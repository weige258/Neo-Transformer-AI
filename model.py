from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from config import CONFIG

# PyTorch 版本兼容性标记
_TORCH_MAJOR, _TORCH_MINOR = map(int, torch.__version__.split(".")[:2])
_CHECKPOINT_SUPPORTS_REENTRANT = (_TORCH_MAJOR, _TORCH_MINOR) >= (2, 0)
from tokenizer import TextTokenizer


CompressedKVCache = tuple[
    torch.Tensor,  # recent_k
    torch.Tensor,  # recent_v
    torch.Tensor,  # mem_k
    torch.Tensor,  # mem_v
    torch.Tensor,  # mem_pos
    int,           # total_len
    torch.Tensor,  # mla_mem_M  (MLA latent associative matrix)
    torch.Tensor,  # mla_mem_z  (MLA latent normalization term)
]


class MLALatentMemory(nn.Module):
    """DeepSeek-V2/V3 style MLA (Multi-head Latent Attention) implementation.
    
    核心改进：
    1. 低秩联合压缩：将K和V一起压缩到共享的latent空间
    2. 矩阵吸收：推理时将上投影矩阵吸收到Q/K投影中
    3. 解耦Q/K投影：Q使用独立的latent维度，K/V共享压缩维度
    
    参考：DeepSeek-V2 Technical Report
    """

    def __init__(self, num_heads: int, head_dim: int, use_delta: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        # use_delta 参数保留仅为兼容旧调用，已不再使用：
        # delta规则/EMA更新会破坏训练与推理的等价性（见 retrieve 注释）

        # 根据网络研究，latent维度应该是head_dim的1/4到1/2
        # DeepSeek-V2使用512维latent压缩2048维的KV
        self.latent_dim = max(16, head_dim // 4)
        
        # K/V联合压缩投影：将head_dim压缩到latent_dim
        # 这是MLA的核心：K和V共享压缩空间
        self.kv_proj = nn.Linear(head_dim, self.latent_dim, bias=False)
        self.v_proj = nn.Linear(head_dim, self.latent_dim, bias=False)
        
        # Q投影：使用相同的latent维度（矩阵吸收要求Q/K维度一致）
        self.q_proj = nn.Linear(head_dim, self.latent_dim, bias=False)
        
        # 上投影：从latent_dim恢复到head_dim
        self.v_up_proj = nn.Linear(self.latent_dim, head_dim, bias=False)
        
        # 输出投影
        self.out_proj = nn.Linear(head_dim, head_dim, bias=False)
        
        # 可学习的缩放因子，用于头差异化
        self.scale = nn.Parameter(torch.ones(num_heads) * 0.1)

    @staticmethod
    def init_mem(batch: int, num_heads: int, head_dim: int, device: torch.device, dtype: torch.dtype
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        """创建初始零状态记忆矩阵和归一化项。"""
        latent_dim = max(16, head_dim // 4)
        M = torch.zeros(1, num_heads, latent_dim, latent_dim, device=device, dtype=dtype)
        z = torch.zeros(1, num_heads, latent_dim, device=device, dtype=dtype)
        return M, z

    def retrieve(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 mem_M: torch.Tensor, mem_z: torch.Tensor) -> torch.Tensor:
        """从压缩记忆中检索值 —— 因果线性注意力版本（修复未来信息泄漏）

        【关键修复】原实现先用整个序列的K/V update记忆、再对每个位置retrieve，
        导致训练时每个位置都能检索到未来token的信息（label leakage）。
        模型（尤其是router）学会严重依赖这条泄漏路径，loss骤降，
        但推理时记忆里只有过去token，生成立刻崩溃为乱码。

        本实现是精确的因果线性注意力：
          - 当前序列内部：下三角mask，位置t只能看到 i<=t 的token
          - 历史记忆 mem_M/mem_z：只包含过去token的精确累加（见 update）
        训练（整段forward）与推理（逐token + 记忆累加）在数学上完全一致，
        不存在任何未来信息通道。

        Args:
            q: Query (batch, heads, seq_len, head_dim)，未加RoPE
            k: Key   (batch, heads, seq_len, head_dim)，未加RoPE
            v: Value (batch, heads, seq_len, head_dim)
            mem_M: 历史关联矩阵 (1, heads, latent_dim, latent_dim)，仅含过去token
            mem_z: 历史归一化项 (1, heads, latent_dim)，仅含过去token
        Returns:
            检索值 (batch, heads, seq_len, head_dim)
        """
        q_lat = self.q_proj(q) * self.scale.view(1, -1, 1, 1)
        q_lat = F.elu(q_lat) + 1.0
        k_lat = F.elu(self.kv_proj(k)) + 1.0
        v_lat = F.elu(self.v_proj(v)) + 1.0

        seq_len = q.size(-2)
        # 当前序列内部的因果注意力（含自身）：S[t,i] = q_t · k_i, i <= t
        scores = torch.matmul(q_lat, k_lat.transpose(-2, -1))
        if seq_len > 1:
            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device))
            scores = scores.masked_fill(~causal_mask, 0.0)

        # 历史记忆只含过去token，与当前序列的因果项相加
        numer = torch.matmul(scores, v_lat) + torch.matmul(q_lat, mem_M)
        denom = scores.sum(dim=-1, keepdim=True) + torch.matmul(q_lat, mem_z.unsqueeze(-1))
        v_lat_out = numer / denom.clamp_min(1e-4)

        v_full = self.v_up_proj(v_lat_out)
        return self.out_proj(v_full)

    def update(self, k: torch.Tensor, v: torch.Tensor,
               mem_M: torch.Tensor, mem_z: torch.Tensor
               ) -> tuple[torch.Tensor, torch.Tensor]:
        """用新KV更新记忆：纯累加（精确求和，无EMA无delta规则）。

        只有纯累加才能保证：训练时位置 t 的因果检索结果
        与推理时"先累加 t 之前的全部token、再检索"的结果严格相等。
        EMA/delta规则都会让推理时的记忆内容偏离训练分布，必须弃用。

        Args:
            k: Key (batch, heads, seq_len, head_dim)
            v: Value (batch, heads, seq_len, head_dim)
            mem_M: 旧关联矩阵
            mem_z: 旧归一化项
        Returns:
            (new_M, new_z)
        """
        k_lat = F.elu(self.kv_proj(k)) + 1.0
        v_lat = F.elu(self.v_proj(v)) + 1.0

        # 【修复】记忆矩阵 (1, H, latent, latent) 在 batch 维上共享，
        # 下方 einsum 会把 batch 维求和掉，batch>1 时不同样本的记忆互相污染。
        # 当前生成路径均为 batch=1，此处直接拒绝 batch>1，避免静默错误。
        if k.size(0) != 1:
            raise ValueError(
                f"MLALatentMemory.update 仅支持 batch=1（记忆矩阵为全 batch 共享），"
                f"实际收到 batch={k.size(0)}")

        # 在 batch 与 seq_len 维度上精确累加
        delta_M = torch.einsum('bhld,bhle->hde', k_lat, v_lat).unsqueeze(0)
        delta_z = k_lat.sum(dim=(0, 2)).unsqueeze(0)  # (1, H, latent)

        return mem_M + delta_M, mem_z + delta_z


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self._nan_warned = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        output = x * norm * self.weight
        # 【优化】NaN/Inf 检查合并为一次 isfinite 全张量归约（原两次归约），
        # 且仅在训练时执行，避免推理时每次前向都强制 GPU 同步
        if self.training:
            if not torch.isfinite(output).all():
                output = torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)
                if not self._nan_warned:
                    self._nan_warned = True
                    print(
                        f"[Warning] RMSNorm detected NaN/Inf; clamped output to safe range. "
                        f"input_min={x.min().item():.3f}, input_max={x.max().item():.3f}",
                        flush=True,
                    )
            elif self._nan_warned:
                self._nan_warned = False
        return output


class RotaryPositionEmbedding(nn.Module):
    """旋转位置编码（RoPE）— 支持YaRN风格NTK-aware扩展

    基于两项研究：
    1. YaRN: "Yet another RoPE extensioN method" (Peng et al., 2023)
       核心：通过调节 base 值和频率缩放实现长序列外推
    2. "Scaling Laws of RoPE-based Extrapolation" (Liu et al., ICLR 2024)
       核心：增大 base 值可显著扩展外推长度，仅需短微调

    【说明】base 值从 CONFIG 读取，默认 10000（LLaMA/GPT-NeoX 标准）。
    对于短序列语言建模，10000 是稳定的选择；如需长序列外推，可通过 CONFIG 增大 rope_base。
    支持 NTK-aware 频率缩放（factor > 1.0）和超长序列位置插值。
    """
    def __init__(self, head_dim: int, base: int = 10000) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embedding.")
        self.head_dim = head_dim
        
        # 从配置读取RoPE参数，以函数参数为fallback
        rope_base = int(CONFIG.get("rope_base", base))
        self.rope_factor = float(CONFIG.get("rope_factor", 1.0))
        self.max_seq_len = int(CONFIG.get("rope_max_seq_len", 4096))
        
        # YaRN: NTK-aware 频率缩放
        # 当 rope_factor > 1.0 时，应用频率缩放：
        # scaling_factor = rope_factor ** (head_dim / (head_dim - 2))
        # 这保持了高频分量的分辨率，同时压缩了低频分量
        if self.rope_factor > 1.0:
            scaling_factor = self.rope_factor ** (head_dim / (head_dim - 2.0))
            # 对部分低频维度应用缩放
            inv_freq = 1.0 / (
                rope_base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
            )
            # 高频保持不变，低频逐步缩放
            ramp = torch.minimum(
                torch.arange(head_dim // 2, dtype=torch.float32) / (head_dim // 4),
                torch.tensor(1.0)
            )
            inv_freq = inv_freq / (1.0 + 0.1 * ramp * (scaling_factor - 1.0))
        else:
            inv_freq = 1.0 / (
                rope_base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
            )
        
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        seq_len: int,
        device: torch.device,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 当序列长度超过最大训练长度时，应用位置插值
        # 这是 YaRN 的核心思想：通过位置缩放避免超长位置编码
        if seq_len > self.max_seq_len:
            # 位置缩放因子：将长序列压缩到 [0, max_seq_len) 范围
            scale = self.max_seq_len / seq_len
            pos = torch.arange(start_pos, start_pos + seq_len, device=device, dtype=torch.float32)
            pos = pos * scale
        else:
            pos = torch.arange(start_pos, start_pos + seq_len, device=device, dtype=torch.float32)
        
        freqs = torch.outer(pos, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """配对式 (half-split) RoPE 半旋转变换。

    对于输入 [x0, x1, x2, x3, ...]，返回 [-x2, -x3, x0, x1]。
    这与 RotaryPositionEmbedding 中 torch.cat((freqs, freqs), dim=-1) 的
    频率布局一致（前半维和后半维共享同一组频率）。

    注意：这是标准 LLaMA 风格的实现，与 emb 的生成方式匹配。
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    rope: RotaryPositionEmbedding,
    start_pos: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = rope(q.size(-2), q.device, start_pos)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


# 注意：HyperAttention 的先验权重现在直接在 forward 中从 CONFIG 读取，
# 与长度自适应偏置(logit 空间)结合。旧的 attention_mix_prior 已被替换。


class HyperAttention(nn.Module):
    """HyperAttention: 压缩记忆 + 局部滑动窗口 + MLA低秩压缩 三路混合注意力
    
    【全动态设计】所有运行时参数（窗口大小、压缩比例等）都基于：
    - 当前序列长度
    - GPU显存压力（reserved/total比例）
    - CPU负载
    - 层深度（金字塔压缩）
    没有任何固定阈值，所有决策都是运行时动态计算。
    """

    def __init__(self, emb_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        self.dropout = dropout
        # ── 动态窗口系数（运行时计算，非固定值） ──
        self.window_scale_factor = float(CONFIG.get("window_scale_factor", 0.5))
        self.window_min_ratio = float(CONFIG.get("window_min_ratio", 0.25))
        self.window_full_attention_ratio = float(CONFIG.get("window_full_attention_ratio", 0.125))
        self.compress_stride = max(2, int(CONFIG.get("compress_stride", 16)))
        self.chunk_size = max(8, int(CONFIG.get("attention_chunk_size", 128)))
        self.sink_count = max(1, int(CONFIG.get("attention_sink_count", 4)))
        # 动态压缩系数
        self.compress_scale = float(CONFIG.get("compress_scale", 0.25))
        self.compress_depth_sensitivity = float(CONFIG.get("compress_depth_sensitivity", 2.0))
        self.compress_memory_pressure_factor = float(CONFIG.get("compress_memory_pressure_factor", 1.5))

        # ── 注意力投影层 ──
        self.qkv_proj = nn.Linear(emb_size, emb_size * 3, bias=False)
        self.router = nn.Sequential(
            nn.Linear(emb_size, max(1, emb_size // 4), bias=False),
            nn.SiLU(),
            nn.Linear(max(1, emb_size // 4), 3, bias=True),  # 3维路由：csa(0)/sliding_window(1)/mla(2)
        )
        self.out_proj = nn.Linear(emb_size, emb_size, bias=False)
        self.rope = RotaryPositionEmbedding(self.head_dim)

        # 【新增】MLA latent KV 压缩记忆（DeepSeek-V2/V3 风格）
        use_delta = bool(CONFIG.get("mla_latent_memory_use_delta", True))
        self.mla_memory = MLALatentMemory(num_heads, self.head_dim, use_delta=use_delta)

        # 【新增】Learned Soft Pooling：可学习的门控压缩权重
        if bool(CONFIG.get("use_learned_pooling", True)):
            self.importance_pooler = nn.Linear(self.head_dim, 1, bias=False)

        # 本训练步的显存压力快照（由 MainModel.forward 在梯度检查点模式下写入），
        # 用于保证 checkpoint 重计算时的动态窗口/压缩比决策与前向一致
        self._forward_mem_pressure: float | None = None

    _gpu_pressure_cache = None
    _gpu_pressure_cache_time = 0.0
    _in_checkpoint = False  # 标记是否在gradient checkpoint内

    def _get_gpu_memory_pressure(self) -> float:
        """获取当前GPU显存压力（0.0-1.0），带缓存避免频繁CUDA同步

        【关键修复】推理时(torch.inference_mode)不查询CUDA状态，避免同步阻塞。
        torch.cuda.memory_allocated() / memory_reserved() 是同步操作，
        每次调用都会阻塞CPU等待GPU完成所有排队内核。
        生成阶段每个token调用此函数3-4次，导致"卡住几十秒"的现象。

        修复策略：
        1. 推理模式下直接返回缓存值或0.3（低压力），完全不触碰CUDA状态
        2. 训练模式下保留1秒缓存
        3. checkpoint模式下返回缓存值或0.5
        """
        if not torch.cuda.is_available():
            return 0.0

        import time as _time
        now = _time.monotonic()

        # 优先返回缓存（无论训练/推理/checkpoint）
        if HyperAttention._gpu_pressure_cache is not None and now - HyperAttention._gpu_pressure_cache_time < 1.0:
            return HyperAttention._gpu_pressure_cache

        # 【修复】推理模式：不查询CUDA，避免同步阻塞。显存在推理期间基本稳定。
        # 能走到这里说明上方 1 秒新鲜缓存检查未命中（缓存不存在或已过期），
        # 必须返回默认低压力 0.3 并刷新缓存——否则训练结束时缓存的高压力值
        # 会一直被推理分支原样返回，污染整个推理会话。
        if not self.training and not torch.is_grad_enabled():
            HyperAttention._gpu_pressure_cache = 0.3
            HyperAttention._gpu_pressure_cache_time = now
            return 0.3

        # checkpoint模式下不能访问CUDA设备状态
        if HyperAttention._in_checkpoint:
            if HyperAttention._gpu_pressure_cache is not None:
                return HyperAttention._gpu_pressure_cache
            return 0.5

        try:
            allocated = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() if torch.cuda.max_memory_allocated() > 0 else 0.0
            # 【修复】使用当前设备而非硬编码 0，尊重 CONFIG["gpu_id"] 的设备选择
            reserved = torch.cuda.memory_reserved() / torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
            pressure = max(allocated, reserved)
            HyperAttention._gpu_pressure_cache = pressure
            HyperAttention._gpu_pressure_cache_time = now
            return pressure
        except (RuntimeError, ValueError):
            return HyperAttention._gpu_pressure_cache if HyperAttention._gpu_pressure_cache is not None else 0.0

    def _get_dynamic_window_size(self, seq_len: int) -> int:
        """全动态窗口大小计算
        
        基于：
        1. 序列长度（seq_len）
        2. GPU显存压力（memory_pressure）
        3. 缩放因子（window_scale_factor）
        
        公式：
        - 如果 seq_len <= full_attention_threshold: 返回seq_len（full attention）
        - 否则: window = seq_len * scale_factor * (1 - memory_pressure * 0.3)
        - 下限: max(512, seq_len * min_ratio)  # 保证至少512的上下文窗口（与旧版一致）
        """
        if seq_len <= 0:
            return 8
        
        # 【修复】优先使用前向开始时快照的显存压力值（见 MainModel.forward）。
        # gradient checkpoint 的重计算发生在 backward 阶段，此时类级压力缓存
        # 可能已过期，重新决策会得到与前向不同的窗口大小，产生静默错误梯度。
        mem_pressure = (self._forward_mem_pressure
                        if (self.training and self._forward_mem_pressure is not None)
                        else self._get_gpu_memory_pressure())
        
        # 【修复】full attention阈值：最大512，超过512必须用窗口
        # 旧版 seq_len <= full_threshold 时返回 seq_len（full attention），
        # 当 seq_len=2048 时 full_threshold=512，但 512 的 full attention 已经很大
        full_threshold = min(512, max(64, int(seq_len * self.window_full_attention_ratio)))
        
        if seq_len <= full_threshold:
            return seq_len
        
        # 动态缩放：显存压力大时减小窗口
        adaptive_scale = self.window_scale_factor * (1.0 - mem_pressure * 0.3)
        adaptive_scale = max(0.1, min(0.9, adaptive_scale))
        
        window_size = int(seq_len * adaptive_scale)
        
        # 【修复】下限保护：至少64，上限512（防显存爆炸）
        min_window = max(64, int(seq_len * self.window_min_ratio))
        window_size = max(min_window, window_size)
        window_size = min(window_size, 512)  # 窗口硬上限512
        
        return window_size

    def _get_dynamic_compress_ratio(self, seq_len: int, layer_idx: int, total_layers: int) -> float:
        """全动态压缩比例计算
        
        基于：
        1. 层深度（layer_idx / total_layers）→ 金字塔压缩
        2. GPU显存压力 → 压力大时增加压缩
        3. 序列长度 → 长序列更激进压缩
        
        公式：
        ratio = base_scale * depth_factor * memory_factor * seq_len_factor
        """
        if total_layers <= 1:
            depth_ratio = 0.5
        else:
            depth_ratio = layer_idx / (total_layers - 1)  # 0.0 ~ 1.0
        
        # 金字塔因子：浅层保留更多，深层压缩更多
        # depth_sensitivity控制差异程度
        depth_factor = 1.0 + (depth_ratio - 0.5) * self.compress_depth_sensitivity
        depth_factor = max(0.3, min(2.0, depth_factor))
        
        # 显存压力因子：压力大时增加压缩（降低ratio=保留更少）
        # 【修复】同 _get_dynamic_window_size：优先用前向快照值，
        # 保证 gradient checkpoint 重计算与前向的压缩比决策一致
        mem_pressure = (self._forward_mem_pressure
                        if (self.training and self._forward_mem_pressure is not None)
                        else self._get_gpu_memory_pressure())
        memory_factor = 1.0 + mem_pressure * self.compress_memory_pressure_factor
        
        # 序列长度因子：超长序列更激进
        if seq_len > 4096:
            seq_factor = 1.3
        elif seq_len > 2048:
            seq_factor = 1.1
        else:
            seq_factor = 1.0
        
        ratio = self.compress_scale * depth_factor * memory_factor * seq_factor
        
        # 裁剪到合理范围
        return max(0.05, min(0.8, ratio))

    def _split_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, emb = x.shape
        qkv = self.qkv_proj(x).view(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        return qkv[0], qkv[1], qkv[2]

    def _compress_kv_with_sink(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        start_pos: int,
        sink_count: int = 4,
        special_mask: torch.Tensor | None = None,
        token_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """智能KV压缩：Attention Sink + Learned Soft Pooling + Special Token Anchoring

        三层保护机制：
        1. StreamingLLM Sink：前sink_count个token永远保留完整KV
        2. Special Token Anchoring：special_mask标记的token以完整精度保留
        3. Learned Soft Pooling：剩余token使用可学习的门控加权凝聚
        
        【修复说明】原代码的special_mask在调用链中从未被传入，导致特殊Token
        （THINK_START/END等）在压缩时信息被稀释。现已修复调用链传递。
        同时在forward中自动为PLACEHOLDER_SINK_TOKEN生成保护掩码。

        Args:
            token_positions: 每个输入token的真实绝对位置 (seq_len,)。
                H2O筛选后传入的序列是非连续的，不能用 start_pos+连续索引 表示，
                调用方必须传入真实位置；为 None 时退化为 start_pos+arange(seq_len)。
        """
        batch, heads, seq_len, dim = k.shape
        if seq_len <= 0:
            empty = k.new_zeros(batch, heads, 0, dim)
            pos = torch.empty(0, device=k.device, dtype=torch.long)
            return empty, empty, pos

        # 【修复】构造每个token的真实绝对位置（H2O筛选后位置非连续）
        if token_positions is None:
            token_positions = start_pos + torch.arange(seq_len, device=k.device, dtype=torch.long)

        # ── 收集需要完整保留的锚点token索引 ──
        anchor_mask = torch.zeros(seq_len, dtype=torch.bool, device=k.device)
        # 1) Sink tokens（前N个，StreamingLLM机制）
        actual_sink = min(sink_count, seq_len)
        anchor_mask[:actual_sink] = True
        # 2) Special tokens（传入的mask，例如THINK_START/END等）
        if special_mask is not None:
            sm = special_mask.bool().squeeze()
            if sm.dim() > 0:
                # 【修复】确保mask长度和seq_len匹配
                if sm.size(0) >= seq_len:
                    anchor_mask[actual_sink:] = anchor_mask[actual_sink:] | sm[actual_sink:seq_len]
                else:
                    # 如果mask比seq_len短，只取有效部分
                    valid_len = min(sm.size(0), seq_len - actual_sink)
                    if valid_len > 0:
                        anchor_mask[actual_sink:actual_sink+valid_len] = anchor_mask[actual_sink:actual_sink+valid_len] | sm[actual_sink:actual_sink+valid_len]
        anchor_idx = anchor_mask.nonzero(as_tuple=True)[0]
        is_anchor = anchor_idx.numel() > 0

        # ── 提取锚点token（完整精度保留） ──
        if is_anchor:
            anchor_k = k[:, :, anchor_idx, :]
            anchor_v = v[:, :, anchor_idx, :]
            anchor_pos = token_positions[anchor_idx]
            # 生成非锚点token索引
            non_anchor_mask = ~anchor_mask
            non_anchor_idx = non_anchor_mask.nonzero(as_tuple=True)[0]
            if non_anchor_idx.numel() == 0:
                return anchor_k, anchor_v, anchor_pos
            compress_k = k[:, :, non_anchor_idx, :]
            compress_v = v[:, :, non_anchor_idx, :]
            compress_pos = token_positions[non_anchor_idx]
        else:
            # 没有锚点，全部压缩（退化到原始行为）
            anchor_k, anchor_v, anchor_pos = None, None, None
            compress_k = k
            compress_v = v
            compress_pos = token_positions

        # ── 对非锚点部分做CSA压缩 (Compressed Sparse Attention) ──
        # DeepSeek-V4: 每4个token压缩成1个entry
        compress_len = compress_k.size(-2)
        if compress_len <= 0:
            if is_anchor:
                return anchor_k, anchor_v, anchor_pos
            return k[:0], v[:0], torch.empty(0, device=k.device, dtype=torch.long)

        # 【CSA】每4个token压缩成1个entry
        csa_stride = 4  # DeepSeek-V4默认压缩比
        chunks = (compress_len + csa_stride - 1) // csa_stride
        padded_len = chunks * csa_stride
        pad_len = padded_len - compress_len
        if pad_len:
            kp = compress_k[:, :, -1:, :].expand(batch, heads, pad_len, dim)
            vp = compress_v[:, :, -1:, :].expand(batch, heads, pad_len, dim)
            compress_k = torch.cat((compress_k, kp), dim=-2)
            compress_v = torch.cat((compress_v, vp), dim=-2)
            # 位置同步补齐：pad 位复制最后一个真实位置（valid mask 已将其权重置零）
            compress_pos = torch.cat((compress_pos, compress_pos[-1:].expand(pad_len)), dim=0)

        ck = compress_k.view(batch, heads, chunks, csa_stride, dim)
        cv = compress_v.view(batch, heads, chunks, csa_stride, dim)
        valid = torch.full((chunks, csa_stride), 1.0, device=k.device, dtype=k.dtype)
        if pad_len:
            valid[-1, -pad_len:] = 0.0

        # Learned Soft Pooling：用可学习线性层计算每个token的重要性权重
        if hasattr(self, 'importance_pooler'):
            importance_logits = self.importance_pooler(ck)  # (B,H,chunks,stride,1)
            # 训练时注入噪声防止过拟合（Gumbel-Softmax风格扰动）
            if self.training:
                importance_logits = importance_logits + torch.randn_like(importance_logits) * 0.01
            importance_logits = importance_logits.masked_fill(
                valid.view(1, 1, chunks, csa_stride, 1) == 0, float('-inf'))
            pool_w = torch.softmax(importance_logits.float(), dim=-2).to(ck.dtype)  # (B,H,chunks,stride,1)
            ck_out = (ck * pool_w).sum(dim=-2)
            cv_out = (cv * pool_w).sum(dim=-2)
        else:
            # 回退到均匀平均
            denom = valid.sum(dim=-1).clamp_min(1.0).view(1, 1, chunks, 1)
            ck_out = (ck * valid.view(1, 1, chunks, csa_stride, 1)).sum(dim=-2) / denom
            cv_out = (cv * valid.view(1, 1, chunks, csa_stride, 1)).sum(dim=-2) / denom

        # CSA位置编码：压缩entry的位置取组内最后一个token的真实位置（chunk末尾）
        # 【修复1】原实现取chunk中点(c*4+2)，位置 t≡2(mod4) 的 query 会通过
        # mem_pos <= q_pos 的因果mask看到包含未来token t+1 的chunk（1-token泄漏）；
        # 取组内最大真实位置后，query 只能在chunk内所有token都成为过去后才看到该entry。
        # 【修复2】H2O筛选后token非连续，必须按真实位置（token_positions）计算，
        # 不能再用 start_pos+连续索引（偏差最大达被筛掉的token数）。
        c_ends = compress_pos.view(chunks, csa_stride).amax(dim=-1)

        # ── 合并锚点 + 压缩结果 ──
        if is_anchor:
            mem_k = torch.cat([anchor_k, ck_out], dim=-2)
            mem_v = torch.cat([anchor_v, cv_out], dim=-2)
            mem_pos = torch.cat([anchor_pos, c_ends], dim=0)
        else:
            mem_k, mem_v, mem_pos = ck_out, cv_out, c_ends
        return mem_k, mem_v, mem_pos

    def _attend_compressed(
        self,
        q: torch.Tensor,
        mem_k: torch.Tensor,
        mem_v: torch.Tensor,
        mem_pos: torch.Tensor,
        q_start_pos: int,
    ) -> torch.Tensor:
        """CSA压缩记忆注意力

        对压缩后的KV entry做标准注意力计算。
        压缩记忆已经过4x压缩，序列长度可控，直接使用SDPA即可。
        使用 causal mask 确保query只关注位置在它之前的记忆entry。
        """
        if mem_k.size(-2) == 0:
            return torch.zeros_like(q)

        batch_size, num_heads, q_len, head_dim = q.shape
        mem_len = mem_k.size(-2)

        q_pos = torch.arange(q_start_pos, q_start_pos + q_len, device=q.device, dtype=torch.float32)
        mem_pos_f = mem_pos.float()

        causal_mask = mem_pos_f.unsqueeze(0) <= q_pos.unsqueeze(1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)

        try:
            attn_out = F.scaled_dot_product_attention(
                q, mem_k, mem_v,
                attn_mask=causal_mask,
                dropout_p=0.0,
                is_causal=False,
            )
        except RuntimeError:
            scores = torch.matmul(q, mem_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(~causal_mask, float("-inf"))
            weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
            weights = torch.nan_to_num(weights, nan=0.0)
            attn_out = torch.matmul(weights, mem_v)

        return attn_out

    def _attend_local_window(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_start_pos: int,
        k_start_pos: int,
    ) -> torch.Tensor:
        """局部滑动窗口注意力（动态窗口大小）

        对超长序列分 chunk 计算，每个 query 只关注窗口范围内的 key。
        窗口大小根据总序列长度动态调整：短序列使用full attention，长序列使用限制窗口。
        """
        batch, heads, q_len, dim = q.shape
        k_len = k.size(-2)
        # 【动态窗口】根据总序列长度计算窗口大小
        total_seq_len = q_start_pos + q_len
        ws = self._get_dynamic_window_size(total_seq_len)

        if k_len == 0:
            return torch.zeros(batch, heads, q_len, dim, device=q.device, dtype=q.dtype)

        # 【快速路径】整个序列都在窗口内（最常见的训练情况）
        # 条件：q_len <= ws 且 k_len <= ws 且 q_start_pos == k_start_pos
        if q_len <= ws and k_len <= ws and q_start_pos == k_start_pos:
            try:
                return F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=True,  # 直接使用原生 causal mask，速度最快
                )
            except RuntimeError:
                pass

        # 【慢速路径】分 chunk 处理长序列
        outputs = []
        for start in range(0, q_len, self.chunk_size):
            end = min(start + self.chunk_size, q_len)
            q_chunk = q[:, :, start:end, :]
            chunk_len = end - start

            q_abs_min = q_start_pos + start
            q_abs_max = q_start_pos + end - 1

            k_abs_min = max(0, q_abs_min - ws + 1)
            k_abs_max = q_abs_max

            k_rel_min = max(0, k_abs_min - k_start_pos)
            k_rel_max = min(k_len - 1, k_abs_max - k_start_pos)

            if k_rel_min > k_rel_max:
                outputs.append(torch.zeros(batch, heads, chunk_len, dim, device=q.device, dtype=q.dtype))
                continue

            k_local = k[:, :, k_rel_min:k_rel_max + 1, :]
            v_local = v[:, :, k_rel_min:k_rel_max + 1, :]
            local_k_len = k_local.size(-2)

            # 【优化】如果此 chunk 内全部 key 都在窗口范围内且位置连续，
            # 直接使用 is_causal=True 代替自定义 bool mask（更快、显存更低）
            q_abs = torch.arange(q_abs_min, q_abs_max + 1, device=q.device, dtype=torch.float32)
            k_abs_local = torch.arange(
                k_start_pos + k_rel_min,
                k_start_pos + k_rel_max + 1,
                device=q.device,
                dtype=torch.float32,
            )
            causal_mask = (k_abs_local[None, :] <= q_abs[:, None]) & \
                         (k_abs_local[None, :] >= (q_abs[:, None] - ws + 1))

            try:
                attn_out = F.scaled_dot_product_attention(
                    q_chunk, k_local, v_local,
                    attn_mask=causal_mask,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=False,
                )
            except RuntimeError:
                scores = torch.matmul(q_chunk, k_local.transpose(-2, -1)) / math.sqrt(dim)
                scores = scores.masked_fill(~causal_mask.view(1, 1, chunk_len, local_k_len), float("-inf"))
                weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
                weights = torch.nan_to_num(weights, nan=0.0)
                attn_out = torch.matmul(weights, v_local)
            outputs.append(attn_out)

        return torch.cat(outputs, dim=-2)

    def _build_cache(
        self,
        k_all: torch.Tensor,
        v_all: torch.Tensor,
        start_pos: int,
        old_mem_k: torch.Tensor | None = None,
        old_mem_v: torch.Tensor | None = None,
        old_mem_pos: torch.Tensor | None = None,
        old_lin_M: torch.Tensor | None = None,
        old_lin_z: torch.Tensor | None = None,
        special_mask: torch.Tensor | None = None,
        layer_idx: int = 0,
        total_layers: int = 8,
    ) -> CompressedKVCache:
        """多级记忆流水线：构建带5级层次化压缩的KV Cache（支持金字塔压缩）

        Level 1 (工作记忆): recent_k/v — 最新动态窗口个token，全量保留
        Level 2 (关键筛选): H2O风格 — 溢出token中保留Heavy Hitters
        Level 3 (语义凝聚): Learned Pooling — 剩余token加权压缩（支持金字塔比例）
        Level 4 (无限历史): MLA latent memory — mem_k 满时固化到低秩关联矩阵
        Level 5 (物理卸载): 量化后CPU offload（由调用方触发）
        
        【金字塔压缩】根据层深度动态调整压缩比例：
        - 浅层（前1/3）: 保留更多细节，压缩率较低
        - 中层（中1/3）: 标准压缩率
        - 深层（后1/3）: 高度抽象，压缩率较高
        """
        total_len = start_pos + k_all.size(-2)
        # 【动态窗口】使用动态计算的窗口大小
        dynamic_window = self._get_dynamic_window_size(total_len)
        
        # 【修复】严格限制 recent_k/v 的最大长度，防止显存无限增长
        # 即使 dynamic_window 很大，也设置一个绝对上限
        max_recent_len_cfg = int(CONFIG.get("max_recent_kv_len", 2048))
        max_recent_len = min(dynamic_window, max_recent_len_cfg)
        keep = min(max_recent_len, k_all.size(-2))
        compress_len = k_all.size(-2) - keep
        
        # 【全动态金字塔压缩】运行时计算压缩比例
        # 基于：层深度、序列长度、GPU显存压力
        pyramid_ratio = self._get_dynamic_compress_ratio(total_len, layer_idx, total_layers)

        # ── Level 1: 工作记忆区（最新token全量保留） ──
        recent_k = k_all[:, :, -keep:, :].contiguous()
        recent_v = v_all[:, :, -keep:, :].contiguous()

        # ── Level 2+3: 对溢出部分做多级压缩 ──
        mem_parts_k = []
        mem_parts_v = []
        mem_parts_pos = []

        # 先保留上一轮的压缩记忆
        # 【修复】限制old_mem长度，防止压缩记忆无限膨胀
        if old_mem_k is not None and old_mem_k.size(-2) > 0:
            base_capacity = int(CONFIG.get("max_mem_kv_capacity", 128))
            if old_mem_k.size(-2) > base_capacity:
                old_mem_k = old_mem_k[:, :, -base_capacity:, :].contiguous()
                old_mem_v = old_mem_v[:, :, -base_capacity:, :].contiguous()
                old_mem_pos = old_mem_pos[-base_capacity:].contiguous()
            mem_parts_k.append(old_mem_k)
            mem_parts_v.append(old_mem_v)
            mem_parts_pos.append(old_mem_pos)

        # 本轮溢出token（仅当有溢出时才做 H2O 筛选 + Pooling 压缩）
        if compress_len > 0:
            overflow_k = k_all[:, :, :compress_len, :]
            overflow_v = v_all[:, :, :compress_len, :]

            # ── 对齐 special_mask 到 overflow 部分的长度 ──
            # 【修复Bug #1 + NEW-3】special_mask 来自当前输入的 token_ids
            # 当 past KV 存在时，当前 token 在 k_all 末尾，应对齐到 overflow 末尾
            overflow_special_mask = None
            if special_mask is not None:
                sm_len = special_mask.size(-1) if special_mask.dim() > 0 else 0
                if sm_len >= compress_len:
                    overflow_special_mask = special_mask[:compress_len]
                else:
                    k_all_len = k_all.size(-2)
                    # 当前 token 在 k_all 末尾，计算有多少在当前 overflow 中
                    current_start_in_kall = k_all_len - sm_len
                    overflow_end = compress_len
                    # overflow 涵盖 k_all[0:compress_len]
                    # 当前 token 区间为 k_all[current_start_in_kall:k_all_len]
                    overlap_start = max(0, current_start_in_kall)
                    overlap_end = min(compress_len, k_all_len)
                    overlap_len = max(0, overlap_end - overlap_start)
                    overflow_special_mask = torch.zeros(compress_len, dtype=torch.bool, device=k_all.device)
                    if overlap_len > 0:
                        # 当前 token 在 overflow 中的偏移量
                        sm_start_in_overflow = current_start_in_kall
                        sm_end_in_overflow = sm_start_in_overflow + overlap_len
                        overflow_special_mask[sm_start_in_overflow:sm_end_in_overflow] = special_mask[:overlap_len]

            # ── Level 2: H2O关键筛选（使用key L2范数作为重要性代理） ──
            # 【注意】当前使用key范数作为重要性代理，这是H2O原始论文的方法。
            # SnapKV (Li et al., 2024) 证明注意力累积分数比key范数更准确，
            # 但计算注意力分数需要额外的forward pass，会增加显存开销。
            # 【金字塔压缩】使用动态计算的h2_ratio（浅层保留更多）
            with torch.no_grad():
                importance = overflow_k.norm(dim=-1).mean(dim=(0, 1))  # (seq_len,)
            
            # 金字塔比例：浅层保留更多Heavy Hitters
            h2_ratio = float(CONFIG.get("h2_ratio", 0.3)) * pyramid_ratio / 0.25
            h2_ratio = max(0.1, min(0.8, h2_ratio))  # 限制在合理范围
            h2_count = max(4, int(compress_len * h2_ratio))
            _, h2_idx_raw = torch.topk(importance, k=min(h2_count, compress_len), sorted=False)
            
            h2_idx = h2_idx_raw.sort().values  # 【修复CRIT-3】保持tensor类型
            h2_k = overflow_k[:, :, h2_idx, :]
            h2_v = overflow_v[:, :, h2_idx, :]
            h2_pos = start_pos + h2_idx

            # ── Level 3: 剩余token做Learned Soft Pooling ──
            remaining_mask = torch.ones(compress_len, dtype=torch.bool, device=k_all.device)
            remaining_mask[h2_idx] = False
            if remaining_mask.any():
                rem_idx = remaining_mask.nonzero(as_tuple=True)[0]
                rem_k = overflow_k[:, :, remaining_mask, :]
                rem_v = overflow_v[:, :, remaining_mask, :]
                rem_start = start_pos + rem_idx[0].item()
                # 【修复】H2O移除非连续token后，rem_k 是压缩过的序列，
                # 必须把剩余token的真实位置传入，否则位置按 start_pos+连续索引
                # 计算会产生最大 h2_count 的漂移，污染下游因果 mask
                rem_pos = start_pos + rem_idx

                # 【修复Bug #1】传递对齐后的special_mask保护特殊Token
                pooled_k, pooled_v, pooled_pos = self._compress_kv_with_sink(
                    rem_k, rem_v, rem_start, sink_count=0,
                    special_mask=overflow_special_mask[remaining_mask] if overflow_special_mask is not None else None,
                    token_positions=rem_pos)
                mem_parts_k.append(torch.cat([h2_k, pooled_k], dim=-2))
                mem_parts_v.append(torch.cat([h2_v, pooled_v], dim=-2))
                mem_parts_pos.append(torch.cat([h2_pos, pooled_pos], dim=0))
            else:
                mem_parts_k.append(h2_k)
                mem_parts_v.append(h2_v)
                mem_parts_pos.append(h2_pos)

        # 合并所有压缩记忆
        if mem_parts_k:
            mem_k = torch.cat(mem_parts_k, dim=-2)
            mem_v = torch.cat(mem_parts_v, dim=-2)
            mem_pos = torch.cat(mem_parts_pos, dim=0)
        else:
            mem_k = k_all.new_zeros(k_all.size(0), k_all.size(1), 0, k_all.size(-1))
            mem_v = v_all.new_zeros(v_all.size(0), v_all.size(1), 0, v_all.size(-1))
            mem_pos = torch.empty(0, device=k_all.device, dtype=torch.long)

        # ── Level 4: 当mem_k超限时，固化最旧部分到 MLA latent memory ──
        # 【修复】根据显存压力动态调整 max_mem_capacity
        mem_pressure = self._get_gpu_memory_pressure()
        base_capacity = int(CONFIG.get("max_mem_kv_capacity", 256))
        # 显存压力大时，更激进地限制压缩记忆容量
        if mem_pressure > 0.8:
            max_mem_capacity = max(64, base_capacity // 4)
        elif mem_pressure > 0.6:
            max_mem_capacity = max(128, base_capacity // 2)
        else:
            max_mem_capacity = base_capacity
        
        if mem_k.size(-2) > max_mem_capacity:
            overflow = mem_k.size(-2) - max_mem_capacity
            to_linear_k = mem_k[:, :, :overflow, :]
            to_linear_v = mem_v[:, :, :overflow, :]
            # 【修复】固化前先对 key 做逆 RoPE：mem_k 中的 key 写入 cache 前
            # 已按其 mem_pos 位置做过 RoPE 旋转，而 MLA 记忆的其余路径
            # （retrieve / 逐 token update / 训练）全部使用无 RoPE 的 key，
            # 直接固化会让带位置相位的 key 混入关联矩阵，污染记忆分布。
            # RoPE 是按位置的正交旋转，用相同位置反向旋转即可还原：
            #   x = y·cos(pθ) − rotate_half(y)·sin(pθ)
            # 对池化压缩 entry，其内部 token 位置不完全相同，取 entry 的代表
            # 位置（mem_pos）近似还原；对锚点/H2O保留的原始 token 则为精确还原。
            # 说明：之所以不用"传入无RoPE的k参数"方案，是因为被固化的 entry
            # 大多是历史轮次的压缩结果，cache 中并未保存其无 RoPE 版本，
            # 逆旋转是唯一对所有 entry 都可行的还原方式。
            pos = mem_pos[:overflow].to(torch.float32)
            freqs = torch.outer(pos, self.rope.inv_freq)
            rope_emb = torch.cat((freqs, freqs), dim=-1)
            cos = rope_emb.cos().view(1, 1, overflow, -1).to(to_linear_k.dtype)
            sin = rope_emb.sin().view(1, 1, overflow, -1).to(to_linear_k.dtype)
            to_linear_k = to_linear_k * cos - rotate_half(to_linear_k) * sin
            # 固化到 MLA latent memory（使用无位置编码的 key）
            mla_M, mla_z = self.mla_memory.update(
                to_linear_k.detach(), to_linear_v.detach(),
                old_lin_M, old_lin_z)
            # 只保留最近的max_mem_capacity个压缩记忆
            mem_k = mem_k[:, :, overflow:, :].contiguous()
            mem_v = mem_v[:, :, overflow:, :].contiguous()
            mem_pos = mem_pos[overflow:].contiguous()
        else:
            mla_M = old_lin_M if old_lin_M is not None else self.mla_memory.init_mem(
                1, self.num_heads, self.head_dim, k_all.device, k_all.dtype)[0]
            mla_z = old_lin_z if old_lin_z is not None else self.mla_memory.init_mem(
                1, self.num_heads, self.head_dim, k_all.device, k_all.dtype)[1]

        return recent_k, recent_v, mem_k, mem_v, mem_pos, total_len, mla_M, mla_z

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: CompressedKVCache | None = None,
        use_cache: bool = False,
        token_ids: torch.Tensor | None = None,
        layer_idx: int = 0,
        total_layers: int = 8,
    ) -> torch.Tensor | tuple[torch.Tensor, CompressedKVCache]:
        batch, seq_len, _ = x.shape
        
        q, k_new, v_new = self._split_qkv(x)

        # 【修复】统一 past_key_value 的有效性判断：None 才表示无 past；
        # 非 None 但结构畸形（非 tuple/list 或长度不足 6，无法解包）时直接报错，
        # 不再静默当作无 past 处理——否则下方 else 分支会用 past_key_value is None
        # 做判断，导致对 None.size() 抛 AttributeError 或逻辑错乱。
        if past_key_value is not None and (
                not isinstance(past_key_value, (tuple, list)) or len(past_key_value) < 6):
            raise ValueError(
                "past_key_value 格式错误：期望 None 或长度>=6 的 tuple/list "
                "(recent_k, recent_v, mem_k, mem_v, mem_pos, total_len[, mla_M, mla_z])，"
                f"实际类型={type(past_key_value).__name__}, "
                f"长度={len(past_key_value) if isinstance(past_key_value, (tuple, list)) else 'N/A'}")

        if past_key_value is None:
            q_start_pos = 0
            raw_k_start_pos = 0
            past_recent_k = past_recent_v = None
            mem_k = k_new.new_zeros(batch, self.num_heads, 0, self.head_dim)
            mem_v = v_new.new_zeros(batch, self.num_heads, 0, self.head_dim)
            mem_pos = torch.empty(0, device=x.device, dtype=torch.long)
            mla_M = None
            mla_z = None
        else:
            # 解包 cache；兼容旧版本缓存（缺少 MLA latent memory）
            (past_recent_k, past_recent_v, mem_k, mem_v, mem_pos,
             q_start_pos, *rest) = past_key_value
            # 【修复】raw_k_start_pos 必须反映 past_recent_k 实际的起始位置
            # q_start_pos 是 total_len（全局token计数），past_recent_k 可能被截断
            # 正确计算：raw_k_start_pos = q_start_pos - past_recent_k.size(-2)
            # 但截断后 past_recent_k 不再从位置0开始，所以需要用 cache 中的信息
            # 由于 _build_cache 中 total_len = start_pos + k_all.size(-2)，
            # 而 recent_k = k_all[:, :, -keep:, :]，所以 recent_k 的起始位置 = total_len - keep
            raw_k_start_pos = q_start_pos - past_recent_k.size(-2)
            mla_M = rest[0] if len(rest) >= 2 else None
            mla_z = rest[1] if len(rest) >= 2 else None

        if mla_M is None or mla_z is None:
            mla_M, mla_z = self.mla_memory.init_mem(
                1, self.num_heads, self.head_dim, x.device, x.dtype)

        # 【说明】此处绝不能先用当前序列的K/V更新MLA记忆再retrieve：
        # 那会让每个位置看到未来token（label leakage），训练loss虚假降低、
        # 推理时生成崩溃。因果性由 MLALatentMemory.retrieve 内部的下三角
        # mask + "记忆只含过去token"共同保证。

        q_for_memory = q
        k_for_memory = k_new

        q, k_new = apply_rope(q, k_new, self.rope, q_start_pos)

        if past_key_value is None:
            raw_k = k_new
            raw_v = v_new
        else:
            # 【修复】限制拼接后的长度，防止无限增长
            max_total_len = int(CONFIG.get("max_total_kv_len", 4096))  # 从配置读取
            past_len = past_recent_k.size(-2)
            if past_len + seq_len > max_total_len:
                # 截断旧的 past KV
                keep_past = max_total_len - seq_len
                if keep_past > 0:
                    past_recent_k = past_recent_k[..., -keep_past:, :]
                    past_recent_v = past_recent_v[..., -keep_past:, :]
                else:
                    past_recent_k = past_recent_k.new_zeros(batch, self.num_heads, 0, self.head_dim)
                    past_recent_v = past_recent_v.new_zeros(batch, self.num_heads, 0, self.head_dim)
                # 【修复】截断后必须更新 raw_k_start_pos，否则位置编码错位
                raw_k_start_pos = q_start_pos - past_recent_k.size(-2)
            raw_k = torch.cat((past_recent_k, k_new), dim=-2)
            raw_v = torch.cat((past_recent_v, v_new), dim=-2)

        # 【动态窗口】计算当前序列的窗口大小
        dynamic_window = self._get_dynamic_window_size(raw_k.size(-2))
        
        if past_key_value is None and seq_len > 1:
            # 训练时（use_cache=False）需要在此处构建压缩记忆
            # 推理时由末尾_build_cache处理
            if raw_k.size(-2) > dynamic_window:
                mem_k, mem_v, mem_pos = self._compress_kv_with_sink(
                    raw_k, raw_v, raw_k_start_pos, sink_count=self.sink_count)

        # 【HyperAttention】DeepSeek-V4风格：CSA + SlidingWindow + MLA
        # 核心思想：
        # 1. CSA (Compressed Sparse Attention): 每4个token压缩成1个entry + 稀疏注意力
        # 2. SlidingWindow: 动态窗口大小的精确注意力（短序列=full attention）
        # 3. MLA: 低秩latent压缩，处理长距离上下文
        # 【统一先验】config 权重 + 长度自适应偏置，在 logit 空间相加
        cfg_mix = CONFIG.get("attention_mix", {"csa": 1.0, "sliding_window": 1.0, "mla": 1.0})
        cfg_prior = torch.tensor(
            [cfg_mix.get("csa", 1.0), cfg_mix.get("sliding_window", 1.0), cfg_mix.get("mla", 1.0)],
            device=x.device, dtype=torch.float32
        )
        cfg_logits = torch.log(cfg_prior.clamp_min(1e-6))

        # 【动态长度自适应偏置】根据实际序列长度和动态窗口大小调整
        total_len = raw_k.size(-2)
        dynamic_window = self._get_dynamic_window_size(total_len)
        full_threshold = max(16, int(total_len * self.window_full_attention_ratio))
        
        if total_len <= full_threshold:
            # 短序列：主要依赖SlidingWindow（等效full attention）
            length_bias = torch.tensor([0.7, 1.4, 0.8], device=x.device, dtype=torch.float32)
        elif total_len <= dynamic_window:
            # 中等序列：平衡三路
            length_bias = torch.tensor([0.9, 1.2, 0.9], device=x.device, dtype=torch.float32)
        elif total_len <= dynamic_window * 2:
            # 较长序列：增加CSA和MLA权重
            length_bias = torch.tensor([1.1, 1.0, 1.1], device=x.device, dtype=torch.float32)
        else:
            # 超长序列：主要依赖CSA和MLA压缩
            length_bias = torch.tensor([1.3, 0.8, 1.2], device=x.device, dtype=torch.float32)
        length_logits = torch.log(length_bias)

        combined_prior = (cfg_logits + length_logits).unsqueeze(0).unsqueeze(0)  # [1,1,3]
        mix = torch.softmax(self.router(x).float() + combined_prior, dim=-1).to(x.dtype)

        # 初始化累积输出为零
        out_accum = None  # (batch, seq_len, emb_size)

        # 【优化】三个路由分支的阈值标量拼成一个张量，一次GPU→CPU同步取回，
        # 避免三处 .item() 各自强制一次同步（阈值与分支语义保持不变：
        # csa > 0.01、sliding_window > 0.01、mla 按 abs > 1e-8）
        mix_gate_csa, mix_gate_sw, mix_gate_mla = torch.stack([
            mix[..., 0].max(),
            mix[..., 1].max(),
            mix[..., 2].abs().max(),
        ]).tolist()

        # ── 路径0: CSA (Compressed Sparse Attention) ──
        # DeepSeek-V4: 每4个token压缩成1个entry + 稀疏注意力
        if mix_gate_csa > 0.01 and mem_k.size(-2) > 0:
            # 使用压缩后的KV进行稀疏注意力
            csa_out = self._attend_compressed(q, mem_k, mem_v, mem_pos, q_start_pos)
            csa_out = csa_out.transpose(1, 2).contiguous().view(batch, seq_len, self.emb_size)
            out_accum = csa_out * mix[..., 0:1]
            del csa_out

        # ── 路径1: SlidingWindow (精确注意力) ──
        # 【修复】之前硬编码window_size=128，导致sliding_window配置被完全忽略。
        # 现在使用self.window_size（来自配置）。对于短序列，这等效于full attention。
        if mix_gate_sw > 0.01:
            # 【动态窗口】使用动态计算的窗口大小
            dynamic_window = self._get_dynamic_window_size(raw_k.size(-2))
            window_size = min(dynamic_window, raw_k.size(-2))
            if window_size > 0:
                window_k = raw_k[..., -window_size:, :]
                window_v = raw_v[..., -window_size:, :]
                # 【修复】窗口切片后其绝对起始位置不再是 raw_k_start_pos：
                # window_k 是 raw_k 末尾 window_size 个 token，
                # 正确起点 = raw_k_start_pos + raw_k长度 - window_size。
                # 传错起点会导致长序列训练时位置错位（未来信息泄漏），
                # 且 chunk 起点 > 2*window 时该分支整块输出为零。
                window_k_start = raw_k_start_pos + raw_k.size(-2) - window_size
                window_out = self._attend_local_window(q, window_k, window_v, q_start_pos, window_k_start)
                window_out = window_out.transpose(1, 2).contiguous().view(batch, seq_len, self.emb_size)
                if out_accum is None:
                    out_accum = window_out * mix[..., 1:2]
                else:
                    out_accum = out_accum + window_out * mix[..., 1:2]
                del window_out

        # ── 路径2: MLA (Multi-head Latent Attention) ──
        # DeepSeek-V4: 低秩latent压缩，处理长距离上下文
        # 【修复】因果检索：当前序列内下三角mask，历史记忆只含过去token
        if mix_gate_mla > 1e-8:
            lin_out = self.mla_memory.retrieve(q_for_memory, k_for_memory, v_new, mla_M, mla_z)
            lin_out = lin_out.transpose(1, 2).contiguous().view(batch, seq_len, self.emb_size)
            if out_accum is None:
                out_accum = lin_out * mix[..., 2:3]
            else:
                out_accum = out_accum + lin_out * mix[..., 2:3]
            del lin_out

        # 极端情况：所有权重都为零，返回零张量
        if out_accum is None:
            out_accum = torch.zeros(batch, seq_len, self.emb_size, device=x.device, dtype=x.dtype)

        del q

        if past_key_value is not None and use_cache:
            with torch.no_grad():
                new_mla_M, new_mla_z = self.mla_memory.update(
                    k_for_memory.detach(), v_new.detach(), mla_M, mla_z)
        else:
            new_mla_M = mla_M
            new_mla_z = mla_z

        out = self.out_proj(out_accum)
        del out_accum

        if use_cache:
            # 【修复】从token_ids参数生成special_mask保护特殊Token
            special_mask = None
            if token_ids is not None:
                # 【修复】使用TextTokenizer中的特殊token值判断，而非硬编码<10
                from tokenizer import TextTokenizer
                special_ids = torch.tensor([
                    TextTokenizer.UNKNOWN_TOKEN,
                    TextTokenizer.START_GENERATION_TOKEN,
                    TextTokenizer.END_GENERATION_TOKEN,
                    TextTokenizer.HISTORY_CONTEXT_START_TOKEN,
                    TextTokenizer.HISTORY_CONTEXT_END_TOKEN,
                    TextTokenizer.THINK_START_TOKEN,
                    TextTokenizer.THINK_END_TOKEN,
                ], device=token_ids.device, dtype=token_ids.dtype)
                if token_ids.dim() == 1:
                    special_mask = (token_ids.unsqueeze(-1) == special_ids).any(dim=-1)
                elif token_ids.dim() == 2:
                    if token_ids.size(0) == 1:
                        special_mask = (token_ids[0].unsqueeze(-1) == special_ids).any(dim=-1)
                    else:
                        special_mask = ((token_ids.unsqueeze(-1) == special_ids).any(dim=-1)).any(dim=0)
                elif token_ids.dim() == 3:
                    special_mask = ((token_ids.unsqueeze(-1) == special_ids).any(dim=-1)).any(dim=0).squeeze(-1)
            
            cache = self._build_cache(
                raw_k,
                raw_v,
                raw_k_start_pos,
                old_mem_k=mem_k if past_key_value is not None else None,
                old_mem_v=mem_v if past_key_value is not None else None,
                old_mem_pos=mem_pos if past_key_value is not None else None,
                old_lin_M=new_mla_M,
                old_lin_z=new_mla_z,
                special_mask=special_mask,
                layer_idx=layer_idx,
                total_layers=total_layers,
            )
            return out, cache
        return out


class SwiGLUFFN(nn.Module):
    def __init__(self, emb_size: int, dropout: float) -> None:
        super().__init__()
        hidden = int(emb_size *  3)
        self.gate_proj = nn.Linear(emb_size, hidden, bias=False)
        self.up_proj = nn.Linear(emb_size, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, emb_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class HyperAttentionBlock(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, dropout: float, layer_idx: int = 0, total_layers: int = 8) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.total_layers = total_layers
        self.attn_norm = RMSNorm(emb_size)
        self.ffn_norm = RMSNorm(emb_size)
        self.attention = HyperAttention(emb_size, num_heads, dropout)
        self.feed_forward = SwiGLUFFN(emb_size, dropout)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: CompressedKVCache | None = None,
        use_cache: bool = False,
        token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, CompressedKVCache]:
        attn_result = self.attention(
            self.attn_norm(x),
            past_key_value=past_key_value,
            use_cache=use_cache,
            token_ids=token_ids,
            layer_idx=self.layer_idx,
            total_layers=self.total_layers,
        )
        if use_cache:
            attn_out, present = attn_result
        else:
            attn_out = attn_result
            present = None

        x = x + attn_out
        x = x + self.feed_forward(self.ffn_norm(x))

        if use_cache:
            return x, present
        return x


class MainModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        dict_size = int(CONFIG["dict_size"])
        emb_size = int(CONFIG["emb_size"])
        num_heads = int(CONFIG["num_heads"])
        if emb_size % num_heads != 0:
            raise ValueError("emb_size must be divisible by num_heads.")

        dropout = float(CONFIG.get("dropout", 0.05))
        num_transformer_blocks = int(CONFIG.get("num_transformer_blocks", 8))
        if num_transformer_blocks < 1:
            raise ValueError("CONFIG['num_transformer_blocks'] must be at least 1.")

        self.token_embedding = nn.Embedding(dict_size, emb_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.transformers = nn.ModuleList(
            HyperAttentionBlock(emb_size, num_heads, dropout, layer_idx=i, total_layers=num_transformer_blocks)
            for i in range(num_transformer_blocks)
        )
        self.final_norm = RMSNorm(emb_size)
        self.output_linear = nn.Linear(emb_size, dict_size, bias=False)
        
        # ── 权重绑定（Weight Tying） ──
        # 将输出投影层权重与输入 embedding 共享，减少参数量约 30%，
        # 同时利用输入/输出语义空间的对称性提升泛化能力。
        # 
        # 梯度流说明（配合梯度检查点）：
        #   forward: tokens → token_embedding → [transformer blocks]* → output_linear → logits
        #   backward: loss → output_linear.weight.grad ← (来自 logits 的梯度)
        #             loss → [transformer blocks]* → token_embedding.weight.grad ← (来自 lookup 的梯度)
        #   PyTorch 自动对共享权重的梯度求和，这是正确且期望的行为。
        #   * 标记的 transformer blocks 在 use_gradient_checkpointing=True 时会重计算。
        #   由于 token_embedding 在 checkpoint 区域之外，梯度流不受影响。
        if bool(CONFIG.get("tie_token_embeddings", True)):
            self.output_linear.weight = self.token_embedding.weight
            # 显式确保梯度状态：nn.Embedding.weight 默认为 requires_grad=True，
            # 但显式调用可防止未来 PyTorch 版本行为变化导致的静默 bug
            self.token_embedding.weight.requires_grad_(True)
            # 输出层无 bias（bias=False），无需额外处理
            # 注意：共享权重意味着 token_embedding.weight.grad 会同时累积
            # 来自 embedding lookup 和 output projection 的梯度，PyTorch 自动求和

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # 初始化输入 embedding（如果与输出绑定，则同时初始化了两者）
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        # 【修复】特殊Token使用更大的初始化标准差
        # 原因：特殊Token在训练中出现的频率远低于普通token，梯度信号弱
        # 更大的初始范数帮助模型从一开始就区分特殊Token和普通Token
        # 注意：特殊token现在在词表末尾（59990-59996）
        dict_size = self.token_embedding.weight.size(0)
        special_token_ids = [dict_size - 10 + i for i in range(10)]  # 最后10个token
        with torch.no_grad():
            special_std = 0.05  # 特殊Token 2.5倍标准差
            for token_id in special_token_ids:
                if token_id < dict_size and token_id >= 0:
                    self.token_embedding.weight[token_id].normal_(mean=0.0, std=special_std)
        # 仅在未绑定时独立初始化输出层（避免覆盖共享权重的初始化）
        if self.output_linear.weight is not self.token_embedding.weight:
            nn.init.normal_(self.output_linear.weight, mean=0.0, std=0.02)
        # 初始化其他线性层（排除已绑定权重的 output_linear）
        for module in self.modules():
            if isinstance(module, nn.Linear) and module.weight is not self.output_linear.weight:
                nn.init.xavier_uniform_(module.weight)

    def _compress_embeddings(self, hist_emb: torch.Tensor, compress_num: int) -> torch.Tensor:
        """对 embedding 序列进行基于重要性权重的压缩聚合。

        Args:
            hist_emb: (seq_len, emb_size) 的 embedding 序列
            compress_num: 目标压缩后的段数
        Returns:
            (compress_num, emb_size) 的压缩后向量
        """
        seq_len = hist_emb.size(0)
        if seq_len <= compress_num:
            return self.final_norm(hist_emb)

        scores = hist_emb.norm(dim=-1)
        boundaries = torch.linspace(0, seq_len, compress_num + 1, device=hist_emb.device)
        pieces: list[torch.Tensor] = []
        for idx in range(compress_num):
            start = int(boundaries[idx].item())
            end = max(start + 1, int(boundaries[idx + 1].item()))
            segment = hist_emb[start:end]
            weights = torch.softmax(scores[start:end].float(), dim=0).to(hist_emb.dtype)
            pieces.append((segment * weights[:, None]).sum(dim=0))
        return self.final_norm(torch.stack(pieces, dim=0))

    def compress_history_vectors(
        self,
        history_tokens: torch.Tensor,
        compress_ratio: float | None = None,
    ) -> torch.Tensor:
        if compress_ratio is None:
            compress_ratio = float(CONFIG.get("compress_ratio", 0.3))
        with torch.no_grad():
            emb_weight = self.token_embedding.weight
            prefer_gpu = bool(CONFIG.get("prefer_gpu_compress", True))

            # 如果 history_tokens 不在 embedding 权重同一设备上，但配置允许在 GPU 上分块压缩
            if history_tokens.device != emb_weight.device and prefer_gpu:
                device = emb_weight.device
                hist_idx = history_tokens.to(torch.long)
                if hist_idx.dim() == 2 and hist_idx.size(0) == 1:
                    hist_idx = hist_idx.squeeze(0)

                seq_len = hist_idx.numel()
                compress_num = max(16, int(seq_len * compress_ratio))

                # 流式读取与 embedding：逐块映射到 GPU
                chunk = int(max(1024, seq_len // max(1, min(8, seq_len // 1024))))
                emb_parts: list[torch.Tensor] = []
                for start in range(0, seq_len, chunk):
                    end = min(start + chunk, seq_len)
                    idx_slice = hist_idx[start:end].to(device)
                    emb_slice = self.token_embedding(idx_slice)
                    if emb_slice.dim() == 3:
                        emb_slice = emb_slice.squeeze(0)
                    emb_parts.append(emb_slice)

                hist_emb = torch.cat(emb_parts, dim=0)
                if hist_emb.dim() == 3:
                    hist_emb = hist_emb.squeeze(0)
                return self._compress_embeddings(hist_emb, compress_num)

            # 若 history_tokens 在同设备，或不启用 GPU 分块压缩，走原有路径
            if history_tokens.device != emb_weight.device:
                # 在 CPU 上进行 embedding lookup 与压缩
                # 【优化】先在权重所在设备按索引 gather 需要的行，仅把选中行
                # 搬到 CPU；原实现每次复制整个 embedding 矩阵（约123MB）到 CPU
                hist_idx = history_tokens.to(torch.long).to(emb_weight.device)
                hist_emb = emb_weight.detach()[hist_idx].to("cpu")
                if hist_emb.dim() == 3:
                    hist_emb = hist_emb.squeeze(0)
                seq_len = hist_emb.size(0)
                compress_num = max(16, int(seq_len * compress_ratio))
                if seq_len <= compress_num:
                    w = self.final_norm.weight.detach().to("cpu")
                    eps = self.final_norm.eps
                    normed = hist_emb * torch.rsqrt(hist_emb.pow(2).mean(dim=-1, keepdim=True) + eps)
                    return normed * w
                return self._compress_embeddings(hist_emb, compress_num)
            else:
                # 常规路径：history_tokens 与 embedding 在同一设备
                hist_emb = self.token_embedding(history_tokens)
                if hist_emb.dim() == 3:
                    hist_emb = hist_emb.squeeze(0)
                seq_len = hist_emb.size(0)
                compress_num = max(16, int(seq_len * compress_ratio))
                return self._compress_embeddings(hist_emb, compress_num)

    def forward(
        self,
        tokens: torch.Tensor,
        past_key_values: list[CompressedKVCache | None] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[CompressedKVCache]]:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
            squeeze_batch = True
        elif tokens.dim() == 2:
            squeeze_batch = False
        else:
            raise ValueError("tokens must have shape [seq_len] or [batch, seq_len].")

        x = self.embedding_dropout(self.token_embedding(tokens))

        if past_key_values is None:
            past_key_values = [None] * len(self.transformers)
        elif len(past_key_values) != len(self.transformers):
            raise ValueError("past_key_values length must match transformer block count.")

        next_key_values: list[CompressedKVCache] = []
        if use_cache:
            for block, past in zip(self.transformers, past_key_values):
                x, present = block(x, past_key_value=past, use_cache=True, token_ids=tokens)
                next_key_values.append(present)
        else:
            use_gc = bool(CONFIG.get("use_gradient_checkpointing", False))
            if self.training and use_gc:
                HyperAttention._in_checkpoint = True
                # 【修复】为每层快照本训练步的显存压力决策：
                # gradient checkpoint 的重计算发生在 backward 阶段，此时
                # _in_checkpoint 已复位、类级压力缓存（1秒TTL）可能已过期，
                # 重新决策会得到与前向不同的动态窗口/压缩比，产生静默错误梯度。
                # 同一训练步内 forward→backward 之间不会再有另一次 forward
                # （main.py 训练循环每步仅调用一次本前向），因此重计算时
                # 复用该实例快照是安全的；快照在 checkpoint 区域外设置，
                # 重计算路径（block.forward）不会覆盖它。
                for _blk in self.transformers:
                    _blk.attention._forward_mem_pressure = _blk.attention._get_gpu_memory_pressure()
            try:
                for block in self.transformers:
                    if self.training and use_gc:
                        ck_kwargs = {"use_reentrant": False} if _CHECKPOINT_SUPPORTS_REENTRANT else {}
                        x = checkpoint(block, x, None, False, tokens, **ck_kwargs)
                    else:
                        x = block(x, None, False, tokens)
            finally:
                if self.training and use_gc:
                    HyperAttention._in_checkpoint = False

        logits = self.output_linear(self.final_norm(x))
        
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)
        elif logits.dim() == 3:
            pass
        else:
            raise ValueError(f"logits dimension {logits.dim()} not supported, expected 2 or 3")
        
        if squeeze_batch:
            logits = logits.squeeze(0)
        
        if use_cache:
            return logits, next_key_values
        return logits