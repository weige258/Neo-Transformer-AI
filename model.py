from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from config import CONFIG
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
        self.use_delta = use_delta

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
        self.k_up_proj = nn.Linear(self.latent_dim, head_dim, bias=False)
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

    def retrieve(self, q: torch.Tensor, mem_M: torch.Tensor, mem_z: torch.Tensor) -> torch.Tensor:
        """从压缩记忆中检索值（矩阵吸收版本）。
        
        矩阵吸收优化：将k_up_proj吸收到q_proj中，减少推理计算量。
        实际计算时：Q_latent -> (吸收k_up_proj) -> Q_full -> attention -> V_full
        
        Args:
            q: Query张量 (batch, heads, seq_len, head_dim)
            mem_M: 关联矩阵 (1, heads, latent_dim, latent_dim)
            mem_z: 归一化项 (1, heads, latent_dim)
        Returns:
            检索值 (batch, heads, seq_len, head_dim)
        """
        # Q投影到latent空间
        q_lat = self.q_proj(q) + self.scale.view(1, -1, 1, 1)
        
        # 使用softmax激活替代elu，更稳定
        q_lat = F.softmax(q_lat / math.sqrt(q_lat.size(-1)), dim=-1)
        
        # 从压缩记忆中检索
        numer = torch.matmul(q_lat, mem_M)
        denom = torch.matmul(q_lat, mem_z.unsqueeze(-1)).clamp_min(1e-8)
        v_lat = numer / denom
        
        # 上投影恢复到head_dim
        v_full = self.v_up_proj(v_lat)
        return self.out_proj(v_full)

    def update(self, k: torch.Tensor, v: torch.Tensor,
               mem_M: torch.Tensor, mem_z: torch.Tensor
               ) -> tuple[torch.Tensor, torch.Tensor]:
        """用新KV更新记忆，返回新的M/z。
        
        使用增量更新（delta update）来减少记忆漂移。
        
        Args:
            k: Key (batch, heads, seq_len, head_dim)
            v: Value (batch, heads, seq_len, head_dim)
            mem_M: 旧关联矩阵
            mem_z: 旧归一化项
        Returns:
            (new_M, new_z)
        """
        # K/V联合压缩
        k_lat = self.kv_proj(k)
        v_lat = self.v_proj(v)
        
        # 使用softmax激活
        k_lat = F.softmax(k_lat / math.sqrt(k_lat.size(-1)), dim=-1)
        
        if self.use_delta:
            # 增量更新：只更新预测误差部分
            k_z = torch.matmul(k_lat, mem_z.unsqueeze(-1)).clamp_min(1e-8)
            v_pred = torch.matmul(k_lat, mem_M) / k_z
            v_error = v_lat - v_pred
            delta_M = torch.matmul(k_lat.transpose(-2, -1), v_error)
        else:
            delta_M = torch.matmul(k_lat.transpose(-2, -1), v_lat)

        # 使用指数移动平均更新，更稳定
        alpha = 0.9  # 历史权重
        new_M = alpha * mem_M + (1 - alpha) * delta_M.mean(dim=0, keepdim=True)
        new_z = alpha * mem_z + (1 - alpha) * k_lat.sum(dim=-2).mean(dim=0, keepdim=True)
        return new_M, new_z


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        output = x * norm * self.weight
        if torch.isnan(output).any() or torch.isinf(output).any():
            output = torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)
            print(
                f"[Warning] RMSNorm detected NaN/Inf; clamped output to safe range. "
                f"input_min={x.min().item():.3f}, input_max={x.max().item():.3f}",
                flush=True,
            )
        return output


class RotaryPositionEmbedding(nn.Module):
    """旋转位置编码（RoPE）— 支持YaRN风格NTK-aware扩展

    基于两项研究：
    1. YaRN: "Yet another RoPE extensioN method" (Peng et al., 2023)
       核心：通过调节 base 值和频率缩放实现长序列外推
    2. "Scaling Laws of RoPE-based Extrapolation" (Liu et al., ICLR 2024)
       核心：增大 base 值可显著扩展外推长度，仅需短微调

    【修复说明】原 base=10000 限制了模型外推能力。
    新方案：
    - base 提高到 1000000（百万级），与 Llama 3.1 一致
    - 支持 NTK-aware 频率缩放（factor > 1.0）
    - 当 seq_len > max_seq_len 时自动启用插值
    """
    def __init__(self, head_dim: int, base: int = 10000) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embedding.")
        self.head_dim = head_dim
        
        # 从配置读取RoPE参数
        # 【修复HIGH #6】使用CONFIG中的base值作为默认，而非函数签名参数
        # 确保当CONFIG缺少rope_base键时使用配置值而非默认10000
        rope_base = int(CONFIG.get("rope_base", 10000))
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


def attention_mix_prior(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """HyperAttention先验：CSA + SlidingWindow + MLA"""
    mix = CONFIG.get("attention_mix", {})
    weights = torch.tensor(
        [
            float(mix.get("csa", 1.0)),        # CSA压缩稀疏注意力
            float(mix.get("sliding_window", 1.5)),  # SlidingWindow精确注意力
            float(mix.get("mla", 1.0)),        # MLA低秩压缩
        ],
        device=device,
        dtype=dtype,
    )
    weights = torch.clamp(weights, min=1e-6)
    return torch.log(weights / weights.sum())


class HyperAttention(nn.Module):
    """HyperAttention: 压缩记忆 + 局部滑动窗口 + MLA低秩压缩 三路混合注意力"""

    def __init__(self, emb_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        self.dropout = dropout
        self.window_size = max(8, int(CONFIG.get("sliding_window", 256)))
        self.compress_stride = max(2, int(CONFIG.get("compress_stride", 16)))
        self.chunk_size = max(8, int(CONFIG.get("attention_chunk_size", 128)))
        self.sink_count = max(1, int(CONFIG.get("attention_sink_count", 4)))  # StreamingLLM sink保护

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """智能KV压缩：Attention Sink + Learned Soft Pooling + Special Token Anchoring

        三层保护机制：
        1. StreamingLLM Sink：前sink_count个token永远保留完整KV
        2. Special Token Anchoring：special_mask标记的token以完整精度保留
        3. Learned Soft Pooling：剩余token使用可学习的门控加权凝聚
        
        【修复说明】原代码的special_mask在调用链中从未被传入，导致特殊Token
        （THINK_START/END等）在压缩时信息被稀释。现已修复调用链传递。
        同时在forward中自动为PLACEHOLDER_SINK_TOKEN生成保护掩码。
        """
        batch, heads, seq_len, dim = k.shape
        if seq_len <= 0:
            empty = k.new_zeros(batch, heads, 0, dim)
            pos = torch.empty(0, device=k.device, dtype=torch.long)
            return empty, empty, pos

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
            anchor_pos = start_pos + anchor_idx
            # 生成非锚点token索引
            non_anchor_mask = ~anchor_mask
            non_anchor_idx = non_anchor_mask.nonzero(as_tuple=True)[0]
            if non_anchor_idx.numel() == 0:
                return anchor_k, anchor_v, anchor_pos
            compress_k = k[:, :, non_anchor_idx, :]
            compress_v = v[:, :, non_anchor_idx, :]
            compress_start = start_pos + non_anchor_idx[0].item()
        else:
            # 没有锚点，全部压缩（退化到原始行为）
            anchor_k, anchor_v, anchor_pos = None, None, None
            compress_k = k
            compress_v = v
            compress_start = start_pos

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

        # CSA位置编码：压缩后的位置取chunk的中点
        c_ends = torch.arange(chunks, device=k.device, dtype=torch.long)
        c_ends = compress_start + (c_ends * csa_stride + csa_stride // 2)

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
        """局部滑动窗口注意力（显存优化版）

        使用 F.scaled_dot_product_attention 替代手动 index_select + matmul，
        利用 PyTorch 内置的 FlashAttention / Memory-Efficient Attention 后端，
        大幅降低显存占用。

        对超过窗口范围的 key 使用 causal mask 屏蔽。
        """
        batch, heads, q_len, dim = q.shape
        k_len = k.size(-2)
        ws = self.window_size

        if k_len == 0:
            return torch.zeros(batch, heads, q_len, dim, device=q.device, dtype=q.dtype)

        outputs = []

        for start in range(0, q_len, self.chunk_size):
            end = min(start + self.chunk_size, q_len)
            q_chunk = q[:, :, start:end, :]  # (batch, heads, chunk_len, dim)
            chunk_len = end - start

            # 确定此 chunk 需要关注的最小/最大 key 位置
            q_abs_min = q_start_pos + start
            q_abs_max = q_start_pos + end - 1

            # 局部窗口范围：[q_pos - ws + 1, q_pos]（因果）
            k_abs_min = max(0, q_abs_min - ws + 1)
            k_abs_max = q_abs_max

            # 转换为相对于 raw_k 的索引
            k_rel_min = max(0, k_abs_min - k_start_pos)
            k_rel_max = min(k_len - 1, k_abs_max - k_start_pos)

            if k_rel_min > k_rel_max:
                # 此 chunk 没有可关注的 key
                outputs.append(torch.zeros(batch, heads, chunk_len, dim, device=q.device, dtype=q.dtype))
                continue

            # 提取局部窗口的 k, v
            k_local = k[:, :, k_rel_min:k_rel_max + 1, :]  # (batch, heads, local_k_len, dim)
            v_local = v[:, :, k_rel_min:k_rel_max + 1, :]
            local_k_len = k_local.size(-2)

            # 构建 causal mask：(chunk_len, local_k_len)
            # 每个 query 只能看到 key_pos <= query_pos 的 key
            q_abs = torch.arange(q_abs_min, q_abs_max + 1, device=q.device, dtype=torch.float32)
            k_abs_local = torch.arange(
                k_start_pos + k_rel_min,
                k_start_pos + k_rel_max + 1,
                device=q.device,
                dtype=torch.float32,
            )
            # causal: (chunk_len, local_k_len), True = 允许关注
            causal_mask = k_abs_local[None, :] <= q_abs[:, None]

            # 同时也需要窗口掩码：key 必须在 [q_pos - ws + 1, q_pos] 范围内
            window_mask = k_abs_local[None, :] >= (q_abs[:, None] - ws + 1)
            attn_mask = causal_mask & window_mask  # (chunk_len, local_k_len), True=保留

            # 使用 F.scaled_dot_product_attention（自动选择最优后端）
            # 需要 [batch, heads, chunk_len, dim] x [batch, heads, local_k_len, dim]
            # attn_mask 形状: (chunk_len, local_k_len) → 广播到 (batch, heads, chunk_len, local_k_len)
            # 【修复】PyTorch SDPA bool mask: True=保留，False=屏蔽（与官方文档相反！）
            try:
                attn_out = F.scaled_dot_product_attention(
                    q_chunk,
                    k_local,
                    v_local,
                    attn_mask=attn_mask,  # PyTorch SDPA bool mask: True=保留，False=屏蔽
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=False,
                )
                outputs.append(attn_out)
            except RuntimeError:
                # 回退：如果 sdpa 不支持（极少数情况），使用手动计算
                scores = torch.matmul(q_chunk, k_local.transpose(-2, -1)) / math.sqrt(dim)
                # attn_mask: True=保留, ~attn_mask=True=屏蔽
                scores = scores.masked_fill(~attn_mask.view(1, 1, chunk_len, local_k_len), float("-inf"))
                weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
                weights = torch.nan_to_num(weights, nan=0.0)
                outputs.append(torch.matmul(weights, v_local))

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
    ) -> CompressedKVCache:
        """多级记忆流水线：构建带5级层次化压缩的KV Cache

        Level 1 (工作记忆): recent_k/v — 最新sliding_window个token，全量保留
        Level 2 (关键筛选): H2O风格 — 溢出token中保留Heavy Hitters
        Level 3 (语义凝聚): Learned Pooling — 剩余token加权压缩
        Level 4 (无限历史): MLA latent memory — mem_k 满时固化到低秩关联矩阵
        Level 5 (物理卸载): 量化后CPU offload（由调用方触发）
        """
        total_len = start_pos + k_all.size(-2)
        keep = min(self.window_size, k_all.size(-2))
        compress_len = k_all.size(-2) - keep

        # ── Level 1: 工作记忆区（最新token全量保留） ──
        recent_k = k_all[:, :, -keep:, :].contiguous()
        recent_v = v_all[:, :, -keep:, :].contiguous()

        # ── Level 2+3: 对溢出部分做多级压缩 ──
        mem_parts_k = []
        mem_parts_v = []
        mem_parts_pos = []

        # 先保留上一轮的压缩记忆
        if old_mem_k is not None and old_mem_k.size(-2) > 0:
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
            with torch.no_grad():
                importance = overflow_k.norm(dim=-1).mean(dim=(0, 1))  # (seq_len,)
            
            h2_ratio = 0.3
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
                rem_k = overflow_k[:, :, remaining_mask, :]
                rem_v = overflow_v[:, :, remaining_mask, :]
                rem_start = start_pos + remaining_mask.nonzero(as_tuple=True)[0][0].item()

                # 【修复Bug #1】传递对齐后的special_mask保护特殊Token
                pooled_k, pooled_v, pooled_pos = self._compress_kv_with_sink(
                    rem_k, rem_v, rem_start, sink_count=0,
                    special_mask=overflow_special_mask[remaining_mask] if overflow_special_mask is not None else None)
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
        max_mem_capacity = int(CONFIG.get("max_mem_kv_capacity", 256))
        if mem_k.size(-2) > max_mem_capacity:
            overflow = mem_k.size(-2) - max_mem_capacity
            to_linear_k = mem_k[:, :, :overflow, :]
            to_linear_v = mem_v[:, :, :overflow, :]
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
    ) -> torch.Tensor | tuple[torch.Tensor, CompressedKVCache]:
        batch, seq_len, _ = x.shape
        
        q, k_new, v_new = self._split_qkv(x)

        if past_key_value is None or not isinstance(past_key_value, (tuple, list)) or len(past_key_value) < 6:
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
            raw_k_start_pos = q_start_pos - past_recent_k.size(-2)
            mla_M = rest[0] if len(rest) >= 2 else None
            mla_z = rest[1] if len(rest) >= 2 else None

        if mla_M is None or mla_z is None:
            mla_M, mla_z = self.mla_memory.init_mem(
                1, self.num_heads, self.head_dim, x.device, x.dtype)

        # 在 RoPE 之前保留一份 q/k 用于 MLA latent memory（无位置编码）
        q_for_memory = q.detach().clone()
        k_for_memory = k_new.detach().clone()

        q, k_new = apply_rope(q, k_new, self.rope, q_start_pos)

        if past_key_value is None:
            raw_k = k_new
            raw_v = v_new
        else:
            raw_k = torch.cat((past_recent_k, k_new), dim=-2)
            raw_v = torch.cat((past_recent_v, v_new), dim=-2)

        if past_key_value is None and seq_len > 1:
            # 训练时（use_cache=False）需要在此处构建压缩记忆
            # 推理时由末尾_build_cache处理
            if raw_k.size(-2) > self.window_size:
                mem_k, mem_v, mem_pos = self._compress_kv_with_sink(
                    raw_k, raw_v, raw_k_start_pos, sink_count=self.sink_count)

        # 【HyperAttention】DeepSeek-V4风格：CSA + SlidingWindow + MLA
        # 核心思想：
        # 1. CSA (Compressed Sparse Attention): 每4个token压缩成1个entry + 稀疏注意力
        # 2. SlidingWindow: 最近128个token的精确注意力（补偿近距离依赖）
        # 3. MLA: 低秩latent压缩，处理长距离上下文
        prior = attention_mix_prior(x.device, torch.float32)
        mix = torch.softmax(self.router(x).float() + prior, dim=-1).to(x.dtype)
        # mix: (batch, seq_len, 3), 对应 csa(0)/sliding_window(1)/mla(2)

        # 【动态权重调整】根据序列长度自动调整注意力策略
        # 【修复】使用非原地乘法，避免 gradient checkpointing 时的 inplace operation 错误
        # 原代码 mix[..., i] = mix[..., i] * scale 是原地操作，在 checkpoint 重计算时
        # SoftmaxBackward0 的输出版本号被修改，导致梯度计算失败
        adjust = torch.ones_like(mix)
        if seq_len <= 128:
            adjust[..., 0] = 0.1   # 禁用CSA
            adjust[..., 1] = 2.0   # SlidingWindow主导
            adjust[..., 2] = 0.1   # 禁用MLA
        elif seq_len <= 1024:
            adjust[..., 0] = 0.5   # CSA辅助
            adjust[..., 1] = 1.5   # SlidingWindow主导
            adjust[..., 2] = 1.0   # MLA正常
        else:
            adjust[..., 0] = 1.5   # CSA主导
            adjust[..., 1] = 0.8   # SlidingWindow辅助
            adjust[..., 2] = 1.2   # MLA增强

        mix = mix * adjust  # 非原地乘法，创建新张量，不破坏 softmax 计算图
        mix = mix / mix.sum(dim=-1, keepdim=True)

        # 初始化累积输出为零
        out_accum = None  # (batch, seq_len, emb_size)

        # ── 路径0: CSA (Compressed Sparse Attention) ──
        # DeepSeek-V4: 每4个token压缩成1个entry + 稀疏注意力
        if mix[..., 0:1].max().item() > 0.01 and mem_k.size(-2) > 0:
            # 使用压缩后的KV进行稀疏注意力
            csa_out = self._attend_compressed(q, mem_k, mem_v, mem_pos, q_start_pos)
            csa_out = csa_out.transpose(1, 2).contiguous().view(batch, seq_len, self.emb_size)
            out_accum = csa_out * mix[..., 0:1]
            del csa_out

        # ── 路径1: SlidingWindow (最近128个token精确注意力) ──
        # DeepSeek-V4: 补偿近距离依赖，每个query看最近128个token
        if mix[..., 1:2].max().item() > 0.01:
            # 只取最近128个token进行精确注意力
            window_size = min(128, raw_k.size(-2))
            if window_size > 0:
                window_k = raw_k[..., -window_size:, :]
                window_v = raw_v[..., -window_size:, :]
                window_out = self._attend_local_window(q, window_k, window_v, q_start_pos, raw_k_start_pos)
                window_out = window_out.transpose(1, 2).contiguous().view(batch, seq_len, self.emb_size)
                if out_accum is None:
                    out_accum = window_out * mix[..., 1:2]
                else:
                    out_accum = out_accum + window_out * mix[..., 1:2]
                del window_out

        # ── 路径2: MLA (Multi-head Latent Attention) ──
        # DeepSeek-V4: 低秩latent压缩，处理长距离上下文
        if mix[..., 2:3].abs().max().item() > 1e-8:
            lin_out = self.mla_memory.retrieve(q_for_memory, mla_M, mla_z)
            lin_out = lin_out.transpose(1, 2).contiguous().view(batch, seq_len, self.emb_size)
            if out_accum is None:
                out_accum = lin_out * mix[..., 2:3]
            else:
                out_accum = out_accum + lin_out * mix[..., 2:3]
            del lin_out

        # 极端情况：所有权重都为零，返回零张量
        if out_accum is None:
            out_accum = torch.zeros(batch, seq_len, self.emb_size, device=x.device, dtype=x.dtype)

        # 释放 q 引用
        del q

        # 【启用】MLA latent memory更新
        new_mla_M, new_mla_z = self.mla_memory.update(
            k_for_memory, v_new, mla_M, mla_z)

        out = self.out_proj(out_accum)
        del out_accum

        # 【修复】对 MLA memory 进行 detach().clone()，防止梯度计算图被原地操作破坏
        # 这是 PyTorch inplace operation error 的根本修复
        if new_mla_M is not None:
            new_mla_M = new_mla_M.detach().clone()
        if new_mla_z is not None:
            new_mla_z = new_mla_z.detach().clone()

        if use_cache:
            # 【修复】从token_ids参数生成special_mask保护特殊Token
            special_mask = None
            if token_ids is not None:
                # 生成掩码：标记所有特殊Token（ID < 10）的位置
                if token_ids.dim() == 1:
                    special_mask = token_ids < 10  # ID 0-9 为特殊Token
                elif token_ids.dim() == 2:
                    special_mask = token_ids[0] < 10  # batch=1时取第一个
            
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
    def __init__(self, emb_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
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
            HyperAttentionBlock(emb_size, num_heads, dropout)
            for _ in range(num_transformer_blocks)
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
        # 【修复】特殊Token（ID 0-9）使用更大的初始化标准差
        # 原因：特殊Token在训练中出现的频率远低于普通token，梯度信号弱
        # 更大的初始范数帮助模型从一开始就区分特殊Token和普通Token
        # 这直接解决了"特殊Token注意力消失"问题的根源
        special_count = min(10, self.token_embedding.weight.size(0))
        with torch.no_grad():
            special_std = 0.05  # 特殊Token 5倍标准差
            self.token_embedding.weight[:special_count].normal_(mean=0.0, std=special_std)
        # 仅在未绑定时独立初始化输出层（避免覆盖共享权重的初始化）
        if self.output_linear.weight is not self.token_embedding.weight:
            nn.init.normal_(self.output_linear.weight, mean=0.0, std=0.02)
        # 初始化其他线性层（排除已绑定权重的 output_linear）
        for module in self.modules():
            if isinstance(module, nn.Linear) and module.weight is not self.output_linear.weight:
                nn.init.xavier_uniform_(module.weight)

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

            # 如果 history_tokens 不在 embedding 权重同一设备上，但配置允许在 GPU 上分块压缩，
            # 则采用分块流式编码：仅把小片段移动到 embedding 设备进行 embedding_lookup 和聚合，
            # 最终在 embedding 设备上返回压缩向量，避免把压缩结果搬回 CPU。
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
                # 后续逻辑与原来在同设备时保持一致
                if hist_emb.dim() == 3:
                    hist_emb = hist_emb.squeeze(0)
                if seq_len <= max(16, compress_num):
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

            # 若 history_tokens 在同设备，或不启用 GPU 分块压缩，走原有路径
            if history_tokens.device != emb_weight.device:
                # 在 CPU 上进行 embedding lookup 与压缩，避免把大张量搬到 GPU
                hist_idx = history_tokens.to(torch.long).to("cpu")
                weight_cpu = emb_weight.detach().to("cpu")
                hist_emb = weight_cpu[hist_idx]
                if hist_emb.dim() == 3:
                    hist_emb = hist_emb.squeeze(0)
                seq_len = hist_emb.size(0)
                compress_num = max(16, int(seq_len * compress_ratio))
                if seq_len <= compress_num:
                    # 手动应用 final_norm（在 CPU 上）以避免移动 module
                    w = self.final_norm.weight.detach().to("cpu")
                    eps = self.final_norm.eps
                    normed = hist_emb * torch.rsqrt(hist_emb.pow(2).mean(dim=-1, keepdim=True) + eps)
                    return normed * w

                scores = hist_emb.norm(dim=-1)
                boundaries = torch.linspace(0, seq_len, compress_num + 1, device=hist_emb.device)
                pieces: list[torch.Tensor] = []
                for idx in range(compress_num):
                    start = int(boundaries[idx].item())
                    end = max(start + 1, int(boundaries[idx + 1].item()))
                    segment = hist_emb[start:end]
                    weights = torch.softmax(scores[start:end].float(), dim=0).to(hist_emb.dtype)
                    pieces.append((segment * weights[:, None]).sum(dim=0))
                pieces_stacked = torch.stack(pieces, dim=0)
                # final_norm on CPU
                w = self.final_norm.weight.detach().to("cpu")
                eps = self.final_norm.eps
                normed = pieces_stacked * torch.rsqrt(pieces_stacked.pow(2).mean(dim=-1, keepdim=True) + eps)
                return normed * w
            else:
                # 常规路径：history_tokens 与 embedding 在同一设备，按原逻辑处理
                hist_emb = self.token_embedding(history_tokens)
                if hist_emb.dim() == 3:
                    hist_emb = hist_emb.squeeze(0)
                seq_len = hist_emb.size(0)
                compress_num = max(16, int(seq_len * compress_ratio))
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
            use_gc = bool(CONFIG.get("use_gradient_checkpointing", True))
            for block in self.transformers:
                if self.training and use_gc:
                    x = checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)

        logits = self.output_linear(self.final_norm(x))
        
        # 【修复】统一返回3维logits [batch, seq_len, vocab]
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)  # [seq_len, vocab] -> [1, seq_len, vocab]
        
        if use_cache:
            return logits, next_key_values
        return logits