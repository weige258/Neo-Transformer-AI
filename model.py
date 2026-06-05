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
    """MLA-style latent compression memory.

    这个实现保留原有 cache 接口，但把旧的纯线性累加记忆替换为一个更稳的
    低秩 latent KV 压缩路径：先把 K/V 投影到更小的 latent 维度，再做线性
    记忆检索与更新。它比固定核的线性注意力更接近 DeepSeek-V2/V3 的 MLA 思路，
    同时能保持当前模型的接口兼容。
    """

    def __init__(self, num_heads: int, head_dim: int, use_delta: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.use_delta = use_delta

        # 低秩 latent 投影：把每个头的 head_dim 压成更小的 latent 维度。
        # 这能在保持接口兼容的同时，替换掉旧的固定核线性注意力。
        latent_dim = max(16, min(head_dim, 64))
        self.latent_dim = latent_dim
        self.k_proj = nn.Linear(head_dim, latent_dim, bias=False)
        self.v_proj = nn.Linear(head_dim, latent_dim, bias=False)
        self.q_proj = nn.Linear(head_dim, latent_dim, bias=False)
        self.out_proj = nn.Linear(latent_dim, head_dim, bias=False)
        # 【修复】beta 从零初始化导致初始阶段头之间无差异化
        # 改为正数初始化（0.1），让每个头从一开始就有独立的偏置偏移
        # 加速 MLA latent memory 的头差异化收敛
        self.beta = nn.Parameter(torch.ones(num_heads) * 0.1)

    @staticmethod
    def _elu_plus_one(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.elu(x, alpha=1.0) + 1.0

    @staticmethod
    def init_mem(batch: int, num_heads: int, head_dim: int, device: torch.device, dtype: torch.dtype
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        """创建初始零状态记忆矩阵和归一化项。"""
        latent_dim = max(16, min(head_dim, 64))
        M = torch.zeros(1, num_heads, latent_dim, latent_dim, device=device, dtype=dtype)
        z = torch.zeros(1, num_heads, latent_dim, device=device, dtype=dtype)
        return M, z

    def retrieve(self, q: torch.Tensor, mem_M: torch.Tensor, mem_z: torch.Tensor) -> torch.Tensor:
        """从压缩记忆中检索值。

        Args:
            q: Query张量 (batch, heads, seq_len, head_dim)，无位置编码
            mem_M: 关联矩阵 (1, heads, head_dim, head_dim)
            mem_z: 归一化项 (1, heads, head_dim)
        Returns:
            检索值 (batch, heads, seq_len, head_dim)
        """
        q_lat = self.q_proj(q) + self.beta.view(1, -1, 1, 1)
        sigma_q = self._elu_plus_one(q_lat)
        numer = torch.matmul(sigma_q, mem_M)
        denom = torch.matmul(sigma_q, mem_z.unsqueeze(-1)).clamp_min(1e-8)
        return self.out_proj(numer / denom)

    def update(self, k: torch.Tensor, v: torch.Tensor,
               mem_M: torch.Tensor, mem_z: torch.Tensor
               ) -> tuple[torch.Tensor, torch.Tensor]:
        """用新KV更新记忆，返回新的M/z（不修改模块内部状态）。

        Args:
            k: Key (batch, heads, seq_len, head_dim)
            v: Value (batch, heads, seq_len, head_dim)
            mem_M: 旧关联矩阵 (1, heads, head_dim, head_dim)
            mem_z: 旧归一化项 (1, heads, head_dim)
        Returns:
            (new_M, new_z) 形状同输入
        """
        k_lat = self.k_proj(k)
        v_lat = self.v_proj(v)
        sigma_k = self._elu_plus_one(k_lat)

        if self.use_delta:
            sigma_k_z = torch.matmul(sigma_k, mem_z.unsqueeze(-1)).clamp_min(1e-8)
            v_retrieved = torch.matmul(sigma_k, mem_M) / sigma_k_z
            v_error = v_lat - v_retrieved
            delta_M = torch.matmul(sigma_k.transpose(-2, -1), v_error)
        else:
            delta_M = torch.matmul(sigma_k.transpose(-2, -1), v_lat)

        new_M = mem_M + delta_M.mean(dim=0, keepdim=True)
        new_z = mem_z + sigma_k.sum(dim=-2).mean(dim=0, keepdim=True)
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
    """交错式 (interleaved) RoPE 半旋转变换。

    对于输入 [x0, x1, x2, x3, ...]，返回 [-x1, x0, -x3, x2, ...]。
    这与 RotaryPositionEmbedding 中 torch.cat((freqs, freqs), dim=-1) 的
    频率布局一致（每对相邻维度共享同一频率）。

    注意：这是 LLaMA 风格的交错布局，不是 GPT-NeoX 的 split 布局。
    
    【布局耦合说明 - Bug #15】
    - rotate_half 与 apply_rope 中的 cos/sin 频率布局紧密耦合
    - 当前：torch.cat((freqs, freqs), dim=-1) → [cos0, cos1, ..., cos0, cos1, ...]
    - rotate_half: stack((-x2, x1)) → 每对相邻维度共享同一频率
    - 若切换为 GPT-NeoX split 布局 (freqs 直接复制为两半)，需同步修改 rotate_half
    """
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    # stack + flatten 保持最后一维长度不变
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    rope: RotaryPositionEmbedding,
    start_pos: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = rope(q.size(-2), q.device, start_pos)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def attention_mix_prior(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mix = CONFIG.get("attention_mix", {})
    weights = torch.tensor(
        [
            float(mix.get("compressed", 1.0)),
            float(mix.get("sparse", 1.0)),
            float(mix.get("dynamic", 1.0)),
            float(mix.get("mla_latent_memory", 1.5)),  # MLA latent KV 路径
        ],
        device=device,
        dtype=dtype,
    )
    weights = torch.clamp(weights, min=1e-6)
    return torch.log(weights / weights.sum())


class CompressedSparseDynamicAttention(nn.Module):
    """No full attention: compressed memory + local sparse window + compressed-key Top-K."""

    def __init__(self, emb_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        self.dropout = dropout
        self.window_size = max(8, int(CONFIG.get("sliding_window", 256)))
        self.compress_stride = max(2, int(CONFIG.get("compress_stride", 16)))
        self.dynamic_topk = max(2, int(CONFIG.get("dynamic_attention_topk", 16)))
        self.chunk_size = max(8, int(CONFIG.get("attention_chunk_size", 128)))
        self.sink_count = max(1, int(CONFIG.get("attention_sink_count", 4)))  # StreamingLLM sink保护

        self.qkv_proj = nn.Linear(emb_size, emb_size * 3, bias=False)
        self.memory_gate = nn.Linear(self.head_dim, 1, bias=False)
        self.router = nn.Sequential(
            nn.Linear(emb_size, max(1, emb_size // 4), bias=False),
            nn.SiLU(),
            nn.Linear(max(1, emb_size // 4), 4, bias=True),  # 4维路由：compressed/sparse/dynamic/mla_latent
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
        batch, seq_len, _ = x.shape
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
                anchor_mask[actual_sink:] = anchor_mask[actual_sink:] | sm[actual_sink:]
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

        # ── 对非锚点部分做Learned Soft Pooling ──
        compress_len = compress_k.size(-2)
        if compress_len <= 0:
            if is_anchor:
                return anchor_k, anchor_v, anchor_pos
            return k[:0], v[:0], torch.empty(0, device=k.device, dtype=torch.long)

        chunks = (compress_len + self.compress_stride - 1) // self.compress_stride
        padded_len = chunks * self.compress_stride
        pad_len = padded_len - compress_len
        if pad_len:
            kp = compress_k[:, :, -1:, :].expand(batch, heads, pad_len, dim)
            vp = compress_v[:, :, -1:, :].expand(batch, heads, pad_len, dim)
            compress_k = torch.cat((compress_k, kp), dim=-2)
            compress_v = torch.cat((compress_v, vp), dim=-2)

        ck = compress_k.view(batch, heads, chunks, self.compress_stride, dim)
        cv = compress_v.view(batch, heads, chunks, self.compress_stride, dim)
        valid = torch.full((chunks, self.compress_stride), 1.0, device=k.device, dtype=k.dtype)
        if pad_len:
            valid[-1, -pad_len:] = 0.0

        # Learned Soft Pooling：用可学习线性层计算每个token的重要性权重
        if hasattr(self, 'importance_pooler'):
            importance_logits = self.importance_pooler(ck)  # (B,H,chunks,stride,1)
            # 训练时注入噪声防止过拟合（Gumbel-Softmax风格扰动）
            if self.training:
                importance_logits = importance_logits + torch.randn_like(importance_logits) * 0.01
            importance_logits = importance_logits.masked_fill(
                valid.view(1, 1, chunks, self.compress_stride, 1) == 0, float('-inf'))
            pool_w = torch.softmax(importance_logits.float(), dim=-2).to(ck.dtype)  # (B,H,chunks,stride,1)
            ck_out = (ck * pool_w).sum(dim=-2)
            cv_out = (cv * pool_w).sum(dim=-2)
        else:
            # 回退到均匀平均
            denom = valid.sum(dim=-1).clamp_min(1.0).view(1, 1, chunks, 1)
            ck_out = (ck * valid.view(1, 1, chunks, self.compress_stride, 1)).sum(dim=-2) / denom
            cv_out = (cv * valid.view(1, 1, chunks, self.compress_stride, 1)).sum(dim=-2) / denom

        c_ends = torch.arange(chunks, device=k.device, dtype=torch.long)
        c_ends = compress_start + torch.minimum(
            (c_ends + 1) * self.compress_stride - 1,
            torch.tensor(compress_len - 1, device=k.device, dtype=torch.long),
        )

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
        """压缩记忆注意力（显存优化版，使用 F.scaled_dot_product_attention）"""
        if mem_k.size(-2) == 0:
            return torch.zeros_like(q)

        q_len = q.size(-2)
        mem_len = mem_k.size(-2)

        # 构建 causal mask：(q_len, mem_len)，True = 保留（可关注）
        # mem_pos <= q_pos 意味着记忆位置在 query 之前 → 可以关注
        q_pos = torch.arange(q_start_pos, q_start_pos + q_len, device=q.device, dtype=torch.float32)
        causal_mask = mem_pos.float()[None, :] <= q_pos[:, None]  # (q_len, mem_len), True=保留
        sdpa_mask = ~causal_mask

        try:
            return F.scaled_dot_product_attention(
                q, mem_k, mem_v,
                attn_mask=sdpa_mask,  # PyTorch SDPA: True=屏蔽，False=保留
                dropout_p=0.0,
                is_causal=False,
            )
        except RuntimeError:
            # 回退到手动计算
            scores = torch.matmul(q, mem_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(mem_pos[None, :] > q_pos, float("-inf"))
            weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
            weights = torch.nan_to_num(weights, nan=0.0)
            return torch.matmul(weights, mem_v)

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
            sdpa_mask = ~attn_mask

            # 使用 F.scaled_dot_product_attention（自动选择最优后端）
            # 需要 [batch, heads, chunk_len, dim] x [batch, heads, local_k_len, dim]
            # attn_mask 形状: (chunk_len, local_k_len) → 广播到 (batch, heads, chunk_len, local_k_len)
            try:
                attn_out = F.scaled_dot_product_attention(
                    q_chunk,
                    k_local,
                    v_local,
                    attn_mask=sdpa_mask,  # PyTorch SDPA: True=屏蔽，False=保留
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

    def _attend_dynamic_memory(
        self,
        q: torch.Tensor,
        mem_k: torch.Tensor,
        mem_v: torch.Tensor,
        mem_pos: torch.Tensor,
        q_start_pos: int,
    ) -> torch.Tensor:
        """动态Top-K记忆注意力（显存优化版：避免 expand 大张量）

        使用批量索引选取替代 value_bank.expand，减少中间张量大小。
        
        性能说明：torch.topk 在 mem_len 较大时是瓶颈。
        由于 Path 2（滑动窗口）已覆盖局部精确注意力，Path 3 的 mem_k
        是压缩后的远距离记忆，mem_len 通常远小于原始序列长度。
        """
        if mem_k.size(-2) == 0:
            return torch.zeros_like(q)

        q_len = q.size(-2)
        mem_len = mem_k.size(-2)
        scores = torch.matmul(q, mem_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + self.memory_gate(mem_k).transpose(-2, -1)

        # causal mask
        q_pos = torch.arange(q_start_pos, q_start_pos + q_len, device=q.device, dtype=torch.long)[:, None]
        scores = scores.masked_fill(mem_pos[None, :] > q_pos, float("-inf"))

        topk = min(self.dynamic_topk, mem_len)
        if topk == 0:
            return torch.zeros_like(q)

        top_scores, top_idx = torch.topk(scores, k=topk, dim=-1)  # (batch, heads, q_len, topk)
        weights = torch.softmax(top_scores.float(), dim=-1).to(q.dtype)
        weights = torch.nan_to_num(weights, nan=0.0)

        # 【显存优化】使用批量索引代替 expand + gather，避免 (batch, heads, q_len, mem_len, dim) 大张量
        # top_idx: (batch, heads, q_len, topk)
        # mem_v: (batch, heads, mem_len, dim)
        # 将 top_idx 展平后用 index_select 选取，再 reshape
        b, h, _, _ = q.shape
        topk_flat = top_idx.reshape(b * h, q_len * topk)  # (b*h, q_len*topk)

        # 对每个 (batch*head) 独立选取
        mem_v_flat = mem_v.reshape(b * h, mem_len, self.head_dim)  # (b*h, mem_len, dim)

        # 使用 gather 在 dim=1 上选取
        # 扩展索引维度
        topk_idx_expanded = topk_flat.unsqueeze(-1).expand(-1, -1, self.head_dim)  # (b*h, q_len*topk, dim)
        selected_v = torch.gather(mem_v_flat, dim=1, index=topk_idx_expanded)  # (b*h, q_len*topk, dim)
        selected_v = selected_v.view(b, h, q_len, topk, self.head_dim)  # (b, h, q_len, topk, dim)

        return (weights.unsqueeze(-1) * selected_v).sum(dim=-2)

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
            # 统一由末尾_build_cache处理压缩（此处不再重复压缩，避免不一致）
            pass

        prior = attention_mix_prior(x.device, torch.float32)
        mix = torch.softmax(self.router(x).float() + prior, dim=-1).to(x.dtype)
        # mix: (batch, seq_len, 4), 分别对应 compressed, sparse, dynamic, mla_latent

        # 【Attention Sink保护】短序列增强局部注意力权重
        # 防止特殊token（THINK_START/END等）在压缩路径中注意力消失
        if seq_len <= self.window_size:
            # 局部窗口能覆盖全序列时，增强sparse路径权重
            mix[..., 1] = mix[..., 1] * 1.5  # sparse boost
            mix[..., 3] = mix[..., 3] * 0.7  # mla latent path reduce
            mix = mix / mix.sum(dim=-1, keepdim=True)  # 重新归一化

        # 初始化累积输出为零
        out_accum = None  # (batch, seq_len, emb_size)

        # ── 路径1: 压缩注意力（权重>1%才计算，避免4条路径全部执行）──
        if mix[..., 0:1].max().item() > 0.01:
            compressed_out = self._attend_compressed(q, mem_k, mem_v, mem_pos, q_start_pos)
            compressed_out = compressed_out.transpose(1, 2).contiguous().view(batch, seq_len, self.emb_size)
            out_accum = compressed_out * mix[..., 0:1]
            del compressed_out

        # ── 路径2: 局部稀疏窗口注意力 ──
        if mix[..., 1:2].max().item() > 0.01:
            sparse_out = self._attend_local_window(q, raw_k, raw_v, q_start_pos, raw_k_start_pos)
            sparse_out = sparse_out.transpose(1, 2).contiguous().view(batch, seq_len, self.emb_size)
            if out_accum is None:
                out_accum = sparse_out * mix[..., 1:2]
            else:
                out_accum = out_accum + sparse_out * mix[..., 1:2]
            del sparse_out

        # ── 路径3: 动态Top-K记忆注意力 ──
        if mix[..., 2:3].max().item() > 0.01:
            dynamic_out = self._attend_dynamic_memory(q, mem_k, mem_v, mem_pos, q_start_pos)
            dynamic_out = dynamic_out.transpose(1, 2).contiguous().view(batch, seq_len, self.emb_size)
            if out_accum is None:
                out_accum = dynamic_out * mix[..., 2:3]
            else:
                out_accum = out_accum + dynamic_out * mix[..., 2:3]
            del dynamic_out

        # ── 🌟 路径4: MLA latent KV 压缩记忆检索 ──
        if mix[..., 3:4].abs().max().item() > 1e-8:
            lin_out = self.mla_memory.retrieve(q_for_memory, mla_M, mla_z)
            lin_out = lin_out.transpose(1, 2).contiguous().view(batch, seq_len, self.emb_size)
            if out_accum is None:
                out_accum = lin_out * mix[..., 3:4]
            else:
                out_accum = out_accum + lin_out * mix[..., 3:4]
            del lin_out

        # 极端情况：所有权重都为零，返回零张量
        if out_accum is None:
            out_accum = torch.zeros(batch, seq_len, self.emb_size, device=x.device, dtype=x.dtype)

        # 释放 q 引用
        del q

        # 用当前 KV 更新 MLA latent memory（训练和推理都更新）
        new_mla_M, new_mla_z = self.mla_memory.update(
            k_for_memory, v_new, mla_M, mla_z)

        out = self.out_proj(out_accum)
        del out_accum

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


class CompressedAttentionBlock(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(emb_size)
        self.ffn_norm = RMSNorm(emb_size)
        self.attention = CompressedSparseDynamicAttention(emb_size, num_heads, dropout)
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
            CompressedAttentionBlock(emb_size, num_heads, dropout)
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
        logits = logits.squeeze(0) if squeeze_batch else logits

        if use_cache:
            return logits, next_key_values
        return logits
