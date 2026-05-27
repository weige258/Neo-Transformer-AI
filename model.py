from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from config import CONFIG


CompressedKVCache = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 修复：移除nan_to_num，让数值异常暴露以便调试
        # 根据PyTorch最佳实践，NaN/Inf应该被检测而非掩盖
        # 如果上游产生NaN/Inf，应该立即报错中断训练
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: int = 10000) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embedding.")
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        seq_len: int,
        device: torch.device,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        self.window_size = max(8, int(CONFIG.get("sliding_window", 96)))
        self.compress_stride = max(2, int(CONFIG.get("compress_stride", 16)))
        self.dynamic_topk = max(2, int(CONFIG.get("dynamic_attention_topk", 16)))
        self.chunk_size = max(8, int(CONFIG.get("attention_chunk_size", 64)))

        self.qkv_proj = nn.Linear(emb_size, emb_size * 3, bias=False)
        self.memory_gate = nn.Linear(self.head_dim, 1, bias=False)
        self.router = nn.Sequential(
            nn.Linear(emb_size, max(1, emb_size // 4), bias=False),
            nn.SiLU(),
            nn.Linear(max(1, emb_size // 4), 3, bias=True),
        )
        self.out_proj = nn.Linear(emb_size, emb_size, bias=False)
        self.rope = RotaryPositionEmbedding(self.head_dim)

    def _split_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape
        qkv = self.qkv_proj(x).view(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        return qkv[0], qkv[1], qkv[2]

    def _compress_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        start_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, heads, seq_len, dim = k.shape
        if seq_len <= 0:
            empty = k.new_zeros(batch, heads, 0, dim)
            pos = torch.empty(0, device=k.device, dtype=torch.long)
            return empty, empty, pos

        chunks = (seq_len + self.compress_stride - 1) // self.compress_stride
        padded_len = chunks * self.compress_stride
        pad_len = padded_len - seq_len
        if pad_len:
            k_pad = k[:, :, -1:, :].expand(batch, heads, pad_len, dim)
            v_pad = v[:, :, -1:, :].expand(batch, heads, pad_len, dim)
            k = torch.cat((k, k_pad), dim=-2)
            v = torch.cat((v, v_pad), dim=-2)

        k_chunks = k.view(batch, heads, chunks, self.compress_stride, dim)
        v_chunks = v.view(batch, heads, chunks, self.compress_stride, dim)
        valid = torch.full(
            (chunks, self.compress_stride),
            1.0,
            device=k.device,
            dtype=k.dtype,
        )
        if pad_len:
            valid[-1, -pad_len:] = 0.0
        denom = valid.sum(dim=-1).clamp_min(1.0).view(1, 1, chunks, 1)
        mem_k = (k_chunks * valid.view(1, 1, chunks, self.compress_stride, 1)).sum(dim=-2) / denom
        mem_v = (v_chunks * valid.view(1, 1, chunks, self.compress_stride, 1)).sum(dim=-2) / denom

        ends = torch.arange(chunks, device=k.device, dtype=torch.long)
        ends = start_pos + torch.minimum(
            (ends + 1) * self.compress_stride - 1,
            torch.tensor(seq_len - 1, device=k.device, dtype=torch.long),
        )
        return mem_k, mem_v, ends

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

        # 构建 causal mask：(q_len, mem_len)，True = 屏蔽
        q_pos = torch.arange(q_start_pos, q_start_pos + q_len, device=q.device, dtype=torch.float32)
        # mem_pos[None, :] > q_pos[:, None] 意味着 key 在 query 之后 → 屏蔽
        causal_mask = mem_pos.float()[None, :] > q_pos[:, None]  # (q_len, mem_len)

        try:
            return F.scaled_dot_product_attention(
                q, mem_k, mem_v,
                attn_mask=causal_mask,
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
            attn_mask = causal_mask & window_mask  # (chunk_len, local_k_len)

            # 使用 F.scaled_dot_product_attention（自动选择最优后端）
            # 需要 [batch, heads, chunk_len, dim] x [batch, heads, local_k_len, dim]
            # attn_mask 形状: (chunk_len, local_k_len) → 广播到 (batch, heads, chunk_len, local_k_len)
            try:
                attn_out = F.scaled_dot_product_attention(
                    q_chunk,
                    k_local,
                    v_local,
                    attn_mask=attn_mask.logical_not(),  # True = 屏蔽
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=False,
                )
                outputs.append(attn_out)
            except RuntimeError:
                # 回退：如果 sdpa 不支持（极少数情况），使用手动计算
                scores = torch.matmul(q_chunk, k_local.transpose(-2, -1)) / math.sqrt(dim)
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
    ) -> CompressedKVCache:
        total_len = start_pos + k_all.size(-2)
        keep = min(self.window_size, k_all.size(-2))
        compress_len = k_all.size(-2) - keep

        recent_k = k_all[:, :, -keep:, :].contiguous()
        recent_v = v_all[:, :, -keep:, :].contiguous()

        mem_parts_k = []
        mem_parts_v = []
        mem_parts_pos = []
        if old_mem_k is not None and old_mem_k.size(-2) > 0:
            mem_parts_k.append(old_mem_k)
            mem_parts_v.append(old_mem_v)
            mem_parts_pos.append(old_mem_pos)

        if compress_len > 0:
            new_mem_k, new_mem_v, new_mem_pos = self._compress_kv(
                k_all[:, :, :compress_len, :],
                v_all[:, :, :compress_len, :],
                start_pos,
            )
            mem_parts_k.append(new_mem_k)
            mem_parts_v.append(new_mem_v)
            mem_parts_pos.append(new_mem_pos)

        if mem_parts_k:
            mem_k = torch.cat(mem_parts_k, dim=-2)
            mem_v = torch.cat(mem_parts_v, dim=-2)
            mem_pos = torch.cat(mem_parts_pos, dim=0)
        else:
            mem_k = k_all.new_zeros(k_all.size(0), k_all.size(1), 0, k_all.size(-1))
            mem_v = v_all.new_zeros(v_all.size(0), v_all.size(1), 0, v_all.size(-1))
            mem_pos = torch.empty(0, device=k_all.device, dtype=torch.long)

        return recent_k, recent_v, mem_k, mem_v, mem_pos, total_len

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: CompressedKVCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, CompressedKVCache]:
        batch, seq_len, _ = x.shape
        q, k_new, v_new = self._split_qkv(x)

        if past_key_value is None:
            q_start_pos = 0
            raw_k_start_pos = 0
            past_recent_k = past_recent_v = None
            mem_k = k_new.new_zeros(batch, self.num_heads, 0, self.head_dim)
            mem_v = v_new.new_zeros(batch, self.num_heads, 0, self.head_dim)
            mem_pos = torch.empty(0, device=x.device, dtype=torch.long)
        else:
            past_recent_k, past_recent_v, mem_k, mem_v, mem_pos, q_start_pos = past_key_value
            raw_k_start_pos = q_start_pos - past_recent_k.size(-2)

        q, k_new = apply_rope(q, k_new, self.rope, q_start_pos)

        if past_key_value is None:
            raw_k = k_new
            raw_v = v_new
        else:
            raw_k = torch.cat((past_recent_k, k_new), dim=-2)
            raw_v = torch.cat((past_recent_v, v_new), dim=-2)

        if past_key_value is None and seq_len > 1:
            mem_k, mem_v, mem_pos = self._compress_kv(raw_k, raw_v, raw_k_start_pos)

        # 【显存优化】预计算 mix 权重，然后顺序计算三种注意力并累加
        # 避免同时持有三种注意力的中间张量，大幅降低峰值显存
        prior = attention_mix_prior(x.device, torch.float32)
        mix = torch.softmax(self.router(x).float() + prior, dim=-1).to(x.dtype)
        # mix: (batch, seq_len, 3), 分别对应 compressed, sparse, dynamic 权重

        # 初始化累积输出为零
        out_accum = None  # (batch, seq_len, emb_size)

        # ── 路径1: 压缩注意力 ──
        if mix[..., 0:1].abs().max().item() > 1e-8:
            compressed_out = self._attend_compressed(q, mem_k, mem_v, mem_pos, q_start_pos)
            compressed_out = compressed_out.transpose(1, 2).contiguous().view(batch, seq_len, self.emb_size)
            out_accum = compressed_out * mix[..., 0:1]
            del compressed_out

        # ── 路径2: 局部稀疏窗口注意力 ──
        if mix[..., 1:2].abs().max().item() > 1e-8:
            sparse_out = self._attend_local_window(q, raw_k, raw_v, q_start_pos, raw_k_start_pos)
            sparse_out = sparse_out.transpose(1, 2).contiguous().view(batch, seq_len, self.emb_size)
            if out_accum is None:
                out_accum = sparse_out * mix[..., 1:2]
            else:
                out_accum = out_accum + sparse_out * mix[..., 1:2]
            del sparse_out

        # ── 路径3: 动态Top-K记忆注意力 ──
        if mix[..., 2:3].abs().max().item() > 1e-8:
            dynamic_out = self._attend_dynamic_memory(q, mem_k, mem_v, mem_pos, q_start_pos)
            dynamic_out = dynamic_out.transpose(1, 2).contiguous().view(batch, seq_len, self.emb_size)
            if out_accum is None:
                out_accum = dynamic_out * mix[..., 2:3]
            else:
                out_accum = out_accum + dynamic_out * mix[..., 2:3]
            del dynamic_out

        # 极端情况：所有权重都为零，返回零张量
        if out_accum is None:
            out_accum = torch.zeros(batch, seq_len, self.emb_size, device=x.device, dtype=x.dtype)

        # 释放 q 引用（raw_k/raw_v 在 use_cache 时仍需保留）
        del q

        out = self.out_proj(out_accum)
        del out_accum

        if use_cache:
            cache = self._build_cache(
                raw_k,
                raw_v,
                raw_k_start_pos,
                old_mem_k=mem_k if past_key_value is not None else None,
                old_mem_v=mem_v if past_key_value is not None else None,
                old_mem_pos=mem_pos if past_key_value is not None else None,
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
    ) -> torch.Tensor | tuple[torch.Tensor, CompressedKVCache]:
        attn_result = self.attention(
            self.attn_norm(x),
            past_key_value=past_key_value,
            use_cache=use_cache,
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
        num_transformer_blocks = int(CONFIG.get("num_transformer_blocks", 2))
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
        if bool(CONFIG.get("tie_token_embeddings", True)):
            self.output_linear.weight = self.token_embedding.weight

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        if self.output_linear.weight is not self.token_embedding.weight:
            nn.init.normal_(self.output_linear.weight, mean=0.0, std=0.02)
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
                x, present = block(x, past_key_value=past, use_cache=True)
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
