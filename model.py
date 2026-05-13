from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from config import CONFIG


KVCache = tuple[torch.Tensor, torch.Tensor]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


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
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    rope: RotaryPositionEmbedding,
    start_pos: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = rope(q.size(-2), q.device, start_pos)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def causal_mask(
    q_len: int,
    k_len: int,
    past_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    q_pos = torch.arange(past_len, past_len + q_len, device=device)[:, None]
    k_pos = torch.arange(k_len, device=device)[None, :]
    blocked = k_pos > q_pos
    mask = torch.zeros(q_len, k_len, device=device, dtype=dtype)
    return mask.masked_fill(blocked, float("-inf"))


def attention_mix_prior(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mix = CONFIG.get("attention_mix", {})
    weights = torch.tensor(
        [
            float(mix.get("hybrid", 1.0)),
            float(mix.get("sparse", 1.0)),
            float(mix.get("dynamic", 1.0)),
        ],
        device=device,
        dtype=dtype,
    )
    weights = torch.clamp(weights, min=1e-6)
    return torch.log(weights / weights.sum())


class HybridSparseDynamicAttention(nn.Module):
    """One fused attention unit: full hybrid attention + sparse attention + adaptive key attention."""

    def __init__(self, emb_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        self.dropout = dropout
        self.window_size = int(CONFIG.get("sliding_window", 128))
        self.sparse_stride = max(2, int(CONFIG.get("sparse_stride", 8)))
        self.dynamic_topk = max(4, int(CONFIG.get("dynamic_attention_topk", 32)))

        self.qkv_proj = nn.Linear(emb_size, emb_size * 3, bias=False)
        self.key_score_proj = nn.Linear(self.head_dim, 1, bias=False)
        self.router = nn.Sequential(
            nn.Linear(emb_size, emb_size // 4, bias=False),
            nn.SiLU(),
            nn.Linear(emb_size // 4, 3, bias=True),
        )
        self.out_proj = nn.Linear(emb_size, emb_size, bias=False)
        self.rope = RotaryPositionEmbedding(self.head_dim)

    def _split_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        return qkv[0], qkv[1], qkv[2]

    def _full_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        past_len: int,
    ) -> torch.Tensor:
        if past_len == 0:
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        mask = causal_mask(q.size(-2), k.size(-2), past_len, q.device, q.dtype)
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )

    def _sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        past_len: int,
    ) -> torch.Tensor:
        q_len, k_len = q.size(-2), k.size(-2)
        q_pos = torch.arange(past_len, past_len + q_len, device=q.device)[:, None]
        k_pos = torch.arange(k_len, device=q.device)[None, :]
        local = k_pos >= (q_pos - self.window_size + 1)
        strided = (k_pos % self.sparse_stride) == 0
        self_token = k_pos == q_pos
        allowed = (k_pos <= q_pos) & (local | strided | self_token)
        mask = torch.zeros(q_len, k_len, device=q.device, dtype=q.dtype)
        mask = mask.masked_fill(~allowed, float("-inf"))
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )

    def _dynamic_key_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        past_len: int,
    ) -> torch.Tensor:
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        scores = scores + self.key_score_proj(k).transpose(-2, -1)

        q_len, k_len = q.size(-2), k.size(-2)
        q_pos = torch.arange(past_len, past_len + q_len, device=q.device)[:, None]
        k_pos = torch.arange(k_len, device=q.device)[None, :]
        scores = scores.masked_fill(k_pos > q_pos, float("-inf"))

        topk = min(self.dynamic_topk, k_len)
        top_scores, top_idx = torch.topk(scores, k=topk, dim=-1)
        weights = torch.softmax(top_scores.float(), dim=-1).to(q.dtype)
        weights = torch.nan_to_num(weights, nan=0.0)

        value_bank = v.unsqueeze(2).expand(-1, -1, q_len, -1, -1)
        gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim)
        selected_v = torch.gather(value_bank, dim=3, index=gather_idx)
        return (weights.unsqueeze(-1) * selected_v).sum(dim=-2)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
        batch, seq_len, _ = x.shape
        q, k_new, v_new = self._split_qkv(x)

        past_len = 0
        if past_key_value is not None:
            past_k, past_v = past_key_value
            past_len = past_k.size(-2)
        else:
            past_k = past_v = None

        q, k_new = apply_rope(q, k_new, self.rope, past_len)
        k = k_new if past_k is None else torch.cat((past_k, k_new), dim=-2)
        v = v_new if past_v is None else torch.cat((past_v, v_new), dim=-2)

        full_out = self._full_attention(q, k, v, past_len)
        sparse_out = self._sparse_attention(q, k, v, past_len)
        dynamic_out = self._dynamic_key_attention(q, k, v, past_len)

        full_out = full_out.transpose(1, 2).contiguous().view(batch, seq_len, self.emb_size)
        sparse_out = sparse_out.transpose(1, 2).contiguous().view(batch, seq_len, self.emb_size)
        dynamic_out = dynamic_out.transpose(1, 2).contiguous().view(batch, seq_len, self.emb_size)

        prior = attention_mix_prior(x.device, torch.float32)
        mix = torch.softmax(self.router(x).float() + prior, dim=-1).to(x.dtype)
        out = (
            full_out * mix[..., 0:1]
            + sparse_out * mix[..., 1:2]
            + dynamic_out * mix[..., 2:3]
        )
        out = self.out_proj(out)

        if use_cache:
            return out, (k, v)
        return out


class SwiGLUFFN(nn.Module):
    def __init__(self, emb_size: int, dropout: float) -> None:
        super().__init__()
        hidden = int(emb_size * 8 / 3)
        self.gate_proj = nn.Linear(emb_size, hidden, bias=False)
        self.up_proj = nn.Linear(emb_size, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, emb_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(x))


class HybridAttentionBigBlock(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(emb_size)
        self.ffn_norm = RMSNorm(emb_size)
        self.attention = HybridSparseDynamicAttention(emb_size, num_heads, dropout)
        self.feed_forward = SwiGLUFFN(emb_size, dropout)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
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

        dropout = float(CONFIG.get("dropout", 0.1))
        num_big_blocks = int(CONFIG.get("num_big_blocks", 2))
        if num_big_blocks != 2:
            raise ValueError("This architecture requires CONFIG['num_big_blocks'] == 2.")

        self.token_embedding = nn.Embedding(dict_size, emb_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.transformers = nn.ModuleList(
            [
                HybridAttentionBigBlock(emb_size, num_heads, dropout)
                for _ in range(num_big_blocks)
            ]
        )
        self.final_norm = RMSNorm(emb_size)
        self.output_linear = nn.Linear(emb_size, dict_size, bias=False)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
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
        past_key_values: list[KVCache | None] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[KVCache]]:
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

        next_key_values: list[KVCache] = []
        if use_cache:
            for block, past in zip(self.transformers, past_key_values):
                x, present = block(x, past_key_value=past, use_cache=True)
                next_key_values.append(present)
        else:
            for block in self.transformers:
                if self.training:
                    x = checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)

        logits = self.output_linear(self.final_norm(x))
        logits = logits.squeeze(0) if squeeze_batch else logits

        if use_cache:
            return logits, next_key_values
        return logits
