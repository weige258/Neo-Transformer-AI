from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from config import CONFIG

KVCache = tuple[torch.Tensor, torch.Tensor]
LinearKVCache = tuple[torch.Tensor, torch.Tensor, int]
AttentionCache = KVCache | LinearKVCache


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization for dynamic dimensions"""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.weight.size(0):
            weight = self.weight[:x.size(-1)]
        else:
            weight = self.weight

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        rms = torch.clamp(rms, min=1e-6, max=1e6)
        x = x / rms
        return x * weight


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
        positions = torch.arange(
            start_pos,
            start_pos + seq_len,
            device=device,
            dtype=torch.float32,
        )
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def _attention_schedule(config: dict[str, Any]) -> list[str]:
    mix = config.get("attention_mix")
    if isinstance(mix, dict) and mix:
        base: list[str] = []
        for name, count in mix.items():
            base.extend([str(name)] * max(0, int(count)))
    else:
        base = (
            ["lightning"] * int(config.get("num_linear_layers", 2))
            + ["flash"] * int(config.get("num_flash_layers", 6))
        )
    if not base:
        base = ["flash"]
    return base * max(1, int(config.get("num_big_blocks", 1)))


def _apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    rope: RotaryPositionEmbedding,
    start_pos: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = rope(seq_len=q.size(-2), device=q.device, start_pos=start_pos)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class LightningAttention(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        self.dropout = dropout
        
        self.qkv_proj = nn.Linear(emb_size, emb_size * 3, bias=False)
        self.gate_proj = nn.Linear(emb_size, emb_size, bias=False)
        self.out_proj = nn.Linear(emb_size, emb_size, bias=False)
        self.rope = RotaryPositionEmbedding(self.head_dim)

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor, past_key_value=None, use_cache: bool = False):
        batch, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        past_len = 0 if past_key_value is None else past_key_value[2]
        cos, sin = self.rope(seq_len=seq_len, device=x.device, start_pos=past_len)
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)

        q = self.feature_map(q)
        k = self.feature_map(k)

        if use_cache:
            if past_key_value is not None:
                S, Z, past_len = past_key_value
            else:
                S = torch.zeros(batch, self.num_heads, self.head_dim, self.head_dim,
                                device=x.device, dtype=x.dtype)
                Z = torch.zeros(batch, self.num_heads, self.head_dim,
                                device=x.device, dtype=x.dtype)
                past_len = 0

            kv_new = torch.einsum('bhld,bhle->bhde', k, v)
            S_new = S + kv_new
            Z_new = Z + k.sum(dim=2)

            num = torch.einsum('bhld,bhde->bhle', q, S_new)
            den = torch.einsum('bhld,bhd->bhl', q, Z_new)
            den = torch.clamp(den, min=1e-6)
            out = num / den.unsqueeze(-1)

            present_key_value = (S_new, Z_new, past_len + seq_len)
        else:
            chunk_size = 256
            num_chunks = (seq_len + chunk_size - 1) // chunk_size
            
            num = torch.zeros_like(q)
            den = torch.zeros(q.size(0), q.size(1), q.size(2), device=q.device, dtype=q.dtype)
            
            S_carry = torch.zeros(batch, self.num_heads, self.head_dim, self.head_dim,
                                  device=q.device, dtype=q.dtype)
            Z_carry = torch.zeros(batch, self.num_heads, self.head_dim,
                                  device=q.device, dtype=q.dtype)
            
            for i in range(num_chunks):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, seq_len)
                
                q_chunk = q[:, :, start:end, :]
                k_chunk = k[:, :, start:end, :]
                v_chunk = v[:, :, start:end, :]
                
                kv_chunk = torch.einsum('bhld,bhle->bhlde', k_chunk, v_chunk)
                kv_cumsum_chunk = kv_chunk.cumsum(dim=2)
                kv_cumsum_chunk = kv_cumsum_chunk + S_carry.unsqueeze(2)
                
                k_cumsum_chunk = k_chunk.cumsum(dim=2)
                k_cumsum_chunk = k_cumsum_chunk + Z_carry.unsqueeze(2)
                
                num_chunk = torch.einsum('bhld,bhlde->bhle', q_chunk, kv_cumsum_chunk)
                den_chunk = torch.einsum('bhld,bhld->bhl', q_chunk, k_cumsum_chunk)
                
                num[:, :, start:end, :] = num_chunk
                den[:, :, start:end] = den_chunk
                
                S_carry = S_carry + kv_chunk.sum(dim=2)
                Z_carry = Z_carry + k_chunk.sum(dim=2)
            
            den = torch.clamp(den, min=1e-6)
            out = num / den.unsqueeze(-1)

            present_key_value = None

        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(batch, seq_len, self.num_heads * self.head_dim)
        out = out * torch.sigmoid(self.gate_proj(x))
        out = self.out_proj(out)

        if use_cache:
            return out, present_key_value
        return out


class FlashAttention(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        self.dropout = dropout
        
        self.qkv_proj = nn.Linear(emb_size, emb_size * 3, bias=False)
        self.out_proj = nn.Linear(emb_size, emb_size, bias=False)
        self.rope = RotaryPositionEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor, past_key_value=None, use_cache: bool = False):
        batch, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k_new, v_new = qkv[0], qkv[1], qkv[2]

        past_len = 0
        if past_key_value is not None:
            past_k, past_v = past_key_value
            past_len = past_k.size(-2)
            v = torch.cat([past_v, v_new], dim=-2)
        else:
            past_k = None
            v = v_new
        
        q, k_new = _apply_rope(q, k_new, self.rope, past_len)
        
        k = k_new if past_k is None else torch.cat([past_k, k_new], dim=-2)

        output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=past_key_value is None,
        )

        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        output = self.out_proj(output)

        if use_cache:
            return output, (k, v)
        return output


class SlidingWindowAttention(FlashAttention):
    def __init__(self, emb_size: int, num_heads: int, dropout: float, window_size: int):
        super().__init__(emb_size, num_heads, dropout)
        self.window_size = max(1, int(window_size))

    def forward(self, x: torch.Tensor, past_key_value=None, use_cache: bool = False):
        batch, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k_new, v_new = qkv[0], qkv[1], qkv[2]

        past_len = 0
        if past_key_value is not None:
            past_k, past_v = past_key_value
            past_len = past_k.size(-2)
        else:
            past_k = past_v = None

        q, k_new = _apply_rope(q, k_new, self.rope, past_len)
        k_all = k_new if past_k is None else torch.cat([past_k, k_new], dim=-2)
        v_all = v_new if past_v is None else torch.cat([past_v, v_new], dim=-2)

        if use_cache and seq_len == 1:
            k_window = k_all[:, :, -self.window_size :, :]
            v_window = v_all[:, :, -self.window_size :, :]
            output = F.scaled_dot_product_attention(
                q,
                k_window,
                v_window,
                dropout_p=0.0,
                is_causal=False,
            )
        else:
            total_len = k_all.size(-2)
            q_pos = torch.arange(past_len, past_len + seq_len, device=x.device)[:, None]
            k_pos = torch.arange(total_len, device=x.device)[None, :]
            blocked = (k_pos > q_pos) | (k_pos < (q_pos - self.window_size + 1))
            attn_mask = torch.zeros(seq_len, total_len, device=x.device, dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(blocked, float("-inf"))
            output = F.scaled_dot_product_attention(
                q,
                k_all,
                v_all,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )

        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        output = self.out_proj(output)

        if use_cache:
            return output, (k_all, v_all)
        return output


class LatentCompressedAttention(FlashAttention):
    def __init__(self, emb_size: int, num_heads: int, dropout: float, stride: int):
        super().__init__(emb_size, num_heads, dropout)
        self.stride = max(1, int(stride))
        self.mix_proj = nn.Linear(emb_size * 2, emb_size, bias=False)

    def _compress(self, x: torch.Tensor) -> torch.Tensor:
        batch, heads, seq_len, dim = x.shape
        chunks = (seq_len + self.stride - 1) // self.stride
        padded_len = chunks * self.stride
        if padded_len != seq_len:
            pad = x[:, :, -1:, :].expand(batch, heads, padded_len - seq_len, dim)
            x = torch.cat([x, pad], dim=-2)
        return x.view(batch, heads, chunks, self.stride, dim).mean(dim=-2)

    def forward(self, x: torch.Tensor, past_key_value=None, use_cache: bool = False):
        batch, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k_new, v_new = qkv[0], qkv[1], qkv[2]

        past_len = 0
        if past_key_value is not None:
            past_k, past_v = past_key_value
            past_len = past_k.size(-2)
        else:
            past_k = past_v = None

        q, k_new = _apply_rope(q, k_new, self.rope, past_len)
        k_all = k_new if past_k is None else torch.cat([past_k, k_new], dim=-2)
        v_all = v_new if past_v is None else torch.cat([past_v, v_new], dim=-2)
        k_latent = self._compress(k_all)
        v_latent = self._compress(v_all)

        scores = torch.matmul(q, k_latent.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if not (use_cache and seq_len == 1):
            total_len = k_all.size(-2)
            chunk_start = torch.arange(k_latent.size(-2), device=x.device) * self.stride
            chunk_start = chunk_start.clamp(max=total_len - 1)[None, :]
            q_pos = torch.arange(past_len, past_len + seq_len, device=x.device)[:, None]
            scores = scores.masked_fill(chunk_start > q_pos, float("-inf"))

        weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        weights = torch.nan_to_num(weights, nan=0.0)
        compressed = torch.matmul(weights, v_latent)

        local_size = min(k_all.size(-2), self.stride)
        if use_cache and seq_len == 1:
            local = F.scaled_dot_product_attention(
                q,
                k_all[:, :, -local_size:, :],
                v_all[:, :, -local_size:, :],
                dropout_p=0.0,
                is_causal=False,
            )
        else:
            total_len = k_all.size(-2)
            q_pos = torch.arange(past_len, past_len + seq_len, device=x.device)[:, None]
            k_pos = torch.arange(total_len, device=x.device)[None, :]
            blocked = (k_pos > q_pos) | (k_pos < (q_pos - self.stride + 1))
            attn_mask = torch.zeros(seq_len, total_len, device=x.device, dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(blocked, float("-inf"))
            local = F.scaled_dot_product_attention(
                q,
                k_all,
                v_all,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
            )

        output = torch.cat([compressed, local], dim=-1)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        output = self.mix_proj(output)

        if use_cache:
            return output, (k_all, v_all)
        return output


class VariableDimAdaptiveFFN(nn.Module):
    def __init__(
        self,
        emb_size: int,
        dropout: float,
        base_ffn_ratio: float = 3.0,
        adaptive_range: float = 0.5,
    ) -> None:
        super().__init__()
        self.base_ffn_size = int(emb_size * base_ffn_ratio)
        self.adaptive_range = adaptive_range
        self.max_ffn_size = int(self.base_ffn_size * (1 + adaptive_range))
        self.dim_predictor = nn.Sequential(
            nn.Linear(emb_size, max(1, emb_size // 4), bias=False),
            nn.SiLU(),
            nn.Linear(max(1, emb_size // 4), 1, bias=False),
            nn.Sigmoid(),
        )
        self.gate_proj = nn.Linear(emb_size, self.max_ffn_size, bias=False)
        self.up_proj = nn.Linear(emb_size, self.max_ffn_size, bias=False)
        self.down_proj = nn.Linear(self.max_ffn_size, emb_size, bias=False)
        self.hidden_norm = RMSNorm(self.max_ffn_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim_ratio = self.dim_predictor(x.mean(dim=1))
        actual_ratio = 1.0 + (dim_ratio - 0.5) * 2 * self.adaptive_range
        actual_ratio = torch.nan_to_num(actual_ratio, nan=1.0).clamp(
            min=1 - self.adaptive_range,
            max=1 + self.adaptive_range,
        )
        actual_ffn_size = int(self.base_ffn_size * actual_ratio.mean().item())
        actual_ffn_size = max(1, min(actual_ffn_size, self.max_ffn_size))

        gate = F.silu(F.linear(x, self.gate_proj.weight[:actual_ffn_size, :]))
        up = F.linear(x, self.up_proj.weight[:actual_ffn_size, :])
        hidden = self.hidden_norm(gate * up)
        down = F.linear(hidden, self.down_proj.weight[:, :actual_ffn_size])
        return self.dropout(down)


class AdaptiveAttentionTransformerBlock(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, dropout: float, attention_type: str = "flash") -> None:
        super().__init__()
        self.attention_type = attention_type
        self.rms_norm1 = RMSNorm(emb_size)
        self.rms_norm2 = RMSNorm(emb_size)
        
        if attention_type == "flash":
            self.attention = FlashAttention(emb_size, num_heads, dropout)
        elif attention_type == "sliding":
            self.attention = SlidingWindowAttention(
                emb_size,
                num_heads,
                dropout,
                int(CONFIG.get("sliding_window", 128)),
            )
        elif attention_type in {"latent", "latent_compress", "compressed"}:
            self.attention = LatentCompressedAttention(
                emb_size,
                num_heads,
                dropout,
                int(CONFIG.get("latent_compress_stride", 8)),
            )
        else:
            self.attention = LightningAttention(emb_size, num_heads, dropout)
        
        self.feed_forward = VariableDimAdaptiveFFN(emb_size, dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        past_key_value: AttentionCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, AttentionCache]:
        attn_output = self.attention(
            self.rms_norm1(x),
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        
        if use_cache:
            attention_output, present_key_value = attn_output
        else:
            attention_output = attn_output
            present_key_value = None
        
        x = x + attention_output
        
        ff_output = self.feed_forward(self.rms_norm2(x))
        x = x + ff_output
        
        if use_cache:
            return x, present_key_value
        return x


class MainModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        dict_size = int(CONFIG["dict_size"])
        emb_size = int(CONFIG["emb_size"])
        num_heads = int(CONFIG["num_heads"])
        if emb_size % num_heads != 0:
            raise ValueError("emb_size must be divisible by num_heads.")

        attention_types = _attention_schedule(CONFIG)
        dropout = float(CONFIG["dropout"])

        self.token_embedding = nn.Embedding(dict_size, emb_size)
        self.embedding_dropout = nn.Dropout(dropout)
        
        self.transformers = nn.ModuleList()
        
        for attention_type in attention_types:
            self.transformers.append(
                AdaptiveAttentionTransformerBlock(
                    emb_size=emb_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    attention_type=attention_type,
                )
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

    def compress_history_vectors(self, history_tokens: torch.Tensor, compress_ratio: float = None) -> torch.Tensor:
        """Compress history embeddings with unlabeled score-weighted segments."""
        if compress_ratio is None:
            compress_ratio = float(CONFIG.get("compress_ratio", 0.3))
        
        with torch.no_grad():
            hist_emb = self.token_embedding(history_tokens)
            if hist_emb.dim() == 1:
                hist_emb = hist_emb.unsqueeze(0)
            seq_len, emb_size = hist_emb.shape
            compress_num = max(16, int(seq_len * compress_ratio))
            
            if seq_len <= compress_num:
                return self.final_norm(hist_emb)
            
            scores = torch.norm(hist_emb, dim=-1)
            boundaries = torch.linspace(0, seq_len, compress_num + 1, device=hist_emb.device)
            compress_vectors = []
            for i in range(compress_num):
                start = int(boundaries[i].item())
                end = max(start + 1, int(boundaries[i + 1].item()))
                segment = hist_emb[start:end]
                segment_scores = scores[start:end].unsqueeze(-1)
                weights = torch.softmax(segment_scores.float(), dim=0).to(hist_emb.dtype)
                compress_vectors.append((segment * weights).sum(dim=0))

            compress_tensor = torch.stack(compress_vectors)
            return self.final_norm(compress_tensor)

    def forward(
        self,
        tokens: torch.Tensor,
        past_key_values: list[AttentionCache | None] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[AttentionCache]]:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
            squeeze_batch = True
        elif tokens.dim() == 2:
            squeeze_batch = False
        else:
            raise ValueError("tokens must have shape [seq_len] or [batch, seq_len].")

        x = self.token_embedding(tokens)
        x = self.embedding_dropout(x)

        next_key_values: list[AttentionCache] = []
        if past_key_values is None:
            past_key_values = [None] * len(self.transformers)
        elif len(past_key_values) != len(self.transformers):
            raise ValueError("past_key_values length must match transformer layer count.")

        if use_cache:
            for block, past_key_value in zip(self.transformers, past_key_values):
                x, present_key_value = block(
                    x,
                    past_key_value=past_key_value,
                    use_cache=True,
                )
                next_key_values.append(present_key_value)
        else:
            for block in self.transformers:
                x = checkpoint(block, x, use_reentrant=False)

        x = self.final_norm(x)
        logits = self.output_linear(x)
        logits = logits.squeeze(0) if squeeze_batch else logits

        if use_cache:
            return logits, next_key_values
        return logits
