from typing import Dict

import torch
from torch import nn


CONFIG: Dict[str, int | float] = {
    "dict_size": 60000,
    "max_length": 256,
    "emb_size": 1024,
    "num_heads": 8,
    "num_layers": 16,
    "dropout": 0.1,
    "temperature": 0.8,
}

KVCache = tuple[torch.Tensor, torch.Tensor]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


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


class SwiGLUFeedForward(nn.Module):
    def __init__(self, emb_size: int, dropout: float) -> None:
        super().__init__()
        ffn_size = emb_size * 4
        self.gate_proj = nn.Linear(emb_size, ffn_size, bias=False)
        self.up_proj = nn.Linear(emb_size, ffn_size, bias=False)
        self.down_proj = nn.Linear(ffn_size, emb_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        return self.dropout(x)


class MultiHeadFeedForward(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if emb_size % num_heads != 0:
            raise ValueError("emb_size must be divisible by num_heads.")

        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        self.multi_head_ffn = nn.ModuleList(
            [
                SwiGLUFeedForward(self.head_dim, dropout)
                for _ in range(num_heads)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        split_heads = torch.chunk(x, self.num_heads, dim=-1)
        head_outputs = [
            ffn(head_x)
            for ffn, head_x in zip(self.multi_head_ffn, split_heads)
        ]
        return torch.cat(head_outputs, dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if emb_size % num_heads != 0:
            raise ValueError("emb_size must be divisible by num_heads.")

        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(emb_size, emb_size * 3, bias=False)
        self.out_proj = nn.Linear(emb_size, emb_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionEmbedding(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
        batch, seq_len, _ = x.shape

        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        past_len = 0 if past_key_value is None else past_key_value[0].size(-2)
        cos, sin = self.rope(seq_len=seq_len, device=x.device, start_pos=past_len)
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        total_kv_len = k.size(-2)

        if seq_len > 1 or past_len == 0:
            key_positions = torch.arange(total_kv_len, device=x.device)
            query_positions = torch.arange(
                past_len,
                past_len + seq_len,
                device=x.device,
            )
            causal_mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0),
                float("-inf"),
            )

        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)

        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        output = self.out_proj(output)

        if use_cache:
            return output, (k, v)
        return output


class TransformerBlock(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.rms_norm1 = RMSNorm(emb_size)
        self.rms_norm2 = RMSNorm(emb_size)
        self.attention = CausalSelfAttention(emb_size, num_heads, dropout)
        self.feed_forward = MultiHeadFeedForward(emb_size, num_heads, dropout)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
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
        x = x + self.feed_forward(self.rms_norm2(x))

        if use_cache:
            return x, present_key_value
        return x


class MainModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        dict_size = int(CONFIG["dict_size"])
        emb_size = int(CONFIG["emb_size"])
        num_heads = int(CONFIG["num_heads"])
        num_layers = int(CONFIG["num_layers"])
        dropout = float(CONFIG["dropout"])

        self.token_embedding = nn.Embedding(dict_size, emb_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.transformers = nn.ModuleList(
            [
                TransformerBlock(
                    emb_size=emb_size,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
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

        seq_len = tokens.size(1)
        past_len = 0
        if past_key_values:
            first_cache = past_key_values[0]
            if first_cache is not None:
                past_len = first_cache[0].size(-2)

        total_len = seq_len + past_len
        if total_len > int(CONFIG["max_length"]):
            raise ValueError(
                f"Input length {total_len} exceeds max_length={CONFIG['max_length']}."
            )

        x = self.token_embedding(tokens)
        x = self.embedding_dropout(x)

        next_key_values: list[KVCache] = []
        if past_key_values is None:
            past_key_values = [None] * len(self.transformers)

        for block, past_key_value in zip(self.transformers, past_key_values):
            if use_cache:
                x, present_key_value = block(
                    x,
                    past_key_value=past_key_value,
                    use_cache=True,
                )
                next_key_values.append(present_key_value)
            else:
                x = block(x)

        x = self.final_norm(x)
        logits = self.output_linear(x)
        logits = logits.squeeze(0) if squeeze_batch else logits

        if use_cache:
            return logits, next_key_values
        return logits
