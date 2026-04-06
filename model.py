from typing import Dict

import torch
from torch import nn


CONFIG: Dict[str, int | float] = {
    "dict_size": 60000,
    "max_length": 256,
    "emb_size": 512,
    "num_heads": 8,
    "num_layers": 8,
    "dropout": 0.1,
    "temperature": 0.8,
}


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

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if emb_size % num_heads != 0:
            raise ValueError("emb_size must be divisible by num_heads.")

        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(emb_size, emb_size, bias=False)
        self.k_proj = nn.Linear(emb_size, emb_size, bias=False)
        self.v_proj = nn.Linear(emb_size, emb_size, bias=False)
        self.out_proj = nn.Linear(emb_size, emb_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionEmbedding(self.head_dim)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden = x.shape
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self._shape(self.q_proj(x))
        k = self._shape(self.k_proj(x))
        v = self._shape(self.v_proj(x))

        cos, sin = self.rope(seq_len=seq_len, device=x.device)
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))
        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)

        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.out_proj(output)


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


class TransformerBlock(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.rms_norm1 = RMSNorm(emb_size)
        self.rms_norm2 = RMSNorm(emb_size)
        self.attention = CausalSelfAttention(emb_size, num_heads, dropout)
        self.feed_forward = SwiGLUFeedForward(emb_size, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.rms_norm1(x))
        x = x + self.feed_forward(self.rms_norm2(x))
        return x


class MainModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        dict_size = int(CONFIG["dict_size"])
        emb_size = int(CONFIG["emb_size"])
        max_length = int(CONFIG["max_length"])
        num_heads = int(CONFIG["num_heads"])
        num_layers = int(CONFIG["num_layers"])
        dropout = float(CONFIG["dropout"])

        self.token_embedding = nn.Embedding(dict_size, emb_size)
        self.position_embedding = nn.Embedding(emb_size, emb_size)
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
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.output_linear.weight, mean=0.0, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear) and module.weight is not self.output_linear.weight:
                nn.init.xavier_uniform_(module.weight)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
            squeeze_batch = True
        elif tokens.dim() == 2:
            squeeze_batch = False
        else:
            raise ValueError("tokens must have shape [seq_len] or [batch, seq_len].")

        seq_len = tokens.size(1)
        if seq_len > int(CONFIG["max_length"]):
            raise ValueError(
                f"Input length {seq_len} exceeds max_length={CONFIG['max_length']}."
            )

        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        x = self.embedding_dropout(x)

        for block in self.transformers:
            x = block(x)

        x = self.final_norm(x)
        logits = self.output_linear(x)
        return logits.squeeze(0) if squeeze_batch else logits
