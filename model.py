from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F


CONFIG: Dict[str, int | float] = {
    "dict_size": 60000,
    "emb_size": 512,
    "num_heads": 8,
    "num_layers": 8,
    "dropout": 0.1,
    "temperature": 0.8,
}

KVCache = tuple[torch.Tensor, torch.Tensor]
LinearKVCache = tuple[torch.Tensor, torch.Tensor, int]


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization for dynamic dimensions"""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 检查最后一个维度是否与weight维度匹配
        if x.size(-1) != self.weight.size(0):
            # 如果不匹配，则使用相应部分的weight
            weight = self.weight[:x.size(-1)]
        else:
            weight = self.weight
        
        # 【新增】输入前检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)
        
        rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        
        # 【修改】更安全的clamp
        rms = torch.clamp(rms, min=1e-6, max=1e6)  # 防止过大或过小
        
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


class LinearAttention(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        self.dropout = dropout
        
        self.qkv_proj = nn.Linear(emb_size, emb_size * 3, bias=False)
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
        q, k, v = qkv[0], qkv[1], qkv[2]

        past_len = 0 if past_key_value is None else past_key_value[0].size(-2)
        cos, sin = self.rope(seq_len=seq_len, device=x.device, start_pos=past_len)
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        output = self.out_proj(output)

        if use_cache:
            return output, (k, v)
        return output


class VariableDimAdaptiveFFN(nn.Module):
    """可变维度自适应前馈网络
    
    根据输入特征的统计特性动态调整隐藏层维度，提高模型的表达能力和效率。
    核心思想：复杂输入使用更大维度，简单输入使用较小维度。
    """
    def __init__(self, emb_size: int, dropout: float, base_ffn_ratio: float = 3.0, adaptive_range: float = 0.5) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.base_ffn_ratio = base_ffn_ratio
        self.adaptive_range = adaptive_range
        
        # 基础FFN尺寸
        self.base_ffn_size = int(emb_size * base_ffn_ratio)
        
        # 维度预测器：根据输入特征预测合适的隐藏层维度比例
        self.dim_predictor = nn.Sequential(
            nn.Linear(emb_size, emb_size // 4, bias=False),
            nn.SiLU(),
            nn.Linear(emb_size // 4, 1, bias=False),
            nn.Sigmoid()  # 输出[0, 1]范围的比例因子
        )
        
        # 最大可能的FFN尺寸（用于预分配）
        self.max_ffn_size = int(self.base_ffn_size * (1 + adaptive_range))
        
        # 主FFN网络（使用最大尺寸以保证兼容性）
        self.gate_proj = nn.Linear(emb_size, self.max_ffn_size, bias=False)
        self.up_proj = nn.Linear(emb_size, self.max_ffn_size, bias=False)
        self.down_proj = nn.Linear(self.max_ffn_size, emb_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # 添加RMSNorm以提高数值稳定性
        self.hidden_norm = RMSNorm(self.max_ffn_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, emb_size]
        Returns:
            output: [batch, seq_len, emb_size]
        """
        batch, seq_len, emb_size = x.shape
        
        # 1. 计算自适应维度比例（增加数值稳定性）
        x_mean = x.mean(dim=1)  # [batch, emb_size]
        dim_ratio = self.dim_predictor(x_mean)  # [batch, 1] 范围[0, 1]，已经是Sigmoid输出
        
        # 2. 应用SwiGLU激活（使用完整维度）
        gate_full = torch.nn.functional.silu(self.gate_proj(x))  # [batch, seq_len, max_ffn_size]
        up_full = self.up_proj(x)  # [batch, seq_len, max_ffn_size]
        
        # 3. 应用动态维度裁剪，只保留前actual_size个维度
        # 根据维度比例计算实际使用的FFN大小
        # 将比例映射到 [1-adaptive_range, 1+adaptive_range] 范围内
        actual_ratio = 1.0 + (dim_ratio - 0.5) * 2 * self.adaptive_range  # [1-range, 1+range]
        actual_ratio = torch.clamp(actual_ratio, min=1-self.adaptive_range, max=1+self.adaptive_range)
        
        # 防止 NaN 导致 int() 崩溃
        if torch.isnan(actual_ratio).any():
            actual_ffn_size = self.base_ffn_size
        else:
            actual_ffn_size = int(self.base_ffn_size * actual_ratio.mean().item())  # scalar
        
        # 确保actual_ffn_size不为0
        actual_ffn_size = max(1, actual_ffn_size)  # 确保至少为 1
        
        # 裁剪到实际大小
        gate = gate_full[:, :, :actual_ffn_size]  # [batch, seq_len, actual_size]
        up = up_full[:, :, :actual_ffn_size]      # [batch, seq_len, actual_size]
        
        hidden = gate * up                        # [batch, seq_len, actual_size]
        
        # 4. 对裁剪后的hidden进行RMSNorm，而不是对零填充后的张量
        # 这样可以避免零填充对均值计算的影响
        hidden = self.hidden_norm(hidden)  # [batch, seq_len, actual_size]
        
        # 5. 将hidden扩展到max_ffn_size维度，以适配down_proj
        # 使用零填充，但这次我们不会对零填充进行归一化
        padded_hidden = torch.zeros(batch, seq_len, self.max_ffn_size, device=hidden.device, dtype=hidden.dtype)
        padded_hidden[:, :, :actual_ffn_size] = hidden
        
        # 6. 输出投影 - 确保维度匹配
        # down_proj期望输入是max_ffn_size维度，所以我们需要确保输入是正确维度
        down = self.down_proj(padded_hidden)  # [batch, seq_len, emb_size]
        
        return self.dropout(down)


class AdaptiveAttentionTransformerBlock(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, dropout: float, attention_type: str = "flash") -> None:
        super().__init__()
        self.attention_type = attention_type
        self.rms_norm1 = RMSNorm(emb_size)
        self.rms_norm2 = RMSNorm(emb_size)
        
        if attention_type == "flash":
            self.attention = FlashAttention(emb_size, num_heads, dropout)
        else:
            self.attention = LinearAttention(emb_size, num_heads, dropout)
        
        self.feed_forward = VariableDimAdaptiveFFN(emb_size, dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        past_key_value: KVCache | LinearKVCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache | LinearKVCache]:
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
        num_layers = int(CONFIG.get("num_layers", 8))
        dropout = float(CONFIG["dropout"])

        self.token_embedding = nn.Embedding(dict_size, emb_size)
        self.embedding_dropout = nn.Dropout(dropout)
        
        self.transformers = nn.ModuleList()
        
        for i in range(num_layers):
            attention_type = "flash" if i % 2 == 0 else "linear"
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

    def forward(
        self,
        tokens: torch.Tensor,
        past_key_values: list[KVCache | LinearKVCache | None] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[KVCache | LinearKVCache]]:
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
                if len(first_cache) == 3:
                    past_len = first_cache[2]
                else:
                    past_len = first_cache[0].size(-2)

        x = self.token_embedding(tokens)
        x = self.embedding_dropout(x)

        next_key_values: list[KVCache | LinearKVCache] = []
        if past_key_values is None:
            past_key_values = [None] * len(self.transformers)
        
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
                x = block(x)

        x = self.final_norm(x)
        logits = self.output_linear(x)
        logits = logits.squeeze(0) if squeeze_batch else logits

        if use_cache:
            return logits, next_key_values
        return logits