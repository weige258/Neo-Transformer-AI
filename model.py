from typing import Dict

import torch
from torch import nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F


CONFIG: Dict[str, int | float] = {
    "dict_size": 60000,
    "emb_size": 512,
    "num_heads": 8,
    "num_layers_linear": 4,   # 线性注意力层数
    "num_layers_dynamic": 4, # 标准注意力层数
    "dropout": 0.1,
    "temperature": 1.2,
    # 动态Token选择配置
    "dynamic_token_top_k_ratio": 0.3,
    "attention_sink_tokens": 4,
    # 动态窗口配置（针对6GB显存优化）
    "min_window_size": 256,
    "max_window_size": 1024,
    "window_complexity_threshold": 0.5,
}

KVCache = tuple[torch.Tensor, torch.Tensor]


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
    """
    因果线性注意力，O(n) 复杂度。
    使用 ELU+1 核函数，训练时用 cumsum，推理时用递归状态 (S, Z)。
    """
    def __init__(self, emb_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(emb_size, emb_size * 3, bias=False)
        self.out_proj = nn.Linear(emb_size, emb_size, bias=False)
        # 【关键】线性注意力与 RoPE 数学不兼容，禁用 RoPE
        self.rope = None

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor, past_key_value=None, use_cache: bool = False):
        batch, seq_len, emb_size = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 特征映射
        q = self.feature_map(q) * self.scale  # [B, H, L, D]
        k = self.feature_map(k)               # [B, H, L, D]

        if use_cache:
            # 推理阶段：缓存格式为 (S, Z, past_len)
            # S: [B, H, D, D] 是累积的 k^T v
            # Z: [B, H, D]    是累积的 k
            if past_key_value is not None:
                S, Z, past_len = past_key_value
            else:
                S = torch.zeros(batch, self.num_heads, self.head_dim, self.head_dim,
                                device=x.device, dtype=x.dtype)
                Z = torch.zeros(batch, self.num_heads, self.head_dim,
                                device=x.device, dtype=x.dtype)
                past_len = 0

            # 更新全局状态
            kv_new = torch.einsum('bhld,bhle->bhde', k, v)  # [B, H, D, D]
            S_new = S + kv_new
            Z_new = Z + k.sum(dim=2)  # [B, H, D]

            # 计算输出
            num = torch.einsum('bhld,bhde->bhle', q, S_new)  # [B, H, L, D]
            den = torch.einsum('bhld,bhd->bhl', q, Z_new)  # [B, H, L]
            den = torch.clamp(den, min=1e-6)  # 更安全的做法
            out = num / den.unsqueeze(-1)

            present_key_value = (S_new, Z_new, past_len + seq_len)
        else:
            # 训练阶段：因果 cumsum
            # 正确的因果线性注意力实现，添加chunk机制避免内存爆炸
            seq_len = q.size(2)
            chunk_size = 256  # 与推理路径一致的分块大小
            num_chunks = (seq_len + chunk_size - 1) // chunk_size
            
            # 初始化输出和累积值
            num = torch.zeros_like(q)
            den = torch.zeros(q.size(0), q.size(1), q.size(2), device=q.device, dtype=q.dtype)
            
            # 初始化跨chunk累积状态
            S_carry = torch.zeros(batch, self.num_heads, self.head_dim, self.head_dim,
                                  device=q.device, dtype=q.dtype)  # [B, H, D, D] - 累积的KV乘积
            Z_carry = torch.zeros(batch, self.num_heads, self.head_dim,
                                  device=q.device, dtype=q.dtype)  # [B, H, D] - 累积的K值
            
            # 分块处理
            for i in range(num_chunks):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, seq_len)
                
                # 获取当前块的q, k, v
                q_chunk = q[:, :, start:end, :]
                k_chunk = k[:, :, start:end, :]
                v_chunk = v[:, :, start:end, :]
                
                # 计算当前块的外积 (k^T * v)
                kv_chunk = torch.einsum('bhld,bhle->bhlde', k_chunk, v_chunk)  # [B, H, C, D, D]
                
                # 计算当前块内部的累积
                kv_cumsum_chunk = kv_chunk.cumsum(dim=2)  # [B, H, C, D, D]
                # 加上之前所有块的累积值
                kv_cumsum_chunk = kv_cumsum_chunk + S_carry.unsqueeze(2)  # [B, H, 1, D, D] broadcast to [B, H, C, D, D]
                
                k_cumsum_chunk = k_chunk.cumsum(dim=2)  # [B, H, C, D]
                # 加上之前所有块的累积值
                k_cumsum_chunk = k_cumsum_chunk + Z_carry.unsqueeze(2)  # [B, H, 1, D] broadcast to [B, H, C, D]
                
                # 计算分子和分母
                num_chunk = torch.einsum('bhld,bhlde->bhle', q_chunk, kv_cumsum_chunk)  # [B, H, C, D]
                den_chunk = torch.einsum('bhld,bhld->bhl', q_chunk, k_cumsum_chunk)  # [B, H, C]
                
                # 存储结果
                num[:, :, start:end, :] = num_chunk
                den[:, :, start:end] = den_chunk
                
                # 更新carry状态为当前块的总和，供下一个chunk使用
                S_carry = S_carry + kv_chunk.sum(dim=2)  # [B, H, D, D] - 累积所有之前的KV乘积
                Z_carry = Z_carry + k_chunk.sum(dim=2)   # [B, H, D] - 累积所有之前的K值
            
            den = torch.clamp(den, min=1e-6)  # 更安全的做法
            out = num / den.unsqueeze(-1)

            present_key_value = None

        # 重新排列维度并应用输出投影
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(batch, seq_len, self.num_heads * self.head_dim)
        out = self.out_proj(out)

        if use_cache:
            return out, present_key_value
        return out


class H2OKVCompressor:
    """
    H2O (Heavy Hitter Oracle) KV缓存压缩实现
    只保留对输出贡献最大的少量KV向量
    """
    @staticmethod
    def compress(k_cache: torch.Tensor, v_cache: torch.Tensor, keep_ratio=0.5, protected_tokens=4):
        """
        压缩KV缓存，只保留最重要的向量
        """
        seq_len = k_cache.size(-2)
        if seq_len <= protected_tokens * 2:
            return k_cache, v_cache
        
        num_keep = max(protected_tokens + 1, int(seq_len * keep_ratio))
        
        # 计算重要性得分：使用 K 的 L2 范数（作为 Heavy Hitter 的简化指标）
        with torch.no_grad():
            # [batch, heads, seq_len]
            scores = torch.norm(k_cache, dim=-1)
            
            # 保护最新的几个 token 不被删除
            scores[:, :, -protected_tokens:] = float('inf')
            
            # 找到得分最高的前 num_keep 个索引
            _, indices = torch.topk(scores, num_keep, dim=-1, sorted=True)
            indices = indices.sort(dim=-1).values # 保持时间顺序
            
        # 重新索引获取压缩后的 KV
        k_compressed = k_cache.gather(-2, 
                                     indices.unsqueeze(-1).expand(-1, -1, -1, k_cache.size(-1)))
        v_compressed = v_cache.gather(-2, 
                                     indices.unsqueeze(-1).expand(-1, -1, -1, v_cache.size(-1)))
        
        return k_compressed, v_compressed


class BlockwiseAttention(nn.Module):
    """
    分块局部-全局注意力机制
    
    核心设计:
    1. 将序列分成固定大小的块
    2. 块内使用完整的自注意力（局部精细）
    3. 块间通过全局汇总 token 交互（全局粗粒度）
    4. 可选: 滑动窗口覆盖相邻块边界
    """
    def __init__(
        self,
        emb_size: int,
        num_heads: int,
        dropout: float,
        block_size: int = 256,
        num_global_tokens: int = 4,
        use_sliding_window: bool = True,
        sliding_window_size: int = 128,
    ) -> None:
        super().__init__()
        if emb_size % num_heads != 0:
            raise ValueError("emb_size must be divisible by num_heads.")
        
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.block_size = block_size
        self.num_global_tokens = num_global_tokens
        self.use_sliding_window = use_sliding_window
        self.sliding_window_size = sliding_window_size
        
        # QKV 投影
        self.qkv_proj = nn.Linear(emb_size, emb_size * 3, bias=False)
        self.out_proj = nn.Linear(emb_size, emb_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # RoPE 位置编码
        self.rope = RotaryPositionEmbedding(self.head_dim)
        
        # 全局 token 的专用投影（用于跨块交互）
        if num_global_tokens > 0:
            self.global_q_proj = nn.Linear(emb_size, emb_size, bias=False)
            self.global_k_proj = nn.Linear(emb_size, emb_size, bias=False)
            self.global_v_proj = nn.Linear(emb_size, emb_size, bias=False)
            self.global_out_proj = nn.Linear(emb_size, emb_size, bias=False)
        
        # KV 压缩器（复用 H2O）
        self.kv_compressor = H2OKVCompressor

    def _create_block_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        创建分块局部注意力掩码
        
        掩码规则:
        - 同块内 token 互相可见
        - 全局 token 可见所有 token
        - 所有 token 可见全局 token
        - 滑动窗口内的相邻块 token 可见（可选）
        """
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        
        # 1. 同块内互相可见
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        for b in range(num_blocks):
            start = b * self.block_size
            end = min((b + 1) * self.block_size, seq_len)
            mask[start:end, start:end] = True
        
        # 2. 全局 token 机制
        # 假设前 num_global_tokens 个 token 是全局 token（如 CLS 或特殊标记）
        if self.num_global_tokens > 0:
            global_end = min(self.num_global_tokens, seq_len)
            # 全局 token 可见所有 token
            mask[:global_end, :] = True
            # 所有 token 可见全局 token
            mask[:, :global_end] = True
        
        # 3. 滑动窗口（覆盖相邻块边界）
        if self.use_sliding_window:
            for i in range(seq_len):
                window_start = max(0, i - self.sliding_window_size)
                window_end = min(seq_len, i + self.sliding_window_size + 1)
                mask[i, window_start:window_end] = True
        
        # 4. 保持因果性（只可见当前及之前位置）
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        mask = mask & causal_mask
        
        return mask

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
        batch, seq_len, _ = x.shape
        
        # QKV 投影
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # RoPE 位置编码
        past_len = 0 if past_key_value is None else past_key_value[0].size(-2)
        cos, sin = self.rope(seq_len=seq_len, device=x.device, start_pos=past_len)
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)

        # KV 缓存处理（与 FlashAttentionWithKVCache 相同）
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
            
            # 应用 H2O 压缩
            if k.size(-2) > 1024:
                k, v = self.kv_compressor.compress(k, v, keep_ratio=0.5, protected_tokens=4)

        # 生成分块掩码
        total_len = k.size(-2)  # 包含 past_key_value 的总长度
        if past_key_value is None:
            # 训练/首次前向：使用分块稀疏掩码
            attn_mask = self._create_block_mask(total_len, x.device)
            # 扩展为 [batch, num_heads, seq_len, total_len] 的布尔掩码
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, total_len, total_len]
            attn_mask = attn_mask.expand(batch, self.num_heads, -1, -1)
            
            # 转换为填充掩码（True 表示需要 mask 的位置）
            # scaled_dot_product_attention 需要: True = 需要 mask (不参与注意力)
            # 我们的 mask: True = 可以参与注意力
            # 所以需要取反
            attn_mask = ~attn_mask
        else:
            # 推理时：不使用掩码（因果性由逐 token 生成保证）
            attn_mask = None

        # 使用 FlashAttention（支持自定义掩码）
        if attn_mask is not None:
            # 需要填充掩码时，使用 attn_mask 参数
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False,  # 我们已经通过 attn_mask 实现了因果性
            )
        else:
            # 推理时无掩码
            output = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False,
            )

        # 重新整理输出
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
    """自适应注意力Transformer块
    
    支持多种注意力机制：标准Blockwise注意力、线性注意力，
    支持GRPO（Group Relative Policy Optimization）强化学习训练。
    """
    def __init__(self, emb_size: int, num_heads: int, dropout: float, 
                 top_k_ratio: float = 0.3, attention_type: str = "standard") -> None:
        super().__init__()
        self.attention_type = attention_type
        self.rms_norm1 = RMSNorm(emb_size)
        self.rms_norm2 = RMSNorm(emb_size)
        
        if attention_type == "linear":
            self.attention = LinearAttention(emb_size, num_heads)
        else:
            self.attention = BlockwiseAttention(emb_size, num_heads, dropout)
        
        self.feed_forward = VariableDimAdaptiveFFN(emb_size, dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
        batch, seq_len, emb_size = x.shape
        
        # 使用标准注意力机制处理所有tokens
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
        
        # FFN
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
        num_layers_linear = int(CONFIG.get("num_layers_linear", 6))
        num_layers_dynamic = int(CONFIG.get("num_layers_dynamic", 10))
        dropout = float(CONFIG["dropout"])
        
        # 动态Token选择配置
        top_k_ratio = float(CONFIG.get("dynamic_token_top_k_ratio", 0.3))

        self.token_embedding = nn.Embedding(dict_size, emb_size)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # 构建混合架构：策略优化Transformer + 动态窗口Transformer
        self.transformers = nn.ModuleList()
        
        # 顺序排列：先添加所有线性注意力层，再添加所有标准注意力层
        for _ in range(num_layers_linear):
            self.transformers.append(
                AdaptiveAttentionTransformerBlock(
                    emb_size=emb_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    top_k_ratio=top_k_ratio,
                    attention_type="linear",  # 线性注意力
                )
            )
        
        for _ in range(num_layers_dynamic):
            self.transformers.append(
                AdaptiveAttentionTransformerBlock(
                    emb_size=emb_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    top_k_ratio=top_k_ratio,
                    attention_type="standard",  # 标准注意力
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

        x = self.token_embedding(tokens)
        x = self.embedding_dropout(x)

        next_key_values: list[KVCache] = []
        if past_key_values is None:
            past_key_values = [None] * len(self.transformers)
        
        if use_cache:
            # 使用缓存时，不应用 gradient checkpoint
            for block, past_key_value in zip(self.transformers, past_key_values):
                x, present_key_value = block(
                    x,
                    past_key_value=past_key_value,
                    use_cache=True,
                )
                next_key_values.append(present_key_value)
        else:
            # 不使用缓存时，应用 gradient checkpoint 来节省内存
            def checkpoint_forward(block, x):
                return block(x)
            
            for block in self.transformers:
                x = checkpoint.checkpoint(checkpoint_forward, block, x)

        x = self.final_norm(x)
        logits = self.output_linear(x)
        logits = logits.squeeze(0) if squeeze_batch else logits

        if use_cache:
            return logits, next_key_values
        return logits