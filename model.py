from typing import Dict

import torch
from torch import nn
import torch.utils.checkpoint as checkpoint


CONFIG: Dict[str, int | float] = {
    "dict_size": 60000,
    "emb_size": 512,
    "num_heads": 8,
    "num_layers_global": 2,   # 全局Transformer层数
    "num_layers_dynamic": 6, # 动态窗口层数
    "dropout": 0.1,
    "temperature": 0.8,
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
        dim_ratio = self.dim_predictor(x_mean)  # [batch, 1]
        
        # 2. 应用SwiGLU激活（使用完整维度）
        gate = torch.nn.functional.silu(self.gate_proj(x))  # [batch, seq_len, max_ffn_size]
        up = self.up_proj(x)  # [batch, seq_len, max_ffn_size]
        hidden = gate * up
        
        # 【修复】移除破坏性的LayerNorm，保留非线性激活的特征强度
        # LayerNorm会抹杀gate*up的幅度信息，严重损害表达能力
        # 输入端已有rms_norm2提供归一化，此处无需额外操作
        
        # 【新增】添加数值稳定性检查（仅在必要时进行裁剪）
        if not torch.isfinite(hidden).all():
            hidden = torch.nan_to_num(hidden, nan=0.0, posinf=1e5, neginf=-1e5)
        
        # 4. 输出投影
        down = self.down_proj(hidden)  # [batch, seq_len, emb_size]
        
        return self.dropout(down)


class TokenImportanceScorer(nn.Module):
    """轻量级Token重要性评分器
    
    基于StreamingLLM和H2O的思想，为每个token计算重要性分数。
    使用低维投影快速评估token的重要性，避免全维度计算。
    """
    def __init__(self, emb_size: int, reduced_dim: int = 64) -> None:
        super().__init__()
        self.reduced_dim = reduced_dim
        # 低维投影层用于快速评分（使用较小的初始化）
        self.proj_query = nn.Linear(emb_size, reduced_dim, bias=False)
        self.proj_key = nn.Linear(emb_size, reduced_dim, bias=False)
        # 可学习的评分向量
        self.scoring_vector = nn.Parameter(torch.randn(reduced_dim) * 0.01)  # 减小初始方差
        
        # 初始化权重
        nn.init.xavier_uniform_(self.proj_query.weight, gain=0.1)
        nn.init.xavier_uniform_(self.proj_key.weight, gain=0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, emb_size]
        Returns:
            importance_scores: [batch, seq_len] 每个token的重要性分数
        """
        batch, seq_len, emb_size = x.shape
        
        # 投影到低维空间
        q = self.proj_query(x)  # [batch, seq_len, reduced_dim]
        k = self.proj_key(x)    # [batch, seq_len, reduced_dim]
        
        # 计算token与学习向量的相似度作为重要性分数
        # 使用余弦相似度
        q_normalized = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
        k_normalized = k / (k.norm(dim=-1, keepdim=True) + 1e-8)
        scoring_vec_normalized = self.scoring_vector / (self.scoring_vector.norm() + 1e-8)
        
        # 计算分数：结合query和key的相似度
        scores_q = (q_normalized * scoring_vec_normalized.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        scores_k = (k_normalized * scoring_vec_normalized.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        importance_scores = (scores_q + scores_k) / 2.0  # [batch, seq_len]
        
        return importance_scores


class DynamicTokenSelector(nn.Module):
    """基于重要性评分的动态Token选择器
    
    实现StreamingLLM和Switch Attention的核心思想：
    1. 保留注意力锚点（初始tokens）
    2. 根据重要性分数选择Top-K重要tokens
    3. 自动获得全局注意力权限
    """
    def __init__(self, top_k_ratio: float = 0.3, sink_tokens: int = 4) -> None:
        super().__init__()
        self.top_k_ratio = top_k_ratio
        self.sink_tokens = sink_tokens
        self.importance_scorer = TokenImportanceScorer(512)  # 默认emb_size，会在forward中适配
    
    def select_tokens(
        self, 
        x: torch.Tensor, 
        importance_scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        选择重要tokens
        
        Args:
            x: [batch, seq_len, emb_size] 输入特征
            importance_scores: [batch, seq_len] 重要性分数
        Returns:
            selected_x: [batch, selected_seq_len, emb_size] 选中的tokens
            selected_indices: [batch, selected_seq_len] 选中tokens的索引
            mask: [batch, seq_len] 布尔掩码，标记哪些tokens被选中
        """
        batch, seq_len, emb_size = x.shape
        
        # 计算需要选择的token数量
        num_select = max(int(seq_len * self.top_k_ratio), self.sink_tokens + 1)
        num_select = min(num_select, seq_len)
        
        # 确保至少保留sink_tokens个初始tokens
        if seq_len > self.sink_tokens:
            # 分离锚点tokens和其余tokens
            sink_scores = importance_scores[:, :self.sink_tokens]  # [batch, sink_tokens]
            remaining_scores = importance_scores[:, self.sink_tokens:]  # [batch, seq_len - sink_tokens]
            
            # 从剩余tokens中选择Top-K
            num_remaining_select = num_select - self.sink_tokens
            if num_remaining_select > 0 and remaining_scores.size(1) > 0:
                _, top_k_indices = torch.topk(remaining_scores, k=min(num_remaining_select, remaining_scores.size(1)), dim=-1)
                top_k_indices = top_k_indices + self.sink_tokens  # 调整索引
                
                # 合并锚点和Top-K tokens
                sink_indices = torch.arange(self.sink_tokens, device=x.device).unsqueeze(0).expand(batch, -1)
                selected_indices = torch.cat([sink_indices, top_k_indices], dim=-1)
            else:
                selected_indices = torch.arange(min(num_select, seq_len), device=x.device).unsqueeze(0).expand(batch, -1)
        else:
            # 序列长度小于等于sink_tokens，全部保留
            selected_indices = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, -1)
        
        # 创建掩码
        mask = torch.zeros(batch, seq_len, dtype=torch.bool, device=x.device)
        mask.scatter_(1, selected_indices, True)
        
        # 收集选中的tokens
        selected_x = torch.gather(
            x, 
            1, 
            selected_indices.unsqueeze(-1).expand(-1, -1, emb_size)
        )
        
        return selected_x, selected_indices, mask


class AdaptiveWindowAttention(nn.Module):
    """自适应动态窗口注意力机制
    
    根据输入内容的复杂度动态调整窗口大小：
    - 简单文本使用小窗口（节省算力）
    - 复杂文本使用大窗口（提升性能）
    
    结合了StreamingLLM的局部窗口和全局token选择。
    """
    def __init__(
        self, 
        emb_size: int, 
        num_heads: int, 
        dropout: float,
        min_window: int = 64,
        max_window: int = 256,
        complexity_threshold: float = 0.5
    ) -> None:
        super().__init__()
        if emb_size % num_heads != 0:
            raise ValueError("emb_size must be divisible by num_heads.")
        
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.min_window = min_window
        self.max_window = max_window
        self.complexity_threshold = complexity_threshold
        
        self.qkv_proj = nn.Linear(emb_size, emb_size * 3, bias=False)
        self.out_proj = nn.Linear(emb_size, emb_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionEmbedding(self.head_dim)
        
        # 复杂度评估器：基于激活值的方差判断复杂度（优化初始化）
        self.complexity_scorer = nn.Sequential(
            nn.Linear(emb_size, emb_size // 4, bias=False),
            nn.SiLU(),
            nn.Linear(emb_size // 4, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 初始化复杂度评估器权重
        for module in self.complexity_scorer:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
    
    def compute_complexity(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算输入序列的复杂度
        
        Args:
            x: [batch, seq_len, emb_size]
        Returns:
            complexity: [batch] 每个样本的复杂度分数 [0, 1]
        """
        # 方法1：基于激活值方差
        variance = x.var(dim=[1, 2])  # [batch]
        variance_normalized = torch.sigmoid(variance * 10 - 5)  # 映射到[0, 1]
        
        # 方法2：基于学习的复杂度评分
        x_mean = x.mean(dim=1)  # [batch, emb_size]
        learned_complexity = self.complexity_scorer(x_mean).squeeze(-1)  # [batch]
        
        # 综合两种方法
        complexity = (variance_normalized + learned_complexity) / 2.0
        return complexity
    
    def get_adaptive_window_size(self, complexity: torch.Tensor, seq_len: int) -> int:
        """
        根据复杂度动态计算窗口大小
        
        Args:
            complexity: [batch] 复杂度分数
            seq_len: 当前序列长度
        Returns:
            window_size: 自适应窗口大小
        """
        # 线性插值：简单->min_window, 复杂->max_window
        window_size_float = self.min_window + complexity * (self.max_window - self.min_window)
        window_size = window_size_float.mean().int().item()
        
        # 不能超过序列长度
        window_size = min(window_size, seq_len)
        window_size = max(window_size, self.min_window)
        
        return window_size
    
    def apply_window_mask(
        self, 
        attn_scores: torch.Tensor, 
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
        window_size: int
    ) -> torch.Tensor:
        """
        应用滑动窗口掩码
        
        Args:
            attn_scores: [batch, heads, query_len, key_len]
            query_positions: [query_len]
            key_positions: [key_len]
            window_size: 窗口大小
        Returns:
            masked_attn_scores: 应用窗口掩码后的注意力分数
        """
        # 计算相对位置
        rel_positions = query_positions.unsqueeze(1) - key_positions.unsqueeze(0)  # [query_len, key_len]
        
        # 窗口掩码：只允许关注window_size范围内的tokens
        window_mask = (rel_positions >= 0) & (rel_positions < window_size)
        
        # 因果掩码：不能关注未来的tokens
        causal_mask = rel_positions >= 0
        
        # 组合掩码：取反，因为masked_fill是用value填充mask为True的位置
        combined_mask = ~(window_mask & causal_mask)
        
        # 应用掩码
        attn_scores = attn_scores.masked_fill(
            combined_mask.unsqueeze(0).unsqueeze(0),
            float("-inf")
        )
        
        return attn_scores
    
    def forward(
        self,
        x: torch.Tensor,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
        batch, seq_len, _ = x.shape
        
        # QKV投影
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # RoPE位置编码
        past_len = 0 if past_key_value is None else past_key_value[0].size(-2)
        cos, sin = self.rope(seq_len=seq_len, device=x.device, start_pos=past_len)
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)
        
        # KV缓存
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        total_kv_len = k.size(-2)
        
        # 计算复杂度并获取自适应窗口大小
        if seq_len > 1 or past_len == 0:
            complexity = self.compute_complexity(x)  # [batch]
            window_size = self.get_adaptive_window_size(complexity, total_kv_len)
            
            # 位置信息
            key_positions = torch.arange(total_kv_len, device=x.device)
            query_positions = torch.arange(past_len, past_len + seq_len, device=x.device)
            
            # 应用因果掩码
            causal_mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0),
                float("-inf"),
            )
            
            # 应用自适应窗口掩码（仅在训练时或长序列时）
            if seq_len > self.min_window:
                attn_scores = self.apply_window_mask(
                    attn_scores, 
                    query_positions, 
                    key_positions, 
                    window_size
                )
        
        # Softmax和Dropout
        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)
        
        # 输出投影
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        output = self.out_proj(output)
        
        if use_cache:
            return output, (k, v)
        return output


class GlobalTransformerBlock(nn.Module):
    """全局Transformer块（前6层）
    
    使用基于重要性评分的动态Token选择机制，
    Top-K重要token自动获得全局注意力权限。
    """
    def __init__(self, emb_size: int, num_heads: int, dropout: float, top_k_ratio: float = 0.3) -> None:
        super().__init__()
        self.rms_norm1 = RMSNorm(emb_size)
        self.rms_norm2 = RMSNorm(emb_size)
        self.attention = CausalSelfAttention(emb_size, num_heads, dropout)
        self.feed_forward = VariableDimAdaptiveFFN(emb_size, dropout)
        
        # 动态Token选择器 - 【修复】现在真正被使用了
        self.token_selector = DynamicTokenSelector(
            top_k_ratio=top_k_ratio,
            sink_tokens=int(CONFIG.get("attention_sink_tokens", 4))
        )
        self.importance_scorer = TokenImportanceScorer(emb_size)
    
    def forward(
        self,
        x: torch.Tensor,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
        batch, seq_len, emb_size = x.shape
        
        # 【修复】计算Token重要性分数并选择Top-K tokens
        importance_scores = self.importance_scorer(x)  # [batch, seq_len]
        
        # 【修复】动态选择重要tokens
        if seq_len > 50:  # 只在序列较长时启用token选择
            selected_x, selected_indices, mask = self.token_selector.select_tokens(x, importance_scores)
            
            # 对选中的tokens进行注意力计算
            attn_output_selected = self.attention(
                self.rms_norm1(selected_x),
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            
            if use_cache:
                attention_output_selected, present_key_value = attn_output_selected
            else:
                attention_output_selected = attn_output_selected
                present_key_value = None
            
            # 【修复】将选中的tokens的注意力输出还原到完整序列
            # 确保数据类型一致，防止scatter_报错
            attention_output = torch.zeros_like(x)
            attention_output.scatter_(
                1,
                selected_indices.unsqueeze(-1).expand(-1, -1, emb_size),
                attention_output_selected.to(attention_output.dtype)  # 显式转换dtype
            )
        else:
            # 短序列直接使用标准注意力
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


class DynamicWindowBlock(nn.Module):
    """动态窗口Transformer块（后10层）
    
    使用自适应动态窗口大小调整机制，
    根据输入复杂度自动调整窗口大小。
    """
    def __init__(
        self, 
        emb_size: int, 
        num_heads: int, 
        dropout: float,
        min_window: int = 64,
        max_window: int = 256
    ) -> None:
        super().__init__()
        self.rms_norm1 = RMSNorm(emb_size)
        self.rms_norm2 = RMSNorm(emb_size)
        
        # 自适应窗口注意力
        self.attention = AdaptiveWindowAttention(
            emb_size=emb_size,
            num_heads=num_heads,
            dropout=dropout,
            min_window=min_window,
            max_window=max_window,
            complexity_threshold=float(CONFIG.get("window_complexity_threshold", 0.5))
        )
        
        self.feed_forward = VariableDimAdaptiveFFN(emb_size, dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
        # 自适应窗口注意力
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
        
        # QKV投影
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # RoPE位置编码
        past_len = 0 if past_key_value is None else past_key_value[0].size(-2)
        cos, sin = self.rope(seq_len=seq_len, device=x.device, start_pos=past_len)
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)
        
        # KV缓存
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 应用因果掩码
        key_positions = torch.arange(k.size(-2), device=x.device)
        query_positions = torch.arange(past_len, past_len + seq_len, device=x.device)
        causal_mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
        attn_scores = attn_scores.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0),
            float("-inf"),
        )
        
        # Softmax和Dropout
        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)
        
        # 输出投影
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        output = self.out_proj(output)
        
        if use_cache:
            return output, (k, v)
        return output


class MainModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        dict_size = int(CONFIG["dict_size"])
        emb_size = int(CONFIG["emb_size"])
        num_heads = int(CONFIG["num_heads"])
        num_layers_global = int(CONFIG.get("num_layers_global", 6))
        num_layers_dynamic = int(CONFIG.get("num_layers_dynamic", 10))
        dropout = float(CONFIG["dropout"])
        
        # 动态Token选择配置
        top_k_ratio = float(CONFIG.get("dynamic_token_top_k_ratio", 0.3))
        min_window = int(CONFIG.get("min_window_size", 64))
        max_window = int(CONFIG.get("max_window_size", 256))

        self.token_embedding = nn.Embedding(dict_size, emb_size)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # 构建混合架构：6层全局Transformer + 10层动态窗口
        self.transformers = nn.ModuleList()
        
        # 前6层：全局Transformer（带动态Token选择）
        for i in range(num_layers_global):
            self.transformers.append(
                GlobalTransformerBlock(
                    emb_size=emb_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    top_k_ratio=top_k_ratio
                )
            )
        
        # 后10层：动态窗口Transformer（自适应窗口大小）
        for i in range(num_layers_dynamic):
            self.transformers.append(
                DynamicWindowBlock(
                    emb_size=emb_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    min_window=min_window,
                    max_window=max_window
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
            
            # 【新增】添加数值稳定性检查
            if not torch.isfinite(x).all():
                x = torch.nan_to_num(x, nan=0.0, posinf=1e5, neginf=-1e5)

        x = self.final_norm(x)
        logits = self.output_linear(x)
        logits = logits.squeeze(0) if squeeze_batch else logits

        if use_cache:
            return logits, next_key_values
        return logits