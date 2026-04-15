from typing import Dict

import torch
from torch import nn
import torch.utils.checkpoint as checkpoint


CONFIG: Dict[str, int | float] = {
    "dict_size": 60000,
    "emb_size": 256,
    "num_heads": 8,
    "num_layers": 8,
    "dropout": 0.1,
    "temperature": 1.0,
    "moe_num_experts": 6,  
    "moe_top_k": 2,        
    "moe_capacity_factor": 1.25,
    "gradient_accumulation_steps": 4,  # 梯度累积步数，模拟batch_size=4
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
        ffn_size = emb_size * 3
        self.gate_proj = nn.Linear(emb_size, ffn_size, bias=False)
        self.up_proj = nn.Linear(emb_size, ffn_size, bias=False)
        self.down_proj = nn.Linear(ffn_size, emb_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        return self.dropout(x)


class ExpertNetwork(nn.Module):
    """单个专家网络"""
    def __init__(self, emb_size: int, dropout: float) -> None:
        super().__init__()
        self.network = SwiGLUFeedForward(emb_size, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MoEGating(nn.Module):
    """MoE门控网络"""
    def __init__(self, emb_size: int, num_experts: int, top_k: int = 2) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(emb_size, num_experts, bias=False)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, emb_size]
        Returns:
            gate_weights: 所有专家的权重 [batch, seq_len, num_experts]
            top_k_weights: Top-K专家的权重 [batch, seq_len, top_k]
            top_k_indices: Top-K专家的索引 [batch, seq_len, top_k]
        """
        # 计算门控分数
        gate_logits = self.gate(x)  # [batch, seq_len, num_experts]
        gate_weights = torch.softmax(gate_logits, dim=-1)
        
        # 选择Top-K专家
        top_k_weights, top_k_indices = torch.topk(gate_weights, k=self.top_k, dim=-1)
        # 重新归一化Top-K权重
        top_k_weights = torch.softmax(top_k_weights, dim=-1)
        
        return gate_weights, top_k_weights, top_k_indices


class MoELayer(nn.Module):
    """Mixture of Experts层"""
    def __init__(self, emb_size: int, num_experts: int, top_k: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 创建多个专家网络
        self.experts = nn.ModuleList([
            ExpertNetwork(emb_size, dropout)
            for _ in range(num_experts)
        ])
        
        # 门控网络
        self.gate = MoEGating(emb_size, num_experts, top_k)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        使用完全向量化的MoE前向传播，彻底消除Python循环
        
        Args:
            x: [batch, seq_len, emb_size]
        Returns:
            output: [batch, seq_len, emb_size]
            aux_loss: 辅助损失用于负载均衡
        """
        batch, seq_len, emb_size = x.shape
        num_tokens = batch * seq_len
        
        # 1. 计算门控权重
        gate_weights, top_k_weights, top_k_indices = self.gate(x)
        
        # 2. 展平输入
        x_flat = x.view(num_tokens, emb_size)  # [num_tokens, emb_size]
        top_k_indices_flat = top_k_indices.view(num_tokens, self.top_k)  # [num_tokens, top_k]
        top_k_weights_flat = top_k_weights.view(num_tokens, self.top_k)  # [num_tokens, top_k]
        
        # 3. 【核心优化】批量并行调用所有专家，生成 [num_experts, num_tokens, emb_size]
        # 使用torch.stack批量处理，GPU自动并行化
        expert_outputs = torch.stack([expert(x_flat) for expert in self.experts])  # [num_experts, num_tokens, emb_size]
        
        # 确保数据类型一致（修复AMP下的dtype不匹配问题）
        if expert_outputs.dtype != x_flat.dtype:
            expert_outputs = expert_outputs.to(x_flat.dtype)
        
        # 4. 【向量化聚合】使用einsum替代for循环
        # 构建one-hot选择矩阵：[num_tokens, num_experts]
        one_hot_selection = torch.zeros(num_tokens, self.num_experts, device=x.device)
        one_hot_selection.scatter_(1, top_k_indices_flat, 1.0)  # 对每个token的top-k专家位置设为1
        
        # 加权选择：将top_k权重分散到对应专家位置
        weighted_selection = torch.zeros(num_tokens, self.num_experts, device=x.device)
        weighted_selection.scatter_add_(1, top_k_indices_flat, top_k_weights_flat)
        
        # 5. 使用einsum进行批量加权求和
        # expert_outputs: [num_experts, num_tokens, emb_size]
        # weighted_selection: [num_tokens, num_experts]
        # 结果: [num_tokens, emb_size]
        final_output_flat = torch.einsum('nte,tn->te', expert_outputs, weighted_selection)
        
        # 6. 恢复形状
        final_output = final_output_flat.view(batch, seq_len, emb_size)
        
        # 7. 计算负载均衡辅助损失
        aux_loss = self._compute_load_balancing_loss(gate_weights, top_k_indices)
        
        return final_output, aux_loss
    
    def _compute_load_balancing_loss(self, gate_weights: torch.Tensor, top_k_indices: torch.Tensor) -> torch.Tensor:
        """
        计算负载均衡损失,确保所有专家被均匀使用
        """
        # 计算每个专家的平均激活概率（重要性）
        importance = gate_weights.mean(dim=[0, 1])  # [num_experts]
        
        # 修复：基于top_k_indices计算实际负载（每个专家被选中的比例）
        batch, seq_len = top_k_indices.shape[:2]
        num_tokens = batch * seq_len
        
        # 展平top_k_indices
        top_k_indices_flat = top_k_indices.view(num_tokens, self.top_k)
        
        # 计算每个专家被选中的次数
        load = torch.zeros(self.num_experts, device=gate_weights.device)
        for k in range(self.top_k):
            load.scatter_add_(0, top_k_indices_flat[:, k], 
                              torch.ones(num_tokens, device=gate_weights.device))
        
        # 归一化为比例
        load = load / num_tokens  # [num_experts]
        
        # 负载均衡损失 = 重要性与负载的点积 * 专家数
        loss = self.num_experts * torch.sum(importance * load)
        return loss


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
        
        # 默认使用MoE架构
        num_experts = int(CONFIG.get("moe_num_experts", 8))
        top_k = int(CONFIG.get("moe_top_k", 2))
        self.feed_forward = MoELayer(emb_size, num_experts, top_k, dropout)

    def _forward_impl(self, x: torch.Tensor, past_key_value: KVCache | None = None, use_cache: bool = False):
        """实际的前向传播实现"""
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
        
        # MoE前馈网络
        ff_output, aux_loss = self.feed_forward(self.rms_norm2(x))
        x = x + ff_output
        
        if use_cache:
            return x, present_key_value, aux_loss
        return x, aux_loss

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache, torch.Tensor]:
        """
        使用梯度检查点的前向传播，训练时节省显存
        """
        # 只在训练且不使用cache时使用checkpoint
        if self.training and not use_cache:
            # checkpoint需要保存输入以便反向传播时重新计算
            return checkpoint.checkpoint(
                self._forward_impl, 
                x, 
                past_key_value, 
                use_cache,
                use_reentrant=False  # 推荐使用非reentrant模式
            )
        else:
            return self._forward_impl(x, past_key_value, use_cache)


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
    ) -> torch.Tensor | tuple[torch.Tensor, list[KVCache]] | tuple[torch.Tensor, list[KVCache], torch.Tensor]:
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

        total_aux_loss = torch.tensor(0.0, device=x.device)
        
        for block, past_key_value in zip(self.transformers, past_key_values):
            if use_cache:
                x, present_key_value, aux_loss = block(
                    x,
                    past_key_value=past_key_value,
                    use_cache=True,
                )
                next_key_values.append(present_key_value)
            else:
                x, aux_loss = block(x)
            
            total_aux_loss = total_aux_loss + aux_loss

        x = self.final_norm(x)
        logits = self.output_linear(x)
        logits = logits.squeeze(0) if squeeze_batch else logits

        if use_cache:
            return logits, next_key_values, total_aux_loss
        return logits, total_aux_loss