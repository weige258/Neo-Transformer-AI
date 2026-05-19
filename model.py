from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from config import CONFIG

# === 行动智能头常量 ===
ACTION_THINKING = 0  # 思考中
ACTION_ANSWER = 1    # 回答中
ACTION_END = 2       # 结束

# 状态转移掩码矩阵：1=允许，0=禁止
# 行=当前状态，列=下一状态
#        T   A   E
# 允许更灵活的多段思考/回答序列（例如：T→T→A→T→A→A→E）
TRANSITION_MASK = [
    [1.0, 1.0, 1.0],  # THINKING → (T, A, E)
    [1.0, 1.0, 1.0],  # ANSWER   → (T, A, E)
    [0.0, 0.0, 1.0],  # END      → (T, A, E)
]


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
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
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
        if mem_k.size(-2) == 0:
            return torch.zeros_like(q)

        scores = torch.matmul(q, mem_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        q_pos = torch.arange(q_start_pos, q_start_pos + q.size(-2), device=q.device)[:, None]
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
        batch, heads, q_len, dim = q.shape
        k_len = k.size(-2)
        outputs = []
        offsets = torch.arange(
            -self.window_size + 1,
            1,
            device=q.device,
            dtype=torch.long,
        )

        for start in range(0, q_len, self.chunk_size):
            end = min(start + self.chunk_size, q_len)
            q_chunk = q[:, :, start:end, :]
            q_abs = torch.arange(q_start_pos + start, q_start_pos + end, device=q.device)
            rel_idx = q_abs[:, None] + offsets[None, :] - k_start_pos
            valid = (rel_idx >= 0) & (rel_idx < k_len)
            gather_idx = rel_idx.clamp(0, max(k_len - 1, 0)).reshape(-1)

            selected_k = k.index_select(dim=2, index=gather_idx)
            selected_v = v.index_select(dim=2, index=gather_idx)
            selected_k = selected_k.view(batch, heads, end - start, self.window_size, dim)
            selected_v = selected_v.view(batch, heads, end - start, self.window_size, dim)

            scores = (q_chunk.unsqueeze(-2) * selected_k).sum(dim=-1) / math.sqrt(dim)
            scores = scores.masked_fill(~valid.view(1, 1, end - start, self.window_size), float("-inf"))
            weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
            weights = torch.nan_to_num(weights, nan=0.0)
            outputs.append((weights.unsqueeze(-1) * selected_v).sum(dim=-2))

        return torch.cat(outputs, dim=-2)

    def _attend_dynamic_memory(
        self,
        q: torch.Tensor,
        mem_k: torch.Tensor,
        mem_v: torch.Tensor,
        mem_pos: torch.Tensor,
        q_start_pos: int,
    ) -> torch.Tensor:
        if mem_k.size(-2) == 0:
            return torch.zeros_like(q)

        scores = torch.matmul(q, mem_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + self.memory_gate(mem_k).transpose(-2, -1)
        q_pos = torch.arange(q_start_pos, q_start_pos + q.size(-2), device=q.device)[:, None]
        scores = scores.masked_fill(mem_pos[None, :] > q_pos, float("-inf"))

        topk = min(self.dynamic_topk, mem_k.size(-2))
        top_scores, top_idx = torch.topk(scores, k=topk, dim=-1)
        weights = torch.softmax(top_scores.float(), dim=-1).to(q.dtype)
        weights = torch.nan_to_num(weights, nan=0.0)

        value_bank = mem_v.unsqueeze(2).expand(-1, -1, q.size(-2), -1, -1)
        gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim)
        selected_v = torch.gather(value_bank, dim=3, index=gather_idx)
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

        compressed_out = self._attend_compressed(q, mem_k, mem_v, mem_pos, q_start_pos)
        sparse_out = self._attend_local_window(q, raw_k, raw_v, q_start_pos, raw_k_start_pos)
        dynamic_out = self._attend_dynamic_memory(q, mem_k, mem_v, mem_pos, q_start_pos)

        compressed_out = compressed_out.transpose(1, 2).contiguous().view(batch, seq_len, self.emb_size)
        sparse_out = sparse_out.transpose(1, 2).contiguous().view(batch, seq_len, self.emb_size)
        dynamic_out = dynamic_out.transpose(1, 2).contiguous().view(batch, seq_len, self.emb_size)

        prior = attention_mix_prior(x.device, torch.float32)
        mix = torch.softmax(self.router(x).float() + prior, dim=-1).to(x.dtype)
        out = (
            compressed_out * mix[..., 0:1]
            + sparse_out * mix[..., 1:2]
            + dynamic_out * mix[..., 2:3]
        )
        out = self.out_proj(out)

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


class ActionHead(nn.Module):
    """行动智能头：从最终隐层状态预测 [思考, 回答, 结束] 行动。"""
    def __init__(self, emb_size: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(emb_size, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, 3, bias=True)  # 3 actions: THINKING, ANSWER, END
        # 置信/终止分支：输出单个标量，表示当前 token 的终止/置信度（可用于策略或 RL）
        self.fc3 = nn.Linear(hidden_dim, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, emb_size] or [seq_len, emb_size]
        h = F.silu(self.fc1(x))
        a_logits = self.fc2(h)
        t_logit = self.fc3(h)
        # Concatenate: last dim -> [THINK, ANSWER, END, TERM]
        return torch.cat([a_logits, t_logit], dim=-1)  # [batch, seq_len, 4] or [seq_len, 4]


def compute_action_labels(
    tokens: torch.Tensor,
    think_start_id: int = 5,
    think_end_id: int = 6,
    end_id: int = 2,
    temperature: float = 0.5,
) -> torch.Tensor:
    """计算每个时间步的硬行动标签 [THINKING, ANSWER, END]。
    
    基于 token 序列中特殊标记位置生成硬标签（非软标签，避免模糊）：
    - THINK_START ~ THINK_END 之间 → THINKING=1
    - 连续思考区域之间的"中转间隙"（TE→TS 无需回答）→ THINKING=1
    - 思考区域外（非中转）→ ANSWER=1
    - 每个 END_GENERATION 位置 → END=1
    """
    seq_len = tokens.size(-1) if tokens.dim() > 1 else tokens.size(0)
    device = tokens.device

    labels = torch.zeros(seq_len, 3, device=device, dtype=torch.float32)

    think_start_positions = (tokens == think_start_id).nonzero(as_tuple=True)[-1]
    think_end_positions = (tokens == think_end_id).nonzero(as_tuple=True)[-1]
    end_positions = (tokens == end_id).nonzero(as_tuple=True)[-1]

    # --- 1) 构建思考区域列表（成对匹配，支持嵌套退化为平铺）---
    # 简单栈匹配：每个 THINK_START 找其后的 THINK_END
    think_regions = []  # [(start, end), ...]
    te_idx = 0
    for ts in think_start_positions:
        # 找最近的未匹配 THINK_END
        while te_idx < len(think_end_positions) and think_end_positions[te_idx] < ts:
            te_idx += 1
        if te_idx < len(think_end_positions):
            te = think_end_positions[te_idx]
            te_idx += 1
            think_regions.append((ts.item(), te.item()))
        else:
            # 没有配对的 END，标记到序列末尾
            think_regions.append((ts.item(), seq_len - 1))

    # --- 3) 建立位置→区域映射 ---
    # 先确定每个位置是否在某个思考区域内
    in_think = torch.zeros(seq_len, device=device, dtype=torch.bool)
    for s, e in think_regions:
        in_think[s:e+1] = True

    # --- 4) 检测"思考中转间隙" ---
    # 标记短间隙（≤3 token）为中转：这允许模型在多个思考段之间存在短过渡
    is_transition_gap = torch.zeros(seq_len, device=device, dtype=torch.bool)
    for i in range(len(think_regions) - 1):
        gap_start = think_regions[i][1] + 1  # 前一个区域的 TE + 1
        gap_end = think_regions[i + 1][0]     # 后一个区域的 TS
        gap_len = gap_end - gap_start
        if 0 < gap_len <= 3:
            is_transition_gap[gap_start:gap_end] = True

    # --- 5) 标记 ANSWER 区域 ---
    # 不在思考区域内且不是中转间隙 → ANSWER
    is_answer = (~in_think) & (~is_transition_gap)

    # --- 6) 计算各维标签 ---
    # THINKING: 在思考区域内 + 中转间隙
    think_weight = torch.zeros(seq_len, device=device)
    think_weight[in_think] = 1.0
    # 对中转间隙赋予较高的 thinking 倾向，以鼓励短暂的连续思考序列
    think_weight[is_transition_gap] = 0.8

    # ANSWER: 不在思考区域、不是中转间隙、不是 END
    answer_weight = is_answer.float()

    # END: 每个 END_GENERATION_TOKEN 位置
    end_weight = torch.zeros(seq_len, device=device)
    for ep in end_positions:
        ep_idx = ep.item()
        end_weight[ep_idx] = 1.0
        # END 前 3 个位置开始渐入
        fade_start = max(0, ep_idx - 3)
        for fi in range(fade_start, ep_idx):
            dist = (fi - fade_start) / max(ep_idx - fade_start, 1)
            end_weight[fi] = max(end_weight[fi].item(), float(dist * dist))

    # 确保每个位置标签和为 1
    # 优先级: END > THINKING > ANSWER
    labels[:, ACTION_END] = end_weight.clamp(0.0, 1.0)
    remaining = 1.0 - end_weight
    labels[:, ACTION_THINKING] = think_weight * remaining
    labels[:, ACTION_ANSWER] = answer_weight * remaining

    row_sum = labels.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    labels = labels / row_sum

    # 应用温度参数对标签做平滑处理，输出为软标签分布
    temp = max(1e-6, float(temperature))
    labels = torch.clamp(labels, min=1e-12)
    # 通过 log + softmax 实现温度缩放，避免数值不稳定
    labels = torch.softmax(torch.log(labels) / temp, dim=-1)

    return labels  # [seq_len, 3]


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

        # === 行动智能头 ===
        action_hidden = int(CONFIG.get("action_hidden_dim", 128))
        self.action_head = ActionHead(emb_size, hidden_dim=action_hidden)
        # === 价值头 (value head) 用于 PPO 的 value 预测 ===
        self.value_head = nn.Linear(emb_size, 1, bias=True)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        if self.output_linear.weight is not self.token_embedding.weight:
            nn.init.normal_(self.output_linear.weight, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear) and module.weight is not self.output_linear.weight:
                nn.init.xavier_uniform_(module.weight)
        # 初始化行动头
        for name, param in self.action_head.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        # 初始化 value head
        if hasattr(self, 'value_head'):
            nn.init.xavier_uniform_(self.value_head.weight)
            if self.value_head.bias is not None:
                nn.init.zeros_(self.value_head.bias)

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
            for block in self.transformers:
                if self.training:
                    x = checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)

        hidden = self.final_norm(x)
        logits = self.output_linear(hidden)
        action_logits = self.action_head(hidden)
        value_preds = self.value_head(hidden)
        # squeeze batch dimension if input was 1D
        logits = logits.squeeze(0) if squeeze_batch else logits
        action_logits = action_logits.squeeze(0) if squeeze_batch else action_logits
        value_preds = value_preds.squeeze(0) if squeeze_batch else value_preds

        if use_cache:
            return logits, action_logits, value_preds, next_key_values
        return logits, action_logits, value_preds
