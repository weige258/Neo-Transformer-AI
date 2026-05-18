from typing import Any, Dict

CONFIG: Dict[str, Any] = {
    "dict_size": 60000,
    "emb_size": 512,
    "num_heads": 8,
    "num_transformer_blocks": 8,
    "attention_mix": {
        "compressed": 2,
        "sparse": 1.3,
        "dynamic": 1,
    },
    "sliding_window": 96,
    "compress_stride": 16,
    "dynamic_attention_topk": 16,
    "attention_chunk_size": 64,
    "tie_token_embeddings": True,
    "dropout": 0.2,
    "compress_trigger_len": 1200,
    "compress_trigger_entropy": 0.7,
    "compress_ratio": 0.3,
    # === 解码策略（防止退化生成） ===
    "temperature": 0.7,               # 温度：越高越随机（0.6→0.7）
    "repetition_penalty": 1.15,        # 重复惩罚：>1 惩罚已生成 token
    "top_k": 50,                       # Top-K 采样
    "top_p": 0.9,                      # Top-P (Nucleus) 采样
    "max_thinking_steps": 200,         # 思考块最大步数（防止死锁）
    # === 行动智能头 (Action Head) ===
    "action_loss_coef": 0.3,          # 行动损失权重
    "action_label_temperature": 0.3,   # 标签温度
    "action_temperature": 0.5,         # 生成时行动采样温度
    "action_hidden_dim": 128,          # 行动头隐藏层维度
    "min_generate_tokens": 4,          # 生成时最少 token 数
}
