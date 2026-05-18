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
    "dropout": 0.05,
    "temperature": 0.8,
    "compress_trigger_len": 1200,
    "compress_trigger_entropy": 0.7,
    "compress_ratio": 0.3,
    # === 行动智能头 (Action Head) ===
    "action_loss_coef": 0.3,          # 行动损失权重（提高以加速行动头学习）
    "action_label_temperature": 0.3,   # 软标签平滑温度（更低=标签更硬更清晰）
    "action_temperature": 0.5,         # 生成时行动采样温度
    "action_hidden_dim": 128,          # 行动头隐藏层维度
    "min_generate_tokens": 8,          # 生成时最少 token 数（防止过早结束）
}
