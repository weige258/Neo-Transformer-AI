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
    "use_tree_rl_generation": True,
    "tree_rl_beam_width": 4,
    "tree_rl_max_generate_tokens": 100,
    "compress_trigger_len": 1200,
    "compress_trigger_entropy": 0.7,
    "compress_ratio": 0.3,
}
