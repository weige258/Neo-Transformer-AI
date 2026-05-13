from typing import Any, Dict

CONFIG: Dict[str, Any] = {
    "dict_size": 60000,
    "emb_size": 512,
    "num_heads": 8,
    "num_big_blocks": 2,
    "attention_mix": {
        "hybrid": 3,
        "sparse": 2,
        "dynamic": 2,
    },
    "sliding_window": 128,
    "sparse_stride": 8,
    "dynamic_attention_topk": 32,
    "dropout": 0.1,
    "temperature": 0.8,
    "compress_trigger_len": 1200,
    "compress_trigger_entropy": 0.7,
    "compress_ratio": 0.3,
}
