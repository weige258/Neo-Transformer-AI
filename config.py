from typing import Any, Dict

CONFIG: Dict[str, Any] = {
    "dict_size": 60000,
    "emb_size": 512,
    "num_heads": 8,
    "num_big_blocks": 1,
    "attention_mix": {
        "lightning": 3,
        "latent_compress": 1,
        "sliding": 2,
        "flash": 2,
    },
    "latent_compress_stride": 8,
    "sliding_window": 128,
    "dropout": 0.1,
    "temperature": 0.7,
    "compress_trigger_len": 1200,
    "compress_trigger_entropy": 0.7,
    "compress_ratio": 0.3,
}
