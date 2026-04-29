from typing import Dict

CONFIG: Dict[str, int | float] = {
    "dict_size": 60000,
    "emb_size": 512,
    "num_heads": 8,
    "num_linear_layers": 2,
    "num_flash_layers": 6,
    "dropout": 0.1,
    "temperature": 0.7,
    "compress_trigger_len": 1200,
    "compress_trigger_entropy": 0.7,
    "compress_ratio": 0.3,
}