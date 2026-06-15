from abc import abstractmethod
import math

import torch

from config import CONFIG


class Tokenizer:
    @staticmethod
    @abstractmethod
    def encode(text: str) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def decode(tokens: torch.Tensor) -> str:
        raise NotImplementedError


class TextTokenizer(Tokenizer):
    UNKNOWN_TOKEN = 0
    START_GENERATION_TOKEN = 1
    END_GENERATION_TOKEN = 2
    HISTORY_CONTEXT_START_TOKEN = 3
    HISTORY_CONTEXT_END_TOKEN = 4
    THINK_START_TOKEN = 5
    THINK_END_TOKEN = 6
    _SURROGATE_START = 0xD800
    _SURROGATE_END = 0xDFFF

    @staticmethod
    def _is_valid_token(idx: int) -> bool:
        """检查token ID是否为有效的Unicode码点
        """
        if idx <= 0:
            return False
        # Unicode最大码点是0x10FFFF = 1114111
        if idx > 0x10FFFF:
            return False
        return not (TextTokenizer._SURROGATE_START <= idx <= TextTokenizer._SURROGATE_END)

    @staticmethod
    def encode(text: str) -> torch.Tensor:
        if not isinstance(text, str):
            if isinstance(text, float) and (math.isnan(text) or math.isinf(text)):
                text = ""
            else:
                text = str(text)

        tensor: list[int] = []
        dict_size = int(CONFIG["dict_size"])

        for letter in text:
            idx = ord(letter)
            # 【修复】简单直接：有效且在词表范围内就保留，否则映射为UNKNOWN
            # 避免哈希桶导致的编解码不一致和冲突问题
            if TextTokenizer._is_valid_token(idx) and idx < dict_size:
                tensor.append(idx)
            else:
                tensor.append(TextTokenizer.UNKNOWN_TOKEN)

        if len(tensor) == 0:
            tensor = [TextTokenizer.UNKNOWN_TOKEN]

        return torch.tensor(tensor, dtype=torch.long)

    @staticmethod
    def decode(tokens: torch.Tensor) -> str:
        text: list[str] = []
        for idx in tokens:
            idx_int = int(idx)
            if idx_int in (
                TextTokenizer.UNKNOWN_TOKEN,
                TextTokenizer.START_GENERATION_TOKEN,
                TextTokenizer.END_GENERATION_TOKEN,
                TextTokenizer.HISTORY_CONTEXT_START_TOKEN,
                TextTokenizer.HISTORY_CONTEXT_END_TOKEN,
                TextTokenizer.THINK_START_TOKEN,
                TextTokenizer.THINK_END_TOKEN,
            ):
                continue
            if not TextTokenizer._is_valid_token(idx_int):
                continue
            text.append(chr(idx_int))
        return "".join(text)


def decode(indices: torch.Tensor) -> str:
    return TextTokenizer.decode(indices)