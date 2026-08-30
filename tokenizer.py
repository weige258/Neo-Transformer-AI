from abc import abstractmethod
import math
import unicodedata

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
    # 【修复】将特殊token移到词表末尾，避免与Unicode码点(0-6)冲突
    # 使用dict_size附近的值，确保不会与普通字符冲突
    _BASE_OFFSET = 59990  # 基于dict_size=60000的偏移
    UNKNOWN_TOKEN = _BASE_OFFSET + 0
    START_GENERATION_TOKEN = _BASE_OFFSET + 1
    END_GENERATION_TOKEN = _BASE_OFFSET + 2
    HISTORY_CONTEXT_START_TOKEN = _BASE_OFFSET + 3
    HISTORY_CONTEXT_END_TOKEN = _BASE_OFFSET + 4
    THINK_START_TOKEN = _BASE_OFFSET + 5
    THINK_END_TOKEN = _BASE_OFFSET + 6
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
        # 【修复】排除Unicode私用区(U+E000-U+F8FF)
        # 特殊token 59990-59996 恰落在私用区(U+EA56-U+EA5C)，
        # 不排除则外部输入(如爬虫数据)可直接注入特殊token
        if 0xE000 <= idx <= 0xF8FF:
            return False
        return not (TextTokenizer._SURROGATE_START <= idx <= TextTokenizer._SURROGATE_END)

    @staticmethod
    def encode(text: str) -> torch.Tensor:
        if not isinstance(text, str):
            if isinstance(text, float) and (math.isnan(text) or math.isinf(text)):
                text = ""
            else:
                text = str(text)

        if not text:
            return torch.tensor([], dtype=torch.long)

        # Normalize compatibility characters so fullwidth punctuation and alphanumerics map into trainable token range.
        text = unicodedata.normalize('NFKC', text)

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