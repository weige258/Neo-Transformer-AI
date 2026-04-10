from abc import abstractmethod

import torch

from model import CONFIG


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
    START_TOKEN = 1
    END_TOKEN = int(CONFIG["dict_size"]) - 1
    THINK_TOKEN = 2
    THINK_END_TOKEN = 3
    _SURROGATE_START = 0xD800
    _SURROGATE_END = 0xDFFF

    @staticmethod
    def _is_valid_token(idx: int) -> bool:
        if idx <= 0 or idx >= TextTokenizer.END_TOKEN:
            return False
        return not (TextTokenizer._SURROGATE_START <= idx <= TextTokenizer._SURROGATE_END)

    @staticmethod
    def encode(text: str) -> torch.Tensor:
        tensor: list[int] = []
        for letter in text:
            idx = ord(letter)
            if TextTokenizer._is_valid_token(idx):
                tensor.append(idx)
            else:
                tensor.append(TextTokenizer.UNKNOWN_TOKEN)
        return torch.tensor(tensor, dtype=torch.long)

    @staticmethod
    def decode(tokens: torch.Tensor) -> str:
        text: list[str] = []
        for idx in tokens:
            idx_int = int(idx)
            if idx_int in (TextTokenizer.UNKNOWN_TOKEN, TextTokenizer.START_TOKEN, TextTokenizer.END_TOKEN, TextTokenizer.THINK_TOKEN, TextTokenizer.THINK_END_TOKEN):
                continue
            if not TextTokenizer._is_valid_token(idx_int):
                continue
            text.append(chr(idx_int))
        return "".join(text)


def decode(indices: torch.Tensor) -> str:
    return TextTokenizer.decode(indices)
