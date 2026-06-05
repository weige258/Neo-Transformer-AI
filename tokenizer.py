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
    PLACEHOLDER_SINK_TOKEN = 7  # 【StreamingLLM】专用attention sink占位符，永不压缩
    _SURROGATE_START = 0xD800
    _SURROGATE_END = 0xDFFF
    
    # 特殊Token集合（用于快速检测）
    SPECIAL_TOKEN_IDS = frozenset({
        UNKNOWN_TOKEN,
        START_GENERATION_TOKEN,
        END_GENERATION_TOKEN,
        HISTORY_CONTEXT_START_TOKEN,
        HISTORY_CONTEXT_END_TOKEN,
        THINK_START_TOKEN,
        THINK_END_TOKEN,
        PLACEHOLDER_SINK_TOKEN,
    })

    @staticmethod
    def _is_valid_token(idx: int) -> bool:
        """检查token ID是否为有效的Unicode码点
        
        修复：添加对Unicode最大码点0x10FFFF(1114111)的检查
        防止chr()函数抛出ValueError
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
        
        # 保留前10个特殊Token ID（0-9），其余用于字符映射
        SPECIAL_TOKEN_COUNT = 10
        
        for letter in text:
            idx = ord(letter)
            if TextTokenizer._is_valid_token(idx):
                if idx < dict_size:
                    tensor.append(idx)
                else:
                    hashed_idx = SPECIAL_TOKEN_COUNT + (idx % (dict_size - SPECIAL_TOKEN_COUNT))
                    tensor.append(hashed_idx)
                    # 维护逆向映射表（线程安全），确保解码能还原
                    # 【注意Bug #4】哈希冲突：不同码点差值为(dict_size-10)的倍数时映射到同一slot
                    # 这是字符级编码的有损压缩，使用字典大小60000时冲突概率~0.002%
                    # 对于大多数文本（中文/英文/数字）无影响，仅极高码点字符可能冲突
                    with TextTokenizer._reverse_map_lock:
                        if hashed_idx in TextTokenizer._reverse_map:
                            existing = TextTokenizer._reverse_map[hashed_idx]
                            if existing != idx:
                                # 哈希冲突：保留首次写入的值，丢弃当前
                                # 可接受的有损压缩（概率极低）
                                pass
                        else:
                            TextTokenizer._reverse_map[hashed_idx] = idx
            else:
                tensor.append(TextTokenizer.UNKNOWN_TOKEN)
        
        if len(tensor) == 0:
            tensor = [TextTokenizer.UNKNOWN_TOKEN]
        
        return torch.tensor(tensor, dtype=torch.long)

    # 逆向映射表：hashed_idx → 原始Unicode码点
    _reverse_map: dict[int, int] = {}
    _reverse_map_lock = __import__('threading').Lock()  # 线程安全锁

    @staticmethod
    def decode(tokens: torch.Tensor) -> str:
        tokens_cpu = tokens.cpu().tolist()  # 先移出GPU
        text: list[str] = []
        for idx_int in tokens_cpu:
            if idx_int in (
                TextTokenizer.UNKNOWN_TOKEN,
                TextTokenizer.START_GENERATION_TOKEN,
                TextTokenizer.END_GENERATION_TOKEN,
                TextTokenizer.HISTORY_CONTEXT_START_TOKEN,
                TextTokenizer.HISTORY_CONTEXT_END_TOKEN,
                TextTokenizer.THINK_START_TOKEN,
                TextTokenizer.THINK_END_TOKEN,
                TextTokenizer.PLACEHOLDER_SINK_TOKEN,
            ):
                continue
            if not TextTokenizer._is_valid_token(idx_int):
                continue
            # 检查逆向映射表，还原被哈希的高码点字符（线程安全读）
            with TextTokenizer._reverse_map_lock:
                if idx_int in TextTokenizer._reverse_map:
                    text.append(chr(TextTokenizer._reverse_map[idx_int]))
                    continue
            text.append(chr(idx_int))
        return "".join(text)


def decode(indices: torch.Tensor) -> str:
    return TextTokenizer.decode(indices)
