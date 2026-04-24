import math
import torch
import torch.nn.functional as F
from typing import List, Tuple

from model import CONFIG
from tokenizer import TextTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# GRPO配置
GRPO_CONFIG = {
    "num_samples": 4,  # 每组采样数量
    "temperature": 1.0,  # 采样温度(提高多样性)
    "kl_coefficient": 0.01,  # KL散度系数
}

def _compute_grpo_advantages(rewards: List[float]) -> List[float]:
    """计算GRPO的优势函数(组内标准化)"""
    if len(rewards) < 2:
        return [0.0] * len(rewards)
    
    mean_reward = sum(rewards) / len(rewards)
    std_reward = (sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)) ** 0.5
    
    # 【关键修复】防止std为0或极小值
    if std_reward < 1e-6:
        std_reward = 1e-6  # 使用最小标准差，而非直接返回0
    
    advantages = [(r - mean_reward) / std_reward for r in rewards]
    return advantages


def _majority_vote(outputs: List[str]) -> str:
    """多数投票选择共识答案"""
    if not outputs:
        return ""
    
    # 简单统计出现次数最多的输出
    count_dict = {}
    for output in outputs:
        count_dict[output] = count_dict.get(output, 0) + 1
    
    # 返回出现次数最多的
    return max(count_dict, key=count_dict.get)


def _generate_with_sampling(model, prompt: torch.Tensor, num_samples: int) -> List[Tuple[torch.Tensor, str]]:
    """带采样的生成函数,用于GRPO/TTRL"""
    samples = []
    
    with torch.inference_mode():
        for _ in range(num_samples):
            current_prompt = prompt.clone()
            result = model(current_prompt, use_cache=True)
            if isinstance(result, tuple):
                logits, past_key_values = result
            else:
                logits = result
            
            generated_tokens = []
            # 移除采样长度限制，让模型生成更完整的回答
            step = 0
            max_steps = 100  # 设置一个合理的最大步数，避免无限循环
            while step < max_steps:
                try:
                    next_logits = logits[-1]
                    probs = torch.softmax(next_logits / GRPO_CONFIG["temperature"], dim=-1)
                    index = int(torch.multinomial(probs, 1).item())
                    
                    if index == TextTokenizer.END_GENERATION_TOKEN:
                        break
                    
                    generated_tokens.append(index)
                    next_token = torch.tensor([index], device=device)
                    current_prompt = torch.cat([current_prompt, next_token])
                    
                    result = model(
                        next_token,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    if isinstance(result, tuple):
                        logits, past_key_values = result
                    else:
                        logits = result
                    
                    step += 1
                except Exception:
                    break
            
            if generated_tokens:
                generated_tensor = torch.tensor(generated_tokens, device=device)
                decoded_text = TextTokenizer.decode(generated_tensor)
                samples.append((generated_tensor, decoded_text))
    
    return samples


def _compute_heuristic_reward(text: str) -> float:
    """计算启发式奖励(无标签场景)"""
    if not text:
        return 0.0
    
    reward = 0.0
    
    # 1. 长度奖励(鼓励生成完整回答，但避免过长或过短)
    length = len(text)
    if length < 5:
        # 对过短文本进行强惩罚
        length_score = length / 10.0 * 0.5
    elif length > 100:
        # 对过长文本进行惩罚
        length_score = 1.0 - (length - 100) / 200.0
        length_score = max(0.3, length_score)
    else:
        length_score = 1.0
    reward += length_score * 0.2
    
    # 2. 多样性奖励(鼓励不同字符和词汇)
    unique_chars = len(set(text))
    diversity_score = min(unique_chars / 20.0, 1.0)
    # 对重复字符进行额外惩罚
    if unique_chars < len(text) * 0.3:
        diversity_score *= 0.5
    reward += diversity_score * 0.25
    
    # 3. 连贯性奖励(避免重复)
    words = text.split()
    if len(words) > 1:
        unique_words = len(set(words))
        repetition_ratio = unique_words / len(words)
        # 对重复词汇进行强惩罚
        if repetition_ratio < 0.5:
            repetition_ratio *= 0.5
        reward += repetition_ratio * 0.3
    else:
        # 对单字回答进行惩罚
        reward += 0.05
    
    # 4. 自一致性奖励(检查文本是否自相矛盾)
    consistency_score = 1.0
    # 简单检查：避免明显的自相矛盾
    if "不是" in text and "是" in text:
        consistency_score = 0.5
    # 检查是否有重复的短语
    if len(text) > 10:
        for i in range(len(text) - 2):
            phrase = text[i:i+3]
            if text.count(phrase) > 2:
                consistency_score *= 0.7
                break
    reward += consistency_score * 0.1
    
    # 5. 格式奖励(鼓励完整的句子结构)
    format_score = 0.0
    if text.endswith(".") or text.endswith("！") or text.endswith("？"):
        format_score = 1.0
    elif text.endswith(",") or text.endswith("；"):
        format_score = 0.5
    reward += format_score * 0.15
    
    return reward