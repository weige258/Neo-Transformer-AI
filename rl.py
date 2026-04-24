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

# 树状搜索强化学习配置
TREE_SEARCH_CONFIG = {
    "max_depth": 3,  # 树的最大深度
    "branch_factor": 2,  # 每个节点的分支数
    "num_simulations": 4,  # 模拟次数
    "exploration_constant": 1.0,  # UCT探索常数
}


def _compute_grpo_advantages(rewards: List[float]) -> List[float]:
    """计算GRPO的优势函数(组内标准化)"""
    if len(rewards) < 2:
        return [0.0] * len(rewards)
    
    mean_reward = sum(rewards) / len(rewards)
    std_reward = (sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)) ** 0.5
    
    if std_reward < 1e-8:
        return [0.0] * len(rewards)
    
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


class TreeNode:
    """树搜索节点"""
    def __init__(self, tokens: torch.Tensor, text: str, parent=None, past_key_values=None):
        self.tokens = tokens  # 到当前节点的token序列
        self.text = text  # 解码后的文本
        self.parent = parent
        self.children: List['TreeNode'] = []
        self.visits = 0  # 访问次数
        self.value = 0.0  # 节点价值(奖励)
        self.untried_actions: List[Tuple[int, str]] = []  # 未尝试的动作
        self.past_key_values = past_key_values  # 【优化】缓存KV状态,避免重复计算
    
    def is_fully_expanded(self) -> bool:
        """检查是否所有动作都已尝试"""
        # 如果untried_actions为None，表示尚未生成，返回False
        if self.untried_actions is None:
            return False
        # 否则检查是否为空
        return len(self.untried_actions) == 0
    
    def best_child(self, exploration_constant: float) -> 'TreeNode':
        """使用UCT公式选择最佳子节点"""
        if not self.children:
            return None
        
        uct_scores = []
        for child in self.children:
            if child.visits == 0:
                uct_scores.append(float('inf'))
            else:
                exploitation = child.value / child.visits
                exploration = exploration_constant * (math.log(self.visits) / child.visits) ** 0.5
                uct_scores.append(exploitation + exploration)
        
        best_idx = uct_scores.index(max(uct_scores))
        return self.children[best_idx]
    
    def update(self, reward: float):
        """更新节点统计信息"""
        self.visits += 1
        self.value += reward


def _expand_node(model, node: TreeNode, prompt: torch.Tensor, branch_factor: int) -> TreeNode | None:
    """扩展节点:生成新的子节点"""
    if node.is_fully_expanded():
        return None
    
    # 首次扩展时，生成候选动作并初始化untried_actions
    if node.untried_actions is None:
        candidates = []
        with torch.inference_mode():
            for _ in range(branch_factor):
                # 【修复】使用非缓存模式避免KV Cache重复计算和错位
                result = model(node.tokens, use_cache=False)
                
                try:
                    # 取最后一个token的logits
                    next_logits = result[-1] if result.dim() > 1 else result
                    probs = torch.softmax(next_logits / GRPO_CONFIG["temperature"], dim=-1)
                    index = int(torch.multinomial(probs, 1).item())
                    
                    if index == TextTokenizer.END_GENERATION_TOKEN:
                        continue
                    
                    decoded_piece = TextTokenizer.decode(torch.tensor([index]))
                    if not decoded_piece:
                        continue
                    
                    candidates.append((index, decoded_piece, None))  # 不使用KV缓存
                except Exception:
                    continue
        
        if not candidates:
            # 如果没有候选动作，标记为已完全扩展
            node.untried_actions = []
            return None
        
        # 初始化未尝试的动作列表
        node.untried_actions = [(c[0], c[1]) for c in candidates]
    
    # 从现有的未尝试动作中选择一个
    if not node.untried_actions:
        return None
    
    # 获取第一个未尝试的动作
    action_index, action_text = node.untried_actions[0]
    
    # 从未尝试动作列表中移除
    node.untried_actions.pop(0)
    
    # 创建新节点 - 【修复】不传递KV缓存，因为使用非缓存模式
    new_tokens = torch.cat([node.tokens, torch.tensor([action_index], device=device)])
    new_text = node.text + action_text
    child_node = TreeNode(new_tokens, new_text, parent=node, past_key_values=None)
    node.children.append(child_node)
    
    return child_node


def _simulate(model, node: TreeNode, max_steps: int = 10) -> float:
    """模拟:从当前节点随机rollout到终止"""
    current_tokens = node.tokens.clone()
    current_text = node.text
    
    with torch.inference_mode():
        for step in range(max_steps):
            # 【修复】使用非缓存模式避免KV Cache重复计算
            result = model(current_tokens, use_cache=False)
            
            try:
                # 取最后一个token的logits
                next_logits = result[-1] if result.dim() > 1 else result
                probs = torch.softmax(next_logits / CONFIG["temperature"], dim=-1)
                index = int(torch.multinomial(probs, 1).item())
                
                if index == TextTokenizer.END_GENERATION_TOKEN:
                    break
                
                decoded_piece = TextTokenizer.decode(torch.tensor([index]))
                if not decoded_piece:
                    break
                
                current_tokens = torch.cat([current_tokens, torch.tensor([index], device=device)])
                current_text += decoded_piece
            except Exception:
                break
    
    # 返回基于文本质量的奖励(无标签时使用启发式规则)
    reward = _compute_heuristic_reward(current_text)
    return reward


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


def _tree_search(model, prompt: torch.Tensor, question: str) -> Tuple[torch.Tensor, str, float]:
    """执行简化的蒙特卡洛树搜索
    
    Returns:
        best_new_tokens: 新生成的token序列（不包含prompt）
        best_text: 完整文本（包含prompt + 新生成部分）
        best_reward: 最佳路径的奖励
    """
    # 【修复】初始化根节点时不预计算KV-Cache
    root_tokens = prompt.clone()
    root_text = TextTokenizer.decode(root_tokens[root_tokens != 0])
    
    # 不预计算根节点的KV状态，让_expand_node自己处理
    root = TreeNode(root_tokens, root_text, past_key_values=None)
    
    # 初始化未尝试动作为None,表示尚未生成
    root.untried_actions = None  # 实际在_expand_node中动态生成
    
    max_depth = TREE_SEARCH_CONFIG["max_depth"]
    branch_factor = TREE_SEARCH_CONFIG["branch_factor"]
    num_simulations = TREE_SEARCH_CONFIG["num_simulations"]
    exploration_constant = TREE_SEARCH_CONFIG["exploration_constant"]
    
    # MCTS主循环
    for _ in range(num_simulations):
        # 1. Selection: 选择节点
        node = root
        while node.children and node.is_fully_expanded():
            node = node.best_child(exploration_constant)
            if node is None:
                break
        
        if node is None:
            continue
        
        # 2. Expansion: 扩展节点
        if not node.is_fully_expanded() and len(node.text.split()) < max_depth * 5:
            child = _expand_node(model, node, prompt, branch_factor)
            if child:
                node = child
        
        # 3. Simulation: 模拟
        reward = _simulate(model, node)
        
        # 4. Backpropagation: 反向传播
        while node:
            node.update(reward)
            node = node.parent
    
    # 选择访问次数最多的节点作为最佳路径
    best_node = root
    for child in root.children:
        if child.visits > best_node.visits:
            best_node = child
    
    best_reward = best_node.value / best_node.visits if best_node.visits > 0 else 0.0
    
    # 【关键修复】提取新生成的tokens（去除prompt部分）
    prompt_len = len(prompt)
    if len(best_node.tokens) > prompt_len:
        best_new_tokens = best_node.tokens[prompt_len:]
    else:
        best_new_tokens = torch.tensor([], device=device, dtype=torch.long)
    
    return best_new_tokens, best_node.text, best_reward
