import sys

import torch

from model import CONFIG, MainModel
from record import record_loss
from tokenizer import TextTokenizer


if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model() -> MainModel:
    try:
        loaded = torch.load("model.pth", map_location=device, weights_only=False)
        if isinstance(loaded, dict):
            model = MainModel().to(device)
            model.load_state_dict(loaded)
            print("Loaded model state dict.", flush=True)
        else:
            model = loaded.to(device)
            print("Loaded full model.", flush=True)

        if (
            not hasattr(model, "transformers")
            or len(model.transformers) == 0
            or not hasattr(model.transformers[0], "rms_norm1")
        ):
            print("Detected old model structure; creating new model.", flush=True)
            model = MainModel().to(device)
        return model
    except Exception as e:
        print(f"Failed to load model: {e}", flush=True)
        model = MainModel().to(device)
        print("Created new model.", flush=True)
        return model


# 【性能优化】启用TensorFloat32加速矩阵运算(消除UserWarning)
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

# 启用自动混合精度训练
scaler = torch.amp.GradScaler()
use_amp = torch.cuda.is_available() and torch.cuda.get_device_capability(device)[0] >= 7

print(f"Using device: {device}", flush=True)
print(f"AMP enabled: {use_amp}", flush=True)
model = _load_model()

# 【性能优化】启用PyTorch 2.0编译加速(需要Triton支持)
compile_success = False
if torch.cuda.is_available():
    # 检查是否有Triton支持
    try:
        import triton
        has_triton = True
    except ImportError:
        has_triton = False
    
    if has_triton:
        import warnings
        import logging
        
        # 临时禁用torch._inductor的日志输出
        inductor_logger = logging.getLogger("torch._inductor")
        old_level = inductor_logger.level
        inductor_logger.setLevel(logging.ERROR)
        
        try:
            # 使用"reduce-overhead"模式适合小batch的推理场景
            model = torch.compile(model, mode="reduce-overhead")
            
            # 测试编译是否成功(首次调用会触发编译)
            with torch.inference_mode():
                test_input = torch.tensor([TextTokenizer.START_GENERATION_TOKEN], device=device).unsqueeze(0)
                _ = model(test_input)
            
            compile_success = True
        except Exception:
            # 编译失败,回退到标准模式
            model = _load_model()
        finally:
            # 恢复日志级别
            inductor_logger.setLevel(old_level)

if not compile_success:
    print("[Info] Running without torch.compile optimization.", flush=True)

total_params = sum(param.numel() for param in model.parameters())
print(f"模型参数: {total_params / 1e+8}亿", flush=True)

loss_func = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)

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


def _prepare_training_data(ask_text: str, answer_text: str, hist_context: str = None):
    """准备单个样本的训练数据"""
    if ask_text is None or answer_text is None:
        return None, None, None
    
    ask_tensor = TextTokenizer.encode(ask_text).to(device)
    answer_tensor = TextTokenizer.encode(answer_text).to(device)
    
    if answer_tensor.numel() == 0:
        return None, None, None

    if hist_context is not None and hist_context.strip():
        history_tensor = TextTokenizer.encode(hist_context).to(device)
        train_tensor = torch.cat(
            [
                torch.tensor([TextTokenizer.START_GENERATION_TOKEN], device=device),
                history_tensor,
                torch.tensor([TextTokenizer.END_GENERATION_TOKEN], device=device),
                torch.tensor([TextTokenizer.HISTORY_CONTEXT_START_TOKEN], device=device),
                ask_tensor,
                torch.tensor([TextTokenizer.START_GENERATION_TOKEN], device=device),
                answer_tensor,
                torch.tensor([TextTokenizer.END_GENERATION_TOKEN], device=device),
            ]
        )
        target_mask = torch.cat(
            [
                torch.zeros(ask_tensor.numel() + 1, dtype=torch.bool, device=device),
                torch.ones(answer_tensor.numel() + 1, dtype=torch.bool, device=device),
            ]
        )
        history_mask_len = 1 + history_tensor.numel() + 1
        target_mask = torch.cat([
            torch.zeros(history_mask_len, dtype=torch.bool, device=device),
            target_mask
        ])
        preview = torch.cat(
            [answer_tensor, torch.tensor([TextTokenizer.END_GENERATION_TOKEN], device=device)]
        )
    else:
        train_tensor = torch.cat(
            [
                ask_tensor,
                torch.tensor([TextTokenizer.START_GENERATION_TOKEN], device=device),
                answer_tensor,
                torch.tensor([TextTokenizer.END_GENERATION_TOKEN], device=device),
            ]
        )
        target_mask = torch.cat(
            [
                torch.zeros(ask_tensor.numel() + 1, dtype=torch.bool, device=device),
                torch.ones(answer_tensor.numel() + 1, dtype=torch.bool, device=device),
            ]
        )
        preview = torch.cat(
            [answer_tensor, torch.tensor([TextTokenizer.END_GENERATION_TOKEN], device=device)]
        )
    
    return train_tensor, target_mask, preview


def _compute_grpo_advantages(rewards: list[float]) -> list[float]:
    """计算GRPO的优势函数(组内标准化)"""
    if len(rewards) < 2:
        return [0.0] * len(rewards)
    
    mean_reward = sum(rewards) / len(rewards)
    std_reward = (sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)) ** 0.5
    
    if std_reward < 1e-8:
        return [0.0] * len(rewards)
    
    advantages = [(r - mean_reward) / std_reward for r in rewards]
    return advantages


def _majority_vote(outputs: list[str]) -> str:
    """多数投票选择共识答案"""
    if not outputs:
        return ""
    
    # 简单统计出现次数最多的输出
    count_dict = {}
    for output in outputs:
        count_dict[output] = count_dict.get(output, 0) + 1
    
    # 返回出现次数最多的
    return max(count_dict, key=count_dict.get)


def _generate_with_sampling(prompt: torch.Tensor, num_samples: int) -> list[tuple[torch.Tensor, str]]:
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
        self.children: list['TreeNode'] = []
        self.visits = 0  # 访问次数
        self.value = 0.0  # 节点价值(奖励)
        self.untried_actions: list[tuple[int, str]] = []  # 未尝试的动作
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
                exploration = exploration_constant * (2 * (self.visits ** 0.5) / child.visits) ** 0.5
                uct_scores.append(exploitation + exploration)
        
        best_idx = uct_scores.index(max(uct_scores))
        return self.children[best_idx]
    
    def update(self, reward: float):
        """更新节点统计信息"""
        self.visits += 1
        self.value += reward


def _expand_node(node: TreeNode, prompt: torch.Tensor, branch_factor: int) -> TreeNode | None:
    """扩展节点:生成新的子节点"""
    if node.is_fully_expanded():
        return None
    
    # 采样多个候选动作
    candidates = []
    with torch.inference_mode():
        for _ in range(branch_factor):
            # 【关键优化】复用父节点的KV-Cache,只传入最后一个token
            if node.past_key_values is not None and len(node.tokens) > 0:
                # 只传入最新的一个token,利用缓存加速
                last_token = node.tokens[-1:].clone()
                result = model(last_token, past_key_values=node.past_key_values, use_cache=True)
            else:
                # 根节点或无缓存时,传入完整序列
                result = model(node.tokens, use_cache=True)
            
            if isinstance(result, tuple):
                logits, past_key_values = result
            else:
                logits = result
                past_key_values = None
            
            try:
                next_logits = logits[-1]
                probs = torch.softmax(next_logits / GRPO_CONFIG["temperature"], dim=-1)
                index = int(torch.multinomial(probs, 1).item())
                
                if index == TextTokenizer.END_GENERATION_TOKEN:
                    continue
                
                decoded_piece = TextTokenizer.decode(torch.tensor([index]))
                if not decoded_piece:
                    continue
                
                candidates.append((index, decoded_piece, past_key_values))
            except Exception:
                continue
    
    if not candidates:
        # 如果没有候选动作，标记为已完全扩展
        node.untried_actions = []
        return None
    
    # 首次扩展时，初始化未尝试的动作列表
    if node.untried_actions is None:
        node.untried_actions = [(c[0], c[1]) for c in candidates]
    
    # 选择一个未尝试的动作
    action = candidates[0]
    action_index, action_text, action_kv_cache = action
    
    # 从未尝试动作列表中移除
    if (action_index, action_text) in node.untried_actions:
        node.untried_actions.remove((action_index, action_text))
    
    # 创建新节点 - 【优化】传递KV-Cache给子节点
    new_tokens = torch.cat([node.tokens, torch.tensor([action_index], device=device)])
    new_text = node.text + action_text
    child_node = TreeNode(new_tokens, new_text, parent=node, past_key_values=action_kv_cache)
    node.children.append(child_node)
    
    return child_node


def _simulate(node: TreeNode, max_steps: int = 10) -> float:
    """模拟:从当前节点随机rollout到终止"""
    current_tokens = node.tokens.clone()
    current_text = node.text
    
    with torch.inference_mode():
        # 【关键优化】复用节点的KV-Cache作为起始状态
        past_kv = node.past_key_values
        
        for step in range(max_steps):
            # 如果有缓存,只传入最后一个token;否则传入完整序列
            if past_kv is not None and len(current_tokens) > 0:
                last_token = current_tokens[-1:].clone()
                result = model(last_token, past_key_values=past_kv, use_cache=True)
            else:
                result = model(current_tokens, use_cache=True)
            
            if isinstance(result, tuple):
                logits, past_kv = result  # 更新缓存供下一步使用
            else:
                logits = result
                past_kv = None
            
            try:
                next_logits = logits[-1]
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


def _tree_search(prompt: torch.Tensor, question: str) -> tuple[torch.Tensor, str, float]:
    """执行简化的蒙特卡洛树搜索
    
    Returns:
        best_new_tokens: 新生成的token序列（不包含prompt）
        best_text: 完整文本（包含prompt + 新生成部分）
        best_reward: 最佳路径的奖励
    """
    # 【优化】初始化根节点时预计算KV-Cache
    root_tokens = prompt.clone()
    root_text = TextTokenizer.decode(root_tokens[root_tokens != 0])
    
    # 预计算根节点的KV状态,避免后续重复计算
    with torch.inference_mode():
        result = model(root_tokens, use_cache=True)
        if isinstance(result, tuple):
            _, root_past_kv = result
        else:
            root_past_kv = None
    
    root = TreeNode(root_tokens, root_text, past_key_values=root_past_kv)
    
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
            child = _expand_node(node, prompt, branch_factor)
            if child:
                node = child
        
        # 3. Simulation: 模拟
        reward = _simulate(node)
        
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


def train(ask: str = None, answer: str = None, history_context: str = None) -> None:
    """单步训练函数 - 集成GRPO和TTRL"""
    # 单文本训练模式
    if ask is None and answer is None:
        return
    
    if ask is None:
        print(f"\n---Single text training:\n{answer}", flush=True)
        print("\n---Learning tokens:", flush=True)

        text_tensor = TextTokenizer.encode(answer).to(device)
        if text_tensor.numel() < 2:
            return

        train_tensor = torch.cat(
            [
                torch.tensor([TextTokenizer.START_GENERATION_TOKEN], device=device),
                text_tensor,
                torch.tensor([TextTokenizer.END_GENERATION_TOKEN], device=device),
            ]
        )
        target_mask = torch.ones(train_tensor.numel(), dtype=torch.bool, device=device)
        preview = train_tensor
        _run_train_step(train_tensor, target_mask, preview, show_preview=True)  # 单文本模式显示预览
        return

    # QA训练模式
    print(f"\n---Train question:\n{ask}", flush=True)
    print(f"\n---Train answer:\n{answer}", flush=True)
    
    # 【关键修复】如果有明确的参考答案，直接执行标准SFT
    if answer and answer.strip():
        train_tensor, target_mask, preview = _prepare_training_data(ask, answer, history_context)
        if train_tensor is None:
            return
        _run_train_step(train_tensor, target_mask, preview, show_preview=False)  # QA模式不重复显示
        return
    
    # 【GRPO】无标签场景：使用模型生成的样本来进行强化学习
    prompt = torch.cat(
        [
            TextTokenizer.encode(ask).to(device),
            torch.tensor([TextTokenizer.START_GENERATION_TOKEN], device=device),
        ]
    )
  
    # 【GRPO】为问题生成多个候选答案
    samples = _generate_with_sampling(prompt, GRPO_CONFIG["num_samples"])
    if not samples:
        return
    
    # 【GRPO】计算每个候选答案的奖励
    rewards = []
    for sample in samples:
        _, decoded_text = sample
        reward = _compute_heuristic_reward(decoded_text)
        rewards.append(reward)
    
    # 【GRPO】计算优势函数
    advantages = _compute_grpo_advantages(rewards)
    
    # 【GRPO正确实现】对每个样本分别计算加权Loss并累加
    total_loss = torch.tensor(0.0, device=device)
    valid_samples = 0
    
    for idx, (sample_tokens, sample_text) in enumerate(samples):
        advantage = advantages[idx]
        
        # 跳过优势为0的样本（组内表现平均）
        if abs(advantage) < 1e-6:
            continue
        
        # 构建该样本的训练数据
        full_tokens = torch.cat([prompt, sample_tokens])
        # 只对生成的部分计算loss（mask掉prompt部分）
        target_mask = torch.cat([
            torch.zeros(len(prompt), dtype=torch.bool, device=device),
            torch.ones(len(sample_tokens), dtype=torch.bool, device=device),
        ])
        
        # 执行带优势加权的训练步骤
        _run_train_step(full_tokens, target_mask, sample_tokens, advantage_weight=advantage, show_preview=False)
        valid_samples += 1
    
    if valid_samples > 0:
        print(f"[GRPO] Trained on {valid_samples} samples with advantages: {[f'{a:.2f}' for a in advantages]}", flush=True)


def generation(text: str, max_generate_tokens: int|None = None, history_context: str = None) -> str:
    """生成函数 - 集成树状搜索强化学习
    
    Args:
        text: 输入文本/问题
        max_generate_tokens: 最大生成token数
        history_context: 历史上下文(可选)
    
    Returns:
        生成的文本
    """
    model.eval()
    output_text = ""

    # 构建prompt(支持历史上下文)
    if history_context and history_context.strip():
        history_tensor = TextTokenizer.encode(history_context).to(device)
        text_tensor = TextTokenizer.encode(text).to(device)
        prompt = torch.cat([
            torch.tensor([TextTokenizer.START_GENERATION_TOKEN], device=device),
            history_tensor,
            torch.tensor([TextTokenizer.END_GENERATION_TOKEN], device=device),
            torch.tensor([TextTokenizer.HISTORY_CONTEXT_START_TOKEN], device=device),
            text_tensor,
            torch.tensor([TextTokenizer.START_GENERATION_TOKEN], device=device),
        ])
    else:
        prompt = torch.cat([
            TextTokenizer.encode(text).to(device),
            torch.tensor([TextTokenizer.START_GENERATION_TOKEN], device=device),
        ])

    print("\n---Generated reply:", flush=True)

    min_new_tokens = 1
    if max_generate_tokens is not None:
        max_generate_tokens = max(1, int(max_generate_tokens))
 
    with torch.inference_mode():
        # 【树状搜索强化学习】在生成阶段执行MCTS搜索
        best_new_tokens, best_full_text, tree_reward = _tree_search(prompt, text)
        
        # 【关键修复】MCTS只决定前N个token，然后继续自回归生成
        # 如果MCTS找到了高质量的前缀且长度合理，使用它作为起始
        mcts_used = False
        if len(best_new_tokens) > 0 and tree_reward > 0.3:
            # 将MCTS生成的tokens拼接到prompt
            current_prompt = torch.cat([prompt, best_new_tokens])
            # 解码MCTS生成的部分
            mcts_text = TextTokenizer.decode(best_new_tokens[best_new_tokens != 0])
            if mcts_text:
                print(mcts_text, end="", flush=True)
                output_text += mcts_text
                mcts_used = True
        else:
            current_prompt = prompt.clone()
        
        # 【修复】继续标准自回归生成（无论是否使用了MCTS）
        result = model(current_prompt, use_cache=True)
        if isinstance(result, tuple):
            logits, past_key_values = result
        else:
            logits = result

        step = 0
        while max_generate_tokens is None or step < max_generate_tokens:
            try:
                next_logits = logits[-1]
                if step < min_new_tokens:
                    next_logits = next_logits.clone()
                    next_logits[TextTokenizer.END_GENERATION_TOKEN] = float("-inf")

                probs = torch.softmax(next_logits / CONFIG["temperature"], dim=-1)
                index = int(torch.multinomial(probs, 1).item())

                if index == TextTokenizer.END_GENERATION_TOKEN:
                    break

                decoded_piece = TextTokenizer.decode(torch.tensor([index]))
                if decoded_piece:
                    print(decoded_piece, end="", flush=True)
                    output_text += decoded_piece

                next_token = torch.tensor([index], device=device)
                current_prompt = torch.cat([current_prompt, next_token])
                
                # 推理时也需要处理返回值
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
            except Exception as e:
                print(f"Error during generation: {e}", flush=True)
                break
        return output_text


def _run_train_step(train_tensor: torch.Tensor, target_mask: torch.Tensor, preview: torch.Tensor, advantage_weight: float = 1.0, show_preview: bool = True) -> None:
    """执行单步训练
    
    Args:
        train_tensor: 训练张量
        target_mask: 目标掩码
        preview: 预览张量
        advantage_weight: 优势加权因子(用于GRPO)
        show_preview: 是否显示预览输出(默认True,QA模式下可设为False避免重复)
    """
    model.train()
    optimizer.zero_grad()

    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        result = model(train_tensor, use_cache=False)
        if isinstance(result, tuple):
            logits = result[0]
        else:
            logits = result

        # 应用目标掩码并进行 next-token prediction 对齐
        # 对于 next-token prediction，targets 应该是 train_tensor 右移一位
        # 确保 logits 和 targets 长度相同
        if len(train_tensor) > 1:
            # 正确的 next-token prediction 对齐
            # logits 对应位置 i，targets 对应位置 i+1
            masked_logits = logits[:-1][target_mask[:-1]]
            masked_targets = train_tensor[1:][target_mask[:-1]]
            
            if len(masked_logits) > 0 and len(masked_targets) > 0:
                loss = loss_func(masked_logits, masked_targets)
            else:
                loss = torch.tensor(0.0, device=device)
        else:
            loss = torch.tensor(0.0, device=device)
        # 应用GRPO优势加权
        loss = loss * advantage_weight
        record_loss(loss.item())

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    # 输出训练预览(可选)
    if show_preview:
        try:
            decoded_preview = TextTokenizer.decode(preview[preview != 0])
            print(decoded_preview, end="", flush=True)
        except Exception as e:
            print(f"[Warning] Failed to decode preview: {e}", flush=True)
        print("", flush=True)