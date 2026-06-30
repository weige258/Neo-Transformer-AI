from typing import List, Tuple, Optional
import sys
import os
import torch
import time
import logging
import math
from collections import Counter
from config import CONFIG
from model import MainModel
from record import record_loss, evaluate_rl_readiness
from tokenizer import TextTokenizer
from rl import SelfRewardModel, LightweightPPO

# ═══════════════════════════════════════════════════════
# 全动态计算辅助函数
# ═══════════════════════════════════════════════════════

def _get_gpu_free_memory_ratio() -> float:
    """获取GPU空闲显存比例（0.0-1.0）"""
    if not torch.cuda.is_available():
        return 1.0
    try:
        total = torch.cuda.get_device_properties(0).total_memory
        reserved = torch.cuda.memory_reserved()
        allocated = torch.cuda.memory_allocated()
        # free = total - reserved + (reserved - allocated)  # 包含缓存
        free = total - allocated
        return max(0.0, min(1.0, free / total))
    except (RuntimeError, ValueError):
        return 1.0

def _get_cpu_free_memory_ratio() -> float:
    """获取CPU空闲内存比例（0.0-1.0）"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return max(0.0, min(1.0, mem.available / mem.total))
    except ImportError:
        return 1.0

def _estimate_question_complexity(text: str) -> float:
    """估计问题复杂度（0.0-1.0）
    
    基于多种启发式特征：
    - 问题长度
    - 标点复杂度
    - 关键词密度（为什么/如何/解释等）
    - 代码/数学符号存在性
    """
    if not text:
        return 0.0
    
    score = 0.0
    
    # 1. 长度因子（长尾分布，用对数压缩）
    length_score = min(1.0, math.log1p(len(text)) / math.log1p(500))
    score += length_score * 0.25
    
    # 2. 复杂标点密度（问号、分号、括号等）
    complex_punct = sum(1 for c in text if c in '？?;；:：（）()[]{}')
    punct_score = min(1.0, complex_punct / 5)
    score += punct_score * 0.15
    
    # 3. 复杂关键词
    complex_keywords = ['为什么', '如何', '解释', '比较', '分析', '推导', '证明', '优化', '设计', '实现']
    keyword_count = sum(1 for kw in complex_keywords if kw in text)
    keyword_score = min(1.0, keyword_count / 3)
    score += keyword_score * 0.25
    
    # 4. 代码/数学符号
    code_math_symbols = sum(1 for c in text if c in '`=+-*/<>[]{}|&^%~')
    code_score = min(1.0, code_math_symbols / 10)
    score += code_score * 0.2
    
    # 5. 句子数量（多句子通常更复杂）
    sentence_count = text.count('。') + text.count('.') + text.count('？') + text.count('?')
    sentence_score = min(1.0, sentence_count / 5)
    score += sentence_score * 0.15
    
    return max(0.0, min(1.0, score))

def _compute_repeat_score(token_history: List[int]) -> float:
    """计算重复模式得分（0.0-1.0）
    
    基于最近生成token的重复程度。
    0.0 = 无重复，1.0 = 严重重复
    """
    if len(token_history) < 8:
        return 0.0
    
    # 检测2-gram、3-gram、4-gram的连续重复
    max_repeat_score = 0.0
    for n in [2, 3, 4]:
        if len(token_history) >= n * 3:
            last_n = tuple(token_history[-n:])
            prev_n = tuple(token_history[-(n*2):-n])
            prev_n2 = tuple(token_history[-(n*3):-(n*2)])
            if last_n == prev_n == prev_n2:
                # 3次连续重复，得分基于n-gram长度
                max_repeat_score = max(max_repeat_score, n / 4.0)
    
    # 检测局部重复率（最近32个token中唯一token的比例）
    recent = token_history[-32:]
    if len(recent) >= 8:
        unique_ratio = len(set(recent)) / len(recent)
        # unique_ratio越低，重复越严重
        repeat_score = max(0.0, 1.0 - unique_ratio * 2.0)
        max_repeat_score = max(max_repeat_score, repeat_score)
    
    return min(1.0, max_repeat_score)

def _compute_diversity_score(token_history: List[int]) -> float:
    """计算N-gram多样性得分（0.0-1.0）
    
    基于最近生成token的多样性。
    1.0 = 非常多样，0.0 = 完全重复
    """
    if len(token_history) < 8:
        return 1.0
    
    recent = token_history[-64:]
    if len(recent) < 8:
        return 1.0
    
    # 计算2-gram、3-gram、4-gram的唯一比例
    diversity_scores = []
    for n in [2, 3, 4]:
        if len(recent) >= n + 4:
            ngrams = [tuple(recent[i:i+n]) for i in range(len(recent) - n + 1)]
            if ngrams:
                unique_ratio = len(set(ngrams)) / len(ngrams)
                diversity_scores.append(unique_ratio)
    
    if not diversity_scores:
        return 1.0
    
    return max(0.0, min(1.0, sum(diversity_scores) / len(diversity_scores)))

def _compute_entropy_trend(entropy_history: List[float]) -> float:
    """计算熵趋势（最近几步的平均变化）
    
    正值：熵在增加（生成变得更多样）
    负值：熵在减少（生成变得更确定/可能重复）
    """
    if len(entropy_history) < 5:
        return 0.0
    
    recent = entropy_history[-10:]
    if len(recent) < 3:
        return 0.0
    
    # 计算线性趋势（简单差分平均）
    diffs = [recent[i] - recent[i-1] for i in range(1, len(recent))]
    return sum(diffs) / len(diffs)


# 【显存优化】设置 PyTorch CUDA 内存分配策略，避免显存碎片化
# expandable_segments:True 允许内存段动态扩展，减少碎片
# max_split_size_mb:512 限制最大分割块大小，减少碎片化导致的OOM
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True,max_split_size_mb:512')

if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────
# 全局训练状态
# ──────────────────────────────────────────────────────────
model = MainModel()
model.to(device)

# 检查是否有可用的预训练权重
pretrained_path = "model.pth"
if os.path.exists(pretrained_path):
    try:
        file_size = os.path.getsize(pretrained_path)
        if file_size < 1024:
            print(f"Warning: {pretrained_path} 文件过小 ({file_size} 字节)，可能已损坏，跳过加载")
            bak_path = pretrained_path + ".bak"
            try:
                os.rename(pretrained_path, bak_path)
                print(f"已将损坏文件重命名为 {bak_path}")
            except Exception:
                pass
        else:
            try:
                state_dict = torch.load(pretrained_path, map_location=device, weights_only=True)
            except Exception:
                print(f"Warning: weights_only=True 加载失败，尝试 weights_only=False ...")
                state_dict = torch.load(pretrained_path, map_location=device, weights_only=False)
            model.load_state_dict(state_dict)
            print(f"Loaded pretrained model from {pretrained_path}")
    except Exception as e:
        print(f"Warning: Failed to load pretrained model: {e}")
        print(f"将从随机初始化权重开始训练，新权重会自动覆盖 {pretrained_path}")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=float(CONFIG.get("lr", 3e-4)),
    weight_decay=float(CONFIG.get("weight_decay", 0.01)),
)

# 学习率调度器：SGDR + ReduceLROnPlateau 组合
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau

sgdr_scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=int(CONFIG.get("sgdr_t0", 1500)),
    T_mult=int(CONFIG.get("sgdr_t_mult", 2)),
    eta_min=float(CONFIG.get("min_lr", 1e-6)),
)
plateau_scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=float(CONFIG.get("plateau_factor", 0.5)),
    patience=int(CONFIG.get("plateau_patience", 500)),
    min_lr=float(CONFIG.get("min_lr", 1e-7)),
)

class CombinedScheduler:
    """SGDR + ReduceLROnPlateau 组合调度器"""
    def __init__(self, sgdr, plateau, warmup_steps=300, base_lr=3e-4):
        self.sgdr = sgdr
        self.plateau = plateau
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.step_count = 0
        self.best_loss = float('inf')
        self.plateau_counter = 0
        
    def step(self, loss=None):
        self.step_count += 1
        
        # Warmup阶段
        if self.step_count <= self.warmup_steps:
            warmup_factor = self.step_count / self.warmup_steps
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.base_lr * warmup_factor
            return optimizer.param_groups[0]['lr']
        
        # SGDR调度
        self.sgdr.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # ReduceLROnPlateau（每10步检查一次）
        if loss is not None and self.step_count % 10 == 0:
            self.plateau.step(loss)
            plateau_lr = optimizer.param_groups[0]['lr']
            if plateau_lr < current_lr:
                current_lr = plateau_lr
        
        return current_lr

lr_scheduler = CombinedScheduler(sgdr_scheduler, plateau_scheduler)

# 混合精度训练 (AMP)
use_amp = bool(CONFIG.get("use_amp", True))
amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
scaler = torch.amp.GradScaler('cuda', enabled=use_amp and amp_dtype == torch.float16)

print(f"Using device: {device}")
print(f"AMP enabled: {use_amp} AMP dtype: {amp_dtype}")

loss_func = torch.nn.CrossEntropyLoss()

# 梯度累积步数
GRADIENT_ACCUMULATION_STEPS = int(CONFIG.get("gradient_accumulation_steps", 4))

# 学习率调度器步进间隔（Gemini修复：防止SGDR震荡过于频繁）
LR_SCHEDULER_STEP_INTERVAL = int(CONFIG.get("lr_scheduler_step_interval", 4))

# PPO配置（Gemini修复：增加episode收集量，减少方差）
RL_MIN_EPISODES = int(CONFIG.get("rl_min_episodes", 32))
RL_UPDATE_BATCH_SIZE = int(CONFIG.get("rl_update_batch_size", 8))
RL_UPDATE_INTERVAL = int(CONFIG.get("rl_update_interval", 4))

# 训练轮数计数器
training_rounds = 0
optimizer_step_count = 0

# 自奖励模型和PPO训练器
# 【修复】PPO使用独立优化器，避免与SFT共享导致梯度污染
reward_model = SelfRewardModel(device)
ppo_trainer = LightweightPPO(model, reward_model, device)


def _build_train_sequence(segments: List[Tuple[torch.Tensor | int, bool]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """构建训练序列和对应的target mask
    
    Args:
        segments: 列表，每个元素是 (tokens_or_special_id, should_compute_loss)
    
    Returns:
        (train_tensor, target_mask)
    """
    tokens = []
    mask = []
    
    for item, compute_loss in segments:
        if isinstance(item, int):
            # 特殊token
            tokens.append(item)
            mask.append(compute_loss)
        else:
            # 普通token tensor
            item_list = item.tolist() if isinstance(item, torch.Tensor) else list(item)
            tokens.extend(item_list)
            mask.extend([compute_loss] * len(item_list))
    
    train_tensor = torch.tensor(tokens, dtype=torch.long, device=device)
    target_mask = torch.tensor(mask, dtype=torch.bool, device=device)
    return train_tensor, target_mask


def _prepare_training_data(ask: str, answer: str, history_context: str = None) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """准备QA训练数据"""
    ask_tensor = TextTokenizer.encode(ask)
    answer_tensor = TextTokenizer.encode(answer)
    
    if ask_tensor.numel() == 0 or answer_tensor.numel() == 0:
        return None, None, None
    
    if history_context and history_context.strip():
        history_tensor = TextTokenizer.encode(history_context)
        train_tensor, target_mask = _build_train_sequence([
            (TextTokenizer.START_GENERATION_TOKEN, False),
            (history_tensor, False),
            (TextTokenizer.END_GENERATION_TOKEN, False),
            (TextTokenizer.HISTORY_CONTEXT_START_TOKEN, False),
            (ask_tensor, False),
            (TextTokenizer.START_GENERATION_TOKEN, True),  # 【修复】学习在ask后输出START
            (answer_tensor, True),
            (TextTokenizer.END_GENERATION_TOKEN, True),
        ])
    else:
        train_tensor, target_mask = _build_train_sequence([
            (ask_tensor, False),
            (TextTokenizer.START_GENERATION_TOKEN, True),  # 【修复】学习在ask后输出START
            (answer_tensor, True),
            (TextTokenizer.END_GENERATION_TOKEN, True),
        ])
    
    preview = answer_tensor
    return train_tensor, target_mask, preview











def _min_p_sampling(logits: torch.Tensor, min_p: float, temperature: float = 1.0) -> torch.Tensor:
    """Min-p 采样 (Nguyen et al., ICLR 2025)

    设计说明（温度处理流程）：
    1. 本函数内部：用 logits/temperature 计算概率 → 确定过滤阈值
    2. 返回原始 logits（只过滤不合格token为-inf，不修改logits值）
    3. 调用方：对返回的 logits 再次执行 softmax(logits/temperature) 采样
    这样温度只被应用一次（在调用方的softmax中），本函数的温度仅用于确定过滤阈值。
    """
    if min_p <= 0.0:
        return logits
    
    logits_temp = logits / temperature
    probs = torch.softmax(logits_temp, dim=-1)
    p_max = probs.max().item()
    threshold = p_max * min_p
    
    logits_filtered = torch.where(
        probs < threshold, 
        torch.full_like(logits, float("-inf")), 
        logits
    )
    return logits_filtered


def _top_p_sampling(logits: torch.Tensor, top_p: float, temperature: float = 1.0) -> torch.Tensor:
    """Top-p (Nucleus) 采样

    设计说明（与_min_p_sampling一致）：
    1. 本函数内部：用 logits/temperature 计算概率 → 确定过滤阈值
    2. 返回原始 logits（只过滤不合格token为-inf，不修改logits值）
    3. 调用方：对返回的 logits 再次执行 softmax(logits/temperature) 采样
    这样温度只被应用一次（在调用方的softmax中），本函数的温度仅用于确定过滤阈值。
    """
    if top_p >= 1.0 or top_p <= 0.0:
        return logits
    
    logits_temp = logits / temperature
    probs = torch.softmax(logits_temp, dim=-1)
    
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    mask = cumsum_probs > top_p
    
    if mask.numel() > 0:
        mask[0] = False
    
    sorted_logits_orig = logits[sorted_indices]
    sorted_logits_orig = torch.where(mask, torch.full_like(sorted_logits_orig, float("-inf")), sorted_logits_orig)
    
    logits_filtered = torch.full_like(logits, float("-inf"))
    logits_filtered[sorted_indices] = sorted_logits_orig
    
    return logits_filtered


def generation(text: str, history_context: str = None, max_generate_tokens: int|None = None, thinking_available: bool = True) -> str:
    """生成函数 (Min-p采样 + Repetition Penalty + CoT完整性保护)

    解码策略：
    - Min-p 采样替代 top-k/top-p
    - Repetition Penalty 防止重复输出
    - 思考阶段长度限制 + 强制回答期确保 CoT→回复 完整过渡
    """
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    
    if not text or not isinstance(text, str):
        return "无效输入"
    
    # 【全动态生成长度】运行时根据多因素动态计算最大生成长度
    # 基于：问题长度、问题复杂度、GPU空闲显存、CPU空闲内存、历史生成熵趋势
    question_len = len(text) if text else 0
    
    # 1. 计算问题复杂度（基于字符特征）
    complexity_score = _estimate_question_complexity(text)
    
    # 2. 获取硬件状态
    gpu_free_ratio = _get_gpu_free_memory_ratio()
    cpu_free_ratio = _get_cpu_free_memory_ratio()
    
    # 3. 动态计算基础长度
    gen_len_base_ratio = float(CONFIG.get("gen_len_base_ratio", 8.0))
    gen_len_complexity_factor = float(CONFIG.get("gen_len_complexity_factor", 1.5))
    gen_len_memory_sensitivity = float(CONFIG.get("gen_len_memory_sensitivity", 0.3))
    gen_len_entropy_sensitivity = float(CONFIG.get("gen_len_entropy_sensitivity", 0.5))
    
    # 基础长度 = 问题长度 * 基础倍数 * 复杂度因子
    base_len = question_len * gen_len_base_ratio * (1.0 + complexity_score * (gen_len_complexity_factor - 1.0))
    
    # 显存调节：显存紧张时降低长度
    memory_factor = 1.0 - (1.0 - gpu_free_ratio) * gen_len_memory_sensitivity
    memory_factor *= 1.0 - (1.0 - cpu_free_ratio) * gen_len_memory_sensitivity * 0.5
    
    # 动态长度
    dynamic_max_len = int(base_len * memory_factor)
    
    # 绝对边界保护
    gen_min = int(CONFIG.get("gen_len_min_absolute", 64))
    gen_max = int(CONFIG.get("gen_len_max_absolute", 4096))
    dynamic_max_len = max(gen_min, min(gen_max, dynamic_max_len))
    
    # 设置绝对上限防止极端情况下的死循环
    absolute_max_tokens = min(gen_max, dynamic_max_len * 2)
    has_token_limit = max_generate_tokens is not None
    if max_generate_tokens is None:
        max_generate_tokens = dynamic_max_len
    
    model.eval()
    output_text = ""

    # 【修复】生成prompt与训练格式一致
    # 训练格式: ask + START_GENERATION + THINK_START + think + THINK_END + answer + END
    # 生成格式: ask + START_GENERATION（然后模型自己生成后续内容）
    text_tensor = TextTokenizer.encode(text).to(device)
    
    if history_context and history_context.strip():
        history_tensor = TextTokenizer.encode(history_context).to(device)
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
            text_tensor,
            torch.tensor([TextTokenizer.START_GENERATION_TOKEN], device=device),
        ])

    print("\n---Generated reply:", flush=True)

    max_generate_tokens = max(1, int(max_generate_tokens))
    
    # 【全动态EDT温度】系数
    temp_base = float(CONFIG.get("temp_base", 0.7))
    temp_entropy_scale = float(CONFIG.get("temp_entropy_scale", 0.4))
    temp_repetition_sensitivity = float(CONFIG.get("temp_repetition_sensitivity", 0.6))
    temp_length_decay = float(CONFIG.get("temp_length_decay", 0.001))
    temp_min_clip = float(CONFIG.get("temp_min_clip", 0.3))
    temp_max_clip = float(CONFIG.get("temp_max_clip", 1.5))
    enable_edt = bool(CONFIG.get("enable_edt", True))
    
    min_p = float(CONFIG.get("min_p", 0.04))
    top_k = int(CONFIG.get("top_k", 50))
    top_p = float(CONFIG.get("top_p", 0.9))
    
    # 【全动态重复惩罚】系数
    rep_penalty_scale = float(CONFIG.get("rep_penalty_scale", 0.15))
    rep_penalty_length_factor = float(CONFIG.get("rep_penalty_length_factor", 0.002))
    rep_penalty_repeat_sensitivity = float(CONFIG.get("rep_penalty_repeat_sensitivity", 2.0))
    rep_penalty_entropy_factor = float(CONFIG.get("rep_penalty_entropy_factor", 0.8))
    presence_penalty = float(CONFIG.get("presence_penalty", 0.1))
    
    # 【全动态强制回答步数】运行时计算
    force_answer_scale = float(CONFIG.get("force_answer_scale", 1.2))
    force_answer_min_absolute = int(CONFIG.get("force_answer_min_absolute", 16))
    force_answer_complexity_exp = float(CONFIG.get("force_answer_complexity_exp", 0.5))
    force_answer_min_steps = max(force_answer_min_absolute, int(question_len * force_answer_scale * (1.0 + complexity_score ** force_answer_complexity_exp)))

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        thinking_started = False
        force_answer_steps = 0
        
        # 【修复】推理prompt与训练格式一致
        # 训练格式: ask + START_GENERATION + THINK_START + think + THINK_END + answer + END
        # 推理格式: ask + START_GENERATION（然后模型自己生成后续内容）
        if thinking_available:
            # 给模型机会自己生成THINK_START
            thinking_started = False
        
        result = model(prompt, use_cache=True)
        if isinstance(result, tuple):
            logits, past_key_values = result
        else:
            logits = result
            past_key_values = None

        step = 0
        # 【修复】使用滑动窗口替代累积 Counter，防止长生成时过度惩罚
        generated_tokens = Counter()
        _token_history = []  # 记录最近生成的 token 序列
        _entropy_history = []  # 【全动态】记录生成过程中的熵历史
        # 【修复】累积完整的生成token序列，用于在推理时构建正确的special_mask保护特殊Token
        _full_generated_ids = []
        
        # 思维链决策：
        # force_thinking_chain=True（默认）：强制注入THINK_START_TOKEN，确保模型先思考再回答
        # force_thinking_chain=False：让模型自己决定是否使用思维链
        force_thinking = bool(CONFIG.get("force_thinking_chain", True))
        
        if thinking_available and force_thinking and not thinking_started:
            think_start_token = torch.tensor([TextTokenizer.THINK_START_TOKEN], device=device)
            result = model(think_start_token, past_key_values=past_key_values, use_cache=True)
            if isinstance(result, tuple):
                logits, past_key_values = result
            else:
                logits = result
                past_key_values = None
            thinking_started = True
            print(f"{BLUE}", end="", flush=True)
        
        hard_limit = absolute_max_tokens + 100
        temperature = temp_base
        
        while step < min(max_generate_tokens, hard_limit):
            try:
                # 获取最后一个token的logits
                if logits.dim() == 3:
                    next_logits = logits[0, -1].clone()
                else:
                    next_logits = logits[-1].clone()

                # 强制回答阶段：禁止特殊token
                if force_answer_steps > 0:
                    next_logits[TextTokenizer.END_GENERATION_TOKEN] = float("-inf")
                    next_logits[TextTokenizer.UNKNOWN_TOKEN] = float("-inf")
                    next_logits[TextTokenizer.START_GENERATION_TOKEN] = float("-inf")
                    # 【修复】强制回答阶段禁止所有特殊token，包括THINK_START和THINK_END
                    next_logits[TextTokenizer.THINK_START_TOKEN] = float("-inf")
                    next_logits[TextTokenizer.THINK_END_TOKEN] = float("-inf")
                    next_logits[TextTokenizer.HISTORY_CONTEXT_START_TOKEN] = float("-inf")
                    next_logits[TextTokenizer.HISTORY_CONTEXT_END_TOKEN] = float("-inf")
                    force_answer_steps -= 1

                # 【BUG #2修复】Repetition penalty + Frequency penalty + N-gram阻断 三重防重复
                frequency_penalty = float(CONFIG.get("frequency_penalty", 0.3))
                
                # 【全动态重复惩罚】运行时计算惩罚强度
                current_len = len(_token_history)
                
                # 1. 计算重复模式得分（0.0-1.0）
                repeat_score = _compute_repeat_score(_token_history)
                
                # 2. 计算N-gram多样性得分（0.0-1.0，越高越多样）
                diversity_score = _compute_diversity_score(_token_history)
                
                # 3. 计算熵趋势（最近10步的平均熵变化）
                entropy_trend = _compute_entropy_trend(_entropy_history)
                
                # 4. 全动态惩罚公式
                # penalty = 1.0 + scale * (length_factor + repeat_sensitivity * repeat_score - entropy_factor * entropy_trend)
                length_factor = min(1.0, current_len * rep_penalty_length_factor)
                repetition_penalty = 1.0 + rep_penalty_scale * (
                    length_factor 
                    + rep_penalty_repeat_sensitivity * repeat_score 
                    - rep_penalty_entropy_factor * max(0, entropy_trend)
                )
                # 多样性调节：多样性低时增强惩罚
                repetition_penalty *= 1.0 + (1.0 - diversity_score) * 0.5
                
                # 裁剪到合理范围
                repetition_penalty = max(1.0, min(2.0, repetition_penalty))
                
                if repetition_penalty > 1.0 and len(_token_history) > 0:
                    # 动态惩罚窗口：基于生成长度
                    window_size = min(128, max(32, current_len // 2))
                    recent_tokens = _token_history[-window_size:]
                    recent_counter = Counter(recent_tokens)
                    # 【新增】Presence Penalty：记录窗口内出现过的所有token（不管次数）
                    recent_set = set(recent_tokens)
                    for token_id, count in recent_counter.items():
                        if token_id < next_logits.size(0) and count > 0:
                            penalty = repetition_penalty ** min(count, 3)
                            if next_logits[token_id] > 0:
                                next_logits[token_id] /= penalty
                            else:
                                next_logits[token_id] *= penalty
                            if frequency_penalty > 0:
                                next_logits[token_id] -= frequency_penalty * min(count, 3)
                            if presence_penalty > 0 and token_id in recent_set:
                                next_logits[token_id] -= presence_penalty

                # 【修复】N-gram重复阻断：只在严重重复时触发
                # 旧版：2-gram重复3次就阻断（过于激进）
                # 新版：只在4-gram重复3次时才阻断（更宽松）
                blocked_tokens = set()
                if len(_token_history) >= 16:  # 增加触发长度阈值
                    for n in [4]:  # 只检查4-gram，避免误伤正常重复
                        if len(_token_history) >= n * 3:
                            last_n = tuple(_token_history[-n:])
                            prev_n = tuple(_token_history[-(n*2):-n])
                            prev_n2 = tuple(_token_history[-(n*3):-(n*2)])
                            if last_n == prev_n == prev_n2:
                                repeat_token = last_n[-1]
                                if repeat_token < next_logits.size(0):
                                    next_logits[repeat_token] = float("-inf")
                                    blocked_tokens.add(repeat_token)
                                    print(f"[Gen] Blocked 4-gram repeat", flush=True)
                
                # 【新增】阻断后回退策略：如果阻断导致概率分布异常，提高温度并放宽采样
                if blocked_tokens:
                    # 检查是否所有高概率token都被阻断
                    temp_probs = torch.softmax(next_logits / max(temperature, 0.1), dim=-1)
                    if temp_probs.max().item() < 0.1 or torch.isnan(temp_probs).any():
                        # 概率分布紊乱，紧急回退：提高温度，允许更多token
                        temperature = min(1.2, temperature * 1.5)
                        print(f"[Gen] Emergency fallback: temp={temperature:.2f}", flush=True)

                # 【全动态EDT温度】基于多因素的temperature计算
                raw_probs = torch.softmax(next_logits, dim=-1)
                entropy = -(raw_probs * torch.log(raw_probs + 1e-10)).sum().item()
                _entropy_history.append(entropy)
                
                if enable_edt:
                    # 多因素temperature：
                    # temp = base + entropy_scale * (target_entropy - entropy) - length_decay * current_len - repetition_sensitivity * repeat_score
                    # 【修复】target_entropy 从 1.0 调整为 log(top_k)
                    # 词表60000的均匀分布熵≈11.0，目标1.0导致温度几乎总被压到下限
                    # log(top_k)=log(50)≈3.9 是更合理的目标：模型应在top-50内集中选择
                    target_entropy = max(2.0, math.log(max(top_k, 2)))
                    temperature = temp_base + temp_entropy_scale * (target_entropy - entropy)
                    # 长度衰减：长生成时降低温度稳定输出
                    temperature -= temp_length_decay * current_len
                    # 重复敏感度：检测到重复时提高温度打破循环
                    temperature += temp_repetition_sensitivity * repeat_score
                    # 裁剪
                    temperature = max(temp_min_clip, min(temp_max_clip, temperature))
                else:
                    temperature = temp_base

                # Min-p 采样（返回原始logits，只过滤不合格的token）
                if min_p > 0.0:
                    next_logits = _min_p_sampling(next_logits, min_p, temperature)

                # 【新增】Top-p (Nucleus) 采样：累积概率过滤
                if top_p < 1.0 and top_p > 0.0:
                    next_logits = _top_p_sampling(next_logits, top_p, temperature)

                # 【新增】top-k过滤：限制采样范围，防止选中极低概率token
                if top_k > 0:
                    vals, indices = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits = torch.full_like(next_logits, float("-inf"))
                    next_logits[indices] = vals

                # 【修复】应用temperature后计算概率
                # _min_p_sampling返回的是原始logits，需要在这里除temperature
                probs = torch.softmax(next_logits / temperature, dim=-1)
                
                # 【修复】防止所有logits为-inf导致softmax产生nan
                if torch.isnan(probs).any() or probs.sum().item() < 1e-6:
                    print(f"\n[Warning] 概率分布异常，使用随机采样", flush=True)
                    probs = torch.ones_like(probs) / probs.size(0)
                
                index = int(torch.multinomial(probs, 1).item())
                
                # 思考阶段或强制回答阶段：禁止 END_TOKEN
                # 【修复】循环重采样直到不是END_TOKEN，防止一次重采样仍得到END
                max_resample = 10
                resample_count = 0
                while index == TextTokenizer.END_GENERATION_TOKEN and (force_answer_steps > 0 or thinking_started) and resample_count < max_resample:
                    next_logits[index] = float("-inf")
                    probs = torch.softmax(next_logits / temperature, dim=-1)
                    if torch.isinf(next_logits).all() or probs.sum().item() < 1e-6:
                        break
                    index = int(torch.multinomial(probs, 1).item())
                    resample_count += 1
                
                # 处理特殊token
                should_skip_output = False
                
                if index == TextTokenizer.THINK_END_TOKEN:
                    if thinking_available and thinking_started:
                        # 正常结束思考阶段，进入回答阶段
                        thinking_started = False
                        force_answer_steps = force_answer_min_steps
                        print(f"\n{GREEN}", end="", flush=True)
                        should_skip_output = True
                    elif thinking_available and not thinking_started:
                        # 【修复】模型没有使用CoT（没有THINK_START），但生成了THINK_END
                        # 这种情况下忽略THINK_END，继续生成
                        should_skip_output = True
                    else:
                        break

                elif index == TextTokenizer.END_GENERATION_TOKEN:
                    if thinking_started and thinking_available:
                        # 【修复】思考阶段异常结束：模型没生成THINK_END就输出END
                        # 不直接break，而是强制过渡到回答阶段
                        print(f"\n{YELLOW}[CoT Guard] 思考阶段异常结束，强制过渡到回答{RESET}", flush=True)
                        thinking_started = False
                        force_answer_steps = force_answer_min_steps
                        should_skip_output = True
                    else:
                        # 正常结束或非思考阶段
                        if not has_token_limit and step >= absolute_max_tokens - 1:
                            print(f"\n[Stop] 达到绝对安全上限({absolute_max_tokens})，强制结束", flush=True)
                        break

                elif index == TextTokenizer.THINK_START_TOKEN:
                    if thinking_available and not thinking_started:
                        # 【修复】模型在生成过程中决定开始CoT
                        thinking_started = True
                        should_skip_output = True
                        print(f"{BLUE}", end="", flush=True)
                    elif not thinking_available:
                        should_skip_output = True

                elif index == TextTokenizer.START_GENERATION_TOKEN:
                    # 【修复】防止模型重复输出START_GENERATION_TOKEN
                    should_skip_output = True

                # 输出解码
                if not should_skip_output:
                    decoded_piece = TextTokenizer.decode(torch.tensor([index]))
                    
                    if decoded_piece:
                        if thinking_started:
                            print(f"{BLUE}{decoded_piece}{RESET}", end="", flush=True)
                        else:
                            print(f"{GREEN}{decoded_piece}{RESET}", end="", flush=True)
                        
                        output_text += decoded_piece

                # 记录生成的token
                if index not in (
                    TextTokenizer.THINK_START_TOKEN,
                    TextTokenizer.THINK_END_TOKEN,
                    TextTokenizer.START_GENERATION_TOKEN,
                    TextTokenizer.END_GENERATION_TOKEN,
                ):
                    generated_tokens[index] += 1
                    _token_history.append(index)  # 【修复】同时记录到滑动窗口历史
                
                # 【修复】累积完整token历史（包含特殊Token），用于构建special_mask
                _full_generated_ids.append(index)

                # 下一步
                next_token = torch.tensor([index], device=device)
                result = model(
                    next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                if isinstance(result, tuple):
                    logits, past_key_values = result
                else:
                    logits = result
                    past_key_values = None
                
                step += 1
                
            except Exception as e:
                import traceback
                print(f"\nError during generation at step {step}: {e}", flush=True)
                traceback.print_exc()
                break
        
        # CoT 完整性检测
        if thinking_started and thinking_available:
            print(f"\n{YELLOW}[CoT Guard] 未检测到回答，正在强制过渡...{RESET}", flush=True)
            
            think_end_token = torch.tensor([TextTokenizer.THINK_END_TOKEN], device=device)
            result = model(think_end_token, past_key_values=past_key_values, use_cache=True)
            if isinstance(result, tuple):
                logits, past_key_values = result
            else:
                logits = result
                past_key_values = None
            
            thinking_started = False
            force_answer_steps = force_answer_min_steps
            step += 1
            
            # 强制回答阶段
            max_force_answer_steps = max_generate_tokens - step
            force_answer_count = 0
            while step < max_generate_tokens and force_answer_count < max_force_answer_steps:
                try:
                    if logits.dim() == 3:
                        next_logits = logits[0, -1].clone()
                    else:
                        next_logits = logits[-1].clone()
                    
                    # 【BUG #2修复】不再永久封死END_TOKEN，只在force_answer_steps>0时禁止
                    # 允许模型在适当时候自然输出END_TOKEN结束
                    if force_answer_steps > 0:
                        force_answer_steps -= 1
                        next_logits[TextTokenizer.END_GENERATION_TOKEN] = float("-inf")
                    
                    next_logits[TextTokenizer.UNKNOWN_TOKEN] = float("-inf")
                    next_logits[TextTokenizer.START_GENERATION_TOKEN] = float("-inf")
                    next_logits[TextTokenizer.THINK_START_TOKEN] = float("-inf")
                    next_logits[TextTokenizer.THINK_END_TOKEN] = float("-inf")
                    next_logits[TextTokenizer.HISTORY_CONTEXT_START_TOKEN] = float("-inf")
                    next_logits[TextTokenizer.HISTORY_CONTEXT_END_TOKEN] = float("-inf")
                    
                    # 【全动态重复惩罚】与主循环一致
                    current_len = len(_token_history)
                    repeat_score = _compute_repeat_score(_token_history)
                    diversity_score = _compute_diversity_score(_token_history)
                    entropy_trend = _compute_entropy_trend(_entropy_history)
                    
                    length_factor = min(1.0, current_len * rep_penalty_length_factor)
                    repetition_penalty = 1.0 + rep_penalty_scale * (
                        length_factor 
                        + rep_penalty_repeat_sensitivity * repeat_score 
                        - rep_penalty_entropy_factor * max(0, entropy_trend)
                    )
                    repetition_penalty *= 1.0 + (1.0 - diversity_score) * 0.5
                    repetition_penalty = max(1.0, min(2.0, repetition_penalty))
                    
                    if repetition_penalty > 1.0 and len(_token_history) > 0:
                        window_size = min(128, max(32, current_len // 2))
                        recent_tokens = _token_history[-window_size:]
                        recent_counter = Counter(recent_tokens)
                        recent_set = set(recent_tokens)
                        for token_id, count in recent_counter.items():
                            if token_id < next_logits.size(0) and count > 0:
                                penalty = repetition_penalty ** min(count, 3)
                                if next_logits[token_id] > 0:
                                    next_logits[token_id] /= penalty
                                else:
                                    next_logits[token_id] *= penalty
                                if frequency_penalty > 0:
                                    next_logits[token_id] -= frequency_penalty * min(count, 3)
                                if presence_penalty > 0 and token_id in recent_set:
                                    next_logits[token_id] -= presence_penalty

                    # 【新增】N-gram重复阻断（强制回答阶段同样需要）
                    blocked_tokens_fallback = set()
                    if len(_token_history) >= 16:
                        for n in [4]:
                            if len(_token_history) >= n * 3:
                                last_n = tuple(_token_history[-n:])
                                prev_n = tuple(_token_history[-(n*2):-n])
                                prev_n2 = tuple(_token_history[-(n*3):-(n*2)])
                                if last_n == prev_n == prev_n2:
                                    repeat_token = last_n[-1]
                                    if repeat_token < next_logits.size(0):
                                        next_logits[repeat_token] = float("-inf")
                                        blocked_tokens_fallback.add(repeat_token)
                    
                    # 【新增】强制回答阶段阻断后回退
                    if blocked_tokens_fallback:
                        temp_probs = torch.softmax(next_logits / max(temperature, 0.1), dim=-1)
                        if temp_probs.max().item() < 0.1 or torch.isnan(temp_probs).any():
                            temperature = min(1.2, temperature * 1.5)
                    
                    # 【全动态EDT温度】强制回答阶段同样使用
                    raw_probs = torch.softmax(next_logits, dim=-1)
                    entropy = -(raw_probs * torch.log(raw_probs + 1e-10)).sum().item()
                    _entropy_history.append(entropy)
                    
                    if enable_edt:
                        target_entropy = max(2.0, math.log(max(top_k, 2)))
                        temperature = temp_base + temp_entropy_scale * (target_entropy - entropy)
                        temperature -= temp_length_decay * current_len
                        temperature += temp_repetition_sensitivity * repeat_score
                        temperature = max(temp_min_clip, min(temp_max_clip, temperature))
                    else:
                        temperature = temp_base
                    
                    # Min-p（修复版：内部已处理temperature）
                    if min_p > 0.0:
                        next_logits = _min_p_sampling(next_logits, min_p, temperature)
                    
                    # 【新增】Top-p (Nucleus) 采样
                    if top_p < 1.0 and top_p > 0.0:
                        next_logits = _top_p_sampling(next_logits, top_p, temperature)
                    
                    # 【新增】top-k过滤（强制回答阶段同样需要）
                    if top_k > 0:
                        vals, indices = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                        next_logits = torch.full_like(next_logits, float("-inf"))
                        next_logits[indices] = vals
                    
                    probs = torch.softmax(next_logits / temperature, dim=-1)
                    
                    # 【修复】防止所有logits为-inf导致softmax产生nan
                    if torch.isnan(probs).any() or probs.sum().item() < 1e-6:
                        print(f"\n[Warning] 强制回答阶段概率分布异常，使用随机采样", flush=True)
                        probs = torch.ones_like(probs) / probs.size(0)
                    
                    index = int(torch.multinomial(probs, 1).item())
                    
                    # 【BUG #2修复】END_TOKEN现在可以被正常采样选中
                    if index == TextTokenizer.END_GENERATION_TOKEN:
                        break
                    
                    decoded_piece = TextTokenizer.decode(torch.tensor([index]))
                    if decoded_piece:
                        print(f"{GREEN}{decoded_piece}{RESET}", end="", flush=True)
                        output_text += decoded_piece
                    
                    if index not in (TextTokenizer.THINK_START_TOKEN, TextTokenizer.THINK_END_TOKEN,
                                     TextTokenizer.START_GENERATION_TOKEN, TextTokenizer.END_GENERATION_TOKEN):
                        generated_tokens[index] += 1
                        _token_history.append(index)  # 【修复】记录到滑动窗口历史
                    
                    next_token = torch.tensor([index], device=device)
                    result = model(next_token, past_key_values=past_key_values, use_cache=True)
                    if isinstance(result, tuple):
                        logits, past_key_values = result
                    else:
                        logits = result
                        past_key_values = None
                    
                    step += 1
                    force_answer_count += 1
                except Exception:
                    break

        # 生成完成后清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return output_text


def train(ask: str = None, think: str = None, answer: str = None, history_context: str = None) -> None:
    """单步训练函数
    
    Args:
        ask: 问题文本
        think: 思维链/推理过程（可选，用于CoT训练）
        answer: 答案文本
        history_context: 历史对话上下文
    """
    model.train()
    t0 = time.time()
    
    def _sanitize(text):
        if text is None:
            return None
        text = str(text).strip()
        # 过滤掉表示 NaN 的字符串
        if text.lower() in ('nan', 'inf', '-inf', 'none', 'null'):
            return None
        return text
    
    ask = _sanitize(ask)
    think = _sanitize(think)
    answer = _sanitize(answer)
    history_context = _sanitize(history_context)
    
    # ANSI颜色代码
    WHITE = '\033[97m'     # 问题 - 白色
    BLUE = '\033[94m'      # 思考 - 蓝色
    GREEN = '\033[92m'     # 回答 - 绿色
    YELLOW = '\033[93m'    # 单文本 - 黄色
    RESET = '\033[0m'      # 重置颜色
    
    # 单文本训练模式
    if ask is None and answer is None:
        return
    
    if ask is None:
        print(f"\n---Train{RESET}", flush=True)

        text_tensor = TextTokenizer.encode(answer)
        if text_tensor.numel() < 2:
            return

        train_tensor, target_mask = _build_train_sequence([
            (TextTokenizer.START_GENERATION_TOKEN, True),
            (text_tensor, True),
            (TextTokenizer.END_GENERATION_TOKEN, True),
        ])
        preview = train_tensor
        _run_train_step(train_tensor, target_mask, preview, show_preview=True, preview_color=YELLOW)
        return

    # QA训练模式
    print(f"\n---Train{RESET}", flush=True)
    print(f"{WHITE}{ask}{RESET}", flush=True)
    
    if answer and answer.strip():
        if think and think.strip():
            print(f"{BLUE}{think}{RESET}", flush=True)
            print(f"{GREEN}{answer}{RESET}", flush=True)
            
            ask_tensor = TextTokenizer.encode(ask)
            think_tensor = TextTokenizer.encode(think)
            answer_tensor = TextTokenizer.encode(answer)
            
            if history_context:
                history_tensor = TextTokenizer.encode(history_context)
                # CoT 训练序列 Loss Mask 设计说明：
                # target_mask[i]=True → logits[i-1] 预测 token[i] 时计算loss
                # 关键：THINK_END_TOKEN 的 mask=True 确保其前面位置的logit学习输出THINK_END
                #       answer 首token 的 mask=True 确保 THINK_END 位置的logit学习输出回答首字符
                #       这保证了"思维链→回答"的过渡被正确训练，防止模型在THINK_END后直接结束
                train_tensor, target_mask = _build_train_sequence([
                    (TextTokenizer.START_GENERATION_TOKEN, False),
                    (history_tensor, False),
                    (TextTokenizer.END_GENERATION_TOKEN, False),
                    (TextTokenizer.HISTORY_CONTEXT_START_TOKEN, False),
                    (ask_tensor, False),
                    (TextTokenizer.START_GENERATION_TOKEN, True), # 【修复】学习在ask后输出START
                    (TextTokenizer.THINK_START_TOKEN, True),     # 学习在START后输出THINK_START
                    (think_tensor, True),                        # 学习生成思维链内容
                    (TextTokenizer.THINK_END_TOKEN, True),       # 学习在思考结束时输出THINK_END
                    (answer_tensor, True),                       # ★ 学习从THINK_END过渡到回答
                    (TextTokenizer.END_GENERATION_TOKEN, True),  # 学习在回答结束时输出END
                ])
                # preview 保持在 CPU，避免不必要的 GPU 移动
                preview = torch.cat([think_tensor, answer_tensor])
                _run_train_step(train_tensor, target_mask, preview, show_preview=False)
            else:
                # CoT 训练序列 Loss Mask（无历史上下文版本）
                # 同上：THINK_END→answer 的过渡是关键，两者 mask 皆为 True
                train_tensor, target_mask = _build_train_sequence([
                    (ask_tensor, False),                         # 问题不参与loss
                    (TextTokenizer.START_GENERATION_TOKEN, True), # 【修复】学习在ask后输出START
                    (TextTokenizer.THINK_START_TOKEN, True),     # 学习在START后输出THINK_START
                    (think_tensor, True),                        # 学习生成思维链内容
                    (TextTokenizer.THINK_END_TOKEN, True),       # 学习在思考结束时输出THINK_END
                    (answer_tensor, True),                       # ★ 学习从THINK_END过渡到回答
                    (TextTokenizer.END_GENERATION_TOKEN, True),  # 学习在回答结束时输出END
                ])
                preview = torch.cat([think_tensor, answer_tensor])
                _run_train_step(train_tensor, target_mask, preview, show_preview=False)
            return
        
        print(f"{GREEN}{answer}{RESET}", flush=True)
        train_tensor, target_mask, preview = _prepare_training_data(ask, answer, history_context)
        if train_tensor is None:
            return
        _run_train_step(train_tensor, target_mask, preview, show_preview=False)
    
    # 自奖励评估 —— 智能 RL 切换（基于 SuperRL Adaptive Switch 设计）
    try:
        # 计算奖励（自动记录到历史）
        total_reward, reward_breakdown = reward_model.compute_total_reward(
            think_text=think,
            answer_text=answer,
            context=history_context
        )

        # 智能决策是否启用 RL 训练（仅由奖励质量决定，不再受训练轮数限制）
        should_enable_rl, rl_decision_reason = reward_model.should_enable_rl()

        if should_enable_rl:
            # 启用 RL 训练：收集 episode 并更新策略
            ppo_trainer.collect_episode(
                prompt=ask if ask else "",
                think_text=think if think else "",
                answer_text=answer if answer else "",
                context=history_context
            )
            # 【Gemini修复】收集足够episode后才更新策略，减少优势函数方差
            # 条件：1) 训练轮数>0  2) 到达检查间隔  3) episode数>=最小阈值
            if (training_rounds > 0 and 
                (training_rounds % RL_UPDATE_INTERVAL) == 0 and
                len(ppo_trainer.episode_data['rewards']) >= RL_MIN_EPISODES):
                ppo_update_result = ppo_trainer.update_policy(batch_size=RL_UPDATE_BATCH_SIZE)

                # 每100步打印一次 RL 状态
                if training_rounds % 100 == 0:
                    print(f"[RL Smart Switch] ✅ 启用RL训练 | "
                          f"奖励={total_reward:.3f} | "
                          f"原因: {rl_decision_reason}", flush=True)
        else:
            # 暂停 RL 训练：仅执行 SFT，不收集 RL episode
            # 每50步打印一次切换状态
            if training_rounds % 50 == 0:
                print(f"[RL Smart Switch] ⏸️ 暂停RL训练 | "
                      f"奖励={total_reward:.3f} | "
                      f"原因: {rl_decision_reason}", flush=True)
    except Exception as e:
        print(f"[Warning] RL step failed (non-fatal): {e}", flush=True)


def _detach_kv_cache(past_kv, max_cache_len: int = None):
    """递归 detach KV Cache 中的所有张量，切断计算图，并严格限制缓存长度。

    past_kv 是 list[CompressedKVCache]，其中 CompressedKVCache 是
    tuple[recent_k, recent_v, mem_k, mem_v, mem_pos, total_len, mla_M, mla_z]。
    必须 detach 后才能跨 chunk 复用，
    否则 backward 后中间值被释放会导致 "backward a second time" 错误。
    
    【修复】max_cache_len 从 2048 降到 1024，更严格限制显存占用。
    同时限制 mem_k/mem_v 的长度，防止压缩记忆无限增长。
    """
    if past_kv is None:
        return None
    
    if max_cache_len is None:
        max_cache_len = int(CONFIG.get("kv_cache_max_len", 1024))
    
    detached = []
    max_mem_len = max(128, max_cache_len // 4)  # 压缩记忆更严格限制
    for cache_tuple in past_kv:
        detached_tuple = []
        for i, t in enumerate(cache_tuple):
            if isinstance(t, torch.Tensor):
                dt = t.detach()
                # recent_k/v (索引0,1) 限制为 max_cache_len
                # mem_k/v (索引2,3) 限制为 max_mem_len
                limit = max_cache_len if i < 2 else max_mem_len
                if dt.dim() >= 2 and dt.shape[-2] > limit:
                    dt = dt[..., -limit:, :].contiguous()
                detached_tuple.append(dt)
            else:
                detached_tuple.append(t)
        detached.append(tuple(detached_tuple))
    return detached


def _estimate_safe_chunk_size(seq_len: int = 0) -> int:
    """全动态分块大小计算
    
    基于：
    1. GPU总显存和已用显存
    2. CPU空闲内存
    3. 序列长度
    4. 系统负载
    
    没有任何固定值，完全运行时计算。
    """
    emb_size = int(CONFIG.get("emb_size", 512))
    num_layers = int(CONFIG.get("num_transformer_blocks", 8))
    # 【修复】修正bytes_per_token计算：
    # 旧版：emb_size * num_layers * 8 = 512*8*8 = 32,768 bytes/token（严重高估）
    # 新版：emb_size * num_layers * 2 = 512*8*2 = 8,192 bytes/token（bf16激活值更合理）
    # 实际每个token的激活值 = hidden_size * num_layers * dtype_size
    bytes_per_token = emb_size * num_layers * 2
    
    # 获取硬件状态
    gpu_free_ratio = _get_gpu_free_memory_ratio()
    cpu_free_ratio = _get_cpu_free_memory_ratio()
    
    # 获取配置系数
    chunk_memory_ratio = float(CONFIG.get("chunk_memory_ratio", 0.15))
    chunk_seq_len_factor = float(CONFIG.get("chunk_seq_len_factor", 0.3))
    chunk_min_absolute = int(CONFIG.get("chunk_min_absolute", 128))
    chunk_max_ratio = float(CONFIG.get("chunk_max_ratio", 0.5))
    chunk_cpu_pressure_factor = float(CONFIG.get("chunk_cpu_pressure_factor", 0.2))
    
    # 【修复】计算可用显存时扣除模型固定开销
    # 模型固定开销 = 参数 + 梯度 + 优化器状态(Adam = 2x参数)
    # 估算：参数量 * 4 (fp32) * 3 (参数+梯度+优化器)
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        # 【修复】使用正确的配置键名 "dict_size" 而非 "vocab_size"
        vocab_size = int(CONFIG.get("dict_size", 60000))
        # 【修复】更准确的参数估算：
        # embedding: vocab_size * emb_size
        # 每个transformer block: 
        #   - qkv_proj: emb_size * emb_size * 3
        #   - out_proj: emb_size * emb_size
        #   - router: emb_size * (emb_size//4) + (emb_size//4) * 3
        #   - SwiGLU FFN: gate(emb*hidden) + up(emb*hidden) + down(hidden*emb), hidden=3*emb
        #   - RMSNorm: 2 * emb_size (attn_norm + ffn_norm)
        #   - MLA memory: kv_proj(head*latent) + v_proj(head*latent) + q_proj(head*latent) + v_up_proj(latent*head) + out_proj(head*head)
        # output: emb_size * vocab_size (与embedding共享时忽略)
        hidden = emb_size * 3  # SwiGLU hidden size
        attn_params = emb_size * emb_size * 3 + emb_size * emb_size  # qkv + out
        ffn_params = emb_size * hidden * 2 + hidden * emb_size  # gate + up + down
        router_params = emb_size * max(1, emb_size // 4) + max(1, emb_size // 4) * 3
        norm_params = emb_size * 2
        # MLA 参数 (粗略)
        head_dim = emb_size // 8  # num_heads=8
        latent_dim = max(16, head_dim // 4)
        mla_params = head_dim * latent_dim * 3 + latent_dim * head_dim + head_dim * head_dim
        block_params = attn_params + ffn_params + router_params + norm_params + mla_params
        
        param_count = vocab_size * emb_size + block_params * num_layers
        if not bool(CONFIG.get("tie_token_embeddings", True)):
            param_count += emb_size * vocab_size  # output layer
        
        model_overhead = param_count * 4 * 3  # fp32 * 3 (param + grad + optimizer)
        
        # 可用显存 = 空闲显存 - 模型固定开销 - 安全余量(15%)
        free_memory = max(0, total_memory * gpu_free_ratio - model_overhead)
        free_memory = min(free_memory, total_memory * gpu_free_ratio * 0.85)  # 保留15%安全余量
    else:
        free_memory = 8 * 1024**3  # 默认8GB
    
    # 显存因子（显存紧张时更保守）
    memory_factor = gpu_free_ratio * (1.0 - (1.0 - cpu_free_ratio) * chunk_cpu_pressure_factor)
    
    # 计算chunk大小
    # 1. 基于显存（使用更保守的比例）
    # 【修复】chunk_memory_ratio从0.15降到0.08，更保守
    conservative_ratio = chunk_memory_ratio * 0.5  # 使用一半的比例
    mem_based_chunk = int(free_memory * conservative_ratio * memory_factor / bytes_per_token)
    
    # 2. 基于序列长度
    seq_based_chunk = int(seq_len * chunk_seq_len_factor)
    
    # 取两者较小值，但不超过seq_len的max_ratio
    chunk_size = min(mem_based_chunk, seq_based_chunk)
    chunk_size = min(chunk_size, int(seq_len * chunk_max_ratio))
    
    # 绝对边界
    chunk_size = max(chunk_min_absolute, chunk_size)
    
    # 如果seq_len很短，尝试完整序列
    if seq_len > 0 and seq_len < chunk_min_absolute * 2:
        chunk_size = max(chunk_size, seq_len)
    
    return chunk_size


def _chunked_forward_backward(
    train_tensor: torch.Tensor,
    target_mask: torch.Tensor,
    chunk_size: int,
    overlap: int = 64,
    grad_scale: int = 1,
) -> float | None:
    """MLA 压缩上下文分段训练：每个 chunk 通过 CSA+H2O+MLA 三路压缩上下文感知全量历史，
    梯度跨块累积，零截断。返回平均 loss，None 表示全部 OOM 跳过。

    原理：每次 model(seg, past_key_values=cache, use_cache=True) 返回的 cache
    包含 HyperAttention._build_cache() 构建的三层压缩记忆：
      - CSA (Compressed Sparse Attention): 压缩稀疏注意力KV
      - H2O (Heavy Hitter Oracle): 高注意力分数token保留
      - MLA (Multi-head Latent Attention): 低秩潜在注意力记忆
    下一 chunk 的注意力计算自动使用此三路混合，实现长上下文感知。

    Args:
        grad_scale: 梯度累积步数，每个 chunk 的 loss 会除以该值以保持梯度量级一致
    """
    seq_len = train_tensor.numel()
    step = max(1, chunk_size - overlap)

    chunk_losses = []
    past_kv = None

    for seg_start in range(0, seq_len, step):
        seg_end = min(seg_start + chunk_size, seq_len)
        seg = train_tensor[seg_start:seg_end].to(device)
        seg_mask = target_mask[seg_start:seg_end]

        try:
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                result = model(seg.unsqueeze(0) if seg.dim() == 1 else seg,
                               past_key_values=past_kv, use_cache=True)
                if isinstance(result, tuple):
                    logits, past_kv = result
                else:
                    logits = result
                    past_kv = None

                if seg.numel() > 1 and seg_mask.any():
                    if logits.dim() == 3:
                        logits_2d = logits.squeeze(0)
                    else:
                        logits_2d = logits

                    if past_kv is not None and seg_start > 0:
                        current_logits = logits_2d[-seg.numel():]
                        overlap_offset = min(overlap, seg.numel() - 1)
                        non_overlap_mask = seg_mask[overlap_offset + 1:].to(device)
                        if non_overlap_mask.any():
                            pred = current_logits[overlap_offset:-1][non_overlap_mask]
                            tgt = seg[overlap_offset + 1:].to(device)[non_overlap_mask]
                            loss_chunk = loss_func(pred, tgt)
                        else:
                            loss_chunk = torch.tensor(0.0, device=device)
                    else:
                        mask_bool = seg_mask[1:].to(device)
                        if mask_bool.any():
                            pred = logits_2d[:-1][mask_bool]
                            tgt = seg[1:].to(device)[mask_bool]
                            loss_chunk = loss_func(pred, tgt)
                        else:
                            loss_chunk = torch.tensor(0.0, device=device)
                else:
                    loss_chunk = torch.tensor(0.0, device=device)

                # 【修复】统一缩放：每个 chunk 的 loss 除以 grad_scale
                # 保持与标准训练路径的梯度量级一致
                loss_scaled = loss_chunk / grad_scale

            if loss_scaled.requires_grad:
                if scaler.is_enabled():
                    scaler.scale(loss_scaled).backward()
                else:
                    loss_scaled.backward()

            chunk_losses.append(loss_chunk.detach())
            del seg, logits, loss_chunk, loss_scaled

            # 【修复】强制清理中间变量，防止显存泄漏
            if past_kv is not None:
                past_kv = _detach_kv_cache(past_kv)
                # 显存紧张时主动清理缓存
                if torch.cuda.is_available() and seg_start % 4 == 0:
                    torch.cuda.empty_cache()

        except RuntimeError as e_oom:
            if "out of memory" in str(e_oom).lower():
                smaller = max(128, chunk_size // 2)
                if smaller < chunk_size:
                    print(f"[Memory] Chunk OOM at [{seg_start}:{seg_end}], 缩半到{smaller}重试", flush=True)
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    safe_past = _detach_kv_cache(past_kv) if past_kv is not None else None
                    sub = _chunk_one_segment(seg, seg_mask, safe_past, smaller, grad_scale=grad_scale)
                    if sub is not None:
                        sub_loss, past_kv = sub
                        chunk_losses.append(sub_loss.detach())
                        del sub_loss
                    else:
                        continue
                else:
                    print(f"[Memory] 最小chunk仍OOM，跳过此段", flush=True)
                    continue
            else:
                raise

    if not chunk_losses:
        return None
    avg_loss = torch.stack([l.to(device) for l in chunk_losses]).mean()
    return avg_loss.item()


def _chunk_one_segment(
    seg: torch.Tensor, seg_mask: torch.Tensor, past_kv, chunk_size: int,
    grad_scale: int = 1,
) -> tuple[torch.Tensor, any] | None:
    """OOM 回退：将单个 segment 进一步细分后训练，返回 (avg_loss, past_kv) 或 None。

    【修复】统一梯度缩放：每个子 chunk 的 loss 除以 grad_scale，
    与主循环的缩放一致。细分后的多个子 chunk 会自然累加它们的梯度，
    这与长序列提供更多训练信号的直觉相符。
    """
    seg_len = seg.numel()
    step = max(1, chunk_size // 2)
    seg_losses = []
    local_past = past_kv

    for s in range(0, seg_len, step):
        e = min(s + chunk_size, seg_len)
        sub = seg[s:e].to(device)
        sub_mask = seg_mask[s:e]

        try:
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                result = model(sub.unsqueeze(0) if sub.dim() == 1 else sub,
                               past_key_values=local_past, use_cache=True)
                if isinstance(result, tuple):
                    logits, local_past = result
                else:
                    logits = result
                    local_past = None
                if sub.numel() > 1 and sub_mask.any():
                    if logits.dim() == 3:
                        logits_2d = logits.squeeze(0)
                    else:
                        logits_2d = logits

                    if local_past is not None and s > 0:
                        current_logits = logits_2d[-sub.numel():]
                        mask_bool = sub_mask[1:].to(device)
                        if mask_bool.any():
                            pred = current_logits[:-1][mask_bool]
                            tgt = sub[1:].to(device)[mask_bool]
                            loss_sub = loss_func(pred, tgt)
                        else:
                            loss_sub = torch.tensor(0.0, device=device)
                    else:
                        mask_bool = sub_mask[1:].to(device)
                        if mask_bool.any():
                            pred = logits_2d[:-1][mask_bool]
                            tgt = sub[1:].to(device)[mask_bool]
                            loss_sub = loss_func(pred, tgt)
                        else:
                            loss_sub = torch.tensor(0.0, device=device)
                else:
                    loss_sub = torch.tensor(0.0, device=device)

                # 【修复】统一缩放
                loss_scaled = loss_sub / grad_scale

            if loss_scaled.requires_grad:
                if scaler.is_enabled():
                    scaler.scale(loss_scaled).backward()
                else:
                    loss_scaled.backward()
            seg_losses.append(loss_sub.detach())
            del sub, logits, loss_sub, loss_scaled

            if local_past is not None:
                local_past = _detach_kv_cache(local_past)

        except RuntimeError:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            continue

    if seg_losses:
        avg = torch.stack(seg_losses).mean()
        return avg, local_past
    return None


def _run_train_step(train_tensor: torch.Tensor, target_mask: torch.Tensor, preview: torch.Tensor, show_preview: bool = True, preview_color: str = None) -> float:
    """执行单步训练（显存感知自适应分段，零截断）

    核心策略（按优先级）：
    1. 序列能放入显存 → 标准前向+反向（含梯度累积）
    2. 序列过长但可分块 → MLA压缩上下文分段训练
        模型自动在 chunk 间传递 CSA+H2O+MLA 三路压缩上下文，
        每个 chunk 能感知全量历史（非简单截断），梯度跨块累积。
    3. 所有方法都失败 → 才跳过（不做截断！）

    【修复】梯度累积与优化器步进逻辑重构：
    - training_rounds 在函数开头立即递增，确保所有判断条件基于最新值
    - zero_grad 在新累积周期开始时执行 (training_rounds % GAS == 0)
    - optimizer.step 在累积周期结束时执行 (training_rounds % GAS == 0)
    - 分段训练路径统一缩放 loss / GRADIENT_ACCUMULATION_STEPS
    """
    global training_rounds, optimizer_step_count

    t0 = time.time()

    training_rounds += 1

    model.train()
    seq_len = train_tensor.numel()

    # ── 显存状态检查 ──
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        total_mem = float(props.total_memory)
        reserved = float(torch.cuda.memory_reserved(idx))
        allocated = float(torch.cuda.memory_allocated(idx))
        used = max(reserved, allocated)
        free_bytes = max(0.0, total_mem - used)
        mem_ratio = used / total_mem
    else:
        free_bytes = float('inf')
        mem_ratio = 0.0
        total_mem = float('inf')

    # 定期显存监控
    if training_rounds > 0 and training_rounds % 100 == 0 and torch.cuda.is_available():
        print(f"[Memory] Step {training_rounds}: Used={used/1024**3:.2f}/{total_mem/1024**3:.2f}GB "
              f"({mem_ratio*100:.1f}%), Free={free_bytes/1024**3:.2f}GB, SeqLen={seq_len}", flush=True)

        cache_thresh = float(CONFIG.get("gpu_cache_clear_threshold_gb", 4.0))
        if reserved / 1024**3 > cache_thresh:
            torch.cuda.empty_cache()
            print(f"[Memory] Cleared GPU cache", flush=True)

    # 【修复】梯度累积管理：在新累积周期开始时清零梯度
    # 使用 (training_rounds - 1) % GAS == 0 确保第一个样本前清零
    # 旧逻辑 training_rounds % GAS == 0 导致第1个样本不清零，梯度会累积到第4个样本
    if ((training_rounds - 1) % GRADIENT_ACCUMULATION_STEPS) == 0:
        optimizer.zero_grad(set_to_none=True)

    # ── 显存安全网关 ──
    # 【修复】降低skip阈值到0.85，更早触发保护
    # 旧版0.92太晚，此时已经OOM风险很高
    skip_thresh = float(CONFIG.get("gpu_memory_skip_ratio", 0.85))
    if mem_ratio >= skip_thresh:
        print(f"[Memory] 显存占用过高 ({mem_ratio:.1%}), 主动清理后跳过本样本", flush=True)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return float('inf')
    
    # ── 策略选择：全动态计算分块大小 ──
    # 【全动态分块】基于实时硬件状态计算
    safe_chunk = _estimate_safe_chunk_size(seq_len)
    
    # 【新增】显存预检：如果预计训练后会超过阈值，提前使用更小的chunk
    # 估算每个token的显存占用（使用保守估计）
    est_bytes_per_token = int(CONFIG.get("emb_size", 512)) * int(CONFIG.get("num_transformer_blocks", 8)) * 2
    estimated_usage = mem_ratio + (seq_len * est_bytes_per_token / total_mem if torch.cuda.is_available() else 0)
    if estimated_usage > skip_thresh and seq_len > 256:
        print(f"[Memory] 预计显存不足 (当前{mem_ratio:.1%}, 估计{estimated_usage:.1%}), 强制使用小chunk训练", flush=True)
        safe_chunk = min(safe_chunk, 256)  # 强制小chunk
    
    # 【全动态overlap】基于chunk_size和序列长度动态计算
    chunk_overlap_base = int(CONFIG.get("chunk_overlap_base", 32))
    chunk_overlap_scale = float(CONFIG.get("chunk_overlap_scale", 0.02))
    overlap = int(chunk_overlap_base + safe_chunk * chunk_overlap_scale)
    overlap = min(overlap, safe_chunk // 4)  # overlap不超过chunk的1/4

    try:
        if seq_len <= safe_chunk:
            # ✅ 策略 1: 标准训练（序列完整放入 GPU）
            train_tensor_gpu = train_tensor.to(device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                result = model(train_tensor_gpu, use_cache=False)
                if isinstance(result, tuple):
                    logits = result[0]
                else:
                    logits = result

                if seq_len > 1:
                    mask_bool = target_mask[1:].to(device)
                    if mask_bool.any():
                        if logits.dim() == 3:
                            logits_2d = logits.squeeze(0)
                        else:
                            logits_2d = logits
                        pred = logits_2d[:-1][mask_bool]
                        tgt = train_tensor_gpu[1:][mask_bool]
                        loss = loss_func(pred, tgt)
                    else:
                        loss = torch.tensor(0.0, device=device)
                else:
                    loss = torch.tensor(0.0, device=device)

            raw_loss_val = loss.item()
            # 统一缩放：所有路径的 loss 都除以 GRADIENT_ACCUMULATION_STEPS
            loss = loss / GRADIENT_ACCUMULATION_STEPS

            if loss.requires_grad:
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

        else:
            # ✅ 策略 2: MLA压缩上下文分段训练（CSA+H2O+MLA三路压缩）
            chunk_size = safe_chunk

            print(f"[Memory] MLA压缩上下文训练: seq={seq_len}, chunk={chunk_size}, overlap={overlap}, "
                  f"free={free_bytes/1024**3:.2f}GB (CSA+H2O+MLA三路压缩)", flush=True)

            # 【修复】传入梯度累积缩放因子，确保分段路径与标准路径梯度量级一致
            loss_val = _chunked_forward_backward(
                train_tensor, target_mask, chunk_size, overlap,
                grad_scale=GRADIENT_ACCUMULATION_STEPS
            )
            if loss_val is None:
                print(f"[Memory] 所有分段均 OOM，跳过本样本", flush=True)
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                return float('inf')
            raw_loss_val = loss_val
            loss = torch.tensor(loss_val, device=device)

        # ── 统一的梯度后处理 ──
        if not torch.isnan(loss) and not torch.isinf(loss):
            # 【修复】step 条件与 zero_grad 条件一致：都在周期边界执行
            # training_rounds 已递增，当 % GAS == 0 表示累积周期完成
            # 注意：这里保持 training_rounds % GAS == 0，因为 step 应该在第4个样本后执行
            should_step = (training_rounds % GRADIENT_ACCUMULATION_STEPS) == 0

            if scaler.is_enabled():
                if should_step:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        optimizer.zero_grad(set_to_none=True)
                        scaler.update()
                        for param in model.parameters():
                            if param.grad is not None and torch.isnan(param.grad).any():
                                param.grad = None
                        print(f"[Warning] NaN/Inf gradient, skipping optimizer step", flush=True)
                        return float('inf')
                    scaler.step(optimizer)
                    scaler.update()
            else:
                if should_step:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        optimizer.zero_grad(set_to_none=True)
                        for param in model.parameters():
                            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                                param.grad = None
                        print(f"[Warning] NaN/Inf gradient, skipping optimizer step", flush=True)
                        return float('inf')
                    optimizer.step()
        else:
            print(f"[Warning] Invalid loss: {loss}, skipping optimizer step", flush=True)
            return float('inf')

        # 学习率调度 — SGDR + ReduceLROnPlateau（动态 LR，适配无限训练）
        # 【修复】step 条件统一为 % GAS == 0
        if should_step:
            optimizer_step_count += 1
            # 【Gemini修复】每N个optimizer step才更新一次学习率，防止SGDR震荡过于频繁
            if optimizer_step_count % LR_SCHEDULER_STEP_INTERVAL == 0:
                current_lr = lr_scheduler.step(loss=raw_loss_val)

        # Preview 输出
        if show_preview:
            try:
                decoded = TextTokenizer.decode(preview[preview != 0])
                RESET = '\033[0m'
                if preview_color:
                    print(f"{preview_color}{decoded}{RESET}", end="", flush=True)
                else:
                    print(decoded, end="", flush=True)
            except Exception as e:
                print(f"[Warning] Preview decode failed: {e}", flush=True)
            print("", flush=True)

        record_loss(raw_loss_val)

        return loss.item()

    except RuntimeError as e:
        error_msg = str(e)
        if "CUDA" in error_msg or "cuda" in error_msg.lower() or "out of memory" in error_msg.lower():
            print(f"[CUDA Error] {e}", flush=True)
            optimizer.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            return float('inf')
        else:
            print(f"[RuntimeError] {e}, skipping", flush=True)
            return float('inf')
    except Exception as e:
        print(f"[Error] 未知错误: {e}, skipping", flush=True)
        return float('inf')
    finally:
        try:
            elapsed = time.time() - t0
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 1024**3
                resv = torch.cuda.memory_reserved() / 1024**3
                print(f"[Profile] Step {training_rounds}: {elapsed:.3f}s, "
                      f"Alloc={alloc:.2f}GB, Resv={resv:.2f}GB", flush=True)
        except Exception:
            pass