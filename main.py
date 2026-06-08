from typing import List, Tuple, Optional
import sys
import os
import torch
import time
import logging
from collections import Counter
from config import CONFIG
from model import MainModel
from record import record_loss, evaluate_rl_readiness
from tokenizer import TextTokenizer
from rl import SelfRewardModel, LightweightPPO


# 【显存优化】设置 PyTorch CUDA 内存分配策略，避免显存碎片化
# expandable_segments:True 允许内存段动态扩展，减少碎片
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

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
        state_dict = torch.load(pretrained_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"Loaded pretrained model from {pretrained_path}")
    except Exception as e:
        print(f"Warning: Failed to load pretrained model: {e}")

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
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

print(f"Using device: {device}")
print(f"AMP enabled: {use_amp} AMP dtype: {amp_dtype}")

loss_func = torch.nn.CrossEntropyLoss()

# 梯度累积步数
GRADIENT_ACCUMULATION_STEPS = int(CONFIG.get("gradient_accumulation_steps", 4))

# 训练轮数计数器
training_rounds = 0
optimizer_step_count = 0

# 自奖励模型和PPO训练器
reward_model = SelfRewardModel(device)
ppo_trainer = LightweightPPO(model, reward_model, device, external_optimizer=optimizer)


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
            (TextTokenizer.START_GENERATION_TOKEN, False),
            (answer_tensor, True),
            (TextTokenizer.END_GENERATION_TOKEN, True),
        ])
    else:
        train_tensor, target_mask = _build_train_sequence([
            (ask_tensor, False),
            (TextTokenizer.START_GENERATION_TOKEN, False),
            (answer_tensor, True),
            (TextTokenizer.END_GENERATION_TOKEN, True),
        ])
    
    preview = answer_tensor
    return train_tensor, target_mask, preview


def _is_garbage_token(token_id: int) -> bool:
    """检测 token 解码后是否为垃圾/不可显示字符

    垃圾类型包括:
      - 控制字符 (0-31, 127) —— 但排除特殊功能token (1-9)
      - 代理对 (0xD800-0xDFFF)
      - 私用区 (0xE000-0xF8FF)
      - 非主流 Unicode 块 (如加拿大音节文字 0x1400-0x167F)
      - 特殊 Unicode 控制字符
    """
    # 特殊功能token (1-9) 不是垃圾，是模型需要学习输出的控制token
    if 1 <= token_id <= 9:
        return False
    if not TextTokenizer._is_valid_token(token_id):
        return True
    # 控制字符（排除特殊token 1-9）
    if token_id < 32 or token_id == 127:
        return True
    # 加拿大土著音节文字 (ᓀ ᓓ 等 — 训练数据中不应出现)
    if 0x1400 <= token_id <= 0x167F:
        return True
    # 切罗基文字
    if 0x13A0 <= token_id <= 0x13FF:
        return True
    # 高棉文符号
    if 0x19E0 <= token_id <= 0x19FF:
        return True
    # 缅甸文扩展
    if 0xAA60 <= token_id <= 0xAA7F:
        return True
    # 彝文音节
    if 0xA000 <= token_id <= 0xA4CF:
        return True
    # 私用区
    if 0xE000 <= token_id <= 0xF8FF:
        return True
    # 标记/组合字符块 
    if 0x0300 <= token_id <= 0x036F:
        return True
    if 0x1AB0 <= token_id <= 0x1AFF:
        return True
    if 0xFE00 <= token_id <= 0xFE0F:
        return True
    return False


# 预计算垃圾token掩码（避免每步Python循环）
_GARBAGE_MASK = None
_GARBAGE_FORCE_MASK = None  # 强制回答阶段的非CJK字符降权掩码

def _init_quality_masks(vocab_size: int):
    """初始化矢量化质量过滤掩码（仅调用一次）"""
    global _GARBAGE_MASK, _GARBAGE_FORCE_MASK
    if _GARBAGE_MASK is not None:
        return
    
    import torch
    _GARBAGE_MASK = torch.zeros(vocab_size, dtype=torch.bool)
    _GARBAGE_FORCE_MASK = torch.zeros(vocab_size, dtype=torch.bool)
    
    limit = min(vocab_size, 0x2000)
    for token_id in range(limit):
        if _is_garbage_token(token_id):
            _GARBAGE_MASK[token_id] = True
    
    # 强制回答阶段：标记非语言字符（用于降权，不封死）
    for token_id in range(0x80, min(vocab_size, 0x10000)):
        if _GARBAGE_MASK[token_id]:
            continue
        # ASCII 可打印、CJK、全角标点 → 允许
        if 32 <= token_id < 127:
            continue
        if 0x4E00 <= token_id <= 0x9FFF:
            continue
        if 0x3400 <= token_id <= 0x4DBF:
            continue
        if 0xFF00 <= token_id <= 0xFFEF:
            continue
        if 0x3000 <= token_id <= 0x303F:
            continue
        _GARBAGE_FORCE_MASK[token_id] = True  # 非语言字符，需要降权


def _apply_token_quality_filter(
    logits: torch.Tensor,
    force_answer: bool = False,
) -> torch.Tensor:
    """对 logits 应用 token 质量过滤（矢量化，O(1)每步）

    将已知垃圾 token 的概率设为 -inf。
    在强制回答阶段，非 CJK/ASCII 字符降权 5.0。
    """
    global _GARBAGE_MASK, _GARBAGE_FORCE_MASK
    vocab_size = logits.size(-1)
    _init_quality_masks(vocab_size)
    
    # 垃圾token → -inf（矢量化，单条指令）
    mask_slice = _GARBAGE_MASK[:vocab_size]
    logits[mask_slice] = float("-inf")
    
    # 强制回答阶段：非语言字符降权
    if force_answer:
        force_slice = _GARBAGE_FORCE_MASK[:vocab_size]
        logits[force_slice] -= 5.0
    
    return logits


def _check_repetition_stop(generated_tokens: list, threshold: int = 5) -> tuple[bool, str]:
    """重复循环检测停止（业界标准方案）
    
    检测生成中的重复模式，防止模型陷入无限循环或"复读机"状态。
    这是比困惑度早停更有效、更稳定的质量控制策略。
    
    Args:
        generated_tokens: 已生成的token列表
        threshold: 连续相同token的阈值
    
    Returns:
        (should_stop: bool, detected_pattern: str)
    """
    if len(generated_tokens) < threshold:
        return False, ""
    
    # 策略1: 检测连续相同token
    recent_tokens = generated_tokens[-threshold:]
    if all(t == recent_tokens[0] for t in recent_tokens):
        return True, f"连续{threshold}个相同token"
    
    # 策略2: 检测重复的n-gram序列（2-gram, 3-gram）
    for n in [2, 3]:
        if len(generated_tokens) < n * 3:
            continue
        
        # 取最后3个n-gram
        ngrams = []
        for i in range(len(generated_tokens) - n + 1):
            ngram = tuple(generated_tokens[i:i+n])
            ngrams.append(ngram)
        
        # 检查最后3个n-gram是否重复
        if len(ngrams) >= 3:
            last_ngrams = ngrams[-3:]
            if len(set(last_ngrams)) == 1:
                return True, f"重复{n}-gram模式"
    
    return False, ""


def _min_p_sampling(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    """Min-p 采样 (Nguyen et al., ICLR 2025)

    核心思想: 以最大概率 token 为锚点，只保留概率 ≥ p_max × min_p 的 token。
    比 top-k + top-p 更优：
      - 高置信度时（p_max 大）→ 自动收紧候选集，避免噪声
      - 低置信度时（p_max 小）→ 自动放宽，保留多样性
    无需手动调节 k 或 p，一个参数适配所有场景。
    """
    if min_p <= 0.0:
        return logits
    probs = torch.softmax(logits, dim=-1)
    p_max = probs.max().item()
    threshold = p_max * min_p
    # 将低于阈值的 token 设为 -inf
    logits = torch.where(probs < threshold, torch.full_like(logits, float("-inf")), logits)
    return logits


def generation(text: str, history_context: str = None, max_generate_tokens: int|None = None, thinking_available: bool = True) -> str:
    """生成函数 (Min-p采样 + Token质量过滤 + CoT完整性保护)

    基于 ICLR 2025 Min-p 论文和大量实验优化的解码策略：
    - Min-p 采样替代 top-k/top-p，天然过滤低概率垃圾token
    - Token 质量过滤器阻止非语言字符（加拿大音节、私用区等）
    - 思考阶段长度限制 + 强制回答期确保 CoT→回复 完整过渡
    - 垃圾token连胜检测：连续N个垃圾立即停止
    """
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    
    if not text or not isinstance(text, str):
        return "无效输入"
    
    # 【修复】None 表示无限制生成，由模型自己决定何时结束（通过 END_TOKEN）
    # 设置一个绝对上限防止极端情况下的死循环（如模型始终不输出 END_TOKEN）
    absolute_max_tokens = 4096  # 绝对安全上限
    has_token_limit = max_generate_tokens is not None
    if max_generate_tokens is None:
        max_generate_tokens = absolute_max_tokens
    
    model.eval()
    output_text = ""

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

    max_generate_tokens = max(1, int(max_generate_tokens))
    
    # 读取采样参数
    temperature = float(CONFIG.get("temperature", 0.5))
    min_p = float(CONFIG.get("min_p", 0.05))
    repetition_penalty = float(CONFIG.get("repetition_penalty", 1.02))
    repetition_stop_threshold = int(CONFIG.get("repetition_stop_threshold", 8))
    force_answer_min_steps = int(CONFIG.get("force_answer_min_steps", 16))
    max_consecutive_garbage = int(CONFIG.get("max_consecutive_garbage", 3))
    # 【新增】最大重采样次数，防止死循环
    max_resample_attempts = int(CONFIG.get("max_resample_attempts", 10))

    with torch.inference_mode():
        thinking_started = False
        force_answer_steps = 0
        consecutive_garbage = 0
        
        # 注入THINK_START
        if thinking_available:
            thinking_started = True
            think_start_tensor = torch.tensor([TextTokenizer.THINK_START_TOKEN], device=device)
            prompt = torch.cat([prompt, think_start_tensor])
        
        result = model(prompt, use_cache=True)
        if isinstance(result, tuple):
            logits, past_key_values = result
        else:
            logits = result
            past_key_values = None

        step = 0
        generated_tokens = Counter()
        generated_sequence = []  # 用于重复检测的完整序列
        
        while step < max_generate_tokens:
            try:
                # 获取最后一个token的logits
                if logits.dim() == 3:
                    next_logits = logits[0, -1].clone()
                else:
                    next_logits = logits[-1].clone()
                
                # Token 质量过滤
                next_logits = _apply_token_quality_filter(
                    next_logits,
                    force_answer=(force_answer_steps > 0),
                )

                # 强制回答阶段：禁止特殊token
                if force_answer_steps > 0:
                    next_logits[TextTokenizer.END_GENERATION_TOKEN] = float("-inf")
                    next_logits[TextTokenizer.UNKNOWN_TOKEN] = float("-inf")
                    next_logits[TextTokenizer.THINK_START_TOKEN] = float("-inf")
                    next_logits[TextTokenizer.THINK_END_TOKEN] = float("-inf")
                    next_logits[TextTokenizer.HISTORY_CONTEXT_START_TOKEN] = float("-inf")
                    next_logits[TextTokenizer.HISTORY_CONTEXT_END_TOKEN] = float("-inf")
                    force_answer_steps -= 1

                # Repetition penalty
                if repetition_penalty > 1.0 and len(generated_tokens) > 0:
                    for token_id, count in generated_tokens.items():
                        if token_id < next_logits.size(0):
                            penalty = repetition_penalty ** min(count, 3)
                            if next_logits[token_id] > 0:
                                next_logits[token_id] /= penalty
                            else:
                                next_logits[token_id] *= penalty

                # Min-p 采样
                if min_p > 0.0:
                    next_logits = _min_p_sampling(next_logits, min_p)

                probs = torch.softmax(next_logits / temperature, dim=-1)
                
                # 【修复】带重采样限制的token选择
                resample_count = 0
                index = int(torch.multinomial(probs, 1).item())
                
                while True:
                    # 检查是否是垃圾token（但保留必要的特殊token）
                    is_garbage = _is_garbage_token(index) and index not in (
                        TextTokenizer.THINK_START_TOKEN,
                        TextTokenizer.THINK_END_TOKEN,
                        TextTokenizer.END_GENERATION_TOKEN,
                        TextTokenizer.START_GENERATION_TOKEN,
                    )
                    
                    # 检查是否是被禁止的token（思考阶段禁止END）
                    is_forbidden = False
                    if index == TextTokenizer.END_GENERATION_TOKEN:
                        if force_answer_steps > 0 or thinking_started:
                            is_forbidden = True
                    
                    if not is_garbage and not is_forbidden:
                        break
                    
                    # 重采样
                    resample_count += 1
                    if resample_count >= max_resample_attempts:
                        print(f"\n[Stop] 重采样次数超限({resample_count})，强制结束", flush=True)
                        break
                    
                    if is_garbage:
                        consecutive_garbage += 1
                        if consecutive_garbage >= max_consecutive_garbage:
                            print(f"\n[Stop] 连续{consecutive_garbage}个垃圾token，强制结束", flush=True)
                            break
                        next_logits[index] = float("-inf")
                    elif is_forbidden:
                        next_logits[index] = float("-inf")
                    
                    # 重新计算概率
                    probs = torch.softmax(next_logits / temperature, dim=-1)
                    # 检查是否所有token都被屏蔽
                    if torch.isinf(next_logits).all() or probs.sum().item() < 1e-6:
                        print(f"\n[Stop] 所有token被屏蔽，强制结束", flush=True)
                        break
                    index = int(torch.multinomial(probs, 1).item())
                
                # 检查是否因重采样超限而退出
                if resample_count >= max_resample_attempts:
                    break
                if consecutive_garbage >= max_consecutive_garbage:
                    break
                
                # 重置垃圾计数（有效token）
                consecutive_garbage = 0
                
                # 处理特殊token
                should_skip_output = False
                
                if index == TextTokenizer.THINK_END_TOKEN:
                    if thinking_available and thinking_started:
                        thinking_started = False
                        force_answer_steps = force_answer_min_steps
                        print(f"\n{GREEN}", end="", flush=True)
                        should_skip_output = True
                    else:
                        break

                elif index == TextTokenizer.END_GENERATION_TOKEN:
                    # 【修复】无限制模式下，END_TOKEN 是正常终止条件
                    # 只有在达到绝对上限时才视为异常
                    if not has_token_limit and step >= absolute_max_tokens - 1:
                        print(f"\n[Stop] 达到绝对安全上限({absolute_max_tokens})，强制结束", flush=True)
                    break

                elif index == TextTokenizer.THINK_START_TOKEN:
                    if thinking_available and not thinking_started:
                        thinking_started = True
                        should_skip_output = True
                    elif not thinking_available:
                        should_skip_output = True

                # 重复检测
                if not thinking_started and index not in (
                    TextTokenizer.THINK_START_TOKEN,
                    TextTokenizer.THINK_END_TOKEN,
                    TextTokenizer.START_GENERATION_TOKEN,
                    TextTokenizer.END_GENERATION_TOKEN,
                ):
                    generated_sequence.append(index)
                    if step >= 5:
                        should_stop, pattern = _check_repetition_stop(generated_sequence, repetition_stop_threshold)
                        if should_stop:
                            print(f"\n[Stop] 检测到重复模式({pattern})，提前结束", flush=True)
                            break

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
                print(f"Error during generation: {e}", flush=True)
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
                    
                    next_logits = _apply_token_quality_filter(next_logits, force_answer=True)
                    next_logits[TextTokenizer.END_GENERATION_TOKEN] = float("-inf")
                    next_logits[TextTokenizer.UNKNOWN_TOKEN] = float("-inf")
                    next_logits[TextTokenizer.THINK_START_TOKEN] = float("-inf")
                    next_logits[TextTokenizer.THINK_END_TOKEN] = float("-inf")
                    next_logits[TextTokenizer.HISTORY_CONTEXT_START_TOKEN] = float("-inf")
                    next_logits[TextTokenizer.HISTORY_CONTEXT_END_TOKEN] = float("-inf")
                    
                    if force_answer_steps > 0:
                        force_answer_steps -= 1
                    
                    # Repetition penalty
                    if repetition_penalty > 1.0 and len(generated_tokens) > 0:
                        for token_id, count in generated_tokens.items():
                            if token_id < next_logits.size(0):
                                penalty = repetition_penalty ** min(count, 3)
                                if next_logits[token_id] > 0:
                                    next_logits[token_id] /= penalty
                                else:
                                    next_logits[token_id] *= penalty
                    
                    # Min-p
                    if min_p > 0.0:
                        next_logits = _min_p_sampling(next_logits, min_p)
                    
                    probs = torch.softmax(next_logits / temperature, dim=-1)
                    
                    # 【修复】强制回答阶段也使用重采样限制
                    resample_count = 0
                    index = int(torch.multinomial(probs, 1).item())
                    
                    while True:
                        is_garbage = _is_garbage_token(index) and index not in (TextTokenizer.END_GENERATION_TOKEN,)
                        if not is_garbage:
                            break
                        
                        resample_count += 1
                        if resample_count >= max_resample_attempts:
                            break
                        
                        consecutive_garbage += 1
                        if consecutive_garbage >= max_consecutive_garbage:
                            break
                        
                        next_logits[index] = float("-inf")
                        probs = torch.softmax(next_logits / temperature, dim=-1)
                        if torch.isinf(next_logits).all() or probs.sum().item() < 1e-6:
                            break
                        index = int(torch.multinomial(probs, 1).item())
                    
                    if resample_count >= max_resample_attempts or consecutive_garbage >= max_consecutive_garbage:
                        break
                    
                    consecutive_garbage = 0
                    
                    if index == TextTokenizer.END_GENERATION_TOKEN:
                        # 【修复】无限制模式下，END_TOKEN 是正常终止条件
                        if not has_token_limit and step >= absolute_max_tokens - 1:
                            print(f"\n[Stop] 强制回答阶段达到绝对安全上限，结束", flush=True)
                        break
                    
                    decoded_piece = TextTokenizer.decode(torch.tensor([index]))
                    if decoded_piece:
                        print(f"{GREEN}{decoded_piece}{RESET}", end="", flush=True)
                        output_text += decoded_piece
                    
                    if index not in (TextTokenizer.THINK_START_TOKEN, TextTokenizer.THINK_END_TOKEN,
                                     TextTokenizer.START_GENERATION_TOKEN, TextTokenizer.END_GENERATION_TOKEN):
                        generated_tokens[index] += 1
                    
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
                    (TextTokenizer.START_GENERATION_TOKEN, False),
                    (TextTokenizer.THINK_START_TOKEN, True),     # 【修复】学习在START后输出THINK_START
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
                    (TextTokenizer.START_GENERATION_TOKEN, False), # 分隔符不参与loss
                    (TextTokenizer.THINK_START_TOKEN, True),     # 【修复】学习在START后输出THINK_START
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
    # 【修复】先检查 SFT 最低训练轮数，未达标则完全跳过 PPO
    rl_min_rounds = int(CONFIG.get("rl_min_training_rounds", 100000))
    if training_rounds < rl_min_rounds:
        # SFT 预热阶段：完全不运行 RL，避免干扰梯度累积
        if training_rounds % 100 == 0:
            print(f"[RL Gate] ⏸️ SFT预热阶段 (round {training_rounds}/{rl_min_rounds})，RL 已禁用", flush=True)
    else:
        try:
            # 计算奖励（自动记录到历史）
            total_reward, reward_breakdown = reward_model.compute_total_reward(
                think_text=think,
                answer_text=answer,
                context=history_context
            )
            
            # 智能决策是否启用 RL 训练
            should_enable_rl, rl_decision_reason = reward_model.should_enable_rl()
            
            if should_enable_rl:
                # 启用 RL 训练：收集 episode 并更新策略
                ppo_trainer.collect_episode(
                    prompt=ask if ask else "",
                    think_text=think if think else "",
                    answer_text=answer if answer else "",
                    context=history_context
                )
                if training_rounds > 0 and (training_rounds % 4) == 0:
                    ppo_update_result = ppo_trainer.update_policy(batch_size=4)
                    
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


def _detach_kv_cache(past_kv):
    """递归 detach KV Cache 中的所有张量，切断计算图。

    past_kv 是 list[CompressedKVCache]，其中 CompressedKVCache 是
    tuple[6个Tensor + 1个int]。必须 detach 后才能跨 chunk 复用，
    否则 backward 后中间值被释放会导致 "backward a second time" 错误。
    """
    if past_kv is None:
        return None
    detached = []
    for cache_tuple in past_kv:
        # cache_tuple: (recent_k, recent_v, mem_k, mem_v, mem_pos, total_len)
        # 前5个是 Tensor，第6个是 int
        detached_tuple = tuple(
            t.detach() if isinstance(t, torch.Tensor) else t
            for t in cache_tuple
        )
        detached.append(detached_tuple)
    return detached


def _estimate_safe_chunk_size(free_bytes: float, safety_factor: float = 0.7) -> int:
    """根据当前空闲显存动态估算安全的分块大小（token数）。"""
    emb_size = int(CONFIG.get("emb_size", 512))
    num_layers = int(CONFIG.get("num_transformer_blocks", 8))
    bytes_per_token = emb_size * num_layers * 8  # 含注意力开销
    safe_bytes = free_bytes * safety_factor
    chunk_size = max(128, int(safe_bytes / bytes_per_token))
    return min(chunk_size, 2048)


def _chunked_forward_backward(
    train_tensor: torch.Tensor,
    target_mask: torch.Tensor,
    chunk_size: int,
    overlap: int = 64,
) -> float | None:
    """KV Cache 分段训练：完整上下文传递，梯度跨块累积，零截断。
    返回平均 loss，None 表示全部 OOM 跳过。
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
                    mask_bool = seg_mask[1:].to(device)
                    if mask_bool.any():
                        pred = logits_2d[:-1][mask_bool]
                        tgt = seg[1:].to(device)[mask_bool]
                        loss_chunk = loss_func(pred, tgt)
                    else:
                        loss_chunk = torch.tensor(0.0, device=device)
                else:
                    loss_chunk = torch.tensor(0.0, device=device)

                num_chunks = max(1, (seq_len + step - 1) // step)
                loss_scaled = loss_chunk / (num_chunks * GRADIENT_ACCUMULATION_STEPS)

            # 【修复】零 loss（无 grad_fn）时跳过 backward，避免崩溃
            if loss_scaled.requires_grad:
                if scaler.is_enabled():
                    scaler.scale(loss_scaled).backward()
                else:
                    loss_scaled.backward()

            chunk_losses.append(loss_chunk.detach())
            del seg, logits, loss_chunk, loss_scaled

            # 【关键】detach past_kv，切断上一chunk的计算图
            # 否则下一个 chunk 的 forward 复用 past_kv 时，
            # backward 会报 "backward a second time" 错误
            if past_kv is not None:
                past_kv = _detach_kv_cache(past_kv)

        except RuntimeError as e_oom:
            if "out of memory" in str(e_oom).lower():
                smaller = max(128, chunk_size // 2)
                if smaller < chunk_size:
                    print(f"[Memory] Chunk OOM at [{seg_start}:{seg_end}], 缩半到{smaller}重试", flush=True)
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    # detach past_kv 后再传入，防止计算图冲突
                    safe_past = _detach_kv_cache(past_kv) if past_kv is not None else None
                    sub = _chunk_one_segment(seg, seg_mask, safe_past, smaller, num_chunks)
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
    num_chunks: int = 1,
):
    """递归重试单个段，返回 (avg_loss, past_kv) 或 None。
    
    【修复】num_chunks 用于正确缩放梯度：
    loss_scaled = loss_sub / (num_chunks * GRADIENT_ACCUMULATION_STEPS)
    确保递归细分后的梯度贡献与正常 chunk 一致，避免梯度被放大 num_chunks 倍。
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
                    mask_bool = sub_mask[1:].to(device)
                    if mask_bool.any():
                        pred = logits_2d[:-1][mask_bool]
                        tgt = sub[1:].to(device)[mask_bool]
                        loss_sub = loss_func(pred, tgt)
                    else:
                        loss_sub = torch.tensor(0.0, device=device)
                else:
                    loss_sub = torch.tensor(0.0, device=device)
                # 【修复】除以 num_chunks，与正常 chunk 梯度缩放一致
                loss_scaled = loss_sub / (num_chunks * GRADIENT_ACCUMULATION_STEPS)

            if loss_scaled.requires_grad:
                if scaler.is_enabled():
                    scaler.scale(loss_scaled).backward()
                else:
                    loss_scaled.backward()
            seg_losses.append(loss_sub.detach())
            del sub, logits, loss_sub, loss_scaled

            # detach local_past，切断计算图（与主函数一致）
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
    2. 序列过长但可分块 → KV Cache 分段训练，梯度跨块累积，完整保留上下文
    3. 显存极端紧张 → 启用历史上下文压缩 + 分段训练
    4. 所有方法都失败 → 才跳过（不做截断！）
    """
    global training_rounds

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

    # 梯度累积管理
    if (training_rounds % GRADIENT_ACCUMULATION_STEPS) == 0:
        optimizer.zero_grad(set_to_none=True)

    # ── 显存安全网关 ──
    skip_thresh = float(CONFIG.get("gpu_memory_skip_ratio", 0.92))
    if mem_ratio >= skip_thresh:
        print(f"[Memory] 显存占用过高 ({mem_ratio:.1%}), 主动清理后跳过本样本", flush=True)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        training_rounds += 1  # 即使跳过也要递增，保持调度器节奏
        return float('inf')

    # ── 策略选择：估算此序列所需显存 ──
    safe_chunk = _estimate_safe_chunk_size(free_bytes, safety_factor=0.65)

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
                        # 【修复】logits 是3维 [batch, seq_len, vocab]，需要 squeeze 或按正确维度索引
                        if logits.dim() == 3:
                            logits_2d = logits.squeeze(0)  # [1, seq_len, vocab] -> [seq_len, vocab]
                        else:
                            logits_2d = logits
                        pred = logits_2d[:-1][mask_bool]
                        tgt = train_tensor_gpu[1:][mask_bool]
                        loss = loss_func(pred, tgt)
                    else:
                        loss = torch.tensor(0.0, device=device)
                else:
                    loss = torch.tensor(0.0, device=device)

            raw_loss_val = loss.item()  # 保存原始 loss 用于 ReduceLROnPlateau
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            
            # 【修复】标准训练路径必须显式调用 backward()
            # 原代码缺少此调用，导致梯度从未被计算，训练完全无效！
            # requires_grad 检查防止零 loss（无 grad_fn）时的崩溃
            if loss.requires_grad:
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

        else:
            # ✅ 策略 2: KV Cache 分段训练（完整上下文，零截断）
            chunk_size = min(safe_chunk, int(CONFIG.get("max_forward_chunk", 512)))
            overlap = int(CONFIG.get("dynamic_segment_overlap", 32))

            print(f"[Memory] 启用 KV-Cache 分段训练: seq_len={seq_len}, chunk={chunk_size}, "
                  f"overlap={overlap}, free={free_bytes/1024**3:.2f}GB", flush=True)

            loss_val = _chunked_forward_backward(train_tensor, target_mask, chunk_size, overlap)
            if loss_val is None:
                print(f"[Memory] 所有分段均 OOM，跳过本样本", flush=True)
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                training_rounds += 1  # 即使跳过也要递增，保持调度器节奏
                return float('inf')
            raw_loss_val = loss_val  # 保存原始 loss 用于 ReduceLROnPlateau
            loss = torch.tensor(loss_val, device=device)

        # ── 统一的梯度后处理 ──
        if not torch.isnan(loss) and not torch.isinf(loss):
            if scaler.is_enabled():
                # 【修复】unscale 和 clip 只在 step 前调用一次，避免梯度累积时重复除法
                if (training_rounds + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        optimizer.zero_grad(set_to_none=True)
                        scaler.update()
                        for param in model.parameters():
                            if param.grad is not None and torch.isnan(param.grad).any():
                                param.grad = None
                        print(f"[Warning] NaN/Inf gradient, skipping optimizer step", flush=True)
                        training_rounds += 1  # 即使跳过也要递增，保持调度器节奏
                        return float('inf')
                    scaler.step(optimizer)
                    scaler.update()
            else:
                if (training_rounds + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        optimizer.zero_grad(set_to_none=True)
                        for param in model.parameters():
                            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                                param.grad = None
                        print(f"[Warning] NaN/Inf gradient, skipping optimizer step", flush=True)
                        training_rounds += 1  # 即使跳过也要递增，保持调度器节奏
                        return float('inf')
                    optimizer.step()
        else:
            print(f"[Warning] Invalid loss: {loss}, skipping optimizer step", flush=True)
            training_rounds += 1  # 即使跳过也要递增，保持调度器节奏
            return float('inf')

        training_rounds += 1

        # 学习率调度 — SGDR + ReduceLROnPlateau（动态 LR，适配无限训练）
        if (training_rounds % GRADIENT_ACCUMULATION_STEPS) == 0:
            global optimizer_step_count
            optimizer_step_count += 1
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

        return loss.item()

    except RuntimeError as e:
        error_msg = str(e)
        if "CUDA" in error_msg or "cuda" in error_msg.lower() or "out of memory" in error_msg.lower():
            print(f"[CUDA Error] {e}", flush=True)
            optimizer.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            training_rounds += 1  # 即使出错也要递增，保持调度器节奏
            return float('inf')
        else:
            print(f"[RuntimeError] {e}, skipping", flush=True)
            training_rounds += 1  # 即使出错也要递增，保持调度器节奏
            return float('inf')
    except Exception as e:
        print(f"[Error] 未知错误: {e}, skipping", flush=True)
        training_rounds += 1  # 即使出错也要递增，保持调度器节奏
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