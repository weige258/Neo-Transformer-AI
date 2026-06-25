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
    """Min-p 采样 (Nguyen et al., ICLR 2025) - 修复版

    核心思想: 以最大概率 token 为锚点，只保留概率 ≥ p_max × min_p 的 token。
    正确流程: 先应用temperature → softmax计算概率 → min_p过滤 → 返回处理后的logits
    
    修复: 原实现先softmax再返回logits，导致调用方二次softmax，破坏概率分布。
    """
    if min_p <= 0.0:
        return logits
    
    # 先应用temperature
    logits_temp = logits / temperature
    
    # 计算概率分布
    probs = torch.softmax(logits_temp, dim=-1)
    p_max = probs.max().item()
    threshold = p_max * min_p
    
    # 将低于阈值的token设为-inf（在temperature后的logits上操作）
    logits_filtered = torch.where(
        probs < threshold, 
        torch.full_like(logits_temp, float("-inf")), 
        logits_temp
    )
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
    
    # None 表示无限制生成，由模型自己决定何时结束（通过 END_TOKEN）
    # 设置一个绝对上限防止极端情况下的死循环（如模型始终不输出 END_TOKEN）
    absolute_max_tokens = 4096  # 绝对安全上限
    has_token_limit = max_generate_tokens is not None
    if max_generate_tokens is None:
        max_generate_tokens = absolute_max_tokens  # 无限制时使用绝对上限作为安全网
    
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
    
    temperature = float(CONFIG.get("temperature", 0.5))
    min_p = float(CONFIG.get("min_p", 0.05))
    repetition_penalty = float(CONFIG.get("repetition_penalty", 1.02))
    force_answer_min_steps = int(CONFIG.get("force_answer_min_steps", 16))

    with torch.inference_mode():
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
        
        # 【修复】如果启用了思维链且模型没有自己生成THINK_START，强制注入
        if thinking_available and not thinking_started:
            # 检查模型是否生成了THINK_START_TOKEN
            if logits.dim() == 3:
                first_logits = logits[0, -1]
            else:
                first_logits = logits[-1]
            
            # 如果THINK_START_TOKEN不是最高概率，强制注入
            if torch.argmax(first_logits).item() != TextTokenizer.THINK_START_TOKEN:
                # 强制注入THINK_START_TOKEN
                think_start_token = torch.tensor([TextTokenizer.THINK_START_TOKEN], device=device)
                result = model(think_start_token, past_key_values=past_key_values, use_cache=True)
                if isinstance(result, tuple):
                    logits, past_key_values = result
                else:
                    logits = result
                    past_key_values = None
                thinking_started = True
                print(f"{BLUE}", end="", flush=True)
        
        while step < max_generate_tokens:
            try:
                # 获取最后一个token的logits
                if logits.dim() == 3:
                    next_logits = logits[0, -1].clone()
                else:
                    next_logits = logits[-1].clone()

                # 强制回答阶段：禁止特殊token（但允许THINK_START如果还没开始思考）
                if force_answer_steps > 0:
                    next_logits[TextTokenizer.END_GENERATION_TOKEN] = float("-inf")
                    next_logits[TextTokenizer.UNKNOWN_TOKEN] = float("-inf")
                    next_logits[TextTokenizer.START_GENERATION_TOKEN] = float("-inf")
                    # 【修复】如果还没开始思考，允许生成THINK_START
                    if thinking_started:
                        next_logits[TextTokenizer.THINK_START_TOKEN] = float("-inf")
                    next_logits[TextTokenizer.THINK_END_TOKEN] = float("-inf")
                    next_logits[TextTokenizer.HISTORY_CONTEXT_START_TOKEN] = float("-inf")
                    next_logits[TextTokenizer.HISTORY_CONTEXT_END_TOKEN] = float("-inf")
                    force_answer_steps -= 1

                # 【BUG #2修复】Repetition penalty + Frequency penalty 双重防重复
                # Repetition penalty: 乘法惩罚，对已出现的token施加倍率惩罚
                # Frequency penalty: 加法惩罚，每次出现减去固定值，更平滑有效
                frequency_penalty = float(CONFIG.get("frequency_penalty", 0.3))
                
                if repetition_penalty > 1.0 and len(_token_history) > 0:
                    window_size = 64
                    recent_tokens = _token_history[-window_size:]
                    recent_counter = Counter(recent_tokens)
                    for token_id, count in recent_counter.items():
                        if token_id < next_logits.size(0):
                            # Repetition penalty (乘法)
                            penalty = repetition_penalty ** min(count, 3)
                            if next_logits[token_id] > 0:
                                next_logits[token_id] /= penalty
                            else:
                                next_logits[token_id] *= penalty
                            # Frequency penalty (加法) - 每次出现减去固定值
                            if frequency_penalty > 0:
                                next_logits[token_id] -= frequency_penalty * min(count, 5)

                # Min-p 采样（修复版：内部已处理temperature）
                if min_p > 0.0:
                    next_logits = _min_p_sampling(next_logits, min_p, temperature)

                # 直接从处理后的logits计算概率（不再重复除temperature）
                probs = torch.softmax(next_logits, dim=-1)
                
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
                    probs = torch.softmax(next_logits, dim=-1)
                    if torch.isinf(next_logits).all() or probs.sum().item() < 1e-6:
                        break
                    index = int(torch.multinomial(probs, 1).item())
                    resample_count += 1
                
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
                    
                    # 【BUG #2修复】Repetition penalty + Frequency penalty（与主循环一致）
                    if repetition_penalty > 1.0 and len(_token_history) > 0:
                        window_size = 64
                        recent_tokens = _token_history[-window_size:]
                        recent_counter = Counter(recent_tokens)
                        for token_id, count in recent_counter.items():
                            if token_id < next_logits.size(0):
                                penalty = repetition_penalty ** min(count, 3)
                                if next_logits[token_id] > 0:
                                    next_logits[token_id] /= penalty
                                else:
                                    next_logits[token_id] *= penalty
                                if frequency_penalty > 0:
                                    next_logits[token_id] -= frequency_penalty * min(count, 5)
                    
                    # Min-p（修复版：内部已处理temperature）
                    if min_p > 0.0:
                        next_logits = _min_p_sampling(next_logits, min_p, temperature)
                    
                    probs = torch.softmax(next_logits, dim=-1)
                    
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
                        mask_bool = seg_mask[1:].to(device)
                        if mask_bool.any():
                            pred = current_logits[:-1][mask_bool]
                            tgt = seg[1:].to(device)[mask_bool]
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

            if past_kv is not None:
                past_kv = _detach_kv_cache(past_kv)

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

    # 【修复】training_rounds 在函数开头立即递增
    # 确保后续所有条件判断（zero_grad、step、scheduler）基于统一的递增值
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
    # 当 training_rounds % GAS == 0 时，表示上一周期已结束，开始新周期
    if (training_rounds % GRADIENT_ACCUMULATION_STEPS) == 0:
        optimizer.zero_grad(set_to_none=True)

    # ── 显存安全网关 ──
    skip_thresh = float(CONFIG.get("gpu_memory_skip_ratio", 0.92))
    if mem_ratio >= skip_thresh:
        print(f"[Memory] 显存占用过高 ({mem_ratio:.1%}), 主动清理后跳过本样本", flush=True)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
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
            chunk_size = min(safe_chunk, int(CONFIG.get("max_forward_chunk", 512)))
            overlap = int(CONFIG.get("dynamic_segment_overlap", 32))

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
        if (training_rounds % GRADIENT_ACCUMULATION_STEPS) == 0:
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