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

# ═══════════════════════════════════════════════════════
# 全动态计算辅助函数
# ═══════════════════════════════════════════════════════

def _get_gpu_free_memory_ratio() -> float:
    """获取GPU空闲显存比例（0.0-1.0）"""
    if not torch.cuda.is_available():
        return 1.0
    try:
        total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
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
    """计算重复模式得分（0.0-1.0）"""
    if len(token_history) < 8:
        return 0.0
    
    max_repeat_score = 0.0
    for n in [2, 3, 4]:
        if len(token_history) >= n * 3:
            last_n = tuple(token_history[-n:])
            prev_n = tuple(token_history[-(n*2):-n])
            prev_n2 = tuple(token_history[-(n*3):-(n*2)])
            if last_n == prev_n == prev_n2:
                max_repeat_score = max(max_repeat_score, n / 4.0)
    
    recent = token_history[-32:]
    if len(recent) >= 8:
        unique_ratio = len(set(recent)) / len(recent)
        repeat_score = max(0.0, 1.0 - unique_ratio * 2.0)
        max_repeat_score = max(max_repeat_score, repeat_score)
    
    return min(1.0, max_repeat_score)

def _compute_entropy_trend(entropy_history: List[float]) -> float:
    """计算熵趋势"""
    if len(entropy_history) < 5:
        return 0.0
    recent = entropy_history[-10:]
    if len(recent) < 3:
        return 0.0
    diffs = [recent[i] - recent[i-1] for i in range(1, len(recent))]
    return sum(diffs) / len(diffs)


def _is_text_like_token(idx: int) -> bool:
    """判断一个 token 是否更像文本内容，而不是乱码或控制字符。"""
    if idx in {
        TextTokenizer.UNKNOWN_TOKEN,
        TextTokenizer.START_GENERATION_TOKEN,
        TextTokenizer.END_GENERATION_TOKEN,
        TextTokenizer.HISTORY_CONTEXT_START_TOKEN,
        TextTokenizer.HISTORY_CONTEXT_END_TOKEN,
        TextTokenizer.THINK_START_TOKEN,
        TextTokenizer.THINK_END_TOKEN,
    }:
        return False

    if not isinstance(idx, int) or idx <= 0 or idx > 0x10FFFF:
        return False

    # 【简化】只排除控制字符和不可打印字符
    ch = chr(idx)
    if not ch.isprintable() and ch not in {'\n', '\t', '\r', ' '}:
        return False
    
    return True


def _looks_meaningful_text(text: str) -> bool:
    """判断生成文本是否像有效自然语言，而不是纯乱码。"""
    if not text or not text.strip():
        return False
    if len(text.strip()) < 1:
        return False

    chinese_chars = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
    letters = sum(1 for ch in text if ch.isalpha())
    digits = sum(1 for ch in text if ch.isdigit())
    total = len(text)

    if total == 0:
        return False

    if chinese_chars + letters > 0:
        return True

    if digits > total * 0.5:
        return False

    meaningful = chinese_chars + letters + digits
    if meaningful > total * 0.3:
        return True

    return False


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
autocast_device_type = "cuda" if device.type == "cuda" else "cpu"


# ──────────────────────────────────────────────────────────
# 全局训练状态
# ──────────────────────────────────────────────────────────
model = MainModel()
model.to(device)

# 检查是否有可用的预训练权重
pretrained_path = "model.pth"
# 新格式检查点中恢复的优化器状态与步数计数（旧格式纯 state_dict 时为默认值）
_pretrained_optimizer_state = None
_pretrained_optimizer_step_count = 0
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
            # 【修复】不再回退 weights_only=False（pickle 反序列化风险），
            # 加载失败直接抛出，走下方既有警告路径（不覆盖旧文件）
            checkpoint = torch.load(pretrained_path, map_location=device, weights_only=True)
            # 向后兼容：新格式为 dict（含优化器状态），旧格式为纯 state_dict
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                state_dict = checkpoint["model"]
                _pretrained_optimizer_state = checkpoint.get("optimizer")
                _pretrained_optimizer_step_count = int(checkpoint.get("optimizer_step_count", 0))
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict)
            print(f"Loaded pretrained model from {pretrained_path}")
    except Exception as e:
        print(f"Warning: Failed to load pretrained model: {e}")
        print(f"将从随机初始化权重开始训练，新权重会自动覆盖 {pretrained_path}")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=float(CONFIG.get("base_learning_rate", 1e-4)),
    weight_decay=float(CONFIG.get("weight_decay", 0.02)),
    betas=(float(CONFIG.get("adam_beta1", 0.9)), float(CONFIG.get("adam_beta2", 0.98))),
    eps=float(CONFIG.get("adam_eps", 1e-6)),
)

# 新格式检查点包含优化器状态时，恢复之（失败则保持全新优化器，不影响模型权重）
if _pretrained_optimizer_state is not None:
    try:
        optimizer.load_state_dict(_pretrained_optimizer_state)
        print("已恢复优化器状态")
    except Exception as e:
        print(f"Warning: 无法恢复优化器状态: {e}")

# 学习率调度器：常数学习率（无warmup，无SGDR，无ReduceLROnPlateau）
# 【修复】完全移除所有复杂调度，使用最简单的常数学习率
# 原因：50轮训练步数太少，复杂调度器只会干扰收敛
class ConstantScheduler:
    """常数学习率调度器 - 最简单最稳定"""
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.step_count = 0
        
    def step(self, loss=None):
        self.step_count += 1
        return self.optimizer.param_groups[0]['lr']

lr_scheduler = ConstantScheduler(optimizer)


def _save_checkpoint() -> None:
    """保存模型检查点到 model.pth，按 checkpoint_interval 控制频率。

    【修复】原子写入：先写临时文件再 os.replace，避免写一半崩溃损坏唯一权重。
    保存内容包含优化器状态与步数计数，便于断点续训。
    """
    interval = int(CONFIG.get("checkpoint_interval", 1000))
    if interval <= 0:
        return
    if optimizer_step_count <= 0 or optimizer_step_count % interval != 0:
        return
    try:
        tmp_path = pretrained_path + ".tmp"
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "optimizer_step_count": optimizer_step_count,
        }, tmp_path)
        os.replace(tmp_path, pretrained_path)
        print(f"[Checkpoint] Saved model at optimizer step {optimizer_step_count} to {pretrained_path}", flush=True)
    except Exception as e:
        print(f"[Warning] Failed to save checkpoint: {e}", flush=True)


# 混合精度训练 (AMP)
use_amp = bool(CONFIG.get("use_amp", True)) and torch.cuda.is_available()
amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
scaler = torch.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

print(f"Using device: {device}")
print(f"AMP enabled: {use_amp} AMP dtype: {amp_dtype}")

loss_func = torch.nn.CrossEntropyLoss()

# 梯度累积步数
GRADIENT_ACCUMULATION_STEPS = int(CONFIG.get("gradient_accumulation_steps", 4))

# 学习率调度器步进间隔（Gemini修复：防止SGDR震荡过于频繁）
LR_SCHEDULER_STEP_INTERVAL = int(CONFIG.get("lr_scheduler_step_interval", 4))

# 训练轮数计数器
training_rounds = 0
# 新格式检查点可恢复优化器步数计数（旧格式为0）
optimizer_step_count = _pretrained_optimizer_step_count


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
            (TextTokenizer.START_GENERATION_TOKEN, True),  # 学习在ask后输出START
            (TextTokenizer.THINK_START_TOKEN, True),       # 学习在START后输出THINK_START
            (answer_tensor, True),                         # 学习生成回答（无思考内容时直接回答）
            (TextTokenizer.THINK_END_TOKEN, True),         # 学习在回答前输出THINK_END（结束空思考）
            (TextTokenizer.END_GENERATION_TOKEN, True),    # 学习在回答结束时输出END
        ])
    else:
        train_tensor, target_mask = _build_train_sequence([
            (ask_tensor, False),
            (TextTokenizer.START_GENERATION_TOKEN, True),  # 学习在ask后输出START
            (TextTokenizer.THINK_START_TOKEN, True),       # 学习在START后输出THINK_START
            (answer_tensor, True),                         # 学习生成回答（无思考内容时直接回答）
            (TextTokenizer.THINK_END_TOKEN, True),         # 学习在回答前输出THINK_END（结束空思考）
            (TextTokenizer.END_GENERATION_TOKEN, True),    # 学习在回答结束时输出END
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
    
    # 固定最大生成长度，仅由配置上限控制，不再运行时动态调整
    gen_min = int(CONFIG.get("gen_len_min_absolute", 64))
    gen_max = int(CONFIG.get("gen_len_max_absolute", 4096))
    if max_generate_tokens is None:
        max_generate_tokens = gen_max
    max_generate_tokens = max(gen_min, min(gen_max, int(max_generate_tokens)))
    absolute_max_tokens = max_generate_tokens
    has_token_limit = True
    
    model.eval()
    output_text = ""

    # 生成prompt与训练格式一致:
    # 训练: ask + START_GENERATION + THINK_START + think + THINK_END + answer + END
    # 生成: ask + START_GENERATION + THINK_START（模型生成think+THINK_END+answer+END）
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
            torch.tensor([TextTokenizer.THINK_START_TOKEN], device=device),
        ])
    else:
        prompt = torch.cat([
            text_tensor,
            torch.tensor([TextTokenizer.START_GENERATION_TOKEN], device=device),
            torch.tensor([TextTokenizer.THINK_START_TOKEN], device=device),
        ])

    print("\n---Generated reply:", flush=True)

    max_generate_tokens = max(1, int(max_generate_tokens))
    
    # 【EDT温度 & 重复惩罚】系数
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
    min_generation_steps_before_stop = max(1, int(CONFIG.get("min_generation_steps_before_stop", 4)))
    
    # 重复惩罚系数
    rep_penalty_scale = float(CONFIG.get("rep_penalty_scale", 0.25))
    rep_penalty_length_factor = float(CONFIG.get("rep_penalty_length_factor", 0.002))
    rep_penalty_repeat_sensitivity = float(CONFIG.get("rep_penalty_repeat_sensitivity", 2.0))
    rep_penalty_entropy_factor = float(CONFIG.get("rep_penalty_entropy_factor", 0.8))
    frequency_penalty = float(CONFIG.get("frequency_penalty", 0.3))
    presence_penalty = float(CONFIG.get("presence_penalty", 0.1))
    
    # 【全动态强制回答步数】运行时计算
    force_answer_scale = float(CONFIG.get("force_answer_scale", 1.2))
    force_answer_min_absolute = int(CONFIG.get("force_answer_min_absolute", 16))
    force_answer_complexity_exp = float(CONFIG.get("force_answer_complexity_exp", 0.5))
    # 【修复】计算 question_len 和 complexity_score（基于输入文本）
    question_len = len(text) if text else 0
    complexity_score = _estimate_question_complexity(text)
    force_answer_min_steps = max(force_answer_min_absolute, int(question_len * force_answer_scale * (1.0 + complexity_score ** force_answer_complexity_exp)))

    with torch.inference_mode():
        # prompt已包含THINK_START，模型从思考阶段（蓝色）开始
        thinking_started = True
        force_answer_steps = 0
        print(f"{BLUE}", end="", flush=True)
        
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
        
        # 【修复】移除硬编码的2048限制，让配置中的gen_len_max_absolute生效
        hard_limit = max_generate_tokens
        temperature = temp_base
        _generation_start_time = time.time()
        _generation_timeout = 120.0  # 生成超时120秒
        
        while step < hard_limit:
            try:
                # 【修复】生成超时保护，防止卡死
                if time.time() - _generation_start_time > _generation_timeout:
                    print(f"\n[Timeout] 生成超时({_generation_timeout}s)，强制结束", flush=True)
                    break
                
                # 获取最后一个token的logits
                if logits.dim() == 3:
                    next_logits = logits[0, -1].clone()
                else:
                    next_logits = logits[-1].clone()

                # 前min_generation_steps步禁止输出END，防止立即终止
                if step < min_generation_steps_before_stop:
                    next_logits[TextTokenizer.END_GENERATION_TOKEN] = float("-inf")

                # 强制回答阶段：禁止特殊token
                if force_answer_steps > 0:
                    next_logits[TextTokenizer.END_GENERATION_TOKEN] = float("-inf")
                    next_logits[TextTokenizer.UNKNOWN_TOKEN] = float("-inf")
                    next_logits[TextTokenizer.START_GENERATION_TOKEN] = float("-inf")
                    next_logits[TextTokenizer.THINK_START_TOKEN] = float("-inf")
                    next_logits[TextTokenizer.THINK_END_TOKEN] = float("-inf")
                    next_logits[TextTokenizer.HISTORY_CONTEXT_START_TOKEN] = float("-inf")
                    next_logits[TextTokenizer.HISTORY_CONTEXT_END_TOKEN] = float("-inf")
                    force_answer_steps -= 1

                # 屏蔽surrogate范围token，防止生成无法解码的空字符
                next_logits[0xD800:0xE000] = float("-inf")

                # 温度缩放
                logits_for_sample = next_logits / max(temperature, 0.01)

                # 滑动窗口硬阻断：最近20个token中出现>=3次的直接屏蔽
                if _token_history:
                    recent_window = _token_history[-20:]
                    recent_counts = Counter(recent_window)
                    for tok, count in recent_counts.items():
                        if count >= 3:
                            logits_for_sample[tok] = float("-inf")
                        elif count >= 2:
                            logits_for_sample[tok] -= frequency_penalty * count * 2.0  # 增强惩罚
                        elif count >= 1:
                            logits_for_sample[tok] -= frequency_penalty * 0.5  # count=1时也施加轻惩罚

                # 【连字惩罚】阻止连续输出相同字符，防止"给给给"、"迎迎迎"等结巴现象
                if _token_history:
                    prev_token = _token_history[-1]
                    logits_for_sample[prev_token] -= 5.0  # 强力惩罚紧邻的相同字符

                # argmax (greedy) 解码 - 用于验证模型学习效果
                index = int(torch.argmax(logits_for_sample).item())
                
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
                        # 思考阶段异常结束：强制过渡到回答阶段
                        print(f"\n{YELLOW}[CoT Guard] 思考中输出END，强制过渡到回答{RESET}", flush=True)
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
                    
                    # 屏蔽surrogate范围token，防止生成无法解码的空字符
                    next_logits[0xD800:0xE000] = float("-inf")

                    # 温度缩放
                    logits_fa = next_logits / max(temperature, 0.01)

                    # 滑动窗口硬阻断：最近20个token中出现>=3次的直接屏蔽
                    if _token_history:
                        recent_window = _token_history[-20:]
                        recent_counts = Counter(recent_window)
                        for tok, count in recent_counts.items():
                            if count >= 3:
                                logits_fa[tok] = float("-inf")
                            elif count >= 2:
                                logits_fa[tok] -= frequency_penalty * count

                    # top-k 采样
                    if top_k > 0 and top_k < logits_fa.size(-1):
                        v, _ = torch.topk(logits_fa, min(top_k, logits_fa.size(-1)))
                        logits_fa[logits_fa < v[-1]] = float('-inf')
                    pf = torch.softmax(logits_fa, dim=-1)
                    pf = torch.nan_to_num(pf, nan=0.0)
                    if pf.sum() > 0:
                        index = int(torch.multinomial(pf, num_samples=1).item())
                    else:
                        index = int(torch.argmax(next_logits).item())
                    
                    # END_TOKEN结束强制回答
                    if index == TextTokenizer.END_GENERATION_TOKEN:
                        break

                    if index not in (
                        TextTokenizer.THINK_START_TOKEN,
                        TextTokenizer.THINK_END_TOKEN,
                        TextTokenizer.START_GENERATION_TOKEN,
                        TextTokenizer.END_GENERATION_TOKEN,
                    ) and not _is_text_like_token(index):
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

        if not _looks_meaningful_text(output_text):
            if text and text.strip():
                output_text = f"我已收到你的问题：{text}"
            else:
                output_text = "我已收到你的消息。"

        # 生成完成后清理GPU缓存
        past_key_values = None  # 释放KV cache引用
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return output_text


def train(ask: str = None, think: str = None, answer: str = None, history_context: str = None) -> float | None:
    """单步训练函数
    
    Args:
        ask: 问题文本
        think: 思维链/推理过程（可选，用于CoT训练）
        answer: 答案文本
        history_context: 历史对话上下文
    
    Returns:
        loss值，如果训练失败返回None
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
        return None
    
    if ask is None:
        print(f"\n---Train{RESET}", flush=True)

        text_tensor = TextTokenizer.encode(answer)
        if text_tensor.numel() < 2:
            return None

        train_tensor, target_mask = _build_train_sequence([
            (TextTokenizer.START_GENERATION_TOKEN, True),
            (text_tensor, True),
            (TextTokenizer.END_GENERATION_TOKEN, True),
        ])
        preview = train_tensor
        return _run_train_step(train_tensor, target_mask, preview, show_preview=True, preview_color=YELLOW)

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
                return _run_train_step(train_tensor, target_mask, preview, show_preview=False)
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
                return _run_train_step(train_tensor, target_mask, preview, show_preview=False)
        
        print(f"{GREEN}{answer}{RESET}", flush=True)
        train_tensor, target_mask, preview = _prepare_training_data(ask, answer, history_context)
        if train_tensor is None:
            return None
        return _run_train_step(train_tensor, target_mask, preview, show_preview=False)

    # 【修复】原 RL/PPO 分支已删除：上面 answer 非空时均已先行返回，
    # 该分支只有 answer 为空才可能到达，所有真实调用方都传 answer，实际永远不可达；
    # 且一旦触发会用空 answer 收集无意义 episode。answer 为空时直接结束（返回 None）。


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
                # 【修复】mem_pos（索引4，1-D）也要同步截断，
                # 否则 mem_k/mem_v 截断后与 mem_pos 长度不一致
                elif i == 4 and dt.dim() == 1 and dt.shape[0] > limit:
                    dt = dt[-limit:].contiguous()
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
    # 【修复】bytes_per_token必须考虑forward+backward的完整开销
    # forward: 激活值 = emb_size * num_layers * 2 (bf16)
    # backward: 梯度激活值 ≈ 2x forward
    # KV cache: emb_size * num_layers * 2 * 2 (K+V, bf16)
    # 总计约 8x 简单激活值
    bytes_per_token = emb_size * num_layers * 2 * 8
    
    # 获取硬件状态
    gpu_free_ratio = _get_gpu_free_memory_ratio()
    cpu_free_ratio = _get_cpu_free_memory_ratio()
    
    # 获取配置系数
    chunk_memory_ratio = float(CONFIG.get("chunk_memory_ratio", 0.15))
    chunk_seq_len_factor = float(CONFIG.get("chunk_seq_len_factor", 0.3))
    chunk_min_absolute = int(CONFIG.get("chunk_min_absolute", 128))
    chunk_max_ratio = float(CONFIG.get("chunk_max_ratio", 0.5))
    chunk_cpu_pressure_factor = float(CONFIG.get("chunk_cpu_pressure_factor", 0.2))
    
    # 【修复】可用显存直接采用 total - allocated 口径：
    # gpu_free_ratio 已基于 (total - allocated) 计算，allocated 本身包含
    # 模型参数/梯度/优化器状态，此处不再重复扣除模型固定开销
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
        free_memory = total_memory * gpu_free_ratio * 0.85  # 保留15%安全余量
    else:
        free_memory = 8 * 1024**3  # 默认8GB
    
    # 显存因子（显存紧张时更保守）
    memory_factor = gpu_free_ratio * (1.0 - (1.0 - cpu_free_ratio) * chunk_cpu_pressure_factor)
    
    # 计算chunk大小
    # 1. 基于显存（使用更合理的比例）
    # 【修复】使用配置中的比例，不再强制减半，配合config.py中的新值
    mem_based_chunk = int(free_memory * chunk_memory_ratio * memory_factor / bytes_per_token)
    
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
    
    # 【修复】chunk大小硬上限1024，配合config.py中的更大chunk配置
    chunk_size = min(chunk_size, 1024)
    
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
    chunk_weights = []  # 各 chunk 有效 loss token 数，用于加权平均
    past_kv = None
    max_chunks = 32  # 【修复】最大chunk数限制，防止无限循环
    chunk_count = 0

    for seg_start in range(0, seq_len, step):
        chunk_count += 1
        if chunk_count > max_chunks:
            print(f"[Memory] 超过最大chunk数({max_chunks})，截断剩余序列", flush=True)
            break
        seg_end = min(seg_start + chunk_size, seq_len)
        seg = train_tensor[seg_start:seg_end].to(device)
        seg_mask = target_mask[seg_start:seg_end]

        try:
            with torch.autocast(device_type=autocast_device_type, dtype=amp_dtype, enabled=use_amp):
                result = model(seg.unsqueeze(0) if seg.dim() == 1 else seg,
                               past_key_values=past_kv, use_cache=True)
                if isinstance(result, tuple):
                    logits, past_kv = result
                else:
                    logits = result
                    past_kv = None

                n_tokens = 0
                if seg.numel() > 1 and seg_mask.any():
                    if logits.dim() == 3:
                        logits_2d = logits.squeeze(0)
                    else:
                        logits_2d = logits

                    # 所有chunk都使用 seg[1:] 计算loss（语言模型预测下一个token）
                    # 第一个token没有前一个token来预测它，所以自然地从索引1开始
                    mask_bool = seg_mask[1:].to(device)
                    # 【修复】overlap 区的 target 在上一 chunk 已计过一次 loss，
                    # 非首个 chunk 将前 overlap-1 个 target 位置的 mask 置 False，
                    # 保证每个 token 只计一次 loss（KV/past 传递逻辑不变，仍输入完整 seg）
                    if seg_start > 0 and overlap > 1:
                        mask_bool[:overlap - 1] = False
                    if mask_bool.any():
                        pred = logits_2d[:-1][mask_bool]
                        tgt = seg[1:].to(device)[mask_bool]
                        loss_chunk = loss_func(pred, tgt)
                        n_tokens = int(mask_bool.sum().item())
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
            chunk_weights.append(n_tokens)
            del seg, logits, loss_chunk, loss_scaled

            # 【修复】强制清理中间变量，防止显存泄漏
            if past_kv is not None:
                past_kv = _detach_kv_cache(past_kv)
            # 【修复】删除每个 chunk 后的 empty_cache：它会同步设备并强制重新
            # 向驱动申请显存，长序列几十个 chunk 每步都付出代价且基本无效

        except RuntimeError as e_oom:
            if "out of memory" in str(e_oom).lower():
                smaller = max(128, chunk_size // 2)
                if smaller < chunk_size:
                    print(f"[Memory] Chunk OOM at [{seg_start}:{seg_end}], 缩半到{smaller}重试", flush=True)
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    safe_past = _detach_kv_cache(past_kv) if past_kv is not None else None
                    # 非首个 chunk 的前 overlap-1 个 target 已由上一 chunk 计过 loss，跳过
                    skip_first = (overlap - 1) if (seg_start > 0 and overlap > 1) else 0
                    sub = _chunk_one_segment(seg, seg_mask, safe_past, smaller,
                                             grad_scale=grad_scale, skip_first=skip_first)
                    if sub is not None:
                        sub_loss, past_kv, sub_tokens = sub
                        chunk_losses.append(sub_loss.detach())
                        chunk_weights.append(sub_tokens)
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
    # 【修复】按各 chunk 有效 token 数加权平均，避免稀疏 chunk 与密集 chunk 等权
    losses_t = torch.stack([l.to(device) for l in chunk_losses])
    weights_t = torch.tensor(chunk_weights, dtype=torch.float32, device=device)
    total_weight = weights_t.sum()
    if total_weight > 0:
        avg_loss = (losses_t * weights_t).sum() / total_weight
    else:
        avg_loss = losses_t.mean()
    return avg_loss.item()


def _chunk_one_segment(
    seg: torch.Tensor, seg_mask: torch.Tensor, past_kv, chunk_size: int,
    grad_scale: int = 1,
    skip_first: int = 0,
) -> tuple[torch.Tensor, any, int] | None:
    """OOM 回退：将单个 segment 进一步细分后训练，返回 (avg_loss, past_kv, 有效token数) 或 None。

    【修复】统一梯度缩放：每个子 chunk 的 loss 除以 grad_scale，
    与主循环的缩放一致。细分后的多个子 chunk 会自然累加它们的梯度，
    这与长序列提供更多训练信号的直觉相符。

    Args:
        skip_first: 首个子 chunk 开头已由上一 chunk 计过 loss 的 target 数，跳过避免重复计权
    """
    seg_len = seg.numel()
    step = max(1, chunk_size // 2)
    seg_losses = []
    seg_weights = []  # 各子 chunk 有效 loss token 数，用于加权平均
    local_past = past_kv

    for s in range(0, seg_len, step):
        e = min(s + chunk_size, seg_len)
        sub = seg[s:e].to(device)
        sub_mask = seg_mask[s:e]

        try:
            with torch.autocast(device_type=autocast_device_type, dtype=amp_dtype, enabled=use_amp):
                result = model(sub.unsqueeze(0) if sub.dim() == 1 else sub,
                               past_key_values=local_past, use_cache=True)
                if isinstance(result, tuple):
                    logits, local_past = result
                else:
                    logits = result
                    local_past = None
                n_tokens = 0
                if sub.numel() > 1 and sub_mask.any():
                    if logits.dim() == 3:
                        logits_2d = logits.squeeze(0)
                    else:
                        logits_2d = logits

                    # 【修复】overlap 区的 target 在上一子 chunk 已计过一次 loss，
                    # 非首个子 chunk 跳过前 (chunk_size - step - 1) 个 target 位置，
                    # 首个子 chunk 跳过 skip_first 个（主循环 overlap 区），保证每个 token 只计一次
                    skip = skip_first if s == 0 else max(0, chunk_size - step - 1)

                    if local_past is not None and s > 0:
                        current_logits = logits_2d[-sub.numel():]
                        mask_bool = sub_mask[1:].to(device)
                        if skip > 0:
                            mask_bool[:skip] = False
                        if mask_bool.any():
                            pred = current_logits[:-1][mask_bool]
                            tgt = sub[1:].to(device)[mask_bool]
                            loss_sub = loss_func(pred, tgt)
                            n_tokens = int(mask_bool.sum().item())
                        else:
                            loss_sub = torch.tensor(0.0, device=device)
                    else:
                        mask_bool = sub_mask[1:].to(device)
                        if skip > 0:
                            mask_bool[:skip] = False
                        if mask_bool.any():
                            pred = logits_2d[:-1][mask_bool]
                            tgt = sub[1:].to(device)[mask_bool]
                            loss_sub = loss_func(pred, tgt)
                            n_tokens = int(mask_bool.sum().item())
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
            seg_weights.append(n_tokens)
            del sub, logits, loss_sub, loss_scaled

            if local_past is not None:
                local_past = _detach_kv_cache(local_past)
            # 【修复】删除每个子 chunk 后的 empty_cache：同步设备且对防显存累积基本无效

        except RuntimeError:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            continue

    if seg_losses:
        # 【修复】按各子 chunk 有效 token 数加权平均，避免稀疏 chunk 与密集 chunk 等权
        losses_t = torch.stack(seg_losses)
        weights_t = torch.tensor(seg_weights, dtype=torch.float32, device=losses_t.device)
        total_weight = weights_t.sum()
        if total_weight > 0:
            avg = (losses_t * weights_t).sum() / total_weight
        else:
            avg = losses_t.mean()
        return avg, local_past, int(total_weight.item())
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

    # 【修复】删除超长序列左截断（保留尾部会丢掉 ask/history/START，
    # 且使分块路径对超长序列永不生效）；超长序列交给下方既有分块训练路径处理

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

    # ── 显存安全网关 ──
    skip_thresh = float(CONFIG.get("gpu_memory_skip_ratio", 0.80))
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
    
    chunk_overlap_base = int(CONFIG.get("chunk_overlap_base", 32))
    chunk_overlap_scale = float(CONFIG.get("chunk_overlap_scale", 0.02))
    overlap = int(chunk_overlap_base + safe_chunk * chunk_overlap_scale)
    overlap = min(overlap, safe_chunk // 4)  # overlap不超过chunk的1/4

    try:
        if seq_len <= safe_chunk:
            # ✅ 策略 1: 标准训练（序列完整放入 GPU）
            train_tensor_gpu = train_tensor.to(device)
            with torch.autocast(device_type=autocast_device_type, dtype=amp_dtype, enabled=use_amp):
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
            del logits, train_tensor_gpu

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

        # 【修复】mask 全空时 loss 无梯度、不参与 backward，若仍执行 optimizer step，
        # AdamW 的 weight decay 会在空样本上持续衰减权重；此处直接跳过整个 step
        if seq_len <= 1 or not target_mask[1:].any():
            print(f"[Warning] target_mask 全空，无有效训练信号，跳过 optimizer step", flush=True)
            optimizer.zero_grad(set_to_none=True)
            return float('inf')

        # ── 统一的梯度后处理 ──
        if not torch.isnan(loss) and not torch.isinf(loss):
            should_step = (training_rounds % GRADIENT_ACCUMULATION_STEPS) == 0

            if scaler.is_enabled():
                if should_step:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(CONFIG.get("max_grad_norm", 5.0)))
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
                    optimizer.zero_grad(set_to_none=True)
            else:
                if should_step:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(CONFIG.get("max_grad_norm", 5.0)))
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        optimizer.zero_grad(set_to_none=True)
                        for param in model.parameters():
                            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                                param.grad = None
                        print(f"[Warning] NaN/Inf gradient, skipping optimizer step", flush=True)
                        return float('inf')
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        else:
            print(f"[Warning] Invalid loss: {loss}, skipping optimizer step", flush=True)
            return float('inf')

        # 学习率调度
        if should_step:
            optimizer_step_count += 1
            current_lr = lr_scheduler.step(loss=raw_loss_val)
            if optimizer_step_count % 100 == 0:
                print(f"[LR] Step {optimizer_step_count}, current LR: {current_lr:.2e}, loss: {raw_loss_val:.4f}", flush=True)
            _save_checkpoint()

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

        if training_rounds % 10 == 0 or training_rounds <= 5:
            print(f"[Loss] Step {training_rounds}: loss={raw_loss_val:.4f}", flush=True)

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