from typing import List, Tuple, Optional
import sys
import os
import torch
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
# 安全的训练序列构建工具
# ──────────────────────────────────────────────────────────

def _build_train_sequence(
    segments: List[Tuple[torch.Tensor | int, bool]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """根据 (token/常量, is_target) 段列表构建 train_tensor 和 target_mask。

    每个段为 (data, is_target)：
      - data: 可以是 torch.Tensor（token 序列）或 int（单个特殊 token）
      - is_target: True = 该段参与 loss 计算，False = 仅作为上下文

    返回 (train_tensor, target_mask)，两者长度严格相等。
    """
    tensors: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []

    for data, is_target in segments:
        if isinstance(data, int):
            t = torch.tensor([data], device=device, dtype=torch.long)
        elif isinstance(data, torch.Tensor):
            t = data.to(device=device, dtype=torch.long)
        else:
            raise TypeError(f"_build_train_sequence: unexpected segment type {type(data)}")

        if t.numel() == 0:
            continue

        tensors.append(t)
        masks.append(torch.full((t.numel(),), is_target, device=device, dtype=torch.bool))

    if not tensors:
        # 返回一个最小的哑元序列
        dummy = torch.tensor([TextTokenizer.UNKNOWN_TOKEN], device=device, dtype=torch.long)
        return dummy, torch.tensor([False], device=device, dtype=torch.bool)

    train_tensor = torch.cat(tensors, dim=0)
    target_mask = torch.cat(masks, dim=0)

    assert target_mask.numel() == train_tensor.numel(), (
        f"_build_train_sequence: mask len {target_mask.numel()} != train len {train_tensor.numel()}"
    )
    return train_tensor, target_mask


def _load_model() -> MainModel:
    try:
        # 安全加载：不使用不存在的 `weights_only` 参数，使用 map_location
        loaded = torch.load("model.pth", map_location=device)
        model = MainModel().to(device)
        
        # 【修复】严格校验键匹配，避免加载不匹配的权重导致静默随机初始化
        # 根据PyTorch最佳实践，应该使用strict=True或在不匹配时抛出异常
        model_state = model.state_dict()
        
        # 检查是否有shape不匹配的键
        shape_mismatched = []
        for k, v in loaded.items():
            if k in model_state:
                if v.shape != model_state[k].shape:
                    shape_mismatched.append(f"{k}: expected {model_state[k].shape}, got {v.shape}")
            else:
                shape_mismatched.append(f"{k}: 模型中不存在此键")
        
        if shape_mismatched:
            error_msg = "\n".join(shape_mismatched)
            raise ValueError(
                f"权重文件与模型结构不匹配，加载中止！\n"
                f"不匹配的键：\n{error_msg}\n\n"
                f"请检查：\n"
                f"1. dict_size、emb_size、层数等配置是否与保存权重时的配置一致\n"
                f"2. 是否使用了错误的model.pth文件"
            )
        
        # 严格加载
        model.load_state_dict(loaded, strict=True)
        print("Loaded model state dict with strict validation.", flush=True)
        return model
    except FileNotFoundError:
        print("model.pth not found. Creating new model.", flush=True)
        model = MainModel().to(device)
        print("Created new model.", flush=True)
        return model
    except ValueError as e:
        # 捕获形状不匹配的异常并终止程序
        print(f"\n{'='*60}", flush=True)
        print(f"[ERROR] {e}", flush=True)
        print(f"{'='*60}\n", flush=True)
        print("程序已终止。请修正配置或权重文件后重新运行。", flush=True)
        import sys
        sys.exit(1)
    except Exception as e:
        print(f"Failed to load model: {e}", flush=True)
        model = MainModel().to(device)
        print("Created new model.", flush=True)
        return model


# 【性能优化】启用TensorFloat32加速矩阵运算(消除UserWarning)
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

# 检查GPU能力，对于较老的GPU（compute capability < 8.0）禁用AMP
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(device)
    # bfloat16 有和 float32 一样的动态范围，不会溢出
    use_amp = True
    if cap[0] >= 8:
        amp_dtype = torch.bfloat16  # Ampere (A100, RTX 30xx/40xx) 才支持 bfloat16
    else:
        amp_dtype = torch.float16  # 老显卡用float16，同样能降显存
else:
    use_amp = False
    amp_dtype = torch.float32

# 【修复】仅float16启用scaler，bfloat16无需缩放，避免梯度爆炸
# 【PyTorch 2.x 更新】使用 torch.amp.GradScaler('cuda') 替代已弃用的 torch.cuda.amp.GradScaler
scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and amp_dtype == torch.float16))

print(f"Using device: {device}", flush=True)
print(f"AMP enabled: {use_amp}, AMP dtype: {amp_dtype}", flush=True)
model = _load_model()
# 检索模块已移除；使用向量压缩与卸载策略来在训练/推理时减小显存占用
print("[Info] Retrieval module removed; using vector compression/offload.", flush=True)


def _get_gpu_memory_ratio(device=None) -> float:
    """返回当前 GPU 的显存使用比例（0.0 当不可用）。
    使用 torch.cuda.memory_reserved/allocated 和 device 总显存计算。
    """
    try:
        if not torch.cuda.is_available():
            return 0.0
        # 解析 device index
        if device is None:
            idx = torch.cuda.current_device()
        else:
            if isinstance(device, torch.device):
                if device.type == 'cuda' and device.index is not None:
                    idx = device.index
                else:
                    idx = torch.cuda.current_device()
            else:
                idx = int(device)
        props = torch.cuda.get_device_properties(idx)
        total = float(props.total_memory)
        reserved = float(torch.cuda.memory_reserved(idx))
        allocated = float(torch.cuda.memory_allocated(idx))
        used = max(reserved, allocated)
        return used / total if total > 0 else 0.0
    except Exception:
        return 0.0


def auto_compress_trigger(history_tensor) -> bool:
    """仅在 GPU 内存占用接近配置阈值时返回 True，否则返回 False。
    该函数不会直接执行压缩；调用点应决定何时真正压缩并卸载。
    """
    try:
        ratio = _get_gpu_memory_ratio(history_tensor.device if history_tensor is not None else None)
        thresh = float(CONFIG.get("compress_on_memory_ratio", 0.9))
        return ratio >= thresh
    except Exception:
        return False

# 【显存优化】关闭torch.compile，避免额外显存占用
print("[Info] Running without torch.compile optimization (disabled for memory efficiency).", flush=True)

total_params = sum(param.numel() for param in model.parameters())
print(f"模型参数: {total_params / 1e+8}亿", flush=True)

loss_func = torch.nn.CrossEntropyLoss().to(device)

# 【学习率配置】从CONFIG读取优化器参数
base_lr = float(CONFIG.get("base_learning_rate", 2e-4))
weight_decay = float(CONFIG.get("weight_decay", 0.01))
adam_beta1 = float(CONFIG.get("adam_beta1", 0.9))
adam_beta2 = float(CONFIG.get("adam_beta2", 0.999))
adam_epsilon = float(CONFIG.get("adam_epsilon", 1e-8))
optimizer_type = CONFIG.get("optimizer_type", "adamw").lower()

# 根据配置选择优化器
if optimizer_type == "adamw":
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        betas=(adam_beta1, adam_beta2),
        eps=adam_epsilon,
        weight_decay=weight_decay,
        foreach=torch.cuda.is_available(),
    )
elif optimizer_type == "adam":
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=base_lr,
        betas=(adam_beta1, adam_beta2),
        eps=adam_epsilon,
        weight_decay=weight_decay,
        foreach=torch.cuda.is_available(),
    )
elif optimizer_type == "sgd":
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=0.9,
        weight_decay=weight_decay,
    )
else:
    print(f"[Warning] Unknown optimizer type '{optimizer_type}', falling back to AdamW", flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        betas=(adam_beta1, adam_beta2),
        eps=adam_epsilon,
        weight_decay=weight_decay,
        foreach=torch.cuda.is_available(),
    )

print(f"[Info] Optimizer: {optimizer_type.upper()}, LR: {base_lr:.2e}, Weight Decay: {weight_decay:.2e}", flush=True)

# 【学习率调度器】实现Warmup + Cosine Decay
GRADIENT_ACCUMULATION_STEPS = 4
training_rounds = 0

# 学习率调度器配置
warmup_steps = int(CONFIG.get("warmup_steps", 300))
warmup_init_lr = float(CONFIG.get("warmup_init_lr", 1e-7))
min_learning_rate = float(CONFIG.get("min_learning_rate", 1e-6))
total_training_steps = int(CONFIG.get("total_training_steps", 30000))
cosine_decay_enabled = bool(CONFIG.get("cosine_decay_enabled", True))
lr_scheduler_type = CONFIG.get("lr_scheduler_type", "cosine")

def get_learning_rate(current_step: int) -> float:
    """计算当前步的学习率（支持Warmup + 多种调度策略）
    
    Args:
        current_step: 当前训练步数
        
    Returns:
        当前学习率
    """
    if current_step < warmup_steps:
        # Warmup阶段：线性增长从warmup_init_lr到base_lr
        warmup_progress = current_step / max(warmup_steps - 1, 1)
        return warmup_init_lr + (base_lr - warmup_init_lr) * warmup_progress
    else:
        # Warmup结束后，根据调度器类型计算学习率
        if lr_scheduler_type == "cosine" and cosine_decay_enabled:
            # Cosine Decay：从base_lr衰减到min_learning_rate
            progress = (current_step - warmup_steps) / max(total_training_steps - warmup_steps, 1)
            progress = min(progress, 1.0)  # 限制在[0, 1]
            
            import math
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_learning_rate + (base_lr - min_learning_rate) * cosine_decay
        elif lr_scheduler_type == "linear":
            # Linear Decay：线性衰减到min_learning_rate
            progress = (current_step - warmup_steps) / max(total_training_steps - warmup_steps, 1)
            progress = min(progress, 1.0)
            return base_lr - (base_lr - min_learning_rate) * progress
        else:
            # Constant：保持base_lr不变
            return base_lr

def apply_learning_rate(step: int):
    """应用学习率到优化器
    
    Args:
        step: 当前训练步数
    """
    lr = get_learning_rate(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

# 打印学习率调度器配置
print(f"[Info] LR Scheduler: {lr_scheduler_type}, Warmup: {warmup_steps} steps, "
      f"Total Steps: {total_training_steps}, Min LR: {min_learning_rate:.2e}", flush=True)

# 初始化强化学习模块
if device.type != "meta":
    reward_model = SelfRewardModel(device)
    ppo_trainer = LightweightPPO(
        model=model,
        reward_model=reward_model,
        device=device,
        learning_rate=float(CONFIG.get("ppo_learning_rate", 5e-7)),
        min_learning_rate=float(CONFIG.get("ppo_min_learning_rate", 1e-8)),
        warmup_steps=int(CONFIG.get("ppo_warmup_steps", 200)),
        total_training_steps=int(CONFIG.get("total_training_steps", 30000)),
        clip_ratio=0.2,
        entropy_coef=0.02,
        gamma=0.99,
        ppo_epochs=int(CONFIG.get("ppo_epochs", 2)),
        mini_batch_num=int(CONFIG.get("ppo_mini_batch_num", 4)),
    )
    print("[Info] Self-reward model and RL modules initialized.", flush=True)



def auto_compress_trigger(history_tensor: torch.Tensor, attn_weights: torch.Tensor = None) -> bool:
    """无标记自动触发压缩：长度/注意力熵双判断"""
    seq_len = history_tensor.numel()
    compress_trigger_len = int(CONFIG.get("compress_trigger_len", 512))
    compress_trigger_entropy = float(CONFIG.get("compress_trigger_entropy", 0.8))
    
    if seq_len > compress_trigger_len:
        return True
    
    if attn_weights is not None:
        attn_soft = torch.softmax(attn_weights, dim=-1)
        entropy = -torch.sum(attn_soft * torch.log(attn_soft + 1e-8), dim=-1).mean()
        return entropy > compress_trigger_entropy
    
    return False


def _prepare_training_data(ask_text: str, answer_text: str, hist_context: str = None):
    """准备单个样本的训练数据（使用安全的段构建方式）"""
    if ask_text is None or answer_text is None:
        return None, None, None

    ask_tensor = TextTokenizer.encode(ask_text).to(device)
    answer_tensor = TextTokenizer.encode(answer_text).to(device)

    if answer_tensor.numel() == 0:
        return None, None, None

    # 【新增】检查序列长度，防止长文本显存爆炸
    # 现在优先使用检索/向量索引处理超长历史，避免直接截断或跳过样本
    estimated_total_len = ask_tensor.numel() + answer_tensor.numel() + 10  # +10为特殊token
    if hist_context:
        estimated_total_len += TextTokenizer.encode(hist_context).numel() + 10

    if estimated_total_len > 4096:
        # 当序列非常长时：仅在 GPU 显存占用接近阈值时才进行向量压缩并卸载到CPU/磁盘，
        # 否则保留原始历史（避免不必要的压缩开销）
        logging.debug(f"样本估算过长（estimated_len={estimated_total_len}），评估是否需要压缩")
        try:
            # 仅在显存占用足够高时触发压缩
            mem_ratio = _get_gpu_memory_ratio(device)
            mem_thresh = float(CONFIG.get("compress_on_memory_ratio", 0.9))
            logging.debug(f"当前GPU显存占用比={mem_ratio:.3f}, 阈值={mem_thresh}")
            if mem_ratio >= mem_thresh and hist_context and hist_context.strip():
                history_tokens = TextTokenizer.encode(hist_context).to(device)
                # 计算压缩向量（在模型当前device上计算），然后卸载到CPU并保存到磁盘临时文件
                with torch.no_grad():
                    comp = model.compress_history_vectors(history_tokens)
                comp_cpu = comp.cpu()
                import time
                ts = int(time.time() * 1000)
                path = f"compressed_history_{ts}.pt"
                torch.save({"vectors": comp_cpu}, path)
                logging.info(f"历史上下文已压缩并卸载到 {path} （shape={tuple(comp_cpu.shape)})")
                # 释放GPU上的历史tokens并清理缓存
                try:
                    del history_tokens
                except Exception:
                    pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # 不把完整历史放回训练序列；使用空历史占位符
                hist_context = ""
            else:
                logging.debug("未达到显存压缩阈值，保留原始历史（暂不压缩）")
        except Exception as e:
            logging.warning(f"历史压缩或卸载失败，继续使用原始历史（可能风险OOM）: {e}")

    segments: list = []

    if hist_context is not None and hist_context.strip():
        history_tensor = TextTokenizer.encode(hist_context).to(device)

        # 如果当前 GPU 显存已经接近阈值，则上游应已在 estimated_total_len 分支
        # 触发压缩并卸载；此处仅检查是否仍需清理历史以避免混合输入问题
        if auto_compress_trigger(history_tensor):
            logging.debug(
                f"检测到高显存占用（seq_len={history_tensor.numel()}），假定已在上游执行压缩并卸载，忽略原始历史"
            )
            # 清除历史以避免在训练序列中混合连续向量
            history_tensor = torch.tensor([], dtype=torch.long, device=device)

        segments = [
            (TextTokenizer.START_GENERATION_TOKEN, False),
            (history_tensor, False),
            (TextTokenizer.END_GENERATION_TOKEN, False),
            (TextTokenizer.HISTORY_CONTEXT_START_TOKEN, False),
            (ask_tensor, False),
            (TextTokenizer.START_GENERATION_TOKEN, False),
            (answer_tensor, True),
            (TextTokenizer.END_GENERATION_TOKEN, True),
        ]
        preview = torch.cat(
            [answer_tensor, torch.tensor([TextTokenizer.END_GENERATION_TOKEN], device=device)]
        )
    else:
        segments = [
            (ask_tensor, False),
            (TextTokenizer.START_GENERATION_TOKEN, False),
            (answer_tensor, True),
            (TextTokenizer.END_GENERATION_TOKEN, True),
        ]
        preview = torch.cat(
            [answer_tensor, torch.tensor([TextTokenizer.END_GENERATION_TOKEN], device=device)]
        )

    train_tensor, target_mask = _build_train_sequence(segments)
    return train_tensor, target_mask, preview


def train(ask: str = None, think: str = None, answer: str = None, history_context: str = None) -> None:
    """单步训练函数
    
    Args:
        ask: 问题文本
        think: 思维链/推理过程（可选，用于CoT训练）
        answer: 答案文本
        history_context: 历史对话上下文
    """
    model.train()
    
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

        text_tensor = TextTokenizer.encode(answer).to(device)
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
            
            ask_tensor = TextTokenizer.encode(ask).to(device)
            think_tensor = TextTokenizer.encode(think).to(device)
            answer_tensor = TextTokenizer.encode(answer).to(device)
            
            if history_context:
                history_tensor = TextTokenizer.encode(history_context).to(device)
                train_tensor, target_mask = _build_train_sequence([
                    (TextTokenizer.START_GENERATION_TOKEN, False),
                    (history_tensor, False),
                    (TextTokenizer.END_GENERATION_TOKEN, False),
                    (TextTokenizer.HISTORY_CONTEXT_START_TOKEN, False),
                    (ask_tensor, False),
                    (TextTokenizer.START_GENERATION_TOKEN, False),
                    (TextTokenizer.THINK_START_TOKEN, False),
                    (think_tensor, True),
                    (TextTokenizer.THINK_END_TOKEN, True),
                    (answer_tensor, True),
                    (TextTokenizer.END_GENERATION_TOKEN, True),
                ])
                preview = torch.cat([think_tensor, answer_tensor])
                _run_train_step(train_tensor, target_mask, preview, show_preview=False)
            else:
                train_tensor, target_mask = _build_train_sequence([
                    (ask_tensor, False),
                    (TextTokenizer.START_GENERATION_TOKEN, False),
                    (TextTokenizer.THINK_START_TOKEN, False),
                    (think_tensor, True),
                    (TextTokenizer.THINK_END_TOKEN, True),
                    (answer_tensor, True),
                    (TextTokenizer.END_GENERATION_TOKEN, True),
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


def _apply_top_k_top_p(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """Top-K + Top-P 组合采样 (业界黄金标准)
    
    先用Top-K过滤极端低概率token，再用Top-P动态调整候选集大小。
    """
    # Top-K: 只保留概率最高的K个token
    if 0 < top_k < logits.size(-1):
        top_k_values, _ = torch.topk(logits, top_k)
        threshold = top_k_values[..., -1]
        logits = torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)
    
    # Top-P (Nucleus): 保留累积概率达到P的最小集合
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        
        # 标记需要移除的token
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        # 反向映射到原始顺序
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        logits = torch.where(indices_to_remove, torch.full_like(logits, float("-inf")), logits)
    
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


def generation(text: str, history_context: str = None, max_generate_tokens: int|None = None, thinking_available: bool = True) -> str:
    """生成函数 (增强版: Top-K + Top-P + 重复惩罚 + 重复检测停止)
    
    Args:
        text: 输入文本/问题
        history_context: 历史上下文(可选)
        max_generate_tokens: 最大生成token数
        thinking_available: 是否启用思维链生成（默认True）
    
    Returns:
        生成的文本
    """
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RESET = '\033[0m'
    
    if not text or not isinstance(text, str):
        return "无效输入"
    
    # 【新增】如果未指定max_generate_tokens，使用配置中的默认值
    if max_generate_tokens is None:
        max_generate_tokens = int(CONFIG.get("max_generation_len", 512))
    
    model.eval()
    output_text = ""

    if history_context and history_context.strip():
        history_tensor = TextTokenizer.encode(history_context).to(device)
        
        # 【修复】与_prepare_training_data保持一致：跳过压缩触发的样本
        # 原因：compress_history_vectors返回2维连续向量，与1维token序列不兼容
        # 直接拼接会导致"Tensors must have same number of dimensions: got 1 and 2"错误
        if auto_compress_trigger(history_tensor):
            logging.warning(f"生成时历史上下文过长（seq_len={history_tensor.numel()}），"
                           f"跳过压缩，使用原始序列")
            # 不压缩，直接使用原始token序列（可能会被模型的最大长度限制截断）
        
        text_tensor = TextTokenizer.encode(text).to(device)
        
        # 统一处理：始终使用1维token序列
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

    # 【新增】检查prompt长度，防止输入过长
    max_seq_len = int(CONFIG.get("max_seq_len", 1024))
    if prompt.numel() > max_seq_len:
        logging.warning(f"生成时prompt过长（seq_len={prompt.numel()}），截断到{max_seq_len}")
        prompt = prompt[:max_seq_len]

    print("\n---Generated reply:", flush=True)

    min_new_tokens = 1
    max_generate_tokens = max(1, int(max_generate_tokens))
    
    # 读取采样参数
    top_k = int(CONFIG.get("top_k", 0))
    top_p = float(CONFIG.get("top_p", 1.0))
    repetition_penalty = float(CONFIG.get("repetition_penalty", 1.0))
    repetition_stop_threshold = int(CONFIG.get("repetition_stop_threshold", 5))

    with torch.inference_mode():
        thinking_started = False
        if thinking_available:
            has_think_token = (prompt == TextTokenizer.THINK_START_TOKEN).any()
            if has_think_token:
                # 已包含THINK_START_TOKEN，直接标记为已开始，不再追加
                thinking_started = True
            else:
                # 未包含，追加THINK_START_TOKEN并标记为已开始
                thinking_started = True
                think_start_tensor = torch.tensor([TextTokenizer.THINK_START_TOKEN], device=device)
                prompt = torch.cat([prompt, think_start_tensor])
        
        result = model(prompt, use_cache=True)
        if isinstance(result, tuple):
            logits, past_key_values = result
        else:
            logits = result

        step = 0
        
        # 【修复】使用Counter记录已生成token的频率，实现基于频率的重复惩罚
        generated_tokens = Counter()
        
        while step < max_generate_tokens:
            try:
                next_logits = logits[-1]
                if step < min_new_tokens:
                    next_logits = next_logits.clone()
                    next_logits[TextTokenizer.END_GENERATION_TOKEN] = float("-inf")

                # 【修复】应用基于频率的 repetition_penalty
                # 根据Hugging Face标准实现：token出现频率越高，惩罚越强
                if repetition_penalty > 1.0 and len(generated_tokens) > 0:
                    for token_id, count in generated_tokens.items():
                        if token_id < next_logits.size(0):  # 确保token_id在有效范围内
                            # 频率越高，惩罚指数增长：penalty = repetition_penalty ^ count
                            penalty = repetition_penalty ** count
                            if next_logits[token_id] > 0:
                                next_logits[token_id] /= penalty
                            else:
                                next_logits[token_id] *= penalty

                # 1️⃣ 应用 Top-K + Top-P 组合采样
                if top_k > 0 or top_p < 1.0:
                    next_logits = _apply_top_k_top_p(next_logits, top_k, top_p)

                probs = torch.softmax(next_logits / CONFIG["temperature"], dim=-1)
                index = int(torch.multinomial(probs, 1).item())

                should_skip_output = False
                
                if index == TextTokenizer.THINK_END_TOKEN:
                    if thinking_available and thinking_started:
                        thinking_started = False
                        print(f"\n{GREEN}", end="", flush=True)
                        should_skip_output = True
                    else:
                        break
                
                elif index == TextTokenizer.END_GENERATION_TOKEN:
                    # 【修复】思维阶段触发结束token，直接结束生成（移除强制切换逻辑）
                    break

                elif index == TextTokenizer.THINK_START_TOKEN:
                    if thinking_available and not thinking_started:
                        thinking_started = True
                        should_skip_output = True
                    elif not thinking_available:
                        should_skip_output = True

                # 2️⃣ 重复检测停止（业界标准方案）
                if index not in (
                    TextTokenizer.THINK_START_TOKEN,
                    TextTokenizer.THINK_END_TOKEN,
                    TextTokenizer.START_GENERATION_TOKEN,
                    TextTokenizer.END_GENERATION_TOKEN,
                ):
                    if step >= 5:  # 至少生成5个token后启用检测
                        should_stop, pattern = _check_repetition_stop(list(generated_tokens) + [index], repetition_stop_threshold)
                        if should_stop:
                            print(f"\n[Stop] 检测到重复模式({pattern})，提前结束", flush=True)
                            break

                if not should_skip_output:
                    decoded_piece = TextTokenizer.decode(torch.tensor([index]))
                    
                    if decoded_piece:
                        if thinking_started:
                            print(f"{BLUE}{decoded_piece}{RESET}", end="", flush=True)
                        else:
                            print(f"{GREEN}{decoded_piece}{RESET}", end="", flush=True)
                        
                        output_text += decoded_piece

                # 【修复】将生成的token添加到Counter中用于frequency_penalty
                if index not in (
                    TextTokenizer.THINK_START_TOKEN,
                    TextTokenizer.THINK_END_TOKEN,
                    TextTokenizer.START_GENERATION_TOKEN,
                    TextTokenizer.END_GENERATION_TOKEN,
                ):
                    generated_tokens[index] += 1

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
                
                step += 1
            except Exception as e:
                print(f"Error during generation: {e}", flush=True)
                break
        
        # 【新增】生成完成后清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return output_text


def _run_train_step(train_tensor: torch.Tensor, target_mask: torch.Tensor, preview: torch.Tensor, show_preview: bool = True, preview_color: str = None) -> float:
    """执行单步训练
    
    Args:
        train_tensor: 训练张量
        target_mask: 目标掩码
        preview: 预览张量
        show_preview: 是否显示预览输出(默认True,QA模式下可设为False避免重复)
        preview_color: 预览文本颜色(可选)
    
    Returns:
        当前训练步骤的损失值
    """
    global training_rounds
    
    model.train()
    
    # 【新增】显存监控：每100步输出一次显存使用情况
    if training_rounds > 0 and training_rounds % 100 == 0 and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[Memory] Step {training_rounds}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB", flush=True)
        
        # 【新增】如果reserved显存过高，定期清理缓存
        if reserved > 5.0:  # 超过5GB时清理
            torch.cuda.empty_cache()
            print(f"[Memory] Cleared GPU cache", flush=True)
    
    if (training_rounds % GRADIENT_ACCUMULATION_STEPS) == 0:
        optimizer.zero_grad(set_to_none=True)

    try:
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
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
                masked_logits = logits[:-1][target_mask[1:]]
                masked_targets = train_tensor[1:][target_mask[1:]]
                
                if len(masked_logits) > 0 and len(masked_targets) > 0:
                    if torch.isnan(masked_logits).any() or torch.isinf(masked_logits).any():
                        print(f"[Warning] NaN/Inf in logits. "
                              f"train_tensor range: [{train_tensor.min()}, {train_tensor.max()}], "
                              f"seq_len: {len(train_tensor)}, "
                              f"preview: {TextTokenizer.decode(preview[:50])[:100]}", flush=True)
                        return float('inf')
                    
                    if torch.isnan(masked_targets).any() or torch.isinf(masked_targets).any():
                        print(f"[Warning] NaN or Inf detected in targets, skipping this step", flush=True)
                        return float('inf')
                    
                    loss = loss_func(masked_logits, masked_targets)
                    
                    if torch.isnan(loss):
                        print(f"[Warning] NaN loss detected, skipping this step", flush=True)
                        return float('inf')
                else:
                    loss = torch.tensor(0.0, device=device)
            else:
                loss = torch.tensor(0.0, device=device)
            
            # 【修复】损失缩放，适配梯度累积
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            
            record_loss(loss.item())

        # 检查损失是否有效
        if not torch.isnan(loss) and not torch.isinf(loss):
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    # 【修复】完善的NaN梯度处理流程
                    optimizer.zero_grad(set_to_none=True)
                    
                    # 更新scaler状态，避免影响后续步骤
                    scaler.update()
                    
                    # 检查并清理模型参数中的NaN
                    nan_params = 0
                    for param in model.parameters():
                        if param.grad is not None and torch.isnan(param.grad).any():
                            param.grad = None
                            nan_params += 1
                    
                    print(f"[Warning] NaN/Inf gradient detected (grad_norm={grad_norm:.4f}), "
                          f"cleaned {nan_params} parameter gradients, skipping optimizer step", flush=True)
                    return float('inf')
                
                if (training_rounds + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    scaler.step(optimizer)
                    scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    # 【修复】完善的NaN梯度处理（无AMP场景）
                    optimizer.zero_grad(set_to_none=True)
                    
                    # 检查并清理模型参数中的NaN
                    nan_params = 0
                    for param in model.parameters():
                        if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                            param.grad = None
                            nan_params += 1
                    
                    print(f"[Warning] NaN/Inf gradient detected (grad_norm={grad_norm:.4f}), "
                          f"cleaned {nan_params} parameter gradients, skipping optimizer step", flush=True)
                    return float('inf')
                
                if (training_rounds + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    optimizer.step()
        else:
            print(f"[Warning] Invalid loss detected: {loss}, skipping optimizer step", flush=True)
            return float('inf')
        
        training_rounds += 1
        
        # 【学习率调度】在每个训练步后更新学习率
        if (training_rounds % GRADIENT_ACCUMULATION_STEPS) == 0:
            current_lr = apply_learning_rate(training_rounds)
            # 每100步打印一次学习率信息
            if training_rounds % 100 == 0:
                print(f"[Info] Step {training_rounds}, Current LR: {current_lr:.2e}, "
                      f"Base LR: {base_lr:.2e}, Min LR: {min_learning_rate:.2e}", flush=True)

        if show_preview:
            try:
                decoded_preview = TextTokenizer.decode(preview[preview != 0])
                RESET = '\033[0m'
                if preview_color:
                    print(f"{preview_color}{decoded_preview}{RESET}", end="", flush=True)
                else:
                    print(decoded_preview, end="", flush=True)
            except Exception as e:
                print(f"[Warning] Failed to decode preview: {e}", flush=True)
            print("", flush=True)
        
        return loss.item()
    
    except RuntimeError as e:
        # 【修复】捕获CUDA运行时错误
        error_msg = str(e)
        if "CUDA" in error_msg or "cuda" in error_msg.lower():
            print(f"[CUDA Error] CUDA运行时错误: {e}", flush=True)
            print(f"[CUDA Error] 尝试恢复CUDA状态...", flush=True)
            
            # 清理优化器状态和梯度
            optimizer.zero_grad(set_to_none=True)
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    print(f"[CUDA Error] GPU缓存已清理，跳过当前样本", flush=True)
                except Exception as cleanup_error:
                    print(f"[CUDA Error] 清理GPU缓存失败: {cleanup_error}", flush=True)
            
            return float('inf')
        else:
            # 其他RuntimeError
            print(f"[RuntimeError] {e}, skipping this sample", flush=True)
            return float('inf')
    except Exception as e:
        # 【修复】捕获所有其他异常
        print(f"[Error] 训练步骤发生未知错误: {e}, skipping this sample", flush=True)
        return float('inf')
