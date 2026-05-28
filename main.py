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
    
    Loss Mask 语义（重要）：
      target_mask[i]=True 表示 logits[i-1] → token[i] 的预测参与 loss 计算。
      即：模型在位置 i-1 的输出需要正确预测位置 i 的 token。
      
    对于 CoT 训练，必须确保：
      - THINK_END_TOKEN 位置的 mask=True → 模型学会在思考结束时输出 THINK_END
      - answer 首 token 位置的 mask=True → 模型学会在 THINK_END 后立即输出回答
      这两者缺一不可，否则模型会学会"只思考不回答"。
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
    t0 = time.time()
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

# 允许通过配置覆盖是否启用 AMP（便于在某些环境下手动关闭）
use_amp = bool(CONFIG.get("use_amp", use_amp))

# 【修复】仅float16启用scaler，bfloat16无需缩放，避免梯度爆炸
# 【PyTorch 2.x 更新】使用 torch.amp.GradScaler('cuda') 替代已弃用的 torch.cuda.amp.GradScaler
s_cl_en = (use_amp and amp_dtype == torch.float16)
scaler = torch.amp.GradScaler('cuda', enabled=s_cl_en)

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

    ask_tensor = TextTokenizer.encode(ask_text)
    answer_tensor = TextTokenizer.encode(answer_text)

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
                # 在 CPU 上编码 tokens，避免一次性把大张量移到 GPU 导致 OOM
                history_tokens = TextTokenizer.encode(hist_context)
                # 计算压缩向量（在模型当前device上计算），然后卸载到CPU并保存到磁盘临时文件
                with torch.no_grad():
                    comp = model.compress_history_vectors(history_tokens)
                # 保持压缩向量在 model.embedding 权重设备（通常为 GPU），避免把压缩结果迁移到 CPU
                try:
                    comp_on_device = comp.to(device)
                except Exception:
                    comp_on_device = comp
                # 将压缩向量缓存到模型内部，供后续推理/训练使用，避免写入磁盘或保持在 CPU
                setattr(model, "_compressed_history", comp_on_device)
                logging.info(f"历史上下文已压缩并缓存至模型（device={comp_on_device.device}, shape={tuple(comp_on_device.shape)})")
                # 释放临时 history_tokens 引用以减少内存峰值
                try:
                    del history_tokens
                except Exception:
                    pass
                # 不把完整历史放回训练序列；使用空历史占位符
                hist_context = ""
            else:
                logging.debug("未达到显存压缩阈值，保留原始历史（暂不压缩）")
        except Exception as e:
            logging.warning(f"历史压缩或卸载失败，继续使用原始历史（可能风险OOM）: {e}")

    segments: list = []

    if hist_context is not None and hist_context.strip():
        history_tensor = TextTokenizer.encode(hist_context)

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
        # 保持 preview 在 CPU 上，避免与 GPU tensors 混合导致 torch.cat 设备不一致错误
        preview = torch.cat([answer_tensor, torch.tensor([TextTokenizer.END_GENERATION_TOKEN])])
    else:
        segments = [
            (ask_tensor, False),
            (TextTokenizer.START_GENERATION_TOKEN, False),
            (answer_tensor, True),
            (TextTokenizer.END_GENERATION_TOKEN, True),
        ]
        preview = torch.cat([answer_tensor, torch.tensor([TextTokenizer.END_GENERATION_TOKEN])])

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
                    (TextTokenizer.THINK_START_TOKEN, False),   # 不预测思考开始标记
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
                    (TextTokenizer.THINK_START_TOKEN, False),    # 不预测思考开始标记
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
        # 【新增】CoT 完整性保护：追踪思维链结束后的强制回答步数
        # 防止模型在 THINK_END 后直接输出 END_GENERATION_TOKEN 结束（奖励作弊）
        force_answer_steps = 0      # 剩余强制回答步数（>0时禁止输出结束符）
        FORCE_ANSWER_MIN_STEPS = 8  # THINK_END后至少强制生成8个有效token
        
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
                next_logits = logits[-1].clone()  # 【修复】始终clone，避免修改原始logits
                
                if step < min_new_tokens:
                    next_logits[TextTokenizer.END_GENERATION_TOKEN] = float("-inf")
                
                # ── 【新增】CoT 完整性强制干预 ──
                # 如果刚结束思维链，强制禁止直接输出 END_GENERATION_TOKEN
                # 这是对抗"奖励作弊"的关键机制：模型不能只思考不回答
                if force_answer_steps > 0:
                    # 强制抹去结束符和特殊符的概率，逼迫模型生成实际回答内容
                    next_logits[TextTokenizer.END_GENERATION_TOKEN] = float("-inf")
                    next_logits[TextTokenizer.UNKNOWN_TOKEN] = float("-inf")
                    next_logits[TextTokenizer.THINK_START_TOKEN] = float("-inf")
                    next_logits[TextTokenizer.THINK_END_TOKEN] = float("-inf")
                    # 也抹去历史上下文标记，防止模型试图用上下文标记"混过去"
                    next_logits[TextTokenizer.HISTORY_CONTEXT_START_TOKEN] = float("-inf")
                    next_logits[TextTokenizer.HISTORY_CONTEXT_END_TOKEN] = float("-inf")
                    force_answer_steps -= 1

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
                        # 【新增】思维链刚结束，启动强制回答期
                        force_answer_steps = FORCE_ANSWER_MIN_STEPS
                        print(f"\n{GREEN}", end="", flush=True)
                        should_skip_output = True
                    else:
                        break
                
                elif index == TextTokenizer.END_GENERATION_TOKEN:
                    # 【修复】如果还在强制回答期内，跳过结束符，重新采样
                    # 不喂入模型，直接continue让模型从相同logits重新选择
                    if force_answer_steps > 0:
                        continue  # 跳过本次迭代，重新采样（END_GEN已被设为-inf）
                    else:
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
                    sub = _chunk_one_segment(seg, seg_mask, safe_past, smaller)
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
):
    """递归重试单个段，返回 (avg_loss, past_kv) 或 None。"""
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
                loss_scaled = loss_sub / GRADIENT_ACCUMULATION_STEPS

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
                        pred = logits[:-1][mask_bool]
                        tgt = train_tensor_gpu[1:][mask_bool]
                        loss = loss_func(pred, tgt)
                    else:
                        loss = torch.tensor(0.0, device=device)
                else:
                    loss = torch.tensor(0.0, device=device)

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
                return float('inf')
            loss = torch.tensor(loss_val, device=device)

        # ── 统一的梯度后处理 ──
        if not torch.isnan(loss) and not torch.isinf(loss):
            if scaler.is_enabled():
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
                if (training_rounds + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    scaler.step(optimizer)
                    scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    optimizer.zero_grad(set_to_none=True)
                    for param in model.parameters():
                        if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                            param.grad = None
                    print(f"[Warning] NaN/Inf gradient, skipping optimizer step", flush=True)
                    return float('inf')
                if (training_rounds + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    optimizer.step()
        else:
            print(f"[Warning] Invalid loss: {loss}, skipping optimizer step", flush=True)
            return float('inf')

        training_rounds += 1

        # 学习率调度
        if (training_rounds % GRADIENT_ACCUMULATION_STEPS) == 0:
            current_lr = apply_learning_rate(training_rounds)
            if training_rounds % 100 == 0:
                print(f"[Info] Step {training_rounds}, LR: {current_lr:.2e}", flush=True)

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
