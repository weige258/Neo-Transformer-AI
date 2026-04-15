import sys
from typing import overload

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


# 启用自动混合精度训练
scaler = torch.amp.GradScaler()
use_amp = torch.cuda.is_available() and torch.cuda.get_device_capability(device)[0] >= 7

print(f"Using device: {device}", flush=True)
print(f"AMP enabled: {use_amp}", flush=True)
model = _load_model()

total_params = sum(param.numel() for param in model.parameters())
print(f"模型参数: {total_params / 1e+8}亿", flush=True)

loss_func = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)

# ==================== 智能Batch优化配置 ====================
gradient_accumulation_steps = int(CONFIG.get("gradient_accumulation_steps", 1))
accumulation_counter = 0
accumulated_loss = 0.0

# 智能Batch管理
MAX_BATCH_SIZE = 8  # 最大batch size（根据6GB显存设定）
MIN_BATCH_SIZE = 1  # 最小batch size
CURRENT_BATCH_SIZE = 1  # 当前batch size
BATCH_GROWTH_FACTOR = 2.0  # batch size增长因子（提高到2倍加速增长）
BATCH_SHRINK_FACTOR = 0.5  # batch size缩减因子
OOM_COUNT = 0  # OOM计数器
SUCCESSFUL_STEPS = 0  # 成功步数计数器
OOM_THRESHOLD = 2  # 连续OOM次数阈值后缩小batch
GROWTH_THRESHOLD = 5  # 降低到5，更快触发batch size增长

print(f"Gradient accumulation steps: {gradient_accumulation_steps}", flush=True)
if gradient_accumulation_steps > 1:
    print(f"Effective batch size increased by {gradient_accumulation_steps}x", flush=True)
print(f"Smart Batch Optimization: enabled (min={MIN_BATCH_SIZE}, max={MAX_BATCH_SIZE}, current={CURRENT_BATCH_SIZE})", flush=True)

# ==================== 学习率调度器配置 ====================
# 配置总训练步数(根据实际情况调整)
TOTAL_TRAINING_STEPS = 100000
MIN_LR = 1e-6  # 最小学习率
PATIENCE = 10  # ReduceLROnPlateau的耐心值

# 1. Cosine Annealing 余弦退火调度器
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=TOTAL_TRAINING_STEPS,
    eta_min=MIN_LR
)

# 2. ReduceLROnPlateau 根据loss自动调整学习率
plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',      # 监控loss,越小越好
    factor=0.5,      # 每次减少到原来的0.5倍
    patience=PATIENCE,  # 等待PATIENCE个step没有改善就降低学习率
    min_lr=MIN_LR    # 最小学习率
)

# 当前使用的调度器类型 ('cosine' 或 'plateau')
current_scheduler_type = 'cosine'  # 默认使用余弦退火

print(f"Learning rate schedulers initialized:", flush=True)
print(f"  - Cosine Annealing: T_max={TOTAL_TRAINING_STEPS}, min_lr={MIN_LR}", flush=True)
print(f"  - ReduceLROnPlateau: factor=0.5, patience={PATIENCE}, min_lr={MIN_LR}", flush=True)
print(f"  - Current scheduler: {current_scheduler_type}", flush=True)


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


def train(ask: str = None, answer: str = None, history_context: str = None) -> None:
    global CURRENT_BATCH_SIZE, OOM_COUNT, SUCCESSFUL_STEPS
    
    def _adjust_batch_size(oom_detected: bool = False):
        """实时智能调整batch size"""
        global CURRENT_BATCH_SIZE, OOM_COUNT, SUCCESSFUL_STEPS
        
        if oom_detected:
            OOM_COUNT += 1
            SUCCESSFUL_STEPS = 0
            
            if OOM_COUNT >= OOM_THRESHOLD:
                old_batch_size = CURRENT_BATCH_SIZE
                CURRENT_BATCH_SIZE = max(MIN_BATCH_SIZE, int(CURRENT_BATCH_SIZE * BATCH_SHRINK_FACTOR))
                if CURRENT_BATCH_SIZE != old_batch_size:
                    print(f"[BATCH] OOM detected! Reducing batch size: {old_batch_size} → {CURRENT_BATCH_SIZE}", flush=True)
                OOM_COUNT = 0  # 重置计数器
        else:
            OOM_COUNT = 0
            SUCCESSFUL_STEPS += 1
            
            # 连续成功后尝试增大batch size
            if SUCCESSFUL_STEPS >= GROWTH_THRESHOLD and CURRENT_BATCH_SIZE < MAX_BATCH_SIZE:
                old_batch_size = CURRENT_BATCH_SIZE
                CURRENT_BATCH_SIZE = min(MAX_BATCH_SIZE, int(CURRENT_BATCH_SIZE * BATCH_GROWTH_FACTOR))
                if CURRENT_BATCH_SIZE != old_batch_size:
                    print(f"[BATCH] GPU stable. Increasing batch size: {old_batch_size} → {CURRENT_BATCH_SIZE}", flush=True)
                SUCCESSFUL_STEPS = 0  # 重置计数器
    
    def _run_train_step(
        train_tensor: torch.Tensor,
        target_mask: torch.Tensor,
        preview: torch.Tensor,
    ) -> None:
        global accumulation_counter, accumulated_loss
        
        if train_tensor.numel() < 2:
            return

        prompt = train_tensor[:-1]
        targets = train_tensor[1:]
        loss_mask = target_mask[1:] & (targets != 0)
        if loss_mask.numel() == 0 or not bool(loss_mask.any()):
            return

        model.train()
        
        # 如果不是第一步，不清除梯度（用于累积）
        should_zero_grad = (accumulation_counter % gradient_accumulation_steps == 0)
        if should_zero_grad:
            optimizer.zero_grad(set_to_none=True)

        def compute_loss():
            logits = model(prompt)
            if isinstance(logits, tuple):
                logits = logits[0]

            if logits.dim() == 1:
                logits = logits.unsqueeze(0)

            masked_logits = logits[loss_mask]
            masked_targets = targets[loss_mask]
            if masked_targets.numel() == 0:
                return None
            
            if masked_logits.dim() == 1:
                masked_logits = masked_logits.unsqueeze(0)
                masked_targets = masked_targets.unsqueeze(0)

            loss = loss_func(masked_logits, masked_targets)
            
            # 梯度累积：将损失除以累积步数
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            
            return loss

        try:
            if use_amp:
                with torch.amp.autocast('cuda'):
                    loss = compute_loss()
                    if loss is None:
                        return
                    
                    # 检测NaN或Inf
                    if not torch.isfinite(loss).item():
                        print(f"[WARNING] Loss is NaN or Inf! Skipping this batch.", flush=True)
                        return
                    
                    record_loss(loss.item() * gradient_accumulation_steps)  # 记录真实loss
                    accumulated_loss += loss.item() * gradient_accumulation_steps
                
                scaler.scale(loss).backward()
            else:
                loss = compute_loss()
                if loss is None:
                    return
                
                # 检测NaN或Inf
                if not torch.isfinite(loss).item():
                    print(f"[WARNING] Loss is NaN or Inf! Skipping this batch.", flush=True)
                    return
                
                record_loss(loss.item() * gradient_accumulation_steps)  # 记录真实loss
                accumulated_loss += loss.item() * gradient_accumulation_steps
                loss.backward()
            
            # 成功完成前向和反向传播
            _adjust_batch_size_from_global(oom_detected=False)
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # OOM错误处理
                print(f"[BATCH] CUDA OOM detected! Clearing cache...", flush=True)
                torch.cuda.empty_cache()
                _adjust_batch_size_from_global(oom_detected=True)
                return  # 跳过当前批次
            else:
                raise  # 其他错误继续抛出
        
        accumulation_counter += 1
        
        # 只在累积足够步数后才更新参数
        if accumulation_counter % gradient_accumulation_steps == 0:
            # 梯度裁剪：防止梯度爆炸
            if use_amp:
                scaler.unscale_(optimizer)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            # 更新学习率调度器
            if current_scheduler_type == 'cosine':
                cosine_scheduler.step()
            elif current_scheduler_type == 'plateau':
                plateau_scheduler.step(accumulated_loss / gradient_accumulation_steps)
            
            # 打印当前学习率(每100步打印一次)
            if not hasattr(_run_train_step, '_step_counter'):
                _run_train_step._step_counter = 0
            _run_train_step._step_counter += 1
            
            if _run_train_step._step_counter % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                
                # 显存监控
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**2
                    reserved = torch.cuda.memory_reserved() / 1024**2
                    max_allocated = torch.cuda.max_memory_allocated() / 1024**2
                    
                    print(f"[LR] Current learning rate: {current_lr:.6f} (scheduler: {current_scheduler_type}, effective_batch={gradient_accumulation_steps})", flush=True)
                    print(f"[GPU] Memory - Allocated: {allocated:.1f}MB, Reserved: {reserved:.1f}MB, Peak: {max_allocated:.1f}MB, Batch: {CURRENT_BATCH_SIZE}", flush=True)
                else:
                    print(f"[LR] Current learning rate: {current_lr:.6f} (scheduler: {current_scheduler_type}, effective_batch={gradient_accumulation_steps})", flush=True)
            
            accumulated_loss = 0.0

        try:
            print(TextTokenizer.decode(preview[preview != 0]), end="", flush=True)
        except Exception as e:
            print(e, flush=True)
        print("", flush=True)

    # 单样本训练模式（保持向后兼容）
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
        _run_train_step(train_tensor, target_mask, preview)
        return

    # QA训练模式
    print(f"\n---Train question:\n{ask}", flush=True)
    print("\n---Train answer:", flush=True)
    
    train_tensor, target_mask, preview = _prepare_training_data(ask, answer, history_context)
    if train_tensor is None:
        return
    
    _run_train_step(train_tensor, target_mask, preview)


def train_batch(samples: list[tuple[str, str, str | None]]) -> None:
    """
    批量训练接口 - 实时智能Batch优化
    
    Args:
        samples: 列表，每个元素为 (ask, answer, history_context) 元组
    """
    global CURRENT_BATCH_SIZE
    
    if not samples:
        return
    
    print(f"\n[BATCH] Starting batch training with {len(samples)} samples", flush=True)
    print(f"[BATCH] Current batch size: {CURRENT_BATCH_SIZE}", flush=True)
    
    # 准备所有样本数据
    prepared_data = []
    for i, sample in enumerate(samples):
        if len(sample) == 2:
            ask, answer = sample
            history = None
        else:
            ask, answer, history = sample
        
        train_tensor, target_mask, preview = _prepare_training_data(ask, answer, history)
        if train_tensor is not None:
            prepared_data.append((train_tensor, target_mask, preview))
    
    if not prepared_data:
        print("[BATCH] No valid samples to train", flush=True)
        return
    
    print(f"[BATCH] Valid samples: {len(prepared_data)}", flush=True)
    
    # 按当前batch size分组处理
    batch_idx = 0
    for start_idx in range(0, len(prepared_data), CURRENT_BATCH_SIZE):
        end_idx = min(start_idx + CURRENT_BATCH_SIZE, len(prepared_data))
        batch_samples = prepared_data[start_idx:end_idx]
        
        actual_batch_size = len(batch_samples)
        if actual_batch_size < CURRENT_BATCH_SIZE:
            print(f"[BATCH] Processing final mini-batch: {actual_batch_size}/{CURRENT_BATCH_SIZE}", flush=True)
        
        print(f"[BATCH] Processing batch {batch_idx + 1}: samples [{start_idx}:{end_idx}] (size={actual_batch_size})", flush=True)
        
        # 对batch中的每个样本执行训练步骤
        for sample_idx, (train_tensor, target_mask, preview) in enumerate(batch_samples):
            print(f"  [SAMPLE {sample_idx + 1}/{actual_batch_size}] Training...", end="", flush=True)
            _run_single_sample_in_batch(train_tensor, target_mask, preview)
            print(" Done", flush=True)
        
        batch_idx += 1
    
    print(f"[BATCH] Batch training completed. Final batch size: {CURRENT_BATCH_SIZE}", flush=True)


def _run_single_sample_in_batch(
    train_tensor: torch.Tensor,
    target_mask: torch.Tensor,
    preview: torch.Tensor,
) -> None:
    """在batch模式下运行单个样本的训练步骤"""
    global accumulation_counter, accumulated_loss
    
    if train_tensor.numel() < 2:
        return

    prompt = train_tensor[:-1]
    targets = train_tensor[1:]
    loss_mask = target_mask[1:] & (targets != 0)
    if loss_mask.numel() == 0 or not bool(loss_mask.any()):
        return

    model.train()
    
    should_zero_grad = (accumulation_counter % gradient_accumulation_steps == 0)
    if should_zero_grad:
        optimizer.zero_grad(set_to_none=True)

    def compute_loss():
        logits = model(prompt)
        if isinstance(logits, tuple):
            logits = logits[0]

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        masked_logits = logits[loss_mask]
        masked_targets = targets[loss_mask]
        if masked_targets.numel() == 0:
            return None
        
        if masked_logits.dim() == 1:
            masked_logits = masked_logits.unsqueeze(0)
            masked_targets = masked_targets.unsqueeze(0)

        loss = loss_func(masked_logits, masked_targets)
        
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        
        return loss

    try:
        if use_amp:
            with torch.amp.autocast('cuda'):
                loss = compute_loss()
                if loss is None:
                    return
                
                if not torch.isfinite(loss).item():
                    print(f"[WARNING] Loss is NaN or Inf! Skipping this batch.", flush=True)
                    return
                
                record_loss(loss.item() * gradient_accumulation_steps)
                accumulated_loss += loss.item() * gradient_accumulation_steps
            
            scaler.scale(loss).backward()
        else:
            loss = compute_loss()
            if loss is None:
                return
            
            if not torch.isfinite(loss).item():
                print(f"[WARNING] Loss is NaN or Inf! Skipping this batch.", flush=True)
                return
            
            record_loss(loss.item() * gradient_accumulation_steps)
            accumulated_loss += loss.item() * gradient_accumulation_steps
            loss.backward()
        
        # 成功完成
        _adjust_batch_size_from_global(oom_detected=False)
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[BATCH] CUDA OOM detected! Clearing cache...", flush=True)
            torch.cuda.empty_cache()
            _adjust_batch_size_from_global(oom_detected=True)
            return
        else:
            raise
    
    accumulation_counter += 1
    
    if accumulation_counter % gradient_accumulation_steps == 0:
        if use_amp:
            scaler.unscale_(optimizer)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        if current_scheduler_type == 'cosine':
            cosine_scheduler.step()
        elif current_scheduler_type == 'plateau':
            plateau_scheduler.step(accumulated_loss / gradient_accumulation_steps)
        
        if not hasattr(_run_single_sample_in_batch, '_step_counter'):
            _run_single_sample_in_batch._step_counter = 0
        _run_single_sample_in_batch._step_counter += 1
        
        if _run_single_sample_in_batch._step_counter % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**2
                reserved = torch.cuda.memory_reserved() / 1024**2
                max_allocated = torch.cuda.max_memory_allocated() / 1024**2
                
                print(f"[LR] Current learning rate: {current_lr:.6f} (scheduler: {current_scheduler_type}, effective_batch={gradient_accumulation_steps})", flush=True)
                print(f"[GPU] Memory - Allocated: {allocated:.1f}MB, Reserved: {reserved:.1f}MB, Peak: {max_allocated:.1f}MB, Batch: {CURRENT_BATCH_SIZE}", flush=True)
            else:
                print(f"[LR] Current learning rate: {current_lr:.6f}", flush=True)
        
        accumulated_loss = 0.0

    try:
        print(TextTokenizer.decode(preview[preview != 0]), end="", flush=True)
    except Exception as e:
        print(e, flush=True)
    print("", flush=True)


def _adjust_batch_size_from_global(oom_detected: bool = False):
    """从全局作用域调用的batch size调整函数"""
    global CURRENT_BATCH_SIZE, OOM_COUNT, SUCCESSFUL_STEPS
    
    if oom_detected:
        OOM_COUNT += 1
        SUCCESSFUL_STEPS = 0
        
        if OOM_COUNT >= OOM_THRESHOLD:
            old_batch_size = CURRENT_BATCH_SIZE
            CURRENT_BATCH_SIZE = max(MIN_BATCH_SIZE, int(CURRENT_BATCH_SIZE * BATCH_SHRINK_FACTOR))
            if CURRENT_BATCH_SIZE != old_batch_size:
                print(f"[BATCH] OOM detected! Reducing batch size: {old_batch_size} → {CURRENT_BATCH_SIZE}", flush=True)
            OOM_COUNT = 0
    else:
        OOM_COUNT = 0
        SUCCESSFUL_STEPS += 1
        
        if SUCCESSFUL_STEPS >= GROWTH_THRESHOLD and CURRENT_BATCH_SIZE < MAX_BATCH_SIZE:
            old_batch_size = CURRENT_BATCH_SIZE
            CURRENT_BATCH_SIZE = min(MAX_BATCH_SIZE, int(CURRENT_BATCH_SIZE * BATCH_GROWTH_FACTOR))
            if CURRENT_BATCH_SIZE != old_batch_size:
                print(f"[BATCH] GPU stable. Increasing batch size: {old_batch_size} → {CURRENT_BATCH_SIZE}", flush=True)
            SUCCESSFUL_STEPS = 0


def generation(text: str, max_generate_tokens: int|None = None) -> str:
    model.eval()
    output_text = ""

    prompt = torch.cat(
        [
            TextTokenizer.encode(text).to(device),
            torch.tensor([TextTokenizer.START_GENERATION_TOKEN], device=device),
        ]
    )

    print("\n---Generated reply:", flush=True)

    min_new_tokens = 1
    if max_generate_tokens is not None:
        max_generate_tokens = max(1, int(max_generate_tokens))
 
    with torch.inference_mode():
        # 推理时不需要辅助损失
        result = model(prompt, use_cache=True)
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
                prompt = torch.cat([prompt, next_token])
                
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


def switch_scheduler(scheduler_type: str = 'cosine'):
    """
    切换学习率调度器
    
    Args:
        scheduler_type: 'cosine' 或 'plateau'
    """
    global current_scheduler_type
    if scheduler_type in ['cosine', 'plateau']:
        current_scheduler_type = scheduler_type
        print(f"Switched to {scheduler_type} scheduler", flush=True)
    else:
        print(f"Invalid scheduler type: {scheduler_type}. Use 'cosine' or 'plateau'", flush=True)