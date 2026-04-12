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
scaler = torch.amp.GradScaler('cuda')
use_amp = torch.cuda.is_available() and torch.cuda.get_device_capability(device)[0] >= 7

print(f"Using device: {device}", flush=True)
print(f"AMP enabled: {use_amp}", flush=True)
model = _load_model()

total_params = sum(param.numel() for param in model.parameters())
print(f"模型参数: {total_params / 1e+8}亿", flush=True)

loss_func = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# ==================== 梯度累积配置 ====================
gradient_accumulation_steps = int(CONFIG.get("gradient_accumulation_steps", 1))
accumulation_counter = 0
accumulated_loss = 0.0

print(f"Gradient accumulation steps: {gradient_accumulation_steps}", flush=True)
if gradient_accumulation_steps > 1:
    print(f"Effective batch size increased by {gradient_accumulation_steps}x for better MoE utilization", flush=True)

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


@overload
def train(ask: str, answer: str) -> None: ...


@overload
def train(text: str) -> None: ...


def train(ask: str, answer: str = "") -> None:
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
            result = model(prompt)
            if isinstance(result, tuple):
                logits, aux_loss = result
            else:
                logits = result
                aux_loss = torch.tensor(0.0, device=device)

            if logits.dim() == 1:
                logits = logits.unsqueeze(0)

            masked_logits = logits[loss_mask]
            masked_targets = targets[loss_mask]
            if masked_targets.numel() == 0:
                return None
            
            if masked_logits.dim() == 1:
                masked_logits = masked_logits.unsqueeze(0)
                masked_targets = masked_targets.unsqueeze(0)

            main_loss = loss_func(masked_logits, masked_targets)
            aux_loss_weight = 0.01
            total_loss = main_loss + aux_loss_weight * aux_loss
            
            # 梯度累积：将损失除以累积步数
            if gradient_accumulation_steps > 1:
                total_loss = total_loss / gradient_accumulation_steps
            
            return total_loss

        if use_amp:
            with torch.amp.autocast('cuda'):
                loss = compute_loss()
                if loss is None:
                    return
                record_loss(loss.item() * gradient_accumulation_steps)  # 记录真实loss
                accumulated_loss += loss.item() * gradient_accumulation_steps
            
            scaler.scale(loss).backward()
        else:
            loss = compute_loss()
            if loss is None:
                return
            
            record_loss(loss.item() * gradient_accumulation_steps)  # 记录真实loss
            accumulated_loss += loss.item() * gradient_accumulation_steps
            loss.backward()
        
        accumulation_counter += 1
        
        # 只在累积足够步数后才更新参数
        if accumulation_counter % gradient_accumulation_steps == 0:
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
                avg_loss = accumulated_loss / gradient_accumulation_steps
                print(f"[LR] Current learning rate: {current_lr:.6f} (scheduler: {current_scheduler_type}, effective_batch={gradient_accumulation_steps})", flush=True)
            
            accumulated_loss = 0.0

        try:
            print(TextTokenizer.decode(preview[preview != 0]), end="", flush=True)
        except Exception as e:
            print(e, flush=True)
        print("", flush=True)

    if not answer:
        print(f"\n---Single text training:\n{ask}", flush=True)
        print("\n---Learning tokens:", flush=True)

        text_tensor = TextTokenizer.encode(ask).to(device)
        if text_tensor.numel() < 2:
            return

        train_tensor = torch.cat(
            [
                torch.tensor([TextTokenizer.START_TOKEN], device=device),
                text_tensor,
                torch.tensor([TextTokenizer.END_TOKEN], device=device),
            ]
        )
        target_mask = torch.ones(train_tensor.numel(), dtype=torch.bool, device=device)
        preview = train_tensor
        _run_train_step(train_tensor, target_mask, preview)
        return

    print(f"\n---Train question:\n{ask}", flush=True)
    print("\n---Train answer:", flush=True)

    ask_tensor = TextTokenizer.encode(ask).to(device)
    answer_tensor = TextTokenizer.encode(answer).to(device)
    if answer_tensor.numel() == 0:
        return

    train_tensor = torch.cat(
        [
            ask_tensor,
            torch.tensor([TextTokenizer.START_TOKEN], device=device),
            answer_tensor,
            torch.tensor([TextTokenizer.END_TOKEN], device=device),
        ]
    )
    target_mask = torch.cat(
        [
            torch.zeros(ask_tensor.numel() + 1, dtype=torch.bool, device=device),
            torch.ones(answer_tensor.numel() + 1, dtype=torch.bool, device=device),
        ]
    )
    preview = torch.cat(
        [answer_tensor, torch.tensor([TextTokenizer.END_TOKEN], device=device)]
    )
    _run_train_step(train_tensor, target_mask, preview)


def generation(text: str, max_generate_tokens: int = 256) -> str:
    model.eval()
    output_text = ""

    prompt = torch.cat(
        [
            TextTokenizer.encode(text).to(device),
            torch.tensor([TextTokenizer.START_TOKEN], device=device),
        ]
    )

    print("\n---Generated reply:", flush=True)

    min_new_tokens = 1
    max_generate_tokens = max(1, int(max_generate_tokens))

    with torch.inference_mode():
        # 推理时不需要辅助损失
        result = model(prompt, use_cache=True)
        if isinstance(result, tuple):
            if len(result) == 3:
                logits, past_key_values, _ = result
            else:
                logits, past_key_values = result
        else:
            logits = result

        for step in range(max_generate_tokens):
            try:
                next_logits = logits[-1]
                if step < min_new_tokens:
                    next_logits = next_logits.clone()
                    next_logits[TextTokenizer.END_TOKEN] = float("-inf")

                probs = torch.softmax(next_logits / CONFIG["temperature"], dim=-1)
                index = int(torch.multinomial(probs, 1).item())

                if index == TextTokenizer.END_TOKEN:
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
                    if len(result) == 3:
                        logits, past_key_values, _ = result
                    else:
                        logits, past_key_values = result
                else:
                    logits = result
            except Exception as e:
                print(e, flush=True)
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
