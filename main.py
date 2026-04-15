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
    """单步训练函数"""
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
        _run_train_step(train_tensor, target_mask, preview)
        return

    # QA训练模式
    print(f"\n---Train question:\n{ask}", flush=True)
    print("\n---Train answer:", flush=True)
    
    train_tensor, target_mask, preview = _prepare_training_data(ask, answer, history_context)
    if train_tensor is None:
        return
    
    _run_train_step(train_tensor, target_mask, preview)


def _run_train_step(
    train_tensor: torch.Tensor,
    target_mask: torch.Tensor,
    preview: torch.Tensor,
) -> None:
    """执行单步训练"""
    if train_tensor.numel() < 2:
        return

    prompt = train_tensor[:-1]
    targets = train_tensor[1:]
    loss_mask = target_mask[1:] & (targets != 0)
    if loss_mask.numel() == 0 or not bool(loss_mask.any()):
        return

    model.train()
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
        return loss

    try:
        if use_amp:
            with torch.amp.autocast('cuda'):
                loss = compute_loss()
                if loss is None:
                    return
                
                # 【增强】检测NaN或Inf
                loss_value = loss.item()
                if not torch.isfinite(loss).item():
                    print(f"[WARNING] Loss is NaN or Inf! Skipping this sample.", flush=True)
                    return
                
                record_loss(loss_value)
            
            scaler.scale(loss).backward()
            
            # 【修复】梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = compute_loss()
            if loss is None:
                return
            
            loss_value = loss.item()
            if not torch.isfinite(loss).item():
                print(f"[WARNING] Loss is NaN or Inf! Skipping this sample.", flush=True)
                return
            
            record_loss(loss_value)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[ERROR] CUDA OOM detected! Clearing cache and skipping sample...", flush=True)
            torch.cuda.empty_cache()
            return
        else:
            raise
    
    try:
        print(TextTokenizer.decode(preview[preview != 0]), end="", flush=True)
    except Exception as e:
        print(e, flush=True)
    print("", flush=True)


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
