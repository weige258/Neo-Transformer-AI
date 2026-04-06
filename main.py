import random
import sys
from typing import Tuple, overload

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


print(f"Using device: {device}", flush=True)
model = _load_model()

total_params = sum(param.numel() for param in model.parameters())
print(f"模型参数: {total_params / 1e+8}亿", flush=True)

loss_func = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)


def _truncate_train_tensor(train_tensor: torch.Tensor, ask_len: int) -> Tuple[torch.Tensor, int]:
    max_train_len = CONFIG["max_length"] + 1
    if train_tensor.numel() <= max_train_len:
        return train_tensor, ask_len

    overflow = train_tensor.numel() - max_train_len
    train_tensor = train_tensor[overflow:]
    ask_len = max(ask_len - overflow, 0)
    return train_tensor, ask_len


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
        if train_tensor.numel() < 2:
            return

        max_train_len = CONFIG["max_length"] + 1
        if train_tensor.numel() > max_train_len:
            train_tensor = train_tensor[-max_train_len:]
            target_mask = target_mask[-max_train_len:]

        prompt = train_tensor[:-1]
        targets = train_tensor[1:]
        loss_mask = target_mask[1:] & (targets != 0)
        if loss_mask.numel() == 0 or not bool(loss_mask.any()):
            return

        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(prompt)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        masked_logits = logits[loss_mask]
        masked_targets = targets[loss_mask]
        if masked_targets.numel() == 0:
            return
        if masked_logits.dim() == 1:
            masked_logits = masked_logits.unsqueeze(0)
            masked_targets = masked_targets.unsqueeze(0)

        loss = loss_func(masked_logits, masked_targets)
        record_loss(loss.item())
        loss.backward()
        optimizer.step()

        try:
            print(TextTokenizer.decode(preview[preview != 0]), end="", flush=True)
        except Exception as e:
            print(e, flush=True)
        print("", flush=True)

    if not answer:
        print(f"\n---Single text training:\n{ask}", flush=True)
        print("\n---Learning tokens:", flush=True)

        text_tensor = TextTokenizer.encode(ask).to(device)
        if text_tensor.numel() == 0:
            return

        train_tensor = torch.cat(
            [
                torch.tensor([TextTokenizer.START_TOKEN], device=device),
                text_tensor,
                torch.tensor([TextTokenizer.END_TOKEN], device=device),
            ]
        )
        target_mask = torch.cat(
            [
                torch.tensor([False], device=device),
                torch.ones(text_tensor.numel() + 1, dtype=torch.bool, device=device),
            ]
        )
        preview = torch.cat(
            [text_tensor, torch.tensor([TextTokenizer.END_TOKEN], device=device)]
        )
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


def generation(text: str) -> str:
    model.eval()
    output_text = ""

    prompt = torch.cat(
        [
            TextTokenizer.encode(text).to(device),
            torch.tensor([TextTokenizer.START_TOKEN], device=device),
        ]
    )
    if prompt.numel() > CONFIG["max_length"]:
        prompt = prompt[-CONFIG["max_length"]:]

    print("\n---Generated reply:", flush=True)

    end_threshold = random.randint(3, 5)
    end_hits = 0

    with torch.inference_mode():
        logits, past_key_values = model(prompt, use_cache=True)

        for _ in range(CONFIG["max_length"]):
            try:
                next_logits = logits[-1]
                probs = torch.softmax(next_logits / CONFIG["temperature"], dim=-1)
                index = int(torch.multinomial(probs, 1).item())

                if index == TextTokenizer.END_TOKEN:
                    end_hits += 1
                    if end_hits >= end_threshold:
                        break

                if index != TextTokenizer.END_TOKEN:
                    decoded_piece = TextTokenizer.decode(torch.tensor([index]))
                    if decoded_piece:
                        print(decoded_piece, end="", flush=True)
                        output_text += decoded_piece

                next_token = torch.tensor([index], device=device)
                prompt = torch.cat([prompt, next_token])

                if prompt.numel() > CONFIG["max_length"]:
                    prompt = prompt[-CONFIG["max_length"]:]
                    logits, past_key_values = model(prompt, use_cache=True)
                else:
                    logits, past_key_values = model(
                        next_token,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
            except Exception as e:
                print(e, flush=True)
    return output_text
