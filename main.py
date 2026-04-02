import random
import torch
from typing import Tuple
from model import MainModel, CONFIG
from record import record_loss

# Special tokens
START_TOKEN = 1
END_TOKEN = CONFIG["dict_size"] - 1

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Encoding/Decoding functions
# Note: This is a simple ASCII-based encoding. For better performance, consider using a proper tokenizer.
def encode(text: str) -> torch.Tensor:
    """Encode text to tensor of indices"""
    tensor = []
    for letter in text:
        try:
            idx = ord(letter)
            # Ensure index is within the dictionary size
            if 0 < idx < CONFIG["dict_size"] - 1:
                tensor.append(idx)
            else:
                tensor.append(0)  # Unknown token
        except Exception:
            tensor.append(0)  # Unknown token
    return torch.tensor(tensor, dtype=torch.long)

def decode(indices: torch.Tensor) -> str:
    """Decode tensor of indices to text"""
    text = []
    for idx in indices:
        try:
            idx_int = int(idx)
            if idx_int != START_TOKEN and idx_int != END_TOKEN:
                text.append(chr(idx_int))
        except Exception:
            continue
    return "".join(text)

def _load_model() -> MainModel:
    """Load model from disk if possible, otherwise create a new one."""
    try:
        loaded = torch.load(f="model.pth", map_location=device, weights_only=False)
        if isinstance(loaded, dict):
            model = MainModel().to(device)
            model.load_state_dict(loaded)
            print("Loaded model state dict.")
        else:
            model = loaded.to(device)
            print("Loaded full model.")

        # Verify model structure compatibility
        if not hasattr(model, "transformers") or len(model.transformers) == 0 or not hasattr(model.transformers[0], "rms_norm1"):
            print("Detected old model structure; creating new model.")
            model = MainModel().to(device)
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        model = MainModel().to(device)
        print("Created new model.")
        return model

# Model initialization
print(f"Using device: {device}")
model = _load_model()

total_params = sum(param.numel() for param in model.parameters())
print(f"Model parameter count: {total_params/1e+8} 亿参数")

# Loss function and optimizer
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

def train(ask: str, answer: str) -> None:
    """Train the model on a single (ask, answer) pair"""
    print(f"\n---Train question:\n{ask}")
    print("\n---Train answer:")

    # Prepare input tensors
    ask_tensor = encode(ask).to(device)
    answer_tensor = encode(answer).to(device)

    # Create training sequence: [ask tokens] + [start token] + [answer tokens] + [end token]
    train_tensor = torch.cat([
        ask_tensor,
        torch.tensor([START_TOKEN], device=device),
        answer_tensor,
        torch.tensor([END_TOKEN], device=device)
    ])
    if train_tensor.numel() < 2:
        return

    train_tensor, loss_start = _truncate_train_tensor(train_tensor, len(ask_tensor))

    # Training loop (single forward with causal mask)
    model.train()
    prompt = train_tensor[:-1]
    targets = train_tensor[1:]
    logits = model(prompt)

    # Only compute loss on answer tokens (after the start token).
    if loss_start >= targets.numel():
        return
    loss_mask = torch.zeros_like(targets, dtype=torch.bool)
    loss_mask[loss_start:] = True

    masked_logits = logits[loss_mask]
    masked_targets = targets[loss_mask]
    if masked_targets.numel() == 0:
        return
    if masked_logits.dim() == 1:
        masked_logits = masked_logits.unsqueeze(0)
        masked_targets = masked_targets.unsqueeze(0)

    loss = loss_func(masked_logits, masked_targets)
    
    # Record loss for monitoring
    record_loss(loss.item())

    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # Print the target tokens for monitoring
    try:
        print(decode(masked_targets), end="", flush=True)
    except Exception:
        pass

    print("", flush=True)  # New line after generation

def generation(text: str) -> str:
    """Generate a response to the input text"""
    model.eval()
    output_text = ""

    # Prepare initial prompt
    prompt = torch.cat([
        encode(text).to(device),
        torch.tensor([START_TOKEN], device=device)
    ])
    if prompt.numel() > CONFIG["max_length"]:
        prompt = prompt[-CONFIG["max_length"]:]

    print("\n---Generated reply:")

    end_threshold = random.randint(3, 5)
    end_hits = 0

    with torch.inference_mode():
        for _ in range(CONFIG["max_length"]):
            try:
                logits = model(prompt)

                # Sample from the output distribution
                next_logits = logits[-1]
                probs = torch.softmax(next_logits / CONFIG["temperature"], dim=-1)
                index = int(torch.multinomial(probs, 1).item())

                if index == END_TOKEN:
                    end_hits += 1
                    if end_hits >= end_threshold:
                        break

                if index != END_TOKEN:
                    print(chr(int(index)), end="")
                    output_text += chr(int(index))

                prompt = torch.cat([prompt, torch.tensor([index], device=device)])
                if prompt.numel() > CONFIG["max_length"]:
                    prompt = prompt[-CONFIG["max_length"]:]
            except Exception:
                continue
    return output_text
