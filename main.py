import sys

import torch

from config import CONFIG
from model import MainModel
from record import record_loss
from tokenizer import TextTokenizer
from rl import SelfRewardModel, LightweightPPO


if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model() -> MainModel:
    try:
        # 安全加载：不使用不存在的 `weights_only` 参数，使用 map_location
        loaded = torch.load("model.pth", map_location=device)
        model = MainModel().to(device)
        
        # 【修复】严格校验键匹配，避免加载不匹配
        model_state = model.state_dict()
        filtered_state = {k: v for k, v in loaded.items() if k in model_state and v.shape == model_state[k].shape}
        missing_keys = [k for k in model_state if k not in filtered_state]
        if missing_keys:
            print(f"[Warning] 缺失权重键: {missing_keys}, 随机初始化", flush=True)
        model.load_state_dict(filtered_state, strict=False)
        print("Loaded model state dict safely.", flush=True)
        return model
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
scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

print(f"Using device: {device}", flush=True)
print(f"AMP enabled: {use_amp}, AMP dtype: {amp_dtype}", flush=True)
model = _load_model()

# 【显存优化】关闭torch.compile，避免额外显存占用
print("[Info] Running without torch.compile optimization (disabled for memory efficiency).", flush=True)

total_params = sum(param.numel() for param in model.parameters())
print(f"模型参数: {total_params / 1e+8}亿", flush=True)

loss_func = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)

GRADIENT_ACCUMULATION_STEPS = 4
training_rounds = 0

# 初始化自奖励模型和强化学习模块
reward_model = SelfRewardModel(device)
ppo_trainer = LightweightPPO(model, reward_model, device, learning_rate=1e-5)

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
    """准备单个样本的训练数据"""
    if ask_text is None or answer_text is None:
        return None, None, None
    
    ask_tensor = TextTokenizer.encode(ask_text).to(device)
    answer_tensor = TextTokenizer.encode(answer_text).to(device)
    
    if answer_tensor.numel() == 0:
        return None, None, None

    if hist_context is not None and hist_context.strip():
        history_tensor = TextTokenizer.encode(hist_context).to(device)
        
        if auto_compress_trigger(history_tensor):
            compressed_hist = model.compress_history_vectors(history_tensor)
            history_tensor = torch.argmax(model.output_linear(compressed_hist), dim=-1)
        
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
        non_target_len = 1 + history_tensor.numel() + 1 + 1 + ask_tensor.numel() + 1
        target_mask = torch.cat([
            torch.zeros(non_target_len, dtype=torch.bool, device=device),
            torch.ones(answer_tensor.numel() + 1, dtype=torch.bool, device=device),
        ])
        assert target_mask.numel() == train_tensor.numel(), f"target_mask length {target_mask.numel()} != train_tensor length {train_tensor.numel()}"
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

        train_tensor = torch.cat(
            [
                torch.tensor([TextTokenizer.START_GENERATION_TOKEN], device=device),
                text_tensor,
                torch.tensor([TextTokenizer.END_GENERATION_TOKEN], device=device),
            ]
        )
        target_mask = torch.ones(train_tensor.numel(), dtype=torch.bool, device=device)
        preview = train_tensor
        _run_train_step(train_tensor, target_mask, preview, show_preview=True, preview_color=YELLOW)
        
        # 【显存优化】训练后主动释放显存，解决泄漏问题
        torch.cuda.empty_cache()
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
            
            if hist_context := history_context:
                history_tensor = TextTokenizer.encode(hist_context).to(device)
                train_tensor = torch.cat(
                    [
                        torch.tensor([TextTokenizer.START_GENERATION_TOKEN], device=device),
                        history_tensor,
                        torch.tensor([TextTokenizer.END_GENERATION_TOKEN], device=device),
                        torch.tensor([TextTokenizer.HISTORY_CONTEXT_START_TOKEN], device=device),
                        ask_tensor,
                        torch.tensor([TextTokenizer.START_GENERATION_TOKEN], device=device),
                        torch.tensor([TextTokenizer.THINK_START_TOKEN], device=device),
                        think_tensor,
                        torch.tensor([TextTokenizer.THINK_END_TOKEN], device=device),
                        answer_tensor,
                        torch.tensor([TextTokenizer.END_GENERATION_TOKEN], device=device),
                    ]
                )
                non_target_len = 1 + history_tensor.numel() + 1 + 1 + ask_tensor.numel() + 1 + 1
                target_len = think_tensor.numel() + 1 + answer_tensor.numel() + 1
                target_mask = torch.cat([
                    torch.zeros(non_target_len, dtype=torch.bool, device=device),
                    torch.ones(target_len, dtype=torch.bool, device=device),
                ])
                assert target_mask.numel() == train_tensor.numel(), f"target_mask length {target_mask.numel()} != train_tensor length {train_tensor.numel()}"
                preview = torch.cat([think_tensor, answer_tensor])
                _run_train_step(train_tensor, target_mask, preview, show_preview=False)
                
                torch.cuda.empty_cache()
            else:
                train_tensor = torch.cat(
                    [
                        ask_tensor,
                        torch.tensor([TextTokenizer.START_GENERATION_TOKEN], device=device),
                        torch.tensor([TextTokenizer.THINK_START_TOKEN], device=device),
                        think_tensor,
                        torch.tensor([TextTokenizer.THINK_END_TOKEN], device=device),
                        answer_tensor,
                        torch.tensor([TextTokenizer.END_GENERATION_TOKEN], device=device),
                    ]
                )
                non_target_len = ask_tensor.numel() + 1 + 1
                target_len = think_tensor.numel() + 1 + answer_tensor.numel() + 1  # think + THINK_END + answer + END
                target_mask = torch.cat([
                    torch.zeros(non_target_len, dtype=torch.bool, device=device),
                    torch.ones(target_len, dtype=torch.bool, device=device),
                ])
                assert target_mask.numel() == train_tensor.numel(), f"target_mask length {target_mask.numel()} != train_tensor length {train_tensor.numel()}"
                preview = torch.cat([think_tensor, answer_tensor])
                _run_train_step(train_tensor, target_mask, preview, show_preview=False)
                
                torch.cuda.empty_cache()
            return
        
        print(f"{GREEN}{answer}{RESET}", flush=True)
        train_tensor, target_mask, preview = _prepare_training_data(ask, answer, history_context)
        if train_tensor is None:
            return
        _run_train_step(train_tensor, target_mask, preview, show_preview=False)
        
        torch.cuda.empty_cache()
    
    # 自奖励评估和PPO强化学习（静默进行，不影响原有训练）
    try:
        reward_model.compute_total_reward(
            think_text=think,
            answer_text=answer,
            context=history_context
        )
        
        # 收集episode数据
        ppo_trainer.collect_episode(
            prompt=ask if ask else "",
            think_text=think if think else "",
            answer_text=answer if answer else "",
            context=history_context
        )
        
        # 定期更新PPO策略
        if training_rounds > 0 and (training_rounds % 4) == 0:
            ppo_trainer.update_policy(batch_size=4)
    except Exception as e:
        pass  # 静默处理错误，不影响原有训练


def generation(text: str, history_context: str = None, max_generate_tokens: int|None = None, thinking_available: bool = True) -> str:
    """生成函数
    
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
    
    model.eval()
    output_text = ""

    if history_context and history_context.strip():
        history_tensor = TextTokenizer.encode(history_context).to(device)
        
        if auto_compress_trigger(history_tensor):
            compressed_hist = model.compress_history_vectors(history_tensor)
            history_tensor = torch.argmax(model.output_linear(compressed_hist), dim=-1)
        
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

    min_new_tokens = 1
    if max_generate_tokens is not None:
        max_generate_tokens = max(1, int(max_generate_tokens))
 
    with torch.inference_mode():
        thinking_started = False
        if thinking_available:
            has_think_token = (prompt == TextTokenizer.THINK_START_TOKEN).any()
            if has_think_token:
                thinking_started = True
            else:
                thinking_started = True
                think_start_tensor = torch.tensor([TextTokenizer.THINK_START_TOKEN], device=device)
                prompt = torch.cat([prompt, think_start_tensor])
        
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

                should_skip_output = False
                
                if index == TextTokenizer.THINK_END_TOKEN:
                    if thinking_available and thinking_started:
                        thinking_started = False
                        print(f"{RESET}\n", end="", flush=True)
                        should_skip_output = True
                    else:
                        break
                
                elif index == TextTokenizer.END_GENERATION_TOKEN:
                    break

                elif index == TextTokenizer.THINK_START_TOKEN:
                    if thinking_available and not thinking_started:
                        thinking_started = True
                        should_skip_output = True
                    elif not thinking_available:
                        should_skip_output = True
                
                if not should_skip_output:
                    decoded_piece = TextTokenizer.decode(torch.tensor([index]))
                    
                    if decoded_piece:
                        if thinking_started:
                            print(f"{BLUE}{decoded_piece}{RESET}", end="", flush=True)
                        else:
                            print(f"{GREEN}{decoded_piece}{RESET}", end="", flush=True)
                        
                        output_text += decoded_piece

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
        
        # 【显存优化】生成后主动释放显存，解决泄漏问题
        with torch.inference_mode():
            torch.cuda.empty_cache()
        
        # 自奖励评估（已移除）
        
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
    
    if (training_rounds % GRADIENT_ACCUMULATION_STEPS) == 0:
        optimizer.zero_grad()

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
            if torch.isnan(grad_norm):
                optimizer.zero_grad()
                print(f"[Warning] NaN gradient detected, skipping optimizer step", flush=True)
                return float('inf')
            
            if (training_rounds + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if torch.isnan(grad_norm):
                optimizer.zero_grad()
                print(f"[Warning] NaN gradient detected, skipping optimizer step", flush=True)
                return float('inf')
            
            if (training_rounds + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
    else:
        print(f"[Warning] Invalid loss detected: {loss}, skipping optimizer step", flush=True)
        return float('inf')
    
    training_rounds += 1

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
