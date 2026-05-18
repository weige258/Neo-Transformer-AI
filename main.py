import sys

import torch
import torch.nn.functional as F

from config import CONFIG
from model import MainModel, compute_action_labels, ACTION_THINKING, ACTION_ANSWER, ACTION_END, TRANSITION_MASK
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
scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and amp_dtype == torch.float16))

print(f"Using device: {device}", flush=True)
print(f"AMP enabled: {use_amp}, AMP dtype: {amp_dtype}", flush=True)
model = _load_model()

# 【显存优化】关闭torch.compile，避免额外显存占用
print("[Info] Running without torch.compile optimization (disabled for memory efficiency).", flush=True)

total_params = sum(param.numel() for param in model.parameters())
print(f"模型参数: {total_params / 1e+8}亿", flush=True)

loss_func = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-4,
    weight_decay=0.01,
    foreach=torch.cuda.is_available(),
)

GRADIENT_ACCUMULATION_STEPS = 1
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

    # 最小生成 tokens
    min_new_tokens = max(4, int(CONFIG.get("min_generate_tokens", 4)))
    if max_generate_tokens is not None:
        max_generate_tokens = max(min_new_tokens + 1, int(max_generate_tokens))

    # 状态机：跟踪当前行动状态
    prev_action = ACTION_THINKING  # 初始状态假设为思考
    thinking_steps = 0             # 当前思考块已累积的步数（防止空块+死锁）
    max_thinking_steps = int(CONFIG.get("max_thinking_steps", 200))
    action_temperature = float(CONFIG.get("action_temperature", 0.5))
    _TAG_CHARS = frozenset({'<', '>', '/', '《', '》'})
    # 解码参数（反退化生成）
    llm_temperature = float(CONFIG.get("temperature", 0.7))
    rep_penalty = float(CONFIG.get("repetition_penalty", 1.15))
    top_k = int(CONFIG.get("top_k", 50))
    top_p = float(CONFIG.get("top_p", 0.9))
    generated_ids: list[int] = []  # 历史 token 用于重复惩罚
    # 转移掩码（log 域，0=禁止 → -inf）
    _trans_mask = torch.tensor(TRANSITION_MASK, device=device, dtype=torch.float32)
    _trans_mask = torch.where(_trans_mask > 0.5, torch.tensor(0.0, device=device),
                              torch.tensor(float('-inf'), device=device))
    
    with torch.inference_mode():
        if thinking_available:
            has_think_token = (prompt == TextTokenizer.THINK_START_TOKEN).any()
            if not has_think_token:
                think_start_tensor = torch.tensor([TextTokenizer.THINK_START_TOKEN], device=device)
                prompt = torch.cat([prompt, think_start_tensor])
        
        result = model(prompt, use_cache=True)
        if isinstance(result, tuple) and len(result) == 3:
            logits, action_logits, past_key_values = result
        elif isinstance(result, tuple) and len(result) == 2:
            logits, past_key_values = result
            action_logits = None
        else:
            logits = result
            action_logits = None

        step = 0
        
        while max_generate_tokens is None or step < max_generate_tokens:
            try:
                # ====================================================
                # 阶段1: Action Head 采样决策（主动控制）
                # ====================================================
                do_think_jump = False      # 是否强制跳转思考/回答
                force_token = None          # 强制注入的 token
                force_token_reason = ""     # 调试用

                if action_logits is not None:
                    act_logit = action_logits[-1].float()  # [3]
                    # 应用状态转移掩码：禁止非法转移
                    act_logit = act_logit + _trans_mask[prev_action]
                    # 采样行动
                    act_probs = torch.softmax(act_logit / action_temperature, dim=-1)
                    
                    # min_new_tokens 内仅禁止 END，保留思考/回答自由度
                    if step < min_new_tokens:
                        act_probs_l = act_probs.clone()
                        act_probs_l[ACTION_END] = 0.0
                        if act_probs_l.sum() > 0:
                            act_probs_l = act_probs_l / act_probs_l.sum()
                            sampled_action = int(torch.multinomial(act_probs_l, 1).item())
                        else:
                            sampled_action = ACTION_ANSWER
                    else:
                        sampled_action = int(torch.multinomial(act_probs, 1).item())

                    # === 行动决策 ===
                    if sampled_action == ACTION_THINKING and prev_action != ACTION_THINKING:
                        # 从回答→思考：注入 THINK_START
                        do_think_jump = True
                        force_token = TextTokenizer.THINK_START_TOKEN
                        thinking_steps = 0  # 重置思考步数计数
                    elif sampled_action == ACTION_ANSWER and prev_action == ACTION_THINKING:
                        # 空块防护：最少思考 3 步
                        # 死锁防护：超过 max_thinking_steps 强制切出
                        if thinking_steps < 3:
                            sampled_action = ACTION_THINKING
                        elif thinking_steps >= max_thinking_steps:
                            # 思考块已达上限，强制切出防止死循环
                            do_think_jump = True
                            force_token = TextTokenizer.THINK_END_TOKEN
                            sampled_action = ACTION_ANSWER
                        else:
                            # 从思考→回答：注入 THINK_END
                            do_think_jump = True
                            force_token = TextTokenizer.THINK_END_TOKEN
                    elif sampled_action == ACTION_END and step >= min_new_tokens:
                        # 结束
                        force_token = TextTokenizer.END_GENERATION_TOKEN
                        force_token_reason = "→END"
                    
                    # 更新状态
                    if sampled_action != ACTION_END:
                        prev_action = sampled_action

                # ====================================================
                # 阶段2: 执行生成（主语言模型）
                # ====================================================
                if force_token == TextTokenizer.END_GENERATION_TOKEN:
                    break

                if do_think_jump and force_token is not None:
                    # 主动注入特殊 token（THINK_START / THINK_END）
                    # 这些不输出到屏幕，不消耗 step，只更新 KV cache
                    inject_token = torch.tensor([force_token], device=device)
                    result = model(inject_token, past_key_values=past_key_values, use_cache=True)
                    if isinstance(result, tuple) and len(result) == 3:
                        logits, action_logits, past_key_values = result
                    elif isinstance(result, tuple) and len(result) == 2:
                        logits, past_key_values = result
                        action_logits = None
                    # 注入不消耗 step，继续下一步循环
                    continue

                # 正常采样下一个 token（含重复惩罚 + Top-K + Top-P）
                next_logits = logits[-1].clone()
                if step < min_new_tokens:
                    next_logits[TextTokenizer.END_GENERATION_TOKEN] = float("-inf")

                # 1) 重复惩罚：已生成过的 token 降低概率
                if rep_penalty != 1.0 and generated_ids:
                    for past_id in set(generated_ids):
                        score = next_logits[past_id].item()
                        if score > 0:
                            next_logits[past_id] = score / rep_penalty
                        else:
                            next_logits[past_id] = score * rep_penalty

                # 2) Top-K 过滤
                if top_k > 0 and top_k < next_logits.size(-1):
                    kth = torch.topk(next_logits, top_k)[0][-1]
                    next_logits[next_logits < kth] = float('-inf')

                # 3) 温度缩放 + softmax
                probs = torch.softmax(next_logits / llm_temperature, dim=-1)
                probs = torch.nan_to_num(probs, nan=0.0)

                # 4) Top-P (Nucleus) 过滤
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumsum = torch.cumsum(sorted_probs, dim=-1)
                    mask = cumsum > top_p
                    mask[1:] = mask[:-1].clone()  # 至少保留一个
                    mask[0] = False
                    to_remove = torch.zeros_like(probs, dtype=torch.bool)
                    to_remove.scatter_(dim=-1, index=sorted_indices[mask], src=torch.ones_like(sorted_indices[mask], dtype=torch.bool))
                    probs[to_remove] = 0.0
                    probs = probs / (probs.sum(dim=-1, keepdim=True).clamp(min=1e-8))

                index = int(torch.multinomial(probs, 1).item())
                generated_ids.append(index)

                # 累积思考步数（在特殊 token 跳过之前更新）
                if prev_action == ACTION_THINKING:
                    thinking_steps += 1
                else:
                    thinking_steps = 0

                # 特殊 token 静默跳过 —— 同步 prev_action
                if index in (TextTokenizer.THINK_START_TOKEN, TextTokenizer.THINK_END_TOKEN,
                             TextTokenizer.HISTORY_CONTEXT_START_TOKEN, TextTokenizer.HISTORY_CONTEXT_END_TOKEN,
                             TextTokenizer.START_GENERATION_TOKEN):
                    if index == TextTokenizer.THINK_START_TOKEN:
                        prev_action = ACTION_THINKING  # LM 自己生成 THINK_START → 同步
                    elif index == TextTokenizer.THINK_END_TOKEN:
                        prev_action = ACTION_ANSWER     # LM 自己生成 THINK_END → 同步
                    next_token = torch.tensor([index], device=device)
                    result = model(next_token, past_key_values=past_key_values, use_cache=True)
                    if isinstance(result, tuple) and len(result) == 3:
                        logits, action_logits, past_key_values = result
                    elif isinstance(result, tuple) and len(result) == 2:
                        logits, past_key_values = result
                        action_logits = None
                    else:
                        logits = result
                        action_logits = None
                    step += 1
                    continue

                if index == TextTokenizer.END_GENERATION_TOKEN:
                    break

                # 颜色：直接使用当前行动状态
                current_color = BLUE if prev_action == ACTION_THINKING else GREEN

                decoded_piece = TextTokenizer.decode(torch.tensor([index]))
                if decoded_piece and decoded_piece in _TAG_CHARS:
                    decoded_piece = ""
                if decoded_piece:
                    print(f"{current_color}{decoded_piece}{RESET}", end="", flush=True)
                    output_text += decoded_piece

                next_token = torch.tensor([index], device=device)
                result = model(next_token, past_key_values=past_key_values, use_cache=True)
                if isinstance(result, tuple) and len(result) == 3:
                    logits, action_logits, past_key_values = result
                elif isinstance(result, tuple) and len(result) == 2:
                    logits, past_key_values = result
                    action_logits = None
                else:
                    logits = result
                    action_logits = None
                
                step += 1
            except Exception as e:
                print(f"Error during generation: {e}", flush=True)
                break
        
        with torch.inference_mode():
            torch.cuda.empty_cache()
        
        return output_text


def train_dynamic(ask: str, response: str, history_context: str = None) -> float:
    """动态格式训练：response 可包含多段 <think>...内容...</think>交替。
    
    自动将文本中的 <think> 和 </think> 替换为特殊 token ID (5=THINK_START, 6=THINK_END)。
    例如 response = "<think>先算12*15</think>12*15=180.<think>再算一半</think>90."
    """
    ask = str(ask).strip() if ask else ""
    response = str(response).strip() if response else ""
    if not ask or not response:
        return float('inf')

    model.train()

    ask_tensor = TextTokenizer.encode(ask).to(device)

    # 手动解析 <think> 和 </think> 标签 → 替换为特殊 token ID
    resp_list = []
    i = 0
    n = len(response)
    TAG_THINK = '<think>'
    TAG_THINK_END = '</think>'
    while i < n:
        if response[i:].startswith(TAG_THINK):
            resp_list.append(TextTokenizer.THINK_START_TOKEN)
            i += len(TAG_THINK)
        elif response[i:].startswith(TAG_THINK_END):
            resp_list.append(TextTokenizer.THINK_END_TOKEN)
            i += len(TAG_THINK_END)
        else:
            ch = response[i]
            idx = ord(ch)
            if TextTokenizer._is_valid_token(idx) and 0 <= idx < int(CONFIG["dict_size"]):
                resp_list.append(idx)
            else:
                resp_list.append(TextTokenizer.UNKNOWN_TOKEN)
            i += 1

    if not resp_list:
        return float('inf')
    resp_tensor = torch.tensor(resp_list, dtype=torch.long, device=device)

    if ask_tensor.numel() == 0 or resp_tensor.numel() == 0:
        return float('inf')

    if history_context and history_context.strip():
        hist_tensor = TextTokenizer.encode(history_context).to(device)
        if auto_compress_trigger(hist_tensor):
            compressed_hist = model.compress_history_vectors(hist_tensor)
            hist_tensor = torch.argmax(model.output_linear(compressed_hist), dim=-1)
        train_tensor = torch.cat([
            torch.tensor([TextTokenizer.START_GENERATION_TOKEN], device=device),
            hist_tensor,
            torch.tensor([TextTokenizer.END_GENERATION_TOKEN], device=device),
            torch.tensor([TextTokenizer.HISTORY_CONTEXT_START_TOKEN], device=device),
            ask_tensor,
            torch.tensor([TextTokenizer.START_GENERATION_TOKEN], device=device),
            resp_tensor,
            torch.tensor([TextTokenizer.END_GENERATION_TOKEN], device=device),
        ])
        non_target_len = 1 + hist_tensor.numel() + 1 + 1 + ask_tensor.numel() + 1
    else:
        train_tensor = torch.cat([
            ask_tensor,
            torch.tensor([TextTokenizer.START_GENERATION_TOKEN], device=device),
            resp_tensor,
            torch.tensor([TextTokenizer.END_GENERATION_TOKEN], device=device),
        ])
        non_target_len = ask_tensor.numel() + 1

    target_len = resp_tensor.numel() + 1
    target_mask = torch.cat([
        torch.zeros(non_target_len, dtype=torch.bool, device=device),
        torch.ones(target_len, dtype=torch.bool, device=device),
    ])
    assert target_mask.numel() == train_tensor.numel(), \
        f"mask len {target_mask.numel()} != train len {train_tensor.numel()}"

    loss_val = _run_train_step(train_tensor, target_mask, resp_tensor, show_preview=False)
    torch.cuda.empty_cache()
    return loss_val


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
        optimizer.zero_grad(set_to_none=True)

    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        result = model(train_tensor, use_cache=False)
        if isinstance(result, tuple):
            logits = result[0]
            action_logits = result[1]
        else:
            logits = result
            action_logits = None

        # === 语言模型损失 (next-token prediction) ===
        if len(train_tensor) > 1:
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
                
                lm_loss = loss_func(masked_logits, masked_targets)
                
                if torch.isnan(lm_loss):
                    print(f"[Warning] NaN loss detected, skipping this step", flush=True)
                    return float('inf')
            else:
                lm_loss = torch.tensor(0.0, device=device)
        else:
            lm_loss = torch.tensor(0.0, device=device)
        
        loss = lm_loss

        # === 行动头损失 (hard label cross-entropy) ===
        if action_logits is not None:
            # 生成硬标签：compute_action_labels 返回 [seq_len, 3] 软标签
            # 取 argmax 转为硬标签 [seq_len]
            action_labels_soft = compute_action_labels(
                train_tensor,
                think_start_id=TextTokenizer.THINK_START_TOKEN,
                think_end_id=TextTokenizer.THINK_END_TOKEN,
                end_id=TextTokenizer.END_GENERATION_TOKEN,
                temperature=float(CONFIG.get("action_label_temperature", 0.5)),
            )
            # 对齐: action_logits[:-1] 对应 action_labels[1:] (next-token prediction)
            act_logits_aligned = action_logits[:-1]  # [seq_len-1, 3]
            act_hard_labels = action_labels_soft[1:].argmax(dim=-1)  # [seq_len-1]
            # 只对 target_mask 选中的位置计算行动损失
            if target_mask[1:].any():
                act_logits_masked = act_logits_aligned[target_mask[1:]]
                act_labels_masked = act_hard_labels[target_mask[1:]]
                action_ce_loss = F.cross_entropy(
                    act_logits_masked.float(), act_labels_masked,
                )
                action_coef = float(CONFIG.get("action_loss_coef", 0.3))
                action_loss = action_coef * action_ce_loss
                loss = loss + action_loss
                if not torch.isnan(action_ce_loss) and not torch.isinf(action_ce_loss):
                    record_loss(action_ce_loss.item())
        
        # 【修复】损失缩放，适配梯度累积
        loss = loss / GRADIENT_ACCUMULATION_STEPS

    # 检查损失是否有效
    if not torch.isnan(loss) and not torch.isinf(loss):
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if torch.isnan(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                print(f"[Warning] NaN gradient detected, skipping optimizer step", flush=True)
                return float('inf')
            
            if (training_rounds + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if torch.isnan(grad_norm):
                optimizer.zero_grad(set_to_none=True)
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
