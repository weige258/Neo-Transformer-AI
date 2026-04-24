import sys
import math

import torch

from model import CONFIG, MainModel
from record import record_loss
from tokenizer import TextTokenizer
from rl import (
    GRPO_CONFIG, TREE_SEARCH_CONFIG, 
    _compute_grpo_advantages, _majority_vote, _generate_with_sampling,
    _tree_search, _compute_heuristic_reward
)


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


# 【性能优化】启用TensorFloat32加速矩阵运算(消除UserWarning)
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

# 启用自动混合精度训练
scaler = torch.amp.GradScaler()
use_amp = torch.cuda.is_available() and torch.cuda.get_device_capability(device)[0] >= 7

print(f"Using device: {device}", flush=True)
print(f"AMP enabled: {use_amp}", flush=True)
model = _load_model()

# 【性能优化】启用PyTorch 2.0编译加速(需要Triton支持)
compile_success = False
if torch.cuda.is_available():
    # 检查是否有Triton支持
    try:
        import triton
        has_triton = True
    except ImportError:
        has_triton = False
    
    if has_triton:
        import warnings
        import logging
        
        # 临时禁用torch._inductor的日志输出
        inductor_logger = logging.getLogger("torch._inductor")
        old_level = inductor_logger.level
        inductor_logger.setLevel(logging.ERROR)
        
        try:
            # 使用"reduce-overhead"模式适合小batch的推理场景
            model = torch.compile(model, mode="reduce-overhead")
            
            # 测试编译是否成功(首次调用会触发编译)
            with torch.inference_mode():
                test_input = torch.tensor([TextTokenizer.START_GENERATION_TOKEN], device=device).unsqueeze(0)
                _ = model(test_input)
            
            compile_success = True
        except Exception as e:
            # 编译失败,回退到标准模式，保留原模型
            print(f"[Warning] Torch compile failed: {e}, continuing with original model", flush=True)
            # 不重新加载模型，保留当前权重
        finally:
            # 恢复日志级别
            inductor_logger.setLevel(old_level)

if not compile_success:
    print("[Info] Running without torch.compile optimization.", flush=True)

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
        # 【关键修复】正确计算target_mask长度
        # train_tensor结构: START + history + END + HIST_START + ask + START + answer + END
        # 非目标部分: START + history + END + HIST_START + ask + START = 1 + h + 1 + 1 + a + 1 = h + a + 4
        # 目标部分: answer + END = ans + 1
        non_target_len = 1 + history_tensor.numel() + 1 + 1 + ask_tensor.numel() + 1
        target_mask = torch.cat([
            torch.zeros(non_target_len, dtype=torch.bool, device=device),
            torch.ones(answer_tensor.numel() + 1, dtype=torch.bool, device=device),
        ])
        # 验证长度匹配
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
    """单步训练函数 - 集成GRPO和TTRL
    
    Args:
        ask: 问题文本
        think: 思维链/推理过程（可选，用于CoT训练）
        answer: 答案文本
        history_context: 历史对话上下文
    """
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
        # 单文本模式：在_run_train_step中输出黄色预览
        _run_train_step(train_tensor, target_mask, preview, show_preview=True, preview_color=YELLOW)
        return

    # QA训练模式
    print(f"\n---Train{RESET}", flush=True)
    print(f"{WHITE}{ask}{RESET}", flush=True)
    
    # 【关键修复】如果有明确的参考答案，直接执行标准SFT
    if answer and answer.strip():
        # 如果提供了思维链，构建带思考的训练数据
        if think and think.strip():
            print(f"{BLUE}{think}{RESET}", flush=True)
            print(f"{GREEN}{answer}{RESET}", flush=True)
            
            # 构建带思维链的训练序列
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
                # 只对思维过程和答案部分计算loss
                # train_tensor结构: START + history + END + HIST_START + ask + START + THINK_START + think + THINK_END + answer + END
                non_target_len = 1 + history_tensor.numel() + 1 + 1 + ask_tensor.numel() + 1 + 1  # START + history + END + HIST_START + ask + START + THINK_START
                target_len = think_tensor.numel() + 1 + answer_tensor.numel() + 1  # think + THINK_END + answer + END
                target_mask = torch.cat([
                    torch.zeros(non_target_len, dtype=torch.bool, device=device),
                    torch.ones(target_len, dtype=torch.bool, device=device),
                ])
                # 【关键修复】确保target_mask长度与train_tensor完全一致
                assert target_mask.numel() == train_tensor.numel(), f"target_mask length {target_mask.numel()} != train_tensor length {train_tensor.numel()}"
                preview = torch.cat([think_tensor, answer_tensor])
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
                # 只对思维过程和答案部分计算loss
                # train_tensor结构: ask + START + THINK_START + think + THINK_END + answer + END
                non_target_len = ask_tensor.numel() + 1 + 1  # ask + START + THINK_START
                target_len = think_tensor.numel() + 1 + answer_tensor.numel() + 1  # think + THINK_END + answer + END
                target_mask = torch.cat([
                    torch.zeros(non_target_len, dtype=torch.bool, device=device),
                    torch.ones(target_len, dtype=torch.bool, device=device),
                ])
                # 【关键修复】确保target_mask长度与train_tensor完全一致
                assert target_mask.numel() == train_tensor.numel(), f"target_mask length {target_mask.numel()} != train_tensor length {train_tensor.numel()}"
                preview = torch.cat([think_tensor, answer_tensor])
            
            _run_train_step(train_tensor, target_mask, preview, show_preview=False)
            return
        
        # 没有思维链，使用标准QA训练
        print(f"{GREEN}{answer}{RESET}", flush=True)
        train_tensor, target_mask, preview = _prepare_training_data(ask, answer, history_context)
        if train_tensor is None:
            return
        _run_train_step(train_tensor, target_mask, preview, show_preview=False)  # QA模式不重复显示
        return
    
    # 【GRPO】无标签场景：使用模型生成的样本来进行强化学习
    prompt = torch.cat(
        [
            TextTokenizer.encode(ask).to(device),
            torch.tensor([TextTokenizer.START_GENERATION_TOKEN], device=device),
        ]
    )
    
    # 【GRPO】为问题生成多个候选答案
    samples = _generate_with_sampling(model, prompt, GRPO_CONFIG["num_samples"])
    if not samples:
        return
    
    # 【GRPO】计算每个候选答案的奖励
    rewards = []
    for sample in samples:
        _, decoded_text = sample
        reward = _compute_heuristic_reward(decoded_text)
        rewards.append(reward)
    
    # 【GRPO】计算优势函数
    advantages = _compute_grpo_advantages(rewards)
    
    # 【修复GRPO】累积所有样本的梯度后再更新
    total_loss = torch.tensor(0.0, device=device)
    valid_samples = 0
    
    # 首先清空梯度
    optimizer.zero_grad()
    
    # 【优化】预计算所有样本的完整tokens
    full_tokens_list = []
    target_mask_list = []
    for sample_tokens, _ in samples:
        full_tokens = torch.cat([prompt, sample_tokens])
        target_mask = torch.cat([
            torch.zeros(len(prompt), dtype=torch.bool, device=device),
            torch.ones(len(sample_tokens), dtype=torch.bool, device=device),
        ])
        full_tokens_list.append(full_tokens)
        target_mask_list.append(target_mask)
    
    # 【优化】一次性计算所有样本的参考策略logits
    with torch.no_grad():
        ref_logits_list = []
        for full_tokens in full_tokens_list:
            ref_result = model(full_tokens, use_cache=False)
            if isinstance(ref_result, tuple):
                ref_logits = ref_result[0]
            else:
                ref_logits = ref_result
            ref_logits_list.append(ref_logits)
    
    for idx, (sample_tokens, sample_text) in enumerate(samples):
        advantage = advantages[idx]
        
        # 跳过优势为0的样本（组内表现平均）
        if abs(advantage) < 1e-6:
            continue
        
        # 获取预计算的完整tokens和目标掩码
        full_tokens = full_tokens_list[idx]
        target_mask = target_mask_list[idx]
        ref_logits = ref_logits_list[idx]
        
        # 【修复】使用相同的训练步骤但不立即更新，而是累积梯度
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            # 【修复1】对完整序列计算当前策略的logits
            result = model(full_tokens, use_cache=False)
            if isinstance(result, tuple):
                logits = result[0]
            else:
                logits = result

            # 应用目标掩码并进行 next-token prediction 对齐
            if len(full_tokens) > 1:
                # 正确的 next-token prediction 对齐
                masked_logits = logits[:-1][target_mask[1:]]
                masked_targets = full_tokens[1:][target_mask[1:]]
                
                if len(masked_logits) > 0 and len(masked_targets) > 0:
                    # 【修复1】获取参考策略的logits（对应生成部分）
                    ref_logits_for_gen = ref_logits[:-1][target_mask[1:]]
                    
                    # 计算当前策略和参考策略的log概率
                    curr_log_probs = torch.log_softmax(masked_logits, dim=-1)
                    ref_log_probs = torch.log_softmax(ref_logits_for_gen, dim=-1)
                    
                    # 计算策略比率
                    curr_log_prob_selected = torch.gather(curr_log_probs, -1, masked_targets.unsqueeze(-1)).squeeze(-1)
                    ref_log_prob_selected = torch.gather(ref_log_probs, -1, masked_targets.unsqueeze(-1)).squeeze(-1)
                    
                    # 计算策略比率
                    ratio = torch.exp(curr_log_prob_selected - ref_log_prob_selected)
                    
                    # 【修复】GRPO标准损失计算：advantage * ratio * log_prob
                    pg_loss = -advantage * ratio * curr_log_prob_selected
                    
                    # 【修复2】修正KL散度的分布顺序：应该是当前策略在前，参考策略在后
                    kl_div = torch.distributions.kl_divergence(
                        torch.distributions.Categorical(logits=masked_logits),
                        torch.distributions.Categorical(logits=ref_logits_for_gen)
                    ).mean()
                    
                    # 总损失 = 策略梯度损失 + KL散度惩罚
                    sample_loss = pg_loss.mean() + GRPO_CONFIG["kl_coefficient"] * kl_div
                    
                    # 累积损失，按样本数平均以保持梯度尺度稳定
                    total_loss += sample_loss / GRPO_CONFIG["num_samples"]
                    valid_samples += 1
                else:
                    continue
            else:
                continue
    
    # 【GRPO】只有在有有效样本时才进行参数更新
    if valid_samples > 0:
        # 应用梯度缩放和更新
        scaler.scale(total_loss).backward()
        
        # 【可选】添加梯度裁剪以提高稳定性
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        print(f"[GRPO] Trained on {valid_samples} samples with advantages: {[f'{a:.2f}' for a in advantages]}", flush=True)
    else:
        print("[GRPO] No valid samples for training, falling back to standard SFT if available", flush=True)


def generation(text: str, history_context: str = None, max_generate_tokens: int|None = None, thinking_available: bool = True) -> str:
    """生成函数 - 集成树状搜索强化学习
    
    Args:
        text: 输入文本/问题
        history_context: 历史上下文(可选)
        max_generate_tokens: 最大生成token数
        thinking_available: 是否启用思维链生成（默认True）
    
    Returns:
        生成的文本
    """
    # ANSI颜色代码
    BLUE = '\033[94m'      # 思考内容 - 蓝色
    GREEN = '\033[92m'     # 回答内容 - 绿色
    RESET = '\033[0m'      # 重置颜色
    
    model.eval()
    output_text = ""

    # 构建prompt(支持历史上下文)
    if history_context and history_context.strip():
        history_tensor = TextTokenizer.encode(history_context).to(device)
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
        # 【树状搜索强化学习】在生成阶段执行MCTS搜索
        best_new_tokens, best_full_text, tree_reward = _tree_search(model, prompt, text)
        
        # 【关键修复】MCTS只决定前N个token，然后继续自回归生成
        # 如果MCTS找到了高质量的前缀且长度合理，使用它作为起始
        mcts_used = False
        if len(best_new_tokens) > 0 and tree_reward > 0.3:
            # 将MCTS生成的tokens拼接到prompt
            current_prompt = torch.cat([prompt, best_new_tokens])
            # 解码MCTS生成的部分
            mcts_text = TextTokenizer.decode(best_new_tokens[best_new_tokens != 0])
            if mcts_text:
                print(mcts_text, end="", flush=True)
                output_text += mcts_text
                mcts_used = True
        else:
            current_prompt = prompt.clone()
        
        # 【新增】如果启用思维链，检查当前prompt是否包含THINK_START_TOKEN
        thinking_started = False
        if thinking_available:
            # 检查当前prompt是否已经包含THINK_START_TOKEN（包括MCTS生成的部分）
            has_think_token = (current_prompt == TextTokenizer.THINK_START_TOKEN).any()
            if has_think_token:
                # 如果prompt中存在THINK_START_TOKEN，则认为思考已经开始
                thinking_started = True
            elif not mcts_used:
                # 如果MCTS没有使用且没有THINK_START_TOKEN，则手动添加
                thinking_started = True
                # 添加THINK_START_TOKEN到prompt
                think_start_tensor = torch.tensor([TextTokenizer.THINK_START_TOKEN], device=device)
                current_prompt = torch.cat([current_prompt, think_start_tensor])
        
        # 【修复】继续标准自回归生成（无论是否使用了MCTS）
        result = model(current_prompt, use_cache=True)
        if isinstance(result, tuple):
            logits, past_key_values = result
        else:
            logits = result

        step = 0
        thinking_content = ""
        answer_content = ""
        
        while max_generate_tokens is None or step < max_generate_tokens:
            try:
                next_logits = logits[-1]
                if step < min_new_tokens:
                    next_logits = next_logits.clone()
                    next_logits[TextTokenizer.END_GENERATION_TOKEN] = float("-inf")

                probs = torch.softmax(next_logits / CONFIG["temperature"], dim=-1)
                index = int(torch.multinomial(probs, 1).item())

                # 【关键修复】先处理特殊token，再决定是否continue
                should_skip_output = False
                
                # 检查是否遇到THINK_END_TOKEN，切换到答案生成模式
                if index == TextTokenizer.THINK_END_TOKEN:
                    if thinking_available and thinking_started:
                        # 思维链结束，切换到答案模式
                        thinking_started = False
                        print(f"{RESET}\n", end="", flush=True)  # 重置颜色并换行
                        should_skip_output = True  # 不输出THINK_END_TOKEN本身
                    else:
                        break
                
                # 检查是否遇到END_GENERATION_TOKEN
                elif index == TextTokenizer.END_GENERATION_TOKEN:
                    break

                # 检查是否遇到THINK_START_TOKEN
                elif index == TextTokenizer.THINK_START_TOKEN:
                    if thinking_available and not thinking_started:
                        # 如果思维链尚未开始但遇到了开始标记，则开启思维链
                        thinking_started = True
                        should_skip_output = True  # 不输出THINK_START_TOKEN本身
                    elif not thinking_available:
                        # 如果不启用思维链，跳过思考过程
                        should_skip_output = True
                
                # 【普通token】解码并输出（带颜色）
                if not should_skip_output:
                    decoded_piece = TextTokenizer.decode(torch.tensor([index]))
                    
                    if decoded_piece:
                        if thinking_started:
                            # 正在生成思考内容 - 蓝色
                            thinking_content += decoded_piece
                            print(f"{BLUE}{decoded_piece}{RESET}", end="", flush=True)
                        else:
                            # 正在生成答案内容 - 绿色
                            answer_content += decoded_piece
                            print(f"{GREEN}{decoded_piece}{RESET}", end="", flush=True)
                        
                        output_text += decoded_piece

                # 【统一处理】更新prompt和logits
                next_token = torch.tensor([index], device=device)
                current_prompt = torch.cat([current_prompt, next_token])
                
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
        
        # 输出总结（无颜色）
        if thinking_available and thinking_content:
            pass
        
        return output_text


def _run_train_step(train_tensor: torch.Tensor, target_mask: torch.Tensor, preview: torch.Tensor, advantage_weight: float = 1.0, show_preview: bool = True, preview_color: str = None) -> None:
    """执行单步训练
    
    Args:
        train_tensor: 训练张量
        target_mask: 目标掩码
        preview: 预览张量
        advantage_weight: 优势加权因子(用于GRPO)
        show_preview: 是否显示预览输出(默认True,QA模式下可设为False避免重复)
        preview_color: 预览文本颜色(可选)
    """
    model.train()
    optimizer.zero_grad()

    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
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
                # 检查masked_logits和masked_targets是否包含NaN或无穷大
                if torch.isnan(masked_logits).any() or torch.isinf(masked_logits).any():
                    print(f"[Warning] NaN or Inf detected in logits, skipping this step", flush=True)
                    return
                
                if torch.isnan(masked_targets).any() or torch.isinf(masked_targets).any():
                    print(f"[Warning] NaN or Inf detected in targets, skipping this step", flush=True)
                    return
                
                loss = loss_func(masked_logits, masked_targets)
                
                # 检查loss是否为NaN
                if torch.isnan(loss):
                    print(f"[Warning] NaN loss detected, skipping this step", flush=True)
                    return
            else:
                loss = torch.tensor(0.0, device=device)
        else:
            loss = torch.tensor(0.0, device=device)
        
        # 检查advantage_weight是否为NaN
        if torch.isnan(torch.tensor(advantage_weight)):
            print(f"[Warning] NaN advantage_weight detected, using 1.0", flush=True)
            advantage_weight = 1.0
            
        # 应用GRPO优势加权
        loss = loss * advantage_weight
        record_loss(loss.item())

    # 检查损失是否有效
    if not torch.isnan(loss) and not torch.isinf(loss):
        scaler.scale(loss).backward()
        
        # 梯度裁剪以防止梯度爆炸
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
    else:
        print(f"[Warning] Invalid loss detected: {loss}, skipping optimizer step", flush=True)

    # 输出训练预览(可选)
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