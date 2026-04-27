import sys
import math

import torch

from model import CONFIG, MainModel
from record import record_loss, get_loss
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


# 【性能优化】启用TensorFloat32加速矩阵运算(消除UserWarning)
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

# 启用自动混合精度训练
scaler = torch.amp.GradScaler()

# 检查GPU能力，对于较老的GPU（compute capability < 8.0）禁用AMP
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(device)
    # bfloat16 有和 float32 一样的动态范围，不会溢出
    use_amp = cap[0] >= 8  # Ampere (A100, RTX 30xx/40xx) 才支持 bfloat16
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
else:
    use_amp = False
    amp_dtype = torch.float32

print(f"Using device: {device}", flush=True)
print(f"AMP enabled: {use_amp}, AMP dtype: {amp_dtype}", flush=True)
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
    # 【修复】过滤 NaN 字符串
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
        current_loss = _run_train_step(train_tensor, target_mask, preview, show_preview=True, preview_color=YELLOW)
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
                current_loss = _run_train_step(train_tensor, target_mask, preview, show_preview=False)
            return
        
        # 没有思维链，使用标准QA训练
        print(f"{GREEN}{answer}{RESET}", flush=True)
        train_tensor, target_mask, preview = _prepare_training_data(ask, answer, history_context)
        if train_tensor is None:
            return
        current_loss = _run_train_step(train_tensor, target_mask, preview, show_preview=False)
        return


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
    
    # 【新增】输入验证
    if not text or not isinstance(text, str):
        return "无效输入"
    
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
        # 直接进入标准生成
        current_prompt = prompt.clone()
        
        # 【新增】如果启用思维链，检查当前prompt是否包含THINK_START_TOKEN
        thinking_started = False
        if thinking_available:
            # 检查当前prompt是否已经包含THINK_START_TOKEN
            has_think_token = (current_prompt == TextTokenizer.THINK_START_TOKEN).any()
            if has_think_token:
                # 如果prompt中存在THINK_START_TOKEN，则认为思考已经开始
                thinking_started = True
            else:
                # 如果没有THINK_START_TOKEN，则手动添加
                thinking_started = True
                # 添加THINK_START_TOKEN到prompt
                think_start_tensor = torch.tensor([TextTokenizer.THINK_START_TOKEN], device=device)
                current_prompt = torch.cat([current_prompt, think_start_tensor])
        
        # 继续标准自回归生成
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


def _run_train_step(train_tensor: torch.Tensor, target_mask: torch.Tensor, preview: torch.Tensor, advantage_weight: float = 1.0, show_preview: bool = True, preview_color: str = None) -> float:
    """执行单步训练
    
    Args:
        train_tensor: 训练张量
        target_mask: 目标掩码
        preview: 预览张量
        advantage_weight: 优势加权因子(用于GRPO)
        show_preview: 是否显示预览输出(默认True,QA模式下可设为False避免重复)
        preview_color: 预览文本颜色(可选)
    
    Returns:
        当前训练步骤的损失值
    """
    model.train()
    
    # 【新增】训练前梯度清理
    optimizer.zero_grad()
    
    # 【新增】损失裁剪（在backward之前）
    if advantage_weight > 10.0:
        advantage_weight = 1.0

    # 【修改】更安全的autocast配置
    with torch.autocast(device_type="cuda", dtype=torch.float32 if amp_dtype == torch.bfloat16 else torch.float32, enabled=use_amp):
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
                # 【修复】增强 NaN 诊断
                if torch.isnan(masked_logits).any() or torch.isinf(masked_logits).any():
                    # 【诊断】打印输入统计信息，定位是哪类样本导致
                    print(f"[Warning] NaN/Inf in logits. "
                          f"train_tensor range: [{train_tensor.min()}, {train_tensor.max()}], "
                          f"seq_len: {len(train_tensor)}, "
                          f"preview: {TextTokenizer.decode(preview[:50])[:100]}", flush=True)
                    return float('inf')
                
                if torch.isnan(masked_targets).any() or torch.isinf(masked_targets).any():
                    print(f"[Warning] NaN or Inf detected in targets, skipping this step", flush=True)
                    return float('inf')
                
                loss = loss_func(masked_logits, masked_targets)
                
                # 检查loss是否为NaN
                if torch.isnan(loss):
                    print(f"[Warning] NaN loss detected, skipping this step", flush=True)
                    return float('inf')
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
        
        # 【新增】梯度裁剪前检查
        scaler.unscale_(optimizer)
        
        # 【新增】梯度裁剪后检查NaN
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if torch.isnan(grad_norm):
            optimizer.zero_grad()
            print(f"[Warning] NaN gradient detected, skipping optimizer step", flush=True)
            return float('inf')
        
        scaler.step(optimizer)
        scaler.update()
    else:
        print(f"[Warning] Invalid loss detected: {loss}, skipping optimizer step", flush=True)
        return float('inf')

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
    
    return loss.item()