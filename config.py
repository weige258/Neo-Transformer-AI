from typing import Any, Dict

CONFIG: Dict[str, Any] = {
    # ═══════════════════════════════════════════════════════
    # 1️⃣ 模型架构参数（架构固定，运行时动态调整行为）
    # ═══════════════════════════════════════════════════════
    "dict_size": 60000,
    "emb_size": 512,
    "num_heads": 8,
    "num_transformer_blocks": 8,
    "tie_token_embeddings": True,
    "dropout": 0.1,                   # 0.1→0.0，小数据不需要dropout                   

    # ═══════════════════════════════════════════════════════
    # 2️⃣ 注意力机制配置（动态系数，无固定阈值）
    # ═══════════════════════════════════════════════════════
    "attention_mix": {
        "csa": 0.1,
        "sliding_window": 5.0,
        "mla": 0.1,
    },
    # ── 动态窗口系数 ──
    # 滑动窗口大小完全运行时动态计算：
    #   window_size = f(seq_len, gpu_memory_pressure, cpu_load)
    # 系数说明：
    #   - window_scale_factor: 窗口大小 = seq_len * scale_factor（受显存压力调节）
    #   - window_min_ratio: 最小窗口占seq_len的比例（防止窗口过小）
    #   - window_full_attention_ratio: seq_len占总上下文比例低于此值时用full attention
    "window_scale_factor": 0.5,       # 窗口缩放因子（0.3-0.8动态范围）
    "window_min_ratio": 0.25,         # 最小窗口比例
    "window_full_attention_ratio": 0.125,  # full attention阈值比例
    "attention_chunk_size": 64,
    "dynamic_attention_topk": 8,
    "rope_base": 10000,
    "rope_factor": 1.0,
    "rope_max_seq_len": 4096,

    # ═══════════════════════════════════════════════════════
    # 3️⃣ 历史上下文压缩（动态系数，无固定压缩率）
    # ═══════════════════════════════════════════════════════
    # 压缩比例完全运行时动态计算：
    #   ratio = f(layer_depth, seq_len, gpu_memory_pressure, token_entropy)
    # 系数说明：
    #   - compress_scale: 压缩比例基础缩放系数
    #   - compress_depth_sensitivity: 层深度敏感度（影响深层/浅层差异程度）
    #   - compress_memory_pressure_factor: 显存压力对压缩的调节系数
    "compress_scale": 0.25,           # 压缩缩放系数
    "compress_depth_sensitivity": 2.0, # 层深度敏感度
    "compress_memory_pressure_factor": 1.5,  # 显存压力因子
    "compress_trigger_entropy": 0.6,
    "compress_stride": 16,
    "use_pyramid_compression": True,
    "compress_on_memory_ratio": 0.80,
    "prefer_gpu_compress": True,
    "max_mem_kv_capacity": 128,        # 压缩记忆容量（256→128，防显存膨胀）
    "h2_ratio": 0.3,
    "use_amp": True,
    "use_gradient_checkpointing": False,
    "gpu_cache_clear_threshold_gb": 4.0,
    "max_forward_chunk": 512,
    
    # ── 显存硬限制配置 ──
    # 【修复】适度放宽KV缓存限制，配合更大的max_seq_len
    "max_recent_kv_len": 1024,       # 512→1024，保留更多近期上下文
    "max_total_kv_len": 2048,        # 1024→2048，配合4096的max_seq_len
    "kv_cache_max_len": 1024,        # 512→1024，更宽松的KV缓存

    # ═══════════════════════════════════════════════════════
    # 4️⃣ 序列长度与显存管理（固定边界）
    # 最大生成长度由配置上限控制，不再运行时动态调整
    "gen_len_min_absolute": 32,       # 64→32，降低最低长度
    "gen_len_max_absolute": 512,      # 最大生成长度
    "dynamic_segment_overlap": 32,
    "gpu_memory_safe_ratio": 0.85,
    "gpu_memory_skip_ratio": 0.80,     # 显存跳过阈值（0.92→0.80，更早保护）

    # ═══════════════════════════════════════════════════════
    # 5️⃣ 生成采样策略
    # ═══════════════════════════════════════════════════════
    # 【修复】使用贪心解码：temperature=0表示argmax
    "temp_base": 0.8,                 # 1.0→0.8，略微降低温度
    "temp_entropy_scale": 0.0,
    "temp_repetition_sensitivity": 0.0,
    "temp_length_decay": 0.0,
    "temp_min_clip": 0.8,
    "temp_max_clip": 1.2,
    "enable_edt": False,
    "min_p": 0.0,                     # 禁用min-p（nanoGPT不用）
    "top_k": 50,                      # nanoGPT风格: 仅top-k
    "top_p": 1.0,                     # 禁用top-p（nanoGPT不用）
    "force_thinking_chain": True,
    "min_generation_steps_before_stop": 8,  # 4→8，更多步数禁止END
    # ═══════════════════════════════════════════════════════
    # 6️⃣ 重复惩罚
    # ═══════════════════════════════════════════════════════
    # 【修复】基于网络研究：repetition_penalty推荐1.05-1.15
    # 我们的rep_penalty_scale计算方式：repetition_penalty = 1.0 + scale * factor
    # 所以scale=0.1时，基础惩罚约1.1，符合推荐范围
    "rep_penalty_scale": 0.1,
    "rep_penalty_length_factor": 0.001,
    "rep_penalty_repeat_sensitivity": 1.0,
    "rep_penalty_entropy_factor": 0.0,
    "frequency_penalty": 1.0,
    "presence_penalty": 0.3,
    # 【修复】减少强制回答步数，避免过度强制
    "force_answer_scale": 2.0,
    "force_answer_min_absolute": 8,
    "force_answer_complexity_exp": 0.3,

    # ═══════════════════════════════════════════════════════
    # 7️⃣ 学习率调度配置
    # ═══════════════════════════════════════════════════════
    # 【修复】使用常数学习率，无warmup，无SGDR，无ReduceLROnPlateau
    # 原因：50轮训练步数太少，复杂调度器只会干扰收敛
    "gradient_accumulation_steps": 1,    # 2→1，每步都更新，最大化学习信号
    "base_learning_rate": 3e-4,          
    # 【移除】所有warmup/SGDR/ReduceLROnPlateau相关参数
    # "warmup_steps": 0,                  # 已移除
    # "warmup_init_lr": 5e-4,             # 已移除
    # "sgdr_t_0": ...                     # 已移除
    # "reduce_lr_patience": ...           # 已移除
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,                 # 0.999→0.95，更快响应新梯度
    "adam_eps": 1e-8,                    # 1e-6→1e-8，标准AdamW值

    # ═══════════════════════════════════════════════════════
    # 8️⃣ 训练数据与采样配置
    # ═══════════════════════════════════════════════════════
    # 【修复】提高最大序列长度，字符级tokenizer需要更长的上下文
    "max_seq_len": 4096,               # 2048→4096，字符级模型需要更长上下文
    "min_seq_len": 8,
    "packing_max_seq_len": 4096,       # 2048→4096，与max_seq_len一致
    "packing_buffer_size": 5000,
    "dataset_shuffle": True,
    "dataset_num_workers": 0,
    "train_data_path": "train_data.txt",
    "eval_data_path": "eval_data.txt",
    "eval_interval": 500,
    "eval_max_samples": 200,
    "checkpoint_interval": 1000,
    "checkpoint_keep_last_n": 3,
    "log_interval": 10,

    # ═══════════════════════════════════════════════════════
    # 9️⃣ 训练Chunk动态计算系数
    # ═══════════════════════════════════════════════════════
    # Chunk大小完全运行时动态计算：
    #   chunk_size = f(gpu_free_memory, gpu_total_memory, seq_len, cpu_free_memory, system_load)
    # 系数说明：
    #   - chunk_memory_ratio: 单chunk占用显存比例
    #   - chunk_seq_len_factor: 序列长度因子
    #   - chunk_min_absolute: 绝对最小chunk（防止过小）
    #   - chunk_max_ratio: 最大chunk占seq_len比例
    #   - chunk_cpu_pressure_factor: CPU压力调节因子
    # 【修复】提高chunk显存占比和序列因子，减少分块数量，保留更多上下文
    "chunk_memory_ratio": 0.25,       # 0.15→0.25，允许更大的chunk
    "chunk_seq_len_factor": 0.5,      # 0.3→0.5，更大的序列因子
    "chunk_min_absolute": 256,        # 128→256，提高最小chunk，减少分块数
    "chunk_max_ratio": 0.75,          # 0.5→0.75，允许更大的chunk比例
    "chunk_cpu_pressure_factor": 0.15, # 0.2→0.15，降低CPU压力影响
    "chunk_overlap_base": 64,         # 32→64，增加overlap保留更多上下文
    "chunk_overlap_scale": 0.03,      # 0.02→0.03，增加overlap缩放

    # ═══════════════════════════════════════════════════════
    # 🔟 强化学习配置（PPO稳定训练）
    # ═══════════════════════════════════════════════════════
    # PPO episode收集与更新策略：
    #   - rl_min_episodes: 收集至少N个episode后才更新策略（防止小样本方差大）
    #   - rl_update_batch_size: 策略更新时的batch size
    #   - rl_update_interval: 每N个training round检查一次是否满足更新条件
    "rl_min_episodes": 32,
    "rl_update_batch_size": 4,        # PPO更新batch size（8→4，减少显存）
    "rl_update_interval": 8,          # 更新检查间隔（4→8，降低PPO频率）
    "rl_enabled": True,              

    # ═══════════════════════════════════════════════════════
    # ⑪ 学习率调度器配置
    # ═══════════════════════════════════════════════════════
    # lr_scheduler步进间隔：每N个optimizer step才更新一次学习率
    #   - 防止大数据集下SGDR震荡过于频繁
    #   - 1 = 每个step都更新（旧行为），4 = 每4个step更新（推荐）
    # 【修复】降低调度间隔，让学习率更灵敏地响应训练状态
    "lr_scheduler_step_interval": 4,  

    # ═══════════════════════════════════════════════════════
    # ⑫ 系统与硬件监控
    # ═══════════════════════════════════════════════════════
    "device": "cuda",
    "seed": 42,
    "compile_model": False,
    "compile_mode": "default",
    "compile_dynamic": True,
    "use_tf32": True,
    "deterministic": False,
    "benchmark": True,
    "gpu_id": 0,
    "multi_gpu": False,
    "attention_sink_count": 4,
    "use_compile_generate": False,
}


def validate_config(cfg: Dict[str, Any]) -> list[str]:
    """运行时配置校验，返回警告列表"""
    warnings = []
    
    emb_size = int(cfg.get("emb_size", 512))
    num_heads = int(cfg.get("num_heads", 8))
    if emb_size % num_heads != 0:
        warnings.append(f"emb_size({emb_size}) must be divisible by num_heads({num_heads})")
    
    dict_size = int(cfg.get("dict_size", 60000))
    if dict_size < 10:
        warnings.append(f"dict_size({dict_size}) is too small, must be >= 10")
    
    num_blocks = int(cfg.get("num_transformer_blocks", 8))
    if num_blocks < 1:
        warnings.append(f"num_transformer_blocks({num_blocks}) must be >= 1")
    
    lr = float(cfg.get("base_learning_rate", 3e-4))
    if lr <= 0 or lr > 0.1:
        warnings.append(f"base_learning_rate({lr}) out of reasonable range (0, 0.1]")
    
    dropout = float(cfg.get("dropout", 0.05))
    if dropout < 0 or dropout >= 1:
        warnings.append(f"dropout({dropout}) must be in [0, 1)")
    
    gen_max = int(cfg.get("gen_len_max_absolute", 4096))
    gen_min = int(cfg.get("gen_len_min_absolute", 256))
    if gen_min > gen_max:
        warnings.append(f"gen_len_min_absolute({gen_min}) > gen_len_max_absolute({gen_max})")
    
    return warnings


_config_warnings = validate_config(CONFIG)
if _config_warnings:
    for w in _config_warnings:
        print(f"[Config Warning] {w}", flush=True)