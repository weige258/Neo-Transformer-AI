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
    "dropout": 0.05,

    # ═══════════════════════════════════════════════════════
    # 2️⃣ 注意力机制配置（动态系数，无固定阈值）
    # ═══════════════════════════════════════════════════════
    "attention_mix": {
        "csa": 1.0,
        "sliding_window": 1.0,
        "mla": 1.0,
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
    "use_gradient_checkpointing": True,
    "gpu_cache_clear_threshold_gb": 4.0,
    "max_forward_chunk": 512,
    
    # ── 显存硬限制配置 ──
    "max_recent_kv_len": 512,        # recent_k/v 绝对最大长度（2048→512，防显存爆炸）
    "max_total_kv_len": 1024,        # 拼接后 raw_k/v 绝对最大长度（4096→1024）
    "kv_cache_max_len": 512,         # _detach_kv_cache 最大长度（1024→512）

    # ═══════════════════════════════════════════════════════
    # 4️⃣ 序列长度与显存管理（全动态计算）
    # ═══════════════════════════════════════════════════════
    # 最大生成长度完全运行时动态计算：
    #   max_len = f(question_len, question_complexity, gpu_free_memory, cpu_free_memory, generation_entropy_history)
    # 系数说明：
    #   - gen_len_base_ratio: 基础长度 = question_len * base_ratio
    #   - gen_len_complexity_factor: 复杂度调节因子
    #   - gen_len_memory_sensitivity: 显存敏感度（显存紧张时降低长度）
    #   - gen_len_entropy_sensitivity: 生成熵敏感度（高熵时允许更长生成）
    "gen_len_base_ratio": 8.0,        # 基础长度倍数（32→8，防生成过长卡死）
    "gen_len_complexity_factor": 1.5,  # 复杂度因子
    "gen_len_memory_sensitivity": 0.3, # 显存敏感度（0.15→0.3，显存紧张时更积极缩短）
    "gen_len_entropy_sensitivity": 0.5, # 熵敏感度
    "gen_len_min_absolute": 64,       # 绝对最小长度（256→64）
    "gen_len_max_absolute": 2048,     # 绝对最大长度（4096→2048，防生成卡死）
    "dynamic_segment_overlap": 32,
    "gpu_memory_safe_ratio": 0.85,
    "gpu_memory_skip_ratio": 0.80,     # 显存跳过阈值（0.92→0.80，更早保护）

    # ═══════════════════════════════════════════════════════
    # 5️⃣ 生成采样策略
    # ═══════════════════════════════════════════════════════
    "temp_base": 0.7,
    "temp_entropy_scale": 0.4,
    "temp_repetition_sensitivity": 0.6,
    "temp_length_decay": 0.001,
    "temp_min_clip": 0.3,
    "temp_max_clip": 1.5,
    "enable_edt": True,
    "min_p": 0.04,
    "top_k": 50,
    "top_p": 0.9,
    "force_thinking_chain": True,
    # ═══════════════════════════════════════════════════════
    # 6️⃣ 重复惩罚
    # ═══════════════════════════════════════════════════════
    "rep_penalty_scale": 0.25,
    "rep_penalty_length_factor": 0.002,
    "rep_penalty_repeat_sensitivity": 2.0,
    "rep_penalty_entropy_factor": 0.8,
    "frequency_penalty": 0.3,
    "presence_penalty": 0.1,
    "force_answer_scale": 8.0,
    "force_answer_min_absolute": 128,
    "force_answer_complexity_exp": 0.5,

    # ═══════════════════════════════════════════════════════
    # 7️⃣ 学习率调度配置
    # ═══════════════════════════════════════════════════════
    "gradient_accumulation_steps": 4,    # 梯度累积步数（1→4，减少显存占用）
    "base_learning_rate": 3e-4,
    "warmup_steps": 300,
    "warmup_init_lr": 1e-6,
    "sgdr_t_0": 1500,
    "sgdr_t_mult": 2,
    "sgdr_eta_min": 1e-6,
    "reduce_lr_patience": 800,
    "reduce_lr_factor": 0.5,
    "reduce_lr_min_lr": 5e-6,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_eps": 1e-8,

    # ═══════════════════════════════════════════════════════
    # 8️⃣ 训练数据与采样配置
    # ═══════════════════════════════════════════════════════
    "max_seq_len": 2048,               # 最大序列长度（99999999→2048，防显存爆炸）
    "min_seq_len": 8,
    "packing_max_seq_len": 2048,       # 打包最大序列长度（99999999→2048）
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
    "chunk_memory_ratio": 0.15,       # chunk显存占比
    "chunk_seq_len_factor": 0.3,      # 序列长度因子
    "chunk_min_absolute": 128,        # 绝对最小chunk
    "chunk_max_ratio": 0.5,           # 最大比例
    "chunk_cpu_pressure_factor": 0.2,  # CPU压力因子
    "chunk_overlap_base": 32,         # 基础overlap
    "chunk_overlap_scale": 0.02,      # overlap缩放

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
    "rl_enabled": False,              # PPO默认禁用（防显存爆炸，手动开启）

    # ═══════════════════════════════════════════════════════
    # ⑪ 学习率调度器配置
    # ═══════════════════════════════════════════════════════
    # lr_scheduler步进间隔：每N个optimizer step才更新一次学习率
    #   - 防止大数据集下SGDR震荡过于频繁
    #   - 1 = 每个step都更新（旧行为），4 = 每4个step更新（推荐）
    "lr_scheduler_step_interval": 4,  # 学习率调度步进间隔

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