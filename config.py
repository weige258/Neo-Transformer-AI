from typing import Any, Dict

CONFIG: Dict[str, Any] = {
    # ═══════════════════════════════════════════════════════
    # 1️⃣ 模型架构参数
    # ═══════════════════════════════════════════════════════
    "dict_size": 60000,              # 词表大小
    "emb_size": 512,                 # 嵌入维度
    "num_heads": 8,                  # 注意力头数
    "num_transformer_blocks": 8,     # Transformer层数
    "tie_token_embeddings": True,    # 绑定输入输出嵌入权重
    "dropout": 0.05,                # 【修复】小模型 0.2 会欠拟合，降至 0.05
    
    # ═══════════════════════════════════════════════════════
    # 2️⃣ 注意力机制配置
    # ═══════════════════════════════════════════════════════
    "attention_mix": {               # 注意力混合权重（4路）
        "compressed": 2,             # 压缩注意力权重
        "sparse": 1.3,               # 稀疏注意力权重
        "dynamic": 1,                # 动态注意力权重
        "mla_latent_memory": 1.5,    # MLA latent KV 压缩权重
    },
    "sliding_window": 256,           # 【修复】从128→256，增大滑动窗口改善长文本上下文覆盖
    "attention_chunk_size": 128,      # 【修复】从64→128，增大注意力块减少分块次数提升长序列质量
    "dynamic_attention_topk": 8,     # 动态注意力Top-K数量（6GB显存推荐8）
    "attention_sink_count": 4,       # 【StreamingLLM】Attention Sink保护数量，前N个token永不压缩
    
    # ═══════════════════════════════════════════════════════
    # 🔟 RoPE 位置编码扩展配置
    # ═══════════════════════════════════════════════════════
    # 基于 YaRN (Yet another RoPE extensioN method, Peng et al., 2023)
    # 和 "Scaling Laws of RoPE-based Extrapolation" (Liu et al., ICLR 2024)
    # 提高 base 值可显著增强长序列外推能力
    "rope_base": 10000,               # 【修复】恢复默认10000，仅训练后按需增大。Scaling Laws论文要求增大base后需微调
    "rope_factor": 1.0,               # 【YaRN】RoPE缩放因子，>1.0时进行位置插值（NTK-aware）
    "rope_max_seq_len": 4096,         # 【YaRN】模型训练时的最大序列长度
    "use_sink_token": False,            # 【修复】StreamingLLM sink token开关，默认关闭（训练后再开）
    
    # ═══════════════════════════════════════════════════════
    # 3️⃣ 历史上下文压缩
    # ═══════════════════════════════════════════════════════
    "compress_trigger_len": 2048,    # 【修复】压缩触发长度阈值，超过此值触发KV压缩（设为None禁用）
    "compress_trigger_entropy": 0.8, # 【修复】熵触发压缩阈值（0-1），低熵=确定性强时可安全压缩
    "compress_stride": 16,           # 压缩步长
    "compress_ratio": 0.25,          # 压缩比例（6GB显存推荐0.25，更激进的压缩）
    "special_token_anchor_count": 10, # 【新增】特殊Token锚定数量(0-9)，始终以完整精度保留
    "special_token_anchor_loss_coef": 0.05,  # 【新增】特殊Token锚定损失权重
    # 运行时显存优化开关
    "use_amp": True,                          # 是否启用自动混合精度（AMP）
    "use_gradient_checkpointing": True,       # 是否在Transformer block上启用梯度检查点
    "gpu_cache_clear_threshold_gb": 4.0,      # 当 reserved 显存超过此值（GB）时定期清理 cache（6GB显卡推荐4GB）
    
    # ═══════════════════════════════════════════════════════
    # 6️⃣ 显存与序列长度管理
    # ═══════════════════════════════════════════════════════
    "gpu_memory_safe_ratio": 0.85,   # 安全显存比例（6GB显卡推荐0.80-0.85）
    "gpu_memory_skip_ratio": 0.92,   # 跳过样本的显存比例阈值（高于此值跳过，不做截断）
    "record_interval": 1000,          # 每N步记录一次loss到record.txt
    
    # ═══════════════════════════════════════════════════════
    # 4️⃣ 生成采样策略 (Min-p Sampling — ICLR 2025 最新方案)
    # ═══════════════════════════════════════════════════════
    # Min-p 论文: "Turning Up the Heat" (Nguyen et al., ICLR 2025)
    # 核心: 动态截断阈值 = 最大概率 × min_p_ratio，天然过滤垃圾token
    # 已被 HuggingFace Transformers / VLLM 等主流框架采纳
    "temperature": 0.8,              # 【修复】小模型降温至0.8，减少噪声
    "min_p": 0.05,                   # Min-p 比例（0.05 为高质量推荐值）
    "top_k": 0,                      # 【修复】关闭top-k，min-p更优
    "top_p": 1.0,                    # 【修复】关闭top-p，min-p替代
    
    # ═══════════════════════════════════════════════════════
    # 5️⃣ 生成质量控制
    # ═══════════════════════════════════════════════════════
    "repetition_penalty": 1.15,      # 提高惩罚强度，抑制中文高频重复
    "repetition_stop_threshold": 8,  # 降低重复触发阈值，避免长串重复文本继续生成
    # ── CoT 与输出完整性保护 ──
    "force_answer_min_steps": 32,    # 【修复】强制回答步数，THINK_END后至少32步禁止END_TOKEN
    "max_consecutive_garbage": 3,    # 【新增】连续垃圾token数阈值（触发重采样/停止）
    
    # ═══════════════════════════════════════════════════════
    # 7️⃣ 学习率调度配置 (SGDR + ReduceLROnPlateau — 适用于无限循环训练)
    # ═══════════════════════════════════════════════════════
    # 设计理念：抛弃固定步数的 Cosine Decay（旧系统3000步后永久平躺的严重缺陷）
    # 改用两大工业级机制协同工作，天然适配 while True 无限训练：
    #
    # ① SGDR (Cosine Annealing with Warm Restarts)
    #    论文: Loshchilov & Hutter, ICLR 2017
    #    核心: LR 在每 T_0 步周期性余弦振荡 → 定期"重启"赋予模型探索能量
    #    每个后续周期的长度 = 前一个周期 × T_mult（逐渐变长，越来越精细）
    #    业界采用: fast.ai, HuggingFace Transformers, PyTorch 官方推荐
    #
    # ② ReduceLROnPlateau（安全网）
    #    核心: 监控 loss，当 loss 不再下降时自动将基准 LR 减半
    #    防止 SGDR 在高 LR 区间无意义地震荡
    #    业界采用: PyTorch 原生调度器，被广泛用于在线/持续学习
    #
    # ── 基础参数 ──
    "gradient_accumulation_steps": 1,            # 梯度累积步数，1=每样本更新
    "base_learning_rate": 3e-4,                  # 初始基准学习率（SGDR 周期峰值）
    "warmup_steps": 300,                         # 预热步数（线性从 init → base_lr）
    "warmup_init_lr": 1e-6,                      # 预热初始学习率
    # ── SGDR 参数 ──
    "sgdr_t_0": 1500,                            # 第一个余弦周期的步数（optimizer steps）
    "sgdr_t_mult": 2,                            # 周期倍增因子（周期逐渐变长: T0, T0×2, T0×4, ...）
    "sgdr_eta_min": 1e-6,                        # 每个周期内的最小学习率
    # ── ReduceLROnPlateau 参数 ──
    "plateau_patience": 500,                     # Loss 不下降的容忍步数（optimizer steps）
    "plateau_factor": 0.5,                       # 触发时 LR 乘以此因子（减半）
    "plateau_threshold": 0.01,                   # 判断 loss 改善的最小相对变化（1%）
    "plateau_cooldown": 300,                     # 降低 LR 后的冷却步数
    "plateau_min_lr": 1e-7,                      # 最低学习率（永不跌破此值）
    
    # ═══════════════════════════════════════════════════════
    # 8️⃣ 强化学习学习率配置
    # ═══════════════════════════════════════════════════════
    # 【优化】基于最新研究大幅降低 RL 学习率
    # 参考: "强化学习阶段的学习率需要设置得非常小（通常在1e-7到1e-6之间）"
    #       "PPO训练: learning_rate 5e-7"
    #       "GRPO训练: learning_rate 6e-7"
    "ppo_learning_rate": 5e-7,                   # 【优化】PPO策略网络学习率（从1e-5降低到5e-7）
    "ppo_min_learning_rate": 1e-8,               # 【优化】PPO最小学习率（从1e-7降低到1e-8）
    "ppo_warmup_steps": 200,                     # 【优化】PPO warmup步数（从100增加到200，更平滑）
    "grpo_learning_rate": 6e-7,                  # 【优化】GRPO学习率（从5e-6降低到6e-7）
    "ttrl_learning_rate": 5e-7,                  # 【优化】TTRL学习率（从1e-5降低到5e-7）
    
    # ═══════════════════════════════════════════════════════
    # 9️⃣ 优化器配置
    # ═══════════════════════════════════════════════════════
    "optimizer_type": "adamw",                   # 优化器类型: "adamw", "adam", "sgd"
    "weight_decay": 0.01,                        # 权重衰减（L2正则化）
    "adam_beta1": 0.9,                           # Adam/AdamW beta1参数
    "adam_beta2": 0.999,                         # Adam/AdamW beta2参数
    "adam_epsilon": 1e-8,                        # Adam/AdamW epsilon参数
    "max_grad_norm": 1.0,                        # 梯度裁剪最大范数

    # ═══════════════════════════════════════════════════════
    # 🔟 MLA latent KV 压缩记忆（DeepSeek-V2/V3 风格）
    # ═══════════════════════════════════════════════════════
    "mla_latent_memory_use_delta": True,      # MLA latent KV 的 Delta 更新开关
    "mla_latent_memory_compress_threshold": 0.85,  # GPU显存占用超过此比例时触发 MLA latent 压缩
    
    # ═══════════════════════════════════════════════════════
    # 1️⃣1️⃣ 智能KV压缩配置
    # ═══════════════════════════════════════════════════════
    "use_learned_pooling": True,          # 启用Learned Soft Pooling（可学习门控压缩）
    "use_kivi_quantization": True,        # 启用KIVI风格KV量化（CPU卸载时压缩）
    "kivi_key_bits": 4,                   # Key量化位数（4-bit, 保留离群值精度）
    "kivi_value_bits": 2,                 # Value量化位数（2-bit, 激进压缩）
    "max_mem_kv_capacity": 256,           # 压缩记忆 mem_kv 最大容量，超限部分固化到 MLA latent memory
    
    # ═══════════════════════════════════════════════════════
    # 【运行时】显存与推理优化开关（仅为配置项，实际需要库支持）
    # ═══════════════════════════════════════════════════════
    "memory_optimizations": {
        "use_flash_attention": False,   # 使用 FlashAttention 内核（需要安装 flash-attn）
        "use_bitsandbytes": False,      # 使用 bitsandbytes 量化/优化（需要安装）
        "use_deepspeed": False,         # 使用 DeepSpeed ZeRO / Offload（需要安装并配置）
        "quantize_level": "int8",      # 默认量化等级: "int8" | "int4" | "none"
    },
    
    # ═══════════════════════════════════════════════════════
    # 🔟 强化学习自动就绪评估
    # ═══════════════════════════════════════════════════════
    # 【优化】基于最新研究调整 RL 就绪评估参数
    # 参考: "RL 应在模型具备基础语言能力后启用，避免过早引入导致训练崩溃"
    #       "SFT → DPO → RL 的渐进式对齐是最佳实践"
    # 系统根据训练损失自动判断是否启用PPO，满足以下条件时自动激活：
    #   1. 最近N条loss平均值 < rl_loss_threshold
    #   2. 最近N条loss标准差 < rl_loss_stability_std_threshold
    #   3. 总训练轮数 >= rl_min_training_rounds
    "rl_loss_threshold": 1.5,                    # 【优化】Loss阈值从1.2提高到1.5，允许更早启用RL
    "rl_loss_stability_window": 10,              # 【优化】稳定性评估窗口从5增加到10，更准确
    "rl_loss_stability_std_threshold": 0.2,      # 【优化】Loss标准差阈值从0.15提高到0.2，更宽容
    "rl_min_training_rounds": 100000,            # 【修复】先彻底禁用 PPO，等 SFT 收敛后再开
    "rl_check_interval": 100,                    # 【优化】RL就绪检查间隔从200降低到100，更及时
    
    # ═══════════════════════════════════════════════════════
    # 1️⃣1️⃣ 强化学习高级配置（基于最新研究新增）
    # ═══════════════════════════════════════════════════════
    # 【新增】KL 惩罚配置（防止策略偏离参考模型太远）
    # 参考: "kl_coef=0.1 是稳定性锚点，当loss震荡大时↑，当acc停滞不前时↓"
    "kl_coef": 0.1,                              # KL 惩罚系数（默认0.1）
    "kl_target": 0.01,                           # KL 散度目标值
    
    # 【新增】PPO 更新配置（基于 Alpha Auto Research 研究）
    # 参考: "mb_num / ppo_epochs 的作用不是让梯度更大，而是让同一批 rollout 被摊开到更多个优化器步"
    "ppo_epochs": 2,                             # PPO epoch 数（推荐2）
    "ppo_mini_batch_num": 4,                     # Mini-batch 数量（推荐4）
    # ppo_epochs * ppo_mini_batch_num = 8，决定"一批rollout能换来多少次参数更新"
    
    # 【新增】采样配置
    "grpo_group_size": 8,                        # GRPO 组内采样数（推荐8）
    "rollout_batch_size": 256,                   # Rollout batch size（推荐256）
}