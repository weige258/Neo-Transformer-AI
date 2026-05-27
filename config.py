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
    "dropout": 0.2,                 # Dropout比率
    
    # ═══════════════════════════════════════════════════════
    # 2️⃣ 注意力机制配置
    # ═══════════════════════════════════════════════════════
    "attention_mix": {               # 注意力混合权重
        "compressed": 2,             # 压缩注意力权重
        "sparse": 1.3,               # 稀疏注意力权重
        "dynamic": 1,                # 动态注意力权重
    },
    "sliding_window": 96,            # 滑动窗口大小
    "attention_chunk_size": 64,      # 注意力块大小
    "dynamic_attention_topk": 16,    # 动态注意力Top-K数量
    
    # ═══════════════════════════════════════════════════════
    # 3️⃣ 历史上下文压缩
    # ═══════════════════════════════════════════════════════
    "compress_trigger_len": 1200,    # 触发压缩的长度阈值
    "compress_trigger_entropy": 0.7, # 触发压缩的熵阈值
    "compress_stride": 16,           # 压缩步长
    "compress_ratio": 0.3,           # 压缩比例
    "compress_on_memory_ratio": 0.9,  # 当 GPU 显存占用超过该比例时触发压缩并卸载（0-1）
    
    # ═══════════════════════════════════════════════════════
    # 【新增】4️⃣ 序列长度限制（防止长文本显存爆炸）
    # ═══════════════════════════════════════════════════════
    # 移除硬性截断：优先在运行时触发向量压缩/卸载以避免显存爆炸
    # 注：已移除 RAG/本地向量检索相关配置；当历史过长或显存紧张时，
    # 会使用 `model.compress_history_vectors()` 生成压缩向量并卸载到 CPU/磁盘。
    "max_generation_len": 512,       # 最大生成长度限制（保留生成长限制）
    
    # ═══════════════════════════════════════════════════════
    # 5️⃣ 生成采样策略 (业界标准: Top-K + Top-P)
    # ═══════════════════════════════════════════════════════
    "temperature": 0.6,              # 温度参数：控制生成随机性
    "top_k": 50,                     # Top-K采样：只从最高概率的K个token采样
    "top_p": 0.95,                   # Top-P核采样：累积概率阈值，动态调整候选集
    
    # ═══════════════════════════════════════════════════════
    # 6️⃣ 生成质量控制 (重复惩罚 + 重复检测停止)
    # ═══════════════════════════════════════════════════════
    "repetition_penalty": 1.2,       # 重复惩罚系数：>1.0启用，防止复读机
    "repetition_stop_threshold": 5,  # 重复停止阈值：连续N个相同token或重复n-gram则停止
    
    # ═══════════════════════════════════════════════════════
    # 7️⃣ 学习率调度配置 (Warmup + Cosine Decay)
    # ═══════════════════════════════════════════════════════
    # 【优化】基于 2025-2026 最新研究调节 SFT 学习率参数
    # 参考: MiniMind 调优指南、GRPO 实战技巧、PPO Epochs 研究
    "base_learning_rate": 2e-4,                  # 基础学习率（峰值学习率）- 提高以便更快收敛（小模型/短训练）
    "min_learning_rate": 1e-6,                   # 最小学习率（衰减终点）
    "warmup_steps": 50,                          # Warmup步数（短warmup，适合小规模快速迭代）
    "warmup_init_lr": 1e-6,                      # Warmup初始学习率
    "cosine_decay_enabled": True,                # 是否启用余弦衰减
    "total_training_steps": 3000,                # 总训练步数（约30轮快速测试场景的默认值）
    "lr_scheduler_type": "cosine",               # 调度器类型: "cosine", "constant", "linear"
    
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
    "rl_min_training_rounds": 2000,              # 【优化】最低训练轮数从3000降低到2000，更早启用RL
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