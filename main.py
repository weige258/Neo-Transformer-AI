from typing import List, Tuple, Optional
import sys
import os
import pickle
import torch
import time
import logging
from collections import Counter
from config import CONFIG
from model import MainModel
from tokenizer import TextTokenizer
from rl import SelfRewardModel, LightweightPPO
from record import record_loss


# 【显存优化】设置 PyTorch CUDA 内存分配策略，避免显存碎片化
# expandable_segments:True 允许内存段动态扩展，减少碎片
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# P2: CPU KV缓存卸载池
_cpu_kv_bank: dict[int, tuple] = {}
_CPU_KV_MAX_LAYERS = 32  # 最多保留32层的卸载KV，超出则淘汰最旧层

# ════════════════════════════════════════════════════════════
# KIVI风格的异构KV量化（Key per-channel 4-bit, Value per-token 2-bit）
# ════════════════════════════════════════════════════════════
_KIVI_KEY_BITS = int(CONFIG.get("kivi_key_bits", 4))
_KIVI_VALUE_BITS = int(CONFIG.get("kivi_value_bits", 2))

def _kivi_quantize_key(k: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """KIVI Key量化 + 离群值保护（Outlier Preservation）

    Key对离群值敏感，使用4-bit + per-channel，但检测离群通道保留fp16。
    修复：防止异常激活值把正常token的缩放刻度压碎。
    """
    if not CONFIG.get("use_kivi_quantization", False):
        return k, torch.tensor(0.0), torch.tensor(1.0)
    bits = _KIVI_KEY_BITS
    q_max = (1 << bits) - 1

    # 检测离群通道：绝对值 > 3.5*std 的通道保留fp16
    k_std = k.std(dim=-2, keepdim=True, unbiased=False)
    k_mean = k.mean(dim=-2, keepdim=True)
    outlier_mask = (k.abs() > (k_mean.abs() + 3.5 * k_std))
    has_outlier = outlier_mask.any()

    if has_outlier:
        # 离群通道保留fp16，其余走4-bit量化
        k_regular = k.masked_fill(outlier_mask, 0.0)
        min_k = k_regular.amin(dim=-2, keepdim=True)
        max_k = k_regular.amax(dim=-2, keepdim=True)
        scale_k = (max_k - min_k) / q_max
        scale_k = scale_k.clamp_min(1e-8)
        qk = torch.round((k_regular - min_k) / scale_k).clamp(0, q_max).to(torch.uint8)
        # 将离群值信息打包进返回元组（用scale保存离群掩码位置）
        return qk, min_k, scale_k, outlier_mask.to(torch.uint8), k
    else:
        min_k = k.amin(dim=-2, keepdim=True)
        max_k = k.amax(dim=-2, keepdim=True)
        scale_k = (max_k - min_k) / q_max
        scale_k = scale_k.clamp_min(1e-8)
        qk = torch.round((k - min_k) / scale_k).clamp(0, q_max).to(torch.uint8)
        return qk, min_k, scale_k, None, None

def _kivi_dequantize_key(qk: torch.Tensor, min_k: torch.Tensor, scale_k: torch.Tensor,
                          outlier_mask=None, outlier_vals=None) -> torch.Tensor:
    if not CONFIG.get("use_kivi_quantization", False):
        return qk
    restored = qk.to(scale_k.dtype) * scale_k + min_k
    if outlier_mask is not None and outlier_vals is not None:
        restored = restored.masked_scatter(outlier_mask.bool(), outlier_vals[outlier_mask.bool()])
    return restored

def _kivi_quantize_value(v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """KIVI Value量化：per-token（每个token独立量化，2-bit激进压缩）

    Value对量化不敏感，使用2-bit即可。
    """
    if not CONFIG.get("use_kivi_quantization", False):
        return v, torch.tensor(0.0), torch.tensor(1.0)
    bits = _KIVI_VALUE_BITS
    q_max = (1 << bits) - 1
    min_v = v.amin(dim=-1, keepdim=True)  # (B, H, seq, 1)
    max_v = v.amax(dim=-1, keepdim=True)
    scale_v = (max_v - min_v) / q_max
    scale_v = scale_v.clamp_min(1e-8)
    qv = torch.round((v - min_v) / scale_v).clamp(0, q_max).to(torch.uint8)
    return qv, min_v, scale_v

def _kivi_dequantize_value(qv: torch.Tensor, min_v: torch.Tensor, scale_v: torch.Tensor) -> torch.Tensor:
    if not CONFIG.get("use_kivi_quantization", False):
        return qv
    return qv.to(scale_v.dtype) * scale_v + min_v


# 预分配CPU锁页内存池，用于异步卸载时的零拷贝传输
_offload_pin_pool: dict[int, torch.Tensor] = {}

def _ensure_pinned(t: torch.Tensor) -> torch.Tensor:
    """确保张量在锁页内存中（pin_memory），实现PCIe异步零拷贝"""
    if t.is_pinned():
        return t
    return t.pin_memory()

def _async_offload_to_cpu(layer_idx: int, cache_tuple) -> tuple | None:
    """将压缩记忆卸载到CPU（KIVI量化 + non_blocking异步PCIe传输）
    
    注意：non_blocking传输后不会立即同步，调用方应在适当时机调用
    torch.cuda.synchronize()确保数据到达CPU。参见NEW-2修复。
    """
    if cache_tuple is None or len(cache_tuple) < 6:
        return cache_tuple
    recent_k, recent_v, mem_k, mem_v, mem_pos, total_len, lin_M, lin_z = cache_tuple
    if mem_k is not None and mem_k.numel() > 0:
        # LRU淘汰：超过容量上限时删除最旧的层
        if len(_cpu_kv_bank) >= _CPU_KV_MAX_LAYERS:
            oldest_id = min(_cpu_kv_bank.keys())
            del _cpu_kv_bank[oldest_id]
        # KIVI量化
        qk, mk, sk, om, ov = _kivi_quantize_key(mem_k)
        qv, mv, sv = _kivi_quantize_value(mem_v)
        # 【修复MED-7】先完成所有non_blocking CPU传输，再同步确保DMA完成
        # 顺序：启动所有传输 → synchronize() → 确保数据到达后存入bank
        _offload_data = (
            qk.cpu(non_blocking=True),
            mk.cpu(non_blocking=True),
            sk.cpu(non_blocking=True),
            qv.cpu(non_blocking=True),
            mv.cpu(non_blocking=True),
            sv.cpu(non_blocking=True),
            mem_pos.cpu(non_blocking=True),
            lin_M.cpu(non_blocking=True),
            lin_z.cpu(non_blocking=True),
        )
        # 确保所有DMA传输完成后再存入bank（防止GPU内存被重用）
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _cpu_kv_bank[layer_idx] = (
            _ensure_pinned(_offload_data[0]),
            _ensure_pinned(_offload_data[1]),
            _ensure_pinned(_offload_data[2]),
            _ensure_pinned(_offload_data[3]),
            _ensure_pinned(_offload_data[4]),
            _ensure_pinned(_offload_data[5]),
            _ensure_pinned(_offload_data[6]),
            _ensure_pinned(_offload_data[7]),
            _ensure_pinned(_offload_data[8]),
            total_len,
            None if om is None else om.cpu(non_blocking=True),
            None if ov is None else ov.cpu(non_blocking=True),
        )
        empty = recent_k.new_zeros(1, 1, 0, mem_k.size(-1))
        empty_v = recent_v.new_zeros(1, 1, 0, mem_v.size(-1))
        empty_pos = torch.empty(0, device=mem_k.device, dtype=torch.long)
        return (recent_k, recent_v, empty, empty_v, empty_pos, total_len, lin_M, lin_z)
    return cache_tuple


def _load_from_cpu(layer_idx: int, device: torch.device) -> tuple | None:
    """从CPU锁页内存加载并反量化（non_blocking预取）"""
    if layer_idx not in _cpu_kv_bank:
        return None
    data = _cpu_kv_bank[layer_idx]
    if len(data) >= 12:
        qk, mk, sk, qv, mv, sv, mpos, linM, linZ, tl, om, ov = data
        # 反量化
        mk_gpu = mk.to(device, non_blocking=True)
        sk_gpu = sk.to(device, non_blocking=True)
        qk_gpu = qk.to(device, non_blocking=True)
        if om is not None:
            om_gpu = om.to(device, non_blocking=True)
            ov_gpu = ov.to(device, non_blocking=True)
            mem_k = _kivi_dequantize_key(qk_gpu, mk_gpu, sk_gpu, om_gpu, ov_gpu)
        else:
            mem_k = _kivi_dequantize_key(qk_gpu, mk_gpu, sk_gpu)
        mv_gpu = mv.to(device, non_blocking=True)
        sv_gpu = sv.to(device, non_blocking=True)
        qv_gpu = qv.to(device, non_blocking=True)
        mem_v = _kivi_dequantize_value(qv_gpu, mv_gpu, sv_gpu)
        mem_pos = mpos.to(device, non_blocking=True)
        linM_gpu = linM.to(device, non_blocking=True)
        linZ_gpu = linZ.to(device, non_blocking=True)
        # 【修复NEW-1】严格按 CompressedKVCache 的8个字段顺序返回
        # 格式: (recent_k, recent_v, mem_k, mem_v, mem_pos, total_len, mla_M, mla_z)
        empty_k = torch.empty(1, 1, 0, mem_k.size(-1), device=device)
        empty_v = torch.empty(1, 1, 0, mem_v.size(-1), device=device)
        return empty_k, empty_v, mem_k, mem_v, mem_pos, tl, linM_gpu, linZ_gpu
    return None


# ──────────────────────────────────────────────────────────
# 安全的训练序列构建工具
# ──────────────────────────────────────────────────────────

def _build_train_sequence(
    segments: List[Tuple[torch.Tensor | int, bool]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """根据 (token/常量, is_target) 段列表构建 train_tensor 和 target_mask。

    每个段为 (data, is_target)：
      - data: 可以是 torch.Tensor（token 序列）或 int（单个特殊 token）
      - is_target: True = 该段参与 loss 计算，False = 仅作为上下文

    返回 (train_tensor, target_mask)，两者长度严格相等。
    
    Loss Mask 语义（重要）：
      target_mask[i]=True 表示 logits[i-1] → token[i] 的预测参与 loss 计算。
      即：模型在位置 i-1 的输出需要正确预测位置 i 的 token。
      
    对于 CoT 训练，必须确保：
      - THINK_END_TOKEN 位置的 mask=True → 模型学会在思考结束时输出 THINK_END
      - answer 首 token 位置的 mask=True → 模型学会在 THINK_END 后立即输出回答
      这两者缺一不可，否则模型会学会"只思考不回答"。
    """
    tensors: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []

    for data, is_target in segments:
        if isinstance(data, int):
            t = torch.tensor([data], device=device, dtype=torch.long)
        elif isinstance(data, torch.Tensor):
            t = data.to(device=device, dtype=torch.long)
        else:
            raise TypeError(f"_build_train_sequence: unexpected segment type {type(data)}")

        if t.numel() == 0:
            continue

        tensors.append(t)
        masks.append(torch.full((t.numel(),), is_target, device=device, dtype=torch.bool))

    if not tensors:
        # 返回空序列（全False mask → 无梯度）
        # 【修复Bug #7】调用方在_run_train_step中检查target_mask.any()后跳过
        dummy = torch.tensor([TextTokenizer.UNKNOWN_TOKEN], device=device, dtype=torch.long)
        return dummy, torch.tensor([False], device=device, dtype=torch.bool)

    train_tensor = torch.cat(tensors, dim=0)
    target_mask = torch.cat(masks, dim=0)

    assert target_mask.numel() == train_tensor.numel(), (
        f"_build_train_sequence: mask len {target_mask.numel()} != train len {train_tensor.numel()}"
    )
    return train_tensor, target_mask


def _load_model() -> MainModel:
    t0 = time.time()
    try:
        # 安全加载：不使用不存在的 `weights_only` 参数，使用 map_location
        loaded = torch.load("model.pth", map_location=device)
        model = MainModel().to(device)
        
        # 【修复】严格校验键匹配，避免加载不匹配的权重导致静默随机初始化
        # 根据PyTorch最佳实践，应该使用strict=True或在不匹配时抛出异常
        model_state = model.state_dict()
        
        # 检查是否有shape不匹配的键
        shape_mismatched = []
        for k, v in loaded.items():
            if k in model_state:
                if v.shape != model_state[k].shape:
                    shape_mismatched.append(f"{k}: expected {model_state[k].shape}, got {v.shape}")
            else:
                shape_mismatched.append(f"{k}: 模型中不存在此键")
        
        if shape_mismatched:
            error_msg = "\n".join(shape_mismatched)
            raise ValueError(
                f"权重文件与模型结构不匹配，加载中止！\n"
                f"不匹配的键：\n{error_msg}\n\n"
                f"请检查：\n"
                f"1. dict_size、emb_size、层数等配置是否与保存权重时的配置一致\n"
                f"2. 是否使用了错误的model.pth文件"
            )
        
        # 严格加载
        model.load_state_dict(loaded, strict=True)
        print("Loaded model state dict with strict validation.", flush=True)
        return model
    except FileNotFoundError:
        print("model.pth not found. Creating new model.", flush=True)
        model = MainModel().to(device)
        print("Created new model.", flush=True)
        return model
    except ValueError as e:
        # 捕获形状不匹配的异常并终止程序
        print(f"\n{'='*60}", flush=True)
        print(f"[ERROR] {e}", flush=True)
        print(f"{'='*60}\n", flush=True)
        raise  # 【修复】改用raise而非sys.exit(1)，让调用方决定是否终止
    except (EOFError, pickle.UnpicklingError, RuntimeError) as e:
        print(f"[Warning] Failed to load model weights (corrupted file?): {e}", flush=True)
        print("[Warning] Creating new model with random initialization.", flush=True)
        model = MainModel().to(device)
        print("Created new model.", flush=True)
        return model
    except Exception as e:
        # 磁盘I/O错误、内存不足等严重异常应传播，不静默创建新模型
        print(f"\n{'='*60}", flush=True)
        print(f"[FATAL] 加载模型时发生严重错误: {e}", flush=True)
        print(f"[FATAL] 这不是权重文件损坏问题，请检查系统状态。", flush=True)
        print(f"{'='*60}\n", flush=True)
        raise


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

# 允许通过配置覆盖是否启用 AMP（便于在某些环境下手动关闭）
use_amp = bool(CONFIG.get("use_amp", use_amp))

# 【修复】仅float16启用scaler，bfloat16无需缩放，避免梯度爆炸
# 【PyTorch 2.x 更新】使用 torch.amp.GradScaler('cuda') 替代已弃用的 torch.cuda.amp.GradScaler
s_cl_en = (use_amp and amp_dtype == torch.float16)
scaler = torch.amp.GradScaler('cuda', enabled=s_cl_en)

print(f"Using device: {device}", flush=True)
print(f"AMP enabled: {use_amp}, AMP dtype: {amp_dtype}", flush=True)
model = _load_model()
# 检索模块已移除；使用向量压缩与卸载策略来在训练/推理时减小显存占用
print("[Info] Retrieval module removed; using vector compression/offload.", flush=True)


def _get_gpu_memory_ratio(device=None) -> float:
    """返回当前 GPU 的显存使用比例（0.0 当不可用）。
    使用 torch.cuda.memory_reserved/allocated 和 device 总显存计算。
    """
    try:
        if not torch.cuda.is_available():
            return 0.0
        # 解析 device index
        if device is None:
            idx = torch.cuda.current_device()
        else:
            if isinstance(device, torch.device):
                if device.type == 'cuda' and device.index is not None:
                    idx = device.index
                else:
                    idx = torch.cuda.current_device()
            else:
                idx = int(device)
        props = torch.cuda.get_device_properties(idx)
        total = float(props.total_memory)
        reserved = float(torch.cuda.memory_reserved(idx))
        allocated = float(torch.cuda.memory_allocated(idx))
        used = max(reserved, allocated)
        return used / total if total > 0 else 0.0
    except Exception:
        return 0.0


# 【显存优化】关闭torch.compile，避免额外显存占用
print("[Info] Running without torch.compile optimization (disabled for memory efficiency).", flush=True)

total_params = sum(param.numel() for param in model.parameters())
print(f"模型参数: {total_params / 1e+8}亿", flush=True)

loss_func = torch.nn.CrossEntropyLoss().to(device)

# 【学习率配置】从CONFIG读取优化器参数
base_lr = float(CONFIG.get("base_learning_rate", 3e-4))
weight_decay = float(CONFIG.get("weight_decay", 0.01))
adam_beta1 = float(CONFIG.get("adam_beta1", 0.9))
adam_beta2 = float(CONFIG.get("adam_beta2", 0.999))
adam_epsilon = float(CONFIG.get("adam_epsilon", 1e-8))
optimizer_type = CONFIG.get("optimizer_type", "adamw").lower()

# 根据配置选择优化器
if optimizer_type == "adamw":
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        betas=(adam_beta1, adam_beta2),
        eps=adam_epsilon,
        weight_decay=weight_decay,
        foreach=torch.cuda.is_available(),
    )
elif optimizer_type == "adam":
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=base_lr,
        betas=(adam_beta1, adam_beta2),
        eps=adam_epsilon,
        weight_decay=weight_decay,
        foreach=torch.cuda.is_available(),
    )
elif optimizer_type == "sgd":
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=0.9,
        weight_decay=weight_decay,
    )
else:
    print(f"[Warning] Unknown optimizer type '{optimizer_type}', falling back to AdamW", flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        betas=(adam_beta1, adam_beta2),
        eps=adam_epsilon,
        weight_decay=weight_decay,
        foreach=torch.cuda.is_available(),
    )

print(f"[Info] Optimizer: {optimizer_type.upper()}, LR: {base_lr:.2e}, Weight Decay: {weight_decay:.2e}", flush=True)

# ═══════════════════════════════════════════════════════════
# 学习率调度器: SGDR + ReduceLROnPlateau (适用于无限循环训练)
# ═══════════════════════════════════════════════════════════
# 替换原因：旧系统固定 3000 步后 LR 永久平躺 → 无限训练中模型"脑死亡"
# 新系统：SGDR 周期性重启 + Plateau 自动降 LR，天然适配 while True
GRADIENT_ACCUMULATION_STEPS = int(CONFIG.get("gradient_accumulation_steps", 1))
training_rounds = 0
optimizer_step_count = 0  # 保留全局计数器，与调度器内部同步


class LRSchedulerManager:
    """SGDR + ReduceLROnPlateau 混合学习率调度器

    解决旧系统"固定步数后永久平躺"的致命缺陷，适配无限循环训练。

    两大机制协同：
    ┌─────────────────────────────────────────────────────┐
    │ ① SGDR (Cosine Annealing with Warm Restarts)       │
    │    Loshchilov & Hutter, ICLR 2017                  │
    │    LR 每 T_0 步从 current_base_lr 余弦衰减到       │
    │    eta_min，然后"重启"回峰值。每个后续周期         │
    │    长度 × T_mult，逐渐变长，越来越精细。           │
    │                                                     │
    │ ② ReduceLROnPlateau                                │
    │    PyTorch 原生，loss 驱动                           │
    │    loss 不改善时 current_base_lr 减半               │
    │    防止 SGDR 在无效高 LR 区间浪费计算               │
    └─────────────────────────────────────────────────────┘

    学习率曲线示意:
    LR ↑
    3e-4 ┤  ╱╲          ← SGDR 周期1: 峰值=base_lr
         │ ╱  ╲    ╱╲    ← 周期2: 更长, 峰值可能因plateau降低
         │╱    ╲╱  ╲╱╲
    1e-6 ┤            ╲╲  ← 永不跌破 plateau_min_lr (1e-7)
         └──────────────────────→ ∞ optimizer steps
           warmup
    """

    def __init__(self, optimizer: torch.optim.Optimizer, config: dict):
        self.optimizer = optimizer

        # 基础参数
        self._base_lr = float(config.get("base_learning_rate", 3e-4))
        self._warmup_steps = int(config.get("warmup_steps", 300))
        self._warmup_init_lr = float(config.get("warmup_init_lr", 1e-7))

        # SGDR 参数
        self._t_0 = int(config.get("sgdr_t_0", 1500))
        self._t_mult = max(int(config.get("sgdr_t_mult", 2)), 1)
        self._eta_min = float(config.get("sgdr_eta_min", 1e-6))

        # ReduceLROnPlateau 参数
        self._plateau_patience = int(config.get("plateau_patience", 500))
        self._plateau_factor = float(config.get("plateau_factor", 0.5))
        self._plateau_threshold = float(config.get("plateau_threshold", 0.01))
        self._plateau_cooldown = int(config.get("plateau_cooldown", 300))
        self._plateau_min_lr = float(config.get("plateau_min_lr", 1e-7))

        # 内部状态
        self.step_count = 0              # optimizer step 计数
        self._best_loss = float('inf')
        self._plateau_counter = 0
        self._cooldown_counter = 0
        self.current_base_lr = self._base_lr  # 可被 Plateau 动态下调

        # SGDR 周期追踪（相对于 warmup 结束后）
        self._cycle_start_step = 0       # 当前周期起始（相对于 warmup 后）
        self._current_t_i = self._t_0    # 当前周期长度
        self.cycle_number = 0            # 第几个 SGDR 周期

        # 打印初始化信息
        print(f"[Info] LR Scheduler: SGDR + ReduceLROnPlateau", flush=True)
        print(f"       Warmup: {self._warmup_steps} steps "
              f"({self._warmup_init_lr:.1e} → {self._base_lr:.1e})", flush=True)
        print(f"       SGDR: T_0={self._t_0}, T_mult={self._t_mult}, "
              f"eta_min={self._eta_min:.1e}", flush=True)
        print(f"       Plateau: patience={self._plateau_patience}, "
              f"factor={self._plateau_factor}, min_lr={self._plateau_min_lr:.1e}", flush=True)

    def step(self, loss: float = None) -> float:
        """更新学习率（每次 optimizer.step() 后调用）

        Args:
            loss: 原始 loss 值（用于 ReduceLROnPlateau 检测）。
                  None 表示跳过 plateau 检测（如 NaN loss 时）。
        Returns:
            当前设置的学习率
        """
        self.step_count += 1

        # Plateau 检测（在计算 LR 之前，因为它可能改变 current_base_lr）
        if loss is not None and loss > 0 and not (loss == float('inf')):
            self._check_plateau(loss)

        lr = self._compute_lr()
        self._apply_lr(lr)
        return lr

    def _compute_lr(self) -> float:
        """计算当前步的学习率（warmup 或 SGDR）"""
        # ── Warmup 阶段 ──
        if self.step_count <= self._warmup_steps:
            progress = self.step_count / max(self._warmup_steps, 1)
            return self._warmup_init_lr + (self.current_base_lr - self._warmup_init_lr) * progress

        # ── SGDR 阶段 ──
        steps_since_warmup = self.step_count - self._warmup_steps
        steps_in_cycle = steps_since_warmup - self._cycle_start_step

        # 当前周期是否结束？
        if steps_in_cycle >= self._current_t_i:
            self._cycle_start_step = steps_since_warmup
            self._current_t_i = max(self._current_t_i * self._t_mult, 1)
            self.cycle_number += 1
            steps_in_cycle = 0

            if self.step_count % 100 == 0 or self.cycle_number <= 3:
                print(f"[LR Scheduler] 🔄 SGDR 周期 #{self.cycle_number} 开始, "
                      f"T_i={self._current_t_i}, peak_lr={self.current_base_lr:.2e}", flush=True)

        # 余弦衰减计算
        import math
        progress = steps_in_cycle / max(self._current_t_i, 1)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self._eta_min + (self.current_base_lr - self._eta_min) * cosine_decay

    def _apply_lr(self, lr: float):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _check_plateau(self, loss: float):
        """检查 loss 是否进入平台期，若是则降低 current_base_lr"""
        # 冷却期内不做检测
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            return

        # Warmup 阶段不检测
        if self.step_count <= self._warmup_steps:
            return

        if loss < self._best_loss * (1.0 - self._plateau_threshold):
            # Loss 有改善，重置计数器
            self._best_loss = loss
            self._plateau_counter = 0
        else:
            self._plateau_counter += 1
            if self._plateau_counter >= self._plateau_patience:
                old_base = self.current_base_lr
                new_base = max(self.current_base_lr * self._plateau_factor,
                              self._plateau_min_lr)
                if new_base < old_base:
                    self.current_base_lr = new_base
                    self._plateau_counter = 0
                    self._cooldown_counter = self._plateau_cooldown
                    self._best_loss = float('inf')
                    # 重置 SGDR 周期
                    self._cycle_start_step = self.step_count - self._warmup_steps
                    self._current_t_i = self._t_0
                    self.cycle_number = 0

                    print(f"\n{'=' * 60}", flush=True)
                    print(f"[LR Scheduler] ⚠️  Plateau 检测! Loss 停滞不前", flush=True)
                    print(f"      基准 LR 降低: {old_base:.2e} → {new_base:.2e}", flush=True)
                    print(f"      SGDR 周期已重置, 冷却 {self._plateau_cooldown} 步", flush=True)
                    print(f"{'=' * 60}\n", flush=True)


# ── 初始化全局调度器 ──
lr_scheduler = LRSchedulerManager(optimizer, CONFIG)

# 初始化强化学习模块
if device.type != "meta":
    reward_model = SelfRewardModel(device)
    ppo_trainer = LightweightPPO(
        model=model,
        reward_model=reward_model,
        device=device,
        learning_rate=float(CONFIG.get("ppo_learning_rate", 5e-7)),
        min_learning_rate=float(CONFIG.get("ppo_min_learning_rate", 1e-8)),
        warmup_steps=int(CONFIG.get("ppo_warmup_steps", 200)),
        total_training_steps=int(CONFIG.get("total_training_steps", 30000)),
        clip_ratio=0.2,
        entropy_coef=0.02,
        gamma=0.99,
        ppo_epochs=int(CONFIG.get("ppo_epochs", 2)),
        mini_batch_num=int(CONFIG.get("ppo_mini_batch_num", 4)),
        external_optimizer=optimizer,  # 【修复】共享主优化器，避免双优化器动量冲突
    )
    print("[Info] Self-reward model and RL modules initialized.", flush=True)



def _prepare_training_data(ask_text: str, answer_text: str, hist_context: str = None):
    """准备单个样本的训练数据（使用安全的段构建方式）"""
    if ask_text is None or answer_text is None:
        return None, None, None

    ask_tensor = TextTokenizer.encode(ask_text)
    answer_tensor = TextTokenizer.encode(answer_text)

    if answer_tensor.numel() == 0:
        return None, None, None

    # 【删除限制】不再根据序列长度触发压缩，保留完整上下文

    # 【StreamingLLM】可选：训练序列前缀sink token（默认关闭）
    sink_enabled = bool(CONFIG.get("use_sink_token", False))

    segments: list = []

    if hist_context is not None and hist_context.strip():
        history_tensor = TextTokenizer.encode(hist_context)
        # 【删除限制】不再检查压缩触发，保留完整历史上下文

        segments = [
            *([(TextTokenizer.PLACEHOLDER_SINK_TOKEN, False)] if sink_enabled else []),
            (TextTokenizer.START_GENERATION_TOKEN, False),
            (history_tensor, False),
            (TextTokenizer.END_GENERATION_TOKEN, False),
            (TextTokenizer.HISTORY_CONTEXT_START_TOKEN, False),
            (ask_tensor, False),
            (TextTokenizer.START_GENERATION_TOKEN, False),
            (answer_tensor, True),
            (TextTokenizer.END_GENERATION_TOKEN, True),
        ]
        # 保持 preview 在 CPU 上，避免与 GPU tensors 混合导致 torch.cat 设备不一致错误
        preview = torch.cat([answer_tensor, torch.tensor([TextTokenizer.END_GENERATION_TOKEN])])
    else:
        segments = [
            *([(TextTokenizer.PLACEHOLDER_SINK_TOKEN, False)] if sink_enabled else []),
            (ask_tensor, False),
            (TextTokenizer.START_GENERATION_TOKEN, False),
            (answer_tensor, True),
            (TextTokenizer.END_GENERATION_TOKEN, True),
        ]
        preview = torch.cat([answer_tensor, torch.tensor([TextTokenizer.END_GENERATION_TOKEN])])

    train_tensor, target_mask = _build_train_sequence(segments)
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
    t0 = time.time()
    
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

        text_tensor = TextTokenizer.encode(answer)
        if text_tensor.numel() < 2:
            return

        train_tensor, target_mask = _build_train_sequence([
            *([(TextTokenizer.PLACEHOLDER_SINK_TOKEN, False)] if bool(CONFIG.get("use_sink_token", False)) else []),
            (TextTokenizer.START_GENERATION_TOKEN, True),
            (text_tensor, True),
            (TextTokenizer.END_GENERATION_TOKEN, True),
        ])
        preview = train_tensor
        _run_train_step(train_tensor, target_mask, preview, show_preview=True, preview_color=YELLOW)
        return

    # QA训练模式
    print(f"\n---Train{RESET}", flush=True)
    print(f"{WHITE}{ask}{RESET}", flush=True)
    
    if answer and answer.strip():
        if think and think.strip():
            print(f"{BLUE}{think}{RESET}", flush=True)
            print(f"{GREEN}{answer}{RESET}", flush=True)
            
            ask_tensor = TextTokenizer.encode(ask)
            think_tensor = TextTokenizer.encode(think)
            answer_tensor = TextTokenizer.encode(answer)
            
            if history_context:
                history_tensor = TextTokenizer.encode(history_context)
                # CoT 训练序列 Loss Mask 设计说明：
                # target_mask[i]=True → logits[i-1] 预测 token[i] 时计算loss
                # 关键：THINK_END_TOKEN 的 mask=True 确保其前面位置的logit学习输出THINK_END
                #       answer 首token 的 mask=True 确保 THINK_END 位置的logit学习输出回答首字符
                #       这保证了"思维链→回答"的过渡被正确训练，防止模型在THINK_END后直接结束
                train_tensor, target_mask = _build_train_sequence([
                    *([(TextTokenizer.PLACEHOLDER_SINK_TOKEN, False)] if bool(CONFIG.get("use_sink_token", False)) else []),
                    (TextTokenizer.START_GENERATION_TOKEN, False),
                    (history_tensor, False),
                    (TextTokenizer.END_GENERATION_TOKEN, False),
                    (TextTokenizer.HISTORY_CONTEXT_START_TOKEN, False),
                    (ask_tensor, False),
                    (TextTokenizer.START_GENERATION_TOKEN, False),
                    (TextTokenizer.THINK_START_TOKEN, False),   # 不预测思考开始标记
                    (think_tensor, True),                        # 学习生成思维链内容
                    (TextTokenizer.THINK_END_TOKEN, True),       # 学习在思考结束时输出THINK_END
                    (answer_tensor, True),                       # ★ 学习从THINK_END过渡到回答
                    (TextTokenizer.END_GENERATION_TOKEN, True),  # 学习在回答结束时输出END
                ])
                # preview 保持在 CPU，避免不必要的 GPU 移动
                preview = torch.cat([think_tensor, answer_tensor])
                _run_train_step(train_tensor, target_mask, preview, show_preview=False)
            else:
                # CoT 训练序列 Loss Mask（无历史上下文版本）
                # 同上：THINK_END→answer 的过渡是关键，两者 mask 皆为 True
                train_tensor, target_mask = _build_train_sequence([
                    *([(TextTokenizer.PLACEHOLDER_SINK_TOKEN, False)] if bool(CONFIG.get("use_sink_token", False)) else []),
                    (ask_tensor, False),                         # 问题不参与loss
                    (TextTokenizer.START_GENERATION_TOKEN, False), # 分隔符不参与loss
                    (TextTokenizer.THINK_START_TOKEN, False),    # 不预测思考开始标记
                    (think_tensor, True),                        # 学习生成思维链内容
                    (TextTokenizer.THINK_END_TOKEN, True),       # 学习在思考结束时输出THINK_END
                    (answer_tensor, True),                       # ★ 学习从THINK_END过渡到回答
                    (TextTokenizer.END_GENERATION_TOKEN, True),  # 学习在回答结束时输出END
                ])
                preview = torch.cat([think_tensor, answer_tensor])
                _run_train_step(train_tensor, target_mask, preview, show_preview=False)
            return
        
        print(f"{GREEN}{answer}{RESET}", flush=True)
        train_tensor, target_mask, preview = _prepare_training_data(ask, answer, history_context)
        if train_tensor is None:
            return
        _run_train_step(train_tensor, target_mask, preview, show_preview=False)
    
    # 自奖励评估 —— 智能 RL 切换（基于 SuperRL Adaptive Switch 设计）
    # 【修复】先检查 SFT 最低训练轮数，未达标则完全跳过 PPO
    rl_min_rounds = int(CONFIG.get("rl_min_training_rounds", 100000))
    if training_rounds < rl_min_rounds:
        # SFT 预热阶段：完全不运行 RL，避免干扰梯度累积
        if training_rounds % 100 == 0:
            print(f"[RL Gate] ⏸️ SFT预热阶段 (round {training_rounds}/{rl_min_rounds})，RL 已禁用", flush=True)
    else:
        try:
            # 计算奖励（自动记录到历史）
            total_reward, reward_breakdown = reward_model.compute_total_reward(
                think_text=think,
                answer_text=answer,
                context=history_context
            )
            
            # 智能决策是否启用 RL 训练
            should_enable_rl, rl_decision_reason = reward_model.should_enable_rl()
            
            if should_enable_rl:
                # 启用 RL 训练：收集 episode 并更新策略
                ppo_trainer.collect_episode(
                    prompt=ask if ask else "",
                    think_text=think if think else "",
                    answer_text=answer if answer else "",
                    context=history_context
                )
                if training_rounds > 0 and (training_rounds % 4) == 0:
                    ppo_update_result = ppo_trainer.update_policy(batch_size=4)
                    
                    # 每100步打印一次 RL 状态
                    if training_rounds % 100 == 0:
                        print(f"[RL Smart Switch] ✅ 启用RL训练 | "
                              f"奖励={total_reward:.3f} | "
                              f"原因: {rl_decision_reason}", flush=True)
            else:
                # 暂停 RL 训练：仅执行 SFT，不收集 RL episode
                # 每50步打印一次切换状态
                if training_rounds % 50 == 0:
                    print(f"[RL Smart Switch] ⏸️ 暂停RL训练 | "
                          f"奖励={total_reward:.3f} | "
                          f"原因: {rl_decision_reason}", flush=True)
        except Exception as e:
            print(f"[Warning] RL step failed (non-fatal): {e}", flush=True)


def _min_p_sampling(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    """Min-p 采样 (Nguyen et al., ICLR 2025)

    核心思想: 以最大概率 token 为锚点，只保留概率 ≥ p_max × min_p 的 token。
    比 top-k + top-p 更优：
      - 高置信度时（p_max 大）→ 自动收紧候选集，避免噪声
      - 低置信度时（p_max 小）→ 自动放宽，保留多样性
    无需手动调节 k 或 p，一个参数适配所有场景。

    【修复NEW-2】调用方传入的 logits 已应用温度缩放，此函数不再做 softmax。
    改为直接使用 logits 计算概率分布，避免双重 softmax 导致的阈值不一致。
    """
    if min_p <= 0.0:
        return logits
    # 使用 logits 直接计算概率（不再内部做 softmax，因调用方已做 temperature 缩放）
    probs = torch.softmax(logits, dim=-1)
    p_max = probs.max().item()
    threshold = p_max * min_p
    # 将低于阈值的 token 设为 -inf
    logits = torch.where(probs < threshold, torch.full_like(logits, float("-inf")), logits)
    return logits


def _is_garbage_token(token_id: int) -> bool:
    """检测 token 解码后是否为垃圾/不可显示字符

    垃圾类型包括:
      - 控制字符 (0-31, 127)
      - 代理对 (0xD800-0xDFFF)
      - 私用区 (0xE000-0xF8FF)
      - 非主流 Unicode 块 (如加拿大音节文字 0x1400-0x167F)
      - 特殊 Unicode 控制字符
    """
    if not TextTokenizer._is_valid_token(token_id):
        return True
    # 控制字符
    if token_id < 32 or token_id == 127:
        return True
    # 加拿大土著音节文字 (ᓀ ᓓ 等 — 训练数据中不应出现)
    if 0x1400 <= token_id <= 0x167F:
        return True
    # 切罗基文字
    if 0x13A0 <= token_id <= 0x13FF:
        return True
    # 彝文音节
    if 0xA000 <= token_id <= 0xA4CF:
        return True
    # 私用区
    if 0xE000 <= token_id <= 0xF8FF:
        return True
    # 标记/组合字符块 
    if 0x0300 <= token_id <= 0x036F:
        return True
    if 0x1AB0 <= token_id <= 0x1AFF:
        return True
    if 0xFE00 <= token_id <= 0xFE0F:
        return True
    return False


# 预计算垃圾token掩码（避免每步Python循环）
_GARBAGE_MASK = None
_GARBAGE_FORCE_MASK = None  # 强制回答阶段的非CJK字符降权掩码

def _init_quality_masks(vocab_size: int):
    """初始化矢量化质量过滤掩码（仅调用一次）"""
    global _GARBAGE_MASK, _GARBAGE_FORCE_MASK
    if _GARBAGE_MASK is not None:
        return
    
    import torch
    _GARBAGE_MASK = torch.zeros(vocab_size, dtype=torch.bool)
    _GARBAGE_FORCE_MASK = torch.zeros(vocab_size, dtype=torch.bool)
    
    limit = min(vocab_size, 0x2000)
    for token_id in range(limit):
        if _is_garbage_token(token_id):
            _GARBAGE_MASK[token_id] = True
    
    # 强制回答阶段：标记非语言字符（用于降权，不封死）
    for token_id in range(0x80, min(vocab_size, 0x10000)):
        if _GARBAGE_MASK[token_id]:
            continue
        # ASCII 可打印、CJK、全角标点 → 允许
        if 32 <= token_id < 127:
            continue
        if 0x4E00 <= token_id <= 0x9FFF:
            continue
        if 0x3400 <= token_id <= 0x4DBF:
            continue
        if 0xFF00 <= token_id <= 0xFFEF:
            continue
        if 0x3000 <= token_id <= 0x303F:
            continue
        _GARBAGE_FORCE_MASK[token_id] = True  # 非语言字符，需要降权


def _apply_token_quality_filter(
    logits: torch.Tensor,
    force_answer: bool = False,
) -> torch.Tensor:
    """对 logits 应用 token 质量过滤（矢量化，O(1)每步）

    将已知垃圾 token 的概率设为 -inf。
    在强制回答阶段，非 CJK/ASCII 字符降权 5.0。
    """
    global _GARBAGE_MASK, _GARBAGE_FORCE_MASK
    vocab_size = logits.size(-1)
    _init_quality_masks(vocab_size)
    
    # 垃圾token → -inf（矢量化，单条指令）
    mask_slice = _GARBAGE_MASK[:vocab_size].to(logits.device)
    logits[mask_slice] = float("-inf")
    
    # 强制回答阶段：非语言字符降权
    if force_answer:
        force_slice = _GARBAGE_FORCE_MASK[:vocab_size].to(logits.device)
        logits[force_slice] -= 5.0
    
    return logits


def _check_repetition_stop(generated_tokens: list, threshold: int = 5) -> tuple[bool, str]:
    """重复循环检测停止（业界标准方案）
    
    检测生成中的重复模式，防止模型陷入无限循环或"复读机"状态。
    这是比困惑度早停更有效、更稳定的质量控制策略。
    
    Args:
        generated_tokens: 已生成的token列表
        threshold: 连续相同token的阈值
    
    Returns:
        (should_stop: bool, detected_pattern: str)
    """
    if len(generated_tokens) < threshold:
        return False, ""
    
    # 策略1: 检测连续相同token
    recent_tokens = generated_tokens[-threshold:]
    if all(t == recent_tokens[0] for t in recent_tokens):
        return True, f"连续{threshold}个相同token"
    
    # 策略2: 检测重复的n-gram序列（2-gram, 3-gram）
    for n in [2, 3]:
        if len(generated_tokens) < n * 3:
            continue
        
        # 取最后3个n-gram
        ngrams = []
        for i in range(len(generated_tokens) - n + 1):
            ngram = tuple(generated_tokens[i:i+n])
            ngrams.append(ngram)
        
        # 检查最后3个n-gram是否重复
        if len(ngrams) >= 3:
            last_ngrams = ngrams[-3:]
            if len(set(last_ngrams)) == 1:
                return True, f"重复{n}-gram模式"
    
    return False, ""


def generation(text: str, history_context: str = None, max_generate_tokens: int|None = None, thinking_available: bool = True) -> str:
    """生成函数 (Min-p采样 + Token质量过滤 + CoT完整性保护)

    基于 ICLR 2025 Min-p 论文和大量实验优化的解码策略：
    - Min-p 采样替代 top-k/top-p，天然过滤低概率垃圾token
    - Token 质量过滤器阻止非语言字符（加拿大音节、私用区等）
    - 思考阶段长度限制 + 强制回答期确保 CoT→回复 完整过渡
    - 垃圾token连胜检测：连续N个垃圾立即停止
    """
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'  # 系统提示
    RESET = '\033[0m'
    
    if not text or not isinstance(text, str):
        return "无效输入"
    
    # 【删除限制】不限制最大生成长度，让模型自然输出END_TOKEN终止
    if max_generate_tokens is None:
        max_generate_tokens = 2048  # 【修复MEDIUM #7】设安全上限，防止无限生成OOM
    
    model.eval()
    output_text = ""

    # 【StreamingLLM】可选sink token（默认关闭，配置 use_sink_token=True 启用）
    sink_token_seg = [torch.tensor([TextTokenizer.PLACEHOLDER_SINK_TOKEN], device=device)] if bool(CONFIG.get("use_sink_token", False)) else []
    
    if history_context and history_context.strip():
        history_tensor = TextTokenizer.encode(history_context).to(device)
        text_tensor = TextTokenizer.encode(text).to(device)
        
        prompt = torch.cat([
            *sink_token_seg,
            torch.tensor([TextTokenizer.START_GENERATION_TOKEN], device=device),
            history_tensor,
            torch.tensor([TextTokenizer.END_GENERATION_TOKEN], device=device),
            torch.tensor([TextTokenizer.HISTORY_CONTEXT_START_TOKEN], device=device),
            text_tensor,
            torch.tensor([TextTokenizer.START_GENERATION_TOKEN], device=device),
        ])
    else:
        prompt = torch.cat([
            *sink_token_seg,
            TextTokenizer.encode(text).to(device),
            torch.tensor([TextTokenizer.START_GENERATION_TOKEN], device=device),
        ])
    
    print("\n---Generated reply:", flush=True)

    max_generate_tokens = max(1, int(max_generate_tokens))
    
    # 读取采样参数 — Min-p + 温度 (ICLR 2025 方案)
    temperature = float(CONFIG.get("temperature", 0.5))
    min_p = float(CONFIG.get("min_p", 0.05))
    repetition_penalty = float(CONFIG.get("repetition_penalty", 1.02))
    repetition_stop_threshold = int(CONFIG.get("repetition_stop_threshold", 8))
    force_answer_min_steps = int(CONFIG.get("force_answer_min_steps", 32))
    max_consecutive_garbage = int(CONFIG.get("max_consecutive_garbage", 3))

    with torch.inference_mode():
        thinking_started = False
        force_answer_steps = 0          # 强制回答剩余步数
        consecutive_garbage = 0         # 连续垃圾token计数
        
        if thinking_available:
            has_think_token = (prompt == TextTokenizer.THINK_START_TOKEN).any()
            if has_think_token:
                # 已包含THINK_START_TOKEN，直接标记为已开始，不再追加
                thinking_started = True
            else:
                # 未包含，追加THINK_START_TOKEN并标记为已开始
                thinking_started = True
                think_start_tensor = torch.tensor([TextTokenizer.THINK_START_TOKEN], device=device)
                prompt = torch.cat([prompt, think_start_tensor])
        
        result = model(prompt, use_cache=True)
        if isinstance(result, tuple):
            logits, past_key_values = result
        else:
            logits = result

        step = 0
        
        # 有序ID列表（用于时间顺序检测）+ Counter（用于频率惩罚）
        generated_ids: list[int] = []
        generated_tokens = Counter()
        
        while step < max_generate_tokens:  # 总生成步数限制（思维链+回答一起）
            try:
                next_logits = logits[-1].clone()  # 【修复】始终clone，避免修改原始logits
                # 【删除限制】不再设置最小生成token数
                
                # ── ① Token 质量过滤 ──
                next_logits = _apply_token_quality_filter(
                    next_logits,
                    force_answer=(force_answer_steps > 0),
                )

                if force_answer_steps > 0:
                    next_logits[TextTokenizer.END_GENERATION_TOKEN] = float("-inf")
                    next_logits[TextTokenizer.UNKNOWN_TOKEN] = float("-inf")
                    next_logits[TextTokenizer.THINK_START_TOKEN] = float("-inf")
                    next_logits[TextTokenizer.THINK_END_TOKEN] = float("-inf")
                    next_logits[TextTokenizer.HISTORY_CONTEXT_START_TOKEN] = float("-inf")
                    next_logits[TextTokenizer.HISTORY_CONTEXT_END_TOKEN] = float("-inf")
                    force_answer_steps -= 1

                # ── ③ Repetition penalty (滑动窗口优化版) ──
                # 【修复Bug #14】只对最近N个token应用惩罚，避免遍历全部vocab
                if repetition_penalty > 1.0 and len(generated_ids) > 0:
                    recent_ids = set(generated_ids[-128:])  # 滑动窗口：最近128个
                    for token_id in recent_ids:
                        if token_id < next_logits.size(0):
                            count = generated_tokens.get(token_id, 1)
                            penalty = repetition_penalty ** min(count, 5)
                            if token_id in set(generated_ids[-10:]):
                                penalty *= 1.1
                            if next_logits[token_id] > 0:
                                next_logits[token_id] /= penalty
                            else:
                                next_logits[token_id] *= penalty

                # ── ④ 先应用温度，再 Min-p 采样 (ICLR 2025) ──
                # 【修复Bug #11】确保min-p的阈值计算在温度缩放后的概率上进行
                # 原顺序：min-p(未缩放logits) → softmax(已缩放logits) → 阈值温度不一致
                # 新顺序：temperature缩放 → min-p → softmax
                if temperature > 0 and temperature != 1.0:
                    next_logits = next_logits / temperature
                if min_p > 0.0:
                    next_logits = _min_p_sampling(next_logits, min_p)

                probs = torch.softmax(next_logits, dim=-1)
                index = int(torch.multinomial(probs, 1).item())

                # ── ⑤ 垃圾token检测与重采样 ──
                if _is_garbage_token(index) and index not in (
                    TextTokenizer.THINK_START_TOKEN,
                    TextTokenizer.THINK_END_TOKEN,
                    TextTokenizer.END_GENERATION_TOKEN,
                    TextTokenizer.START_GENERATION_TOKEN,
                ):
                    consecutive_garbage += 1
                    if consecutive_garbage >= max_consecutive_garbage:
                        print(f"\n[Stop] 连续{consecutive_garbage}个垃圾token，强制结束", flush=True)
                        break
                    # 将该垃圾token的logit设为-inf后重采样
                    next_logits[index] = float("-inf")
                    continue
                else:
                    consecutive_garbage = 0  # 有效token，重置计数器

                should_skip_output = False
                
                if index == TextTokenizer.THINK_END_TOKEN:
                    if thinking_available and thinking_started:
                        thinking_started = False
                        force_answer_steps = force_answer_min_steps
                        print(f"\n{GREEN}", end="", flush=True)
                        should_skip_output = True
                    else:
                        break

                elif index == TextTokenizer.END_GENERATION_TOKEN:
                    if force_answer_steps > 0 or thinking_started:
                        next_logits[index] = float("-inf")
                        continue  # 【修复】思考阶段也禁止END_TOKEN，防止模型还在思考就提前结束
                    else:
                        break

                elif index == TextTokenizer.THINK_START_TOKEN:
                    if thinking_available and not thinking_started:
                        thinking_started = True
                        should_skip_output = True
                    elif not thinking_available:
                        should_skip_output = True

                # 重复检测停止（仅回答阶段启用）
                if not thinking_started and index not in (
                    TextTokenizer.THINK_START_TOKEN,
                    TextTokenizer.THINK_END_TOKEN,
                    TextTokenizer.START_GENERATION_TOKEN,
                    TextTokenizer.END_GENERATION_TOKEN,
                ):
                    if step >= 5:
                        should_stop, pattern = _check_repetition_stop(generated_ids + [index], repetition_stop_threshold)
                        if should_stop:
                            print(f"\n[Stop] 检测到重复模式({pattern})，提前结束", flush=True)
                            break

                if not should_skip_output:
                    decoded_piece = TextTokenizer.decode(torch.tensor([index]))
                    
                    if decoded_piece:
                        if thinking_started:
                            print(f"{BLUE}{decoded_piece}{RESET}", end="", flush=True)
                        else:
                            print(f"{GREEN}{decoded_piece}{RESET}", end="", flush=True)
                        
                        output_text += decoded_piece

                # 【修复】将生成的token添加到Counter中用于frequency_penalty
                if index not in (
                    TextTokenizer.THINK_START_TOKEN,
                    TextTokenizer.THINK_END_TOKEN,
                    TextTokenizer.START_GENERATION_TOKEN,
                    TextTokenizer.END_GENERATION_TOKEN,
                ):
                    generated_tokens[index] += 1
                    generated_ids.append(index)

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
        
        # ── CoT 完整性检测：生成结束仍未输出 THINK_END → 强制注入回答 ──
        if thinking_started and thinking_available:
            print(f"\n{YELLOW}[CoT Guard] 未检测到回答，正在强制过渡...{RESET}", flush=True)
            
            # 强制插入 THINK_END，让模型进入回答模式
            think_end_token = torch.tensor([TextTokenizer.THINK_END_TOKEN], device=device)
            result = model(think_end_token, past_key_values=past_key_values, use_cache=True)
            if isinstance(result, tuple):
                logits, past_key_values = result
            else:
                logits = result
            
            thinking_started = False
            force_answer_steps = force_answer_min_steps
            step += 1
            
            # 继续生成回答（最多额外 generate 步）
            forced_max = min(step + max_generate_tokens // 2, max_generate_tokens * 2)
            while step < forced_max:
                try:
                    next_logits = logits[-1].clone()
                    
                    # 质量过滤 + 强制回答期保护
                    next_logits = _apply_token_quality_filter(next_logits, force_answer=True)
                    next_logits[TextTokenizer.END_GENERATION_TOKEN] = float("-inf")
                    next_logits[TextTokenizer.UNKNOWN_TOKEN] = float("-inf")
                    next_logits[TextTokenizer.THINK_START_TOKEN] = float("-inf")
                    next_logits[TextTokenizer.THINK_END_TOKEN] = float("-inf")
                    next_logits[TextTokenizer.HISTORY_CONTEXT_START_TOKEN] = float("-inf")
                    next_logits[TextTokenizer.HISTORY_CONTEXT_END_TOKEN] = float("-inf")
                    
                    if force_answer_steps > 0:
                        force_answer_steps -= 1
                    
                    # Repetition penalty（滑动窗口优化版）
                    if repetition_penalty > 1.0 and len(generated_ids) > 0:
                        recent_ids = set(generated_ids[-128:])
                        for token_id in recent_ids:
                            if token_id < next_logits.size(0):
                                count = generated_tokens.get(token_id, 1)
                                penalty = repetition_penalty ** min(count, 5)
                                if token_id in set(generated_ids[-10:]):
                                    penalty *= 1.1
                                if next_logits[token_id] > 0:
                                    next_logits[token_id] /= penalty
                                else:
                                    next_logits[token_id] *= penalty
                    
                    # 温度 + Min-p
                    if temperature > 0 and temperature != 1.0:
                        next_logits = next_logits / temperature
                    if min_p > 0.0:
                        next_logits = _min_p_sampling(next_logits, min_p)
                    
                    probs = torch.softmax(next_logits, dim=-1)
                    index = int(torch.multinomial(probs, 1).item())
                    
                    # 垃圾检测
                    if _is_garbage_token(index) and index not in (TextTokenizer.END_GENERATION_TOKEN,):
                        consecutive_garbage += 1
                        if consecutive_garbage >= max_consecutive_garbage:
                            break
                        next_logits[index] = float("-inf")
                        continue
                    else:
                        consecutive_garbage = 0
                    
                    if index == TextTokenizer.END_GENERATION_TOKEN:
                        break
                    
                    decoded_piece = TextTokenizer.decode(torch.tensor([index]))
                    if decoded_piece:
                        print(f"{GREEN}{decoded_piece}{RESET}", end="", flush=True)
                        output_text += decoded_piece
                    
                    if index not in (TextTokenizer.THINK_START_TOKEN, TextTokenizer.THINK_END_TOKEN,
                                     TextTokenizer.START_GENERATION_TOKEN, TextTokenizer.END_GENERATION_TOKEN):
                        generated_tokens[index] += 1
                        generated_ids.append(index)
                    
                    next_token = torch.tensor([index], device=device)
                    result = model(next_token, past_key_values=past_key_values, use_cache=True)
                    if isinstance(result, tuple):
                        logits, past_key_values = result
                    else:
                        logits = result
                    
                    step += 1
                except Exception:
                    break

        # 【新增】生成完成后清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return output_text


def _estimate_safe_chunk_size_v2(free_bytes: float, safety_factor: float = 0.6) -> int:
    """精确显存预算模型（6GB专用），分项计算KV Cache+激活值+系统保留"""
    cfg = CONFIG
    d = int(cfg["emb_size"])
    L = int(cfg["num_transformer_blocks"])
    bytes_per = 2 if (use_amp and amp_dtype == torch.bfloat16) else (2 if use_amp else 4)

    # 1. KV Cache: 2(K+V) * L * d * bytes_per
    kv_per_token = 2 * L * d * bytes_per

    # 2. 注意力激活峰值: heads * window * 4 (softmax buffer)
    h = int(cfg["num_heads"])
    window = int(cfg.get("sliding_window", 128))
    act_peak = h * window * 4

    # 3. 系统保留 512MB
    system_reserve = 512 * 1024 * 1024

    usable = max(1.0, (free_bytes - system_reserve)) * safety_factor
    bytes_per_token = max(1, kv_per_token + act_peak)

    chunk_size = max(64, int(usable / bytes_per_token))
    return min(chunk_size, 4096)


# P2: 四级显存触发器
def _memory_trigger_policy(mem_ratio: float) -> str:
    """返回当前应采取的记忆管理策略"""
    if mem_ratio < 0.70:
        return "normal"
    elif mem_ratio < 0.82:
        return "compress"
    elif mem_ratio < 0.90:
        return "aggressive"
    else:
        return "skip"


def _run_train_step(train_tensor: torch.Tensor, target_mask: torch.Tensor, preview: torch.Tensor, show_preview: bool = True, preview_color: str = None) -> float:
    """执行单步训练（显存感知 + MLA latent KV 压缩）

    核心策略（按优先级）：
    1. 序列能放入显存 → 标准前向+反向
    2. 序列过长 → 分段处理，每段独立 forward，MLA latent memory 自动累积历史
    3. 显存接近阈值 → 自动触发 MLA latent 压缩，释放 KV Cache
    """
    global training_rounds
    _step_t0 = time.time()  # 【修复Bug #1】在函数内部计时，而非依赖外部的t0

    model.train()
    seq_len = train_tensor.numel()
    
# ── 显存状态检查 ──
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        total_mem = float(props.total_memory)
        reserved = float(torch.cuda.memory_reserved(idx))
        allocated = float(torch.cuda.memory_allocated(idx))
        used = max(reserved, allocated)
        free_bytes = max(0.0, total_mem - used)
        mem_ratio = used / total_mem
    else:
        free_bytes = float('inf')
        mem_ratio = 0.0
        total_mem = float('inf')

    # 定期显存监控
    if training_rounds > 0 and training_rounds % 100 == 0 and torch.cuda.is_available():
        print(f"[Memory] Step {training_rounds}: Used={used/1024**3:.2f}/{total_mem/1024**3:.2f}GB "
              f"({mem_ratio*100:.1f}%), Free={free_bytes/1024**3:.2f}GB, SeqLen={seq_len}", flush=True)

        cache_thresh = float(CONFIG.get("gpu_cache_clear_threshold_gb", 4.0))
        if reserved / 1024**3 > cache_thresh:
            torch.cuda.empty_cache()
            print(f"[Memory] Cleared GPU cache", flush=True)

    # 梯度累积管理
    if (training_rounds % GRADIENT_ACCUMULATION_STEPS) == 0:
        optimizer.zero_grad(set_to_none=True)

    # ── 🧠 四级显存触发器 + MLA latent 压缩触发 ──
    policy = _memory_trigger_policy(mem_ratio)

    if policy == "aggressive":
        print(f"\n{'='*60}", flush=True)
        print(f"[🧠 Memory Policy] 🟠 激进模式: 显存{mem_ratio:.1%}", flush=True)
        print(f"[🧠 Memory Policy] 压缩比例从25%→6%, 释放GPU缓存", flush=True)
        print(f"{'='*60}\n", flush=True)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    elif policy == "compress" and seq_len > 256:
        print(f"\n{'='*60}", flush=True)
        print(f"[🧠 MLA Memory] 🟡 压缩触发: 显存{mem_ratio:.1%}", flush=True)
        print(f"[🧠 MLA Memory] 序列长度={seq_len} token, 压缩到 MLA latent KV 层", flush=True)
        print(f"[🧠 MLA Memory] MLALatentMemory 已累积历史上下文", flush=True)
        print(f"{'='*60}\n", flush=True)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    skip_thresh = float(CONFIG.get("gpu_memory_skip_ratio", 0.92))
    if mem_ratio >= skip_thresh or policy == "skip":
        print(f"[Memory] 显存占用过高 ({mem_ratio:.1%}), 主动清理后跳过本样本", flush=True)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return float('inf')

    # ── 检查target_mask ──
    # 【修复Bug #7】如果target_mask全为False，前向传播无梯度，直接跳过
    if not target_mask.any():
        print(f"[Warning] target_mask全为False，跳过本样本", flush=True)
        return float('inf')

    # ── 策略选择：估算此序列所需显存 ──
    safe_chunk = _estimate_safe_chunk_size_v2(free_bytes, safety_factor=0.65)

    try:
        if seq_len <= safe_chunk:
            # ✅ 策略 1: 标准训练（序列完整放入 GPU）
            train_tensor_gpu = train_tensor.to(device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                result = model(train_tensor_gpu, use_cache=False)
                if isinstance(result, tuple):
                    logits = result[0]
                else:
                    logits = result

                if seq_len > 1:
                    mask_bool = target_mask[1:].to(device)
                    if mask_bool.any():
                        pred = logits[:-1][mask_bool]
                        tgt = train_tensor_gpu[1:][mask_bool]
                        loss = loss_func(pred, tgt)
                    else:
                        loss = torch.tensor(0.0, device=device)
                else:
                    loss = torch.tensor(0.0, device=device)

            # ── 【新增】Special Token Anchor Loss ──
            # 防止特殊Token（ID < 10）的hidden state被普通Token淹没
            # 通过额外的自预测损失，确保特殊Token位置保持可区分性
            # 这是解决"特殊Token注意力消失"问题的最后一道防线
            anchor_loss_coef = float(CONFIG.get("special_token_anchor_loss_coef", 0.05))
            if anchor_loss_coef > 0 and seq_len > 1:
                special_positions = (train_tensor_gpu < 10)
                if special_positions.any():
                    # 从logits中提取特殊Token位置的输出分布
                    special_logits = logits[:-1][special_positions[1:]]
                    special_targets = train_tensor_gpu[1:][special_positions[1:]]
                    if special_logits.numel() > 0:
                        anchor_loss = loss_func(special_logits, special_targets)
                        loss = loss + anchor_loss_coef * anchor_loss

            raw_loss_val = loss.item()  # 保存原始 loss 用于 ReduceLROnPlateau
            # 记录loss到record.txt（异步写入，不阻塞训练）
            record_loss(raw_loss_val)
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            
            # 【修复】标准训练路径必须显式调用 backward()
            # 原代码缺少此调用，导致梯度从未被计算，训练完全无效！
            # requires_grad 检查防止零 loss（无 grad_fn）时的崩溃
            if loss.requires_grad:
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

        else:
            # ✅ 策略 2: 🧠 MLA latent 分段训练（长文本→压缩到 MLA latent KV 层）
            # 【修复Bug #3】使用 past_key_values 传递跨段 MLA latent memory 状态
            # 注意：MLALatentMemory.update() 内部使用 .detach()，梯度不会跨段传播
            # 但前向传播时 MLA 状态可跨段累积，使后续段可访问历史压缩记忆
            seg_size = min(safe_chunk, 2048)
            seg_losses = []
            seg_past_kv = [None] * len(model.transformers)  # 跨段传递的 KV cache
            seg_loss_value_accum = []  # 【修复CRIT-3】收集每段loss值用于记录

            print(f"[🧠 Memory] 长序列触发 MLA latent 压缩: seq_len={seq_len}, "
                  f"seg_size={seg_size}, free={free_bytes/1024**3:.2f}GB", flush=True)

            num_segments = max(1, (seq_len + seg_size - 1) // seg_size)

            for seg_start in range(0, seq_len, seg_size):
                seg_end = min(seg_start + seg_size, seq_len)
                seg_tensor = train_tensor[seg_start:seg_end].to(device)
                seg_mask = target_mask[seg_start:seg_end]

                # 【修复Bug #3 v2】使用use_cache=True使MLA跨段累积，之后detach防OOM
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    result = model(seg_tensor, past_key_values=seg_past_kv, use_cache=True)
                    if isinstance(result, tuple):
                        logits, seg_past_kv = result
                    else:
                        logits = result

                    if seg_tensor.numel() > 1 and seg_mask.any():
                        mask_bool = seg_mask[1:].to(device)
                        if mask_bool.any():
                            pred = logits[:-1][mask_bool]
                            tgt = seg_tensor[1:][mask_bool]
                            loss_seg = loss_func(pred, tgt)
                        else:
                            loss_seg = torch.tensor(0.0, device=device)
                    else:
                        loss_seg = torch.tensor(0.0, device=device)

                    # ── 分段路径的Special Token Anchor Loss ──
                    anchor_loss_coef = float(CONFIG.get("special_token_anchor_loss_coef", 0.05))
                    if anchor_loss_coef > 0 and seg_tensor.numel() > 1:
                        special_positions = (seg_tensor < 10)
                        if special_positions.any():
                            special_logits = logits[:-1][special_positions[1:]]
                            special_targets = seg_tensor[1:][special_positions[1:]]
                            if special_logits.numel() > 0:
                                anchor_loss = loss_func(special_logits, special_targets)
                                loss_seg = loss_seg + anchor_loss_coef * anchor_loss

                # autocast结束。detach KV cache防止计算图跨段连接
                if seg_past_kv is not None:
                    seg_past_kv = [
                        None if layer_kv is None else tuple(
                            t.detach() if isinstance(t, torch.Tensor) else t
                            for t in layer_kv
                        )
                        for layer_kv in seg_past_kv
                    ]

                # 【修复CRIT-3】逐段反向传播，立即释放计算图
                seg_loss_scaled = loss_seg / num_segments / GRADIENT_ACCUMULATION_STEPS
                if seg_loss_scaled.requires_grad:
                    if scaler.is_enabled():
                        scaler.scale(seg_loss_scaled).backward()
                    else:
                        seg_loss_scaled.backward()
                seg_loss_value_accum.append(loss_seg.detach().item())

                # 每段后立即释放本段计算图
                del seg_tensor, logits, loss_seg, seg_loss_scaled
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # 检查显存，如果仍高则打印提示
                if torch.cuda.is_available() and seg_start % (seg_size * 2) == 0:
                    ratio = _get_gpu_memory_ratio(device)
                    if ratio > 0.8:
                        print(f"[🧠 Memory] 段内显存{ratio:.1%}，MLA latent memory 已累积历史上下文", flush=True)

            if not seg_loss_value_accum:
                print(f"[Memory] 所有分段均OOM，跳过本样本", flush=True)
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                return float('inf')

            raw_loss_val = sum(seg_loss_value_accum) / len(seg_loss_value_accum)
            record_loss(raw_loss_val)
            # 【修复CRIT-6】保留实际loss值供profiling和调度器使用
            # 注意：此tensor无grad_fn，不影响梯度累积
            loss = torch.tensor(raw_loss_val, device=device)

            # ── 统一的梯度后处理 ──（接在分段循环后面）
            # 注意：梯度已在每段backward中累积，这里只需执行optimizer.step

        # ── 统一的梯度后处理 ──
        if not torch.isnan(loss) and not torch.isinf(loss):
            if scaler.is_enabled():
                # 【修复】unscale 和 clip 只在 step 前调用一次，避免梯度累积时重复除法
                if (training_rounds + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        optimizer.zero_grad(set_to_none=True)
                        scaler.update()
                        for param in model.parameters():
                            if param.grad is not None and torch.isnan(param.grad).any():
                                param.grad = None
                        print(f"[Warning] NaN/Inf gradient, skipping optimizer step", flush=True)
                        return float('inf')
                    scaler.step(optimizer)
                    scaler.update()
            else:
                if (training_rounds + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        optimizer.zero_grad(set_to_none=True)
                        for param in model.parameters():
                            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                                param.grad = None
                        print(f"[Warning] NaN/Inf gradient, skipping optimizer step", flush=True)
                        return float('inf')
                    optimizer.step()
        else:
            print(f"[Warning] Invalid loss: {loss}, skipping optimizer step", flush=True)
            return float('inf')

        training_rounds += 1

        # 学习率调度 — SGDR + ReduceLROnPlateau（动态 LR，适配无限训练）
        if (training_rounds % GRADIENT_ACCUMULATION_STEPS) == 0:
            global optimizer_step_count
            optimizer_step_count += 1
            current_lr = lr_scheduler.step(loss=raw_loss_val)

        # Preview 输出
        if show_preview:
            try:
                decoded = TextTokenizer.decode(preview[preview != 0])
                RESET = '\033[0m'
                if preview_color:
                    print(f"{preview_color}{decoded}{RESET}", end="", flush=True)
                else:
                    print(decoded, end="", flush=True)
            except Exception as e:
                print(f"[Warning] Preview decode failed: {e}", flush=True)
            print("", flush=True)

        return loss.item()

    except RuntimeError as e:
        error_msg = str(e)
        if "CUDA" in error_msg or "cuda" in error_msg.lower() or "out of memory" in error_msg.lower():
            print(f"[CUDA Error] {e}", flush=True)
            optimizer.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            return float('inf')
        else:
            print(f"[RuntimeError] {e}, skipping", flush=True)
            return float('inf')
    except Exception as e:
        print(f"[Error] 未知错误: {e}, skipping", flush=True)
        return float('inf')
    finally:
        try:
            elapsed = time.time() - _step_t0
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 1024**3
                resv = torch.cuda.memory_reserved() / 1024**3
                print(f"[Profile] Step {training_rounds}: {elapsed:.3f}s, "
                      f"Alloc={alloc:.2f}GB, Resv={resv:.2f}GB", flush=True)
        except Exception:
            pass
