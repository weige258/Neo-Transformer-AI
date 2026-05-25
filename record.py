import threading
import time
import os

# Global variables
running_time = 0
total_loss = 0
record_count = 0
record_interval = 1000

# 【修复】新增全局线程锁，防止数据竞争
data_lock = threading.Lock()


def hours_minutes_seconds_to_seconds(time_str: str) -> int:
    """Convert HH:MM:SS string to seconds"""
    try:
        h, m, s = time_str.split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)
    except (ValueError, IndexError):
        return 0


def seconds_to_hours_minutes_seconds(seconds: int) -> str:
    """Convert seconds to HH:MM:SS string"""
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02}:{m:02}:{s:02}"


def get_system_time():
    """Get current system time in format YYYY/MM/DD HH:MM:SS"""
    return time.strftime("%Y/%m/%d %H:%M:%S")


def load_run_time():
    """Load running time from record file"""
    global running_time
    record_file = "record.txt"
    
    if not os.path.exists(record_file):
        # Create the record file if it doesn't exist
        with open(record_file, "w", encoding="utf-8") as f:
            system_time = get_system_time()
            f.write(f"<system_time>{system_time}</system_time><time>00:00:00</time><avg_loss>0</avg_loss>\n")
        running_time = 0
        return
    
    try:
        with open(record_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if lines:
                last_line = lines[-1].strip()
                # Parse running time
                time_start = last_line.find("<time>") + len("<time>")
                time_end = last_line.find("</time>")
                if time_start > 0 and time_end > time_start:
                    time_str = last_line[time_start:time_end]
                    running_time = hours_minutes_seconds_to_seconds(time_str)
    except Exception as e:
        print(f"加载运行时间失败: {e}", flush=True)
        running_time = 0


# Initialize running time
load_run_time()


def time_thread():
    """Thread to track running time"""
    global running_time
    while True:
        with data_lock:
            running_time += 1
        time.sleep(1)


# Start the time tracking thread
threading.Thread(target=time_thread, daemon=True).start()


def record_loss(loss: float):
    """Record training loss"""
    global running_time, total_loss, record_count
    
    try:
        with data_lock:
            total_loss += loss
            record_count += 1
            
            if record_count >= record_interval:
                avg_loss = total_loss / record_count
                record_file = "record.txt"
                
                with open(record_file, "a", encoding="utf-8") as f:
                    time_str = seconds_to_hours_minutes_seconds(running_time)
                    system_time = get_system_time()
                    f.write(f"<system_time>{system_time}</system_time><time>{time_str}</time><avg_loss>{avg_loss:.6f}</avg_loss>\n")
                
                # Reset counters
                total_loss = 0
                record_count = 0
                print(f"记录损失 - 系统时间: {system_time}, 运行时间: {time_str}, 平均损失: {avg_loss:.6f}", flush=True)
            
    except Exception as e:
        print(f"记录损失失败: {e}", flush=True)

def get_loss() -> float:
    """返回当前平均loss，如果分母是0就返回10"""
    global total_loss, record_count
    if record_count > 0:
        return total_loss / record_count
    else:
        return 10.0


def evaluate_rl_readiness(
    loss_threshold: float = 1.2,
    stability_window: int = 5,
    stability_std_threshold: float = 0.15,
    min_training_rounds: int = 3000,
) -> tuple[bool, str]:
    """自动评估当前模型是否已准备好进行强化学习（PPO）。

    评估维度：
      1. 训练步数是否达到最低要求
      2. 近期平均 loss 是否低于阈值
      3. 近期 loss 是否已稳定（标准差低 = 收敛）

    Args:
        loss_threshold: 平均 loss 必须低于此值
        stability_window: 取最近 N 条 record.txt 记录进行分析
        stability_std_threshold: loss 标准差低于此值视为收敛
        min_training_rounds: 最低训练轮数（约 = 记录数 × record_interval）

    Returns:
        (ready: bool, reason: str) — 是否就绪及原因说明
    """
    import os
    from config import CONFIG

    record_file = "record.txt"
    if not os.path.exists(record_file):
        return False, "record.txt 不存在，尚无训练数据"

    try:
        with open(record_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
    except Exception as e:
        return False, f"读取 record.txt 失败: {e}"

    # 解析 loss 记录
    losses = []
    for line in lines:
        try:
            loss_start = line.find("<avg_loss>") + len("<avg_loss>")
            loss_end = line.find("</avg_loss>")
            if loss_start > 0 and loss_end > loss_start:
                losses.append(float(line[loss_start:loss_end]))
        except (ValueError, IndexError):
            continue

    record_interval = int(CONFIG.get("record_interval", 1000))
    estimated_rounds = len(losses) * record_interval

    # 条件 1：最低训练量
    if estimated_rounds < min_training_rounds:
        return False, (
            f"训练量不足：估计 {estimated_rounds} 轮 < 最低 {min_training_rounds} 轮 "
            f"（{len(losses)} 条 loss 记录 × {record_interval}）"
        )

    # 取最近 stability_window 条记录
    recent = losses[-min(len(losses), stability_window):]
    if len(recent) < max(2, stability_window // 2):
        return False, f"loss 记录不足：仅 {len(recent)} 条，需要至少 {max(2, stability_window // 2)} 条"

    import statistics
    avg_loss = statistics.mean(recent)
    std_loss = statistics.stdev(recent) if len(recent) >= 2 else float("inf")

    # 条件 2：loss 低于阈值
    if avg_loss >= loss_threshold:
        return False, (
            f"平均 loss 过高：{avg_loss:.4f} >= {loss_threshold} "
            f"（最近 {len(recent)} 条记录）"
        )

    # 条件 3：loss 已稳定
    if std_loss >= stability_std_threshold:
        return False, (
            f"loss 尚未收敛：标准差 {std_loss:.4f} >= {stability_std_threshold} "
            f"（最近 {len(recent)} 条均值={avg_loss:.4f}）"
        )

    return True, (
        f"✅ RL 就绪：avg_loss={avg_loss:.4f} < {loss_threshold}, "
        f"std={std_loss:.4f} < {stability_std_threshold}, "
        f"估计训练 {estimated_rounds} 轮"
    )