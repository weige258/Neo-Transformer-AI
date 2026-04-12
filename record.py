import threading
import time
import os

# Global variables
running_time = 0
total_loss = 0
record_count = 0
record_interval = 1000


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
        print(f"加载运行时间失败: {e}")
        running_time = 0


# Initialize running time
load_run_time()


def time_thread():
    """Thread to track running time"""
    global running_time
    while True:
        running_time += 1
        time.sleep(1)


# Start the time tracking thread
threading.Thread(target=time_thread, daemon=True).start()


def record_loss(loss: float):
    """Record training loss"""
    global running_time, total_loss, record_count
    
    try:
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
            print(f"记录损失 - 系统时间: {system_time}, 运行时间: {time_str}, 平均损失: {avg_loss:.6f}")
            
    except Exception as e:
        print(f"记录损失失败: {e}")

def get_loss() -> float:
    """返回当前平均loss，如果分母是0就返回10"""
    global total_loss, record_count
    if record_count > 0:
        return total_loss / record_count
    else:
        return 10.0