import json
import os
import torch
import random
import logging
from typing import List, Optional, Dict
from main import train, model, optimizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_dataset_files(dataset_dir: str = "dataset") -> List[str]:
    """Load all dataset JSON files from the specified directory"""
    
    dataset_files = []
    for file_name in os.listdir(dataset_dir):
        if file_name.endswith('.json'):
            dataset_files.append(os.path.join(dataset_dir, file_name))
    
    logging.info(f"Found {len(dataset_files)} dataset files in {dataset_dir}")
    return dataset_files


class StreamingDataset:
    """流式数据集，按需加载数据，避免内存溢出"""
    
    def __init__(self, dataset_dir: str = "dataset"):
        self.dataset_dir = dataset_dir
        self.dataset_files = load_dataset_files(dataset_dir)
        self.total_entries = 0
        self.file_entry_counts = []
        
        self._build_index()
    
    def _build_index(self):
        """构建文件索引，统计每个文件的条目数"""
        for file_path in self.dataset_files:
            count = self._count_entries_in_file(file_path)
            self.file_entry_counts.append(count)
            self.total_entries += count
        
        logging.info(f"Indexed {self.total_entries} total entries across {len(self.dataset_files)} files")
    
    def _count_entries_in_file(self, file_path: str) -> int:
        """快速统计文件中的有效条目数"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            count = 0
            for item in data:
                # 新格式：检查 ask 和 answer 字段
                if "ask" in item and "answer" in item:
                    ask_raw = item.get("ask")
                    answer_raw = item.get("answer")
                    if ask_raw is not None and answer_raw is not None:
                        ask = str(ask_raw).strip()
                        answer = str(answer_raw).strip()
                        if ask and answer:
                            count += 1
            return count
        except Exception as e:
            logging.error(f"Failed to count entries in {file_path}: {e}")
            return 0
    
    def get_random_sample(self) -> Dict[str, Optional[str]]:
        """随机获取一个训练样本（包含ask, think, answer, history）"""
        if self.total_entries == 0:
            raise ValueError("No training data available")
        
        target_idx = random.randint(0, self.total_entries - 1)
        
        cumulative = 0
        for file_idx, count in enumerate(self.file_entry_counts):
            if cumulative + count > target_idx:
                local_idx = target_idx - cumulative
                return self._load_entry_from_file(self.dataset_files[file_idx], local_idx)
            cumulative += count
        
        raise IndexError("Failed to locate entry")
    
    def _load_entry_from_file(self, file_path: str, target_local_idx: int) -> Dict[str, Optional[str]]:
        """从文件中加载指定索引的条目
        
        修复：添加文件大小限制和安全校验，防止恶意JSON导致内存耗尽
        """
        try:
            # 【安全修复】检查文件大小，限制为100MB
            max_file_size = 100 * 1024 * 1024  # 100MB
            file_size = os.path.getsize(file_path)
            if file_size > max_file_size:
                raise ValueError(
                    f"文件 {file_path} 过大 ({file_size / 1024 / 1024:.2f}MB > {max_file_size / 1024 / 1024:.2f}MB)，"
                    f"可能存在安全风险或格式错误"
                )
            
            with open(file_path, "r", encoding="utf-8") as f:
                # 【安全修复】添加JSON加载的异常处理
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    raise ValueError(f"JSON解析失败 {file_path}: {e}") from e
                except MemoryError:
                    raise MemoryError(f"JSON文件 {file_path} 加载时内存不足，文件可能过大或格式异常")
            
            current_idx = 0
            for item in data:
                # 新格式：检查 ask 和 answer 字段
                if "ask" in item and "answer" in item:
                    ask_raw = item.get("ask")
                    answer_raw = item.get("answer")
                    
                    if ask_raw is not None and answer_raw is not None:
                        ask = str(ask_raw).strip()
                        answer = str(answer_raw).strip()
                        
                        if ask and answer:
                            if current_idx == target_local_idx:
                                # 提取所有字段
                                think_raw = item.get("think", "")
                                think = str(think_raw).strip() if think_raw is not None else ""
                                
                                history_raw = item.get("history", [])
                                # 将history数组转换为字符串格式
                                if isinstance(history_raw, list) and len(history_raw) > 0:
                                    # 假设history是 [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}] 格式
                                    history_parts = []
                                    for msg in history_raw:
                                        if isinstance(msg, dict):
                                            role = msg.get("role", "unknown")
                                            content = msg.get("content", "")
                                            if role == "user":
                                                history_parts.append(f"用户: {content}")
                                            elif role == "assistant":
                                                history_parts.append(f"助手: {content}")
                                            else:
                                                history_parts.append(f"{role}: {content}")
                                        elif isinstance(msg, str):
                                            history_parts.append(str(msg))
                                    history_context = "\n".join(history_parts)
                                else:
                                    history_context = ""
                                
                                return {
                                    "ask": ask,
                                    "think": think,
                                    "answer": answer,
                                    "history_context": history_context
                                }
                            current_idx += 1
            
            raise IndexError(f"Entry {target_local_idx} not found in {file_path}")
        
        except Exception as e:
            logging.error(f"Failed to load entry from {file_path}: {e}")
            raise


def main() -> None:
    """Main training loop
    
    修复：精细化异常处理，区分可恢复错误和不可恢复错误
    添加最大连续错误计数，防止在损坏状态下无限循环
    """
    # 使用流式数据集，不再一次性加载所有数据到内存
    dataset = StreamingDataset("dataset")

    if dataset.total_entries == 0:
        logging.error("No training data found, please check dataset files")
        return

    logging.info(f"Initialized streaming dataset with {dataset.total_entries} training samples.")

    local_training_rounds = 0
    save_interval = 500  # Save model every 500 rounds
    
    # 用于ReduceLROnPlateau的损失跟踪
    recent_losses = []
    loss_window_size = 100  # 计算平均loss的窗口大小
    
    # 【修复】添加连续错误计数器，防止无限循环
    consecutive_errors = 0
    max_consecutive_errors = 50  # 最大连续错误次数

    try:
        while True:
            # 流式获取随机训练样本
            sample = dataset.get_random_sample()
            
            ask = sample.get("ask", "")
            think = sample.get("think", "")
            answer = sample.get("answer", "")
            history_context = sample.get("history_context", "")
            
            # Skip empty asks or answers
            if not ask or not answer:
                continue

            # Train on this sample (支持问-思考-答-历史上下文格式)
            try:
                train(
                    ask=ask,
                    think=think if think else None,
                    answer=answer,
                    history_context=history_context if history_context else None
                )
                
                # 【修复】训练成功后重置错误计数器
                consecutive_errors = 0
                            
                local_training_rounds += 1
                
                # 获取当前学习率
                current_lr = optimizer.param_groups[0]['lr']
                
                # 记录最近loss用于监控
                import record
                if record.record_count > 0:
                    avg_recent_loss = record.total_loss / record.record_count
                    recent_losses.append(avg_recent_loss)
                    if len(recent_losses) > loss_window_size:
                        recent_losses.pop(0)
                
                print("*" * 100, flush=True)

                # Save model periodically
                if local_training_rounds % save_interval == 0:
                    torch.save(obj=model.state_dict(), f="model.pth")
                    avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0
                    logging.info(f"Model saved, training rounds: {local_training_rounds}, current LR: {current_lr:.6f}, avg loss: {avg_loss:.6f}")

            except RuntimeError as e:
                # 【修复】区分可恢复和不可恢复的RuntimeError
                if "NaN" in str(e) or "nan" in str(e).lower():
                    consecutive_errors += 1
                    logging.error(
                        f"NaN training error (连续错误 {consecutive_errors}/{max_consecutive_errors}): {e}, "
                        f"skipping this sample"
                    )
                    optimizer.zero_grad(set_to_none=True)
                    
                    if consecutive_errors >= max_consecutive_errors:
                        logging.critical(
                            f"连续{max_consecutive_errors}次NaN错误，训练可能已损坏，中止训练并保存模型"
                        )
                        torch.save(obj=model.state_dict(), f="model.pth")
                        raise RuntimeError("训练因连续NaN错误而中止") from e
                    
                    continue
                elif "out of memory" in str(e).lower():
                    # CUDA OOM是不可恢复的，需要立即中止
                    logging.critical(f"CUDA Out of Memory: {e}")
                    torch.save(obj=model.state_dict(), f="model.pth")
                    raise RuntimeError("训练因显存不足而中止") from e
                else:
                    # 其他RuntimeError也可能是严重的
                    consecutive_errors += 1
                    logging.error(
                        f"RuntimeError (连续错误 {consecutive_errors}/{max_consecutive_errors}): {e}"
                    )
                    
                    if consecutive_errors >= max_consecutive_errors:
                        logging.critical(f"连续{max_consecutive_errors}次RuntimeError，中止训练")
                        torch.save(obj=model.state_dict(), f="model.pth")
                        raise RuntimeError("训练因连续RuntimeError而中止") from e
                    
                    optimizer.zero_grad(set_to_none=True)
                    continue
                    
            except Exception as e:
                # 【修复】其他异常也要计数，并有最大限制
                consecutive_errors += 1
                logging.error(
                    f"Training error (连续错误 {consecutive_errors}/{max_consecutive_errors}): {e}"
                )
                
                if consecutive_errors >= max_consecutive_errors:
                    logging.critical(f"连续{max_consecutive_errors}次错误，中止训练并保存模型")
                    torch.save(obj=model.state_dict(), f="model.pth")
                    raise RuntimeError("训练因连续错误而中止") from e
                
                optimizer.zero_grad(set_to_none=True)
                continue

    except KeyboardInterrupt:
        logging.info("Training interrupted by user.")
        # Save final model
        torch.save(obj=model.state_dict(), f="model.pth")
        logging.info(f"Final model saved, training rounds: {local_training_rounds}")
    except Exception as e:
        logging.error(f"Training loop error: {e}")
        # Save model before exiting
        torch.save(obj=model.state_dict(), f="model.pth")


if __name__ == "__main__":
    main()
