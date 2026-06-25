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
                try:
                    data = json.load(f)
                except MemoryError:
                    logging.warning(f"内存不足，无法加载 {file_path} 进行计数，该文件暂时跳过")
                    import gc; gc.collect()
                    return 0
            
            count = 0
            for item in data:
                if "ask" in item and "answer" in item:
                    ask_raw = item.get("ask")
                    answer_raw = item.get("answer")
                    if ask_raw is not None and answer_raw is not None:
                        ask = str(ask_raw).strip()
                        answer = str(answer_raw).strip()
                        if ask and answer:
                            count += 1
            return count
        except MemoryError:
            logging.warning(f"内存不足，无法处理 {file_path}，该文件暂时跳过")
            import gc; gc.collect()
            return 0
        except Exception as e:
            logging.error(f"Failed to count entries in {file_path}: {e}")
            return 0
    
    def get_random_sample(self, max_retries: int = 10) -> Dict[str, Optional[str]]:
        """随机获取一个训练样本（包含ask, think, answer, history）
        
        当加载某个文件失败（如内存不足）时，自动重试其他随机样本，
        而不是让异常传播导致训练退出。
        """
        if self.total_entries == 0:
            raise ValueError("No training data available")
        
        for attempt in range(max_retries):
            target_idx = random.randint(0, self.total_entries - 1)
            
            cumulative = 0
            for file_idx, count in enumerate(self.file_entry_counts):
                if cumulative + count > target_idx:
                    local_idx = target_idx - cumulative
                    try:
                        result = self._load_entry_from_file(self.dataset_files[file_idx], local_idx)
                        return result
                    except (MemoryError, ValueError) as e:
                        logging.warning(
                            f"加载样本失败 (尝试 {attempt + 1}/{max_retries}): {e}，"
                            f"跳过该样本，重新随机选择"
                        )
                        import gc; gc.collect()
                        break
                    except Exception as e:
                        logging.warning(
                            f"加载样本异常 (尝试 {attempt + 1}/{max_retries}): {e}，"
                            f"跳过该样本，重新随机选择"
                        )
                        break
                cumulative += count
        
        raise RuntimeError(f"连续 {max_retries} 次加载样本均失败，请检查数据集文件")
    
    def _load_entry_from_file(self, file_path: str, target_local_idx: int) -> Dict[str, Optional[str]]:
        """从文件中加载指定索引的条目
        
        修复：添加文件大小限制和安全校验，防止恶意JSON导致内存耗尽
        内存不足时抛出异常，由 get_random_sample 捕获并重试其他样本
        """
        # 【安全修复】检查文件大小，限制为100MB
        max_file_size = 100 * 1024 * 1024  # 100MB
        file_size = os.path.getsize(file_path)
        if file_size > max_file_size:
            raise ValueError(
                f"文件 {file_path} 过大 ({file_size / 1024 / 1024:.2f}MB > {max_file_size / 1024 / 1024:.2f}MB)，"
                f"可能存在安全风险或格式错误"
            )
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    raise ValueError(f"JSON解析失败 {file_path}: {e}") from e
                except MemoryError:
                    import gc; gc.collect()
                    raise MemoryError(f"JSON文件 {file_path} 加载时内存不足，文件可能过大或格式异常")
        except (ValueError, MemoryError):
            raise
        except Exception as e:
            if "内存不足" in str(e) or "MemoryError" in type(e).__name__:
                import gc; gc.collect()
                raise MemoryError(f"JSON文件 {file_path} 加载时内存不足: {e}")
            raise ValueError(f"无法打开文件 {file_path}: {e}")
        
        current_idx = 0
        for item in data:
            if "ask" in item and "answer" in item:
                ask_raw = item.get("ask")
                answer_raw = item.get("answer")
                
                if ask_raw is not None and answer_raw is not None:
                    ask = str(ask_raw).strip()
                    answer = str(answer_raw).strip()
                    
                    if ask and answer:
                        if current_idx == target_local_idx:
                            think_raw = item.get("think", "")
                            think = str(think_raw).strip() if think_raw is not None else ""
                            
                            history_raw = item.get("history", [])
                            if isinstance(history_raw, list) and len(history_raw) > 0:
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


def main() -> None:
    """Main training loop
    
    修复：精细化异常处理，区分可恢复错误和不可恢复错误
    当发生内存不足等可恢复错误时，自动重试或重启训练循环，而不是退出
    只有 KeyboardInterrupt 才会真正退出
    """
    # 使用流式数据集，不再一次性加载所有数据到内存
    dataset = StreamingDataset("dataset")

    if dataset.total_entries == 0:
        logging.error("No training data found, please check dataset files")
        return

    logging.info(f"Initialized streaming dataset with {dataset.total_entries} training samples.")

    local_training_rounds = 0
    save_interval = 500
    consecutive_sample_errors = 0
    max_consecutive_sample_errors = 50
    
    recent_losses = []
    loss_window_size = 100
    
    while True:
        try:
            while True:
                try:
                    sample = dataset.get_random_sample()
                    consecutive_sample_errors = 0
                except (MemoryError, ValueError, RuntimeError) as e:
                    consecutive_sample_errors += 1
                    if consecutive_sample_errors >= max_consecutive_sample_errors:
                        logging.error(
                            f"连续 {max_consecutive_sample_errors} 次加载样本失败: {e}，"
                            f"尝试重建数据集索引..."
                        )
                        try:
                            dataset = StreamingDataset("dataset")
                            if dataset.total_entries == 0:
                                logging.error("重建索引后仍无数据，等待10秒后重试...")
                                import time; time.sleep(10)
                            consecutive_sample_errors = 0
                        except Exception as rebuild_err:
                            logging.error(f"重建数据集索引失败: {rebuild_err}，等待10秒后重试...")
                            import time; time.sleep(10)
                        continue
                    logging.warning(
                        f"获取样本失败 (连续第 {consecutive_sample_errors} 次): {e}，"
                        f"稍后重试..."
                    )
                    import time; time.sleep(min(consecutive_sample_errors * 0.5, 5.0))
                    import gc; gc.collect()
                    continue
                except Exception as e:
                    consecutive_sample_errors += 1
                    logging.warning(f"获取样本异常: {e}，跳过重试...")
                    import gc; gc.collect()
                    continue
                
                ask = sample.get("ask", "")
                think = sample.get("think", "")
                answer = sample.get("answer", "")
                history_context = sample.get("history_context", "")
                
                if not ask or not answer:
                    continue

                try:
                    train(
                        ask=ask,
                        think=think if think else None,
                        answer=answer,
                        history_context=history_context if history_context else None
                    )
                    
                    local_training_rounds += 1
                    
                    current_lr = optimizer.param_groups[0]['lr']
                    
                    import record
                    if record.record_count > 0:
                        avg_recent_loss = record.total_loss / record.record_count
                        recent_losses.append(avg_recent_loss)
                        if len(recent_losses) > loss_window_size:
                            recent_losses.pop(0)
                    
                    print("*" * 100, flush=True)

                    if local_training_rounds % save_interval == 0:
                        torch.save(obj=model.state_dict(), f="model.pth")
                        avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0
                        logging.info(f"Model saved, training rounds: {local_training_rounds}, current LR: {current_lr:.6f}, avg loss: {avg_loss:.6f}")

                except RuntimeError as e:
                    if "NaN" in str(e) or "nan" in str(e).lower():
                        logging.error(f"NaN training error: {e}, skipping this sample")
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    elif "out of memory" in str(e).lower():
                        logging.warning(f"CUDA Out of Memory: {e}, skipping this sample")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    else:
                        logging.error(f"RuntimeError: {e}, skipping this sample")
                        optimizer.zero_grad(set_to_none=True)
                        continue
                        
                except Exception as e:
                    error_msg = str(e)
                    
                    if "cannot convert float NaN to integer" in error_msg or "nan" in error_msg.lower():
                        logging.error(f"NaN training error: {e}, skipping this sample")
                    elif "out of memory" in error_msg.lower():
                        logging.warning(f"CUDA Out of Memory: {e}, skipping this sample")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        logging.error(f"Training error: {e}, skipping this sample")
                    
                    optimizer.zero_grad(set_to_none=True)
                    continue

        except KeyboardInterrupt:
            logging.info("Training interrupted by user.")
            torch.save(obj=model.state_dict(), f="model.pth")
            logging.info(f"Final model saved, training rounds: {local_training_rounds}")
            return
        
        except MemoryError as e:
            logging.error(f"内存不足导致训练循环异常: {e}")
            torch.save(obj=model.state_dict(), f="model.pth")
            logging.info("模型已保存，清理内存后自动重启训练...")
            import gc; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import time; time.sleep(5)
            try:
                dataset = StreamingDataset("dataset")
                if dataset.total_entries == 0:
                    logging.error("重启后无数据，等待30秒后再次尝试...")
                    time.sleep(30)
            except Exception:
                logging.error("重启时重建数据集失败，等待30秒后再次尝试...")
                time.sleep(30)
            continue
        
        except Exception as e:
            logging.error(f"训练循环意外异常: {e}")
            torch.save(obj=model.state_dict(), f="model.pth")
            logging.info("模型已保存，5秒后自动重启训练...")
            import time; time.sleep(5)
            import gc; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                dataset = StreamingDataset("dataset")
            except Exception as rebuild_err:
                logging.error(f"重启时重建数据集失败: {rebuild_err}，等待30秒后再次尝试...")
                time.sleep(30)
            continue


if __name__ == "__main__":
    main()