import json
import torch
import random
import logging
from typing import List, Tuple
from main import train, generation, model, optimizer
from record import get_loss

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_dataset_files(dataset_dir: str = "dataset") -> List[str]:
    """Load all dataset JSON files from the specified directory"""
    import os
    
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
                if "prompt" in item and "response" in item:
                    prompt_raw = item.get("prompt")
                    response_raw = item.get("response")
                    if prompt_raw is not None and response_raw is not None:
                        prompt = str(prompt_raw).strip()
                        response = str(response_raw).strip()
                        if prompt and response:
                            count += 1
            return count
        except Exception as e:
            logging.error(f"Failed to count entries in {file_path}: {e}")
            return 0
    
    def get_random_pair(self) -> Tuple[str, str]:
        """随机获取一个训练对"""
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
    
    def _load_entry_from_file(self, file_path: str, target_local_idx: int) -> Tuple[str, str]:
        """从文件中加载指定索引的条目"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            current_idx = 0
            for item in data:
                if "prompt" in item and "response" in item:
                    prompt_raw = item.get("prompt")
                    response_raw = item.get("response")
                    
                    if prompt_raw is not None and response_raw is not None:
                        prompt = str(prompt_raw).strip()
                        response = str(response_raw).strip()
                        
                        if prompt and response:
                            if current_idx == target_local_idx:
                                return prompt, response
                            current_idx += 1
            
            raise IndexError(f"Entry {target_local_idx} not found in {file_path}")
        
        except Exception as e:
            logging.error(f"Failed to load entry from {file_path}: {e}")
            raise


def main() -> None:
    """Main training loop"""
    # 使用流式数据集，不再一次性加载所有数据到内存
    dataset = StreamingDataset("dataset")

    if dataset.total_entries == 0:
        logging.error("No training data found, please check dataset files")
        return

    logging.info(f"Initialized streaming dataset with {dataset.total_entries} training pairs.")

    training_rounds = 0
    save_interval = 500  # Save model every 500 rounds
    
    # 用于ReduceLROnPlateau的损失跟踪
    recent_losses = []
    loss_window_size = 100  # 计算平均loss的窗口大小

    try:
        while True:
            # 流式获取随机训练对，每次只加载一条数据
            prompt, response = dataset.get_random_pair()
            
            # Skip empty prompts or responses
            if not prompt or not response:
                continue

            # Train on this pair
            try:
                train(prompt, response)
                            
                training_rounds += 1
                
                # 获取当前学习率
                current_lr = optimizer.param_groups[0]['lr']
                
                # 记录最近loss用于监控
                from record import total_loss, record_count
                if record_count > 0:
                    avg_recent_loss = total_loss / record_count
                    recent_losses.append(avg_recent_loss)
                    if len(recent_losses) > loss_window_size:
                        recent_losses.pop(0)
                
                print("*" * 100, flush=True)

                # Save model periodically
                if training_rounds % save_interval == 0:
                    torch.save(obj=model.state_dict(), f="model.pth")
                    avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0
                    logging.info(f"Model saved, training rounds: {training_rounds}, current LR: {current_lr:.6f}, avg loss: {avg_loss:.6f}")

            except Exception as e:
                logging.error(f"Training error: {e}")
                continue

    except KeyboardInterrupt:
        logging.info("Training interrupted by user.")
        # Save final model
        torch.save(obj=model.state_dict(), f="model.pth")
        logging.info(f"Final model saved, training rounds: {training_rounds}")
    except Exception as e:
        logging.error(f"Training loop error: {e}")
        # Save model before exiting
        torch.save(obj=model.state_dict(), f="model.pth")


if __name__ == "__main__":
    main()