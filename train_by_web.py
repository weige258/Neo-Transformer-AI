import torch
import logging
from main import train, generation, model
from crawler import WebCrawler
from record import get_loss

crawler = WebCrawler()


local_training_rounds = 0
save_interval = 300

while True:
    try:
        text = crawler.get()
        if text != None:
            train(answer=text)

            if(get_loss()<2.0):
                    # Generate response to see the progress
                    generation(text)
        
        local_training_rounds += 1
        print("*" * 100, flush=True)

        # Save model periodically
        if local_training_rounds % save_interval == 0:
            torch.save(obj=model.state_dict(), f="model.pth")
            logging.info(f"Model saved, training rounds: {local_training_rounds}")
    except RuntimeError as e:
        # 【修复】专门处理CUDA运行时错误
        error_msg = str(e)
        if "CUDA" in error_msg or "cuda" in error_msg.lower():
            logging.error(f"CUDA运行时错误: {e}")
            logging.info("尝试恢复CUDA状态...")
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    logging.info("GPU缓存已清理，尝试继续训练")
                except Exception as cleanup_error:
                    logging.error(f"清理GPU缓存失败: {cleanup_error}")
                    # 如果清理失败，跳过当前样本
                    continue
            else:
                logging.warning("CUDA不可用，跳过当前样本")
                continue
        else:
            # 其他RuntimeError，记录并跳过
            logging.error(f"RuntimeError: {e}, skipping this sample")
            continue
    except Exception as e:
        logging.error(f"Web crawling error: {e}")
        continue
