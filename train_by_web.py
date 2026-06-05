"""网页爬虫在线训练脚本

注意：此脚本从 main.py 导入 train 和 generation 函数，
会触发 main.py 的模块级初始化（加载模型、创建优化器等）。
建议直接运行 main.py 进行训练，此脚本作为辅助工具。

修复：
1. 添加 sleep 避免 CPU 空转 100%
2. 添加节流控制，防止请求过于频繁
3. 使用 lazy import 减少模块加载副作用
"""

import torch
import logging
import time

from crawler import WebCrawler

# 延迟导入，仅在需要时触发 main.py 的模块级初始化
def _get_main():
    from main import train, generation, model
    return train, generation, model

crawler = WebCrawler()


local_training_rounds = 0
save_interval = 300
empty_poll_sleep = 2.0  # 无数据时休眠2秒

# 【修复Bug #11】首次获取后缓存，避免每次迭代都调用
_train_fn, _gen_fn, _model_obj = None, None, None

while True:
    try:
        text = crawler.get(timeout=5)
        if text is not None:
            if _train_fn is None:
                _train_fn, _gen_fn, _model_obj = _get_main()
            train, generation, model = _train_fn, _gen_fn, _model_obj
            train(answer=text)

            loss_val = 10.0
            try:
                from record import get_loss
                loss_val = get_loss()
            except ImportError:
                pass
            
            if loss_val < 2.0:
                    # Generate response to see the progress
                    generation(text)
            
            local_training_rounds += 1
            print("*" * 100, flush=True)

            # Save model periodically
            if local_training_rounds % save_interval == 0:
                torch.save(obj=model.state_dict(), f="model.pth")
                logging.info(f"Model saved, training rounds: {local_training_rounds}")
        else:
            # 【修复】没有数据时休眠，避免CPU空转100%
            time.sleep(empty_poll_sleep)
            continue
        
        # 【修复】每次训练后短暂休眠，降低CPU占用
        time.sleep(0.1)
        
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
                    time.sleep(1.0)
                    continue
            else:
                logging.warning("CUDA不可用，跳过当前样本")
                time.sleep(1.0)
                continue
        else:
            # 其他RuntimeError，记录并跳过
            logging.error(f"RuntimeError: {e}, skipping this sample")
            time.sleep(0.5)
            continue
    except Exception as e:
        logging.error(f"Web crawling error: {e}")
        time.sleep(1.0)
        continue
