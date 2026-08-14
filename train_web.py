"""网页爬虫在线训练脚本

特性：
- 从随机种子URL开始爬取，自动递归发现新链接
- Ctrl+C 优雅退出：保存模型
- 健壮异常处理：训练永不退出（除Ctrl+C）
"""

import torch
import logging
import time
import signal
import sys

from crawler import WebCrawler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

crawler: WebCrawler | None = None

local_training_rounds = 0
save_interval = 1000
empty_poll_sleep = 2.0


def _create_crawler() -> WebCrawler:
    global crawler
    if crawler is None:
        crawler = WebCrawler(max_workers=4, max_sub_urls_per_page=20)
    return crawler

_train_fn = None
_gen_fn = None
_model_obj = None
_optimizer_obj = None

_shutdown_requested = False


def _signal_handler(sig, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print("\n\n[Ctrl+C] 收到退出信号，正在保存模型和爬虫状态...", flush=True)


signal.signal(signal.SIGINT, _signal_handler)


def _lazy_import():
    global _train_fn, _gen_fn, _model_obj, _optimizer_obj
    if _train_fn is None:
        from main import train, generation, model, optimizer
        _train_fn, _gen_fn, _model_obj, _optimizer_obj = train, generation, model, optimizer


def _save_checkpoint():
    if _model_obj is not None:
        try:
            torch.save(obj=_model_obj.state_dict(), f="model.pth")
            logging.info(f"模型已保存, 训练轮数: {local_training_rounds}")
        except Exception as e:
            logging.error(f"保存模型失败: {e}")


def _print_status():
    if crawler is None:
        logging.info("爬虫尚未初始化，无法显示状态。")
        return
    status = crawler.get_status()
    logging.info(
        f"训练轮数: {local_training_rounds} | "
        f"尝试: {status['attempt_count']} | "
        f"成功: {status['success_count']} | "
        f"失败: {status['fail_count']} | "
        f"成功率: {status['success_rate']:.1f}% | "
        f"队列: {status['queue_size']} | "
        f"缓存: {status['cache_size']}"
    )


_wait_counter = 0

def main() -> None:
    global _wait_counter

    _create_crawler()

    while True:
        if _shutdown_requested:
            _save_checkpoint()
            if crawler is not None:
                crawler.stop()
            logging.info("训练已停止，再见！")
            sys.exit(0)

        try:
            text = crawler.get(timeout=5)

            if text is None:
                _wait_counter += 1
                if _wait_counter % 10 == 1:
                    status = crawler.get_status()
                    logging.info(
                        f"等待网页内容... (已等{_wait_counter * empty_poll_sleep:.0f}秒) | "
                        f"尝试:{status['attempt_count']} 成功:{status['success_count']} "
                        f"成功率:{status['success_rate']:.1f}% | "
                        f"队列:{status['queue_size']} 缓存:{status['cache_size']}"
                    )
                time.sleep(empty_poll_sleep)
                continue

            _wait_counter = 0

            _lazy_import()
            train, generation, model = _train_fn, _gen_fn, _model_obj

            if not text or len(text.strip()) < 20:
                logging.info(f"跳过: 内容过短({len(text.strip()) if text else 0}字符)")
                continue

            logging.info(f"开始训练: {len(text)}字符内容")

            try:
                train(answer=text)
            except RuntimeError as e:
                error_msg = str(e)
                if "out of memory" in error_msg.lower():
                    logging.warning(f"CUDA OOM，跳过此样本")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    _optimizer_obj.zero_grad(set_to_none=True)
                    continue
                elif "nan" in error_msg.lower():
                    logging.error(f"NaN训练错误，跳过此样本")
                    _optimizer_obj.zero_grad(set_to_none=True)
                    continue
                else:
                    logging.error(f"训练RuntimeError: {e}，跳过此样本")
                    _optimizer_obj.zero_grad(set_to_none=True)
                    continue
            except Exception as e:
                logging.error(f"训练异常: {e}，跳过此样本")
                if _optimizer_obj is not None:
                    _optimizer_obj.zero_grad(set_to_none=True)
                continue

            local_training_rounds += 1

            if local_training_rounds % 50 == 0:
                _print_status()
                try:
                    import record
                    avg_loss = record.get_loss()
                    logging.info(f"当前平均Loss: {avg_loss:.6f}")
                except Exception:
                    pass

            if local_training_rounds % save_interval == 0:
                _save_checkpoint()

            print("*" * 100, flush=True)
            time.sleep(0.05)

        except KeyboardInterrupt:
            _save_checkpoint()
            if crawler is not None:
                crawler.stop()
            logging.info("训练已停止，再见！")
            sys.exit(0)

        except MemoryError as e:
            logging.error(f"内存不足: {e}，清理后继续...")
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(5)
            continue

        except Exception as e:
            logging.error(f"Web训练循环异常: {e}")
            time.sleep(2)
            continue


if __name__ == '__main__':
    main()