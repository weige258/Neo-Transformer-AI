import threading
import queue
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import random
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import logging
from datetime import datetime
import string

# 配置日志 - 只输出关键信息
logging.basicConfig(
    level=logging.WARNING,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


class WebCrawler:
    """多线程并发爬虫系统"""
    
    # 成员变量
    url_queue: queue.Queue              # 待爬取URL队列
    visited_urls: set                   # 已爬取URL集合
    failed_urls: set                    # 失败URL集合
    cache: deque                        # 爬取结果缓存（FIFO）
    cache_lock: threading.Lock          # 缓存线程锁
    
    queue_threshold: int                # 队列阈值
    max_workers: int                    # 最大工作线程数
    max_retries: int                    # 最大重试次数
    timeout: int                        # 请求超时时间
    max_cache_size: int                 # 缓存最大大小
    
    is_running: bool                    # 是否运行中
    stop_event: threading.Event         # 停止事件
    executor: ThreadPoolExecutor        # 线程池
    
    def __init__(self, 
                 seed_urls=None,
                 queue_threshold=5,
                 max_workers=4,
                 max_retries=3,
                 timeout=10,
                 max_cache_size=100):
        """
        初始化爬虫
        
        Args:
            seed_urls: 种子URL列表
            queue_threshold: 队列阈值，队列大小小于此值时添加新URL
            max_workers: 最大工作线程数
            max_retries: 最大重试次数
            timeout: 请求超时时间（秒）
            max_cache_size: 缓存最大大小
        """
        self.queue_threshold = queue_threshold
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_cache_size = max_cache_size
        
        # URL队列和已爬取URL集合
        self.url_queue = queue.Queue()
        self.visited_urls = set()
        self.failed_urls = set()
        
        # 数据缓存（FIFO）
        self.cache = deque(maxlen=max_cache_size)
        self.cache_lock = threading.Lock()
        
        # 控制标志
        self.is_running = False
        self.stop_event = threading.Event()
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # 初始化种子URL或随机生成
        if seed_urls:
            # 如果seed_urls是字符串，转换为列表
            if isinstance(seed_urls, str):
                seed_urls = [seed_urls]
            
            # 添加到队列，并检查URL格式
            for url in seed_urls:
                # 如果URL没有协议前缀，添加https://
                if not url.startswith(('http://', 'https://')):
                    url = 'https://' + url
                self.url_queue.put(url)
        else:
            # 如果没有提供种子URL，自动生成随机URL初始化队列
            initial_urls = self._generate_random_urls(max(5, queue_threshold))
            for url in initial_urls:
                self.url_queue.put(url)
        
        # 启动各个后台线程
        self._start_threads()
    
    def _start_threads(self):
        """启动所有后台线程"""
        self.is_running = True
        
        # 启动爬虫工作线程
        for i in range(self.max_workers):
            self.executor.submit(self._crawler_worker)
        
        # 启动队列管理线程
        threading.Thread(target=self._queue_manager, daemon=True).start()
        
        # 启动内存清理线程（每10分钟清理一次）
        threading.Thread(target=self._memory_cleaner, daemon=True).start()
    
    def _generate_random_urls(self, num=5):
        """生成随机URL"""
        result = []
        for _ in range(num):
            chars = string.ascii_letters + string.digits
            middle = ''.join(random.choices(chars, k=random.randint(3, 30)))
            back = random.choice(['com', 'org', 'net', 'io'])
            url = 'http://' + middle + '.' + back
            result.append(url)
        return result
    
    def _crawler_worker(self):
        """爬虫工作线程"""
        while self.is_running and not self.stop_event.is_set():
            try:
                # 获取URL（非阻塞，超时1秒）
                try:
                    url = self.url_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # 跳过已访问或已失败的URL
                if url in self.visited_urls or url in self.failed_urls:
                    self.url_queue.task_done()
                    continue
                
                self.visited_urls.add(url)
                
                # 爬取网页
                success = self._fetch_and_parse(url)
                
                if success:
                    print(f"爬取成功: {url}", flush=True)
                else:
                    self.failed_urls.add(url)
                    print(f"失败url: {url}", flush=True)
                
                self.url_queue.task_done()
                
            except Exception as e:
                pass
    
    def _get_headers(self):
        """获取反爬虫伪装请求头"""
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.59',
            'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Mobile/15E148 Safari/604.1',
        ]
        
        headers = {
            'User-Agent': random.choice(user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0',
            'Referer': 'https://www.google.com/',
        }
        return headers
    
    def _fetch_and_parse(self, url):
        """爬取并解析网页"""
        try:
            # 随机延迟（1-3秒）- 避免请求过快被封IP
            time.sleep(random.uniform(1, 3))
            
            # 获取伪装请求头
            headers = self._get_headers()
            
            # 使用Session保持连接
            session = requests.Session()
            session.headers.update(headers)
            
            # 发送请求（支持重定向和cookie）
            response = session.get(
                url, 
                headers=headers,
                timeout=self.timeout,
                allow_redirects=True,
                verify=True
            )
            response.raise_for_status()
            
            # 解析HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 提取所有链接
            links = soup.find_all('a', href=True)
            sub_urls = []
            
            for link in links:
                href = link['href']
                # 转换相对路径为绝对路径
                absolute_url = urljoin(url, href)
                
                # 同域检查和去重
                if self._is_valid_url(absolute_url) and absolute_url not in self.visited_urls:
                    sub_urls.append(absolute_url)
            
            # 添加子URL到队列（限制数量）
            for sub_url in sub_urls[:5]:  # 每个页面最多添加5个子URL
                if sub_url not in self.visited_urls and sub_url not in self.failed_urls:
                    self.url_queue.put(sub_url)
            
            # 清洗HTML
            cleaned_content = self._clean_html(soup)
            
            # 保存到缓存
            self._add_to_cache({
                'url': url,
                'title': soup.title.string if soup.title else 'N/A',
                'content': cleaned_content,
                'timestamp': datetime.now().isoformat(),
                'sub_urls_count': len(sub_urls)
            })
            
            return True
            
        except requests.RequestException as e:
            logger.warning(f"请求失败 {url}: {e}")
            return False
        except Exception as e:
            logger.error(f"解析错误 {url}: {e}")
            return False
    
    def _clean_html(self, soup):
        """清洗HTML，保留有意义的内容"""
        # 移除脚本和样式
        for script in soup(['script', 'style']):
            script.decompose()
        
        # 获取文本
        text = soup.get_text()
        
        # 清理空白
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text[:1000]  # 限制长度为前1000个字符
    
    def _is_valid_url(self, url):
        """检查URL是否有效"""
        try:
            result = urlparse(url)
            return all([result.scheme in ['http', 'https'], result.netloc])
        except:
            return False
    
    def _queue_manager(self):
        """队列管理线程"""
        while self.is_running and not self.stop_event.is_set():
            try:
                # 检查队列大小
                current_size = self.url_queue.qsize()
                
                if current_size < self.queue_threshold:
                    # 生成新的随机URL
                    new_urls = self._generate_random_urls(self.queue_threshold - current_size)
                    for url in new_urls:
                        if url not in self.visited_urls and url not in self.failed_urls:
                            self.url_queue.put(url)
                
                # 定期检查（每2秒）
                time.sleep(2)
                
            except Exception as e:
                pass
    
    def _memory_cleaner(self):
        """内存清理线程 - 每10分钟清理一次所有成员数据"""
        while self.is_running and not self.stop_event.is_set():
            try:
                # 等待10分钟（600秒）
                time.sleep(600)
                
                if not self.is_running:
                    break
                
                # 清理缓存（保留visited_urls和failed_urls以防止重复爬取）
                print("【内存清理】开始清理缓存...", flush=True)
                print(f"  清理前 - visited_urls: {len(self.visited_urls)}, "
                      f"failed_urls: {len(self.failed_urls)}, "
                      f"cache: {len(self.cache)}", flush=True)
                
                # 清理缓存
                with self.cache_lock:
                    self.cache.clear()
                
                print(f"  清理后 - visited_urls: {len(self.visited_urls)}, "
                      f"failed_urls: {len(self.failed_urls)}, "
                      f"cache: {len(self.cache)}", flush=True)
                print("【内存清理】完成\n", flush=True)
                
            except Exception as e:
                pass
    
    def _add_to_cache(self, data):
        """将数据添加到缓存"""
        with self.cache_lock:
            self.cache.append(data)
    
    def get(self, timeout=10):
        """
        获取成功爬取的清洗过的文本
        
        Args:
            timeout: 超时时间（秒），如果缓存为空，最多等待这么长时间
            
        Returns:
            str: 清洗后的HTML或文本
            None: 超时或无数据
        """
        start_time = time.time()
        
        while True:
            with self.cache_lock:
                if len(self.cache) > 0:
                    data = self.cache.popleft()
                    return data['content']
            
            # 检查超时
            if time.time() - start_time > timeout:
                return None
            
            # 等待100ms后重试
            time.sleep(0.1)
    
    def get_batch(self, count=10, timeout=30):
        """
        批量获取数据
        
        Args:
            count: 获取的数据数量
            timeout: 超时时间（秒）
            
        Returns:
            list: 爬取的数据列表
        """
        result = []
        start_time = time.time()
        
        while len(result) < count:
            if time.time() - start_time > timeout:
                logger.warning(f"[get_batch] 超时，仅获取 {len(result)}/{count} 条数据")
                break
            
            data = self.get(timeout=1)
            if data:
                result.append(data)
            else:
                time.sleep(0.5)
        
        return result
    
    def get_status(self):
        """获取爬虫状态"""
        return {
            'queue_size': self.url_queue.qsize(),
            'visited_count': len(self.visited_urls),
            'failed_count': len(self.failed_urls),
            'cache_size': len(self.cache),
        }
    
    def stop(self, timeout=5):
        """停止爬虫"""
        self.is_running = False
        self.stop_event.set()
        
        # 等待所有任务完成
        try:
            self.executor.shutdown(wait=True, timeout=timeout)
        except:
            pass
    
    def __del__(self):
        """析构函数"""
        if self.is_running:
            self.stop()