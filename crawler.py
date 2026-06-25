import threading
import queue
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import random
import time
from collections import deque, OrderedDict
from concurrent.futures import ThreadPoolExecutor
import logging
from datetime import datetime
import os
import json

import string

logging.basicConfig(
    level=logging.WARNING,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

_TLD_WEIGHTS = {
    "com": 500, "org": 50, "net": 40, "cn": 35, "de": 30, "uk": 28,
    "fr": 25, "ru": 22, "jp": 20, "br": 18, "in": 18, "au": 15,
    "it": 15, "es": 14, "ca": 14, "us": 12, "nl": 12, "pl": 10,
    "ch": 10, "se": 9, "no": 8, "at": 8, "be": 8, "dk": 7, "fi": 7,
    "pt": 7, "cz": 6, "ro": 6, "hu": 6, "gr": 6, "ie": 6, "kr": 6,
    "mx": 8, "ar": 7, "cl": 6, "co": 8, "pe": 5, "ve": 4, "tw": 8,
    "hk": 7, "sg": 7, "my": 5, "th": 5, "vn": 4, "ph": 4, "id": 5,
    "pk": 4, "bd": 3, "za": 5, "ng": 3, "eg": 3, "ke": 3, "ma": 3,
    "io": 15, "me": 8, "cc": 6, "tv": 6, "info": 8, "biz": 5,
    "app": 12, "dev": 10, "ai": 10, "tech": 6, "xyz": 5, "top": 5,
    "site": 4, "online": 4, "club": 3, "live": 3, "pro": 3,
    "sk": 4, "bg": 4, "hr": 3, "si": 2, "ee": 2, "lv": 2, "lt": 2,
    "lu": 2, "mt": 1, "cy": 2, "wiki": 3, "code": 3, "run": 2,
    "fun": 2, "win": 2, "one": 2, "world": 2, "life": 2,
    "ec": 2, "bo": 1, "py": 2, "uy": 1, "tz": 1, "gh": 1, "et": 1,
    "sn": 1, "cm": 1, "lab": 2,
}
_TLDS = list(_TLD_WEIGHTS.keys())
_TLD_WEIGHT_LIST = [_TLD_WEIGHTS[t] for t in _TLDS]


def _generate_random_url():
    """完全随机生成一个URL，如 https://www.a7xk2m.com
    TLD按真实网页出现频率加权采样，com概率最高
    """
    length = random.randint(3, 30)
    chars = string.ascii_lowercase + string.digits
    name = ''.join(random.choice(chars) for _ in range(length))
    tld = random.choices(_TLDS, weights=_TLD_WEIGHT_LIST, k=1)[0]
    prefix = random.choice(["www.", ""])
    return f"https://{prefix}{name}.{tld}"


class WebCrawler:
    """增量递归爬虫系统
    
    特性：
    - 从随机种子URL开始爬取
    - 自动解析HTML中的所有链接，未爬取过的递归加入队列
    - 队列空时自动补充随机种子URL
    - 支持Ctrl+C优雅退出
    - 增量爬取：已访问/失败的URL不会重复爬取
    """

    def __init__(self,
                 seed_urls=None,
                 queue_threshold=5,
                 max_workers=4,
                 max_retries=3,
                 timeout=10,
                 max_cache_size=100,
                 max_sub_urls_per_page=20,
                 state_file="crawler_state.json"):
        self.queue_threshold = queue_threshold
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_cache_size = max_cache_size
        self.max_sub_urls_per_page = max_sub_urls_per_page
        self.state_file = state_file

        self.url_queue = queue.Queue()
        self.visited_urls = OrderedDict()
        self.failed_urls = OrderedDict()
        self._MAX_URL_SET_SIZE = 100000

        self.cache = deque(maxlen=max_cache_size)
        self.cache_lock = threading.Lock()
        self.url_lock = threading.Lock()

        self.is_running = False
        self.stop_event = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        self._attempt_count = 0
        self._success_count = 0
        self._fail_count = 0
        self._stats_lock = threading.Lock()

        self._load_state()

        if seed_urls:
            if isinstance(seed_urls, str):
                seed_urls = [seed_urls]
            for url in seed_urls:
                if not url.startswith(('http://', 'https://')):
                    url = 'https://' + url
                with self.url_lock:
                    if url not in self.visited_urls and url not in self.failed_urls:
                        self.url_queue.put(url)
        elif self.url_queue.qsize() == 0:
            seeds = []
            for _ in range(3):
                seed = _generate_random_url()
                seeds.append(seed)
                self.url_queue.put(seed)
            print(f"[Crawler] 随机生成种子URL: {', '.join(seeds)}", flush=True)

        self._start_threads()

    def _load_state(self):
        if not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            for url in state.get("visited", []):
                self.visited_urls[url] = True
            for url in state.get("failed", []):
                self.failed_urls[url] = True
            for url in state.get("pending", []):
                if url not in self.visited_urls and url not in self.failed_urls:
                    self.url_queue.put(url)
            print(f"[Crawler] 加载状态: visited={len(self.visited_urls)}, "
                  f"failed={len(self.failed_urls)}, pending={self.url_queue.qsize()}", flush=True)
        except Exception as e:
            print(f"[Crawler] 加载状态失败: {e}，从零开始", flush=True)

    def save_state(self):
        try:
            pending = []
            temp_queue = queue.Queue()
            while not self.url_queue.empty():
                try:
                    url = self.url_queue.get_nowait()
                    pending.append(url)
                    temp_queue.put(url)
                except queue.Empty:
                    break
            while not temp_queue.empty():
                try:
                    self.url_queue.put(temp_queue.get_nowait())
                except queue.Empty:
                    break

            with self.url_lock:
                state = {
                    "visited": list(self.visited_urls.keys())[-50000:],
                    "failed": list(self.failed_urls.keys())[-10000:],
                    "pending": pending[-1000:],
                    "timestamp": datetime.now().isoformat(),
                }
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False)
            print(f"[Crawler] 状态已保存: visited={len(self.visited_urls)}, "
                  f"failed={len(self.failed_urls)}, pending={len(pending)}", flush=True)
        except Exception as e:
            print(f"[Crawler] 保存状态失败: {e}", flush=True)

    def _start_threads(self):
        self.is_running = True
        for i in range(self.max_workers):
            self.executor.submit(self._crawler_worker)
        threading.Thread(target=self._queue_manager, daemon=True).start()
        threading.Thread(target=self._memory_cleaner, daemon=True).start()
        threading.Thread(target=self._state_saver, daemon=True).start()

    def _crawler_worker(self):
        while self.is_running and not self.stop_event.is_set():
            try:
                try:
                    url = self.url_queue.get(timeout=1)
                except queue.Empty:
                    continue

                with self.url_lock:
                    if url in self.visited_urls or url in self.failed_urls:
                        self.url_queue.task_done()
                        continue
                    self.visited_urls[url] = True

                with self._stats_lock:
                    self._attempt_count += 1
                    attempt_num = self._attempt_count

                success = self._fetch_and_parse(url)

                if success:
                    with self.url_lock:
                        self.visited_urls[url] = True
                        if len(self.visited_urls) > self._MAX_URL_SET_SIZE:
                            while len(self.visited_urls) > self._MAX_URL_SET_SIZE // 2:
                                self.visited_urls.popitem(last=False)
                    with self._stats_lock:
                        self._success_count += 1
                else:
                    with self.url_lock:
                        self.failed_urls[url] = True
                        if len(self.failed_urls) > self._MAX_URL_SET_SIZE:
                            while len(self.failed_urls) > self._MAX_URL_SET_SIZE // 2:
                                self.failed_urls.popitem(last=False)
                    with self._stats_lock:
                        self._fail_count += 1

                self.url_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.warning(f"爬虫工作线程异常: {e}", exc_info=True)

    def _get_headers(self):
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0',
            'Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1',
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
        }
        return headers

    def _is_valid_url(self, url):
        try:
            result = urlparse(url)
            if result.scheme not in ['http', 'https']:
                return False
            if not result.netloc:
                return False
            path = result.path.lower()
            if any(ext in path for ext in [
                '.pdf', '.jpg', '.jpeg', '.png', '.gif', '.svg', '.ico',
                '.mp3', '.mp4', '.avi', '.mov', '.wmv', '.flv',
                '.zip', '.rar', '.7z', '.tar', '.gz',
                '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
                '.exe', '.dmg', '.apk', '.iso',
                '.css', '.js', '.json', '.xml', '.rss', '.atom',
            ]):
                return False
            if any(skip in url.lower() for skip in [
                'mailto:', 'tel:', 'javascript:', 'data:',
                'login', 'signin', 'signup', 'register',
                'logout', 'signout',
                'facebook.com/login', 'twitter.com/login',
                'accounts.google.com',
            ]):
                return False
            return True
        except Exception:
            return False

    def _fetch_and_parse(self, url):
        try:
            time.sleep(random.uniform(0.1, 0.5))

            headers = self._get_headers()
            session = requests.Session()
            session.headers.update(headers)

            response = session.get(
                url,
                headers=headers,
                timeout=self.timeout,
                allow_redirects=True,
                verify=True
            )
            response.raise_for_status()

            content_type = response.headers.get('Content-Type', '')
            if 'text/html' not in content_type and 'text/plain' not in content_type:
                print(f"[Crawler] ✗ {url} → 非HTML内容: {content_type[:50]}", flush=True)
                return False

            soup = BeautifulSoup(response.content, 'html.parser')

            links = soup.find_all('a', href=True)
            sub_urls = []

            for link in links:
                href = link['href']
                absolute_url = urljoin(url, href)

                if '#' in absolute_url:
                    absolute_url = absolute_url.split('#')[0]
                if '?' in absolute_url and len(absolute_url) > 500:
                    absolute_url = absolute_url.split('?')[0]
                if len(absolute_url) > 500:
                    continue

                if self._is_valid_url(absolute_url):
                    with self.url_lock:
                        if absolute_url not in self.visited_urls and absolute_url not in self.failed_urls:
                            sub_urls.append(absolute_url)

            random.shuffle(sub_urls)
            added_count = 0
            for sub_url in sub_urls[:self.max_sub_urls_per_page]:
                with self.url_lock:
                    if sub_url not in self.visited_urls and sub_url not in self.failed_urls:
                        self.url_queue.put(sub_url)
                        added_count += 1

            cleaned_content = self._clean_html(soup)

            if not cleaned_content or len(cleaned_content.strip()) < 50:
                print(f"[Crawler] ✗ {url} → 内容过短({len(cleaned_content.strip())}字符)", flush=True)
                return False

            self._add_to_cache({
                'url': url,
                'title': soup.title.string.strip() if soup.title and soup.title.string else 'N/A',
                'content': cleaned_content,
                'timestamp': datetime.now().isoformat(),
                'sub_urls_found': len(sub_urls),
                'sub_urls_added': added_count,
            })

            with self._stats_lock:
                rate = self._success_count / max(self._attempt_count, 1) * 100
            print(f"[Crawler] ✓ {url} → {len(cleaned_content)}字符, {len(sub_urls)}链接, +{added_count}入队 (成功率{rate:.1f}%)", flush=True)
            return True

        except requests.exceptions.ConnectionError as e:
            reason = str(e)
            if "Name or service not known" in reason or "getaddrinfo failed" in reason:
                short = "DNS失败(域名不存在)"
            elif "Connection refused" in reason:
                short = "连接被拒绝"
            elif "timed out" in reason:
                short = "连接超时"
            else:
                short = reason[:60]
            print(f"[Crawler] ✗ {url} → {short}", flush=True)
            return False
        except requests.exceptions.TooManyRedirects:
            print(f"[Crawler] ✗ {url} → 重定向过多", flush=True)
            return False
        except requests.RequestException as e:
            print(f"[Crawler] ✗ {url} → 请求错误: {str(e)[:60]}", flush=True)
            return False
        except Exception as e:
            print(f"[Crawler] ✗ {url} → 解析错误: {str(e)[:60]}", flush=True)
            return False

    def _clean_html(self, soup):
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe', 'noscript']):
            tag.decompose()

        for tag in soup.find_all(['div', 'section', 'article']):
            if tag.get('role') in ['navigation', 'banner', 'contentinfo', 'complementary']:
                tag.decompose()

        text = soup.get_text(separator='\n')

        lines = []
        for line in text.splitlines():
            line = line.strip()
            if line and len(line) > 2:
                lines.append(line)

        text = '\n'.join(lines)
        return text[:3000]

    def _queue_manager(self):
        while self.is_running and not self.stop_event.is_set():
            try:
                current_size = self.url_queue.qsize()

                if current_size < self.queue_threshold:
                    new_urls = []
                    for _ in range(max(1, self.queue_threshold - current_size)):
                        url = _generate_random_url()
                        self.url_queue.put(url)
                        new_urls.append(url)
                    print(f"[Crawler] 队列不足({current_size}), 补充{len(new_urls)}个随机URL: {new_urls[0]}...", flush=True)

                time.sleep(5)

            except Exception as e:
                logger.warning(f"队列管理线程异常: {e}", exc_info=True)

    def _memory_cleaner(self):
        while self.is_running and not self.stop_event.is_set():
            try:
                time.sleep(600)
                if not self.is_running:
                    break

                with self.cache_lock:
                    self.cache.clear()

                import gc
                gc.collect()

            except Exception as e:
                logger.warning(f"内存清理线程异常: {e}", exc_info=True)

    def _state_saver(self):
        while self.is_running and not self.stop_event.is_set():
            try:
                time.sleep(300)
                if self.is_running:
                    self.save_state()
            except Exception as e:
                logger.warning(f"状态保存线程异常: {e}", exc_info=True)

    def _add_to_cache(self, data):
        with self.cache_lock:
            self.cache.append(data)

    def get(self, timeout=10):
        start_time = time.time()
        while True:
            with self.cache_lock:
                if len(self.cache) > 0:
                    data = self.cache.popleft()
                    return data['content']
            if time.time() - start_time > timeout:
                return None
            time.sleep(0.1)

    def get_with_meta(self, timeout=10):
        start_time = time.time()
        while True:
            with self.cache_lock:
                if len(self.cache) > 0:
                    return self.cache.popleft()
            if time.time() - start_time > timeout:
                return None
            time.sleep(0.1)

    def get_batch(self, count=10, timeout=30):
        result = []
        start_time = time.time()
        while len(result) < count:
            if time.time() - start_time > timeout:
                break
            data = self.get(timeout=1)
            if data:
                result.append(data)
            else:
                time.sleep(0.5)
        return result

    def get_status(self):
        with self._stats_lock:
            attempt = self._attempt_count
            success = self._success_count
            fail = self._fail_count
            rate = success / max(attempt, 1) * 100
        return {
            'queue_size': self.url_queue.qsize(),
            'visited_count': len(self.visited_urls),
            'failed_count': len(self.failed_urls),
            'cache_size': len(self.cache),
            'attempt_count': attempt,
            'success_count': success,
            'fail_count': fail,
            'success_rate': rate,
        }

    def add_seed_url(self, url):
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        with self.url_lock:
            if url not in self.visited_urls and url not in self.failed_urls:
                self.url_queue.put(url)

    def add_seed_urls(self, urls):
        for url in urls:
            self.add_seed_url(url)

    def stop(self, timeout=5):
        self.is_running = False
        self.stop_event.set()

        self.save_state()

        try:
            import sys
            if sys.version_info >= (3, 9):
                self.executor.shutdown(wait=True, cancel_futures=True)
            else:
                self.executor.shutdown(wait=False)
                start_time = time.time()
                while time.time() - start_time < timeout:
                    if not any(t.is_alive() for t in threading.enumerate()
                              if t != threading.main_thread() and t.name.startswith('ThreadPool')):
                        break
                    time.sleep(0.1)
        except Exception as e:
            logger.warning(f"线程池关闭异常: {e}", exc_info=True)

        print("[Crawler] 爬虫已停止", flush=True)

    def __del__(self):
        if self.is_running:
            self.stop()