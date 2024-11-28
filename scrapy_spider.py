# scrapy_spider.py

import scrapy # type: ignore
import json
from session_utils import initialize_session_state, log_debug

from scrapy.crawler import CrawlerRunner  # type: ignore
from twisted.internet import reactor  # type: ignore
from scrapy.utils.project import get_project_settings  # type: ignore

from urllib.parse import urlparse

class AmandaSpider(scrapy.Spider):
    name = "amanda_spider"

    custom_settings = {
        'REQUEST_FINGERPRINTER_IMPLEMENTATION': '2.7',  # Updated to avoid deprecation
        'FEED_EXPORT_ENCODING': 'utf-8',
        'LOG_LEVEL': 'ERROR',  # Reduce Scrapy logs
    }

    def __init__(self, start_url, max_depth=2, max_pages=100, log_queue=None, *args, **kwargs):
        super(AmandaSpider, self).__init__(*args, **kwargs)
        self.start_urls = [start_url]
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.page_count = 0
        self.crawled_data = []
        self.log_queue = log_queue

    def parse(self, response):
        if self.page_count >= self.max_pages:
            return

        self.page_count += 1
        self.crawled_data.append({
            "source_url": response.url,
            "content": response.text
        })
        self.log_debug(f"Crawled {self.page_count}: {response.url}")

        if self.page_count >= self.max_pages:
            return

        # Extract links and follow them
        for href in response.css('a::attr(href)').getall():
            next_url = response.urljoin(href)
            if self._is_valid_url(next_url):
                yield scrapy.Request(next_url, callback=self.parse)

    @staticmethod
    def _is_valid_url(url):
        parsed = urlparse(url)
        return parsed.scheme in ('http', 'https')

    def closed(self, reason):
        # Save the crawled data
        with open("crawled_data.json", "w", encoding='utf-8') as f:
            json.dump(self.crawled_data, f, ensure_ascii=False, indent=4)
        self.log_debug(f"Spider closed: {reason}")

    def log_debug(self, message):
        if self.log_queue:
            self.log_queue.put(message)

# Function to run Scrapy spider in a separate process
def run_spider_process(start_url, max_depth, max_pages, log_queue):
    try:
        runner = CrawlerRunner(get_project_settings())
        deferred = runner.crawl(AmandaSpider, start_url=start_url, max_depth=max_depth, max_pages=max_pages, log_queue=log_queue)
        deferred.addBoth(lambda _: reactor.stop())
        reactor.run()
    except Exception as e:
        if log_queue:
            log_queue.put(f"Scrapy crawling error: {e}")
