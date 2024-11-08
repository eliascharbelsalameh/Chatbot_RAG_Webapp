# web_crawl.py

import requests
from bs4 import BeautifulSoup  # type: ignore
import json
from urllib.parse import urljoin, urlparse
import time

class WebCrawler:
    def __init__(self, start_url, max_depth=5, delay=1):
        """
        Initializes the WebCrawler.

        Args:
            start_url (str): The URL to start crawling from.
            max_depth (int): The maximum depth to crawl.
            delay (int): Delay in seconds between requests to prevent overloading servers.
        """
        self.start_url = start_url
        self.max_depth = max_depth
        self.delay = delay
        self.visited = set()
        self.data = []

    def is_valid_url(self, url):
        """
        Validates the URL.

        Args:
            url (str): The URL to validate.

        Returns:
            bool: True if the URL is valid, False otherwise.
        """
        parsed = urlparse(url)
        return bool(parsed.netloc) and bool(parsed.scheme)

    def crawl(self):
        """
        Starts the crawling process.
        """
        self._crawl_recursive(self.start_url, 0)
        self._save_data()

    def _crawl_recursive(self, current_url, depth):
        """
        Recursively crawls web pages up to the specified depth.

        Args:
            current_url (str): The current URL to crawl.
            depth (int): The current depth level.
        """
        if depth > self.max_depth:
            print(f"Maximum depth reached at URL: {current_url}")
            return
        if current_url in self.visited:
            print(f"Already visited URL: {current_url}")
            return

        print(f"Crawling: {current_url} | Depth: {depth}")
        self.visited.add(current_url)

        try:
            response = requests.get(current_url, timeout=5)
            if response.status_code != 200:
                print(f"Failed to retrieve {current_url} | Status Code: {response.status_code}")
                return
            soup = BeautifulSoup(response.text, 'html.parser')
            text = self._extract_text(soup)
            self.data.append({
                "source_url": current_url,
                "content": text
            })
        except requests.RequestException as e:
            print(f"Request failed for {current_url} | Error: {e}")
            return

        # Find all anchor tags and extract URLs
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            next_url = urljoin(current_url, href)
            if self.is_valid_url(next_url) and next_url not in self.visited:
                print(f"Queueing URL: {next_url} | Depth: {depth + 1}")
                time.sleep(self.delay)  # Respectful crawling
                self._crawl_recursive(next_url, depth + 1)

    def _extract_text(self, soup):
        """
        Extracts and cleans text from the HTML content.

        Args:
            soup (BeautifulSoup): Parsed HTML content.

        Returns:
            str: Cleaned text content.
        """
        # Remove script and style elements
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()

        # Get text
        text = soup.get_text(separator=' ')

        # Collapse whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        return text

    def _save_data(self, filename="crawled_data.json"):
        """
        Saves the crawled data to a JSON file.

        Args:
            filename (str): The name of the JSON file.
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=4)
            print(f"Data saved to {filename}")
        except IOError as e:
            print(f"Failed to save data to {filename} | Error: {e}")
