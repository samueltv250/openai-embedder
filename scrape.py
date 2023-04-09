import os
from urllib.parse import urlparse, urljoin

from bs4 import BeautifulSoup
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import time
import random


visited_urls = set()
output_folder = "scraped_pages"

def save_html(url, content):
    parsed_url = urlparse(url)
    page_path = os.path.join(output_folder, parsed_url.netloc, parsed_url.path.strip('/'))
    os.makedirs(os.path.dirname(page_path), exist_ok=True)
    with open(page_path + ".html", "w", encoding="utf-8") as file:
        file.write(content)

def is_valid_url(url):
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)

def get_all_links(url, soup):
    internal_urls = set()
    for link in soup.find_all("a"):
        href = link.attrs.get("href")
        if href == "" or href is None:
            continue

        href = urljoin(url, href)
        parsed_href = urlparse(href)
        href = parsed_href.scheme + "://" + parsed_href.netloc + parsed_href.path

        if not is_valid_url(href):
            continue

        if urlparse(url).netloc not in href:
            continue

        internal_urls.add(href)
    return internal_urls

def crawl(url, driver):
    global visited_urls

    if url in visited_urls:
        return

    print(f"Crawling: {url}")
    visited_urls.add(url)

    driver.get(url)
    
    time.sleep(random.uniform(1, 5))  # Sleep for a random time between 1 and 5 seconds.

    html = driver.page_source

    save_html(url, html)

    soup = BeautifulSoup(html, "html.parser")
    internal_links = get_all_links(url, soup)

    for link in internal_links:
        crawl(link, driver)

if __name__ == "__main__":
    start_url = "https://gprovivienda.com/"  # Replace this with the URL you want to scrape

    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)

    crawl(start_url, driver)

    driver.quit()
