#!/usr/bin/env python3
"""
HENRI 7B Agent Swarm Corpus Compiler
------------------------------------
A robust, concurrent crawler designed to scrape, distill, and merge web-based
source URLs into a single, high-density training corpus (Markdown or XML).

Requirements:
    pip install httpx beautifulsoup4

Usage:
    1. Create a file named 'sources.txt' with one URL per line.
    2. Run the script:
       python compile_corpus.py --input sources.txt --output training_corpus.xml --format xml
"""

import asyncio
import argparse
import sys
import os
import re
from typing import List, Dict, Any
from urllib.parse import urlparse
import httpx
from bs4 import BeautifulSoup, Comment

# Standard browser headers to avoid simple bot blocking
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
}

class CorpusCompiler:
    def __init__(self, urls: List[str], concurrency_limit: int = 5, timeout: int = 15, delay: float = 0.5):
        self.urls = urls
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self.timeout = timeout
        self.delay = delay
        self.results: List[Dict[str, Any]] = []

    async def fetch_url(self, client: httpx.AsyncClient, url: str) -> Dict[str, Any]:
        """Fetches a single URL asynchronously with retry logic and rate-limiting."""
        async with self.semaphore:
            print(f"[FETCH] Starting: {url}")
            for attempt in range(3):
                try:
                    response = await client.get(url, timeout=self.timeout)
                    if response.status_code == 200:
                        print(f"[SUCCESS] Fetched {url} (Attempt {attempt + 1})")
                        return {"url": url, "html": response.text, "status": 200, "error": None}
                    elif response.status_code in [403, 404, 500]:
                        print(f"[WARN] HTTP {response.status_code} for {url}")
                        return {"url": url, "html": None, "status": response.status_code, "error": f"HTTP {response.status_code}"}
                except (httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                    print(f"[TIMEOUT] Attempt {attempt + 1} failed for {url}")
                except httpx.RequestError as e:
                    print(f"[ERROR] Attempt {attempt + 1} request failed for {url}: {str(e)}")
                
                if attempt < 2:
                    await asyncio.sleep(self.delay * (attempt + 1))
            
            return {"url": url, "html": None, "status": 0, "error": "Max retries exceeded"}

    def distill_html(self, html: str, url: str) -> Dict[str, str]:
        """Cleans and parses raw HTML to maximize training token density."""
        soup = BeautifulSoup(html, "html.parser")
        
        # 1. Extract Title
        title = soup.title.string.strip() if soup.title else ""
        if not title:
            # Fallback to domain and path
            parsed = urlparse(url)
            title = f"{parsed.netloc}{parsed.path}"
        
        # 2. Strip Noise (elements that pollute LLM training with boilerplate/junk)
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe", "noscript"]):
            element.decompose()
        
        # Remove HTML comments
        for comment in soup.find_all(text=lambda text: isinstance(text, Comment)):
            comment.extract()

        # 3. Parse and Convert Core Text Blocks to Clean Markdown-like prose
        content_lines = []
        for element in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "code", "table"]):
            tag = element.name
            text = element.get_text().strip()
            if not text:
                continue
            
            if tag == "h1":
                content_lines.append(f"\n# {text}\n")
            elif tag == "h2":
                content_lines.append(f"\n## {text}\n")
            elif tag == "h3":
                content_lines.append(f"\n### {text}\n")
            elif tag in ["h4", "h5", "h6"]:
                content_lines.append(f"\n#### {text}\n")
            elif tag == "p":
                content_lines.append(f"\n{text}\n")
            elif tag == "li":
                content_lines.append(f"* {text}")
            elif tag in ["pre", "code"]:
                content_lines.append(f"\n```\n{text}\n```\n")
            elif tag == "table":
                # Convert simple table rows
                rows = []
                for row in element.find_all("tr"):
                    cells = [cell.get_text().strip() for cell in row.find_all(["td", "th"])]
                    rows.append(" | ".join(cells))
                if rows:
                    content_lines.append("\n" + "\n".join(rows) + "\n")
        
        # Deduplicate excessive newlines
        body_text = "\n".join(content_lines)
        body_text = re.sub(r'\n{3,}', '\n\n', body_text)
        
        return {
            "title": title,
            "body": body_text.strip()
        }

    async def run(self):
        """Orchestrates the fetch, parse, and compilation phases."""
        print(f"[START] Commencing crawl of {len(self.urls)} targets...")
        limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
        async with httpx.AsyncClient(headers=DEFAULT_HEADERS, limits=limits, follow_redirects=True) as client:
            tasks = [self.fetch_url(client, url) for url in self.urls]
            raw_responses = await asyncio.gather(*tasks)
            
            print("\n[PROCESS] Fetches complete. Commencing parsing and distillation...")
            success_count = 0
            for resp in raw_responses:
                if resp["html"]:
                    distilled = self.distill_html(resp["html"], resp["url"])
                    self.results.append({
                        "url": resp["url"],
                        "title": distilled["title"],
                        "body": distilled["body"]
                    })
                    success_count += 1
                else:
                    self.results.append({
                        "url": resp["url"],
                        "title": "Failed Connection",
                        "body": f"Extraction failed. Error: {resp['error']} (Status: {resp['status']})"
                    })
            
            print(f"[COMPLETED] Distilled {success_count}/{len(self.urls)} documents successfully.")

    def compile_markdown(self) -> str:
        """Assembles results into a single clean Markdown file."""
        markdown_blocks = [
            "# HENRI 7B Swarm Agent Token-Training Corpus",
            f"Generated: {os.popen('date').read().strip()}",
            f"Total Documents: {len(self.results)}",
            "==================================================\n\n"
        ]
        
        for idx, item in enumerate(self.results, 1):
            markdown_blocks.append(f"## Document {idx}: {item['title']}")
            markdown_blocks.append(f"**URL Source**: {item['url']}")
            markdown_blocks.append(f"**Format**: Markdown-Prose Distillation")
            markdown_blocks.append("---\n")
            markdown_blocks.append(item['body'])
            markdown_blocks.append("\n\n<!-- DOCUMENT_BOUNDARY_MARKER -->\n\n")
            
        return "\n".join(markdown_blocks)

    def compile_xml(self) -> str:
        """Assembles results into a highly-structured XML document."""
        xml_blocks = [
            '<?xml version="1.0" encoding="utf-8"?>',
            '<corpus name="HENRI 7B Agent Swarm Training Data">'
        ]
        
        for idx, item in enumerate(self.results, 1):
            # Clean text values for safe XML embedding
            clean_title = self.escape_xml(item['title'])
            clean_url = self.escape_xml(item['url'])
            clean_body = self.escape_xml(item['body'])
            
            xml_blocks.append(f'  <document id="{idx}">')
            xml_blocks.append(f'    <metadata>')
            xml_blocks.append(f'      <title>{clean_title}</title>')
            xml_blocks.append(f'      <url>{clean_url}</url>')
            xml_blocks.append(f'    </metadata>')
            xml_blocks.append(f'    <content>')
            xml_blocks.append(clean_body)
            xml_blocks.append(f'    </content>')
            xml_blocks.append(f'  </document>')
            
        xml_blocks.append('</corpus>')
        return "\n".join(xml_blocks)

    @staticmethod
    def escape_xml(text: str) -> str:
        """Escapes string characters for valid XML nesting."""
        if not text:
            return ""
        text = text.replace("&", "&amp;")
        text = text.replace("<", "&lt;")
        text = text.replace(">", "&gt;")
        text = text.replace('"', "&quot;")
        text = text.replace("'", "&apos;")
        return text

def parse_args():
    parser = argparse.ArgumentParser(description="Compile raw web URLs into LLM training corpora.")
    parser.add_argument("--input", "-i", default="sources.txt", help="Path to input text file containing URLs (one per line).")
    parser.add_argument("--output", "-o", default="compiled_corpus.xml", help="Path to save compiled document.")
    parser.add_argument("--format", "-f", choices=["xml", "markdown"], default="xml", help="Target output format.")
    parser.add_argument("--concurrency", "-c", type=int, default=5, help="Number of parallel request workers.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.input):
        print(f"[FATAL] Input file '{args.input}' not found. Please populate it with your source URLs.")
        sys.exit(1)
        
    with open(args.input, "r") as f:
        urls = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
        
    if not urls:
        print("[FATAL] No URLs found in input file.")
        sys.exit(1)
        
    compiler = CorpusCompiler(urls, concurrency_limit=args.concurrency)
    
    # Run async loop
    asyncio.run(compiler.run())
    
    # Compile output
    print(f"[WRITE] Generating final file in {args.format.upper()} format...")
    if args.format == "xml":
        compiled_data = compiler.compile_xml()
    else:
        compiled_data = compiler.compile_markdown()
        
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(compiled_data)
        
    print(f"[SUCCESS] Pipeline complete. Training file compiled successfully at: {args.output}")

if __name__ == "__main__":
    main()
