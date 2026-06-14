import sys
import os
import re
import asyncio
import httpx
from bs4 import BeautifulSoup, Comment
from urllib.parse import urlparse

# Reconfigure stdout for UTF-8 encoding support on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Constants
CATALOG_PATH = r"c:\Users\chan\Desktop\HENRI 7B SWARM\HENRI\archive\source-catalog.md"
OUTPUT_DIR = r"c:\Users\chan\Desktop\HENRI 7B SWARM\HENRI\archive\raw_sources"
CONCURRENCY_LIMIT = 5
TIMEOUT = 30  # 30 seconds timeout for larger PDFs

DEFAULT_HEADERS = {
    "User-Agent": "HENRITrainingBot/1.0 (chandler.coleman34@gmail.com) httpx/0.24",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,application/pdf,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

def clean_filename(title: str) -> str:
    # Remove invalid characters for Windows file systems
    cleaned = re.sub(r'[\\/*?:"<>|]', "", title)
    cleaned = cleaned.replace(" ", "_")
    return cleaned[:100]  # Limit filename length

def parse_catalog(catalog_path: str):
    if not os.path.exists(catalog_path):
        print(f"[ERROR] Catalog file not found at: {catalog_path}")
        return []
        
    with open(catalog_path, "r", encoding="utf-8") as f:
        data = f.read()
        
    sources = []
    current_title = None
    lines = data.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        m_url = re.match(r'Verified URL:\s*(https?://\S+)', line)
        if m_url:
            url = m_url.group(1)
            title = current_title if current_title else url
            sources.append((title, url))
            current_title = None
        else:
            if re.match(r'^\d+$', line):
                continue
            if '[URL]' in line:
                current_title = line.replace('[URL]', '').strip()
            else:
                current_title = line
                
    return sources

def distill_html(html: str, url: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    
    # Remove script, style, nav, footer elements
    for element in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe", "noscript"]):
        element.decompose()
        
    for comment in soup.find_all(text=lambda text: isinstance(text, Comment)):
        comment.extract()
        
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
            rows = []
            for row in element.find_all("tr"):
                cells = [cell.get_text().strip() for cell in row.find_all(["td", "th"])]
                rows.append(" | ".join(cells))
            if rows:
                content_lines.append("\n" + "\n".join(rows) + "\n")
                
    body_text = "\n".join(content_lines)
    body_text = re.sub(r'\n{3,}', '\n\n', body_text)
    return body_text.strip()

async def fetch_and_save(client: httpx.AsyncClient, semaphore: asyncio.Semaphore, title: str, url: str):
    async with semaphore:
        # Pre-process arXiv URLs to target PDF directly for full papers
        target_url = url
        is_arxiv_abs = False
        if "arxiv.org/abs/" in url:
            arxiv_id = url.split("arxiv.org/abs/")[-1].strip()
            # We will fetch the PDF instead of the abstract page
            target_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            is_arxiv_abs = True
            
        print(f"[FETCH] Starting: {title} -> {target_url}")
        
        # Determine extension based on target_url first
        parsed_path = urlparse(target_url).path
        is_pdf_url = parsed_path.lower().endswith(".pdf") or "pdf" in target_url.lower()
        
        try:
            response = await client.get(target_url, timeout=TIMEOUT, follow_redirects=True)
            if response.status_code != 200:
                print(f"[WARN] Failed to fetch {target_url} (HTTP {response.status_code})")
                return False
                
            content_type = response.headers.get("content-type", "").lower()
            is_pdf = "application/pdf" in content_type or is_pdf_url
            
            safe_title = clean_filename(title)
            
            if is_pdf:
                file_path = os.path.join(OUTPUT_DIR, f"{safe_title}.pdf")
                with open(file_path, "wb") as f:
                    f.write(response.content)
                print(f"[SAVED] PDF: {file_path}")
            else:
                file_path = os.path.join(OUTPUT_DIR, f"{safe_title}.md")
                distilled = distill_html(response.text, target_url)
                
                # Prepend source info
                header = f"# {title}\nSource URL: {url}\n\n"
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(header + distilled)
                print(f"[SAVED] MD: {file_path}")
                
            return True
            
        except Exception as e:
            print(f"[ERROR] Exception fetching {title} ({target_url}): {str(e)}")
            return False

async def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sources = parse_catalog(CATALOG_PATH)
    print(f"[INIT] Parsed {len(sources)} sources from catalog.")
    
    if not sources:
        print("[FATAL] No sources to scrape.")
        return
        
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    # We use a persistent connection client with limits
    limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
    async with httpx.AsyncClient(headers=DEFAULT_HEADERS, limits=limits, follow_redirects=True) as client:
        tasks = [fetch_and_save(client, semaphore, title, url) for title, url in sources]
        results = await asyncio.gather(*tasks)
        
    success_count = sum(1 for r in results if r)
    print(f"[FINISHED] Successfully scraped {success_count}/{len(sources)} sources.")

if __name__ == "__main__":
    asyncio.run(main())
