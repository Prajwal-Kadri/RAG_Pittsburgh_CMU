from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from bs4.exceptions import ParserRejectedMarkup
from tqdm import tqdm

from .common import normalize_text


def extract_page(url: str) -> dict:
    response = requests.get(url, timeout=30, headers={"User-Agent": "CMU-11711-coursework-bot/1.0"})
    response.raise_for_status()
    if response.content.startswith(b"%PDF-") or "application/pdf" in response.headers.get("Content-Type", ""):
        from pypdf import PdfReader
        text = normalize_text(" ".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(response.content)).pages))
        return {"url": url, "domain": urlparse(url).netloc, "title": Path(urlparse(url).path).name, "text": text}
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()
    title = normalize_text(soup.title.get_text(" ", strip=True) if soup.title else "")
    text = normalize_text(soup.get_text(" ", strip=True))
    return {"url": url, "domain": urlparse(url).netloc, "title": title, "text": text}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--append", action="store_true", help="Preserve existing pages and skip previously saved URLs")
    args = parser.parse_args()

    urls = [line.strip() for line in Path(args.url_file).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if args.append and output.exists():
        seen = {json.loads(line)["url"] for line in output.read_text(encoding="utf-8").splitlines() if line.strip()}
    saved = 0
    with output.open("a" if args.append else "w", encoding="utf-8") as handle:
        for url in tqdm(urls, desc="Collecting"):
            if url in seen:
                continue
            try:
                page = extract_page(url)
                if len(page["text"]) >= 200:
                    handle.write(json.dumps(page, ensure_ascii=False) + "\n")
                    handle.flush()
                    seen.add(url)
                    saved += 1
            except (requests.RequestException, ParserRejectedMarkup) as error:
                print(f"WARNING: skipped {url}: {error}")
    print(f"Saved {saved} new pages to {args.out}")


if __name__ == "__main__":
    main()

