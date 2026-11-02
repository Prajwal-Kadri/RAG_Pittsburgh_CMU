"""Collect an explicit source list, including PDF pages and official social links.

No evaluation questions or reference answers are read. Raw downloads are cached
by content hash; each output record points to the source URL and download hash.
Scanned PDFs can optionally use the macOS Vision helper during local collection.
Babel only needs the finished JSONL corpus, not OCR tools or this collector.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import urljoin, urlsplit

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from .common import normalize_text
from .crawl import extract_html


def ocr_text(data):
    """Read table cells across each visual row instead of down whole columns."""
    if not data.get("lines"):
        return data["text"]
    rows = []
    for line in sorted(data["lines"], key=lambda item: -(item["box"][1] + item["box"][3] / 2)):
        x, y, width, height = line["box"]
        center = y + height / 2
        if rows and abs(center - rows[-1]["center"]) < max(height, rows[-1]["height"]) * 0.5:
            rows[-1]["cells"].append((x, line["text"]))
        else:
            rows.append({"center": center, "height": height, "cells": [(x, line["text"])]})
    return "\n".join(" | ".join(text for _, text in sorted(row["cells"])) for row in rows)


def collect(url, cache, ocr_scanned=False, pdf_page_pattern=None, source_label=None, clean=False):
    response = requests.get(url, timeout=(10, 40), headers={"User-Agent": "CMU-11711-coursework-bot/1.0"})
    response.raise_for_status()
    digest = hashlib.sha256(response.content).hexdigest()
    is_pdf = response.content.startswith(b"%PDF-")
    raw = cache / (digest + (".pdf" if is_pdf else ".html"))
    raw.write_bytes(response.content)
    base = {"source_url": response.url, "domain": urlsplit(response.url).netloc,
            "source_sha256": digest, "collected_at": datetime.now(timezone.utc).isoformat()}
    if is_pdf:
        reader = PdfReader(io.BytesIO(response.content))
        texts = [p.extract_text(extraction_mode="layout") or "" for p in reader.pages]
        ocr_dir = cache / (digest + "-ocr")
        if sum(len(normalize_text(t)) for t in texts) < 200 and ocr_scanned:
            if not all((ocr_dir / f"page-{i+1}.json").exists() for i in range(len(texts))):
                subprocess.run(["swift", str(Path(__file__).with_name("ocr_pdf.swift")),
                                str(raw), str(ocr_dir)], check=True)
        pages = []
        recurring = defaultdict(list)
        for i, text in enumerate(texts, 1):
            kind = "pdf_text"
            if len(normalize_text(text)) < 200 and (ocr_dir / f"page-{i}.json").exists():
                text = ocr_text(json.loads((ocr_dir / f"page-{i}.json").read_text()))
                kind = "pdf_ocr"
            if len(text.strip()) < 60:
                continue
            if pdf_page_pattern and not re.search(pdf_page_pattern, text, re.I):
                continue
            title = source_label or urlsplit(response.url).path.rsplit("/", 1)[-1]
            pages.append({**base, "url": response.url + f"#page={i}", "title": title,
                          "text": normalize_text(f"{title}, page {i}. {text}"),
                          "page": i, "extraction": kind})
            for line in set(normalize_text(line) for line in text.splitlines()):
                if 40 <= len(line) <= 300 and ":" in line and sum(c.isalpha() for c in line) >= 12:
                    recurring[line].append(i)
        # Document-level labelled facts (often in game-note footers) lose their
        # subject when indexed only at the end of a page. Keep one copy with
        # its document title and the pages on which it actually appears.
        for line, locations in recurring.items():
            if len(locations) >= 2:
                key = hashlib.sha256(line.encode()).hexdigest()[:16]
                pages.append({**base, "url": response.url + "#label-" + key, "title": title,
                              "text": normalize_text(title + ". " + line),
                              "pages": locations, "extraction": "pdf_recurring_label"})
        return pages, {"requested_url": url, **base, "records": len(pages),
                       "pdf_pages": len(texts), "pdf_page_pattern": pdf_page_pattern,
                       "status": "ok" if pages else "no_text"}

    soup = BeautifulSoup(response.content, "html.parser")
    if clean:
        from .source_quality import clean_html
        html = clean_html(str(soup))
    else:
        html = str(soup)
    records, _ = extract_html(html, response.url)
    if clean:
        from .source_quality import table_records
        records = [r for r in records if r.get("extraction") != "table_row"] + table_records(html, response.url)
        # Some institutions publish their concise identity/affiliation in the
        # page description rather than body text. This is publisher metadata,
        # preserved verbatim with its source hash, not a generated answer.
        description = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
        value = normalize_text(description.get("content", "")) if description else ""
        if len(value) >= 40 and not any(value in r["text"] for r in records):
            title = normalize_text(soup.title.get_text(" ", strip=True)) if soup.title else response.url
            records.append({"url": response.url + "#published-page-description", "title": title,
                            "text": title + ". Published page description: " + value,
                            "extraction": "published_description"})
    # Some venues publish schema.org microdata instead of JSON-LD. Keep the
    # event name, date and venue together, without adding an inferred date.
    for event in soup.select('[itemscope][itemtype]'):
        if event.get("itemtype", "").rsplit("/", 1)[-1] not in {"Event", "MusicEvent", "TheaterEvent", "ComedyEvent", "SportsEvent"}:
            continue
        fields = []
        for key in ("name", "startDate", "endDate", "doorTime", "location", "performer"):
            nodes = event.select(f'[itemprop="{key}"]')
            if nodes:
                node = nodes[0]
                value = node.get("content") or node.get("datetime") or node.get_text(" ", strip=True)
                fields.append(f"{key}: {normalize_text(value)}")
        if fields:
            text = ". ".join(fields) + "."
            key = hashlib.sha256(text.encode()).hexdigest()[:16]
            title = normalize_text(soup.title.get_text(" ", strip=True)) if soup.title else response.url
            records.append({"url": response.url + "#microdata-" + key, "title": title,
                            "text": title + ". " + text, "extraction": "event_microdata"})
    # These attributes live in footers that the main-text extractor removes.
    # Preserve only hyperlinks actually published by this official source.
    socials = set()
    for anchor in soup.find_all("a", href=True):
        target = urljoin(response.url, anchor["href"])
        host = urlsplit(target).netloc.lower().removeprefix("www.")
        if (host in ({"instagram.com", "facebook.com", "twitter.com", "x.com", "youtube.com", "youtu.be"} if clean else {"instagram.com", "facebook.com", "twitter.com", "x.com"})
                and not re.search(r"/(?:intent|share|sharer)(?:[/.?]|$)", urlsplit(target).path)):
            socials.add((normalize_text(anchor.get_text(" ", strip=True)) or host, target))
    if socials:
        title = normalize_text(soup.title.get_text(" ", strip=True)) if soup.title else response.url
        text = title + ". Published social media links: " + "; ".join(f"{name}: {target}" for name, target in sorted(socials))
        records.append({"url": response.url + "#published-social-links", "title": title,
                        "text": text, "extraction": "published_links"})
    records = [{**record, **base} for record in records]
    if clean:
        from .source_quality import clean_text, record_junk
        records = [{**r, "text": clean_text(r["text"]), "collection_policy": "clean_html_v7"}
                   for r in records if not record_junk(r)]
        records = [r for r in records if r["text"]]
    return records, {"requested_url": url, **base, "records": len(records),
                     "status": "ok" if records else "no_text"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--ocr-scanned-pdfs", action="store_true")
    parser.add_argument("--pdf-page-pattern", help="Keep PDF pages matching a document-section regex")
    parser.add_argument("--source-labels", help="JSON map of PDF URLs to descriptive document titles")
    parser.add_argument("--clean-html", action="store_true", help="Remove navigation and related-content widgets (v7 collection)")
    args = parser.parse_args()
    labels = json.loads(Path(args.source_labels).read_text()) if args.source_labels else {}
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    log = output.with_suffix(".collection_log.jsonl")
    completed = set()
    if log.exists():
        completed = {entry["requested_url"] for entry in map(json.loads, log.read_text().splitlines())
                     if entry["status"] == "ok"}
        if completed and not output.exists():
            parser.error("Collection log exists but output is missing; use a new output path")
    urls = list(dict.fromkeys(line.strip() for line in Path(args.url_file).read_text().splitlines()
                             if line.strip() and not line.lstrip().startswith("#")))
    with output.open("a", encoding="utf-8") as handle, log.open("a", encoding="utf-8") as journal:
        for url in urls:
            if url in completed:
                continue
            try:
                pages, entry = collect(url, cache, args.ocr_scanned_pdfs, args.pdf_page_pattern, labels.get(url), args.clean_html)
                for page in pages:
                    handle.write(json.dumps(page, ensure_ascii=False) + "\n")
                handle.flush()
            except Exception as error:
                # Keep failures visible while preserving already collected pages.
                entry = {"requested_url": url, "status": "error", "error": str(error)}
            journal.write(json.dumps(entry, ensure_ascii=False) + "\n")
            journal.flush()
            print(json.dumps(entry, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
