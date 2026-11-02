"""Collect an explicit URL list with source-labelled paragraph evidence.

Fetches publisher pages, not QA answers. Existing collectors and corpora stay
unchanged. The collection journal and raw content hashes permit reproduction.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote

from bs4 import BeautifulSoup
from pypdf import PdfReader

from .collect_sources import collect
from .common import normalize_text
from .source_quality import clean_html, record_junk


def publisher_blocks(html, base, title):
    soup = BeautifulSoup(clean_html(html), "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    body = soup.find("main") or soup.find(attrs={"role": "main"}) or soup.body or soup
    heading = title
    blocks, seen = [], set()
    for node in body.find_all(["h1", "h2", "h3", "h4", "p", "li", "dl"]):
        if node.find_parent(["p", "li", "dl"]):
            continue
        value = normalize_text(node.get_text(" ", strip=True))
        if node.name.startswith("h"):
            heading = value or heading
            continue
        if len(value) < 45 or len(value.split()) > 450:
            continue
        value = title + ". " + heading + ". " + value
        if value in seen:
            continue
        seen.add(value)
        digest = hashlib.sha256(value.encode()).hexdigest()[:16]
        row = {**base, "url": base["source_url"] + "#block-" + digest,
               "title": title, "text": value, "extraction": "published_block"}
        if not record_junk(row):
            blocks.append(row)
    # Preserve addresses the publisher exposes as mailto links, including
    # their visible labels. These are published contacts, never inferred ones.
    for node in soup.select('a[href^="mailto:"]'):
        address = unquote(node["href"][7:].split("?", 1)[0])
        if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", address):
            continue
        parent = node.find_parent(["p", "li", "div"])
        context = normalize_text(parent.get_text(" ", strip=True)) if parent else ""
        if len(context) > 1400:
            context = normalize_text(node.get_text(" ", strip=True))
        value = title + ". Published contact: " + context + ". Email: " + address + "."
        digest = hashlib.sha256(value.encode()).hexdigest()[:16]
        blocks.append({**base, "url": base["source_url"] + "#contact-" + digest,
                       "title": title, "text": value, "extraction": "published_contact"})
    # Public seal badges can have meaningful aria-labels and filenames while
    # their image alt text is empty. Preserve the publisher's attributes.
    original = BeautifulSoup(html, "html.parser")
    # Published JavaScript promotion cards: read literal string properties,
    # never execute the site's JavaScript. Preserve dates, category and title.
    for script in original.find_all("script"):
        code = script.string or script.get_text()
        match = re.search(r"(?:const|let|var)\s+contentItems\s*=\s*\[(.*?)\]\s*;", code, re.S)
        if not match:
            continue
        for fragment in re.findall(r"\{([^{}]*)\}", match.group(1), re.S):
            fields = {}
            for key, literal in re.findall(r"\b(tabId|subheadline|title|text)\s*:\s*('(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\")", fragment):
                try:
                    fields[key] = ast.literal_eval(literal)
                except (SyntaxError, ValueError):
                    pass
            if not fields.get("title") or not fields.get("subheadline"):
                continue
            value = title + ". Published promotion card. " + "; ".join(k + ": " + v for k, v in fields.items()) + "."
            digest = hashlib.sha256(value.encode()).hexdigest()[:16]
            blocks.append({**base, "url": base["source_url"] + "#promotion-" + digest,
                           "title": title + " — " + fields["title"], "text": value,
                           "extraction": "published_promotion_card"})
    for price in original.select(".elementor-price-table__price"):
        integer = price.select_one(".elementor-price-table__integer-part")
        fraction = price.select_one(".elementor-price-table__fractional-part")
        currency = price.select_one(".elementor-price-table__currency")
        card = price.find_parent(class_="elementor-price-table")
        heading = card.find(["h2", "h3", "h4"]) if card else None
        if not (integer and fraction and heading):
            continue
        amount = (currency.get_text(strip=True) if currency else "") + integer.get_text(strip=True) + "." + fraction.get_text(strip=True)
        value = title + ". Ticket category: " + heading.get_text(" ", strip=True) + ". Published price: " + amount + "."
        digest = hashlib.sha256(value.encode()).hexdigest()[:16]
        blocks.append({**base, "url": base["source_url"] + "#price-" + digest,
                       "title": title + " — " + heading.get_text(" ", strip=True),
                       "text": value, "extraction": "published_price_card"})
    for node in original.find_all("a", href=True):
        image = node.find("img")
        if image is None or not re.search(r"guidestar|candid|seal", str(node), re.I):
            continue
        if image.get("src", "").startswith("data:"):
            continue
        value = (title + ". Published badge. Label: " + node.get("aria-label", "")
                 + ". Image description: " + image.get("alt", "")
                 + ". Image URL: " + image.get("src", "")
                 + ". Destination: " + node["href"] + ".")
        digest = hashlib.sha256(value.encode()).hexdigest()[:16]
        blocks.append({**base, "url": base["source_url"] + "#badge-" + digest,
                       "title": title, "text": value, "extraction": "published_badge"})
    # Keep publisher identity/history statements that happen to be in footers.
    # Legal notices and navigational lists are not promoted to evidence.
    for footer in original.select('footer, .footer, #footer, [role="contentinfo"], .region-footer, #footer-wrapper'):
        for node in footer.find_all("p"):
            value = normalize_text(node.get_text(" ", strip=True))
            if 65 <= len(value) <= 700 and re.search(r"founded|established|season|professional|nonprofit", value, re.I):
                value = title + ". Published organization footer: " + value
                digest = hashlib.sha256(value.encode()).hexdigest()[:16]
                blocks.append({**base, "url": base["source_url"] + "#footer-" + digest,
                               "title": title, "text": value, "extraction": "published_footer"})
    # A source's season/history table can answer counting questions without
    # requiring an LLM to count rows spread across five unrelated chunks.
    for table in original.find_all("table"):
        headers = table.find("tr")
        if not headers or not re.search(r"\b(?:Year|Season)\b", headers.get_text()):
            continue
        years = []
        for row in table.find_all("tr")[1:]:
            cell = row.find(["td", "th"])
            if cell:
                value = cell.get_text(" ", strip=True)
                match = re.fullmatch(r"(20\d{2})(?:[–-]\d{2,4})?\s*[*†]?", value)
                if match:
                    years.append(value.rstrip("*† "))
        years = list(dict.fromkeys(years))
        if len(years) < 4:
            continue
        heading = table.find_previous(["h1", "h2", "h3", "h4"])
        context = heading.get_text(" ", strip=True) if heading else "Published history table"
        value = (title + ". " + context + ". Number of distinct listed season/year rows: "
                 + str(len(years)) + ". Listed seasons/years: " + ", ".join(years) + ". This counts this table's listed rows, not unlisted years.")
        digest = hashlib.sha256(value.encode()).hexdigest()[:16]
        blocks.append({**base, "url": base["source_url"] + "#yearrows-" + digest,
                       "title": title + " — " + context, "text": value,
                       "extraction": "published_year_row_count"})
    return blocks


def pdf_rows(path, base, title, max_pages=None):
    """Keep whitespace-delimited PDF rows with their printed year columns."""
    rows = []
    reader = PdfReader(path)
    for page_number, page in enumerate(reader.pages, 1):
        if max_pages and page_number > max_pages:
            break
        lines = (page.extract_text(extraction_mode="layout") or "").splitlines()
        years, section = [], ""
        for line in lines:
            line = re.sub(r"\$\s+(?=\d)", "$", line)
            cells = [x.strip() for x in re.split(r"\s{2,}", line.strip()) if x.strip()]
            if len(cells) == 1 and cells[0].isupper() and re.search(r"[A-Z]", cells[0]):
                section = cells[0]
            if len(cells) >= 2 and all(re.fullmatch(r"20\d{2}", c) for c in cells):
                years = cells
                continue
            if not years or len(cells) != len(years) + 1 or not any(c.isalpha() for c in cells[0]):
                continue
            if not all(re.fullmatch(r"[$\d,.()%\s−–-]+", c) for c in cells[1:]):
                continue
            value = (title + f". Page {page_number}. " + ("Source section: " + section + ". " if section else "") + "Row: " + cells[0] + ". "
                     + "; ".join(year + ": " + amount for year, amount in zip(years, cells[1:])) + ".")
            digest = hashlib.sha256(value.encode()).hexdigest()[:16]
            rows.append({**base, "url": base["source_url"] + "#pdfrow-" + digest,
                         "title": title + " — " + cells[0], "text": value,
                         "page": page_number, "extraction": "pdf_labelled_year_row"})
    return rows


def collect_one(url, cache, label=None):
    pages, entry = collect(url, cache, clean=True, source_label=label)
    if pages:
        first = pages[0]
        if re.search(r"/errors/not-found|/404(?:/|$)", first["source_url"]):
            raise ValueError("Publisher returned a not-found page")
        base = {k: first[k] for k in ("source_url", "domain", "source_sha256", "collected_at")}
        raw = cache / (first["source_sha256"] + ".html")
        if raw.exists():
            pages.extend(publisher_blocks(raw.read_bytes(), base, first["title"]))
        raw_pdf = cache / (first["source_sha256"] + ".pdf")
        if raw_pdf.exists():
            pages.extend(pdf_rows(raw_pdf, base, first["title"]))
    unique = {row["url"]: row for row in pages}
    entry["records"] = len(unique)
    return list(unique.values()), entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--source-labels", help="JSON mapping publisher PDF URLs to document titles")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    cache, out = Path(args.cache_dir), Path(args.out)
    labels = json.loads(Path(args.source_labels).read_text()) if args.source_labels else {}
    cache.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = out.with_suffix(".collection_log.jsonl")
    completed = set()
    if log.exists():
        completed = {row["requested_url"] for row in map(json.loads, log.read_text().splitlines())
                     if row["status"] == "ok"}
        if completed and not out.exists():
            parser.error("Collection output is missing; choose a new output path")
    urls = list(dict.fromkeys(line.strip() for line in Path(args.url_file).read_text().splitlines()
                             if line.strip() and not line.lstrip().startswith("#")))
    with out.open("a", encoding="utf-8") as handle, log.open("a", encoding="utf-8") as journal:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = {pool.submit(collect_one, url, cache, labels.get(url)): url for url in urls if url not in completed}
            for future in as_completed(pending):
                url = pending[future]
                try:
                    pages, entry = future.result()
                    for page in pages:
                        handle.write(json.dumps(page, ensure_ascii=False) + "\n")
                    handle.flush()
                except Exception as error:
                    entry = {"requested_url": url, "status": "error", "error": str(error)}
                journal.write(json.dumps(entry, ensure_ascii=False) + "\n")
                journal.flush()
                print(json.dumps(entry, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
