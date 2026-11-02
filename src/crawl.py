"""Bounded, resumable collection with row/event/section records.

Only follows configured hosts and paths. Does not read evaluation questions or
answers. Failed/empty/blocked pages are logged rather than treated as evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import time
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from bs4.exceptions import ParserRejectedMarkup
from pypdf.errors import PyPdfError

from .common import normalize_text
from .repair_records import sports_context, context_text, event_start, readable_date, empty_fragment

AGENT = "CMU-11711-coursework-bot/1.0"


def canonical(url: str) -> str:
    parts = urlsplit(url)
    query = [(k, v) for k, v in parse_qsl(parts.query)
             if k in {"year", "season", "view", "page", "grid", "date", "start", "end"}]
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path or "/",
                       urlencode(sorted(query)), ""))


def in_scope(url: str, scope: dict) -> bool:
    p = urlsplit(url)
    return (p.scheme in {"http", "https"} and not p.username and not p.password
            and p.netloc.lower() in scope["hosts"]
            and any(re.search(pattern, p.path) for pattern in scope["paths"])
            and not re.search(r"\.(?:jpe?g|png|gif|svg|mp4|zip|css|js)$", p.path, re.I))


def scalar(value) -> str:
    if isinstance(value, dict):
        return "; ".join(f"{k}: {scalar(v)}" for k, v in value.items()
                         if k in {"name", "streetAddress", "addressLocality", "addressRegion",
                                  "postalCode", "address", "startDate", "endDate", "description"})
    if isinstance(value, list):
        return "; ".join(scalar(v) for v in value)
    return normalize_text(str(value))


def extract_html(html: str, url: str) -> tuple[list[dict], list[str]]:
    soup = BeautifulSoup(html, "html.parser")
    title = normalize_text(soup.title.get_text(" ", strip=True) if soup.title else url)
    heading = soup.find("h1")
    context = normalize_text(heading.get_text(" ", strip=True)) if heading else title
    sport_fields = sports_context(url, title)
    if sport_fields:
        context = context_text(sport_fields) + " " + title
    links = [canonical(urljoin(url, a["href"])) for a in soup.find_all("a", href=True)]
    records = []

    # JSON-LD is structured public source content, not model-generated answers.
    def visit(value):
        if isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, dict):
            types = value.get("@type", [])
            if isinstance(types, str):
                types = [types]
            if any(t in {"Event", "SportsEvent", "MusicEvent", "TheaterEvent", "ComedyEvent"} for t in types):
                fields = [f"{key}: {scalar(value[key])}" for key in
                          ("name", "startDate", "endDate", "location", "performer", "description")
                          if value.get(key)]
                record = context + ". " + ". ".join(fields) + "."
                start = event_start(record)
                if start:
                    record += f" Game date: {readable_date(start)}."
                records.append(("event", record))
            for key, child in value.items():
                if key != "@context" and isinstance(child, (list, dict)):
                    visit(child)
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            visit(json.loads(script.string or script.get_text()))
        except (ValueError, TypeError):
            pass

    # Keep complete rows together, with headers and the page's entity/sport.
    for table in soup.find_all("table"):
        header_row = table.find("tr")
        headers = [normalize_text(c.get_text(" ", strip=True))
                   for c in header_row.find_all("th", recursive=False)] if header_row else []
        for row in table.find_all("tr"):
            cells = row.find_all(["td", "th"], recursive=False)
            if not cells or not row.find("td"):
                continue
            fields = []
            for i, cell in enumerate(cells):
                label = cell.get("data-label") or (headers[i] if i < len(headers) else f"Field {i+1}")
                fields.append(f"{label}: {normalize_text(cell.get_text(' ', strip=True))}")
            records.append(("table_row", context + ". " + "; ".join(fields) + "."))
    for game in soup.select(".sidearm-schedule-game"):
        records.append(("schedule_game", context + ". " + normalize_text(game.get_text(" ", strip=True)) + "."))

    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()
    body = soup.find("main") or soup.find(attrs={"role": "main"}) or soup.body or soup

    # Keep tax/FAQ headings attached to their paragraphs rather than losing them
    # at arbitrary sentence boundaries. Each section becomes a source record.
    current_heading = context
    heading_stack = {}
    paragraphs = []
    for node in body.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        if node.find_parent(["li", "p"]):
            continue
        text = normalize_text(node.get_text(" ", strip=True))
        if node.name.startswith("h"):
            if paragraphs:
                records.append(("section", context + ". " + current_heading + ". " + " ".join(paragraphs)))
            level = int(node.name[1])
            heading_stack = {k: v for k, v in heading_stack.items() if k < level}
            heading_stack[level] = text
            current_heading = " > ".join(heading_stack[k] for k in sorted(heading_stack))
            paragraphs = []
        elif text:
            paragraphs.append(text)
    if paragraphs:
        records.append(("section", context + ". " + current_heading + ". " + " ".join(paragraphs)))

    text = normalize_text(body.get_text(" ", strip=True))
    base = {"url": url, "domain": urlsplit(url).netloc, "title": title,
            "source_url": url, "collected_at": datetime.now(timezone.utc).isoformat()}
    pages = [{**base, "text": text, "extraction": "main_text"}] if len(text) >= 200 else []
    seen = set()
    for kind, text in records:
        text = normalize_text(text)
        if len(text) < 60 or text in seen or (sport_fields and empty_fragment({"text": text, "extraction": kind})):
            continue
        seen.add(text)
        key = hashlib.sha256(text.encode()).hexdigest()[:16]
        pages.append({**base, "url": url + "#record-" + key, "text": text, "extraction": kind,
                      **({"sports_context": sport_fields} if sport_fields else {})})
    return pages, list(dict.fromkeys(links))


def fetch(session: requests.Session, url: str, scope: dict):
    # Check every redirect before following it, so links cannot escape the scope.
    for _ in range(6):
        response = session.get(url, timeout=25, allow_redirects=False)
        if response.is_redirect:
            target = canonical(urljoin(url, response.headers["Location"]))
            if not in_scope(target, scope):
                raise ValueError("redirect outside configured scope")
            url = target
            continue
        response.raise_for_status()
        if response.content.startswith(b"%PDF-"):
            from pypdf import PdfReader
            text = normalize_text(" ".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(response.content)).pages))
            page = {"url": url, "source_url": url, "domain": urlsplit(url).netloc,
                    "title": urlsplit(url).path.rsplit("/", 1)[-1], "text": text, "extraction": "pdf"}
            return ([page] if len(text) >= 200 else []), []
        if "html" not in response.headers.get("Content-Type", "").lower():
            raise ValueError("unsupported content type")
        response.encoding = response.apparent_encoding
        return extract_html(response.text, url)
    raise ValueError("too many redirects")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="data/crawl_scopes.json")
    parser.add_argument("--out", default="data/raw/pages_crawled.jsonl")
    parser.add_argument("--max-pages", type=int, default=160, help="Maximum HTTP page attempts per invocation")
    parser.add_argument("--delay", type=float, default=0.3)
    args = parser.parse_args()
    if args.max_pages < 1 or args.delay < 0:
        parser.error("Require max-pages > 0 and delay >= 0")
    scopes = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = output.with_suffix(".crawl_log.jsonl")
    state_path = output.with_suffix(".state.json")
    # State preserves the frontier so a resumed run can discover children of
    # already downloaded directory pages without downloading those pages again.
    signature = hashlib.sha256(json.dumps(scopes, sort_keys=True).encode()).hexdigest()
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["config_hash"] != signature:
            parser.error("Config changed; choose a new output path for a new crawl")
        queue, visited, counts = deque(state["queue"]), set(state["visited"]), Counter(state["counts"])
    else:
        if output.exists() and output.stat().st_size:
            parser.error("Existing output has no crawl state; use a new output path")
        queue = deque((i, canonical(u), 0) for i, s in enumerate(scopes) for u in s["seeds"])
        visited, counts = set(), Counter()
    existing = {json.loads(line)["url"] for line in output.read_text().splitlines() if line.strip()} if output.exists() else set()
    session = requests.Session()
    session.headers["User-Agent"] = AGENT
    robots = {}
    attempted, saved = 0, 0
    def allowed(url):
        origin = urlunsplit((*urlsplit(url)[:2], "", "", ""))
        if origin not in robots:
            r = session.get(origin + "/robots.txt", timeout=15)
            robot = RobotFileParser()
            if r.status_code in (404, 410):
                robot.parse([])
            elif r.ok:
                robot.parse(r.text.splitlines())
            else:
                robot.parse(["User-agent: *", "Disallow: /"])
            robots[origin] = robot
        return robots[origin].can_fetch(AGENT, url)
    def checkpoint():
        state = {"config_hash": signature, "queue": list(queue), "visited": sorted(visited), "counts": dict(counts)}
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state), encoding="utf-8")
        tmp.replace(state_path)
    with output.open("a", encoding="utf-8") as handle, log_path.open("a", encoding="utf-8") as log:
        while queue and attempted < args.max_pages:
            index, url, depth = queue.popleft()
            scope = scopes[index]
            if url in visited or not in_scope(url, scope) or counts[str(index)] >= scope["max_pages"]:
                continue
            visited.add(url)
            counts[str(index)] += 1
            attempted += 1
            event = {"url": url, "scope": scope["name"], "depth": depth}
            try:
                if not allowed(url):
                    raise ValueError("robots.txt disallows fetch or was unavailable")
                pages, links = fetch(session, url, scope)
                for page in pages:
                    if page["url"] not in existing:
                        handle.write(json.dumps(page, ensure_ascii=False) + "\n")
                        handle.flush()
                        existing.add(page["url"])
                        saved += 1
                if depth < scope.get("max_depth", 1):
                    queued = {entry[1] for entry in queue}
                    for link in links:
                        if link not in visited and link not in queued and in_scope(link, scope):
                            queue.append((index, link, depth + 1))
                            queued.add(link)
                event.update(status="saved" if pages else "empty", records=len(pages))
            except (requests.RequestException, ParserRejectedMarkup, PyPdfError, ValueError, RuntimeError) as error:
                event.update(status="skipped", reason=str(error))
            log.write(json.dumps(event, ensure_ascii=False) + "\n")
            log.flush()
            checkpoint()
            print(f"[{attempted}] {scope['name']}: {event['status']} {url}", flush=True)
            time.sleep(args.delay)
    checkpoint()
    print(f"Saved {saved} records; attempted {attempted} URLs. Log: {log_path}")


if __name__ == "__main__":
    main()
