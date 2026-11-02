"""Source-derived cleanup shared by the v7 collector and retrieval policy.

No question IDs, expected answers, or leaderboard predictions are used here.
"""
import re
import hashlib
from urllib.parse import urldefrag

from bs4 import BeautifulSoup

from .common import normalize_text, tokenize


RELATED = re.compile(r"\b(?:Related Stories\s*(?:&|and)\s*Events|News Feed|Related Videos|You Might Also Like)\b", re.I)
SOCIAL_QUERY = re.compile(r"instagram|facebook|twitter|youtube|social media|handle|username", re.I)


def clean_html(html):
    soup = BeautifulSoup(html, "html.parser")
    # Keep JSON-LD for structured event extraction; remove presentation widgets.
    for node in soup.select("nav, header, footer, aside, form, [role='navigation'], .sidearm-story-related, .sidearm-content-social-links, .sidearm-schedule-stories, .related-posts, .related-stories, .related-events, .news-feed, .skip-link"):
        node.decompose()
    for node in soup.find_all(["h2", "h3", "h4", "h5"]):
        if RELATED.fullmatch(normalize_text(node.get_text(" ", strip=True))):
            # Remove a distinct related-content section, never an entire main/article.
            parent = node.find_parent(["section", "aside"])
            if parent is not None and parent.find(["h1", "article"]) is None:
                parent.decompose()
    return str(soup)


def clean_text(text):
    match = RELATED.search(text)
    if match:
        text = text[:match.start()]
    text = re.sub(r"\bSkip to (?:main )?content\b", " ", text, flags=re.I)
    return normalize_text(text)


def table_records(html, url):
    """Keep each table row with its published heading and column labels."""
    soup = BeautifulSoup(html, "html.parser")
    title = normalize_text(soup.title.get_text(" ", strip=True)) if soup.title else url
    records = []
    for table in soup.find_all("table"):
        first = table.find("tr")
        if not first:
            continue
        cells = first.find_all(["th", "td"], recursive=False)
        labels = [normalize_text(c.get_text(" ", strip=True)) for c in cells]
        has_headers = bool(first.find("th")) or bool(labels and all(v and not re.search(r"\d", v) for v in labels))
        if not has_headers:
            labels = []
        heading = table.find_previous(["h1", "h2", "h3", "h4"])
        context = normalize_text(heading.get_text(" ", strip=True)) if heading else title
        for row in table.find_all("tr"):
            if row is first and has_headers:
                continue
            fields = []
            for i, cell in enumerate(row.find_all(["td", "th"], recursive=False)):
                label = cell.get("data-label") or (labels[i] if i < len(labels) else f"Field {i+1}")
                fields.append(f"{label}: {normalize_text(cell.get_text(' ', strip=True))}")
            if not fields:
                continue
            text = title + ". " + context + ". " + "; ".join(fields) + "."
            key = hashlib.sha256(text.encode()).hexdigest()[:16]
            records.append({"url": url + "#table-" + key, "title": title,
                            "text": text, "extraction": "table_row"})
    return records


def source_page(url):
    return urldefrag(url)[0].rstrip("/")


def record_junk(record):
    text = record.get("text", "")
    if not text.strip():
        return True
    if len(text) < 1500 and re.search(r"\b(?:page not found|page (?:you are|you're) looking for|access denied|verify you are human|checking your browser|just a moment|403 forbidden|404 not found)\b", text, re.I):
        return True
    # Bulk press-release archives mix years and unrelated events in single chunks.
    if "/City-Press-Release-Archive/Special-Events" in record.get("url", ""):
        return True
    kind = record.get("extraction", "")
    if kind == "section" and re.search(r"(?:>|[.!])\s*(?:Related Stories\s*(?:&|and)\s*Events|News Feed|Related Videos|You Might Also Like)\b", text, re.I):
        return True
    if re.search(r"(?:Go To Coaching Staff|There is no additional information to display|\bHeadlines\b|VIEW ALL EVENTS)\s*[.!]?\s*$", text, re.I):
        # Keep a real dated game or a labelled staff/player field.
        if not re.search(r"startDate:|Opponent:|\b(?:Name|Title):", text):
            return True
    if re.search(r"Field 1: Manager & Coaches; Field 2: Title", text):
        return True
    return False


def candidate_junk(record, query):
    if record_junk(record):
        return True
    if record.get("extraction") == "published_links" or "#published-social-links" in record.get("url", ""):
        return not SOCIAL_QUERY.search(query)
    return False


def near_duplicate(left, right, threshold=0.9):
    a, b = tokenize(left), tokenize(right)
    if a == b:
        return True
    # Different dates, accession numbers, counts, etc. are distinct evidence.
    if {w for w in a if any(c.isdigit() for c in w)} != {w for w in b if any(c.isdigit() for c in w)}:
        return False
    if min(len(a), len(b)) < 12 or min(len(a), len(b)) / max(len(a), len(b)) < 0.8:
        return False
    aa, bb = set(zip(a, a[1:])), set(zip(b, b[1:]))
    return bool(aa and bb and len(aa & bb) / len(aa | bb) >= threshold)
