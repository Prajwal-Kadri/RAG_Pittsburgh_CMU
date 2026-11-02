"""Collect painting catalog records through the museum's public website API.

Collect the whole paintings category, independently of evaluation questions.
The API endpoint and public search credentials come from the museum web app.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from .common import normalize_text

BASE = "https://collection.carnegieart.org"
API = BASE + "/api/cmoa_objects/object/_search"


def public_session() -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = "CMU-11711-coursework-bot/1.0"
    response = session.get(BASE, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for script in soup.find_all("script", src=True):
        if not script["src"].startswith("/assets/index-"):
            continue
        bundle = session.get(urljoin(BASE, script["src"]), timeout=30)
        bundle.raise_for_status()
        home = re.search(r'import\("\./(Home-[^"/]+\.js)"\)', bundle.text)
        if not home:
            continue
        source = session.get(BASE + "/assets/" + home.group(1), timeout=30)
        source.raise_for_status()
        # Locate the browser's public collections search login, without logging it.
        login = re.search(r'"(collections:[^"\n]+)"', source.text)
        if login:
            session.auth = tuple(login.group(1).split(":", 1))
            return session
    raise RuntimeError("Museum app format changed: public search configuration was not found")


def catalog_page(record: dict) -> dict:
    identifier = record["id"].rsplit("/", 1)[-1]
    fields = {
        "Title": record.get("title"),
        "Creator": "; ".join(c.get("name", "") for c in record.get("creators", [])),
        "Accession number": record.get("accession_number"),
        "Department": record.get("department"),
        "Creation date": record.get("creation_date"),
        "Acquisition date": record.get("acquisition_date"),
        "Acquisition method": record.get("acqisition_method") or record.get("acquisition_method"),
        "Credit line": record.get("credit_line"),
        "Current location": record.get("current_location"),
        "Medium": record.get("medium_description") or record.get("medium"),
    }
    def value_text(value):
        return "; ".join(str(v) for v in value) if isinstance(value, list) else str(value)
    text = "Carnegie Museum of Art catalog. " + " ".join(
        f"{key}: {value_text(value)}." for key, value in fields.items() if value
    )
    return {"url": BASE + "/objects/" + identifier, "domain": "collection.carnegieart.org",
            "title": record.get("title", ""), "text": normalize_text(text),
            "source_api": API}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw/museum_paintings.jsonl")
    args = parser.parse_args()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if output.exists():
        seen = {json.loads(line)["url"] for line in output.read_text(encoding="utf-8").splitlines() if line.strip()}
    session = public_session()
    offset, total = 0, None
    with output.open("a", encoding="utf-8") as handle:
        while total is None or offset < total:
            response = session.post(API, json={"size": 200, "from": offset,
                "sort": [{"internal_id": "asc"}],
                "query": {"match": {"medium": "paintings"}}}, timeout=60)
            response.raise_for_status()
            hits = response.json()["hits"]
            total = hits["total"]
            if isinstance(total, dict):
                total = total["value"]
            if not hits["hits"] and offset < total:
                raise RuntimeError("Museum API returned an incomplete page")
            for hit in hits["hits"]:
                page = catalog_page(hit["_source"])
                if page["url"] not in seen:
                    handle.write(json.dumps(page, ensure_ascii=False) + "\n")
                    handle.flush()
                    seen.add(page["url"])
            offset += len(hits["hits"])
            print(f"Museum painting records: {offset}/{total}", flush=True)
            time.sleep(0.25)
    print(f"Saved {len(seen)} painting records to {output}")


if __name__ == "__main__":
    main()
