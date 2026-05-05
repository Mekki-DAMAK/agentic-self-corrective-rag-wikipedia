from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from self_rag_pro.ingest.sample_corpus import sample_articles  # noqa: E402
from self_rag_pro.ingest.storage import write_jsonl  # noqa: E402
from self_rag_pro.utils.config import ensure_dirs, load_config  # noqa: E402


def slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def fetch_wikipedia_article(title: str, lang: str = "en") -> dict:
    txt_api = f"https://{lang}.wikipedia.org/w/api.php"
    params: dict[str, str | int | float | None] = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "format": "json",
        "titles": title,
        "redirects": 1,
    }
    headers = {
        "User-Agent": "self-rag-wikipedia-demo/1.0 (educational demo; contact: example@example.com)"
    }
    r = requests.get(txt_api, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    pages = r.json()["query"]["pages"]
    page = next(iter(pages.values()))
    text = page.get("extract", "")
    canonical_title = page.get("title", title)
    return {
        "id": slug(canonical_title),
        "title": canonical_title,
        "text": text,
        "url": f"https://{lang}.wikipedia.org/wiki/{canonical_title.replace(' ', '_')}",
    }


def merge_by_id(docs: list[dict]) -> list[dict]:
    merged = {}
    for doc in docs:
        merged[doc["id"]] = doc
    return list(merged.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    lang = cfg["wikipedia"].get("language", "en")
    docs = []
    for title in tqdm(cfg["wikipedia"]["articles"], desc="Downloading Wikipedia"):
        try:
            doc = fetch_wikipedia_article(title, lang)
            if len(doc["text"]) > 500:
                docs.append(doc)
            else:
                print(f"[WARN] Article too short/skipped: {title}")
        except Exception as exc:
            print(f"[WARN] Failed article '{title}': {exc}")
    if not docs:
        docs = sample_articles()
        print("[WARN] Wikipedia download failed or returned no documents. Using bundled sample corpus.")
    else:
        docs = merge_by_id(docs + sample_articles())
        print("[INFO] Added bundled AI/ML corpus to downloaded Wikipedia documents.")
    write_jsonl(cfg["data"]["wikipedia_jsonl"], docs)
    print(f"Saved {len(docs)} documents to {cfg['data']['wikipedia_jsonl']}")


if __name__ == "__main__":
    main()
