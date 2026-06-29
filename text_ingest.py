#!/usr/bin/env python3
"""
Kutchi TEXT-corpus collector (separate track from the STT audio pipeline).

Monolingual Kutchi text does NOT train the STT model (that needs audio+text).
This corpus feeds: (1) the spelling convention + glossary, (2) a future Kutchi
language model, (3) translator target-side fluency. Register matters — literary
poetry/proverbs are a different style from conversational speech, so we tag it.

Output: text/corpus.jsonl, one JSON object per post. Gujarati-script text is the
deliverable (matches our script-first convention); Devanagari kept for reference.
Everything lands as status="unverified" — a human confirms it's clean Kutchi and
convention-compatible before it counts, exactly like the audio correction loop.

Rights: store provenance, keep the corpus private. Use for research/glossary;
don't redistribute scraped text. See CLAUDE.md guardrails.

Usage:
    python text_ingest.py --sources text_sources.txt --out ./text --max-pages 3

    # text_sources.txt: one entry per line "URL[\\tregister]". Lines starting # ignored.
    # URL may be a category page (posts auto-discovered) or a single post.
    # register defaults to "unknown" (e.g. poetry / proverb / quote / prose).
"""

import argparse
import hashlib
import json
import sys
import time
import unicodedata
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (Kutchi research dataset collection)"}
GUJARATI = (0x0A80, 0x0AFF)
DEVANAGARI = (0x0900, 0x097F)
LATIN = (0x0041, 0x007A)


DEVA_TO_GUJ_OFFSET = 0x180  # Devanagari cp + 0x180 == Gujarati cp (abugidas align 1:1)


def dev_to_guj(text):
    """Transliterate Devanagari -> Gujarati. NFD first so precomposed nukta
    letters (क़ etc.) decompose to base+nukta and map cleanly. Only shift a
    codepoint when a real Gujarati char exists at the target; leave script-
    neutral danda (।॥) and anything without an equivalent untouched."""
    out = []
    for ch in unicodedata.normalize("NFD", text):
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F and cp not in (0x0964, 0x0965):
            tgt = chr(cp + DEVA_TO_GUJ_OFFSET)
            try:
                unicodedata.name(tgt)       # raises if target is unassigned
                out.append(tgt)
            except ValueError:
                out.append(ch)              # no Gujarati counterpart — keep original
        else:
            out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def _count(line, lo, hi):
    return sum(1 for ch in line if lo <= ord(ch) <= hi)


def line_script(line):
    """Dominant script of a line, or None if it's just punctuation/emoji/digits."""
    g, d, l = _count(line, *GUJARATI), _count(line, *DEVANAGARI), _count(line, *LATIN)
    if g == d == l == 0:
        return None
    return max((("gujarati", g), ("devanagari", d), ("latin", l)), key=lambda x: x[1])[0]


def fetch(url):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "lxml")


def discover_posts(soup):
    return [a["href"] for a in soup.select("h2.entry-title a") if a.get("href")]


def extract_post(soup, url):
    title_el = soup.select_one("h1.entry-title, h1")
    title = title_el.get_text(strip=True) if title_el else ""
    body = soup.select_one("div.entry-content")
    if not body:
        return None
    for x in body.select("script,style,ins,iframe,.code-block,.sharedaddy,"
                          ".addtoany_share_save_container,.wp-block-buttons,nav"):
        x.decompose()
    by_script = {"gujarati": [], "devanagari": [], "latin": []}
    for raw in body.get_text("\n", strip=True).splitlines():
        line = raw.strip()
        s = line_script(line)
        if s in by_script and len(line) > 1:
            by_script[s].append(line)
    devanagari = "\n".join(by_script["devanagari"]).strip()
    native_guj = "\n".join(by_script["gujarati"]).strip()
    if native_guj:
        text, origin, n = native_guj, "native_gujarati", len(by_script["gujarati"])
    elif devanagari:
        text, origin, n = dev_to_guj(devanagari), "transliterated_from_devanagari", len(by_script["devanagari"])
    else:
        return None
    pid = hashlib.sha1(url.encode()).hexdigest()[:12]
    return {
        "id": pid,
        "source_url": url,
        "source_site": urlparse(url).netloc,
        "title": title,
        "register": None,                       # filled by caller
        "scripts": [k for k, v in by_script.items() if v],
        "text": text,                           # Gujarati script — the deliverable
        "text_origin": origin,                  # native_gujarati | transliterated_from_devanagari
        "text_devanagari": devanagari,          # original, kept for audit/reversibility
        "n_lines": n,
        "status": "unverified",                 # human confirms clean Kutchi + convention
        "collected_at": str(date.today()),
        "notes": "",
    }


def load_done(path):
    done = set()
    if Path(path).exists():
        for line in open(path, encoding="utf-8"):
            try:
                done.add(json.loads(line)["source_url"])
            except Exception:
                pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", required=True, help="text file: URL[\\tregister] per line")
    ap.add_argument("--out", default="./text", help="output root")
    ap.add_argument("--max-pages", type=int, default=3, help="category pagination depth")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests (be polite)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    corpus = out / "corpus.jsonl"
    done = load_done(corpus)

    entries = []
    for line in Path(args.sources).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        entries.append((parts[0].strip(), parts[1].strip() if len(parts) > 1 else "unknown"))

    n = 0
    with open(corpus, "a", encoding="utf-8") as f:
        for url, register in entries:
            try:
                post_urls = []
                if "/category/" in url:
                    for page in range(1, args.max_pages + 1):
                        purl = url if page == 1 else url.rstrip("/") + f"/page/{page}/"
                        try:
                            post_urls += discover_posts(fetch(purl))
                        except requests.HTTPError:
                            break
                        time.sleep(args.delay)
                    post_urls = list(dict.fromkeys(post_urls))
                    print(f"[i] {url} -> {len(post_urls)} posts", file=sys.stderr)
                else:
                    post_urls = [url]

                for purl in post_urls:
                    if purl in done:
                        continue
                    try:
                        rec = extract_post(fetch(purl), purl)
                    except Exception as e:
                        print(f"[!] {purl}: {e}", file=sys.stderr)
                        continue
                    time.sleep(args.delay)
                    if not rec:
                        print(f"[skip] no gujarati-script text: {purl}", file=sys.stderr)
                        continue
                    rec["register"] = register
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    done.add(purl)
                    n += 1
                    print(f"[+] {rec['n_lines']} lines [{register}] {rec['title'][:50]}", file=sys.stderr)
            except Exception as e:
                print(f"[!] source {url}: {e}", file=sys.stderr)

    print(f"[done] wrote {n} new posts to {corpus}", file=sys.stderr)


if __name__ == "__main__":
    main()
