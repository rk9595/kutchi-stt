#!/usr/bin/env python3
"""
Stage 0 of the Kutchi-GPT curriculum — assemble the text the models train on.

Kutchi has almost no text (that is the whole project's premise), so we gather every
Kutchi sentence we have into one file, and separately fetch a large *Gujarati* corpus
to pretrain on (Gujarati is written in the same script and is linguistically close;
the GPT-2-scale model learns language on Gujarati, then specializes on Kutchi).

  data_lm/kutchi.txt    <- verified STT transcripts + phrasebook + poetry/proverbs,
                           each run through lexicon.normalize() so the LM obeys the
                           same spelling convention as the STT data. ~27 KB today;
                           grows automatically as the STT pipeline verifies more audio.
  data_lm/gujarati.txt  <- Gujarati Wikipedia (via HF `datasets`), size-capped for the Mac.

Usage:
    python lm/build_corpus.py                          # (re)build kutchi.txt
    python lm/build_corpus.py --fetch-gujarati --max-mb 200
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lexicon import Lexicon  # noqa: E402

OUT = ROOT / "data_lm"
# (path, text field, require status=="verified"?) — the verified gate applies only to
# STT output; the text-corpus track is its own thing (status "unverified" is expected).
KUTCHI_SOURCES = [
    ("dataset/verified.jsonl", "transcript", True),    # STT deliverables
    ("data/cards/phrasebook.jsonl", "kutchi_guj", False),  # lesson cards ("/"-variants)
    ("text/corpus.jsonl", "text", False),              # poetry / proverbs / prose
]


def iter_texts(path, field, require_verified):
    p = ROOT / path
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if require_verified and row.get("status") != "verified":
            continue
        val = row.get(field)
        if val:
            yield val


def build_kutchi(lex):
    OUT.mkdir(parents=True, exist_ok=True)
    lines, seen = [], set()
    for path, field, require_verified in KUTCHI_SOURCES:
        n = 0
        for raw in iter_texts(path, field, require_verified):
            # poetry rows are multi-line; phrasebook packs register variants as "a / b"
            for chunk in raw.splitlines():
                for piece in chunk.split("/"):
                    norm, _ = lex.normalize(piece)
                    norm = norm.strip()
                    if len(norm) < 2 or norm in seen:
                        continue
                    seen.add(norm)
                    lines.append(norm)
                    n += 1
        print(f"  {path:32s} +{n} lines", file=sys.stderr)
    text = "\n".join(lines) + "\n"
    (OUT / "kutchi.txt").write_text(text, encoding="utf-8")
    chars = len(text)
    print(f"[kutchi] {len(lines)} lines, {chars} chars ({chars/1024:.1f} KB) "
          f"-> {OUT/'kutchi.txt'}", file=sys.stderr)


def fetch_gujarati(max_mb):
    # Gujarati Wikipedia: clean, no auth. IndicCorpV2 (ai4bharat) is the scale-up
    # source if you later want hundreds of MB to billions of tokens on a GPU.
    from datasets import load_dataset
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / "gujarati.txt"
    budget = max_mb * 1024 * 1024
    print(f"[gujarati] streaming wikimedia/wikipedia 20231101.gu, cap {max_mb} MB",
          file=sys.stderr)
    ds = load_dataset("wikimedia/wikipedia", "20231101.gu", split="train", streaming=True)
    written, docs = 0, 0
    with open(dst, "w", encoding="utf-8") as f:
        for row in ds:
            body = (row.get("text") or "").strip()
            if not body:
                continue
            chunk = body + "\n\n"
            f.write(chunk)
            written += len(chunk.encode("utf-8"))
            docs += 1
            if docs % 2000 == 0:
                print(f"    {docs} docs, {written/1024/1024:.1f} MB", file=sys.stderr)
            if written >= budget:
                break
    print(f"[gujarati] {docs} docs, {written/1024/1024:.1f} MB -> {dst}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch-gujarati", action="store_true")
    ap.add_argument("--max-mb", type=int, default=200)
    args = ap.parse_args()

    lex = Lexicon.load()
    build_kutchi(lex)
    if args.fetch_gujarati:
        fetch_gujarati(args.max_mb)


if __name__ == "__main__":
    main()
