#!/usr/bin/env python3
"""
Kutchi Phase 1 — the spelling convention as code.

spelling_convention.md is the human-readable law; data/lexicon.tsv is its
machine-readable logged-decisions table. With no human annotators in the loop,
every pipeline that writes a `transcript` field MUST pass it through
normalize() so the machine makes the *same* choice every time (prime rule:
consistency beats correctness).

normalize() enforces, in order:
  1. Devanagari -> Gujarati transliteration (one-script rule, D7; also catches
     the Devanagari-keyboard / Hindi-ASR leakage failure mode)
  2. nukta stripped (Implosive Option A: plain letters, NO diacritics)
  3. candrabindu -> anusvara (D4 default)
  4. danda/double-danda -> '.', punctuation restricted to sentence-final . and ? (D8)
  5. lexicon guj_variants folded into the canonical guj spelling
  6. flags (returned, not fixed): Latin remnants, digits (D8 says words),
     leftover Devanagari

Import:  from lexicon import Lexicon;  lex = Lexicon.load()
         text, flags = lex.normalize(raw)
         lex.roman_to_guj("tonjo")  ->  "તોંજો" | None
CLI:     python lexicon.py --check data/cards/phrasebook.jsonl --field kutchi_guj
"""

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

LEXICON_TSV = Path(__file__).parent / "data" / "lexicon.tsv"

GUJARATI = range(0x0A80, 0x0B00)
DEVANAGARI = range(0x0900, 0x0980)
DEV_TO_GUJ_OFFSET = 0x0A80 - 0x0900
NUKTA = "઼"
CANDRABINDU = "ઁ"
ANUSVARA = "ં"
DEV_SPECIALS = {"।": ".", "॥": ".", "ऽ": "", "॰": ""}


def _dev_to_guj(ch):
    cp = ord(ch)
    if cp not in DEVANAGARI:
        return ch
    if ch in DEV_SPECIALS:
        return DEV_SPECIALS[ch]
    tgt = chr(cp + DEV_TO_GUJ_OFFSET)
    return tgt if unicodedata.name(tgt, None) else ch


class Lexicon:
    def __init__(self, roman_map, variant_map):
        self.roman_map = roman_map        # roman variant -> canonical guj
        self.variant_map = variant_map    # guj misspelling -> canonical guj

    @classmethod
    def load(cls, path=LEXICON_TSV):
        roman_map, variant_map = {}, {}
        header_seen = False
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("##"):
                continue
            if not header_seen:
                header_seen = True  # column header row
                continue
            cols = line.split("\t")
            if len(cols) < 2:
                continue
            roman, guj = cols[0].strip(), cols[1].strip()
            guj_variants = cols[2].strip() if len(cols) > 2 else ""
            for r in filter(None, (x.strip().lower() for x in roman.split("|"))):
                roman_map[r] = guj
            for v in filter(None, (x.strip() for x in guj_variants.split("|"))):
                variant_map[v] = guj
        return cls(roman_map, variant_map)

    def roman_to_guj(self, word):
        return self.roman_map.get(word.strip().lower())

    def normalize(self, text):
        flags = []
        t = unicodedata.normalize("NFC", text or "")

        if any(ord(c) in DEVANAGARI for c in t):
            flags.append("devanagari-transliterated")
            t = "".join(_dev_to_guj(c) for c in t)

        if NUKTA in t:
            flags.append("nukta-stripped")
            t = t.replace(NUKTA, "")

        if CANDRABINDU in t:
            flags.append("candrabindu->anusvara")
            t = t.replace(CANDRABINDU, ANUSVARA)

        if "/" in t:
            flags.append("slash-alternates")

        # D8: only sentence punctuation . and ? survive — but the uncertain-word
        # marker "(?)" is itself part of D8 and must be preserved.
        t = t.replace("(?)", "\x00")
        t = re.sub(r"[!;:,‘’“”\"'()\[\]{}<>|/\\*_~`–—-]", " ", t)
        t = t.replace("\x00", "(?)")

        out_words = []
        for w in t.split():
            bare = w.rstrip(".?")
            tail = w[len(bare):]
            canonical = self.variant_map.get(bare)
            out_words.append(canonical + tail if canonical else w)
        t = " ".join(out_words)

        if re.search(r"[A-Za-z]", t):
            flags.append("latin-remnant")
        if re.search(r"[0-9૦-૯]", t):
            flags.append("digits")
        if any(ord(c) in DEVANAGARI for c in t):
            flags.append("devanagari-remnant")
        return t.strip(), flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", required=True, help="jsonl file to audit")
    ap.add_argument("--field", default="transcript")
    args = ap.parse_args()

    lex = Lexicon.load()
    n, changed, flagged = 0, 0, 0
    for line in Path(args.check).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        raw = row.get(args.field) or ""
        if not raw:
            continue
        n += 1
        norm, flags = lex.normalize(raw)
        if norm != raw or flags:
            tag = ",".join(flags) or "respelled"
            ident = row.get("id", "?")
            print(f"[{tag}] id={ident}\n  in : {raw}\n  out: {norm}")
            changed += norm != raw
            flagged += bool(flags)
    print(f"\n{n} rows checked: {changed} would change, {flagged} flagged", file=sys.stderr)


if __name__ == "__main__":
    main()
