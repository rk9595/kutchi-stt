#!/usr/bin/env python3
"""
Kutchi Phase 1 — merge every verified source into dataset/verified.jsonl.

Combines per-track manifests (human-corrected exports, Track-A card alignments,
Track-B machine-consensus rows, self-recordings) into the ONE canonical training
manifest. Rules:

  - only status == "verified" with a non-empty transcript
  - provenance is mandatory: rows without source_id + source_url are REJECTED
    (CLAUDE.md guardrail — needed for dedup and rights)
  - verified_by tiers are preserved, never conflated. Legacy rows without the
    field get "human" (everything before the automation pivot was human-checked).
    On duplicates, the higher tier wins: human > machine-card > machine-consensus.
  - dedup on id, then on (source_id, start, end) to ~0.1 s
  - transcripts are NOT rewritten (prime rule) — rows whose transcript would
    change under lexicon.normalize() are only WARNED about, so convention drift
    is visible without silently editing verified text.

Usage:
    python merge.py                       # default inputs, writes dataset/verified.jsonl
    python merge.py --data extra.jsonl --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

from lexicon import Lexicon

DEFAULT_INPUTS = [
    "dataset/verified.jsonl",             # existing canonical (legacy human tier)
    "data/manifest.for_upload.jsonl",     # human-corrected desk rows
    "data/cards/aligned/manifest.jsonl",  # Track A: card-aligned
    "data/machine_verified.jsonl",        # Track B: consensus
    "data_self/manifest.jsonl",           # self-recordings
]
TIER_RANK = {"human": 0, "machine-card": 1, "machine-consensus": 2}


def rows_from(path):
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)


def tier(row):
    return row.get("verified_by") or "human"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="append", default=None,
                    help="input jsonl (repeatable). default: known track outputs")
    ap.add_argument("--out", default="dataset/verified.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    lex = Lexicon.load()
    inputs = args.data or DEFAULT_INPUTS

    by_id, by_span = {}, {}
    rejected, drift = [], 0
    for path in inputs:
        n_in = 0
        for r in rows_from(path):
            if r.get("status") != "verified" or not (r.get("transcript") or "").strip():
                continue
            n_in += 1
            src = r.get("source_id") or ""
            # self-recordings have no URL; the speaker is the provenance
            if not src or (not r.get("source_url") and not src.startswith("selfrec")):
                rejected.append((path, r.get("id", "?"), "missing-provenance"))
                continue
            r.setdefault("verified_by", "human")
            if tier(r) not in TIER_RANK:
                rejected.append((path, r.get("id", "?"), f"unknown-tier:{tier(r)}"))
                continue

            norm, flags = lex.normalize(r["transcript"])
            if norm != r["transcript"] or flags:
                drift += 1
                print(f"[drift] {r.get('id','?')} ({','.join(flags) or 'respell'}): "
                      f"{r['transcript']!r} -> {norm!r}", file=sys.stderr)

            span = (r["source_id"], round(r.get("start") or 0, 1), round(r.get("end") or 0, 1))
            for key, table in ((r.get("id"), by_id), (span, by_span)):
                old = table.get(key)
                if old is not None and TIER_RANK[tier(old)] <= TIER_RANK[tier(r)]:
                    break
            else:
                by_id[r["id"]] = r
                by_span[span] = r
        if n_in:
            print(f"[in] {path}: {n_in} verified rows", file=sys.stderr)

    # by_id and by_span may disagree after upgrades; by_id is authoritative,
    # then drop id-distinct rows that lost their span slot (span duplicates)
    final = [r for r in by_id.values() if by_span.get(
        (r["source_id"], round(r.get("start") or 0, 1), round(r.get("end") or 0, 1))) is r]

    final.sort(key=lambda r: (r["source_id"], r.get("start") or 0))
    total_min = sum(r.get("duration") or 0 for r in final) / 60
    per = {}
    for r in final:
        key = (r["source_id"], tier(r))
        per[key] = per.get(key, 0) + (r.get("duration") or 0)

    print(f"\n[report] {len(final)} clips, {total_min:.1f} verified min "
          f"({drift} drift warnings, {len(rejected)} rejected)")
    for (src, t), sec in sorted(per.items()):
        print(f"  {src:45s} {t:18s} {sec/60:6.1f} min")
    for path, rid, why in rejected[:10]:
        print(f"  [rejected] {rid} from {path}: {why}", file=sys.stderr)

    if args.dry_run:
        print("[dry-run] not writing", file=sys.stderr)
        return
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
