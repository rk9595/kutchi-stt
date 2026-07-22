#!/usr/bin/env python3
"""
Kutchi Phase 1 — freeze eval, emit train/eval splits.

The eval set is FROZEN: once a clip id is written to dataset/eval_ids.txt it stays
eval forever. Re-run this as verified data grows — existing eval assignments are
never moved, only new data is added. finetune_whisper.py refuses to train on any
id in eval_ids.txt (see CLAUDE.md guardrail).

Two split modes (auto-picked):
  - source: hold out WHOLE sources for eval (no speaker/recording leakage). Used
    when >= 2 sources exist. Eval grows one whole source at a time toward the
    --eval-minutes target. The lock is dataset/eval_sources.txt.
  - clip:   deterministic per-clip holdout. Used when there's only ONE source
    (e.g. your self-recorded MVP: same voice, held-out utterances). Speaker
    overlaps train by design — fine for a personal MVP, NOT for the real model.

Outputs (default in dataset/):
    eval_ids.txt       locked clip ids  <- training MUST exclude these
    eval_sources.txt   locked source ids (source mode only)
    eval.jsonl         frozen eval rows
    train.jsonl        everything else

Usage:
    python split.py --data dataset/verified.jsonl --eval-minutes 90
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def load_verified(paths):
    rows = []
    for dp in paths:
        with open(dp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("status") == "verified" and (r.get("transcript") or "").strip():
                    rows.append(r)
    return rows


def read_lock(path):
    if Path(path).exists():
        return [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]
    return []


def minutes(rows):
    return sum(r.get("duration", 0) for r in rows) / 60


def stable_rank(key, salt):
    # deterministic pseudo-random order, stable across runs and data growth
    return hashlib.sha1(f"{salt}:{key}".encode()).hexdigest()


def split_by_source(rows, out, target_min, salt):
    lock_path = out / "eval_sources.txt"
    locked = set(read_lock(lock_path))

    by_src = {}
    for r in rows:
        by_src.setdefault(r["source_id"], []).append(r)

    eval_min = minutes([r for r in rows if r["source_id"] in locked])
    # add whole unlocked sources (deterministic order) until we hit the target
    pool = sorted((s for s in by_src if s not in locked), key=lambda s: stable_rank(s, salt))
    for s in pool:
        if eval_min >= target_min:
            break
        locked.add(s)
        eval_min += minutes(by_src[s])

    if not locked:
        sys.exit("[!] source mode selected but no sources locked — check --eval-minutes.")

    lock_path.write_text("\n".join(sorted(locked)) + "\n")
    ev = [r for r in rows if r["source_id"] in locked]
    tr = [r for r in rows if r["source_id"] not in locked]
    return ev, tr, f"{len(locked)} source(s) held out"


def split_by_clip(rows, out, target_min, frac, salt):
    # single-source fallback: freeze clips whose stable hash falls in the eval bucket
    ranked = sorted(rows, key=lambda r: stable_rank(r["id"], salt))
    total = minutes(rows)
    want = min(target_min, total * frac)
    ev, acc = [], 0.0
    for r in ranked:
        if acc >= want:
            break
        ev.append(r); acc += r.get("duration", 0) / 60
    ev_ids = {r["id"] for r in ev}
    tr = [r for r in rows if r["id"] not in ev_ids]
    return ev, tr, "single source — clip-level holdout (speaker overlaps train)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="append", default=None,
                    help="verified jsonl (repeatable). default: dataset/verified.jsonl")
    ap.add_argument("--out-dir", default="dataset")
    ap.add_argument("--eval-minutes", type=float, default=90.0,
                    help="target FROZEN eval size (CLAUDE.md target: 60-120 min)")
    ap.add_argument("--eval-frac", type=float, default=0.15,
                    help="clip-mode cap: don't put more than this fraction in eval")
    ap.add_argument("--mode", choices=["auto", "source", "clip"], default="auto")
    ap.add_argument("--seed", default="kutchi-eval-v1", help="deterministic split salt")
    args = ap.parse_args()

    data = args.data or ["dataset/verified.jsonl"]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = load_verified(data)
    if not rows:
        sys.exit("[!] no verified rows found.")
    n_src = len({r["source_id"] for r in rows})

    mode = args.mode
    if mode == "auto":
        mode = "clip" if n_src < 2 else "source"

    if mode == "source":
        ev, tr, note = split_by_source(rows, out, args.eval_minutes, args.seed)
    else:
        if n_src >= 2:
            print("[!] clip mode with multiple sources: eval WILL leak speakers into train.")
        ev, tr, note = split_by_clip(rows, out, args.eval_minutes, args.eval_frac, args.seed)

    if not tr:
        sys.exit("[!] train split is empty — lower --eval-minutes or add more sources.")

    ev_ids = {r["id"] for r in ev}
    leak = ev_ids & {r["id"] for r in tr}
    assert not leak, f"eval/train id overlap: {leak}"

    (out / "eval_ids.txt").write_text("\n".join(sorted(ev_ids)) + "\n")
    with open(out / "eval.jsonl", "w", encoding="utf-8") as f:
        for r in ev:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "train.jsonl", "w", encoding="utf-8") as f:
        for r in tr:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[mode] {mode} — {note}")
    print(f"[eval ] {len(ev):4d} clips  {minutes(ev):6.1f} min  (FROZEN -> eval_ids.txt)")
    print(f"[train] {len(tr):4d} clips  {minutes(tr):6.1f} min")
    if minutes(ev) < args.eval_minutes:
        print(f"[note ] eval is {minutes(ev):.1f} min, below the {args.eval_minutes:.0f} min "
              f"target — it will grow (and stay frozen) as you add verified data.")


if __name__ == "__main__":
    main()
