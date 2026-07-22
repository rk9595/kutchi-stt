#!/usr/bin/env python3
"""
Kutchi Phase 1 — inference + WER/CER for a fine-tuned Whisper model.

Runs the model over verified rows and prints prediction vs reference plus
word- and character-error-rate. CER matters most for Indic scripts (word
boundaries are coarse). Point --model at a fine-tuned dir OR a base model
(e.g. openai/whisper-tiny) to measure the BEFORE baseline.

Usage:
    python eval.py --model ./model_out \
        --data dataset/verified.jsonl \
        --clips-dir data/clips --clips-dir data_smoke/clips --limit 20

Install:
    pip install torch transformers soundfile librosa
"""

import argparse
import json
import sys
from pathlib import Path


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def resolve_clip(clip, clips_dirs):
    for d in clips_dirs:
        p = Path(d) / clip
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="fine-tuned dir or base model id")
    ap.add_argument("--data", required=True, help="manifest jsonl with references")
    ap.add_argument("--clips-dir", action="append", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    try:
        import torch
        import librosa
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
    except ImportError:
        sys.exit("Install deps first:  pip install torch transformers soundfile librosa")

    processor = WhisperProcessor.from_pretrained(args.model, language="gu", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    rows = []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("status") != "verified" or not (r.get("transcript") or "").strip():
                continue
            p = resolve_clip(r["clip"], args.clips_dir)
            if p:
                rows.append((str(p), r["transcript"].strip()))
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        sys.exit("[!] no verified rows with resolvable clips found.")

    w_err = w_tot = c_err = c_tot = 0
    for path, ref in rows:
        audio, _ = librosa.load(path, sr=16000, mono=True)
        feats = processor.feature_extractor(audio, sampling_rate=16000,
                                            return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            ids = model.generate(feats, language="gu", task="transcribe")
        hyp = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

        rw, hw = ref.split(), hyp.split()
        w_err += edit_distance(rw, hw); w_tot += len(rw)
        c_err += edit_distance(ref, hyp); c_tot += len(ref)
        print(f"  ref: {ref}")
        print(f"  hyp: {hyp}\n")

    wer = 100 * w_err / max(w_tot, 1)
    cer = 100 * c_err / max(c_tot, 1)
    print(f"[{len(rows)} clips]  WER {wer:.1f}%   CER {cer:.1f}%")


if __name__ == "__main__":
    main()
