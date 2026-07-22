#!/usr/bin/env python3
"""
Kutchi Phase 1 — self-recording (Option A read-speech).

Inverts the ingest pipeline: instead of audio -> ASR draft -> human correction,
you READ a known Kutchi prompt aloud. The transcript is the prompt, so rows are
written already `status: "verified"` with ZERO correction cost. Single speaker +
clean mic = the fastest path to a working end-to-end MVP.

This is a personal MVP dataset, not the multi-speaker moat. Rows are tagged with
provenance (source_id = selfrec-<speaker>) so they can be filtered/deduped later.

Usage:
    python record.py --prompts prompts.txt --out ./data_self --speaker rakesh

    prompts.txt format:  <Gujarati-script transcript><TAB><optional roman>
    (lines starting with # ignored)

Per clip:  Enter=start recording -> Enter=stop -> [k]eep / [r]edo / [p]lay / [s]kip.
Resume-safe: re-running skips prompts already recorded (matched by id).

Install:
    pip install sounddevice soundfile numpy
    # macOS: grant the terminal Microphone permission in System Settings > Privacy.
"""

import argparse
import hashlib
import json
import re
import sys
import threading
from pathlib import Path

TARGET_SR = 16000  # 16 kHz mono — matches ingest.py / ASR standard


def slug(text, n=40):
    text = re.sub(r"[^a-zA-Z0-9\s-]", "", text or "").strip().lower()
    return re.sub(r"[\s_-]+", "-", text)[:n] or "src"


def prompt_id(speaker, transcript):
    # deterministic from (speaker, text) so re-runs skip already-recorded prompts
    h = hashlib.sha1(f"{speaker}\n{transcript}".encode("utf-8")).hexdigest()[:8]
    return f"selfrec-{speaker}-{h}"


def load_prompts(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        transcript, _, roman = line.partition("\t")
        rows.append((transcript.strip(), roman.strip()))
    return rows


def load_done_ids(manifest_path):
    done = set()
    if Path(manifest_path).exists():
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
    return done


def record_until_enter(sd, np, sr):
    frames = []

    def cb(indata, n, t, status):
        frames.append(indata.copy())

    with sd.InputStream(samplerate=sr, channels=1, dtype="float32", callback=cb):
        input()  # blocks until Enter
    if not frames:
        return np.zeros((0, 1), dtype="float32")
    return np.concatenate(frames, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, help="prompts file (transcript<TAB>roman)")
    ap.add_argument("--out", default="./data_self", help="output root")
    ap.add_argument("--speaker", required=True, help="speaker id, e.g. rakesh")
    ap.add_argument("--sr", type=int, default=TARGET_SR, help="sample rate (Hz)")
    args = ap.parse_args()

    try:
        import numpy as np
        import sounddevice as sd
        import soundfile as sf
    except ImportError:
        sys.exit("Install deps first:  pip install sounddevice soundfile numpy")

    out = Path(args.out)
    clips_dir = out / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.jsonl"

    prompts = load_prompts(args.prompts)
    done = load_done_ids(manifest_path)
    todo = [(t, r) for (t, r) in prompts if prompt_id(args.speaker, t) not in done]

    print(f"[i] {len(prompts)} prompts, {len(done)} already recorded, {len(todo)} to go.")
    print("[i] Controls: Enter=start, Enter=stop, then k=keep r=redo p=play s=skip q=quit\n")

    n_new = 0
    with open(manifest_path, "a", encoding="utf-8") as mf:
        for i, (transcript, roman) in enumerate(todo, 1):
            cid = prompt_id(args.speaker, transcript)
            print(f"[{i}/{len(todo)}] Read aloud:")
            print(f"    {transcript}")
            if roman:
                print(f"    (roman: {roman})")

            audio = None
            while True:
                cmd = input("  Enter=start  (s=skip, q=quit) > ").strip().lower()
                if cmd == "q":
                    print(f"\n[done] wrote {n_new} new clips to {clips_dir}")
                    return
                if cmd == "s":
                    audio = None
                    break
                print("  ● recording… press Enter to stop")
                audio = record_until_enter(sd, np, args.sr)
                dur = len(audio) / args.sr
                if dur < 0.4:
                    print(f"  [!] only {dur:.2f}s captured — likely a misfire, redo.")
                    continue
                print(f"  captured {dur:.1f}s")
                act = input("  k=keep  r=redo  p=play  s=skip > ").strip().lower()
                while act == "p":
                    sd.play(audio, args.sr); sd.wait()
                    act = input("  k=keep  r=redo  p=play  s=skip > ").strip().lower()
                if act == "r":
                    continue
                if act == "s":
                    audio = None
                break

            if audio is None:
                print("  [skip]\n")
                continue

            dur = len(audio) / args.sr
            clip_name = f"{cid}.wav"
            sf.write(clips_dir / clip_name, audio, args.sr, subtype="PCM_16")
            row = {
                "id": cid,
                "clip": clip_name,
                "source_id": f"selfrec-{args.speaker}",
                "source_title": f"self-recorded read-speech ({args.speaker})",
                "source_url": "",
                "start": 0.0,
                "end": round(dur, 3),
                "duration": round(dur, 3),
                "asr_draft": "",
                "asr_model": "",
                "transcript": transcript,   # known by construction = the prompt
                "roman": roman,
                "status": "verified",
                "notes": "self-recorded",
            }
            mf.write(json.dumps(row, ensure_ascii=False) + "\n")
            mf.flush()
            n_new += 1
            print(f"  [saved] {clip_name}\n")

    total = len(done) + n_new
    print(f"[done] wrote {n_new} new clips to {clips_dir}")
    print(f"[done] manifest: {manifest_path}  ({total} verified rows total)")
    mins = sum(json.loads(l).get('duration', 0) for l in open(manifest_path, encoding='utf-8')) / 60
    print(f"[done] ~{mins:.1f} min of verified self-recorded audio")


if __name__ == "__main__":
    main()
