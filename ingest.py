#!/usr/bin/env python3
"""
Kutchi Phase 1 — audio ingestion pipeline.

Turns a list of YouTube (or any yt-dlp-supported) URLs into:
  - 16 kHz mono WAV clips, segmented to ASR-friendly lengths
  - a manifest.jsonl with a rough Gujarati-script ASR draft per clip

The drafts are NOT correct Kutchi. Whisper has no Kutchi; we transcribe as
Gujarati ("gu") only to give a human a starting point to CORRECT, which is
~3-5x faster than transcribing from a blank box. The verified transcripts are
produced by humans in transcribe_correct.html, never by this script.

Usage:
    python ingest.py --urls urls.txt --out ./data --model small

    # urls.txt = one URL per line (videos or playlists). Lines starting with # ignored.

Install (run on YOUR machine with a GPU if possible; CPU works but is slow):
    pip install yt-dlp faster-whisper
    # plus ffmpeg on PATH:  sudo apt install ffmpeg   (or: brew install ffmpeg)

Resume-safe: re-running skips sources already in manifest.jsonl.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path

# ---- tunables -------------------------------------------------------------
MIN_CLIP_SEC = 2.0      # drop clips shorter than this (too little signal)
MAX_CLIP_SEC = 20.0     # split/skip clips longer than this (ASR training likes < ~20s)
TARGET_SR = 16000       # 16 kHz mono is the standard for ASR
# ---------------------------------------------------------------------------


def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def slug(text, n=40):
    # ASCII-only: clip ids become object-storage keys / URLs for remote
    # annotators, so keep them portable. uuid suffix guarantees uniqueness;
    # full title is preserved in the manifest's source_title.
    text = re.sub(r"[^a-zA-Z0-9\s-]", "", text or "").strip().lower()
    return re.sub(r"[\s_-]+", "-", text)[:n] or "src"


def download_audio(url, raw_dir):
    """Download best audio to a 16 kHz mono WAV. Returns (wav_path, title, video_id)."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    # Ask yt-dlp for metadata + extract audio to wav, resampled to 16k mono.
    out_tmpl = str(raw_dir / "%(id)s.%(ext)s")
    cmd = [
        "yt-dlp", "-x", "--audio-format", "wav",
        "--postprocessor-args", f"-ac 1 -ar {TARGET_SR}",
        "--print", "after_move:%(id)s\t%(title)s",
        "--no-playlist" if "list=" not in url else "--yes-playlist",
        "-o", out_tmpl, url,
    ]
    res = run(cmd)
    results = []
    for line in res.stdout.strip().splitlines():
        if "\t" not in line:
            continue
        vid, title = line.split("\t", 1)
        wav = raw_dir / f"{vid}.wav"
        if wav.exists():
            results.append((wav, title, vid))
    return results


def transcribe_segments(wav_path, model):
    """Yield (start, end, draft_text) using faster-whisper as Gujarati."""
    segments, _info = model.transcribe(
        str(wav_path),
        language="gu",          # closest high-resource script Whisper knows well
        vad_filter=True,        # drops long silences / music gaps
        vad_parameters=dict(min_silence_duration_ms=500),
        beam_size=5,
    )
    for s in segments:
        yield s.start, s.end, (s.text or "").strip()


def cut_clip(src_wav, start, end, dst_wav):
    run([
        "ffmpeg", "-y", "-i", str(src_wav),
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
        "-ac", "1", "-ar", str(TARGET_SR), str(dst_wav),
    ])


def load_done_sources(manifest_path):
    done = set()
    if Path(manifest_path).exists():
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["source_id"])
                except Exception:
                    pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", required=True, help="text file, one URL per line")
    ap.add_argument("--out", default="./data", help="output root")
    ap.add_argument("--model", default="small",
                    help="faster-whisper size: tiny/base/small/medium/large-v3")
    ap.add_argument("--device", default="auto", help="cuda / cpu / auto")
    args = ap.parse_args()

    out = Path(args.out)
    clips_dir = out / "clips"
    raw_dir = out / "raw"
    clips_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.jsonl"

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("Install deps first:  pip install yt-dlp faster-whisper  (and ffmpeg)")

    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    compute = "float16" if device == "cuda" else "int8"
    print(f"[i] loading whisper '{args.model}' on {device} ({compute})", file=sys.stderr)
    model = WhisperModel(args.model, device=device, compute_type=compute)

    urls = [l.strip() for l in Path(args.urls).read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.strip().startswith("#")]
    done = load_done_sources(manifest_path)

    n_clips = 0
    with open(manifest_path, "a", encoding="utf-8") as mf:
        for url in urls:
            try:
                for wav, title, vid in download_audio(url, raw_dir):
                    if vid in done:
                        print(f"[skip] already done: {title[:60]}", file=sys.stderr)
                        continue
                    print(f"[+] transcribing: {title[:60]}", file=sys.stderr)
                    src_slug = slug(title)
                    for start, end, draft in transcribe_segments(wav, model):
                        dur = end - start
                        if dur < MIN_CLIP_SEC or dur > MAX_CLIP_SEC:
                            continue
                        cid = f"{src_slug}-{uuid.uuid4().hex[:8]}"
                        clip_name = f"{cid}.wav"
                        cut_clip(wav, start, end, clips_dir / clip_name)
                        row = {
                            "id": cid,
                            "clip": clip_name,                 # relative to clips/
                            "source_id": vid,
                            "source_title": title,
                            "source_url": url,
                            "start": round(start, 3),
                            "end": round(end, 3),
                            "duration": round(dur, 3),
                            "asr_draft": draft,                # Gujarati-script guess; TO BE CORRECTED
                            "asr_model": f"faster-whisper-{args.model}/gu",
                            "transcript": "",                  # filled by human
                            "roman": "",                       # optional romanized, filled by human
                            "status": "pending",               # pending|verified|skip|nonspeech
                            "notes": "",
                        }
                        mf.write(json.dumps(row, ensure_ascii=False) + "\n")
                        mf.flush()
                        n_clips += 1
                    done.add(vid)
            except subprocess.CalledProcessError as e:
                print(f"[!] failed on {url}: {e.stderr[:200]}", file=sys.stderr)
            except Exception as e:
                print(f"[!] error on {url}: {e}", file=sys.stderr)

    print(f"[done] wrote {n_clips} new clips to {clips_dir}", file=sys.stderr)
    print(f"[done] manifest: {manifest_path}", file=sys.stderr)
    print("Next: open transcribe_correct.html, load manifest.jsonl + the clips/ folder.",
          file=sys.stderr)


if __name__ == "__main__":
    main()
