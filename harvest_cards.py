#!/usr/bin/env python3
"""
Kutchi Phase 1 — on-screen CARD harvester for lesson-style videos.

Some sources (the Jagdish Vaghela "Learn Kutchi" playlist) put each phrase on a
full-screen slide: a hand-authored English / Hindi / romanized-Kutchi triple,
one card at a time, over a solid background colour. That text is a native-authored
translation seed we can read straight off the frames — far cheaper than blank-box
transcription.

This script automates the mechanical half of the recipe proven on Part-1:
  1. download the video at low res (yt-dlp)
  2. detect card boundaries by CHROMA shift on the cropped card region
     (each card = a distinct solid bg colour; robust to the animated text and to
      the word-by-word caption highlight that defeats scene-detect/mpdecimate),
     then grab ONE static, fully-rendered frame from the latter half of each window
  3. extract those frames + contact sheets to read
  4. anchor each card window to the audio clips already in manifest.jsonl

What it does NOT do (needs vision + the native lead, same as asr_draft):
  - read the triple text off the frame        -> filled in the reading pass
  - convert romanized Kutchi -> Gujarati script -> per spelling_convention.md,
    Option A; only the native lead ratifies implosive/vowel-length calls.
So cards.jsonl rows carry EMPTY text fields + needs_native_verify=true; the frame,
window timing and aligned clip ids are filled here.

Applies to card-style lesson videos only. Vlogs/comedy/katha have no on-screen
text and still go through the audio+verify pipeline (ingest.py).

Usage:
    python harvest_cards.py --playlist "https://www.youtube.com/playlist?list=..."
    python harvest_cards.py --ids 1oLsZIOGFE8 m5l8aEZzvDA ...
    python harvest_cards.py --urls urls.txt          # video/playlist lines

Output (under --out, default data/cards/, gitignored):
    <video_id>/<video_id>.mp4     low-res source (private, not redistributed)
    <video_id>/frames/c##.jpg     one frame per detected card
    <video_id>/sheet_##.jpg       contact sheets for reading
    <video_id>/cards.jsonl        one row per card (text fields blank)
    harvest_index.jsonl           per-video summary

Resume-safe: a video whose cards.jsonl already exists is skipped (use --force).
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# ---- tunables -------------------------------------------------------------
RES = 360             # download cap; 360p is plenty to read slide text
TOP_FRAC = 0.86       # card lives in the top; the bottom band is the caption ticker
CHROMA_THRESH = 6.0   # |dU|+|dV| between 1s samples that marks a new card bg colour
MIN_CARD_SEC = 6.0    # ignore transitions / blips shorter than a real card display
PICK_FRAC = 0.6       # sample this far into a window: static + fully rendered
SHEET_COLS, SHEET_ROWS = 2, 6
# ---------------------------------------------------------------------------


def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def resolve_ids(args):
    """Return an ordered, de-duped list of (video_id, title) from the given sources."""
    seen, out = set(), []

    def add(vid, title=""):
        if vid and vid not in seen:
            seen.add(vid)
            out.append((vid, title))

    sources = list(args.ids or [])
    if args.urls:
        for l in Path(args.urls).read_text(encoding="utf-8").splitlines():
            l = l.split("#", 1)[0].strip()
            if l:
                sources.append(l)
    if args.playlist:
        sources.append(args.playlist)

    for s in sources:
        if "list=" in s or "/playlist" in s or "/channel/" in s or "/@" in s:
            res = run(["yt-dlp", "--flat-playlist", "--print", "%(id)s\t%(title)s", s])
            for line in res.stdout.strip().splitlines():
                vid, _, title = line.partition("\t")
                add(vid.strip(), title.strip())
        elif "watch?v=" in s or "youtu.be/" in s:
            m = re.search(r"(?:v=|youtu\.be/)([\w-]{11})", s)
            if m:
                add(m.group(1))
        else:
            add(s.strip())  # bare id
    return out


def download_video(vid, vdir):
    """Download the video at <=RES. Returns (path, title). Resume-safe."""
    existing = next((p for p in vdir.glob(f"{vid}.*") if p.suffix != ".jsonl"), None)
    if existing and existing.stat().st_size > 0:
        return existing, ""
    out_tmpl = str(vdir / "%(id)s.%(ext)s")
    res = run([
        "yt-dlp", "-S", f"res:{RES}", "--merge-output-format", "mp4",
        "--no-playlist", "--print", "after_move:%(title)s",
        "-o", out_tmpl, f"https://www.youtube.com/watch?v={vid}",
    ])
    title = res.stdout.strip().splitlines()[-1] if res.stdout.strip() else ""
    path = next(p for p in vdir.glob(f"{vid}.*") if p.suffix != ".jsonl")
    return path, title


def probe_dims(path):
    res = run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(path),
    ])
    w, h = res.stdout.strip().split("x")[:2]
    return int(w), int(h)


def chroma_windows(video, vdir, top_h):
    """Sample 1 fps, read U/V averages of the card region, split into card windows."""
    stats = vdir / "stats.txt"
    run([
        "ffmpeg", "-y", "-i", str(video),
        "-vf", f"crop=iw:{top_h}:0:0,fps=1,signalstats,metadata=print:file={stats}",
        "-f", "null", "-",
    ])
    rows, t, u, v = [], None, None, None
    for line in stats.read_text().splitlines():
        m = re.search(r"pts_time:([\d.]+)", line)
        if m:
            if t is not None and u is not None and v is not None:
                rows.append((t, u, v))
            t, u, v = float(m.group(1)), None, None
        m = re.search(r"signalstats\.UAVG=([\d.]+)", line)
        if m:
            u = float(m.group(1))
        m = re.search(r"signalstats\.VAVG=([\d.]+)", line)
        if m:
            v = float(m.group(1))
    if t is not None and u is not None and v is not None:
        rows.append((t, u, v))
    if not rows:
        return []

    segs, start, pu, pv = [], rows[0][0], rows[0][1], rows[0][2]
    for i in range(1, len(rows)):
        ti, ui, vi = rows[i]
        if abs(ui - pu) + abs(vi - pv) > CHROMA_THRESH:
            segs.append((start, rows[i - 1][0]))
            start = ti
        pu, pv = ui, vi
    segs.append((start, rows[-1][0]))
    return [(s, e) for s, e in segs if (e - s) >= MIN_CARD_SEC]


def extract_frames(video, windows, frames_dir, top_h):
    frames_dir.mkdir(parents=True, exist_ok=True)
    for p in frames_dir.glob("*.jpg"):
        p.unlink()
    picks = []
    for i, (s, e) in enumerate(windows, 1):
        t = s + PICK_FRAC * (e - s)
        dst = frames_dir / f"c{i:02d}.jpg"
        run([
            "ffmpeg", "-y", "-ss", f"{t:.2f}", "-i", str(video),
            "-frames:v", "1", "-vf", f"crop=iw:{top_h}:0:0", "-qscale:v", "2", str(dst),
        ])
        picks.append((t, dst))
    return picks


def build_sheets(frames_dir, vdir):
    for p in vdir.glob("sheet_*.jpg"):
        p.unlink()
    run([
        "ffmpeg", "-y", "-pattern_type", "glob", "-i", str(frames_dir / "c*.jpg"),
        "-filter_complex",
        f"scale=600:-1,tile={SHEET_COLS}x{SHEET_ROWS}:margin=8:padding=6:color=gray",
        str(vdir / "sheet_%02d.jpg"),
    ])


def load_manifest(path):
    by_vid = {}
    if not Path(path).exists():
        return by_vid
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        by_vid.setdefault(r.get("source_id"), []).append(
            (r.get("id"), r.get("start"), r.get("end"))
        )
    return by_vid


def clips_in_window(clips, s, e):
    """Manifest clip ids whose [start,end] overlaps the card window [s,e]."""
    out = []
    for cid, cs, ce in clips:
        if cs is None or ce is None:
            continue
        if cs < e and ce > s:
            out.append(cid)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--playlist", help="playlist/channel URL")
    ap.add_argument("--ids", nargs="*", help="explicit video ids")
    ap.add_argument("--urls", help="text file of video/playlist URLs (one per line)")
    ap.add_argument("--out", default="data/cards", help="output root")
    ap.add_argument("--manifest", default="data/manifest.jsonl",
                    help="ingested clips, for audio alignment")
    ap.add_argument("--force", action="store_true", help="re-harvest even if cards.jsonl exists")
    args = ap.parse_args()

    if not (args.playlist or args.ids or args.urls):
        ap.error("give one of --playlist / --ids / --urls")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    videos = resolve_ids(args)
    if not videos:
        sys.exit("[!] no video ids resolved from the given sources")
    manifest = load_manifest(args.manifest)
    print(f"[i] {len(videos)} video(s) to harvest -> {out}", file=sys.stderr)

    index = []
    for n, (vid, title) in enumerate(videos, 1):
        vdir = out / vid
        vdir.mkdir(parents=True, exist_ok=True)
        cards_path = vdir / "cards.jsonl"
        if cards_path.exists() and not args.force:
            print(f"[skip] {n}/{len(videos)} {vid}: cards.jsonl exists", file=sys.stderr)
            index.append({"video_id": vid, "title": title, "status": "skipped"})
            continue
        try:
            print(f"[+] {n}/{len(videos)} {vid}: downloading", file=sys.stderr)
            video, dl_title = download_video(vid, vdir)
            title = title or dl_title
            _w, h = probe_dims(video)
            top_h = max(2, int(h * TOP_FRAC) // 2 * 2)  # even height for the codec

            print(f"[+] {n}/{len(videos)} {vid}: detecting cards", file=sys.stderr)
            windows = chroma_windows(video, vdir, top_h)
            picks = extract_frames(video, windows, vdir / "frames", top_h)
            if picks:
                build_sheets(vdir / "frames", vdir)

            clips = manifest.get(vid, [])
            with open(cards_path, "w", encoding="utf-8") as f:
                for i, ((s, e), (t, frame)) in enumerate(zip(windows, picks), 1):
                    row = {
                        "video_id": vid,
                        "source_url": f"https://www.youtube.com/watch?v={vid}",
                        "source_title": title,
                        "n": i,
                        "win_start": round(s, 2),
                        "win_end": round(e, 2),
                        "frame_t": round(t, 2),
                        "frame": str(frame.relative_to(out)),
                        "clip_ids": clips_in_window(clips, s, e),
                        "english": "",
                        "hindi": "",
                        "kutchi_roman": "",
                        "kutchi_guj_draft": "",
                        "needs_native_verify": True,
                        "notes": "",
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"    -> {len(windows)} cards, {len(clips)} clips aligned", file=sys.stderr)
            index.append({"video_id": vid, "title": title,
                          "cards": len(windows), "clips": len(clips), "status": "ok"})
        except subprocess.CalledProcessError as e:
            print(f"[!] {vid} failed: {(e.stderr or '')[:200]}", file=sys.stderr)
            index.append({"video_id": vid, "title": title, "status": "error"})
        except Exception as e:
            print(f"[!] {vid} error: {e}", file=sys.stderr)
            index.append({"video_id": vid, "title": title, "status": "error"})

    with open(out / "harvest_index.jsonl", "w", encoding="utf-8") as f:
        for r in index:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = sum(1 for r in index if r.get("status") == "ok")
    total = sum(r.get("cards", 0) for r in index)
    print(f"[done] harvested {total} cards across {ok} video(s)", file=sys.stderr)
    print(f"[next] read {out}/<id>/sheet_*.jpg -> fill english/hindi/kutchi_roman,",
          file=sys.stderr)
    print("       then draft kutchi_guj per spelling_convention.md (native ratifies).",
          file=sys.stderr)


if __name__ == "__main__":
    main()
