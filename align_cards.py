#!/usr/bin/env python3
"""
Kutchi Phase 1, Track A — align harvested card text to the spoken audio.

harvest_cards.py + the Claude reading pass produced data/cards/phrasebook.jsonl:
native-authored Kutchi phrases (roman + Gujarati draft) that the teacher *speaks*
somewhere in each lesson video. This script finds WHERE, cuts that exact
utterance, and emits verified (audio, transcript) rows — STT training pairs with
no human transcription.

Method per video:
  1. extract 16 kHz mono audio from the already-downloaded low-res mp4
  2. faster-whisper with word timestamps, language=gu (cached to asr_gu.json —
     the draft is wrong-by-design; we only need it as a phonetic timeline)
  3. reduce both the card phrase and the ASR word stream to a coarse phonetic
     skeleton (retroflex/dental collapsed, aspiration dropped, vowel classes)
     and fuzzy-match the phrase against word n-grams; the teacher usually says
     the phrase 1-3 times, so take every non-overlapping match >= threshold
  4. cut each match (with padding) to a wav; write a manifest row with
     status=verified, verified_by=machine-card, confidence=match score,
     transcript = lexicon-normalized kutchi_guj (the matched '/'-alternate)

Phrases that never match land in align_report.jsonl with their best score —
they stay out of the dataset, they are not deleted.

Usage:
    python align_cards.py                          # all videos in the phrasebook
    python align_cards.py --vids 1oLsZIOGFE8 ...   # subset
    python align_cards.py --threshold 0.8 --model medium
"""

import argparse
import difflib
import json
import re
import subprocess
import sys
from pathlib import Path

from lexicon import Lexicon

CARDS = Path("data/cards")
PAD_BEFORE, PAD_AFTER = 0.25, 0.35
MIN_CLIP_SEC, MAX_CLIP_SEC = 0.5, 15.0
MIN_SKELETON = 4          # phrases with tinier skeletons match everything
MAX_REPETITIONS = 4

# ---- phonetic skeleton ----------------------------------------------------
# Coarse on purpose: whisper's Gujarati guess at Kutchi audio is wrong in
# exactly these dimensions (aspiration, retroflex/dental, vowel length).
GUJ_CLASS = {}
for chars, cls in [
    ("કખ", "k"), ("ગઘ", "g"), ("ચછ", "c"), ("જઝ", "j"),
    ("ટઠતથ", "t"), ("ડઢદધ", "d"), ("ણનં", "n"),
    ("પફ", "p"), ("બભ", "b"), ("મ", "m"),
    ("ય", "y"), ("રઋ", "r"), ("લળ", "l"), ("વ", "v"),
    ("સશષ", "s"), ("હ", "h"),
    ("અઆા", "a"), ("ઇઈિી", "i"), ("ઉઊુૂ", "u"),
    ("એઍેૅૈઐઁ", "e"), ("ઓઑોૉૌઔ", "o"),
]:
    for ch in chars:
        GUJ_CLASS[ch] = cls

ROMAN_DIGRAPH = [("chh", "c"), ("ch", "c"), ("sh", "s"), ("kh", "k"), ("gh", "g"),
                 ("th", "t"), ("dh", "d"), ("ph", "p"), ("bh", "b"), ("jh", "j"),
                 ("zh", "j"), ("aa", "a"), ("ee", "i"), ("ii", "i"), ("oo", "u"),
                 ("uu", "u"), ("ai", "e"), ("au", "o")]
ROMAN_CLASS = {"z": "j", "w": "v", "f": "p", "q": "k", "x": "k"}


def skeleton_guj(text):
    # whisper writes the Hindi framing (and often the Kutchi phrase itself) in
    # Devanagari; the blocks are codepoint-parallel, so shift into Gujarati first
    out = []
    for ch in text:
        if 0x0900 <= ord(ch) <= 0x097F:
            ch = chr(ord(ch) + 0x180)
        out.append(GUJ_CLASS.get(ch, ""))
    return "".join(out)


def skeleton_roman(text):
    t = text.lower()
    for dg, rep in ROMAN_DIGRAPH:
        t = t.replace(dg, rep)
    out = []
    for ch in t:
        if not ch.isalpha():
            continue
        ch = ROMAN_CLASS.get(ch, ch)
        if not out or out[-1] != ch:      # collapse doubles (incl. implosive bb/dd)
            out.append(ch)
    return "".join(out)


# ---- ASR ------------------------------------------------------------------

def video_path(vid):
    d = CARDS / vid
    return next((p for p in d.glob(f"{vid}.*") if p.suffix in (".mp4", ".webm", ".mkv")), None)


def ensure_wav(vid):
    wav = CARDS / vid / f"{vid}.16k.wav"
    if wav.exists() and wav.stat().st_size > 0:
        return wav
    src = video_path(vid)
    if not src:
        return None
    subprocess.run(["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "16000",
                    "-vn", str(wav)], check=True, capture_output=True)
    return wav


def asr_words(vid, model_name):
    cache = CARDS / vid / f"asr_{model_name}_gu.json"
    if cache.exists():
        return json.loads(cache.read_text())
    wav = ensure_wav(vid)
    if not wav:
        return None
    from faster_whisper import WhisperModel
    if not hasattr(asr_words, "_model"):
        asr_words._model = WhisperModel(model_name, device="auto", compute_type="int8")
    # condition_on_previous_text off: lesson videos with music beds send whisper
    # into repetition loops that decode 5x slower than real time
    segments, _ = asr_words._model.transcribe(
        str(wav), language="gu", word_timestamps=True, vad_filter=True,
        condition_on_previous_text=False, temperature=0.0)
    words = [{"w": w.word.strip(), "s": round(w.start, 2), "e": round(w.end, 2)}
             for seg in segments for w in seg.words if w.word.strip()]
    cache.write_text(json.dumps(words, ensure_ascii=False))
    return words


# ---- matching -------------------------------------------------------------

def find_matches(phrase_skels, words, threshold):
    """All non-overlapping word spans whose skeleton fuzzy-matches any variant.
    Returns [(score, i, j, variant_idx)] best-first, span = words[i:j]."""
    word_skels = [max(skeleton_guj(w["w"]), skeleton_roman(w["w"]), key=len)
                  for w in words]
    target_len = max(len(s) for s in phrase_skels)
    spans = []
    for i in range(len(words)):
        skel = ""
        for j in range(i, len(words)):
            skel += word_skels[j]
            if len(skel) > 1.8 * target_len + 3:
                break
            if len(skel) < 0.5 * target_len:
                continue
            for vi, ps in enumerate(phrase_skels):
                if not ps:
                    continue
                score = difflib.SequenceMatcher(None, ps, skel).ratio()
                if score >= threshold:
                    spans.append((score, i, j + 1, vi))
    spans.sort(reverse=True)
    picked, used = [], set()
    for score, i, j, vi in spans:
        if used & set(range(i, j)):
            continue
        picked.append((score, i, j, vi))
        used |= set(range(i, j))
        if len(picked) >= MAX_REPETITIONS:
            break
    return picked


# ---- main -----------------------------------------------------------------

def cut_clip(vid, start, end, dst):
    subprocess.run(["ffmpeg", "-y", "-ss", f"{start:.2f}", "-to", f"{end:.2f}",
                    "-i", str(CARDS / vid / f"{vid}.16k.wav"),
                    "-ac", "1", "-ar", "16000", str(dst)],
                   check=True, capture_output=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phrasebook", default=str(CARDS / "phrasebook.jsonl"))
    ap.add_argument("--out", default=str(CARDS / "aligned"))
    ap.add_argument("--model", default="small")
    ap.add_argument("--threshold", type=float, default=0.75)
    ap.add_argument("--vids", nargs="*", help="restrict to these video ids")
    args = ap.parse_args()

    lex = Lexicon.load()
    out = Path(args.out)
    clips_dir = out / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    phrases = [json.loads(l) for l in Path(args.phrasebook).read_text(
        encoding="utf-8").splitlines() if l.strip()]
    by_vid = {}
    for p in phrases:
        by_vid.setdefault(p["video_id"], []).append(p)
    def wav_size(v):
        p = video_path(v)
        return p.stat().st_size if p else 0

    # shortest first: early yield, and a pathological long video can't starve the rest
    vids = args.vids or sorted(by_vid, key=wav_size)

    manifest_rows, report_rows = [], []
    total_sec = 0.0
    for vid in vids:
        if not video_path(vid):
            print(f"[!] {vid}: video file missing, skipping", file=sys.stderr)
            continue
        print(f"[+] {vid}: ASR ({args.model}, gu)", file=sys.stderr)
        words = asr_words(vid, args.model)
        if not words:
            print(f"[!] {vid}: no ASR words", file=sys.stderr)
            continue

        for p in by_vid[vid]:
            # parentheticals in the cards are example sentences, not the phrase:
            # matching them pairs a whole spoken sentence with a one-word transcript
            strip = lambda t: re.sub(r"\([^)]*\)", " ", t or "")
            guj_alts = [a.strip() for a in strip(p.get("kutchi_guj")).split("/") if a.strip()]
            rom_alts = [a.strip() for a in strip(p.get("kutchi_roman")).split("/") if a.strip()]
            variants = []   # (guj_text, roman_text, [skeletons])
            for k, guj in enumerate(guj_alts):
                rom = rom_alts[k] if k < len(rom_alts) else (rom_alts[0] if rom_alts else "")
                gs = skeleton_guj(guj)
                skels = [s for s in (gs,) if len(s) >= MIN_SKELETON]
                rs = skeleton_roman(rom)
                # roman helps only when it renders the SAME phrase, so its
                # skeleton must be about as long as the Gujarati one
                if len(rs) >= MIN_SKELETON and gs and len(rs) <= 1.4 * len(gs):
                    skels.append(rs)
                if skels:
                    variants.append((guj, rom, skels))
            if not variants:
                report_rows.append({"phrase_id": p["id"], "video_id": vid,
                                    "reason": "no-usable-skeleton",
                                    "kutchi_guj": p.get("kutchi_guj", "")})
                continue

            flat_skels, flat_var = [], []
            for vi, (_, _, skels) in enumerate(variants):
                for s in skels:
                    flat_skels.append(s)
                    flat_var.append(vi)
            matches = find_matches(flat_skels, words, args.threshold)
            if not matches:
                report_rows.append({"phrase_id": p["id"], "video_id": vid,
                                    "reason": "no-match",
                                    "kutchi_guj": p.get("kutchi_guj", "")})
                continue

            for r, (score, i, j, si) in enumerate(matches, 1):
                guj, rom, _ = variants[flat_var[si]]
                span_skel = "".join(
                    max(skeleton_guj(w["w"]), skeleton_roman(w["w"]), key=len)
                    for w in words[i:j])
                # audio span much longer than the transcript = matched extra speech
                if len(span_skel) > 2.0 * max(1, len(skeleton_guj(guj))):
                    continue
                transcript, flags = lex.normalize(guj)
                if not transcript or any(f in ("latin-remnant", "devanagari-remnant",
                                               "slash-alternates") for f in flags):
                    report_rows.append({"phrase_id": p["id"], "video_id": vid,
                                        "reason": f"bad-transcript:{flags}",
                                        "kutchi_guj": guj})
                    continue
                start = max(0.0, words[i]["s"] - PAD_BEFORE)
                end = words[j - 1]["e"] + PAD_AFTER
                dur = end - start
                if not (MIN_CLIP_SEC <= dur <= MAX_CLIP_SEC):
                    continue
                clip_name = f"card_{vid}_p{p['id']:03d}_r{r}.wav"
                cut_clip(vid, start, end, clips_dir / clip_name)
                manifest_rows.append({
                    "id": f"card-{vid}-p{p['id']:03d}-r{r}",
                    "clip": clip_name,
                    "source_id": vid,
                    "source_url": p.get("source_url", f"https://www.youtube.com/watch?v={vid}"),
                    "source_title": "",
                    "start": round(start, 2), "end": round(end, 2),
                    "duration": round(dur, 2),
                    "asr_draft": " ".join(w["w"] for w in words[i:j]),
                    "asr_model": f"faster-whisper-{args.model}-gu",
                    "transcript": transcript,
                    "roman": rom,
                    "status": "verified",
                    "verified_by": "machine-card",
                    "confidence": round(score, 3),
                    "notes": f"phrasebook id={p['id']} rep={r}; en: {p.get('english','')}",
                })
                total_sec += dur

    with open(out / "manifest.jsonl", "w", encoding="utf-8") as f:
        for r in manifest_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "align_report.jsonl", "w", encoding="utf-8") as f:
        for r in report_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    matched_phrases = {r["id"].rsplit("-r", 1)[0] for r in manifest_rows}
    print(f"[done] {len(manifest_rows)} clips from {len(matched_phrases)} phrases "
          f"-> {total_sec/60:.1f} verified min ({out}/manifest.jsonl)", file=sys.stderr)
    print(f"       {len(report_rows)} phrases unmatched/rejected -> align_report.jsonl",
          file=sys.stderr)


if __name__ == "__main__":
    main()
