#!/usr/bin/env python3
"""
Kutchi Phase 1, Track B — machine-consensus verification of conversational clips.

No human annotators: a clip gets verified when independent machine "ears" agree.
  ear A: gemini-3.5-flash audio    (GEMINI_API_KEY env var)
  ear B: gemini-3.1-pro-preview audio
(The ingest-time whisper draft proved useless as an ear — on real Kutchi it
hallucinates English/Telugu — so consensus is intra-Gemini flash-vs-pro until
our fine-tuned whisper is good enough to join as an independent ear via
--local-model.) Hypotheses go through lexicon.normalize(); agreement =
character error rate between them. Nothing is auto-written to the dataset: agreeing clips land
in adjudication batches that Claude reviews in-session (picks/fixes the final
transcript per spelling_convention.md), then `apply` folds those decisions into
data/machine_verified.jsonl, which merge.py consumes.

Disagreeing clips stay pending WITH their hypotheses stored — the next
bootstrap round (fine-tuned model as ear 3) retries them. Clips Gemini hears
as clearly non-Kutchi (hindi/english/other) are marked skip; "gujarati" is NOT
auto-skipped — Kutchi is routinely mislabeled Gujarati, so those go to
adjudication with a flag.

Subcommands:
    run    --manifest data/manifest.jsonl --clips data/clips [--limit N]
           transcribe + score; resume-safe state in data/machine_verify_state.jsonl
    batch  emit data/mv_batches/batch_###.json for Claude adjudication
    apply  --decisions <file>   fold adjudicated rows into data/machine_verified.jsonl

Typical cost: Gemini Flash audio ≈ pennies for the whole 88-min backlog.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from lexicon import Lexicon

STATE = Path("data/machine_verify_state.jsonl")
VERIFIED_OUT = Path("data/machine_verified.jsonl")
BATCH_DIR = Path("data/mv_batches")
AGREE_CER = 0.15
BATCH_SIZE = 40
EAR_A = "gemini-3.5-flash"          # pinned for reproducibility, not -latest
EAR_B = "gemini-3.1-pro-preview"
NON_KUTCHI = {"hindi", "english", "other", "nonspeech"}
# Lexical language gate (defends against fluent Gujarati passing consensus): real
# Kutchi carries these function words even when an ear mislabels the language;
# Gujarati-only text has the second set and none of the first. Heuristic — it only
# HOLDS suspects for review, never auto-skips, so an incomplete list can't lose data.
KUTCHI_MARKERS = {"આઉં", "મૂં", "મુકે", "પાં", "અસાં", "અઈ", "આય", "થો", "થા", "થી",
                  "વેંધો", "વેજનૂ", "વેરો", "અચિનૂ", "હાલ્યો", "હલ", "હેડા", "હેતે",
                  "ભોલનૂ", "વેનો", "વેનનૂ", "જેંલા", "ચ્યાં", "બોલાયો", "અચો", "હલો"}
GUJARATI_MARKERS = {"છે", "છીએ", "છો", "હતું", "હતી", "હતા", "નથી", "જણાવશો",
                    "રહ્યું", "એટલે", "પરિસ્થિતિ", "કેટલું", "મહત્વ", "જીવનમાં",
                    "સ્વભાવ", "ઘટના", "સંવાદદાતા", "દર્શકને", "જાણવા", "શકું"}
RESPONSE_SCHEMA = {                  # enforced JSON: unescaped quotes in the
    "type": "OBJECT",                # transcript were breaking naive parsing
    "properties": {"language": {"type": "STRING"}, "transcript": {"type": "STRING"}},
    "required": ["language", "transcript"],
}

GEMINI_PROMPT = """\
This audio is a short clip scraped for a KUTCHI (Kachchhi) speech corpus — a
Sindhi-related language of Kutch, Gujarat, usually written in Gujarati script.
BUT the source videos are code-mixed: the speaker may actually be speaking
Gujarati, Hindi, or English here. Transcribe it, and classify the language by
what you HEAR, not by what the corpus wants it to be.

Rules:
- Gujarati script ONLY (no Devanagari, no Latin). Transliterate any embedded
  Hindi/English words into Gujarati script as pronounced.
- Write exactly what is spoken, including fillers. No translation, no cleanup.
- Punctuation: only sentence-final . and ? — no commas.
- Also classify the language actually spoken.

Return JSON: {"language": "kutchi|gujarati|hindi|english|mixed|other|nonspeech",
"transcript": "<Gujarati-script transcription, empty if nonspeech>"}"""


def cer(a, b):
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(len(a), len(b))


def gujarati_suspect(hyp_a, hyp_b, lang_a, lang_b):
    """True when the consensus text reads as Gujarati (not Kutchi): a Gujarati
    signal present (marker word or ear label) AND no Kutchi marker anywhere."""
    import re
    toks = set(re.findall(r"[઀-૿]+", hyp_a + " " + hyp_b))
    if toks & KUTCHI_MARKERS:
        return False
    return bool(toks & GUJARATI_MARKERS) or "gujarati" in {lang_a, lang_b}


def load_jsonl(path):
    if not Path(path).exists():
        return []
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()
            if l.strip()]


def append_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def find_clip(name, clip_dirs):
    for d in clip_dirs:
        p = Path(d) / name
        if p.exists():
            return p
    return None


# ---- ears -----------------------------------------------------------------

def gemini_transcribe(client, wav_path, model=EAR_A, retries=3):
    from google.genai import types
    audio = types.Part.from_bytes(data=wav_path.read_bytes(), mime_type="audio/wav")
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[audio, GEMINI_PROMPT],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=RESPONSE_SCHEMA, temperature=0.0),
            )
            data = json.loads(resp.text)
            return {"language": str(data.get("language", "")).lower(),
                    "transcript": data.get("transcript", "") or ""}
        except Exception as e:
            if attempt == retries - 1:
                return {"language": "error", "transcript": "", "error": str(e)[:200]}
            time.sleep(2 ** attempt)


def local_transcribe(model_name, wav_path):
    from faster_whisper import WhisperModel
    if not hasattr(local_transcribe, "_model"):
        local_transcribe._model = WhisperModel(model_name, device="auto",
                                               compute_type="int8")
    segments, _ = local_transcribe._model.transcribe(str(wav_path), language="gu",
                                                     vad_filter=True)
    return " ".join(s.text.strip() for s in segments).strip()


# ---- subcommands ----------------------------------------------------------

def cmd_run(args):
    lex = Lexicon.load()
    state = {r["id"]: r for r in load_jsonl(STATE)}
    if args.redo_errors:
        # an errored ear means the clip never got a fair two-ear hearing;
        # drop it from state (file rewritten) so this run redoes it
        errored = {i for i, r in state.items()
                   if "error" in (r.get("lang_a"), r.get("lang_b"))}
        if errored:
            state = {i: r for i, r in state.items() if i not in errored}
            STATE.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                     for r in state.values()), encoding="utf-8")
            print(f"[i] --redo-errors: {len(errored)} error-tainted clip(s) requeued",
                  file=sys.stderr)
    manifest = [r for src in args.manifest for r in load_jsonl(src)]
    pending = [r for r in manifest
               if r.get("status") == "pending" and r["id"] not in state]
    if args.limit:
        pending = pending[:args.limit]
    if not pending:
        print("[i] nothing new to verify", file=sys.stderr)
        return

    client = None
    if not args.no_gemini:
        if not os.environ.get("GEMINI_API_KEY"):
            sys.exit("[!] GEMINI_API_KEY not set (or pass --no-gemini for local-only)")
        from google import genai
        # explicit timeout: without it a single wedged HTTP call hangs the run forever
        client = genai.Client(http_options={"timeout": 90_000})

    print(f"[i] {len(pending)} clip(s) to verify ({args.ear_a} x {args.ear_b})",
          file=sys.stderr)
    new_rows, agree, disagree, skip, lang_review = [], 0, 0, 0, 0
    for n, r in enumerate(pending, 1):
        clip = find_clip(r["clip"], args.clips)
        if not clip:
            continue
        ear_a = gemini_transcribe(client, clip, args.ear_a)
        ear_b = gemini_transcribe(client, clip, args.ear_b)
        hyp_a, _ = lex.normalize(ear_a["transcript"])
        hyp_b, _ = lex.normalize(ear_b["transcript"])
        hyp_local = ""
        if args.local_model:
            hyp_local, _ = lex.normalize(local_transcribe(args.local_model, clip))

        agreement = 1.0 - cer(hyp_a, hyp_b) if hyp_a and hyp_b else 0.0
        langs = {ear_a["language"], ear_b["language"]}
        gu_suspect = gujarati_suspect(hyp_a, hyp_b, ear_a["language"], ear_b["language"])
        if langs <= NON_KUTCHI | {"error"} and langs & NON_KUTCHI:
            verdict, skip = "skip", skip + 1        # both ears: not Kutchi
        elif agreement >= 1.0 - AGREE_CER and gu_suspect:
            # high-agreement text that reads as Gujarati, not Kutchi — the silent-accept
            # trap. HOLD for explicit language review; do NOT send to normal adjudication.
            verdict, lang_review = "lang-review", lang_review + 1
        elif agreement >= 1.0 - AGREE_CER:
            verdict, agree = "adjudicate", agree + 1
        else:
            verdict, disagree = "pending", disagree + 1

        row = {**r, "hyp_a": hyp_a, "hyp_b": hyp_b, "hyp_local": hyp_local,
               "ear_models": [args.ear_a, args.ear_b],
               "lang_a": ear_a["language"], "lang_b": ear_b["language"],
               "lang_flag": "gujarati-suspect" if gu_suspect else "",
               "agreement": round(agreement, 3), "mv_verdict": verdict}
        new_rows.append(row)
        append_jsonl(STATE, [row])   # per-clip: a killed run loses nothing
        if n % 25 == 0 or n == len(pending):
            print(f"    {n}/{len(pending)}  agree:{agree} pending:{disagree} skip:{skip}",
                  file=sys.stderr)

    secs = sum(r.get("duration") or 0 for r in new_rows if r["mv_verdict"] == "adjudicate")
    print(f"[done] {agree} clip(s) ({secs/60:.1f} min) ready for adjudication, "
          f"{disagree} disagree (kept pending), {lang_review} held for language review "
          f"(gujarati-suspect), {skip} non-Kutchi skipped.\n"
          f"[next] python machine_verify.py batch", file=sys.stderr)


def cmd_batch(args):
    done_ids = {r["id"] for r in load_jsonl(VERIFIED_OUT)}
    rows = [r for r in load_jsonl(STATE)
            if r.get("mv_verdict") == "adjudicate" and r["id"] not in done_ids]
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    for p in BATCH_DIR.glob("batch_*.json"):
        p.unlink()
    for i in range(0, len(rows), BATCH_SIZE):
        batch = [{"id": r["id"], "duration": r.get("duration"),
                  "lang_a": r.get("lang_a"), "lang_b": r.get("lang_b"),
                  "agreement": r.get("agreement"),
                  "hyp_a": r["hyp_a"], "hyp_b": r["hyp_b"]}
                 for r in rows[i:i + BATCH_SIZE]]
        (BATCH_DIR / f"batch_{i // BATCH_SIZE:03d}.json").write_text(
            json.dumps(batch, ensure_ascii=False, indent=1))
    print(f"[done] {len(rows)} row(s) in {(len(rows)+BATCH_SIZE-1)//BATCH_SIZE} batch file(s) "
          f"under {BATCH_DIR}/ — Claude reads each batch, returns decisions as jsonl rows "
          f'{{"id", "transcript", "status": "verified|skip", "notes"}}, then: '
          f"python machine_verify.py apply --decisions <file>", file=sys.stderr)


def cmd_apply(args):
    lex = Lexicon.load()
    state = {r["id"]: r for r in load_jsonl(STATE)}
    existing = {r["id"] for r in load_jsonl(VERIFIED_OUT)}
    out, n_ver, n_skip = [], 0, 0
    for d in load_jsonl(args.decisions):
        r = state.get(d["id"])
        if not r or d["id"] in existing:
            continue
        if d.get("status") == "verified":
            transcript, flags = lex.normalize(d.get("transcript", ""))
            bad = {"latin-remnant", "devanagari-remnant", "slash-alternates"}
            if not transcript or bad & set(flags):
                print(f"[!] {d['id']}: rejected transcript ({flags})", file=sys.stderr)
                continue
            n_ver += 1
            out.append({**{k: r[k] for k in r if not k.startswith(("hyp_", "mv_", "lang_", "agreement"))},
                        "transcript": transcript,
                        "status": "verified",
                        "verified_by": "machine-consensus",
                        "confidence": r.get("agreement"),
                        "notes": d.get("notes", "")})
        elif d.get("status") == "skip":
            n_skip += 1
            out.append({**{k: r[k] for k in r if not k.startswith(("hyp_", "mv_", "lang_", "agreement"))},
                        "status": "skip", "verified_by": "machine-consensus",
                        "notes": d.get("notes", "")})
    append_jsonl(VERIFIED_OUT, out)
    print(f"[done] +{n_ver} verified, +{n_skip} skipped -> {VERIFIED_OUT}\n"
          f"[next] python merge.py", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run")
    p.add_argument("--manifest", action="append", default=None)
    p.add_argument("--clips", action="append", default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--local-model", default="",
                   help="re-transcribe locally with this faster-whisper model "
                        "(default: reuse asr_draft)")
    p.add_argument("--no-gemini", action="store_true")
    p.add_argument("--redo-errors", action="store_true",
                   help="requeue state rows where an ear errored (504s etc.)")
    p.add_argument("--ear-a", default=EAR_A)
    p.add_argument("--ear-b", default=EAR_B)

    sub.add_parser("batch")

    p = sub.add_parser("apply")
    p.add_argument("--decisions", required=True)

    args = ap.parse_args()
    if args.cmd == "run":
        args.manifest = args.manifest or ["data/manifest.jsonl"]
        args.clips = args.clips or ["data/clips"]
        cmd_run(args)
    elif args.cmd == "batch":
        cmd_batch(args)
    else:
        cmd_apply(args)


if __name__ == "__main__":
    main()
