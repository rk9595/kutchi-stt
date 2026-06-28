# CLAUDE.md — Kutchi STT / LLM project

Operating context for Claude Code. Read this before doing anything in this repo.

## What this project is
Building Kutchi-language AI (STT first, then translation/TTS/chat). We are in **Phase 1:
data**. The bottleneck is not modeling — it's producing verified `(audio, Kutchi text)`
pairs. Models are commodities; the dataset is the moat.

## Locked decisions — do NOT change these without me saying so
- **Audio-first.** Kutchi is oral; real Kutchi *text* barely exists. The corpus is audio.
- **Dialect: Bhuj.** Core Kutch. Don't mix in other regions' speech without flagging.
- **Script: modified Gujarati.** Plus an optional romanized field.
- **Implosives: Option A (plain).** Write /ɓ/→બ, /ɗ/→ડ with NO nukta in the main
  transcript. Capture the implosive only in the `roman` field (bb/dd). See
  spelling_convention.md. Do not "improve" this by adding diacritics.
- **Prime rule: consistency beats correctness.** Never silently re-spell, normalize,
  or "clean up" verified transcripts. Uncovered cases get logged in
  spelling_convention.md, not improvised.
- **Unit of progress: verified MINUTES of audio**, not file count.

## The pipeline (data flow)
```
URLs → ingest.py → clips/*.wav + manifest.jsonl (drafts)
     → transcribe_correct.html (humans) → *_verified.jsonl per annotator
     → merge.py → dataset/verified.jsonl
     → split.py → train.jsonl + eval.jsonl   (eval is FROZEN)
     → fine-tune Whisper → first Kutchi STT
```

## Data schema (manifest.jsonl, one JSON object per line)
`id, clip, source_id, source_url, source_title, start, end, duration,`
`asr_draft, asr_model, transcript, roman, status, notes`
- `asr_draft` = machine guess (Gujarati). Wrong by design; a human-correction seed.
- `transcript` = the deliverable: verified Kutchi in Gujarati script.
- `status` ∈ {pending, verified, skip, nonspeech}. Any script that builds training data
  uses ONLY `status == "verified"`.

## Hard guardrails
- **Never train on the eval set.** split.py writes eval IDs to a locked file; training
  scripts must refuse to load any clip whose id is in it.
- **Keep provenance** (source_id/url) on every row — needed for dedup and rights.
- **Don't redistribute raw audio.** Clips stay private; published artifacts are the
  models + (where licensing allows) transcripts, not the source audio.

## Status
- DONE: ingest.py (audio → clips + drafts), transcribe_correct.html (correction desk),
  spelling_convention.md, README.md.
- NEXT (build in-repo, test against real files): split_for_annotators.py (shard manifest,
  ~5% overlap for QC), merge.py (combine + flag disagreements), split.py (frozen eval +
  train/dev), then a Whisper fine-tune + eval harness once ~10 verified hours exist.

## Targets
5–10 verified hrs → first fine-tune. 50–100 hrs → usable STT. Frozen eval set: 1–2 hrs,
set aside on day one.
