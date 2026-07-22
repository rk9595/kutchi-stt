# CLAUDE.md — Kutchi STT / LLM project

Operating context for Claude Code. Read this before doing anything in this repo.

## What this project is
Building Kutchi-language AI (STT first, then translation/TTS/chat). We are in **Phase 1:
data**. The bottleneck is not modeling — it's producing verified `(audio, Kutchi text)`
pairs. Models are commodities; the dataset is the moat.

## Locked decisions — do NOT change these without me saying so
- **NO human annotators (pivot 2026-07-19).** Solo + Claude, everything automated.
  Verification is machine consensus (Gemini audio + whisper ensemble) plus Claude
  adjudication; card-aligned lesson videos give the high-confidence tier. The ONE
  manual exception: the owner does a one-time listen-check of the frozen eval set.
- **Audio-first.** Kutchi is oral; real Kutchi *text* barely exists. The corpus is audio.
- **Dialect: Bhuj.** Core Kutch. Don't mix in other regions' speech without flagging.
- **Script: modified Gujarati.** Plus an optional romanized field.
- **Implosives: Option A (plain).** Write /ɓ/→બ, /ɗ/→ડ with NO nukta in the main
  transcript. Capture the implosive only in the `roman` field (bb/dd). See
  spelling_convention.md. Do not "improve" this by adding diacritics.
- **Prime rule: consistency beats correctness.** Never silently re-spell, normalize,
  or "clean up" verified transcripts. `lexicon.py` (+ data/lexicon.tsv) is the
  executable spelling convention — every machine-written transcript goes through
  `Lexicon.normalize()` at creation time; already-verified rows are only warned about.
  New spelling decisions get appended to lexicon.tsv AND spelling_convention.md.
- **Unit of progress: verified MINUTES of audio**, not file count.

## The pipeline (data flow — fully automated)
```
Track A  lesson videos → harvest_cards.py → Claude reads card sheets → phrasebook.jsonl
         → align_cards.py (fuzzy-match phrase↔audio, cut utterances)
         → data/cards/aligned/manifest.jsonl        [verified_by=machine-card]
Track B  URLs → ingest.py → clips + drafts → machine_verify.py run (whisper + Gemini ears)
         → agreement gate → machine_verify.py batch → Claude adjudicates
         → machine_verify.py apply → data/machine_verified.jsonl [verified_by=machine-consensus]
Track C  phrasebook.jsonl doubles as (en, hi, kutchi) parallel text for the later LLM phase
Then     merge.py → dataset/verified.jsonl → split.py → train/eval (eval FROZEN,
         one-time human check) → finetune_whisper.py → eval.py CER
Loop     fine-tuned model becomes an extra machine_verify ear → re-run on pending → retrain
```

## Data schema (manifest.jsonl, one JSON object per line)
`id, clip, source_id, source_url, source_title, start, end, duration,`
`asr_draft, asr_model, transcript, roman, status, notes, verified_by, confidence`
- `asr_draft` = machine guess (Gujarati). Wrong by design; a correction seed.
- `transcript` = the deliverable: verified Kutchi in Gujarati script.
- `status` ∈ {pending, verified, skip, nonspeech}. Any script that builds training data
  uses ONLY `status == "verified"`.
- `verified_by` ∈ {human, machine-card, machine-consensus} — tiers are NEVER conflated;
  merge.py preserves them and higher tiers win dedup. `confidence` = match/agreement score.

## Hard guardrails
- **Never train on the eval set.** split.py writes eval IDs to a locked file; training
  scripts must refuse to load any clip whose id is in it.
- **Eval must be human-checked once** (the one manual step). Machine-verified rows may
  train; an unreviewed eval row means CER numbers are not to be quoted as real accuracy.
- **Keep provenance** (source_id/url) on every row — needed for dedup and rights.
  (Self-recordings: `source_id=selfrec-<speaker>` suffices.)
- **Don't redistribute raw audio.** Clips stay private; published artifacts are the
  models + (where licensing allows) transcripts, not the source audio.
- **Sample-check a new source is actually Kutchi** (3 clips) before bulk-processing it.

## Status (2026-07-21)
- DONE: ingest.py, transcribe_correct.html (now only for the eval check),
  spelling_convention.md + lexicon.py/data/lexicon.tsv, harvest_cards.py (+111-phrase
  phrasebook), align_cards.py (Track A run complete), machine_verify.py (Track B full
  88-min backlog run + adjudication complete), merge.py, split.py, finetune_whisper.py,
  eval.py. Smoke-tested end-to-end (whisper-tiny) — chain runs clean.
- DATASET: 479 verified clips, 22.1 min (machine-consensus 9.1, machine-card 8.9,
  human 4.1). 651 clips (33.8 min) remain `pending` — ears disagreed; held for the loop.
- CONTAMINATION FIX (2026-07-21): quarantined source `Eg5tUArUuL4` — 193 clips / 10.3 min
  of Gujarati that passed machine-consensus (it verifies transcription agreement, not
  language). Dropped machine-consensus 19.4→9.1 min; the old "30.5 min" was inflated.
  Quarantined at both `data/machine_verified.jsonl` and `dataset/verified.jsonl` (backups
  `*.bak`), regenerated via merge.py. machine_verify.py now has a lexical language gate
  (`gujarati_suspect()` → verdict `lang-review`) that HOLDS high-agreement Gujarati-reading
  clips. NOT yet built: a known-bad-source blocklist at ingest to stop re-entry entirely.
- Track B ears: gemini-3.5-flash x gemini-3.1-flash-lite (Pro model capped at 250
  req/day on this key's tier; use sparingly as a tie-break ear on near-miss pendings).
- NEXT: keep feeding lesson/card-style URLs (Track A is the highest-yield, zero-cost
  lever) → cross 60 min → freeze eval + owner's one-time eval check → bootstrap round 1
  (fine-tune whisper-small, CER on frozen eval, add model as ear 3, re-verify pendings).

## Targets
5–10 verified hrs → first fine-tune. 50–100 hrs → usable STT. Owner keeps feeding
speech-forward Kutchi URLs (lesson/card-style content is the highest-yield input).
