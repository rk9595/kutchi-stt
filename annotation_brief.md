# Kutchi Speech Annotation — Vendor Brief

**Prepared for:** [Agency name]
**Prepared by:** [Your name], Kutchi language-AI project
**Date:** [date]
**Contact:** [email / phone]

---

## 1. What we're building
A speech-to-text dataset for **Kutchi** — an oral, low-resource language of the Kutch
region (Gujarat, India). Kutchi is **not** Gujarati or Hindi and is not listed as an
independent language in the census. Written Kutchi barely exists, so our corpus is
**audio**, and the deliverable is **verified `(audio clip → Kutchi text)` pairs**.

**Unit of progress: verified minutes of audio** (not files, not clips).

## 2. Dialect and language requirement (read first — this is the gate)
- **Dialect: Bhuj-region Kutchi.** Annotators must be **native Kutchi speakers** who
  understand Bhuj-area speech. General "Indic language" or Gujarati annotators cannot do
  this work — the audio is Kutchi, not Gujarati.
- Our audio is pulled broad and is **heavily code-mixed** (Kutchi with Gujarati / Hindi /
  English). Clips that are *mostly* Gujarati/Hindi are **discarded** (marked `skip`), not
  transcribed. Auditioning and discarding is part of the job and part of the cost.

## 3. The task
This is **correction, not transcription from scratch.** Each clip already carries a rough
machine draft. Per clip the annotator:
1. Listens (2–20s clips), and decides status:
   - **verified** — Kutchi speech, corrected transcript produced
   - **skip** — mostly non-Kutchi, multi-speaker/overlap, or too unclear to transcribe
   - **nonspeech** — music / noise / silence
2. For `verified`: correct the draft into accurate Kutchi in **modified Gujarati script**,
   plus an optional romanized field.

## 4. Script and spelling rules (consistency is the whole game)
Full rules: **`spelling_convention.md`** (attached). The governing principle:

> **Consistency beats correctness.** A word spelled the same way every time is worth more
> than a "more correct" spelling used half the time. Annotators must **never** silently
> re-spell or normalize. Uncovered cases are escalated to the native-speaker lead and
> **logged**, then everyone copies that choice.

Key locked rules:
- **Script: modified Gujarati.** Type **Gujarati script only** — no Devanagari (a hard
  keyboard-setup requirement; mixed keyboards cause silent script-mixing that corrupts data).
- **Implosives: plain (Option A).** Write /ɓ/→બ, /ɗ/→ડ with **no nukta** in the main
  transcript; capture the implosive only in the `roman` field (bb/dd). Do not add diacritics.
- **Code-switching:** transliterate loanwords (incl. English) into Gujarati script as
  pronounced. One script per transcript.
- **Punctuation:** minimal — sentence-final `.` and `?` only.
- **Unclear:** one uncertain word → best guess + `(?)`. Worse than that → `skip`.

## 5. Tooling and data format
- Preferred tool: our browser-based correction desk (`transcribe_correct.html`) — it loads
  the manifest + audio, enforces the schema and status values, has keyboard shortcuts, and
  exports the required JSONL. If you propose using your own platform, output **must** match
  the schema below exactly.
- **Input:** `manifest.jsonl` (one JSON object per line) + a `clips/` folder of WAVs.
- **Output:** the same rows with `transcript`, `roman`, `status`, `notes` filled.

Schema (one JSON object per line):

| field | meaning |
|---|---|
| `id` | unique clip id (do not change) |
| `clip` | wav filename |
| `source_id` / `source_url` / `source_title` | provenance — **must be preserved** |
| `start` / `end` / `duration` | seconds within the source |
| `asr_draft` | machine guess — wrong by design, a starting point only |
| `transcript` | **the deliverable** — verified Kutchi, Gujarati script |
| `roman` | optional romanized form |
| `status` | `pending` → `verified` / `skip` / `nonspeech` |
| `notes` | speaker count, noise, uncertain words |

## 6. Quality control
- **~5% of clips double-annotated** (overlap) to measure inter-annotator agreement.
- We seed a **blind gold set** (clips we have already verified) into each batch and score
  your output against it. Acceptance threshold agreed before scale-up (see §9).
- Disagreements are treated as either ambiguous audio or a spelling-convention gap → we
  update the convention and re-issue, rather than arguing per clip.

## 7. Data handling and rights (contractual)
- Source **audio is confidential** and **must not be redistributed, republished, or reused**
  for any purpose beyond this annotation task. NDA required.
- No uploading clips to third-party/public services outside the agreed toolchain.
- **Provenance fields stay on every row** (needed for rights and dedup).
- Deliverables (transcripts) are **work-for-hire; IP assigns to us.**

## 8. Volume and phasing
- **Pilot (paid):** ~60 minutes of source audio, 2–3 annotators, incl. overlap + gold.
  Purpose: measure real per-annotator throughput and accuracy on Kutchi before scale-up.
- **First milestone: ~10 hours of verified audio** (enough for a first fine-tune). We
  reassess and size the next phase from the pilot's throughput and quality.
- Scale-up is contingent on the pilot meeting acceptance criteria.

**Sourcing model:** if you don't have Bhuj-Kutchi native speakers in-house, a
**managed-service arrangement is acceptable** — you provide project management, tooling,
and QC while we help source native Kutchi speakers (Mumbai/Kutch community, Kutchi Sahitya
Academy) to work under your process. Either model is fine; tell us which you're proposing.

## 9. Acceptance criteria (to finalize together)
- Agreement with gold set ≥ **[target]%** (character-level) on `verified` clips.
- Correct status calls (verified/skip/nonspeech) ≥ **[target]%**.
- Spelling-convention adherence (spot-checked by our native lead).
- Schema-valid JSONL, provenance intact, Gujarati-script-only transcripts.

## 10. Commercials (to discuss)
- Pricing basis: **per minute of source audio processed** (not per verified minute), since
  auditioning and discarding non-Kutchi clips is real work. Alternatively a blended rate —
  propose what you use.
- Include: annotator ramp on the convention, QC/overlap time, and a named native-speaker
  lead who ratifies spelling decisions.
