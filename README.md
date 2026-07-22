# Kutchi STT & LM — building AI for an oral, low-resource language

Kutchi (કચ્છી) is spoken by a few million people across Kutch, Mumbai, and a global
diaspora. India's census folds it under Gujarati; it has no official status, no
standard orthography, and — the fact that shapes this entire project — **almost no
written text**. It is an oral language.

That single fact inverts the usual NLP recipe. You cannot scrape a Kutchi corpus,
because one barely exists. So this repo is **data-infrastructure first**: a set of
pipelines that manufacture verified `(audio, Kutchi text)` pairs, plus a from-scratch
language-model track that shows exactly how far the resulting text can (and cannot)
take you.

This is **Phase 1, in progress** — a public research log, not a finished product.
Numbers below are real and come with their caveats. Nothing here is a released model.

---

## The thesis

> Models are commodities; a verified Kutchi dataset is the moat.

Whisper, GPT-2, IndicWhisper — the architectures are free. What does not exist is
Kutchi supervision. So the unit of progress is not "trained a model," it is
**verified minutes of audio**. Everything is built to grow that number cheaply and
honestly, and to never lie to itself about how much real signal it has.

---

## Two tracks

### Track 1 — Speech-to-text data pipeline (the main line)

```
URLs ──ingest.py──▶ clips/*.wav + manifest.jsonl (rough Gujarati ASR drafts)
                         │
      ┌──────────────────┴───────────────────┐
      ▼ (lesson videos)                        ▼ (conversational audio)
 harvest_cards.py                         machine_verify.py
   read on-screen phrase cards              transcribe each clip with two
   → phrasebook.jsonl                        independent machine "ears"
 align_cards.py                              (Gemini-audio × 2), score agreement
   fuzzy-match phrase ↔ audio,               by character-error-rate
   cut the exact utterance                 ── agree ──▶ Claude adjudicates the
   [verified_by = machine-card]                          final transcript
                                            [verified_by = machine-consensus]
      └──────────────────┬───────────────────┘
                         ▼
              merge.py ──▶ dataset/verified.jsonl
                         ▼
              split.py ──▶ train.jsonl + eval.jsonl   (eval is FROZEN)
                         ▼
              finetune_whisper.py ──▶ eval.py (CER)
                         ▼
              (loop) the fine-tuned model becomes a third ear → re-verify pending clips
```

There are **no human annotators**. Verification is either card-alignment (a native
teacher already wrote the phrase on screen; we just find where they say it) or
machine consensus (two models must agree, then Claude adjudicates against a written
spelling convention). The *only* manual step is a one-time listen-check of the frozen
eval set — because you cannot honestly quote accuracy against an unheard reference.

Every transcript passes through `lexicon.py` (`Lexicon.normalize()`), the spelling
convention encoded as code, so the machine makes the *same* orthographic choice every
time. The prime rule is **consistency beats correctness**: verified text is never
silently re-spelled.

### Track 2 — Kutchi language model from scratch (`lm/`)

A Karpathy-style "build a GPT" curriculum, but for Kutchi — and a live demonstration
of the low-resource ceiling. Kutchi has, at time of writing, on the order of a few
thousand sentences of usable text (tens of KB). You cannot train a GPT from scratch on
that; the model has more parameters than the corpus has characters. The realistic
recipe — and the one implemented here — is **pretrain on Gujarati** (same script,
close language, a real ~110 MB Wikipedia corpus) then **specialize on Kutchi**. Kutchi
ends up a thin adapter on a Gujarati base. That is the honest ceiling, and the code is
written to make it visible, not to hide it.

---

## What is actually done (2026-07)

**Track 1 — pipeline: complete and smoke-tested end-to-end.**
`ingest → harvest_cards → align_cards → machine_verify → merge → split →
finetune_whisper → eval` all run clean on real files.

**Dataset today: 479 verified clips ≈ 22.1 minutes.**

| tier | how it was verified | clips | minutes |
|---|---|---:|---:|
| `machine-card` | phrase-card aligned to audio | 260 | 8.9 |
| `machine-consensus` | two Gemini ears agreed → Claude adjudicated | 174 | 9.1 |
| `human` | pre-pivot hand-checked | 45 | 4.1 |

A further **651 clips (33.8 min)** sit `pending` — the ears disagreed, and they are
held for a later bootstrap round rather than force-labelled.

**A note on honesty:** during this work a whole source (193 clips / 10.3 min) that
*passed* machine-consensus turned out to be **Gujarati, not Kutchi** — consensus
verifies that two models *transcribe the same thing*, not that the thing is Kutchi.
It was quarantined, the inflated minute count was corrected downward, and
`machine_verify.py` gained a lexical language gate that now *holds* high-agreement
Gujarati-looking clips for review. This is the kind of failure a low-resource
pipeline must catch itself.

**Track 2 — LM: trained, and instructive.** See the numbers below.

**Not done:** a validated Kutchi STT model. The fine-tune has only been *smoke-run*
(whisper-tiny, ~18 min of audio) to prove the chain executes — it overfits and its
error rate is meaningless. **No CER is quoted as real accuracy**, because (a) there is
not yet enough data and (b) the eval set has not had its one-time human check. That is
a deliberate guardrail, not an oversight.

---

## How training / fine-tuning was done

### STT (Whisper)

Kutchi is written in Gujarati script, so we fine-tune Whisper with its `gu`
tokenizer/task: the model already renders the glyphs; we are teaching it Kutchi
phonetics and vocabulary. Training reads only `status == "verified"` rows and hard-
refuses any clip id in the frozen `eval_ids.txt`.

**Baseline (illustrative):** off-the-shelf `whisper-tiny` on held-out Kutchi clips
returns roughly **WER/CER > 100 %** — it hallucinates English (`"So, he is going to
say that"` for `તો ઇસકો કચ્છી મેં બોલેંગે`). That is the "before." There is no
credible "after" yet — that is what the next ~40 verified minutes are for.

### LM (from scratch → Gujarati → Kutchi)

Three checkpoints were actually trained on an M-series Mac (MPS, fp32):

**1. Character GPT on Kutchi alone** — 2.73 M params, ~23 K training characters. This
run *is* the lesson: validation loss bottoms out early (~3.05) then **climbs to 6.25**
as training loss collapses toward 0.12. Textbook overfitting on a corpus far too small
for a from-scratch model. It generates Kutchi-*shaped* text — real character
statistics, no meaning.

**2. GPT-2 (~50 M) pretrained on Gujarati Wikipedia** (~110 MB): a proof that the
architecture and data pipeline train a real model — validation loss `6.17 → 5.73`. It
is deliberately undertrained (a laptop, 1500 steps), not a competitive Gujarati LM.

**3. Continue-training that checkpoint on Kutchi** (`--init-from ckpt_gujarati`): this
is the low-resource move — specialize the Gujarati base on the tiny Kutchi corpus.
Validation loss drops to ~0.015, which on a single-batch validation set means the
model has largely *memorized* the corpus. A real sample from the Kutchi checkpoint:

```
મુકે કચ્છી હિન્દી. અનેૂ અચ હથ. પાકે તો સાથ કચ્છી ... પગ ભી હમ કચ્છી મેં ...
```

Fluent-looking Kutchi fragments with Hindi/English leakage — exactly what a
memorizing model on a Gujarati base produces. **The takeaway is the ceiling itself:**
a usable Kutchi LM is gated on Kutchi *text*, and right now transcription (Track 1) is
the only scalable source of it. Every verified audio hour is also an LM corpus hour.

---

## Reproduce

```bash
# --- STT pipeline (deps you install locally) ---
pip install yt-dlp faster-whisper torch transformers datasets soundfile librosa
python ingest.py --urls urls.txt --out ./data --model small        # audio → clips + drafts
python machine_verify.py run --manifest data/manifest.jsonl --clips data/clips  # needs GEMINI_API_KEY
python merge.py                                                     # → dataset/verified.jsonl
python split.py --data dataset/verified.jsonl --eval-minutes 90     # freeze eval + train split
python finetune_whisper.py --data dataset/train.jsonl \
    --clips-dir data/clips --model openai/whisper-small --epochs 8 --out ./model_out
python eval.py --model ./model_out --data dataset/eval.jsonl --clips-dir data/clips

# --- Kutchi GPT (lm/) ---
python lm/build_corpus.py --fetch-gujarati --max-mb 200
python lm/char_gpt.py --steps 3000                                  # watch it overfit — that's the point
python lm/train_gpt2.py --data gujarati --config mac-50m --steps 20000
python lm/train_gpt2.py --data kutchi --init-from data_lm/ckpt_gujarati.pt --steps 2000 --lr 6e-5
python lm/sample.py --ckpt data_lm/ckpt_kutchi.pt --prompt "મુકે"
```

Raw audio, checkpoints, and scraped corpora are **git-ignored** (see rights note).
The repo ships the code, the convention, and the method — not the private data.

---

## Data schema (`manifest.jsonl`, one JSON per line)

`id · clip · source_id · source_url · source_title · start · end · duration ·
asr_draft · asr_model · transcript · roman · status · notes · verified_by · confidence`

- `asr_draft` — machine guess (Gujarati), wrong by design; a correction seed.
- `transcript` — the deliverable: verified Kutchi in Gujarati script.
- `status` ∈ {pending, verified, skip, nonspeech}; training uses only `verified`.
- `verified_by` ∈ {human, machine-card, machine-consensus} — tiers are never
  conflated; higher tiers win on dedup.

---

## Locked design decisions

- **Audio-first.** Kutchi is oral; the corpus is audio, text is a byproduct.
- **Dialect: Bhuj** (core Kutch); other regions flagged, not silently mixed.
- **Script: modified Gujarati**, plus an optional romanized field.
- **Implosives written plain** (/ɓ/→બ, /ɗ/→ડ, no nukta); the implosive is captured
  only in the `roman` field. Consistency over phonetic precision.
- **Never train on the eval set;** an unreviewed eval means CER is not real accuracy.
- **Keep provenance on every row;** don't redistribute raw audio.

---

## Next steps

1. **Cross ~60 verified minutes** — card-style lesson videos are the highest-yield,
   zero-cost lever.
2. **Freeze the eval set and do the one-time listen-check** — this is what turns CER
   from a number into a *claim*.
3. **Bootstrap round 1** — fine-tune `whisper-small`, measure CER on frozen eval, then
   add that model as a third machine-verify ear and re-verify the 651 pending clips.
4. **Grow the loop** toward the 5–10 hr first-real-fine-tune milestone, then 50–100 hr
   for a usable STT.
5. **LM shallow-fusion (stretch):** use the Kutchi LM to rescore Whisper hypotheses,
   feeding language knowledge back into the verifier.

---

## Rights & ethics

Provenance is stored on every row for dedup and licensing. Raw audio stays private;
published artifacts are models and (where licensing allows) transcripts — not source
audio. New sources are sample-checked for actually being Kutchi before bulk processing.
The project is framed as **language preservation**, which it genuinely is: an
endangered oral language the census does not even count separately.

*If you speak Kutchi, or work on low-resource / Indic speech, I'd genuinely like to
compare notes.*
