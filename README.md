# Kutchi LLM — Phase 1: Data

The whole project is bottlenecked here. Models are commodities; a verified Kutchi
dataset is the moat. Phase 1 is **audio-first** because Kutchi is an oral language
and real Kutchi *text* barely exists (even "Kutchmitra" is a Gujarati paper).

The loop: machine pre-transcribes → a **native speaker corrects** → you get
`(audio, verified Kutchi text)` pairs. Correcting a draft is 3–5× faster than
transcribing from scratch, which is what makes a solo/small-team effort feasible.

---

## Locked assumptions (change these on purpose, not by accident)

- **Canonical script: modified Gujarati script**, with an optional romanized field.
  Reason: most Kutchi readers in India read Gujarati script, Whisper transcribes
  Gujarati well (free drafts), and you can reuse Gujarati tokenizers/tooling.
- **First milestone: a usable Kutchi STT**, because it's the long pole and unlocks
  the voice-first v1. Translation/chat data comes after.
- **Unit of progress: verified MINUTES of audio**, not file count. The desk shows this.

---

## The two tools

**`ingest.py`** — feed it URLs, it produces `clips/*.wav` + `manifest.jsonl` with a
rough Gujarati-script draft per clip.

```bash
pip install yt-dlp faster-whisper          # + ffmpeg on PATH
python ingest.py --urls urls.txt --out ./data --model small
# GPU: add nothing (auto-detected). Bigger drafts: --model medium / large-v3
```

**`transcribe_correct.html`** — open in any browser. Load `manifest.jsonl` + the
`clips/` folder. Native speaker corrects each clip. Autosaves to the browser.
Export `kutchi_verified.jsonl` when done. Host it on GitHub Pages to share with
remote annotators (each person's work saves in *their* browser; they export and
send you the JSONL; you merge).

Keyboard: `Space` play · `R` replay · `⌘/Ctrl+Enter` verify · `S` skip · `M` music · `[` `]` move.

---

## Data schema (`manifest.jsonl`, one JSON object per line)

| field | meaning |
|---|---|
| `id` | unique clip id |
| `clip` | wav filename, relative to `clips/` |
| `source_id` / `source_url` / `source_title` | provenance (keep it — you'll need it for rights + dedup) |
| `start` / `end` / `duration` | seconds within the source |
| `asr_draft` | machine guess (Gujarati). **Wrong**, just a starting point |
| `asr_model` | which model made the draft |
| `transcript` | **the deliverable** — verified Kutchi, Gujarati script |
| `roman` | optional romanized form (how people type on phones) |
| `status` | `pending` → `verified` / `skip` / `nonspeech` |
| `notes` | annotator notes (multiple speakers, noise, uncertain word) |

---

## Source inventory (audio = the corpus)

Start broad, pull a lot, let annotators discard. Prioritize **clear single-speaker
speech** over music — songs are hard for ASR.

**Best for STT (speech-forward):**
- Kutchi comedy / skit channels (e.g. "Kutchi Comedy" Bhuj, AJ Brothers) — conversational, clean
- Kutchi vlogs, interviews, news-talk, and **films/short films** (Kutchi Films playlists)
- Kala Varso (@kalavarso) — folk *storytelling* (the spoken parts), Kutch heritage
- Religious discourse / katha / pravachan in Kutchi (long, single-speaker, clean — gold for STT)
- Kutchi YouTube Shorts / Reels creators (search the spelling variants below)

**Use sparingly (music — good for TTS prosody later, weak for STT):**
- Kutchi lokgeet / folk-song collections (Rajshri Gujarati "Kutch Vatanji Vani", "Kutch Te Watha", etc.)

**Search spelling variants** (the language has no standard romanization, so each
spelling surfaces different creators): `kutchi`, `kutchhi`, `kachchhi`, `kachhi`,
`katchi`, `cutchi`, `kutch`, `કચ્છી`, `کچھی`. Add `comedy / vlog / interview /
katha / news / film`.

**Text (scarce — collect, but don't expect volume):**
- Kutchi Sahitya Academy publications; *Vadhod* magazine; community sites
  (kutchimaadu.com, khojawiki.org has vocab/phrase lists)
- Kutchi-language *columns/sections* inside otherwise-Gujarati outlets
- Diaspora orgs with written material (e.g. Kutchi Cultural Association, Sacramento)

**Rights note:** scraping for a research/dataset is one thing; redistributing the
audio is another. Store provenance now, keep clips private, and for anything you
later publish, prefer CC / public-domain / explicitly-permitted sources or get
consent. This matters more once it's a product.

---

## The thing that actually gates you: native speakers

You're in Mumbai — one of the largest Kutchi populations anywhere. This is your
unfair advantage. Without 3–5 reliable native correctors, Phase 1 doesn't move.

- Approach the **Kutchi Sahitya Academy**, Kutchi Visa Oswal / Bhatia / Lohana /
  Memon community associations, and Kutch-region colleges.
- Pitch it as language *preservation*, which it genuinely is — the census doesn't
  even list Kutchi as independent. That framing recruits volunteers a paycheck won't.
- Pay per verified minute if you can; it sets a clean quality incentive.
- Recruit at least **two** speakers per dialect region you target (Bhuj vs Mandvi vs
  Mundra etc. differ) so you can spot-check agreement.

---

## Targets (don't boil the ocean)

| Milestone | Verified audio | What it gets you |
|---|---|---|
| Proof of life | **5–10 hrs** | fine-tune Whisper/IndicWhisper → first Kutchi STT that sort-of works |
| Usable STT | **50–100 hrs** | a demoable voice-in component for v1 |
| Good STT | **300+ hrs** | production-grade; also enough to anchor everything else |
| TTS seed | **3–5 hrs single clean speaker** | one studio voice for Kutchi TTS |

At a realistic ~5–8× real-time correction speed, 50 hrs of verified audio ≈ a few
hundred annotator-hours. With 4 people that's weeks, not years — *if* recruiting works.

---

## QC (build this in from clip #1, not later)

- **Double-annotate ~5%** of clips and measure agreement. Disagreement = ambiguous
  audio or inconsistent spelling convention → fix the convention, document it.
- Keep a **one-page spelling guide** (which Gujarati-script choices map to which
  Kutchi sounds). Inconsistent spelling silently wrecks downstream training.
- Hold out a **frozen eval set** (~1–2 hrs) from day one. Never train on it. It's
  how you'll honestly answer "is the STT actually getting better?"

---

## Next after Phase 1 has ~10 hrs verified
Fine-tune `whisper-large-v3` (or AI4Bharat IndicWhisper / Meta MMS) on
`kutchi_verified.jsonl`, evaluate on the frozen set, and you have a first STT —
the first real Kutchi-specific artifact in the whole stack.
