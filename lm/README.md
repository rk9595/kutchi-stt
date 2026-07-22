# Build-a-GPT-for-Kutchi

A learning project: reproduce Karpathy's "GPT from scratch" builds, but for **Kutchi**.
The goal is understanding how a GPT works end-to-end — not shipping a product. Each file
maps to one of his videos and is written to be *read*.

## The honest data situation

Kutchi has almost no written text (that is this repo's whole premise). We have **~26 KB**
total — verified STT transcripts + phrasebook cards + scraped poetry/proverbs. For scale,
Karpathy's char demo uses 1.1 MB and his GPT-2 124M uses ~10 **billion** tokens. So:

- You **cannot** train a real GPT from scratch on Kutchi alone — it has more parameters
  than we have characters and would just memorize the corpus.
- The fix (how low-resource LMs are actually built): **pretrain on Gujarati** — same
  script, linguistically close, a real ~110 MB Wikipedia corpus — then **specialize on
  Kutchi**. Kutchi ends up a thin layer on a Gujarati base. That is the honest ceiling.

## The curriculum

| Stage | File | Karpathy video | What it teaches |
|------|------|----------------|-----------------|
| 0 | `build_corpus.py` | — | assemble Kutchi text; fetch Gujarati Wikipedia |
| 1 | `char_gpt.py` | *Let's build GPT* | attention, blocks, residual+LN, training loop, **overfitting** |
| T | `train_tokenizer.py` | *tokenizer / minbpe* | BPE, why byte-level BPE fails on Indic script |
| 2 | `gpt2.py` + `train_gpt2.py` | *Let's reproduce GPT-2 (124M)* | real GPT-2 arch + full training pipeline |
| — | `sample.py` | — | generate from a checkpoint |

## Run it

```bash
# Stage 0 — build the corpora (Kutchi from repo data; Gujarati from HF, ~110 MB)
python lm/build_corpus.py --fetch-gujarati --max-mb 200

# Stage 1 — from-scratch char GPT on Kutchi (minutes on an M1). Watch val loss bottom
# out and rise while train loss -> 0: that's overfitting on 26 KB, the whole lesson.
python lm/char_gpt.py --steps 3000

# Stage T — a Gujarati-aware BPE (Unicode-level, not byte-level; ~5 ch/tok on Gujarati)
python lm/train_tokenizer.py --vocab-size 16000

# Stage 2a — validate the architecture loads real GPT-2 (downloads ~500 MB once)
python lm/gpt2.py            # prints ' Paris' in the top-5 for "The capital of France is"

# Stage 2b — pretrain on Gujarati, then continue-train (specialize) on Kutchi
python lm/train_gpt2.py --data gujarati --config mac-50m --steps 20000
python lm/train_gpt2.py --data kutchi --init-from data_lm/ckpt_gujarati.pt --steps 2000 --lr 6e-5

# generate
python lm/sample.py --ckpt data_lm/ckpt_kutchi.pt --prompt "મુકે"
```

## Configs / compute

`--config` picks model shape (all identical to OpenAI's GPT-2, just scaled):

| name | layers | heads | n_embd | ~params | where |
|------|--------|-------|--------|---------|-------|
| `mac-tiny` | 6 | 6 | 384 | ~10M | M1 smoke |
| `mac-50m` | 8 | 8 | 512 | ~50M | M1 overnight (default) |
| `gpt2-124m` | 12 | 12 | 768 | 124M | rented GPU |

The Mac runs fp32 (MPS autocast/`torch.compile` are unreliable — guarded off). The literal
124M-on-10B-tokens run is the same code with `--config gpt2-124m` on a CUDA box; the only
knobs that change are model shape, corpus size, and where it runs.

## How this connects to the STT project

The speech pipeline (`ingest.py` → `machine_verify.py`/`align_cards.py` → `merge.py` →
`dataset/verified.jsonl`) turns Kutchi **audio** into Kutchi **text**. `build_corpus.py`
reads that file, so **every hour of audio we verify grows this LM's training corpus** —
transcription is currently the only scalable source of Kutchi text. All corpus text passes
through `lexicon.normalize()` so the LM obeys the same spelling convention as the STT data.

Stretch (not built): use the LM to rescore Whisper's hypotheses (shallow fusion) — a real
technique that would feed the LM's language knowledge back into the STT verifier.
