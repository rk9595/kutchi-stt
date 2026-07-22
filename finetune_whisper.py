#!/usr/bin/env python3
"""
Kutchi Phase 1 — Whisper fine-tune (SMOKE HARNESS first, real fine-tune later).

Point it at one or more manifest jsonl files; it trains on rows with
status == "verified" only. Kutchi is written in Gujarati script, so we fine-tune
with the Whisper 'gu' tokenizer/task — the model already knows the glyphs, we're
teaching it Kutchi phonetics/vocabulary.

The point of the smoke run is to prove the chain works end to end:
    verified.jsonl -> input features -> train -> saved model -> eval.py
With ~4 min of audio it WILL overfit and be useless as a model. That's expected.
Record ~30-60 min with record.py before reading anything into the WER.

GUARDRAIL (CLAUDE.md): never train on the eval set. Any id listed in --eval-ids
(default dataset/eval_ids.txt if present) is dropped from training, loudly.

Usage:
    # smoke test on what you have today (tiny model, CPU-friendly):
    python finetune_whisper.py \
        --data dataset/verified.jsonl \
        --clips-dir data/clips --clips-dir data_smoke/clips \
        --model openai/whisper-tiny --epochs 20 --out ./model_out

    # after recording yourself:
    python finetune_whisper.py --data data_self/manifest.jsonl \
        --clips-dir data_self/clips --model openai/whisper-small --epochs 8

Install:
    pip install torch transformers datasets soundfile librosa
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


def resolve_clip(clip, clips_dirs):
    for d in clips_dirs:
        p = Path(d) / clip
        if p.exists():
            return p
    return None


def load_rows(data_paths, clips_dirs, eval_ids):
    rows, missing, excluded = [], 0, 0
    for dp in data_paths:
        with open(dp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("status") != "verified":
                    continue
                if not (r.get("transcript") or "").strip():
                    continue
                if r["id"] in eval_ids:
                    excluded += 1
                    continue
                p = resolve_clip(r["clip"], clips_dirs)
                if p is None:
                    missing += 1
                    continue
                rows.append({"id": r["id"], "path": str(p),
                             "text": r["transcript"].strip()})
    if excluded:
        print(f"[guard] excluded {excluded} eval-set clip(s) from training.")
    if missing:
        print(f"[!] {missing} verified row(s) skipped — clip not found in --clips-dir.")
    return rows


@dataclass
class Collator:
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        import torch
        inp = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(inp, return_tensors="pt")
        labels = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features], return_tensors="pt")
        lab = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        # drop the forced BOS the processor prepends; the model adds it back
        if (lab[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            lab = lab[:, 1:]
        batch["labels"] = lab
        return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="append", required=True,
                    help="manifest jsonl (repeatable)")
    ap.add_argument("--clips-dir", action="append", required=True,
                    help="dir to search for clip files (repeatable)")
    ap.add_argument("--eval-ids", default="dataset/eval_ids.txt",
                    help="locked eval-id file to EXCLUDE from training")
    ap.add_argument("--model", default="openai/whisper-tiny")
    ap.add_argument("--out", default="./model_out")
    ap.add_argument("--epochs", type=float, default=20.0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    args = ap.parse_args()

    try:
        import torch
        import librosa
        from transformers import (WhisperProcessor,
                                  WhisperForConditionalGeneration,
                                  Seq2SeqTrainer, Seq2SeqTrainingArguments)
    except ImportError:
        sys.exit("Install deps first:  pip install torch transformers datasets soundfile librosa")

    eval_ids = set()
    if Path(args.eval_ids).exists():
        eval_ids = {l.strip() for l in Path(args.eval_ids).read_text().splitlines() if l.strip()}
        print(f"[guard] loaded {len(eval_ids)} locked eval id(s) from {args.eval_ids}")

    rows = load_rows(args.data, args.clips_dir, eval_ids)
    if not rows:
        sys.exit("[!] no trainable verified rows found. Nothing to do.")
    print(f"[i] training on {len(rows)} verified clip(s) with '{args.model}'")

    processor = WhisperProcessor.from_pretrained(args.model, language="gu", task="transcribe")

    def prepare(row):
        audio, _ = librosa.load(row["path"], sr=16000, mono=True)
        feat = processor.feature_extractor(audio, sampling_rate=16000).input_features[0]
        labels = processor.tokenizer(row["text"]).input_ids
        return {"input_features": feat, "labels": labels}

    from datasets import Dataset
    ds = Dataset.from_list(rows).map(prepare, remove_columns=["id", "path", "text"])
    # Whisper's decoder is capped at 448 tokens; drop rows whose labels exceed it
    # (long clips, or Gujarati byte-fallback on smaller models, blow past this).
    max_labels = 448
    before = len(ds)
    ds = ds.filter(lambda x: len(x["labels"]) <= max_labels)
    if len(ds) < before:
        print(f"[!] dropped {before - len(ds)} row(s) with > {max_labels} label tokens.")

    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.generation_config.language = "gu"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    targs = Seq2SeqTrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        fp16=torch.cuda.is_available(),
        use_cpu=not torch.cuda.is_available(),  # MPS is pathologically slow for Whisper; use CPU
        dataloader_num_workers=0,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Seq2SeqTrainer(
        model=model, args=targs, train_dataset=ds,
        data_collator=Collator(processor),
    )
    trainer.train()
    trainer.save_model(args.out)
    processor.save_pretrained(args.out)
    print(f"[done] model saved to {args.out}")
    print(f"[next] python eval.py --model {args.out} --data <holdout.jsonl> --clips-dir ...")


if __name__ == "__main__":
    main()
