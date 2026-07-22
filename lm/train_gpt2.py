#!/usr/bin/env python3
"""
Stage 2 (training) — the GPT-2 training pipeline (Karpathy's build-nanogpt).

This is where the real-model machinery lives, each piece labelled:
  - a tokenized-shard data loader (tokenize once -> uint16 .npy cache)
  - AdamW with the weight-decay param split (in gpt2.py: configure_optimizers)
  - cosine learning-rate schedule with linear warmup
  - gradient accumulation (simulate a big batch on a small GPU/Mac)
  - gradient clipping at 1.0
  - autocast mixed precision (bf16/fp16 on CUDA; fp32 fallback on MPS/CPU)
  - optional torch.compile (off by default — flaky on MPS)

Two-phase curriculum for a low-resource language:
  1. pretrain on Gujarati (real corpus, model learns the script + grammar)
       python lm/train_gpt2.py --data gujarati --config mac-50m --steps 20000
  2. specialize on Kutchi (continue-train the checkpoint at a low LR)
       python lm/train_gpt2.py --data kutchi --init-from data_lm/ckpt_gujarati.pt \
              --steps 2000 --lr 6e-5
After a crash, resume the same run from its last checkpoint (restores optimizer + step):
       python lm/train_gpt2.py --data kutchi --steps 2000 --lr 6e-5 --resume
Then generate with lm/sample.py.
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from gpt2 import GPT, GPTConfig, build_config

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_lm"


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def tokenize_corpus(name, tok):
    """Tokenize data_lm/<name>.txt -> uint16 .npy (cached). Returns the token array."""
    cache = DATA / f"{name}.tokens.npy"
    if cache.exists():
        return np.load(cache)
    src = DATA / f"{name}.txt"
    lines = [ln for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip()]
    print(f"[tok] encoding {name}.txt ({len(lines)} lines) with encode_batch...")
    # encode_batch is multi-threaded in the Rust tokenizer — orders of magnitude
    # faster than one giant encode(); process in chunks to bound memory.
    eot = tok.token_to_id("<|endoftext|>")
    ids = []
    CHUNK = 50_000
    for i in range(0, len(lines), CHUNK):
        for enc in tok.encode_batch(lines[i:i + CHUNK]):
            ids.extend(enc.ids)
            if eot is not None:
                ids.append(eot)          # document/line boundary
        print(f"[tok]   {min(i+CHUNK,len(lines))}/{len(lines)} lines, "
              f"{len(ids)/1e6:.1f}M tokens", flush=True)
    arr = np.array(ids, dtype=np.uint16)
    np.save(cache, arr)
    print(f"[tok] {len(arr)/1e6:.2f}M tokens -> {cache}")
    return arr


class DataLoaderLite:
    """Streams contiguous (B, T) chunks through the token stream, wrapping at the end."""

    def __init__(self, tokens, B, T, split="train"):
        n = len(tokens)
        cut = int(0.98 * n)
        data = tokens[:cut] if split == "train" else tokens[cut:]
        # tiny corpora (the Kutchi finetune) can have a val slice smaller than one
        # batch — fall back to the whole stream so eval still produces a number.
        if len(data) < B * T + 1:
            data = tokens
        self.tokens = torch.from_numpy(data.astype(np.int64))
        self.B, self.T = B, T
        self.pos = 0
        print(f"[data] {split}: {len(self.tokens)/1e6:.2f}M tokens, "
              f"{len(self.tokens)//(B*T)} batches/epoch")

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.pos:self.pos + B * T + 1]
        if len(buf) < B * T + 1:                 # wrap around
            self.pos = 0
            buf = self.tokens[:B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.pos += B * T
        return x, y


def lr_at(step, warmup, max_steps, max_lr, min_lr):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step > max_steps:
        return min_lr
    ratio = (step - warmup) / (max_steps - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))     # cosine decay
    return min_lr + coeff * (max_lr - min_lr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["gujarati", "kutchi"], default="gujarati")
    ap.add_argument("--config", choices=["mac-tiny", "mac-50m", "gpt2-124m"], default="mac-50m")
    ap.add_argument("--tokenizer", default=str(DATA / "tokenizer.json"))
    ap.add_argument("--init-from", default="", help="checkpoint to warm-start weights from (fresh optimizer + step 0)")
    ap.add_argument("--resume", action="store_true",
                    help="resume from the output checkpoint: restore model, optimizer, and step")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.resume and args.init_from:
        ap.error("--resume and --init-from are mutually exclusive")
    out = args.out or str(DATA / f"ckpt_{args.data}.pt")

    device = pick_device()
    torch.manual_seed(1337)
    tok = Tokenizer.from_file(args.tokenizer)
    vocab_size = tok.get_vocab_size()

    tokens = tokenize_corpus(args.data, tok)
    train_loader = DataLoaderLite(tokens, args.batch_size, args.block_size, "train")
    val_loader = DataLoaderLite(tokens, args.batch_size, args.block_size, "val")

    resume_ckpt = None
    if args.resume:
        resume_ckpt = torch.load(out, map_location=device)
        # architecture must match the checkpoint, not the CLI defaults
        config = GPTConfig(**resume_ckpt["config"])
    else:
        config = build_config(args.config, vocab_size, args.block_size)
    model = GPT(config).to(device)
    if args.resume:
        model.load_state_dict(resume_ckpt["model"])
        print(f"[resume] loaded {out} at step {resume_ckpt.get('step', 0)} (config={config.n_layer}L/{config.n_embd}d)")
    elif args.init_from:
        ckpt = torch.load(args.init_from, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"[init] continued from {args.init_from}")
    n_params = sum(p.numel() for p in model.parameters())
    tokens_per_step = args.batch_size * args.block_size * args.grad_accum
    config_label = "resumed" if args.resume else args.config
    print(f"[i] device={device} config={config_label} params={n_params/1e6:.1f}M "
          f"vocab={vocab_size} tokens/step={tokens_per_step}")

    if args.compile:
        model = torch.compile(model)

    opt = model.configure_optimizers(args.weight_decay, args.lr, device)
    start_step = 0
    if resume_ckpt is not None:
        start_step = resume_ckpt.get("step", 0)
        if "opt" in resume_ckpt:
            opt.load_state_dict(resume_ckpt["opt"])
        else:
            print("[resume] checkpoint has no optimizer state — restarting optimizer fresh")
    # mixed precision: bf16 on CUDA (Ampere+), else run fp32 (MPS autocast is unreliable)
    use_amp = device == "cuda"
    min_lr = args.lr * 0.1

    for step in range(start_step, args.steps + 1):
        t0 = time.time()
        if step % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                losses = []
                for _ in range(20):
                    xb, yb = val_loader.next_batch()
                    _, l = model(xb.to(device), yb.to(device))
                    losses.append(l.item())
            print(f"step {step:6d}  val_loss {sum(losses)/len(losses):.4f}")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "config": config.__dict__, "step": step, "data": args.data}, out)
            model.train()

        lr = lr_at(step, args.warmup, args.steps, args.lr, min_lr)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(args.grad_accum):
            xb, yb = train_loader.next_batch()
            xb, yb = xb.to(device), yb.to(device)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, loss = model(xb, yb)
            else:
                _, loss = model(xb, yb)
            loss = loss / args.grad_accum
            loss_accum += loss.item()
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()

        if step % 50 == 0:
            dt = time.time() - t0
            tps = tokens_per_step / dt
            print(f"step {step:6d}  loss {loss_accum:.4f}  lr {lr:.2e}  "
                  f"norm {norm:.2f}  {dt*1000:.0f}ms  {tps:.0f} tok/s")

    print(f"[done] checkpoint -> {out}")


if __name__ == "__main__":
    main()
