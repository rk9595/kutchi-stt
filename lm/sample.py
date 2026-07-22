#!/usr/bin/env python3
"""
Generate text from a trained GPT-2 checkpoint (lm/train_gpt2.py output).

    python lm/sample.py --ckpt data_lm/ckpt_kutchi.pt --prompt "મુકે" --n 200
"""

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from gpt2 import GPT, GPTConfig

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_lm"


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(DATA / "ckpt_kutchi.pt"))
    ap.add_argument("--tokenizer", default=str(DATA / "tokenizer.json"))
    ap.add_argument("--prompt", default="")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--samples", type=int, default=3)
    args = ap.parse_args()

    device = pick_device()
    tok = Tokenizer.from_file(args.tokenizer)
    ckpt = torch.load(args.ckpt, map_location=device)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[i] {args.ckpt} (step {ckpt.get('step','?')}, trained on {ckpt.get('data','?')})")

    start = tok.encode(args.prompt).ids if args.prompt else [tok.token_to_id("<|endoftext|>") or 0]
    for s in range(args.samples):
        idx = torch.tensor([start], dtype=torch.long, device=device)
        out = model.generate(idx, args.n, temperature=args.temperature, top_k=args.top_k)
        print(f"\n--- sample {s+1} ---")
        print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
