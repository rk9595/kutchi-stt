#!/usr/bin/env python3
"""
Stage 1 — a GPT built from scratch, character-level, on Kutchi.

This is Karpathy's "Let's build GPT: from scratch, in code, spelled out" adapted to
Kutchi. Everything is here and readable: a character tokenizer, a single attention
head, multi-head attention, the feed-forward, a Transformer block with residual
connections + pre-LayerNorm, the full model, the training loop, and generation.

The corpus is ~25 KB of Kutchi (run lm/build_corpus.py first). That is tiny — the
model WILL overfit, and watching train-loss keep falling while val-loss turns back up
is the point: it teaches you what a train/val split, model size, and dropout are for.
Nothing here is Kutchi-specific except the data; that is the lesson.

Usage:
    python lm/char_gpt.py                 # train, print losses, then sample
    python lm/char_gpt.py --sample-only --ckpt data_lm/char_gpt.pt
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data_lm" / "kutchi.txt"

# ---- hyperparameters (small: a few M params, fits the tiny corpus) ---------
BLOCK_SIZE = 128        # context length in characters
N_EMBD = 192
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.2
BATCH_SIZE = 32
LR = 3e-4


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)              # (B, T, head_size)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5   # scaled scores
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # causal mask
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ self.value(x)   # (B, T, head_size)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.ReLU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Communication (attention) followed by computation (MLP), both residual."""

    def __init__(self):
        super().__init__()
        head_size = N_EMBD // N_HEAD
        self.sa = MultiHeadAttention(N_HEAD, head_size)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.ln2 = nn.LayerNorm(N_EMBD)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))     # pre-norm residual
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_embedding(idx)                              # (B,T,C)
        pos = self.position_embedding(torch.arange(T, device=idx.device))  # (T,C)
        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                     # (B,T,vocab)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]           # crop to context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                 # last time step
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def load_data(device):
    text = CORPUS.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join(itos[i] for i in l)
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n].to(device), data[n:].to(device), chars, encode, decode


def get_batch(data):
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data, iters=50):
    out = {}
    model.eval()
    for name, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(iters)
        for k in range(iters):
            xb, yb = get_batch(data)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--eval-every", type=int, default=300)
    ap.add_argument("--ckpt", default=str(ROOT / "data_lm" / "char_gpt.pt"))
    ap.add_argument("--sample-only", action="store_true")
    ap.add_argument("--gen-chars", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    train_data, val_data, chars, encode, decode = load_data(device)
    vocab_size = len(chars)
    print(f"[i] device={device}  vocab={vocab_size}  "
          f"train={len(train_data)} val={len(val_data)} chars")

    model = GPTLanguageModel(vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[i] model: {n_params/1e6:.2f}M params "
          f"(n_layer={N_LAYER} n_head={N_HEAD} n_embd={N_EMBD} block={BLOCK_SIZE})")

    if args.sample_only:
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        for step in range(args.steps + 1):
            if step % args.eval_every == 0:
                l = estimate_loss(model, train_data, val_data)
                gap = l["val"] - l["train"]
                print(f"step {step:5d}  train {l['train']:.4f}  val {l['val']:.4f}"
                      f"  gap {gap:+.4f}")
            xb, yb = get_batch(train_data)
            _, loss = model(xb, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        torch.save(model.state_dict(), args.ckpt)
        print(f"[saved] {args.ckpt}")

    print("\n--- sample ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, args.gen_chars)[0].tolist()))


if __name__ == "__main__":
    main()
