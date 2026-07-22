#!/usr/bin/env python3
"""
Stage 2 (architecture) — GPT-2, reproduced.

This is the model from Karpathy's "Let's reproduce GPT-2 (124M)", written to be exactly
the GPT-2 architecture so it can *load the real OpenAI GPT-2 weights* — that round-trip is
the correctness test that our attention/MLP/embedding wiring is right (from_pretrained()).

Differences from the char model in char_gpt.py, and why they matter:
  - a real (BPE) vocab via a trained tokenizer, not characters       (lm/train_tokenizer.py)
  - learned positional embeddings capped at block_size=1024
  - attention via F.scaled_dot_product_attention (flash) not a python softmax
  - GELU (tanh approx, as GPT-2) instead of ReLU
  - weight tying: token embedding and output projection share weights
  - GPT-2 init: std 0.02, and residual projections scaled by 1/sqrt(2*n_layer)

Configs (identical shape to OpenAI's, we just scale down for the Mac):
  gpt2-124m : 12 layer / 12 head / 768  (the real thing — needs a GPU to train)
  mac-50m   : 8  layer / 8  head / 512  (default; trains on M1 in hours)
  mac-tiny  : 6  layer / 6  head / 384  (fast smoke)
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 16000     # our Gujarati/Kutchi BPE (50257 for real GPT-2)
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512


CONFIGS = {
    "gpt2-124m": dict(n_layer=12, n_head=12, n_embd=768),
    "mac-50m":   dict(n_layer=8,  n_head=8,  n_embd=512),
    "mac-tiny":  dict(n_layer=6,  n_head=6,  n_embd=384),
}


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)   # q,k,v in one matmul
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.RESIDUAL_SCALE = True
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.RESIDUAL_SCALE = True

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight     # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if getattr(module, "RESIDUAL_SCALE", False):
                std *= (2 * self.config.n_layer) ** -0.5      # GPT-2 residual scaling
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, f"seq len {T} > block {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, lr, device):
        # weight-decay only tensors that are matmuls/embeddings (dim >= 2); not biases/norms
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [{"params": decay, "weight_decay": weight_decay},
                  {"params": no_decay, "weight_decay": 0.0}]
        fused = device == "cuda"
        return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8,
                                 fused=fused)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=50):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    @classmethod
    def from_pretrained(cls, model_type="gpt2"):
        """Load OpenAI GPT-2 weights from HuggingFace — the architecture correctness test."""
        from transformers import GPT2LMHeadModel
        sizes = {"gpt2": dict(n_layer=12, n_head=12, n_embd=768)}[model_type]
        config = GPTConfig(vocab_size=50257, block_size=1024, **sizes)
        model = cls(config)
        sd = model.state_dict()
        keys = [k for k in sd if not k.endswith(".attn.bias")]

        hf = GPT2LMHeadModel.from_pretrained(model_type)
        hf_sd = hf.state_dict()
        # HF Conv1D layers are transposed relative to our Linear layers
        transposed = ["attn.c_attn.weight", "attn.c_proj.weight",
                      "mlp.c_fc.weight", "mlp.c_proj.weight"]
        hf_keys = [k for k in hf_sd if not k.endswith(".attn.masked_bias")
                   and not k.endswith(".attn.bias")]
        assert len(hf_keys) == len(keys), f"{len(hf_keys)} vs {len(keys)}"
        for k in hf_keys:
            if any(k.endswith(t) for t in transposed):
                assert hf_sd[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(hf_sd[k].t())
            else:
                assert hf_sd[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(hf_sd[k])
        return model


def build_config(name, vocab_size, block_size):
    cfg = CONFIGS[name]
    return GPTConfig(vocab_size=vocab_size, block_size=block_size, **cfg)


if __name__ == "__main__":
    # correctness test: our from-scratch arch loads real GPT-2 weights and produces
    # sensible next-token predictions — proves the wiring matches OpenAI's GPT-2.
    from transformers import GPT2Tokenizer
    torch.manual_seed(0)
    model = GPT.from_pretrained("gpt2").eval()
    enc = GPT2Tokenizer.from_pretrained("gpt2")
    ids = torch.tensor([enc.encode("The capital of France is")], dtype=torch.long)
    with torch.no_grad():
        logits, _ = model(ids)
    top = torch.topk(logits[0, -1], 5).indices.tolist()
    print("prompt: 'The capital of France is'")
    print("top-5 next tokens:", [repr(enc.decode([t])) for t in top])
