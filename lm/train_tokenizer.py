#!/usr/bin/env python3
"""
Stage T — a Gujarati-aware BPE tokenizer (Karpathy's tokenizer / minbpe lesson).

Why not just reuse GPT-2's tokenizer? GPT-2's BPE was learned on English bytes; on
Gujarati script every character falls back to 2-3 raw UTF-8 bytes, so a word becomes
a dozen meaningless tokens and the model wastes its context and capacity. A tokenizer
trained on our own script turns common Gujarati/Kutchi subwords into single tokens —
that is the whole point of the lesson.

A subtle trap this file exists to teach: **byte-level** BPE (what GPT-2 uses) is a bad
fit for Indic scripts. Each Gujarati codepoint is 3 UTF-8 bytes, so byte-BPE spends most
of its vocab just reassembling single characters before it can learn any subword — you
end up near character-level (~1.5 chars/token). We instead use a **Unicode-level** BPE
(SentencePiece-style Metaspace) whose base alphabet is characters, so the whole vocab
budget goes to real subwords. On Gujarati that jumps to ~4+ chars/token.

We train on the big Gujarati corpus plus the tiny Kutchi one, so Kutchi's frequent
pieces (આય, મુકે, તોંજો, -નૂ …) get tokens even though Kutchi is a rounding error.

Usage:
    python lm/train_tokenizer.py --vocab-size 16000
    # -> data_lm/tokenizer.json  (+ a round-trip / fragmentation report)
"""

import argparse
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_lm"

SPECIAL = ["<|endoftext|>", "<unk>"]
PROBE = "મુકે તોંજો નાલો કોર આય?"   # a Kutchi line to eyeball tokenization


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=16000)
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--out", default=str(DATA / "tokenizer.json"))
    args = ap.parse_args()

    files = [str(DATA / "kutchi.txt")]
    guj = DATA / "gujarati.txt"
    if guj.exists():
        files.insert(0, str(guj))
    else:
        print("[!] data_lm/gujarati.txt missing — training on Kutchi only "
              "(run build_corpus.py --fetch-gujarati for a real vocab)")

    # Unicode-level BPE with Metaspace (▁ marks word starts) — not byte-level.
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Metaspace()
    tok.decoder = decoders.Metaspace()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size, min_frequency=args.min_frequency,
        special_tokens=SPECIAL, show_progress=True)
    print(f"[i] training Unicode BPE vocab={args.vocab_size} on {files}")
    tok.train(files, trainer)
    tok.save(args.out)
    print(f"[saved] {args.out}  (vocab {tok.get_vocab_size()})")

    # --- the lesson: whole-subword tokens, not per-character/per-byte ---
    guj_probe = "ગુજરાત ભારત દેશનું એક રાજ્ય છે અને તેની રાજધાની ગાંધીનગર છે."
    print("\n--- tokenization probe ---")
    for label, s in (("gujarati", guj_probe), ("kutchi  ", PROBE)):
        enc = tok.encode(s)
        rt = tok.decode(enc.ids)
        print(f"{label}: {len(s):3d} chars -> {len(enc.ids):3d} tokens "
              f"({len(s)/len(enc.ids):.2f} ch/tok)  round-trip={'ok' if rt == s else 'FAIL'}")
    print("kutchi tokens:", tok.encode(PROBE).tokens)


if __name__ == "__main__":
    main()
