# Teaching machines a language that was never written down

*Building the first speech dataset and language-model pipeline for Kutchi — a research log, Phase 1.*

---

Kutchi (કચ્છી) is my mother tongue. A few million people speak it — across Kutch, all
over Mumbai, and in a diaspora that reaches East Africa, the Gulf, and California.
India's census does not count it as a language; it is filed under Gujarati. It has no
official status, no standardized spelling, and — the fact that reorganized this entire
project — **almost no written text.** Kutchi is spoken, not written.

Every modern NLP recipe assumes the opposite. You scrape a few billion tokens, you
train, you fine-tune. For Kutchi there is nothing to scrape. So before any model, the
real problem is manufacturing supervision: verified pairs of *(audio, Kutchi text)*,
and enough clean Kutchi text to even attempt a language model.

This is a log of Phase 1. It is not a finished product, and I want to be precise about
what works, what doesn't, and why the things that don't work are the interesting part.

## The thesis: the dataset is the moat

Whisper, GPT-2, IndicWhisper — the architectures are commodities, downloadable for
free. What does not exist anywhere is Kutchi supervision. So I stopped measuring
progress in "models trained" and started measuring it in one number: **verified
minutes of audio.** Everything is built to grow that number cheaply, and — just as
important — to never lie to itself about how much real signal it has.

## Track 1 — manufacturing a speech dataset with no annotators

The original plan used native-speaker annotators. I pivoted away from that to a fully
automated pipeline, for a simple reason: a solo project can't sustain a labelling team,
and consistency is easier to enforce in code than across people. The pipeline has two
ways to produce a verified transcript, neither of which involves a human transcribing
from scratch.

**Card alignment.** Kutchi-teaching videos on YouTube put each phrase on screen — an
English / Hindi / romanized-Kutchi card. A native teacher has already written the
phrase; I just have to find *where in the audio they say it.* A harvester reads the
cards off the video frames; an aligner reduces both the card phrase and the ASR word
stream to a coarse phonetic skeleton, fuzzy-matches them, and cuts the exact utterance.
The result is a verified `(audio, text)` pair with zero transcription. I call this tier
`machine-card`.

**Machine consensus.** For conversational audio (podcasts, comedy, vlogs) there is no
on-screen text. Here each clip is transcribed by two independent machine "ears" —
audio-native models — and I score how much they agree by character-error-rate. When
they agree, the clip goes to an adjudication step where Claude picks the final
transcript against a written spelling convention. Disagreements are *not* forced; they
stay `pending` for a later round. This tier is `machine-consensus`.

A single rule governs all of it: **consistency beats correctness.** Every transcript
passes through a spelling convention encoded as code (`lexicon.py`), so the machine
makes the identical orthographic choice every time — implosives written plain, one
script, punctuation normalized. Verified text is never silently re-spelled.

### The bug that matters more than the pipeline

Partway through, my dataset said 30-plus verified minutes. Then I sampled it and found
a whole source — 193 clips, over 10 minutes — that had sailed through machine consensus
and was **Gujarati, not Kutchi.**

This is the low-resource trap in one incident. Consensus checks that two models
*transcribe the same thing*; it says nothing about *which language that thing is.* And
Kutchi is routinely mislabeled as Gujarati because they share a script and much
vocabulary. Two models confidently agreeing on a Gujarati sentence looks exactly like
success.

I quarantined the source, corrected the inflated count downward (the honest number was
lower than what I'd been telling myself), and added a lexical language gate that now
*holds* high-agreement Gujarati-looking clips for review instead of trusting them.
A dataset you don't audit is a dataset that flatters you.

**Where Track 1 stands:** 479 verified clips, about **22.1 minutes**, across three
tiers (card-aligned, machine-consensus, and legacy human-checked). Another ~34 minutes
sit pending because the models disagreed. The full chain — ingest → verify → merge →
freeze eval → fine-tune Whisper → score — runs clean end to end.

**What I am deliberately *not* claiming:** a working Kutchi speech recognizer. The
fine-tune has only been smoke-run to prove the plumbing; with ~18 minutes of audio it
overfits, and its error rate is meaningless. Off-the-shelf Whisper, for reference,
scores worse than 100% error on Kutchi and hallucinates English — it transcribes
*"so, he is going to say that"* for a sentence that means *"we'll say this in Kutchi."*
There is a real "before." There is not yet a credible "after," and I won't quote one
until the dataset is bigger and the evaluation set has had its one manual check. That
check is the single human step I've kept, precisely because you cannot honestly report
accuracy against a reference no human has ever heard.

## Track 2 — building a Kutchi GPT, and hitting the ceiling on purpose

Alongside the speech work I built a small language-model track — a from-scratch "build
a GPT" curriculum in the spirit of Karpathy's videos, but for Kutchi. It doubles as an
honest demonstration of exactly how far the text takes you.

**A character GPT trained on Kutchi alone** (2.7M parameters, ~23K characters of text)
is the cleanest lesson in the whole repo. Its validation loss drops nicely, bottoms out
around step 500 — and then *climbs*, from 3.05 up past 6.0, while training loss
collapses toward zero. That divergence is the model memorizing a corpus far too small
for it. It ends up generating text with correct Kutchi *character statistics* and no
meaning at all. This is not a failure to fix; it is the point, rendered as a loss curve.

So how do you actually build a language model for a language with tens of kilobytes of
text? You don't start from scratch. **You pretrain on a close, high-resource
neighbor.** Gujarati shares Kutchi's script and much of its structure, and it has a
real ~110 MB Wikipedia corpus. So I trained a ~50M-parameter GPT-2 on Gujarati (a
laptop-scale, deliberately undertrained run — validation loss 6.17 → 5.73), then
**continue-trained that checkpoint on the tiny Kutchi corpus.** Kutchi becomes a thin
adapter on a Gujarati base.

The Kutchi checkpoint's validation loss falls to ~0.015 — which, on a validation set
this small, means it has largely memorized. Here is a real, unedited sample:

> મુકે કચ્છી હિન્દી. અનેૂ અચ હથ. પાકે તો સાથ કચ્છી ... પગ ભી હમ કચ્છી મેં ...

Fluent-*looking* Kutchi fragments, bleeding into Hindi and English. That is precisely
what a memorizing model sitting on a Gujarati base produces. The value isn't the model
— it's that the ceiling is now visible and measurable. A usable Kutchi LM is gated on
Kutchi *text*, and right now transcription is the only scalable way to produce it.
Which closes the loop: **every hour of audio I verify in Track 1 is also an hour of
corpus for Track 2.**

## What Phase 1 actually taught me

- **In low-resource NLP, the model is the easy part.** The hard, uncelebrated work is
  manufacturing and auditing supervision.
- **Automated verification needs a language gate, not just an agreement gate.**
  Two models agreeing is necessary, not sufficient — especially between sibling
  languages that share a script.
- **Honest evaluation is a design constraint, not an afterthought.** Freeze the eval
  set on day one, keep one human check, and refuse to quote accuracy you can't stand
  behind.
- **A related high-resource language is the cheapest data you have.** Gujarati
  pretraining is the only reason a Kutchi LM is even conceivable today.

## Next steps

Cross 60 verified minutes (card-style lesson videos are the highest-yield input),
freeze and human-check the eval set, then run the first real fine-tune of
`whisper-small` and measure character-error-rate honestly. After that, the loop closes:
the fine-tuned model becomes a third "ear" in the verifier, re-scores the pending
clips, and each round makes the next one cheaper. The milestone that matters is 5–10
verified hours — the first point where a Kutchi speech recognizer stops being a
demo and starts being real.

If you speak Kutchi, or you work on low-resource or Indic speech, I'd genuinely like to
compare notes.

---

*Code and method (not the private audio) are open: the pipeline, the spelling
convention as code, and the from-scratch LM curriculum.*
