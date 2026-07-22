# X thread — draft

Honest research-log framing. ~10 posts. Swap in your repo/blog links where marked.
Attach media where noted (loss curve, a clip, a sample screenshot) — threads with a
visual on post 1 travel much further.

---

**1/**
Kutchi is my mother tongue. A few million speakers, huge presence in Mumbai + a global
diaspora — and almost no written text. India's census doesn't even count it as a
language.

So I started building AI for it. Phase 1, honestly documented. 🧵

*(media: a Kutchi clip + its transcript, or the repo card)*

---

**2/**
The problem flips every NLP assumption. You can't scrape a Kutchi corpus because one
barely exists. Kutchi is *oral*.

So before any model, the real work is manufacturing supervision: verified pairs of
(audio → Kutchi text).

The dataset is the moat. The models are commodities.

---

**3/**
My unit of progress isn't "models trained." It's **verified minutes of audio.**

Everything is built to grow that number cheaply — and to never lie to itself about how
much real signal it has.

---

**4/**
The pipeline has NO human annotators. Two ways to get a verified transcript:

① Card alignment: Kutchi-lesson videos show each phrase on screen. A native teacher
already wrote it — I just find where they *say* it in the audio and cut that clip.

② Machine consensus: two audio models transcribe a clip; if they agree, Claude
adjudicates the final text.

---

**5/**
The most important thing I built wasn't the pipeline. It was catching this:

A whole source — 193 clips, 10+ min — passed machine consensus and was **Gujarati, not
Kutchi.**

Consensus checks that two models transcribe the SAME thing. Not that it's the RIGHT
language. Sibling languages sharing a script break naive agreement gates.

---

**6/**
Fix: quarantine the source, correct the inflated count *down* (the honest number was
lower than what I'd told myself), add a lexical language gate that HOLDS
Gujarati-looking clips for review.

A dataset you don't audit is a dataset that flatters you.

---

**7/**
I also built a Kutchi GPT from scratch — and the lesson is a loss curve.

A char-GPT on ~23K characters of Kutchi: val loss drops, then *climbs* while train loss
→ 0. Textbook memorization. It generates Kutchi-shaped text with zero meaning.

That's not a bug to fix. That's the ceiling, rendered honestly.

*(media: the val-loss-climbing curve)*

---

**8/**
So how do you build an LM for a language with tens of KB of text?

You don't start from scratch. You pretrain on a close high-resource neighbor —
**Gujarati** (same script, real 110MB corpus) — then specialize on Kutchi.

Kutchi becomes a thin adapter on a Gujarati base. Real sample:

મુકે કચ્છી હિન્દી... પગ ભી હમ કચ્છી મેં...

Fluent-looking, leaking into Hindi/English. Exactly what a memorizing model on a
Gujarati base produces.

---

**9/**
What I'm NOT claiming: a working Kutchi speech recognizer.

The fine-tune is smoke-tested only — 18 min of audio, it overfits, error rate is
meaningless. I won't quote accuracy until the data is bigger AND the eval set has had
its one human listen-check.

Honest eval is a design constraint, not an afterthought.

---

**10/**
Phase 1 takeaways:

• In low-resource NLP the model is the easy part; manufacturing + auditing supervision
is the hard part.
• Agreement gates need a language gate too.
• A related high-resource language is your cheapest data.

Every audio hour I verify is also LM corpus. The loop closes.

---

**11/**
Next: cross 60 verified min → freeze + human-check eval → first real whisper-small
fine-tune → that model becomes a 3rd "ear" and re-verifies the backlog. Each round gets
cheaper.

Code + method are open (private audio isn't): [REPO LINK]
Full writeup: [BLOG LINK]

If you speak Kutchi or work on Indic/low-resource speech — let's talk. 🙏
