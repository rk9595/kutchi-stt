# Kutchi Spelling & Transcription Convention — Bhuj dialect, Gujarati script

> **Executable form:** `data/lexicon.tsv` + `lexicon.py` encode the logged-decisions
> table below as code; every machine-written transcript passes through
> `Lexicon.normalize()`. When a new decision lands in the table here, add the same
> row to lexicon.tsv (both are append-only). Since the 2026-07-19 automation pivot
> there is no human annotator lead — the lexicon is the ratifier; genuinely open
> calls go to the project owner.

**Version 0.1 (draft — must be ratified by the native-speaker lead).**
Target: Bhuj-region Kutchi, written in modified Gujarati script, for the STT/LLM dataset.

> **Before you use this:** ask the **Kutchi Sahitya Academy** whether they already
> have a spelling standard. If they do, adopt theirs and discard this. Don't invent a
> competing one. This sheet exists only so that, absent a standard, every annotator
> makes the *same* choice every time.

---

## The one rule above all others

**Consistency beats correctness.** A word spelled the same way every time is worth more
to the model than a "more correct" spelling used half the time. When this sheet doesn't
cover a case: the Bhuj native-speaker lead decides, and the decision is **logged in the
word list** (bottom of this doc) so everyone copies it. Never improvise silently.

---

## DECISION 1 — Implosive consonants (the only hard call)

Kutchi has implosives that plain Gujarati script can't write. The robust, contrastive
ones in core Kutchi are the **bilabial /ɓ/** and **retroflex /ɗ/** (e.g. *ɓ* in *bara*
"outside", *ɗ* in *dari* "beard"). /ʄ/ and /ɠ/ are marginal/dialectal — usually ignore.

Pick **one** option for the whole project:

**Option A — Plain (recommended for v1).**
Write implosives with the ordinary Gujarati letter: /ɓ/→**બ**, /ɗ/→**ડ**. No marks.
- *Why:* this is how Bhuj speakers already write Kutchi; fastest to type; nothing to
  hear-discriminate; zero risk of two annotators marking differently. For STT the
  acoustic model still learns the sound from audio — the contrast lives in the waveform.
- *Cost:* a handful of text-only minimal pairs become homographs. Acceptable for v1.
- *Hedge:* capture the implosive in the **romanized field** instead (see Decision 6),
  where it's cheap and optional. Info preserved, main transcript stays clean.

**Option B — Phonemic (only if the lead can hear the contrast reliably).**
Mark the two implosives with **nukta ( ઼ , U+0ABC)**: /ɓ/→**બ઼**, /ɗ/→**ડ઼**. Plain બ/ડ
stay for the non-implosive sounds.
- *Why:* preserves a true phonemic distinction; better for future TTS/linguistic use;
  reversible (you can always strip nukta to get Option-A text, never the reverse).
- *Cost:* harder to type, and **dangerous if applied inconsistently** — that's worse
  than not marking at all. Only choose this if you double-annotate and agreement is high.

**My call:** ship v1 on **Option A**, hedge in the roman field. Revisit only if a
phonetic/TTS need appears. Get the lead + Academy to sign off either way and write the
choice here: `IMPLOSIVE POLICY = ______ (ratified by ____ on ____)`.

---

## DECISION 2 — Consonants (everything else is low-risk)

Spell exactly as standard Gujarati does. These distinctions are native to the script and
speakers already handle them:

| sound | letter | sound | letter |
|---|---|---|---|
| k / kh / g / gh | ક ખ ગ ઘ | retroflex ʈ / ʈʰ / ɖ / ɖʰ / ɳ | ટ ઠ ડ ઢ ણ |
| c / ch / j / jh | ચ છ જ ઝ | dental t̪ / t̪ʰ / d̪ / d̪ʰ / n | ત થ દ ધ ન |
| p / ph / b / bh / m | પ ફ બ ભ મ | y r l v · s / ʃ / h | ય ર લ વ · સ શ હ |

**Perso-Arabic loan sounds** (common in Kutchi: *duniya*, *naseeb*, etc.): **default to
the plain Gujarati letter**, no nukta — z→જ, f→ફ, x→ખ, q→ક. Simpler and consistent.
(Mark with nukta — જ઼ ફ઼ ખ઼ — only if you also chose Option B above, to stay coherent.)

---

## DECISION 3 — Vowels

Use standard Gujarati vowel letters/matras. Rules of thumb:

- **Length is phonemic** — write long vs short faithfully: ઇ/ઈ (i/ī), ઉ/ઊ (u/ū),
  અ/આ (a/ā). Mishearing length is the most common vowel error; replay before guessing.
- **Preserved final short vowels.** Kutchi keeps old short final vowels that Gujarati
  often drops. **Write the final vowel if you hear it** (e.g. attach the matra) rather
  than defaulting to a bare consonant out of Gujarati habit.
- The open front /æ/ and open-mid /ɔ/: use **ઍ** and **ઑ** (candra forms) only when
  clearly heard; otherwise ઐ / ઓ. Let the lead fix the default and log examples.

---

## DECISION 4 — Nasalization

Default to **anusvara ( ં )** for all nasalization. Use **candrabindu ( ઁ )** only if the
lead designates specific words for it. One default = no drift. Don't mix freely.

---

## DECISION 5 — Word boundaries, postpositions, clitics

Kutchi (like Sindhi) has postpositions and can suffix pronouns. This is a big silent-
inconsistency trap (`ghar mein` vs `gharmein`).

- Follow **standard Gujarati spacing**: write postpositions as **separate words**.
- Suffixed/clitic pronouns that are clearly fused in speech: **attach them**, and log
  the pattern the first time it appears so it's spelled identically thereafter.

---

## DECISION 6 — Romanized field (optional, but useful)

The `roman` field models phone-typing input and aids search. Keep it **internally
consistent**, even though real-world phone typing is messy:

- long vowels doubled: `aa ee oo`; retroflex as capitals: `T D N R L`; aspirates add `h`.
- **implosive hedge:** if you can hear it, write `bb`/`dd` for /ɓ/, /ɗ/ here
  (e.g. *bbara*, *ddari*) even when the main transcript uses plain બ/ડ. This is where
  Option A quietly preserves the contrast at no cost to the main text.

---

## DECISION 7 — Code-switching (Gujarati / Hindi / English in Kutchi speech)

Kutchi speech is heavily code-mixed. Keep **everything in one script**:

- Transliterate loanwords — including English — **into Gujarati script** as pronounced
  (e.g. "mobile" → મોબાઇલ). One script per transcript helps the model.
- Established acronyms heard letter-by-letter may stay Latin (e.g. `GST`).
- If a clip is *mostly* another language, mark **status = skip** in the desk and add a
  note — don't bury heavy Gujarati/Hindi monologue inside the Kutchi set.

---

## DECISION 8 — Numbers, fillers, punctuation, unclear audio

- **Numbers:** write as **words, as spoken** (`ٹre` → ત્રે / "three" → spoken Kutchi
  word). Don't convert to digits — the transcript must match the speech.
- **Fillers:** transcribe natural fillers as heard, using these fixed spellings so they
  don't fragment: hesitation `અં`, affirmation `હા`, `હાં`, negation `ના`,
  thinking `ઉં`. Log any new filler once and reuse the spelling.
- **Punctuation:** keep it minimal — sentence-final `.` and `?` only. No commas,
  no Gujarati danda. Punctuation guessing is wasted annotator effort here.
- **Unclear audio:** one uncertain word → write best guess + `(?)` right after it.
  Multiple unclear words or bad audio → don't fudge it, mark **status = skip**.
- **Multiple speakers / heavy overlap:** if you can't cleanly attribute the speech →
  **skip**. Clean single-speaker clips are worth far more than messy ones.

---

## Worked example (fill in real ones with the lead before launch)

| machine draft (wrong) | corrected transcript | roman (optional) | notes |
|---|---|---|---|
| _e.g._ બારા આયો | બારા આયો | bbaara aayo | /ɓ/ heard, plain in main text |
| … | … | … | … |

Add 15–20 real lines here from your first session. New annotators read these **before**
their first clip — examples teach the convention faster than rules do.

---

## Logged decisions (the living part — append, never overwrite)

> Every time the lead resolves a case this sheet didn't cover, add a row. This list,
> not the rules above, is what keeps spelling identical across people and across months.

| date | word / case | decision | reason |
|---|---|---|---|
| 2026-07-18 | IMPLOSIVE POLICY | Option A (plain) | locked in CLAUDE.md; Kutchi Sahitya Academy sign-off still pending |
| 2026-07-18 | verb ending -nu / -inu (venanu, achinu, somnu, lekhnu) | → -નૂ (long ū final) | preserved final long vowel (D3). DRAFT from card harvest — ratify |
| 2026-07-18 | copula "aay" (is/are) | → આય | one spelling across all 111 harvested cards. DRAFT — ratify |
| 2026-07-18 | dative pronouns muke / toke / tonke / munke | → મુકે / તોકે / તોંકે / મુંકે | fused clitic pronouns (D5). DRAFT — ratify |
| 2026-07-18 | possessives tonjo / mujo / asanjo / panjo | → તોંજો / મુજો / અસાંજો / પાંજો | nasal = anusvara (D4). DRAFT — ratify |
| 2026-07-18 | subject pronouns aau / aai / tu | → આઉ / આઈ / તૂ | vowel-length call. DRAFT — ratify |
| 2026-07-18 | interrogatives + present aux: kor / keda / to / ta | → કોર / કેડા / તો / તા | DRAFT — ratify |
| 2026-07-18 | loanwords bazaar / photo / dabeli | → બજાર / ફોટો / દાબેલી | code-switch transliterate (D7); plain letters, no nukta (D2). DRAFT — ratify |
| 2026-07-18 | OPEN: "Get up" (phrasebook id 17) | (?) — Kutchi was off-frame in source card | re-check video or native supplies |
| 2026-07-18 | OPEN: "Rabbit" (phrasebook id 68) | (?) — no Kutchi term on the card (only Gujarati ससलुं) | native supplies Kutchi word |
