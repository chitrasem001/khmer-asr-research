# Chapter 5 — Results and Error Analysis

Draft. CER is computed without spaces after normalization protocol v1 (Chapter 3). Brackets give 95%
bootstrap intervals over clips; "paired" differences are computed on the same clips. Test = four
unseen speakers and unseen sentences (3,712 clips); FLEURS = out of domain (771 clips); seen speakers
= unseen sentences read by speakers mostly seen in training (1,891 clips). Results files:
`experiments/m0_*`, `experiments/m1_eval*`, `experiments/error_analysis/`.

---

## 5.1 Zero-shot baselines (M0)

*Table 5.1. Zero-shot results on the test set (RTX 4060 laptop GPU).*

| Model | Clips | CER [95% CI] | CER with spaces | RTF |
|---|---|---|---|---|
| MMS-1B-all (Khmer adapter) | 3,712 | **9.71%** [9.5–9.9] | 11.64% | 0.007 |
| Whisper medium | 400 | 132.9% [129.7–136.0] | 150.5% | 0.706 |
| Whisper small | 400 | 136.3% [132.5–140.0] | 186.7% | 0.219 |
| Whisper large-v3-turbo | 400 | 177.1% [171.3–183.0] | 173.4% | 0.146 |

On FLEURS, MMS scores 15.49% and the Whisper models 114–138%.

**MMS** is usable without any Khmer training. Its errors are mostly substitutions (5.04%) and
deletions (2.88%), with 1.79% insertions, and CER per test speaker ranges from 7.2% to 12.9%. Part of
its error is structural: the released Khmer vocabulary has no symbol for the digits ៤ ៦ ៨ ៩, so it
can never write them.

**Whisper** fails: 1,087 of the 1,200 evaluated clips exceed 100% CER, because decoding locks onto a
repeated character until the token limit. This agrees with the 103.19% published for
Whisper-large-v3 on FLEURS Khmer [4].

*Table 5.2. Whisper diagnostics on 40 test clips.*

| Model | Detected as Khmer (no language forced) | Detected as other | fp16 output = fp32 output |
|---|---|---|---|
| Whisper small | 38 / 40 | Vietnamese 2 | 38 / 40 |
| Whisper medium | 39 / 40 | Vietnamese 1 | 40 / 40 |
| Whisper large-v3-turbo | 0 / 40 | Vietnamese 29, Thai 10, Indonesian 1 | 39 / 40 |

Numerical precision is not the cause. Whisper small and medium recognize the audio as Khmer and still
collapse into repetition; turbo mostly mistakes it for Vietnamese or Thai. Zero-shot Whisper knows it
hears Khmer but cannot write it — the gap fine-tuning has to close.

## 5.2 Fine-tuning and data scaling (RQ1, RQ2)

*Table 5.3. Fine-tuned models (3.5 passes per subset; the 872 h model one pass).*

| Model | Data | Test CER [95% CI] | FLEURS |
|---|---|---|---|
| Whisper small | 10 h | 19.43% [19.08–19.78] | 51.26% |
| Whisper small | 50 h | 6.40% [6.18–6.63] | 32.82% |
| Whisper small | 100 h | 4.70% [4.48–4.93] | 24.48% |
| **Whisper small** | **872 h** | **2.85%** [2.66–3.08] | 18.61% |
| MMS-1B, new adapter | 10 h | 8.43% [8.25–8.64] | 16.72% |
| MMS-1B, new adapter | 50 h | 6.20% [6.05–6.39] | 15.69% |
| MMS-1B, new adapter | 100 h | 5.24% [5.07–5.42] | 14.97% |

Figure 5.1 (`paper/fig_scaling.png`) plots these results against training hours.

**RQ1.** Fine-tuning reduces CER for both families. Whisper small goes from unusable (136%) to 19.43%
with 10 hours; MMS from 9.71% to 8.43% (−13% relative), and all four test speakers improve.

**RQ2.** Every increase in data helps and no curve has flattened. Doubling from 50 to 100 hours still
cuts test CER by 27% (Whisper) and 15% (MMS), and the full 872 hours take Whisper to 2.85%, with its
its lowest validation CER at the final evaluation of the single pass (1.26% at step 24,000, after
1.43% at 16,000 and a rise to 1.92% at 20,000).
Whisper is far more data-hungry: the weaker model at 10 hours, statistically tied with MMS at 50 hours
(6.40% vs 6.20%, overlapping intervals), and significantly better at 100 hours (4.70% vs 5.24%,
non-overlapping intervals).

**Out of domain** the picture differs. Fine-tuned MMS with a new adapter stays close to zero-shot on
FLEURS and is first *worse* than it (16.72% at 10 h vs 15.49%), recovering only at 100 hours (14.97%):
the new adapter, trained on one corpus, loses some of the generality of Meta's adapter. Whisper
improves most with data (51.26% → 18.61%) but never reaches MMS. A same-corpus test set therefore
overstates how well a model will do on other Khmer speech.

## 5.3 Khmer-specific strategy: adapter initialization (RQ3)

*Table 5.4. New adapter versus adapter started from the released Khmer adapter (M2), same data and
steps; differences in CER points, paired 95% intervals.*

| Data | Adapter start | Test | Seen speakers | FLEURS |
|---|---|---|---|---|
| 10 h | New | 8.43% | — | 16.72% |
| 10 h | Released Khmer adapter | **7.45%** | 7.27% | **15.88%** |
| | Difference | −0.99 [−1.07, −0.91] | — | −0.85 [−1.06, −0.66] |
| 100 h | New | 5.24% | 4.83% | 14.97% |
| 100 h | Released Khmer adapter | **4.62%** | **4.12%** | **14.36%** |
| | Difference | −0.61 [−0.69, −0.54] | −0.71 [−0.79, −0.64] | −0.61 [−0.80, −0.44] |

Starting from the released adapter is better at both sizes and on every test set, at the same training
cost; every interval excludes zero. The gain shrinks with data (about one point at 10 h, 0.6 at
100 h) but does not vanish. It also removes the out-of-domain loss of the new adapter: with 100 hours
the M2 model is **1.13 points [0.84, 1.39] better than zero-shot MMS on FLEURS** (14.36% vs 15.49%),
the only fine-tuned model in this study that improves on it.

**Seed check.** Repeating both 10-hour runs with a second training seed gives 8.46% (new adapter) and
7.39% (M2) on the test set and 16.63% and 15.72% on FLEURS: seed-to-seed differences are at most 0.16
points, far below the M2 gain.

## 5.4 Model families and speaker familiarity (RQ4)

*Table 5.5. Unseen versus seen speakers.*

| Model | Data | Unseen speakers (test) | Seen speakers | Change | FLEURS |
|---|---|---|---|---|---|
| MMS-1B zero-shot | — | 9.71% | 9.84% | +1% | 15.49% |
| MMS-1B, new adapter | 100 h | 5.24% | 4.83% | −8% | 14.97% |
| MMS-1B, M2 | 100 h | 4.62% | 4.12% | −11% | 14.36% |
| Whisper small | 100 h | 4.70% | 3.44% | −27% | 24.48% |
| Whisper small | 872 h | **2.85%** | 1.89% | −34% | 18.61% |

**In and out of domain.** The best Whisper model is 1.77 points [1.59, 1.92] better than the best MMS
model on the test set but 4.25 points [3.27, 5.23] worse on FLEURS, despite almost nine times more
training data. In domain Whisper wins; on other speech MMS wins.

**Speaker familiarity.** Zero-shot MMS scores the same on both speaker sets, as it should. At 100 hours,
fine-tuned MMS — which trains only a small adapter — is 8–11% better on speakers it has heard, while
fully fine-tuned Whisper is 27% better, and 34% with the full corpus. A split that let test speakers
into training would flatter Whisper by about a third and change the comparison between the families.
(73 of the 1,891 seen-speaker clips are read by speaker m-adt4-0002, who is not in training; removing
them changes no figure in Table 5.5 by more than 0.05 points.)

**Cost.** MMS trains 2.3 M parameters but takes about four times longer per step than Whisper small
(464 vs 113 minutes for 100 hours on the A40). Whisper's 1-pass full-data run (282 minutes) cost less
than one MMS 100-hour run.

*Table 5.6. Error types and per-speaker CER on the test set.*

| Model | Sub. | Del. | Ins. | CER with spaces | f-adt1-0003 | f-adt4-0001 | m-adt1-0002 | m-adt3-0001 |
|---|---|---|---|---|---|---|---|---|
| MMS-1B zero-shot | 5.04 | 2.88 | 1.79 | 11.64 | 12.89 | 7.23 | 9.85 | 7.73 |
| MMS-1B, M2, 100 h | 2.35 | 1.40 | 0.88 | 5.68 | 6.54 | 3.63 | 4.90 | 3.01 |
| Whisper small, 100 h | 2.57 | 1.21 | 0.92 | 5.67 | 7.62 | 3.05 | 4.74 | 2.57 |
| Whisper small, 872 h | 1.56 | 0.78 | 0.51 | 3.79 | 4.45 | 2.09 | 3.14 | 1.44 |

Values in %. Substitutions dominate for every model. For every model f-adt1-0003 is the hardest test
speaker and m-adt1-0002 the second hardest.

## 5.5 Error analysis

The analysis uses the saved predictions of five models (MMS zero-shot, MMS new adapter 100 h, M2
100 h, Whisper 100 h, Whisper 872 h) on the test set and FLEURS. Each reference character is counted
as a hit, substitution or deletion, and inserted characters separately, on the same normalized,
space-free alignment as the CER.

*Table 5.7. Test-set error rate by character class ((S + D + I) / N).*

| Character class | Characters | MMS zero-shot | M2 100 h | Whisper 872 h |
|---|---|---|---|---|
| Consonant | 237,149 | 8.2% | 3.9% | 2.5% |
| Dependent vowel | 91,549 | 9.6% | 4.7% | 2.9% |
| Subscript sign (coeng) | 35,598 | 10.8% | 4.3% | 2.4% |
| Diacritic / sign | 25,004 | 18.7% | 8.8% | 3.9% |
| Independent vowel | 2,735 | 35.4% | 18.0% | 15.7% |
| Digit | 388 | 46.6% | 21.6% | 17.0% |
| Latin letter | 515 | 69.5% | 56.7% | 79.6% |

### 5.5.1 Spelling variants

The most frequent error of every model is **subscript da written as subscript ta** (្ដ → ្ត), and the
second is **ឲ written as ឱ** (as in ឲ្យ / ឱ្យ). Of all ដ↔ត confusions, 84% (M2) and 93% (Whisper
872 h) follow the subscript sign. Both pairs sound the same and the corpus uses both spellings —
in the test references ្ដ occurs 643 times and ្ត 2,610 times; ឲ 337 times and ឱ 1,166 times — so no
model can choose the writer's spelling from audio. Counting each pair as one form on both sides lowers
test CER by 3–8% relative (MMS zero-shot 9.71 → 9.44%; M2 100 h 4.62 → 4.38%; Whisper 872 h 2.85 →
2.62%) and changes no ranking.

### 5.5.2 Sound-alike letters and subscripts

The next most frequent substitutions are consonants of the two series that share a sound (ក→គ, ច→ជ,
ប→ព), final m written as nikahit (ម→ំ), and vowel signs that differ in length or height (ឹ→ិ, ូ→ៅ).
Among deletions and insertions, the subscript sign itself leads for every model (for M2: 825 deleted,
392 inserted), followed by ro and the vowel aa — the model writes a cluster with or without a
subscript, or merges and splits syllables.

### 5.5.3 Digits, Latin words and long transcripts

- **Digits** (388 test characters) have 17–22% error after fine-tuning and 47% for zero-shot MMS, which
  cannot write four of the ten digits.
- **Latin letters** (515 test characters, in 46 clips) are the worst class for every model. Whisper
  872 h writes 62% of them (319 of 515) as Khmer characters — it transliterates English words into
  Khmer script — against 32% for M2.
- Clips containing digits or Latin words have **1.6 to 4.6 times** the model's overall test CER
  (Whisper 872 h: 7.67% with digits, 13.06% with Latin words).
- The **24 test transcripts longer than Whisper's decoder limit** score 15.3% for Whisper 872 h and 6.2%
  for M2; the other clips score 2.68% for Whisper. Whisper must be given segmented audio for long
  recordings.
- Looping, the failure of zero-shot Whisper, almost disappears after fine-tuning: one test clip and two
  FLEURS clips still loop for Whisper 872 h, none for MMS.

### 5.5.4 Speakers

Every model is worse on the two female test speakers (Whisper 872 h 3.60% vs 2.14%; M2 5.50% vs
3.78%), mostly because of one speaker, f-adt1-0003 (Table 5.6). With four test speakers this is a
speaker effect; it is not evidence of a gender gap.

## 5.6 Summary

| RQ | Answer |
|---|---|
| RQ1 | Fine-tuning helps both families; Whisper goes from unusable to 19.43% with 10 h, MMS from 9.71% to 8.43%. |
| RQ2 | More data keeps helping up to the full 872 h (Whisper 2.85%); Whisper needs more data than MMS to overtake it. |
| RQ3 | Starting from the released Khmer adapter is better at 10 h and 100 h on every test set (−0.6 to −1.0 points) and gives the best out-of-domain model (FLEURS 14.36%). |
| RQ4 | Whisper wins in domain (2.85% vs 4.62%), MMS out of domain (14.36% vs 18.61%); Whisper gains far more from speaker familiarity. |
