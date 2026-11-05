# Progress Reports

Thesis: *An Experimental Study of Fine-Tuning Strategies for Pretrained Models in Khmer ASR*
Student: Sem Chitra · Advisor: Dr. Mao Makara

Newest report first. Each entry: what was done, what was found (with numbers), decisions, next steps.
Published results by others are marked as such; everything else is our own measurement.

---

## Report 6 — 1 Nov 2026 · Error analysis

### Setup
- Character-level analysis of the saved predictions (no new training or inference) for five
  models: MMS zero-shot, MMS new adapter 100 h, M2 100 h, Whisper small 100 h and 872 h, on the
  test set and FLEURS. Same normalization and space-free alignment as the primary CER.
- Code: `src/khmer_asr/evaluation/error_analysis.py` (5 unit tests), `scripts/error_analysis.py`;
  outputs in `experiments/error_analysis/` (`class_rates.csv`, `confusions.csv`, `subsets.json`).

### Error rate by character class (test set, (S + D + I) / N)

| Class | Characters | MMS zero-shot | M2 100 h | Whisper 872 h |
|---|---|---|---|---|
| Consonant | 237,149 | 8.2% | 3.9% | 2.5% |
| Dependent vowel | 91,549 | 9.6% | 4.7% | 2.9% |
| Subscript sign (coeng) | 35,598 | 10.8% | 4.3% | 2.4% |
| Diacritic / sign | 25,004 | 18.7% | 8.8% | 3.9% |
| Independent vowel | 2,735 | 35.4% | 18.0% | 15.7% |
| Digit | 388 | 46.6% | 21.6% | 17.0% |
| Latin letter | 515 | 69.5% | 56.7% | 79.6% |

### Findings
- **The two most frequent errors of every model are spelling variants, not recognition errors.**
  Subscript da vs ta (្ដ / ្ត: 84–93% of all ដ↔ត errors follow the subscript sign) and the two
  forms of the vowel in ឲ្យ / ឱ្យ. They sound the same and the corpus mixes both (test references:
  ្ដ 643 vs ្ត 2,610; ឲ 337 vs ឱ 1,166). Counting them as equal lowers test CER by 3–8% relative
  (Whisper 872 h 2.85 → 2.62%, M2 100 h 4.62 → 4.38%); rankings do not change.
- **Next: sounds the script writes two ways** — consonant pairs of the two series (ក/គ, ច/ជ, ប/ព),
  final m vs nikahit (ម/ំ), and added or dropped subscripts.
- **Digits and Latin words are the weakest points.** Clips containing them have 1.6–4.6× the
  model's overall CER (Whisper 872 h: 7.67% with digits, 13.06% with Latin words). Whisper writes
  62% of Latin letters as Khmer characters (it transliterates English words), M2 32%.
- **Long transcripts:** the 24 test clips too long for Whisper's decoder score 15.3% for Whisper
  872 h vs 6.2% for M2 (other clips: 2.68%). Long audio must be segmented before Whisper.
- **Speakers:** every model is worse on the female test speakers (Whisper 872 h 3.60% vs 2.14%;
  M2 5.50% vs 3.78%), mostly because of one speaker (f-adt1-0003). With four test speakers this is
  a speaker effect, not evidence of a gender gap.

### Decision needed
- **Normalization v2:** unify ្ដ/្ត and ឲ/ឱ in the primary CER? Fairer to all models, but it changes
  every reported number and must be decided **before the test set is locked**. Until then the
  primary CER keeps protocol v1 and the merged CER is reported as a sensitivity check.

### Next
- Paper §9 (done), slide, advisor meeting (split D, M2, normalization v2, deadline/template).
- Thesis: literature review chapter; chapters 3–5 from the paper.

---

## Report 5 — 30 Oct – 1 Nov 2026 · Khmer-specific strategy (RQ3), full-data model, robustness checks

### Setup
- **M2 (RQ3): start the MMS adapter from Meta's released Khmer (`khm`) adapter** instead of from
  scratch. The 288 adapter tensors are copied as they are; in the output layer, 105 of our 111 (10 h; 110 of 116 at 100 h)
  characters take the released row for the same character, and the 6 characters the released
  vocabulary lacks (`|`, `è`, ៤ ៦ ៨ ៩) start new. Everything else is identical to the M1/S MMS runs
  (same subsets, steps, learning rate), so only the initialisation changes. Check before training:
  the initialised model scores 10.07% on 40 validation clips (zero-shot MMS 10.2%), a scratch head
  249.71% — the copy is correct.
- **Main in-domain model:** Whisper small on the **full training split** (872.0 h, 376,255 clips),
  **one pass** (24,000 steps). Reported separately from the 3.5-pass scaling curve.
- **Seen-speaker test** (`test_seen_speakers`, 1,891 clips, 5 h): sentences no model trained on,
  read by 10 training speakers. Compared with the main test set (unseen speakers), it shows how much
  of a model's accuracy comes from knowing the voice.
- **Second training seed** (43) for the two 10 h MMS runs, on the same data subset.

### Results (CER, spaces ignored; 95% bootstrap interval for the main test set)

| Model | Train data | Validation | **Test** (unseen speakers) | Seen speakers | **FLEURS** |
|---|---|---|---|---|---|
| MMS-1B zero-shot | — | 10.20% | 9.71% [9.5–9.9] | 9.84% | 15.49% |
| MMS-1B, new adapter (M1/S) | 10 h | 8.85% | 8.43% [8.25–8.64] | — | 16.72% |
| **MMS-1B, from khm adapter (M2)** | 10 h | 7.82% | **7.45%** [7.26–7.64] | 7.27% | 15.88% |
| MMS-1B, new adapter (S) | 100 h | 5.33% | 5.24% [5.07–5.42] | 4.83% | 14.97% |
| **MMS-1B, from khm adapter (M2)** | 100 h | 4.54% | **4.62%** [4.47–4.80] | 4.12% | **14.36%** |
| Whisper small (S) | 100 h | 2.40% | 4.70% [4.48–4.93] | 3.44% | 24.48% |
| **Whisper small, full data** | 872 h, 1 pass | 1.26% | **2.85%** [2.66–3.08] | **1.89%** | 18.61% |

Paired bootstrap on the same clips (difference in CER points, 95% interval):

| Comparison | Test | FLEURS | Seen speakers |
|---|---|---|---|
| M2 vs new adapter, 10 h | −0.99 [−1.07, −0.91] | −0.85 [−1.06, −0.66] | — |
| M2 vs new adapter, 100 h | −0.61 [−0.69, −0.54] | −0.61 [−0.80, −0.44] | −0.71 [−0.79, −0.64] |
| M2 100 h vs zero-shot | −5.08 [−5.21, −4.96] | **−1.13 [−1.39, −0.84]** | −5.72 [−5.90, −5.55] |
| Whisper 872 h vs best MMS (M2 100 h) | −1.77 [−1.92, −1.59] | +4.25 [+3.27, +5.23] | −2.23 [−2.38, −2.10] |

Second seed (10 h, same data): new adapter 8.43 / 8.46% test, 16.72 / 16.63% FLEURS; M2 7.45 / 7.39%
test, 15.88 / 15.72% FLEURS. Seed-to-seed differences are ≤ 0.16 points.

### Findings
- **RQ3 — starting from Meta's Khmer adapter helps at both sizes, in and out of domain**, at no extra
  training cost. It is worth about 1 point at 10 h and 0.6 points at 100 h on the test set, roughly
  15× the seed-to-seed noise. **M2 100 h is the first fine-tuned model that is clearly better than
  zero-shot on FLEURS** (14.36% vs 15.49%): the new-adapter runs had lost Meta's generality; keeping
  it fixes that.
- **Full data makes Whisper small near-perfect in domain**: 2.85% test (−39% vs 100 h), 1.89% on seen
  speakers; validation was still improving at the end of the single pass.
- **The in-domain / out-of-domain split remains the main result for RQ4**: the best Whisper
  (872 h) is 1.8 points better than the best MMS (M2 100 h) on the test set but 4.3 points worse on
  FLEURS — even with almost 9× more training data.
- **Unseen-speaker testing matters most for fully fine-tuned models.** On seen speakers Whisper's
  CER is 27% (100 h) and 34% (872 h) lower than on unseen speakers; MMS at 100 h only 8–11%
  (new adapter 8%, M2 11%).
  A split that lets test speakers into training would make Whisper look about a third better than
  it is on new voices.

### Cost and logs
- 30–31 Oct night run (M2 10 h + Whisper 872 h): ≈ $4.70. 31 Oct round 3: $7.60. **Total GPU spend
  ≈ $23** of the $200 budget.
- Pods back results up to the private HF repo (sha256 checked) and stop themselves, so nothing is
  billed while unattended. All per-run logs, predictions and the round logs
  (`experiments/m2_full_logs/`, `experiments/round3_logs/`) are in git; weights are on HF.
- **Problem:** the planned second Whisper pass did not run. Logs pushed to GitHub during the round
  made the pod's `git pull` before that run fail. Fixed: each round now pins its code at the start
  (`SKIP_PULL`), and nothing is pushed while a pod runs.

### Next
- Paper and slides: M2 section, full-data model, seen-speaker table, paired tests.
- Error analysis: which characters are confused; digits, Latin words, over-long clips.
- Optional (needs a top-up): MMS from the khm adapter on the full 872 h — likely the best model out
  of domain (≈ $13); Whisper second pass (≈ $4).
- Advisor: approve split option D (lock the test set) and the M2 strategy.

---

## Report 4 — 29–30 Oct 2026 · Data scaling (RQ2)

### Setup
- One RunPod A40 pod ran four jobs in sequence (`scripts/runpod_scaling.sh`): Whisper small and
  MMS-1B on nested 50 h and 100 h subsets (10 h ⊂ 50 h ⊂ 100 h, seed 42), each for **3.5 passes**
  over its subset — the same as the 10 h pilots — so only the amount of data changes.
- Every model is scored on the same test set (4 unseen speakers, unseen sentences, 3,712 clips) and
  FLEURS (771 clips), with the same normalization and space-free CER.

| Subset | Clips | Hours | Files | Steps (3.5 passes) |
|---|---|---|---|---|
| 10 h | 4,561 | 10.40 | 51 | 1,000 |
| 50 h | 21,193 | 50.02 | 226 | 4,638 |
| 100 h | 43,348 | 100.58 | 435 | 9,485 |

### Results (CER, spaces ignored)

| Model | Train data | Validation | **Test** | FLEURS | Training time |
|---|---|---|---|---|---|
| Whisper small | zero-shot | — | 136.30% † | 124.13% † | — |
| Whisper small | 10 h | 17.86% | 19.43% | 51.26% | 18 min |
| Whisper small | 50 h | 3.81% | 6.40% | 32.82% | 59 min |
| Whisper small | **100 h** | 2.40% | **4.70%** | 24.48% | 113 min |
| MMS-1B | zero-shot | 10.20% | 9.71% | 15.49% | — |
| MMS-1B | 10 h | 8.85% | 8.43% | 16.72% | 51 min |
| MMS-1B | 50 h | 6.38% | 6.20% | 15.69% | 234 min |
| MMS-1B | **100 h** | 5.33% | 5.24% | **14.97%** | 464 min |

† seeded sample (400 test / 200 FLEURS clips) because zero-shot Whisper loops to the token limit.
Whisper validation uses 294 clips, MMS validation all 1,288 — validation columns are not comparable
across families; the test and FLEURS columns are. Inference speed is not compared here: 10 h models
were evaluated on the laptop RTX 4060, 50/100 h models on the A40.

### Findings
- **More data helps both families on the test set.** Whisper 19.43 → 6.40 → 4.70%; MMS 8.43 → 6.20 →
  5.24%. Doubling 50 → 100 h still cuts test CER by 27% (Whisper) and 15% (MMS): neither curve has
  flattened at 100 h.
- **Whisper is far more data-hungry, and overtakes MMS at 100 h.** It is the worst model at 10 h,
  level with MMS at 50 h (6.40% vs 6.20%) and the best at 100 h (4.70% vs 5.24%).
- **Out of domain, MMS stays well ahead.** FLEURS: MMS 16.72 → 15.69 → 14.97% (the 100 h model is the
  first fine-tuned MMS to beat zero-shot, 15.49%); Whisper 51.26 → 32.82 → 24.48%, still above MMS
  zero-shot. Whisper learns this dataset's speakers and domain faster than Khmer in general.
- **The in-domain/out-of-domain gap is large** (best test 4.70% vs best FLEURS 14.97%): a test set
  from the same corpus overstates real-world quality — a key point for the thesis.
- **Choice depends on the use**: best accuracy on this kind of speech → Whisper-100h (also
  Apache-2.0, usable commercially); robustness to other speech → MMS-100h (CC BY-NC, research only).

### Cost and logs
- Pod 14:11 (29 Oct) → 06:00 (30 Oct): **$9.50** (balance 14.94 → 5.44). Total GPU spend so far
  **$10.80** of the $200 budget.
- All logs kept: per-run `console.log`, `log_history.json`, `run_info.json`, `config.yaml`, and
  `experiments/scaling_logs/` (full scaling console, launch log, eval-prep log, upload log).
- Fix during the run: the audio prep for evaluation ran before the package was installed; it was
  rerun on the pod and the script fixed (commit 8dea503).
- **Model backup**: all four checkpoints are in the private Hugging Face repo
  `chitra168/khmer-asr-thesis-models` (sizes verified against the pod); weights are not in git.

### Next
- Scaling chart (CER vs hours, test and FLEURS, both models) for the paper and slides.
- M2 (RQ3): one Khmer-specific strategy — candidates: initialise MMS from Meta's Khmer adapter, or
  add FLEURS training data (targets the out-of-domain gap).
- Optional: Whisper on the full 872 h (about 1 pass, ≈ $3) for the best final model.
- Advisor: approve split option D (lock the test set), the two models, and the M2 choice.

---

## Report 3 — 27–29 Oct 2026

### Done
- **FLEURS Khmer** added as an out-of-domain test (771 clips, 3.15 h, CC BY 4.0), scored with the
  same normalization and CER as our test set.
- **First fine-tuning runs (M1 pilots)** on RunPod, 1× A40 48 GB: MMS-1B and Whisper small, each on
  the same nested 10 h training subset (4,561 clips, 51 files), 1,000 steps, batch 16.
- **Demo app (Khmer ASR Studio)**: FastAPI backend + React/Vite frontend following the mockup;
  Compare tab (test clips or uploads, per-model CER and colored error view) and Results tab.
- **Slide deck** for the advisor meeting (12 slides).

### Zero-shot vs fine-tuned (CER, spaces ignored)

| Model | Our test (unseen speakers + sentences) | FLEURS (out of domain) |
|---|---|---|
| MMS-1B zero-shot | 9.71% | 15.49% |
| **MMS-1B fine-tuned, 10 h** | **8.43%** (−13% rel.) | 16.72% (+8% rel.) |
| Whisper small zero-shot | 136.3% (400-clip sample) | 124.1% (200-clip sample) |
| **Whisper small fine-tuned, 10 h** | **19.43%** | 51.26% (7 clips still loop) |

Validation CER during training (validation = unseen sentences, known speakers):

| Step | MMS (1,288 clips) | Whisper small (294 clips) |
|---|---|---|
| 250 | 10.93% | 89.33% |
| 500 | 9.70% | 34.98% |
| 750 | 9.13% | 20.74% |
| 1,000 | **8.85%** | **17.86%** |

MMS zero-shot on the same validation clips: 10.20%.

### Findings
- **MMS improves in-domain but loses out-of-domain.** All four test speakers improve, but FLEURS
  gets worse for both genders (more substitutions and deletions). Replacing Meta's Khmer adapter
  with one trained on 10 h of read cultural speech from 15 speakers costs generality. Not a
  vocabulary effect: only 0.04% of FLEURS characters are outside the new vocabulary.
- **Whisper learns fast but starts from nothing.** 136% → 17.9% validation CER in 18 minutes of
  training, still falling steeply at step 1,000, but about twice MMS's error with 10 h of data.
- **Whisper generalizes much worse.** Test 19.4% but FLEURS 51.3%: having learned Khmer from 10 h
  of one dataset, it more than doubles its error out of domain, while MMS (pre-trained on Khmer)
  goes 8.4% → 16.7%. With little data, MMS is both more accurate (about 2× on every test speaker)
  and more robust, and transcribes about 12× faster (RTF 0.006 vs 0.070) — evidence for RQ4.
- **The hardest test speaker is the same for both models** (f-adt1-0003: MMS 11.2%, Whisper
  25.6%), so that voice is genuinely harder, not a model quirk.
- **Both curves are still falling at step 1,000**, so the data-scaling runs (50 h, 100 h) are the
  natural next step; Whisper is likely to gain more from them than MMS.
- The released MMS Khmer vocabulary lacks the digits ៤ ៦ ៨ ៩; the fine-tuned adapter uses a
  vocabulary built from the training text.

### Cost and time (RunPod A40, $0.59/h)

| Run | Training | Pod total | Notes |
|---|---|---|---|
| MMS-1B 10 h | 51.4 min | $0.93 | incl. Hugging Face rate-limit stall and a restart |
| Whisper small 10 h | 18.0 min | $0.37 | incl. one crashed attempt (fixed) |

Balance 16.29 → 14.98 USD. Laptop (RTX 4060) for comparison: MMS 6.6 s/step at batch 4.

### Problems found and fixed
- **Hugging Face rate limit (HTTP 429)** on the pod without a login → HF read token on the pod;
  metadata sent from the laptop instead of re-auditing.
- **Split hash mismatch on Linux** → only CRLF vs LF line endings; content identical. CSVs are now
  written with LF everywhere, so the split hash is the same on Windows and Linux (pod check: MATCH).
- **Whisper crashed at its first evaluation**: validation transcripts over 448 tokens (6 of 300)
  were not filtered. Now filtered for training/validation; the test evaluation keeps every clip.
  24 of 3,712 test clips (0.65%) exceed Whisper's limit and count as partial errors.
- **MMS console log lost** when the first pod was deleted; reconstructed from captured lines
  (`experiments/m1_mms/pod_log_summary.md`). `scripts/runpod_train.sh` now keeps the console log in
  the experiment folder (the Whisper run has both its logs).

### Next
- Data scaling (RQ2): 50 h and 100 h for both models.
- M2 candidates: start MMS from Meta's Khmer adapter; add FLEURS training data.
- Advisor meeting: split option D (then lock the test set), the two models, and M2 choice.

---

## Report 2 — 23 Oct 2026

### Done
- **M0 zero-shot baseline** of two model families on the test set (unseen speakers and sentences).
- **Whisper diagnostics** on 40 test clips: language identification and fp16 vs fp32 decoding.
- **Paper draft** (`paper/khmer_asr_paper_draft.docx`, 5 pages), generated from the result files by
  `paper/build_paper.py`, so every number in it matches the experiments.
- **Training code** (`scripts/train.py`) for Whisper (full fine-tuning) and MMS (new Khmer adapter),
  smoke-tested on the local RTX 4060 before any RunPod spending.

### M0 results (test set; CER ignores spaces; 95% bootstrap interval)

| Model | Clips | CER [95% CI] | CER with spaces | RTF |
|---|---|---|---|---|
| **MMS-1B-all (Khmer adapter)** | 3,712 (full) | **9.7% [9.5–9.9%]** | 11.6% | 0.007 |
| Whisper medium | 400 (sample) | 132.9% [129.7–136.0%] | 150.5% | 0.706 |
| Whisper small | 400 (sample) | 136.3% [132.5–140.0%] | 186.7% | 0.219 |
| Whisper large-v3-turbo | 400 (sample) | 177.1% [171.3–183.0%] | 173.4% | 0.146 |

- **MMS** is already usable: 9.7% CER, 7.2–12.9% across the four test speakers. Its released Khmer
  vocabulary has no symbols for the digits ៤ ៦ ៨ ៩, so it can never output them.
- **Whisper fails zero-shot**: 1,087 of 1,200 evaluated clips exceed 100% CER because decoding
  repeats one character until the 440-token limit. Whisper was evaluated on a seeded 100-clip
  sample per speaker because the looping makes full-set decoding take about 16 GPU-hours.

### Whisper diagnostics (40 clips)

| Model | Detected as Khmer (language not forced) | fp16 = fp32 output |
|---|---|---|
| Whisper small | 38 / 40 (2 Vietnamese) | 38 / 40 |
| Whisper medium | 39 / 40 (1 Vietnamese) | 40 / 40 |
| Whisper large-v3-turbo | 0 / 40 (29 Vietnamese, 10 Thai, 1 Indonesian) | 39 / 40 |

Small and medium recognize the language as Khmer and still cannot transcribe it; only turbo
misidentifies it. Precision is not the cause. *(An earlier one-clip spot check had suggested
all Whisper models mistake Khmer for Vietnamese; the 40-clip check corrected this.)*

### Bug found and fixed: 48 kHz audio
- The first M0 run showed one speaker at 19% CER. The cause: the 71 files named `*-of-00724` are
  **48 kHz**, although the dataset declares 16 kHz. Our extraction read raw audio bytes without
  resampling, so those 129 test clips reached the models at one third of the correct speed
  (MMS: 128 of them above 50% CER).
- Fix: all audio is resampled to 16 kHz at extraction; evaluation refuses non-16 kHz clips; the
  audit now records the sample rate of every upload batch. The first M0 results were discarded and
  every model was rerun. MMS moved from 11.6% to 9.7% CER.
- Training is affected the same way (1,794 clips, 4.9 h); they are resampled by the same code.

### Training pipeline (smoke test, 0.31 h of audio, 10 steps, RTX 4060)
| Model | Setup | Peak GPU memory | Time |
|---|---|---|---|
| Whisper small | Full fine-tuning; transcripts over 448 tokens dropped (about 1%) | 4.7 GB at batch 4 | 7.2 min |
| MMS-1B-all | New Khmer adapter + vocabulary built from training text; base frozen (2.3 M trainable) | 6.5 GB at batch 4 | 2.9 min |

Both runs trained, evaluated on validation, saved the best checkpoint and wrote full run records.
Whisper uses about 2.2 tokens per Khmer character (median 220 tokens per sentence). Training subsets
are drawn by whole files (about 40 files ≈ 10 h ≈ 1 GB), nested so 10 h ⊂ 50 h ⊂ 100 h.

### Next
- Advisor meeting: approve split option D (then lock the test set) and the two models to fine-tune
  (Whisper small, MMS-1B-all).
- RunPod pilot: Whisper small on 10 h to measure real time and cost per training hour.
- M1 full runs for both models.

---

## Report 1 — 22 Oct 2026

### Done
- Wrote the thesis proposal draft (`docs/research_proposal.md`, shared doc:
  https://claude.ai/artifact/3yp3ZyQTcB66zBBGdkrZ3y).
- Initial literature review: arXiv, ACL Anthology (LREC), ISCA archive (Interspeech, SLTU), ICASSP.
  No controlled study of modern pretrained models fine-tuned on Khmer was found.
- Set the plan: target submission end of Feb 2027; RunPod budget cap $200 (≤ $25 per run).
- Set up the environment: Python 3.12, PyTorch 2.6 + CUDA, local RTX 4060 Laptop (8 GB).
  Tests and lint pass. Versions pinned in `requirements.lock.txt`.
- Wrote the dataset audit (`scripts/audit_dataset.py`). It reads only the metadata columns of every
  parquet file, so the whole corpus is audited without downloading the 128 GB of audio.

### Findings
- **Dataset size on disk:** 128.2 GB in 4,447 parquet files (revision `e872b0c`).
- **Audio:** 16 kHz, mono. The `duration` column matches the decoded audio length exactly
  (checked on 20 clips).
- **Speakers:** `speaker_metadata.xlsx` lists **21 speakers** (10 female, 11 male), aged 18–66,
  from 10 provinces. The dataset card's "12 speakers" is out of date.
- **Same sentences, many speakers:** a random split would put the same sentence in both training and
  test, so the test set must hold out both **speakers** and **sentences** (details below).

### Full audit results (all 4,447 files, revision `e872b0c`)
Source: `data/manifests/audit_report.json`, `data/manifests/audit_speakers.csv`.

| Item | Dataset card | Our audit |
|---|---|---|
| Utterances | 450,396 | **450,396** (confirmed; all 4,447 files) |
| Total audio | 727.94 h | **1,073.91 h** (sum of the `duration` column) |
| Speakers with audio | 12 | **20**: 9 female, 11 male (`f-adt1-0006` is in the metadata sheet but has no audio) |
| Unique sentences | not stated | **112,517** |
| Topics / subtopics | 61 / 1,749 | 70 / 2,087 |

- **Clip length:** median 8.06 s; 5–95% range 4.5–14.4 s. 636 clips have no duration value, 1 is
  under 0.5 s, and 36 are over 30 s (1.15 h total, up to 1,190 s). These will be filtered out.
- **Every sentence is read by 4 different speakers** (112,436 of 112,517 sentences; the rest by 5–6).
  Readers come in groups of 2 female + 2 male. Two speakers (`f-adt2-0002`, `m-adt2-0002`) are in
  almost every group.
- **Speakers are very unbalanced:** the top 3 speakers have 653 h (61%); 3 speakers have only 16
  utterances each.
- **Transcripts:** none empty; median 102 characters. Some contain Latin letters, quotes and
  brackets (e.g. 20,856 `"` characters), which the normalization rules must handle.

### Split options (decision needed)
Rule: test speakers are never in training, and no sentence read by a test speaker is in training.
Utterances by training speakers of those sentences are "set aside" (usable as a second,
seen-speaker/unseen-sentence test).

| Option | Test speakers | Test pool (h) | Train pool (h) | Set aside (h) |
|---|---|---|---|---|
| A: older speakers (46+) | 5 | 101.5 | 935.7 | 36.7 |
| B: four small young speakers | 4 | 40.5 | 928.6 | 104.8 |
| C: 1 F + 1 M mid-size | 2 | 65.5 | 816.6 | 191.8 |
| **D: 2 F + 2 M, mixed age (recommended)** | 4 | 66.8 | 875.8 | 131.3 |

Option D (`f-adt1-0003`, `f-adt4-0001`, `m-adt1-0002`, `m-adt3-0001`; ages 23–60) tests across
gender and age while keeping 876 h for training. Option A removes all older voices from training,
so it would test age generalization rather than general accuracy. The test set itself will be a
fixed sample of a few hours from the pool, not the whole pool.

### Text normalization protocol (v1)
Built from the character inventory of all 112,517 sentences (190 distinct characters).
The same rules apply to training targets and to both sides of every CER score.

| Found in transcripts | Count | Rule |
|---|---|---|
| Zero-width spaces | 87,242 | Remove |
| Khmer and Latin punctuation (។ ៖ " « » ( ) , -) | e.g. ។ in 112,259 | Replace with a space (not spoken) |
| Spoken marks % & # @, symbols $ + °, repetition sign ៗ | ៗ: 20,476 | Keep (read aloud) |
| Arabic digits / Khmer digits | 85 / 4,475 sentences | Map Arabic → Khmer |
| Latin words ("vitamin A", "Beta-Carotene") | 1,642 sentences | Keep, lowercased |
| Legacy character order (្រ before another subscript; vowel before subscript; ំ before ុ) | 884 sentences | Reorder to the standard order |
| Newlines, no-break spaces, long runs of spaces | 511 / 88 / 1,704 | Collapse to one space |

**Primary CER ignores spaces**, because Khmer word spacing is optional and inconsistent. A
space-sensitive CER is reported as a secondary metric. 21 unit tests cover the rules.

**Reproducibility:** the character counts come from `scripts/audit_dataset.py`, which saves every
character with its Unicode name and counts to `data/manifests/char_inventory.csv`, and the
per-issue sentence counts to `text_issues_in_sentences` in `audit_report.json`.

### Data preparation (option D, provisional)
`scripts/prepare_dataset.py --config configs/data.yaml` filters clips and writes the split.

| Split | Clips | Hours | Speakers | Purpose |
|---|---|---|---|---|
| train | 376,255 | 872.0 | 15 | Fine-tuning |
| validation | 1,288 | 3.0 | 12 | Checkpoint selection (unseen sentences) |
| test | 3,712 | 10.0 | 4 | Final results: unseen speakers and sentences |
| test_seen_speakers | 1,891 | 5.0 | 10 | Unseen sentences, known speakers |

- Dropped 673 clips: 636 missing duration, 1 under 0.5 s, 36 over 30 s.
- Train has 15 speakers, not 16: every sentence of `m-adt4-0002` (male, 66; 1,867 clips, 5.95 h)
  was also read by a test speaker, so this speaker is not used. *(Corrected 23 Oct: the first
  version of this report wrongly named `f-adt2-0001`.)* This removes one of the few older voices
  from training.
- Leakage checks run on every build: no clip, sentence or test speaker is shared with train.
- Clip IDs are `<file>:<row>`; 4 random test rows were checked against the source files
  (transcript, speaker and duration all match).
- The test set is **not locked yet**; it will be locked once the split option is approved.

### Decisions
- Logged in `docs/decisions.md`: deadline, budget, GPU choice, environment, normalization v1,
  space-free primary CER, provisional option D, clip IDs.

### Next
- Agree on the split option with Dr. Mao Makara, then lock the test set.
- Choose and pin the M0 model (Whisper small vs. medium on the 8 GB local GPU).
