# Chapter 1 — Introduction

Draft for the thesis *An Experimental Study of Fine-Tuning Strategies for Pretrained Models in Khmer
ASR* (Sem Chitra; advisor Dr. Mao Makara). Reference numbers [n] refer to the list in Chapter 2.
Results are provisional until the test set is locked (split option D and the normalization protocol
are awaiting approval).

---

## 1.1 Background

Automatic speech recognition (ASR) turns spoken audio into text. It underlies voice search,
captioning, transcription of meetings and archives, and spoken interfaces for people who find typing
hard. For widely spoken, well-resourced languages, modern systems are accurate enough for everyday
use. For many other languages they are not, because the models were trained on little or no speech
in those languages.

Khmer, the national language of Cambodia, is one of these under-resourced languages for ASR. Large
multilingual models now list Khmer among the languages they cover — Whisper [1] and MMS [3] among
them — but their accuracy out of the box is poor. A recent multilingual study reports a character
error rate (CER) of 103.19% for zero-shot Whisper-large-v3 on the FLEURS Khmer test set, and 25.99%
after fine-tuning (*published*) [4]. Even commercial speech models reach only 13–30% CER on Khmer
in SEA-SpeechBench (*published*) [5].

Two recent developments make better Khmer ASR possible. First, pretrained models can be adapted to
a new language with modest data and compute, either by fine-tuning all their weights or by training
a small language-specific part (Chapter 2). Second, a large open Khmer speech corpus is now
available: the Digital Divide Data Khmer ASR Cultural Dataset [10]. Our audit of this corpus counts
450,396 utterances and 1,073.9 hours of read speech from 20 speakers (*ours*, Chapter 3).

## 1.2 Problem statement

Fine-tuning pretrained models on Khmer clearly helps, but it is not known **which choices drive the
improvement or how large each effect is**. Published Khmer results cannot answer this, for three
reasons:

1. **They are not comparable.** Studies use different test sets, different text normalization and
   different handling of spaces — which matters for a language written without spaces between
   words — and many do not document these choices (Chapter 2).
2. **They change several things at once.** A reported score reflects the model, the training data,
   its amount and the fine-tuning method together, so the effect of each cannot be separated.
3. **Their test sets may leak.** In the largest open corpus every sentence is read by four different
   speakers. A random train/test split, or one that only holds out speakers, puts test sentences
   into training, so a model is partly graded on text it has already seen (Chapter 3).

Without a controlled, leakage-free comparison, a developer of Khmer speech technology cannot tell
which model to start from, how much transcribed data is worth collecting, or whether a Khmer-specific
adaptation is worth its cost.

## 1.3 Research questions

- **RQ1 — Fine-tuning.** How much does fine-tuning a pretrained ASR model on Khmer reduce CER,
  compared with the same model without fine-tuning?
- **RQ2 — Data quantity.** How does the amount of fine-tuning data affect CER on a fixed test set?
- **RQ3 — Khmer-specific strategy.** Does one Khmer-specific strategy reduce CER beyond plain
  fine-tuning? The strategy studied is starting the adapter of a multilingual model from its released
  Khmer adapter instead of from scratch.
- **RQ4 — Model family.** How do a CTC model (MMS) and an encoder-decoder model (Whisper) compare on
  CER in and out of domain, and on training cost?

## 1.4 Objectives

1. Audit the full Khmer corpus and build a **leakage-free split** that holds out both test speakers
   and every sentence they read.
2. Define and document **one Khmer text-normalization protocol** and a space-insensitive CER, applied
   identically to training targets and to every score.
3. Measure **zero-shot baselines** for Whisper and MMS on this split and on FLEURS.
4. Fine-tune both families under **controlled conditions**, changing one variable at a time: whether
   to fine-tune, how much data (10, 50, 100 h and the full split), and how the adapter is initialized.
5. Report **uncertainty** (bootstrap intervals, paired tests, a second training seed) and analyse the
   **errors** the models make.
6. Release the code, configurations, split manifests and run logs so that every number can be
   reproduced.

## 1.5 Scope and delimitations

- **Speech type.** Read speech on cultural topics from one corpus, plus FLEURS as an out-of-domain
  test. Spontaneous and conversational speech are out of scope.
- **Speakers.** The corpus has 20 speakers; four are held out for testing. Conclusions about
  individual speakers or about gender are not drawn from four test speakers.
- **Models.** Two pretrained families: Whisper small (full fine-tuning) and MMS-1B (adapter
  training). Larger Whisper models were evaluated zero-shot only, because of the GPU budget.
- **Metric.** CER with spaces removed is the primary metric; word error rate is not reported, as it
  would depend on a word segmenter.
- **Resources.** About 23 USD of rented GPU time (one NVIDIA A40) was used for all fine-tuning, out
  of a 200 USD budget. Every experiment fits a single 48 GB GPU.
- **Licences.** MMS is licensed for non-commercial use only (CC BY-NC 4.0); Whisper is Apache-2.0.
  The dataset is CC BY-SA 4.0.

## 1.6 Contributions

1. **A full-corpus audit** of the DDD Khmer dataset that corrects its card (hours, speakers), finds
   that every sentence is read by four speakers, and finds an upload batch stored at 48 kHz instead of
   16 kHz.
2. **A leakage-free evaluation protocol**: a split that holds out speakers and their sentences, a
   normalization protocol derived from the corpus's full character inventory, and a space-insensitive
   CER, with automatic leakage checks.
3. **Controlled measurements** of fine-tuning (RQ1), data quantity (RQ2) and model family (RQ4) for
   Khmer, with confidence intervals.
4. **A Khmer-specific adaptation result** (RQ3): starting from the released Khmer adapter improves
   MMS at every data size and gives the most robust model.
5. **Two evaluation findings** with consequences beyond this corpus: testing on seen speakers would
   overstate fully fine-tuned models by about a third, and the most frequent "errors" are spelling
   variants the corpus itself mixes.
6. **Open, reproducible artefacts**: code, configurations, manifests, logs of every run, and a demo
   application.

## 1.7 Summary of results

On the leakage-free test set, zero-shot Whisper models fail (CER above 100%) while zero-shot MMS
reaches 9.71% (*ours*). Fine-tuning helps both: Whisper small reaches 4.70% with 100 hours and 2.85%
with the full 872 hours; MMS with the released Khmer adapter as its starting point reaches 4.62%
with 100 hours. Out of domain the ranking reverses — on FLEURS the best MMS model scores 14.36%,
better than zero-shot, and the best Whisper 18.61% — so the best choice depends on how closely the
target speech resembles the training corpus (Chapters 5 and 6).

## 1.8 Organization of the thesis

- **Chapter 2** reviews pretrained speech models, adaptation methods, Khmer resources, the Khmer
  script as it affects ASR, and evaluation practice, and states the research gap.
- **Chapter 3** describes the corpus audit, the leakage-free split, the normalization protocol and
  the metrics.
- **Chapter 4** describes the models, the fine-tuning setup, the experiments (M0, M1, S, M2) and the
  computing environment.
- **Chapter 5** presents the results for RQ1–RQ4, the seen-speaker comparison and the error analysis.
- **Chapter 6** answers the research questions, gives practical recommendations, and discusses
  limitations and future work.
