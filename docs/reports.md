# Progress Reports

Thesis: *An Experimental Study of Fine-Tuning Strategies for Pretrained Models in Khmer ASR*
Student: Sem Chitra · Advisor: Dr. Mao Makara

Newest report first. Each entry: what was done, what was found (with numbers), decisions, next steps.
Published results by others are marked as such; everything else is our own measurement.

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
