# Progress Reports

Thesis: *An Experimental Study of Fine-Tuning Strategies for Pretrained Models in Khmer ASR*
Student: Sem Chitra · Advisor: Dr. Mao Makara

Newest report first. Each entry: what was done, what was found (with numbers), decisions, next steps.
Published results by others are marked as such; everything else is our own measurement.

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

### Decisions
- Logged in `docs/decisions.md`: deadline, budget, GPU choice, environment.
- Split option: **pending** advisor feedback.

### Next
- Agree on the split option with Dr. Mao Makara.
- Write the Khmer normalization rules (Latin letters, quotes, brackets).
- Build `prepare_dataset.py`: filter bad clips, apply the split, lock `test.csv`.
