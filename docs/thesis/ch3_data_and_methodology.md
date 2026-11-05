# Chapter 3 — Dataset and Methodology

Draft. Every figure in this chapter is computed by code in the repository from the dataset at a
pinned revision; the file that holds each figure is named in brackets. The split and the
normalization protocol are provisional until approved; the test set is then locked
(`lock_test: true` in `configs/data.yaml`).

---

## 3.1 The corpus

The study uses the **Khmer ASR Cultural Dataset** from Digital Divide Data [10], licensed CC BY-SA
4.0, at the pinned revision `e872b0c58b4bbbe30c0244def61e39e7b57ea093`. It contains read sentences
on Cambodian cultural topics, stored as 4,447 parquet files with audio and metadata (speaker,
transcript, topic, duration).

### 3.1.1 Full-corpus audit

The dataset card is internally inconsistent, so before any modelling we audited **every file**,
reading only the metadata columns so that the 128 GB of audio did not have to be downloaded
(`scripts/audit_dataset.py`, about 7 minutes; [`data/manifests/audit_report.json`]).

*Table 3.1. Dataset card versus our audit.*

| Item | Dataset card | Our audit |
|---|---|---|
| Utterances | 450,396 | 450,396 |
| Total audio | 727.94 h | **1,073.91 h** |
| Speakers with audio | 12 | **20** (9 female, 11 male, aged 18–66) |
| Unique sentences | not stated | 112,516 |
| Topics / subtopics | 61 / 1,749 | 70 / 2,087 |
| Sample rate | 16 kHz | 16 kHz, **except 71 files at 48 kHz** |

Clip durations have a median of 8.06 s (5th–95th percentile 4.5–14.4 s). 636 clips have no
duration, one is shorter than 0.5 s and 36 are longer than 30 s (the longest 1,190 s). Transcripts
have a median of 100 characters and a maximum of 384; none is empty.

### 3.1.2 Every sentence is read by four speakers

The most consequential finding is that the corpus is built from **112,516 sentences, each read
by several speakers**: 112,487 sentences are read by exactly four different speakers, in groups of
two female and two male readers (26 by five, two by six, one by one). This is not stated on the
card. It means that any split which holds out speakers only — let alone a random split — still
puts the test sentences into training, read by other people. A model could then be scored partly on
text it has memorized. Section 3.2 builds a split that prevents this.

### 3.1.3 Unbalanced speakers

Speaker hours are very unbalanced: the three largest speakers hold 61% of the audio
(256.6, 221.2 and 175.1 h), while three speakers have only 16 clips each (about 0.05 h).

*Table 3.2. Speakers, hours and role in the split (group codes from the dataset; adt4 = aged 60
or over).*

| Speaker | Hours | Role | Speaker | Hours | Role |
|---|---|---|---|---|---|
| m-adt2-0002 | 256.57 | train | m-adt3-0001 | 13.93 | **test** |
| f-adt2-0002 | 221.16 | train | f-adt1-0003 | 6.82 | **test** |
| f-adt1-0002 | 175.10 | train | m-adt2-0001 | 6.63 | train |
| m-adt1-0003 | 98.34 | train | f-adt1-0004 | 6.15 | train |
| m-adt1-0004 | 87.23 | train | m-adt4-0002 | 5.95 | not used for training |
| m-adt1-0005 | 44.98 | train | m-adt2-0003 | 3.79 | train |
| f-adt1-0005 | 41.73 | train | f-adt2-0001 | 0.05 | train |
| f-adt3-0001 | 29.77 | train | m-adt1-0001 | 0.04 | train |
| m-adt4-0001 | 29.58 | train | f-adt1-0001 | 0.04 | train |
| m-adt1-0002 | 23.76 | **test** | | | |
| f-adt4-0001 | 22.29 | **test** | | | |

### 3.1.4 A batch stored at the wrong sample rate

The audit also records the sample rate of every upload batch. The 71 files named `*-of-00724` are
stored at **48 kHz**, although the dataset declares 16 kHz. A pipeline that decodes the raw audio
bytes without resampling feeds these clips to a model at the wrong speed. We discovered this through
near-total recognition failure on exactly these clips in the first baseline run (one test speaker at
19% CER while the others were at 7–10%); the affected clips were 129 test clips, 27 seen-speaker
test clips and 1,794 training clips (4.9 h). All audio is now resampled to 16 kHz mono at extraction
with a high-quality resampler (soxr), evaluation refuses any clip that is not 16 kHz, and the first
baseline results were discarded and rerun.

## 3.2 A leakage-free split

### 3.2.1 Rule

1. Test speakers never appear in training.
2. **No sentence read by a test speaker appears in training**, whoever reads it.
3. Training speakers' readings of the test sentences are set aside. A sample of them forms a second
   test set of **unseen sentences read by known speakers** (`test_seen_speakers`).
4. A validation set of whole held-out sentences, read by training speakers, is used for checkpoint
   selection only; the test sets are never used for any choice.

Sentence identity is the normalized, space-free transcript (Section 3.3), so two readings of the
same sentence with different spacing or character order still count as the same sentence. Clips are
identified as `<file>:<row>`, which is unique and points directly to the audio.

### 3.2.2 Choice of test speakers

Because a test speaker removes every sentence they read from training, the choice of test speakers
decides how much training data is lost. We compared four options (`docs/reports.md`, Report 1):

*Table 3.3. Split options (pool sizes before sampling).*

| Option | Test speakers | Test pool (h) | Train pool (h) | Set aside (h) |
|---|---|---|---|---|
| A: older speakers (46+) | 5 | 101.5 | 935.7 | 36.7 |
| B: four small young speakers | 4 | 40.5 | 928.6 | 104.8 |
| C: 1 F + 1 M mid-size | 2 | 65.5 | 816.6 | 191.8 |
| **D: 2 F + 2 M, mixed age** | 4 | 66.8 | 875.8 | 131.3 |

Option A would remove all older voices from training and so test age generalization rather than
general accuracy; B tests only small, young speakers; C has only one speaker per gender. **Option D**
(f-adt1-0003, f-adt4-0001, m-adt1-0002, m-adt3-0001) covers both genders and ages from 23 to 60
while keeping most of the data for training. It is the provisional choice.

A side effect of option D is that speaker **m-adt4-0002** (5.95 h, one of only three speakers aged
60 or over) drops out of training entirely, because every sentence this speaker read was also read by
a test speaker. Some of this speaker's readings of test sentences fall into the seen-speaker test set
(73 of its 1,891 clips); Chapter 5 shows that removing them changes no result by more than 0.05
points.

### 3.2.3 Filtering and final split

Clips with a missing duration (636), shorter than 0.5 s (1) or longer than 30 s (36) are dropped
(673 in total). From the test speakers' pool a fixed sample of 2.5 h per speaker is drawn (seed 42).

*Table 3.4. Final split (option D, seed 42) [`data/manifests/split_report.json`].*

| Split | Clips | Hours | Speakers | Sentences | Use |
|---|---|---|---|---|---|
| Train | 376,255 | 872.02 | 15 | 94,060 | Fine-tuning |
| Validation | 1,288 | 3.00 | 12 | 321 | Checkpoint selection |
| **Test** | **3,712** | **9.99** | **4** | **3,600** | Main results (unseen speakers and sentences) |
| Test, seen speakers | 1,891 | 5.00 | 10 | 1,839 | Unseen sentences, known speakers |

Leakage checks run on every build: no speaker and no sentence key is shared between the test set and
training (`khmer_asr.data.validation.find_split_overlap`). The training manifest is not committed
(30 MB) but is rebuilt from the configuration and checked against recorded hashes
(`train_utt_ids_sha256 = 3dde14af…`, `test_csv_sha256 = 7788d6df…`, line endings normalized), which
also proves the split is reproducible on another machine; every cloud training run rebuilt it and
matched.

### 3.2.4 Training subsets

For the data-scaling experiment, nested subsets of 10, 50 and 100 hours are drawn from the training
split **by whole parquet file**, in one seeded random order of files, until the target is reached
(10 h ⊂ 50 h ⊂ 100 h). Drawing whole files rather than random clips keeps audio extraction cheap
(a random 10-hour clip sample would touch almost every file of the 128 GB corpus); the cost is that
small subsets cover fewer topics.

*Table 3.5. Training subsets.*

| Subset | Clips | Hours | Files |
|---|---|---|---|
| 10 h | 4,561 | 10.40 | 51 |
| 50 h | 21,193 | 50.02 | 226 |
| 100 h | 43,348 | 100.58 | 435 |
| Full | 376,255 | 872.02 | 3,759 |

### 3.2.5 Out-of-domain test set

**FLEURS** Khmer (`google/fleurs`, `km_kh`, revision `70bb2e8`, CC BY 4.0) [2] is used as an
out-of-domain test: 771 clips, 3.15 h, 335 sentences, recorded by different speakers under different
conditions with different text. Its raw transcripts are normalized with the same protocol as ours,
so our FLEURS numbers are only approximately comparable with published FLEURS results.

## 3.3 Text normalization (protocol v1)

The rules were derived from the inventory of **all 190 distinct characters** in the 112,516
sentences (`data/manifests/char_inventory.csv`). One function, `normalize_khmer_text`, is applied to
training targets and to both sides of every score.

*Table 3.6. Normalization rules.*

| Found in the transcripts | Sentences | Rule |
|---|---|---|
| Zero-width characters | 6,233 | Remove |
| Punctuation (។ ៖ " « » ( ) , -) | e.g. ។ in 112,259 | Replace with a space (not spoken) |
| Spoken marks % & # @, symbols, repetition sign ៗ | — | Keep (read aloud) |
| Arabic / Khmer digits | 85 / 4,475 | Map Arabic to Khmer digits |
| Latin words | 1,642 | Keep, lowercased |
| Legacy character order: subscript ro first / vowel before subscript / nikahit before u | 737 / 114 / 39 | Reorder to the standard order |
| Newlines / double spaces | 500 / 1,704 | Collapse whitespace |

Text is first put in Unicode NFC form, and deprecated characters (found in 2 sentences) are
replaced. Khmer digits are
kept rather than mapped to Arabic digits because they dominate the corpus (4,475 sentences against
85). Latin words are kept because they are spoken (for example vitamin names). After the rules, one
doubled-ro typo remains in the corpus and is left as is. Unit tests cover every rule.

**Known gap.** The error analysis (Chapter 5) shows that the corpus uses two sound-identical
spellings in two places — subscript da and ta, and the two forms of the vowel in ឲ្យ / ឱ្យ — and
that these are the most frequent errors of every model. Protocol v1 does not merge them. A protocol
v2 that does is proposed but must be decided before the test set is locked; until then it is reported
only as a sensitivity check.

## 3.4 Metrics

- **Primary: character error rate without spaces.** CER = (S + D + I) / N, computed on normalized
  text with all spaces removed, summed over the corpus (not averaged per clip). Khmer word spacing is
  optional and inconsistent (1,704 sentences contain double spaces), so spaces would add errors
  unrelated to recognition.
- **Secondary:** CER with spaces; the substitution, deletion and insertion rates; CER per speaker;
  the number of clips above 100% CER (decoding loops) and of empty outputs; real-time factor (RTF);
  parameter count and peak GPU memory.
- **WER is not reported**, because it would depend on a Khmer word segmenter.
- **Uncertainty.** 95% bootstrap intervals over clips (1,000 resamples, seed 42) for each CER, and
  paired bootstrap intervals for the difference between two models on the same clips [23]. Because
  errors of one speaker are correlated, intervals over clips may be optimistic (Section 6.5).

All scoring goes through `khmer_asr.evaluation.metrics`, so normalization is identical everywhere.

## 3.5 Experimental design

Each experiment changes **one variable** relative to a reference run; all use the same test sets,
normalization and evaluation code, and checkpoints are selected on validation only.

*Table 3.7. Experiments.*

| ID | Experiment | Variable changed | RQ |
|---|---|---|---|
| M0 | Pretrained models without fine-tuning | — (baseline) | RQ1 |
| M1 | Fine-tune Whisper small and MMS-1B on 10 h | Khmer fine-tuning | RQ1 |
| S | Same, on 50 h and 100 h (3.5 passes each), and Whisper on the full split (1 pass) | Training hours | RQ2 |
| M2 | MMS adapter started from the released Khmer adapter, 10 h and 100 h | Adapter initialization | RQ3 |
| M3 | Compare the two families across all of the above | Model family | RQ4 |

Robustness checks: a second training seed for both 10-hour MMS runs, scoring on the seen-speaker
test set, and a character-level error analysis.
