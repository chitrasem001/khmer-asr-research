# Chapter 6 — Conclusion and Future Work

Draft. All numbers are CER with spaces removed, normalization protocol v1, on the leakage-free test
set (four unseen speakers, unseen sentences, 3,712 clips) unless marked FLEURS (771 clips) or seen
speakers (1,891 clips). Results are provisional until the test set is locked.

---

## 6.1 Summary

This thesis set out to measure, under controlled conditions, how much pretrained speech models
improve on Khmer when fine-tuned, and which choices drive the improvement. A full audit of the
largest open Khmer speech corpus showed that every sentence is read by four speakers, so we built a
split that holds out both test speakers and every sentence they read, and a single documented
normalization protocol. On this protocol we compared two model families — MMS-1B, a CTC model
adapted through a small language adapter, and Whisper small, an encoder-decoder model fine-tuned in
full — without fine-tuning, with 10 to 872 hours of training data, and with one Khmer-specific
strategy. Eleven training runs cost about 23 USD of GPU time.

## 6.2 Answers to the research questions

**RQ1 — How much does fine-tuning reduce CER?** A great deal for both families, but from very
different starting points. Zero-shot Whisper small does not work for Khmer: it recognizes the
language but its decoding loops, giving CER above 100%. After fine-tuning on 10 hours it reaches
19.43%, and on 100 hours 4.70%. Zero-shot MMS with its released Khmer adapter is already usable at
9.71%; fine-tuning a new adapter brings it to 8.43% with 10 hours and 5.24% with 100 hours.

**RQ2 — How does the amount of data matter?** Every increase helped and no curve flattened within
the range studied. From 10 to 50 to 100 hours, Whisper fell from 19.43% to 6.40% to 4.70% and MMS
from 8.43% to 6.20% to 5.24%. Whisper is far more data-hungry: it is the weaker model at 10 hours,
still slightly but significantly behind MMS at 50 hours (by 0.20 points), and significantly better at
100 hours. Trained once over the
full 872 hours it reaches 2.85% — and its validation error was still falling.

**RQ3 — Does a Khmer-specific strategy help beyond plain fine-tuning?** Yes. Starting the MMS
adapter from the released Khmer adapter, instead of from scratch, lowers CER at both data sizes and
on every test set, at no extra training cost: by 0.99 points at 10 hours (8.43% → 7.45%) and 0.61
points at 100 hours (5.24% → 4.62%) on the test set, and by 0.85 and 0.61 points on FLEURS. All
paired 95% intervals exclude zero, and a second training seed moves the 10-hour results by at most
0.16 points. The strategy also fixes a weakness of the from-scratch adapter, which had lost some of
the released adapter's generality: with 100 hours it is the only fine-tuned model better than
zero-shot MMS on FLEURS (14.36% vs 15.49%).

**RQ4 — How do the two families compare?** It depends on the domain. In domain, Whisper wins: its
best model is 1.77 points better than the best MMS model on the test set (2.85% vs 4.62%). Out of
domain, MMS wins clearly: 14.36% vs 18.61% on FLEURS, although the Whisper model saw almost nine times
more training data. MMS is also more robust to long transcripts, which Whisper's 448-token decoder
cannot produce (6.2% vs 15.3% CER on the 24 test clips whose transcripts exceed
Whisper's decoder limit). Whisper trains faster per hour of
data on the same GPU (113 vs 464 minutes for 100 hours) because MMS runs a 1B-parameter encoder.

## 6.3 Findings about evaluation

Two findings concern how Khmer ASR should be evaluated rather than which model is best.

- **Held-out speakers are essential.** On sentences read by speakers seen in training, fully
  fine-tuned Whisper is 27% (100 h) to 34% (872 h) more accurate than on new speakers, while MMS gains
  only 8–11%. A split that let test speakers into training would therefore overstate Whisper by about
  a third and distort the comparison between the families.
- **Spelling variants inflate error rates.** The two most frequent errors of every model are
  spellings the corpus itself uses interchangeably (subscript da/ta, and the two forms of the vowel in
  ឲ្យ / ឱ្យ). They cannot be recovered from audio. Counting them as equal lowers CER by 3–8% without
  changing any ranking. The remaining weak points are digits, Latin-script words — which Whisper
  tends to transliterate into Khmer — and long transcripts.

## 6.4 Recommendations

- **Speech similar to the training corpus** (read speech, similar recording conditions): use Whisper
  small fine-tuned on as much data as available (2.85% CER here). It is licensed Apache-2.0, so it can
  also be used commercially. Split long audio into segments before decoding.
- **Varied or unknown speech:** use MMS with the adapter started from the released Khmer adapter
  (14.36% on FLEURS). Its licence (CC BY-NC 4.0) restricts it to non-commercial use.
- **When adapting any model with a released language adapter:** start from that adapter rather than
  from scratch; it costs nothing and helped at every data size here.
- **When reporting Khmer ASR results:** hold out both speakers and sentences, document the
  normalization (including spelling variants), use space-insensitive CER, and include an
  out-of-domain test set.

## 6.5 Limitations

- One corpus of read cultural-domain speech with 20 speakers, four of them in the test set; results
  may not transfer to spontaneous or other-domain speech, and speaker or gender effects cannot be
  generalized.
- Most configurations are single training runs; only the 10-hour MMS runs were repeated with a
  second seed. Bootstrap intervals resample clips and may be optimistic, because errors from one
  speaker are correlated.
- The full-data Whisper model was trained for one pass and the scaling runs for 3.5, so it is not a
  point on the same curve; the full-data MMS model was not trained.
- Only Whisper small and MMS-1B were fine-tuned; larger Whisper models were evaluated zero-shot only.
- The test set and normalization protocol are provisional until approved and locked.

## 6.6 Future work

1. **MMS from the released Khmer adapter on the full 872 hours** — likely the best model out of
   domain; a second pass for the full-data Whisper model, whose validation error was still falling.
2. **Normalization protocol v2** that unifies the spelling variants found in the error analysis,
   decided before the test set is locked.
3. **A spontaneous-speech test set**, and more test speakers, to measure robustness beyond read speech
   and to allow speaker-level confidence intervals.
4. **Long-form transcription**: automatic segmentation so Whisper can handle recordings longer than its
   decoder limit.
5. **Targeted data** for the weakest classes — digits, numbers read aloud and English loanwords — and
   noise or channel augmentation for robustness.
6. **Larger or newer models** (Whisper medium or large with parameter-efficient fine-tuning; Omnilingual
   ASR) under the same protocol, so their results stay comparable with this thesis.
