# Chapter 2 — Literature Review

Draft for the thesis *An Experimental Study of Fine-Tuning Strategies for Pretrained Models in Khmer
ASR* (Sem Chitra; advisor Dr. Mao Makara). Last literature search: **5 Nov 2026** (arXiv, ACL
Anthology, general web; IEEE Xplore not searched directly — repeat before submission).

Conventions: numbers reported by other authors are marked *published*; numbers from this thesis
are marked *ours*. Model cards are not peer-reviewed and are marked as such. References are listed
at the end; every entry was opened and checked when it was added (dates in `docs/decisions.md`).

---

## 2.1 Overview

Automatic speech recognition (ASR) for Khmer sits where three lines of work meet: large pretrained
speech models that cover many languages, methods for adapting those models cheaply to one
language, and the practical problems of a low-resource language with its own script. This chapter
reviews each line, then the small body of Khmer-specific work, and ends with the gap this thesis
addresses (Section 2.8, Table 2.3).

## 2.2 From pipelines to end-to-end recognition

Classical ASR combined an acoustic model (hidden Markov models with Gaussian mixtures, later with
neural networks), a pronunciation lexicon and an n-gram language model. The first Khmer
broadcast-news recognizer was built this way and had to choose between word and sub-word units
because written Khmer does not mark word boundaries [6].

End-to-end models replace this pipeline with one network trained from audio and text pairs. Two
families matter for this thesis:

- **CTC models.** Connectionist temporal classification [13] trains a network to emit one label
  (or a blank) per audio frame and sums over all alignments, so no frame-level segmentation is
  needed. Decoding is fast and non-autoregressive, and the output is a character sequence, which
  suits a script without word spaces. MMS (Section 2.3) is a CTC model.
- **Attention encoder-decoder models.** A Transformer [15] encoder reads the audio and an
  autoregressive decoder writes the text token by token, conditioning on what it has already
  written. This gives an internal language model but also the failure modes of autoregression:
  repetition loops and a hard limit on output length. Whisper (Section 2.3) is of this type.

## 2.3 Pretrained multilingual speech models

**Self-supervised pretraining.** wav2vec 2.0 [14] learns speech representations from unlabeled
audio by masking spans of latent features and solving a contrastive task; with very little labeled
data it reaches strong English results (*published*). XLSR [16] extends the method to unlabeled
audio in many languages and finds cross-lingual pretraining better than monolingual pretraining
after fine-tuning; XLS-R [17] scales this to up to 2B parameters and nearly half a million hours in
128 languages (*published*). These models need a fine-tuning step with labeled data before they can
transcribe anything.

**Massively multilingual supervised models.**

- **Whisper** [1] is an encoder-decoder trained with weak supervision on about 680,000 hours of
  audio–transcript pairs collected from the web, and transcribes many languages zero-shot. Its
  decoder has 448 positions, which bounds the length of a transcript.
- **MMS** [3] builds on wav2vec 2.0 pretraining and covers 1,107 languages for ASR. A shared 1B-
  parameter model is specialized to a language by loading a small *language adapter* and a
  language-specific output layer; the released Khmer (`khm`) adapter makes MMS a Khmer recognizer
  out of the box.
- **Omnilingual ASR** [18] extends coverage to more than 1,600 languages with models from 300M to
  7B parameters and is designed so communities can add a language from a few samples
  (*published*). It appeared after the models in this thesis were chosen; a community Khmer model
  built on it is listed in Table 2.2.

**Which pretraining adapts better?** Rouditchenko et al. [8] compare self-supervised (XLS-R) and
weakly supervised (Whisper) pretraining for languages unseen or rare in pretraining and find that
the number of hours of a language seen in pretraining predicts which model adapts better. Khmer is
a rare language for both families, so the question cannot be settled from the literature and has to
be measured — one motivation for comparing a CTC and an encoder-decoder model here (RQ4).

## 2.4 Adapting pretrained models to one language

Fine-tuning all weights of a large model works well but is expensive and produces a full copy per
language or task. **Parameter-efficient** methods train a small number of new weights instead:

- **Adapters** [19] insert small bottleneck layers between the frozen layers of a pretrained network;
  on text classification they come within 0.4% of full fine-tuning while adding 3.6% parameters
  per task (*published*). MMS's language adapters [3] apply the same idea to speech.
- **LoRA** [20] adds trainable low-rank matrices to frozen weight matrices and, for GPT-3, cuts
  trainable parameters by about 10,000 times with no added inference latency (*published*).

Two choices recur in all of these methods and are central to this thesis: **what is trained**
(all weights, as in our Whisper runs, versus a small adapter on a frozen base, as in our MMS runs)
and **where the trainable part starts** (random initialization versus an existing adapter). The
second choice is the strategy tested in our M2 experiment: the released Khmer adapter of MMS is used
as the starting point instead of a random one. We found no study that measures this choice for
Khmer.

**Controlled fine-tuning studies.** The closest methodological template is a controlled study of
Whisper fine-tuning for Assamese, another low-resource language with its own script, which changes
one factor at a time on a fixed test set and reports a 93.10% relative CER reduction over zero-shot
Whisper (*published*) [7]. This thesis follows the same one-variable-at-a-time design.

## 2.5 Benchmarks and data for Khmer

- **FLEURS** [2] provides about 12 hours of read speech per language, including Khmer (`km_kh`),
  with a fixed test split. It is the most common shared yardstick for Khmer results and is used
  here as the out-of-domain test set.
- **Common Voice** [21] is a crowdsourced multilingual corpus collected and validated by volunteers;
  some Khmer community models are trained on it (Table 2.2).
- **SEACrowd** [9] collects data and benchmarks for Southeast Asian languages, and
  **SEA-SpeechBench** [5] evaluates speech understanding across the region. On SEA-SpeechBench,
  commercial speech LLMs reach 13–30% CER on Khmer while open 3–30B models range from 77% to 571%
  (*published*) — evidence that general-purpose speech models are far from solving Khmer.
- **ECCC court speech** used in the WAT Khmer speech-translation task [12] is another Khmer speech
  resource, in a different domain and task.
- **The DDD Khmer ASR Cultural Dataset** [10] (CC BY-SA 4.0) is the largest open Khmer ASR corpus.
  Its card reports 727.94 hours and 12 speakers; our full audit finds 1,073.9 hours, 20 speakers and,
  crucially, that every sentence is read by four speakers (*ours*, Chapter 3). It is the training
  and main test corpus of this thesis.

## 2.6 Khmer script and what it means for ASR

Khmer is written in an abugida: consonant letters carry an inherent vowel, and vowels are written as
dependent signs placed before, after, above or below the consonant. Several properties of the script
shape both modelling and evaluation:

1. **No spaces between words.** Spaces mark phrases rather than words and are used inconsistently.
   Word error rate therefore depends on a word segmenter, and character error rate (CER) is the
   more reliable metric; this thesis reports CER with spaces removed as the primary metric and a
   space-sensitive CER as secondary (Chapter 3).
2. **Subscript consonants.** Consonant clusters are written with a subscript form, encoded in
   Unicode as the sign COENG followed by a consonant. Errors on these signs are a distinct error
   class (*ours*, Chapter 5).
3. **Two consonant series.** Consonants belong to two series that share many sounds and differ in
   how the following vowel is pronounced, so pairs such as ក/គ or ច/ជ can sound alike in context.
4. **Spelling variation.** Some sound-identical spellings are both in use — for example subscript
   da and ta, and the two forms of the vowel in ឲ្យ / ឱ្យ. A recognizer cannot recover a writer's
   choice from audio; our error analysis finds these are the most frequent errors of every model
   (*ours*, Chapter 5).
5. **Encoding order.** The same visible text can be typed in different Unicode orders (legacy
   orders of vowels and subscripts), so text must be normalized before training and scoring.

Prior Khmer work rarely documents how it handles points 1, 4 and 5, which is one reason reported
numbers are hard to compare (Section 2.7).

## 2.7 Khmer ASR results so far

**Peer-reviewed work.** Beyond the HMM-era broadcast-news system [6], the most relevant recent
results come from multilingual studies: BuzzASR [4] trains one Whisper-large-v3 model per language
and reports, for Khmer FLEURS, 103.19% CER zero-shot and 25.99% after full fine-tuning
(*published*). No peer-reviewed study we found fine-tunes a modern pretrained model on the DDD
corpus.

**Community models (model cards, not peer-reviewed).**

*Table 2.2. Khmer models with reported results (as stated on their cards; not verified by us).*

| Model | Base | Training data (as stated) | Reported result |
|---|---|---|---|
| BuzzASR/khmer [4, 11] | Whisper-large-v3 | FLEURS, Common Voice | 25.99% CER, FLEURS |
| sengtha/whisper-base-khmer [11] | Whisper base | DDD corpus + phone/room audio | 22.8% CER FLEURS; ~19% in-domain |
| Darayut/Omnilingual-ASR-Khm [11] | Omnilingual CTC 300M | unnamed Khmer corpus + code-switching | 12.03% CER, FLEURS |
| songhieng/whisper-small-khmer-v1 [22] | Whisper small | not checked | not checked |
| gagan3012/wav2vec2-xlsr-khmer [22] | wav2vec2-XLSR | Common Voice, OpenSLR Khmer | not checked |

These numbers cannot be compared with each other or with ours: test sets, normalization rules and
the treatment of spaces differ or are not stated, and models trained on the DDD corpus may have seen
test sentences, since the corpus has no official split and every sentence is read by four speakers.

## 2.8 Evaluation practice

**Leakage-free splits.** Holding out speakers is standard in ASR evaluation. In the DDD corpus this
is not enough: a held-out speaker's sentences are also read by training speakers, so a model can be
tested on text it memorized. Our split holds out both speakers and every sentence they read
(Chapter 3), and we measure the size of the effect directly: fine-tuned Whisper is 27–34% more
accurate on speakers it trained on than on new ones (*ours*, Chapter 4).

**Uncertainty.** Bisani and Ney [23] introduced bootstrap confidence intervals for ASR error rates
and showed that resampling sentences gives narrower intervals than resampling speakers. This thesis
reports 95% bootstrap intervals over clips and paired bootstrap tests between systems on the same
clips; because errors from one speaker are correlated, intervals over clips may be optimistic, and
speaker-level resampling (or blockwise bootstrap [24]) is a stricter alternative noted as a
limitation.

**Normalization.** Results depend on how both sides of a score are normalized (Section 2.6). FLEURS
ships its own normalized transcripts; we apply one documented protocol to every corpus instead, so
our FLEURS numbers are only approximately comparable to published ones.

## 2.9 Research gap

*Table 2.3. What existing Khmer work covers, and what this thesis adds.*

| Aspect | Prior Khmer work | This thesis |
|---|---|---|
| Test set | FLEURS or unstated; DDD-trained models without a documented split | Leakage-free split of the DDD corpus (unseen speakers and sentences) + FLEURS |
| Normalization | Rarely documented | Protocol v1 from the corpus's full character inventory |
| Model families | One per study | CTC (MMS) and encoder-decoder (Whisper), same protocol |
| Data quantity | Not varied | Nested 10 / 50 / 100 h subsets, plus full 872 h |
| Khmer-specific strategy | Not isolated | Adapter initialisation from the released Khmer adapter, one variable changed |
| Uncertainty | Single numbers | Bootstrap intervals, paired tests, second seed |
| Speaker familiarity | Not measured | Seen- vs unseen-speaker test sets |

The searches in Section 2.1 found no controlled study of modern pretrained models fine-tuned on
Khmer that isolates, on one leakage-free test set with a documented normalization protocol, (1) the
gain from fine-tuning, (2) the effect of data quantity, (3) the effect of a Khmer-specific strategy
and (4) the difference between model families. Recent Khmer work in the searched venues concerns
scene-text recognition rather than speech [25]. This thesis addresses that gap with research
questions RQ1–RQ4.

---

## References

1. A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, I. Sutskever. Robust speech recognition via
   large-scale weak supervision. arXiv:2212.04356, 2022. https://arxiv.org/abs/2212.04356
2. A. Conneau, M. Ma, S. Khanuja, et al. FLEURS: Few-shot learning evaluation of universal
   representations of speech. arXiv:2205.12446, 2022. https://arxiv.org/abs/2205.12446
3. V. Pratap, A. Tjandra, B. Shi, et al. Scaling speech technology to 1,000+ languages.
   arXiv:2305.13516, 2023. https://arxiv.org/abs/2305.13516
4. S. Singh, A. Yadavalli, C. Arnett, A. Warstadt. BuzzASR: A swarm of 100+ monolingual speech
   recognition models. Findings of EMNLP 2026. https://arxiv.org/abs/2609.09554
5. J. Liao, W. Zhang, Z. Liu, et al. SEA-SpeechBench: A large-scale multitask benchmark for speech
   understanding across Southeast Asia. EMNLP 2026. https://arxiv.org/abs/2609.09672
6. S. Seng, S. Sam, L. Besacier, B. Bigi, E. Castelli. First broadcast news transcription system for
   Khmer language. LREC 2008. https://aclanthology.org/L08-1123/
7. G. Das, D. Laskar, H. A. Ahmed, et al. Robust Assamese speech recognition through controlled
   fine-tuning of Whisper models. arXiv:2607.17164, 2026. https://arxiv.org/abs/2607.17164
8. A. Rouditchenko, S. Khurana, S. Thomas, et al. Comparison of multilingual self-supervised and
   weakly-supervised speech pre-training for adaptation to unseen languages. Interspeech 2023.
   https://arxiv.org/abs/2305.12606
9. H. Lovenia et al. SEACrowd: A multilingual multimodal data hub and benchmark suite for Southeast
   Asian languages. arXiv:2406.10118, 2024. https://arxiv.org/abs/2406.10118
10. Digital Divide Data. Khmer ASR Cultural Dataset. CC BY-SA 4.0, revision e872b0c.
    https://huggingface.co/datasets/Digital-Divide-Data/khmer-speech-dataset
11. Model cards (accessed Oct 2026): https://huggingface.co/BuzzASR/khmer ·
    https://huggingface.co/sengtha/whisper-base-khmer · https://huggingface.co/Darayut/Omnilingual-ASR-Khm
12. WAT Khmer–English/French speech translation task (ECCC corpus).
    https://lotus.kuee.kyoto-u.ac.jp/WAT/speech-khenfr/
13. A. Graves, S. Fernández, F. Gomez, J. Schmidhuber. Connectionist temporal classification:
    Labelling unsegmented sequence data with recurrent neural networks. ICML 2006, pp. 369–376.
    https://doi.org/10.1145/1143844.1143891
14. A. Baevski, H. Zhou, A. Mohamed, M. Auli. wav2vec 2.0: A framework for self-supervised learning of
    speech representations. arXiv:2006.11477, 2020 (NeurIPS 2020). https://arxiv.org/abs/2006.11477
15. A. Vaswani et al. Attention is all you need. NeurIPS 2017, arXiv:1706.03762.
    https://arxiv.org/abs/1706.03762
16. A. Conneau, A. Baevski, R. Collobert, A. Mohamed, M. Auli. Unsupervised cross-lingual
    representation learning for speech recognition. arXiv:2006.13979, 2020 (Interspeech 2021).
    https://arxiv.org/abs/2006.13979
17. A. Babu, C. Wang, et al. XLS-R: Self-supervised cross-lingual speech representation learning at
    scale. arXiv:2111.09296, 2021. https://arxiv.org/abs/2111.09296
18. Omnilingual ASR team (FAIR at Meta). Omnilingual ASR: Open-source multilingual speech recognition
    for 1600+ languages. arXiv:2511.09690, 2025. https://arxiv.org/abs/2511.09690
19. N. Houlsby et al. Parameter-efficient transfer learning for NLP. ICML 2019 (PMLR 97),
    arXiv:1902.00751. https://arxiv.org/abs/1902.00751
20. E. J. Hu, Y. Shen, et al. LoRA: Low-rank adaptation of large language models. arXiv:2106.09685,
    2021. https://arxiv.org/abs/2106.09685
21. R. Ardila et al. Common Voice: A massively-multilingual speech corpus. LREC 2020, pp. 4218–4222.
    https://aclanthology.org/2020.lrec-1.520
22. Community model cards (found 5 Nov 2026, contents not checked):
    https://huggingface.co/collections/songhieng/model-automatic-speech-recognition ·
    https://huggingface.co/gagan3012/wav2vec2-xlsr-khmer
23. M. Bisani, H. Ney. Bootstrap estimates for confidence intervals in ASR performance evaluation.
    ICASSP 2004, vol. 1, pp. 409–412. https://doi.org/10.1109/ICASSP.2004.1326009
24. Statistical testing on ASR performance via blockwise bootstrap. arXiv:1912.09508.
    https://arxiv.org/abs/1912.09508
25. KhmerST: A low-resource Khmer scene text detection and recognition benchmark. ACCV 2024,
    arXiv:2410.18277. https://arxiv.org/abs/2410.18277

### To verify before submission
- Repeat the gap search, including IEEE Xplore and the Interspeech 2026 / ICASSP 2027 programmes.
- Author lists marked "et al." or missing in [17], [18], [24], [25]: copy them from the papers.
- Omnilingual ASR training-data figures reported by secondary sources (not used here).
