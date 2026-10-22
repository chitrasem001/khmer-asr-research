# Research Proposal

**Student:** Sem Chitra · **Advisor:** Dr. Mao Makara · **Degree:** Master's thesis
**Status:** Draft for advisory meeting (22 Oct 2026)
**Shared doc version:** https://claude.ai/artifact/3yp3ZyQTcB66zBBGdkrZ3y

## Title
An Experimental Study of Fine-Tuning Strategies for Pretrained Models in Khmer Automatic Speech Recognition

## Problem statement

Off-the-shelf multilingual speech models still transcribe Khmer poorly. Fine-tuning helps, but no
controlled, reproducible study shows which choices drive the improvement.

- Khmer is under-resourced for ASR: FLEURS has only about 12 hours of speech per language
  (Conneau et al., 2022).
- Whisper was trained on about 680,000 hours of audio (Radford et al., 2022), yet Whisper-large-v3
  is reported at 103.19% CER zero-shot on FLEURS Khmer and 25.99% after full fine-tuning
  (BuzzASR, Singh et al., 2026). *Published results, not ours.*
- Even large speech LLMs struggle: on SEA-SpeechBench, Gemini 2.5 Flash scores 13% CER and
  GPT-4o 30% on Khmer, while open 3–30B models range from 77% to 571% (Liao et al., 2026).
- Khmer is written without spaces between words, so WER depends on segmentation. CER is the more
  reliable metric.
- A large corpus now exists: the DDD Khmer ASR Cultural Dataset reports 727.94 hours under
  CC BY-SA 4.0.

The core problem is that reported Khmer results are not comparable. They use different test sets,
normalization and training data, so the separate contributions of fine-tuning, data quantity and
Khmer-specific strategies are unknown.

## Related work and research gap

- **Earlier Khmer ASR:** the first broadcast-news recognizer, built with HMM-era methods, compared
  word and sub-word units (Seng et al., LREC 2008). Other resources include the ECCC court speech
  corpus (WAT shared task).
- **Pretrained models:** Whisper (Radford et al., 2022); MMS, covering 1,107 languages
  (Pratap et al., 2023). How many hours of a language a model saw in pre-training predicts whether
  XLS-R or Whisper adapts better (Rouditchenko et al., Interspeech 2023).
- **Existing Khmer fine-tunes** (model cards, not verified by us):

| Model | Base | Training data | Reported CER |
|---|---|---|---|
| BuzzASR/khmer | Whisper-large-v3 | FLEURS, Common Voice 25 | 25.99% FLEURS |
| sengtha/whisper-base-khmer | Whisper-base | **DDD Khmer dataset** + phone/room audio | 22.8% FLEURS, ~19% in-domain |
| Darayut/Omnilingual-ASR-Khm | Omnilingual CTC 300M | Unnamed Khmer corpus + code-switching | 12.03% FLEURS |

- **Method template:** controlled Whisper fine-tuning for Assamese on a T4 GPU reports a 93.10%
  relative CER reduction (Das et al., 2026).

**Gap.** Searches of arXiv, the ACL Anthology (LREC), the ISCA archive (Interspeech, SLTU) and
ICASSP (22 Oct 2026) found no controlled study of modern pretrained models fine-tuned on Khmer
that isolates, on one locked test set with a documented normalization protocol:
(1) the gain from fine-tuning, (2) the effect of data quantity, and (3) the effect of
Khmer-specific strategies. IEEE Xplore was not searched directly; repeat the search before
submission.

## Research questions
- **RQ1:** How much does fine-tuning a pretrained ASR model on Khmer reduce CER, compared with the
  same model without fine-tuning?
- **RQ2:** How does the amount of fine-tuning data affect CER with a fixed test set?
- **RQ3:** Does one Khmer-specific strategy (normalization, augmentation or fine-tuning method)
  reduce CER beyond plain fine-tuning?
- **RQ4 (optional):** How do two pretrained architectures compare on CER, speed, size and cost?

## Methodology

**Dataset.** DDD Khmer ASR Cultural Dataset: 450,396 utterances, 61 cultural topics, 16 kHz,
CC BY-SA 4.0. Its card figures disagree, so we audited the full corpus (22 Oct 2026, revision
`e872b0c`; see `docs/reports.md`):
- 450,396 utterances and **1,073.91 h** of audio (the card says 727.94 h).
- **20 speakers** with audio: 9 female, 11 male, aged 18–66, 10 provinces (the card says 12).
- **Every sentence is read by 4 different speakers** (112,517 unique sentences). The top 3
  speakers hold 61% of the audio.

Because sentences repeat across speakers, the test set must hold out both speakers and sentences.
Split options A–D are compared in `docs/reports.md`; option D (2 F + 2 M test speakers, mixed age)
keeps 876 h for training.

**Data protocol.**
1. Audit by streaming, without a full download.
2. Deduplicate before splitting.
3. Hold out whole speakers for the test set if possible.
4. Lock the test set before model selection; choose checkpoints with the validation set only.
5. Also report results on FLEURS Khmer as an out-of-domain test.

**Normalization.** One documented Khmer protocol (Unicode form, zero-width characters, spaces,
punctuation, digits), applied in every evaluation.

**Experiments.**

| ID | Experiment | Variable changed | Answers |
|---|---|---|---|
| M0 | Pretrained model, no fine-tuning | — (baseline) | RQ1 |
| M1 | Fine-tune on a Khmer subset | Khmer training | RQ1 |
| S | Fine-tune on e.g. 10 / 50 / 100 h | Training hours | RQ2 |
| M2 | M1 + one Khmer-specific strategy | One strategy | RQ3 |
| M3 | Second architecture (e.g. MMS) | Architecture | RQ4 (optional) |

**Models.** Whisper small or medium as the main model; MMS as the second architecture if time
allows. The final choice depends on license, tokenizer support and VRAM.

**Metrics.** CER is primary. WER is reported only with documented segmentation. Secondary metrics:
real-time factor, model size, peak GPU memory, training time and cost, and a breakdown of
substitution, insertion and deletion errors.

**Reproducibility.** Record seed, config, package versions, model revision, GPU, time and cost
for every run.

## Expected contribution
1. A documented Khmer CER evaluation protocol: normalization, locked splits, scoring code.
2. Controlled results for zero-shot vs. fine-tuned performance, data scaling and one Khmer strategy.
3. An error analysis of typical Khmer recognition mistakes.
4. Open, reproducible code, configs and manifests, plus a speech-to-text demo.

**Out of scope:** spontaneous or conversational speech, new architectures, and generalization
claims beyond what the speaker count supports.

## Timeline

Start: 22 Oct 2026. Target submission: end of Feb 2027 (official deadline to be confirmed, expected ~March 2027).

| Weeks | Dates | Work |
|---|---|---|
| 1–2 | 22 Oct – 4 Nov | Initial literature review, proposal, meeting with advisor |
| 3–4 | 5 – 18 Nov | Environment setup, model selection, dataset audit |
| 5–6 | 19 Nov – 2 Dec | Data preparation, locked splits, evaluation pipeline, M0 baseline |
| 7 | 3 – 9 Dec | Small fine-tuning pilot on RunPod; measure real time and cost |
| 8–12 | 10 Dec – 13 Jan | M1 fine-tuning, M2 strategy, data scaling, optional M3 |
| 13–14 | 14 – 27 Jan | Results, error analysis, demo application |
| 15–18 | 28 Jan – 24 Feb | Thesis writing (Ch. 1–3 drafted in parallel earlier) |
| — | late Feb – Mar | Buffer, revisions, defense preparation |

## Budget

GPU compute on RunPod, on-demand pricing (runpod.io/pricing, checked 22 Oct 2026):

| Item | Assumption | Estimated cost |
|---|---|---|
| GPU: RTX A6000 / A40 (48 GB) | ~200 GPU-hours × $0.33–0.59/h | ~$70–120 |
| Network storage | ~100 GB × $0.07/GB/month × 5 months | ~$35 |
| Contingency | ~30% | ~$45 |
| **Total cap** | | **$200** (max $25 per run) |

**Actual usage so far:** an earlier RunPod session used an RTX 4090 at $0.74/h for about
7–7.5 h (~$5.20–5.55), with ~10 h (~$7.40) also reported. At that rate, the $200 cap covers about
270 GPU-hours, so the budget has headroom. The workload of that session has not been recorded yet,
so it is not used as a throughput estimate.

The GPU-hour figure is an assumption. It will be replaced with measured values after the pilot
experiment, and the budget will be revised if needed. Pods are stopped when idle, and
checkpoints are saved to persistent storage first.

## Questions for the advisor
- What is the official submission deadline, and is there a required thesis template?
- Is a controlled-comparison study (not a new model) a sufficient contribution?
- Is the DDD dataset acceptable with 20 speakers, and is split option D acceptable?
- Should RQ4 stay in scope or be dropped to save time?
- Is university GPU access or funding available?

## References
1. Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2022). Robust Speech Recognition via Large-Scale Weak Supervision. https://arxiv.org/abs/2212.04356
2. Conneau, A., Ma, M., Khanuja, S., et al. (2022). FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech. https://arxiv.org/abs/2205.12446
3. Pratap, V., Tjandra, A., Shi, B., et al. (2023). Scaling Speech Technology to 1,000+ Languages. https://arxiv.org/abs/2305.13516
4. Singh, S., Yadavalli, A., Arnett, C., & Warstadt, A. (2026). BuzzASR: A Swarm of 100+ Monolingual Speech Recognition Models. Findings of EMNLP 2026. https://arxiv.org/abs/2609.09554
5. Liao, J., Zhang, W., Liu, Z., et al. (2026). SEA-SpeechBench: A Large-Scale Multitask Benchmark for Speech Understanding Across Southeast Asia. EMNLP 2026. https://arxiv.org/abs/2609.09672
6. Seng, S., Sam, S., Besacier, L., Bigi, B., & Castelli, E. (2008). First Broadcast News Transcription System for Khmer Language. LREC 2008. https://aclanthology.org/L08-1123/
7. Rouditchenko, A., Khurana, S., Thomas, S., et al. (2023). Comparison of Multilingual Self-Supervised and Weakly-Supervised Speech Pre-Training for Adaptation to Unseen Languages. Interspeech 2023. https://arxiv.org/abs/2305.12606
8. Das, G., Laskar, D., Ahmed, H. A., et al. (2026). Robust Assamese Speech Recognition through Controlled Fine-Tuning of Whisper Models. https://arxiv.org/abs/2607.17164
9. Lovenia, H., et al. (2024). SEACrowd: A Multilingual Multimodal Data Hub and Benchmark Suite for Southeast Asian Languages. https://arxiv.org/abs/2406.10118
10. Digital Divide Data. Khmer ASR Cultural Dataset. CC BY-SA 4.0. https://huggingface.co/datasets/Digital-Divide-Data/khmer-speech-dataset · https://datacollective.mozillafoundation.org/datasets/cml9h5vgc01bxmn075sjeftek
11. Model cards: https://huggingface.co/BuzzASR/khmer · https://huggingface.co/sengtha/whisper-base-khmer · https://huggingface.co/Darayut/Omnilingual-ASR-Khm
12. WAT Khmer–English/French speech translation task (ECCC corpus). https://lotus.kuee.kyoto-u.ac.jp/WAT/speech-khenfr/
13. RunPod pricing, accessed 22 Oct 2026. https://www.runpod.io/pricing
