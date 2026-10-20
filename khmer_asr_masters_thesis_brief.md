# Master's Thesis Project: Khmer Automatic Speech Recognition (ASR)

## 1. Project Overview

I am pursuing a Master's degree in Data Science. I want to research, customize, fine-tune, evaluate, and demonstrate a Khmer speech-to-text model using GPU computing on RunPod.

The goal is to produce a research-based thesis, not merely a software application.

### Proposed thesis title

**An Experimental Study of Fine-Tuning Strategies for Pretrained Models in Khmer Automatic Speech Recognition**

Alternative title:

**Development and Evaluation of a Customized Deep Learning-Based Khmer Speech-to-Text System**

The first title is preferred because it emphasizes research and experimentation.

## 2. Research Direction

The main objective is to investigate how pretrained speech-recognition models can be adapted to recognize Khmer speech more accurately.

I do not want to train a large model from scratch unless experiments demonstrate that it is necessary.

The research should investigate:

1. How much does Khmer-specific fine-tuning improve speech recognition?
2. How does the amount of training data affect recognition performance?
3. Can Khmer-specific preprocessing, audio augmentation, or training strategies improve performance?
4. If feasible, how does another pretrained ASR architecture compare with the main model?
5. What are the trade-offs between recognition quality, inference speed, model size, and GPU cost?

CNN is an architecture family, not a single pretrained model. Only include CNN/CRNN if we identify a suitable pretrained speech-recognition checkpoint and a scientifically fair comparison. Whisper is a candidate, not an automatically selected final solution.

## 3. Candidate Dataset

**Digital Divide Data — Khmer Speech Dataset**

Dataset: https://huggingface.co/datasets/Digital-Divide-Data/khmer-speech-dataset

Previously reported characteristics:
- Approximately 450,000 audio/transcript examples.
- Approximately 128 GB of data.
- Audio paired with Khmer transcripts.
- Parquet-based dataset shards.
- License reported as CC BY-SA 4.0.

These figures and the current schema must be verified against the live dataset card before implementation. Review the attribution requirements and any other applicable license terms.

### Dataset audit checklist

- [ ] Inspect the latest dataset card and revision.
- [ ] Verify the license and attribution requirements.
- [ ] Inspect the actual schema, audio format, sample rate, and duration distribution.
- [ ] Identify missing, corrupted, duplicated, or low-quality examples.
- [ ] Calculate the usable number of samples and total audio hours.
- [ ] Inspect speaker and topic distributions.
- [ ] Determine whether reliable speaker metadata is available.
- [ ] Check the official train/validation/test split, if one exists.
- [ ] Create a reproducible dataset manifest.
- [ ] Avoid downloading the full dataset before the pilot experiment succeeds.

## 4. Research Problem

Khmer has fewer speech-recognition resources than languages with much larger labeled speech corpora. Pretrained multilingual models provide a foundation, but their recognition quality and suitability for Khmer need to be measured.

The research will investigate whether controlled fine-tuning and Khmer-specific adaptations can improve recognition performance.

Do not claim that a particular research gap is new until recent papers, existing Khmer ASR models, and prior experiments have been reviewed.

## 5. Research Questions

**RQ1:** How much does fine-tuning a pretrained ASR model on Khmer speech improve Character Error Rate compared with the same model without additional training?

**RQ2:** How does the quantity of Khmer fine-tuning data affect recognition performance when the test set remains fixed?

**RQ3:** Do selected Khmer-specific preprocessing or training strategies improve recognition performance over a controlled baseline?

**Optional RQ4:** How do two suitable pretrained ASR architectures compare in recognition quality, inference speed, model size, and training cost?

Refine these questions after the literature review and initial feasibility experiments.

## 6. Proposed Experiments

### M0 — Pretrained baseline

Evaluate a selected pretrained ASR model without additional fine-tuning on Khmer.

Record:
- CER
- WER, if a consistent Khmer segmentation protocol is available
- Inference latency
- Model size
- Evaluation dataset size and audio duration

### M1 — Khmer fine-tuning

Fine-tune the same model using a documented subset of the Khmer speech dataset.

Keep the test set fixed and use the validation set for model selection.

Measure whether fine-tuning improves recognition quality.

### M2 — Customized training strategy

Add one clearly defined improvement, such as:
- Khmer transcript normalization
- Appropriate audio augmentation
- A different fine-tuning strategy
- Carefully controlled hyperparameter optimization

Change one major variable at a time whenever possible. This helps identify the cause of any improvement.

### Optional M3 — Second pretrained architecture

If resources and time permit, evaluate a second suitable pretrained ASR model.

Do not force a CNN-versus-Transformer comparison unless suitable models and fair experimental conditions can be established.

### Training-data scaling

Consider subsets such as 10, 50, and 100 hours only after auditing the dataset and confirming that these quantities are available and affordable to train.

Keep the test set unchanged across experiments.

## 7. Evaluation Methodology

### Primary metric: Character Error Rate (CER)

CER is the primary metric because character-level evaluation is useful for Khmer transcription.

Define the normalization rules before evaluating models. Document how Unicode normalization, whitespace, punctuation, and other text transformations are handled.

### Secondary metrics

- WER, using a documented and consistent Khmer segmentation method.
- Inference latency or real-time factor.
- Model size.
- Peak GPU memory usage.
- Training time and estimated GPU cost.
- Substitution, insertion, and deletion error analysis.

Do not compare metrics from different sources unless their evaluation protocols are compatible.

### Data leakage prevention

- Lock the test set before model selection.
- Deduplicate examples before splitting.
- Where feasible, use speaker-independent evaluation.
- Inspect the number and diversity of speakers before selecting a split.
- Document limitations if the dataset has too few independent speakers.
- Do not claim generalization to spontaneous conversations if the corpus mainly contains read or prompted speech.

## 8. Model Selection

Investigate the following model categories:

1. **Whisper:** pretrained encoder-decoder ASR.
2. **Another pretrained ASR model:** select one with a documented fine-tuning pipeline.
3. **CNN/CRNN-based ASR:** consider only if a suitable pretrained speech-recognition checkpoint is available.

For every candidate, investigate:
- Existing Khmer recognition performance.
- Pretraining data and training task.
- Whether it has already been fine-tuned on this dataset.
- License and usage restrictions.
- Khmer tokenizer and decoder support.
- Training requirements and VRAM.
- Available implementation and documentation.
- Whether it offers a meaningful comparison for the thesis.

Do not assume CNN versus Transformer is automatically a fair comparison.

## 9. RunPod Infrastructure

Use RunPod GPU instances for fine-tuning and evaluation.

### Workflow

1. Develop preprocessing and evaluation scripts locally.
2. Store code in Git.
3. Launch a RunPod instance for a small pilot.
4. Use persistent storage or reliable external storage for checkpoints and artifacts.
5. Record the GPU model, configuration, software versions, model revision, seeds, training time, and cost.
6. Save checkpoints and results before stopping the pod.
7. Stop idle GPU instances and verify storage charges separately.

Do not estimate training time or GPU cost until the model, dataset subset, batch size, sequence length, hardware, and current pricing have been checked.

## 10. Suggested Software Stack

- Python
- PyTorch
- Hugging Face Transformers and Datasets, where compatible
- An audio-processing library compatible with the chosen model
- A documented CER/WER evaluation implementation
- Git
- CSV/JSON experiment logs or an experiment-tracking tool

Pin working dependency versions and follow the current documentation for the selected model.

## 11. Final Demonstration

Build a simple Khmer speech-to-text application.

Workflow:

1. Upload or record Khmer speech.
2. Process the audio with the selected model.
3. Display the recognized Khmer text.
4. Optionally display the model name and processing time.

The live demo is supplementary. The actual thesis results must come from the locked test set.

An audio-file upload demo is a safer fallback than relying entirely on live microphone recording during the defense.

## 12. Deliverables

- Thesis proposal.
- Literature review and verified research gap.
- Dataset audit.
- Reproducible dataset manifest and split.
- Preprocessing pipeline.
- Baseline evaluation.
- Fine-tuning scripts and checkpoints.
- Experiment configurations and logs.
- CER/WER evaluation results.
- Error analysis.
- Working demonstration application.
- Thesis report and defense presentation.

## 13. Proposed Thesis Chapters

1. Introduction and problem statement.
2. Literature review and related work.
3. Dataset and research methodology.
4. Model implementation and experimental setup.
5. Results, comparison, and error analysis.
6. Conclusion, limitations, and future work.

Follow the university's official thesis structure if it differs.

## 14. Risks and Mitigation

| Risk | Mitigation |
|---|---|
| Dataset is too large | Audit first and use a documented subset. |
| Existing Khmer models already perform well | Identify a measurable research contribution through the literature review. |
| Limited speaker diversity | Inspect metadata and limit generalization claims appropriately. |
| Data leakage | Deduplicate, define splits early, and lock the test set. |
| RunPod costs increase | Run a small pilot, set a budget, and stop idle instances. |
| Dependencies change | Pin versions and record model revisions. |
| Experiments cannot be reproduced | Save configurations, seeds, manifests, and logs. |
| Demo fails during defense | Prepare a tested audio-file demo and example outputs. |

## 15. Questions to Resolve

- What is the thesis deadline?
- What is the available RunPod budget?
- Which GPU and VRAM capacity are appropriate?
- Which pretrained models have already been evaluated on this dataset?
- How many distinct speakers are available in the current dataset?
- What Khmer CER normalization protocol will be used?
- Which research contribution offers the best balance of novelty, feasibility, and cost?

Resolve the high-impact questions before beginning large-scale training.

## 16. Instructions for Claude AI

Act as a research assistant and machine-learning engineering collaborator for this Master's thesis.

### Work in this order

1. Review the current dataset documentation and recent Khmer ASR literature.
2. Identify suitable pretrained ASR checkpoints and check whether they have already been fine-tuned on this dataset.
3. Recommend a focused research gap and defensible research questions.
4. Propose a realistic pilot experiment and RunPod configuration, stating assumptions.
5. Design the project structure, scripts, and reproducibility practices.
6. Validate the pilot before recommending large-scale training or full dataset download.

### Quality requirements

- Cite current papers, dataset cards, and model cards.
- Distinguish published results from our experimental results.
- Never fabricate metrics, citations, dataset statistics, or costs.
- State assumptions and unresolved questions.
- Prefer small, reproducible experiments before complex architecture changes.
- Keep the project feasible within a Master's thesis timeline and finite GPU budget.
- Respect licensing, attribution, privacy, and dataset-use restrictions.

### Initial reference

Digital Divide Data Khmer Speech Dataset:
https://huggingface.co/datasets/Digital-Divide-Data/khmer-speech-dataset

Treat this as a starting point, not an endorsement of any particular experimental design. Verify current documentation before implementation.
