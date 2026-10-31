"""Model, processor and trainer setup for Whisper (seq2seq) and MMS (CTC adapter) fine-tuning."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoFeatureExtractor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from khmer_asr.evaluation.metrics import cer
from khmer_asr.training.data import (
    ClipDataset,
    CTCCollator,
    WhisperCollator,
    build_ctc_vocab,
    whisper_label_lengths,
)

WHISPER_MAX_LABEL_TOKENS = 448


def _common_args(cfg: dict, output_dir: str) -> dict:
    t = cfg["training"]
    return {
        "output_dir": output_dir,
        "per_device_train_batch_size": t["per_device_train_batch_size"],
        "per_device_eval_batch_size": t.get("per_device_eval_batch_size", t["per_device_train_batch_size"]),
        "gradient_accumulation_steps": t["gradient_accumulation_steps"],
        "learning_rate": float(t["learning_rate"]),
        "warmup_steps": t["warmup_steps"],
        "max_steps": t["max_steps"],
        "gradient_checkpointing": t.get("gradient_checkpointing", False),
        "fp16": t["fp16"] and torch.cuda.is_available(),
        "eval_strategy": "steps",
        "eval_steps": t["eval_steps"],
        "save_steps": t["save_steps"],
        "logging_steps": t["logging_steps"],
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "cer",
        "greater_is_better": False,
        "report_to": [],
        # training.seed varies initialisation and data order only; subsets use experiment.seed
        "seed": t.get("seed", cfg["experiment"]["seed"]),
        "dataloader_num_workers": t.get("dataloader_num_workers", 2),
        "remove_unused_columns": False,
    }


def build_whisper(cfg: dict, train: pd.DataFrame, val: pd.DataFrame, audio_dir: str, output_dir: str):
    m = cfg["model"]
    processor = WhisperProcessor.from_pretrained(m["name_or_path"], revision=m["revision"])
    processor.tokenizer.set_prefix_tokens(language=m["language"], task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(m["name_or_path"], revision=m["revision"])
    model.generation_config.language = m["language"]
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False  # needed with gradient checkpointing
    if m.get("freeze_encoder"):
        model.freeze_encoder()

    # Whisper's decoder holds at most 448 positions; longer transcripts cannot be learned, and
    # the trainer cannot compute an eval loss for them either, so both sets are filtered.
    # The final test evaluation (scripts/evaluate.py) only generates and keeps every clip.
    def fits(frame):
        lengths = whisper_label_lengths(processor.tokenizer, frame["text"].tolist())
        return frame[[n <= WHISPER_MAX_LABEL_TOKENS for n in lengths]]

    n_train, n_val = len(train), len(val)
    train, val = fits(train), fits(val)
    dropped = {"dropped_too_long_train": n_train - len(train), "dropped_too_long_eval": n_val - len(val)}

    def compute_metrics(pred):
        label_ids = np.where(pred.label_ids != -100, pred.label_ids, processor.tokenizer.pad_token_id)
        hyps = processor.batch_decode(pred.predictions, skip_special_tokens=True)
        refs = processor.batch_decode(label_ids, skip_special_tokens=True)
        return {"cer": cer(refs, hyps)}

    args = Seq2SeqTrainingArguments(
        **_common_args(cfg, output_dir),
        predict_with_generate=True,
        generation_max_length=WHISPER_MAX_LABEL_TOKENS,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=ClipDataset(train, audio_dir),
        eval_dataset=ClipDataset(val, audio_dir),
        data_collator=WhisperCollator(processor, model.config.decoder_start_token_id),
        compute_metrics=compute_metrics,
        processing_class=processor,
    )
    return trainer, processor, dropped


# Our CTC special tokens and their names in the released MMS vocabularies.
RELEASED_SPECIAL = {"[PAD]": "<pad>", "[UNK]": "<unk>"}


def head_row_map(new_vocab: dict[str, int], released_vocab: dict[str, int]) -> dict[int, int]:
    """New output-layer row -> released row, for every token both vocabularies share."""
    rows = {}
    for token, i in new_vocab.items():
        old = released_vocab.get(RELEASED_SPECIAL.get(token, token))
        if old is not None:
            rows[i] = old
    return rows


def load_released_adapter(model, new_vocab: dict[str, int], m: dict) -> dict:
    """Start from the released language adapter instead of a random one (M2).

    Adapter weights are copied as they are; output-layer rows are copied for shared tokens, so
    tokens the released vocabulary lacks (e.g. Khmer digits) keep their fresh initialisation.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    lang = m["init_adapter"]
    weights = load_file(hf_hub_download(m["name_or_path"], f"adapter.{lang}.safetensors",
                                        revision=m["revision"]))
    vocab_path = hf_hub_download(m["name_or_path"], "vocab.json", revision=m["revision"])
    released_vocab = json.loads(Path(vocab_path).read_text(encoding="utf-8"))[lang]

    state = model.state_dict()
    adapter = {k: v for k, v in weights.items() if not k.startswith("lm_head.")}
    missing = set(adapter) - set(state)
    if missing:
        raise ValueError(f"Released adapter has keys the model lacks: {sorted(missing)[:3]}")
    rows = head_row_map(new_vocab, released_vocab)
    with torch.no_grad():
        model.load_state_dict(adapter, strict=False)
        new_idx, old_idx = list(rows), list(rows.values())
        model.lm_head.weight[new_idx] = weights["lm_head.weight"][old_idx].to(model.lm_head.weight.dtype)
        model.lm_head.bias[new_idx] = weights["lm_head.bias"][old_idx].to(model.lm_head.bias.dtype)
    return {
        "init_adapter": lang,
        "adapter_tensors_copied": len(adapter),
        "head_rows_copied": len(rows),
        "head_rows_new": sorted(t for t, i in new_vocab.items() if i not in rows),
    }


def build_mms(cfg: dict, train: pd.DataFrame, val: pd.DataFrame, audio_dir: str, output_dir: str):
    """Train a Khmer adapter + CTC head on a vocabulary built from the training text.

    The released `khm` adapter's vocabulary lacks some Khmer digits, so it cannot learn them.
    By default the adapter starts from scratch; `model.init_adapter: khm` starts it from the
    released adapter instead (M2).
    """
    m = cfg["model"]
    vocab = build_ctc_vocab(train["text"])
    vocab_dir = Path(output_dir) / "vocab"
    vocab_dir.mkdir(parents=True, exist_ok=True)
    vocab_file = vocab_dir / "vocab.json"
    vocab_file.write_text(json.dumps({m["language"]: vocab}, ensure_ascii=False), encoding="utf-8")
    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_file),
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
        target_lang=m["language"],
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(m["name_or_path"], revision=m["revision"])
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    model = Wav2Vec2ForCTC.from_pretrained(
        m["name_or_path"],
        revision=m["revision"],
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        pad_token_id=tokenizer.pad_token_id,
        vocab_size=len(tokenizer),
        ignore_mismatched_sizes=True,
    )
    model.init_adapter_layers()
    init_notes = load_released_adapter(model, vocab, m) if m.get("init_adapter") else {}
    model.freeze_base_model()
    for param in model._get_adapters().values():
        param.requires_grad = True

    def compute_metrics(pred):
        ids = np.argmax(pred.predictions, axis=-1)
        label_ids = np.where(pred.label_ids != -100, pred.label_ids, tokenizer.pad_token_id)
        hyps = processor.batch_decode(ids)
        refs = processor.batch_decode(label_ids, group_tokens=False)
        return {"cer": cer(refs, hyps)}

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**_common_args(cfg, output_dir)),
        train_dataset=ClipDataset(train, audio_dir),
        eval_dataset=ClipDataset(val, audio_dir),
        data_collator=CTCCollator(processor),
        compute_metrics=compute_metrics,
        processing_class=processor,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainer, processor, {"vocab_size": len(vocab), "trainable_parameters": trainable, **init_notes}


BUILDERS = {"whisper": build_whisper, "mms": build_mms}
