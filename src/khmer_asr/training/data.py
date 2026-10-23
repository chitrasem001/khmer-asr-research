"""Datasets and collators for fine-tuning on extracted 16 kHz FLAC clips."""
from dataclasses import dataclass

import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

from khmer_asr.data.audio import TARGET_SR, clip_path


class ClipDataset(Dataset):
    """Yields {"audio": float32 array, "text": normalized transcript} from a manifest."""

    def __init__(self, manifest: pd.DataFrame, audio_dir: str):
        self.rows = manifest[["utt_id", "text"]].to_dict("records")
        self.audio_dir = audio_dir

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        row = self.rows[i]
        audio, sr = sf.read(clip_path(self.audio_dir, row["utt_id"]), dtype="float32")
        if sr != TARGET_SR:
            raise ValueError(f"{row['utt_id']} is {sr} Hz; re-extract it")
        return {"audio": audio, "text": row["text"]}


@dataclass
class WhisperCollator:
    processor: object
    decoder_start_token_id: int

    def __call__(self, batch: list[dict]) -> dict:
        feats = self.processor.feature_extractor(
            [b["audio"] for b in batch], sampling_rate=TARGET_SR, return_tensors="pt"
        ).input_features
        labels = self.processor.tokenizer([b["text"] for b in batch], padding=True, return_tensors="pt")
        ids = labels.input_ids.masked_fill(labels.attention_mask.ne(1), -100)
        # The model prepends the decoder start token itself.
        if (ids[:, 0] == self.decoder_start_token_id).all():
            ids = ids[:, 1:]
        return {"input_features": feats, "labels": ids}


@dataclass
class CTCCollator:
    processor: object

    def __call__(self, batch: list[dict]) -> dict:
        inputs = self.processor(
            [b["audio"] for b in batch], sampling_rate=TARGET_SR, return_tensors="pt", padding=True
        )
        labels = self.processor.tokenizer([b["text"] for b in batch], padding=True, return_tensors="pt")
        ids = labels.input_ids.masked_fill(labels.attention_mask.ne(1), -100)
        out = {"input_values": inputs.input_values, "labels": ids}
        if "attention_mask" in inputs:
            out["attention_mask"] = inputs.attention_mask
        return out


def whisper_label_lengths(tokenizer, texts: list[str]) -> list[int]:
    """Token count of each transcript including Whisper's prefix tokens."""
    return [len(ids) for ids in tokenizer(texts).input_ids]


def build_ctc_vocab(texts: pd.Series) -> dict[str, int]:
    """Character vocabulary for a new CTC head: '|' for space, plus [UNK] and [PAD]."""
    chars = sorted({c for t in texts for c in t} - {" "})
    vocab = {c: i for i, c in enumerate(chars)}
    vocab["|"] = len(vocab)
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)
    return vocab


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
