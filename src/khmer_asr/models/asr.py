"""Pretrained ASR models behind one interface: `transcribe(list of 16 kHz arrays) -> list of text`."""
from dataclasses import dataclass

import numpy as np
import torch
from transformers import (
    AutoProcessor,
    Wav2Vec2ForCTC,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

SAMPLING_RATE = 16000


@dataclass
class ModelSpec:
    name_or_path: str
    revision: str | None
    family: str  # "whisper" or "mms"
    language: str | None  # Whisper code ("km"), MMS adapter code ("khm"), None for fine-tuned MMS
    max_new_tokens: int = 440  # Whisper decoder limit is 448 positions incl. 4 prompt tokens


class Transcriber:
    def __init__(self, spec: ModelSpec, device: str = "cuda", fp16: bool = True):
        self.spec = spec
        self.device = device
        self.dtype = torch.float16 if fp16 and device == "cuda" else torch.float32
        kwargs = {"revision": spec.revision, "torch_dtype": self.dtype}
        if spec.family == "whisper":
            self.processor = WhisperProcessor.from_pretrained(spec.name_or_path, revision=spec.revision)
            self.model = WhisperForConditionalGeneration.from_pretrained(spec.name_or_path, **kwargs)
        elif spec.family == "mms":
            # A released checkpoint loads its language adapter by code ("khm"). A fine-tuned
            # checkpoint already contains its own adapter and vocabulary: language is None.
            lang = {"target_lang": spec.language} if spec.language else {}
            self.processor = AutoProcessor.from_pretrained(
                spec.name_or_path, revision=spec.revision, **lang
            )
            self.model = Wav2Vec2ForCTC.from_pretrained(
                spec.name_or_path,
                ignore_mismatched_sizes=bool(lang),
                **lang,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown model family: {spec.family}")
        self.model.to(device).eval()

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @torch.inference_mode()
    def transcribe(self, audio: list[np.ndarray]) -> list[str]:
        if self.spec.family == "whisper":
            feats = self.processor(audio, sampling_rate=SAMPLING_RATE, return_tensors="pt")
            ids = self.model.generate(
                feats.input_features.to(self.device, self.dtype),
                language=self.spec.language,
                task="transcribe",
                max_new_tokens=self.spec.max_new_tokens,
            )
            return self.processor.batch_decode(ids, skip_special_tokens=True)
        inputs = self.processor(
            audio, sampling_rate=SAMPLING_RATE, return_tensors="pt", padding=True
        )
        logits = self.model(inputs.input_values.to(self.device, self.dtype)).logits
        return self.processor.batch_decode(logits.argmax(dim=-1))
