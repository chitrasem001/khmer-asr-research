"""Diagnose zero-shot Whisper on Khmer: detected language, and fp16 vs fp32 decoding.

Writes experiments/m0_baseline/whisper_diagnostics.json.
"""
import argparse
import collections
import json
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from khmer_asr.data.audio import clip_path
from khmer_asr.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--clips-per-speaker", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg, seed = cfg["data"], cfg["experiment"]["seed"]
    manifest = pd.read_csv(data_cfg["test_manifest"])
    clips = manifest.groupby("speaker_id").sample(n=args.clips_per_speaker, random_state=seed)
    audio = [
        sf.read(clip_path(data_cfg["audio_dir"], u), dtype="float32")[0] for u in clips["utt_id"]
    ]

    report = {"clips": len(clips), "seed": seed, "max_new_tokens": args.max_new_tokens, "models": {}}
    for entry in cfg["models"]:
        if entry["family"] != "whisper":
            continue
        name, rev = entry["name_or_path"], entry["revision"]
        print(f"== {name}", flush=True)
        proc = WhisperProcessor.from_pretrained(name, revision=rev)
        feats = proc(audio, sampling_rate=16000, return_tensors="pt").input_features
        outputs = {}
        for dtype in (torch.float16, torch.float32):
            model = WhisperForConditionalGeneration.from_pretrained(
                name, revision=rev, torch_dtype=dtype
            ).cuda().eval()
            texts, langs = [], []
            with torch.inference_mode():
                for i in range(0, len(feats), 8):
                    batch = feats[i : i + 8].cuda().to(dtype)
                    ids = model.generate(
                        batch, language="km", task="transcribe", max_new_tokens=args.max_new_tokens
                    )
                    texts += proc.batch_decode(ids, skip_special_tokens=True)
                    if dtype == torch.float32:
                        lang_ids = model.detect_language(batch)
                        langs += [proc.tokenizer.decode(t).strip("<|>") for t in lang_ids.tolist()]
            outputs[str(dtype)] = texts
            if langs:
                detected = collections.Counter(langs)
            del model
            torch.cuda.empty_cache()
        same = sum(
            a == b for a, b in zip(outputs["torch.float16"], outputs["torch.float32"], strict=True)
        )
        report["models"][name] = {
            "detected_language_without_forcing": dict(detected.most_common()),
            "fp16_equals_fp32_outputs": same,
        }
        print(report["models"][name], flush=True)

    out = Path(cfg["experiment"]["output_dir"]) / "whisper_diagnostics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
