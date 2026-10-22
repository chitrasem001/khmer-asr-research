"""Evaluate one or more models on a manifest (default: the locked test set)."""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import transformers

from khmer_asr.evaluation.run import compute_metrics, transcribe_manifest
from khmer_asr.models.asr import ModelSpec, Transcriber
from khmer_asr.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", default=None, help="Override the config's test manifest")
    parser.add_argument("--models", nargs="*", default=None, help="Only these model names")
    parser.add_argument("--limit", type=int, default=None, help="First N clips (smoke test)")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--audio-dir", default=None, help="Override the config's audio dir")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    manifest_path = args.manifest or data_cfg["test_manifest"]
    manifest = pd.read_csv(manifest_path)
    if args.limit:
        manifest = manifest.head(args.limit)
    manifest_sha = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
    torch.manual_seed(cfg["experiment"]["seed"])
    out_root = Path(args.output_dir or cfg["experiment"]["output_dir"])

    for entry in cfg["models"]:
        if args.models and entry["name_or_path"] not in args.models:
            continue
        spec = ModelSpec(
            entry["name_or_path"], entry["revision"], entry["family"], entry["language"]
        )
        print(f"== {spec.name_or_path} on {len(manifest)} clips")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        transcriber = Transcriber(
            spec, device="cuda" if torch.cuda.is_available() else "cpu", fp16=cfg["inference"]["fp16"]
        )
        predictions, model_seconds = transcribe_manifest(
            transcriber, manifest, args.audio_dir or data_cfg["audio_dir"], entry["batch_size"]
        )
        metrics = compute_metrics(predictions, model_seconds, transcriber)

        out = out_root / spec.name_or_path.split("/")[-1]
        out.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(out / "predictions.csv", index=False, encoding="utf-8")
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        run_info = {
            "date": datetime.now().isoformat(timespec="seconds"),
            "config": args.config,
            "model": entry,
            "dataset_revision": data_cfg["revision"],
            "manifest": manifest_path,
            "manifest_sha256": manifest_sha,
            "limit": args.limit,
            "normalization": "v1 (khmer_asr.preprocessing.text)",
            "seed": cfg["experiment"]["seed"],
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
        }
        (out / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
        )
        (out / "pip_freeze.txt").write_text(freeze.stdout, encoding="utf-8")
        print(
            f"   CER {metrics['cer']:.4f} | CER w/ spaces {metrics['cer_with_spaces']:.4f} | "
            f"RTF {metrics['rtf']:.4f} | {metrics['model_seconds']} s -> {out}"
        )
        del transcriber
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
