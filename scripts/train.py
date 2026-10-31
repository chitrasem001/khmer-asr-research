"""Fine-tune a pretrained ASR model on Khmer.

One command does the whole run (resumable): pick the training subset, extract its audio,
fine-tune, select the best checkpoint on validation, and save the model plus run records.

    python scripts/train.py --config configs/m1_whisper_small.yaml --train-hours 10 --max-steps 300
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import transformers
import yaml

from khmer_asr.data.audio import extract_clips
from khmer_asr.data.subsets import select_train_hours
from khmer_asr.training.data import seed_everything
from khmer_asr.training.finetune import BUILDERS
from khmer_asr.utils import load_config


def git_commit() -> str | None:
    out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return out.stdout.strip() or None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-hours", type=float, default=None, help="Override data.train_hours")
    parser.add_argument("--max-steps", type=int, default=None, help="Override training.max_steps")
    parser.add_argument("--eval-clips", type=int, default=None, help="Override data.eval_clips")
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override train/eval batch size (small GPUs)"
    )
    parser.add_argument("--eval-steps", type=int, default=None, help="Override eval/save steps")
    parser.add_argument("--warmup-steps", type=int, default=None, help="Override warmup steps")
    parser.add_argument(
        "--train-seed", type=int, default=None,
        help="Training seed (init, data order); the subset still uses experiment.seed",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prepare-only", action="store_true", help="Select and extract audio only")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data, exp = cfg["data"], cfg["experiment"]
    if args.train_hours is not None:
        data["train_hours"] = args.train_hours
    if args.max_steps is not None:
        cfg["training"]["max_steps"] = args.max_steps
    if args.eval_clips is not None:
        data["eval_clips"] = args.eval_clips
    if args.batch_size is not None:
        cfg["training"]["per_device_train_batch_size"] = args.batch_size
        cfg["training"]["per_device_eval_batch_size"] = args.batch_size
    if args.eval_steps is not None:
        cfg["training"]["eval_steps"] = cfg["training"]["save_steps"] = args.eval_steps
    if args.warmup_steps is not None:
        cfg["training"]["warmup_steps"] = args.warmup_steps
    if args.train_seed is not None:
        cfg["training"]["seed"] = args.train_seed
    out = Path(args.output_dir or exp["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg["training"].get("seed", exp["seed"]))

    train = select_train_hours(
        pd.read_parquet(data["train_manifest"]), data.get("train_hours"), exp["seed"]
    )
    val = pd.read_csv(data["validation_manifest"])
    if data.get("eval_clips"):
        val = val.sample(n=min(data["eval_clips"], len(val)), random_state=exp["seed"])
    train.to_csv(out / "train_subset.csv", index=False, encoding="utf-8")
    val.to_csv(out / "eval_subset.csv", index=False, encoding="utf-8")
    print(f"train: {len(train)} clips, {train['duration'].sum() / 3600:.2f} h "
          f"from {train['source_file'].nunique()} files; eval: {len(val)} clips", flush=True)

    t0 = time.time()
    for frame in (train, val):
        extract_clips(frame, data["dataset"], data["revision"], data["audio_dir"], workers=8)
    extract_s = time.time() - t0
    print(f"audio ready in {extract_s:.0f} s", flush=True)
    if args.prepare_only:
        return

    build = BUILDERS[cfg["model"]["family"]]
    trainer, processor, notes = build(cfg, train, val, data["audio_dir"], str(out / "checkpoints"))
    print(f"setup: {notes}", flush=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    result = trainer.train()
    train_s = time.time() - t1
    final_eval = trainer.evaluate()

    best = out / "best"
    trainer.save_model(str(best))
    processor.save_pretrained(str(best))

    price = cfg.get("compute", {}).get("gpu_price_per_hour")
    run = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "config": args.config,
        "resolved_config": cfg,
        "train_clips": len(train),
        "train_hours": round(train["duration"].sum() / 3600, 3),
        "eval_clips": len(val),
        "setup_notes": notes,
        "train_loss": result.training_loss,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_eval_cer": trainer.state.best_metric,
        "final_eval": final_eval,
        "extract_seconds": round(extract_s),
        "train_seconds": round(train_s),
        "peak_gpu_memory_mb": (
            round(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else None
        ),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_price_per_hour": price,
        "estimated_gpu_cost_usd": round(price * (extract_s + train_s) / 3600, 2) if price else None,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
    }
    (out / "run_info.json").write_text(json.dumps(run, indent=2, default=str), encoding="utf-8")
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    (out / "log_history.json").write_text(
        json.dumps(trainer.state.log_history, indent=2), encoding="utf-8"
    )
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
    )
    (out / "pip_freeze.txt").write_text(freeze.stdout, encoding="utf-8")
    print(f"best eval CER {trainer.state.best_metric} -> {best}; train {train_s / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
