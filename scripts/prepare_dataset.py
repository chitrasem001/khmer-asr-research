"""Clean, deduplicate and split the dataset; write manifests to data/manifests/."""
import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd

from khmer_asr.data.split import add_ids, filter_examples, split_dataset, split_summary
from khmer_asr.utils import load_config

MANIFEST_COLUMNS = ["utt_id", "source_file", "row_in_file", "speaker_id", "duration", "text"]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--allow-test-change",
        action="store_true",
        help="Overwrite a locked test set (record why in docs/decisions.md)",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    data_cfg, split_cfg = cfg["data"], cfg["split"]
    out = Path(cfg["output"]["dir"])
    out.mkdir(parents=True, exist_ok=True)

    df = add_ids(pd.read_parquet(data_cfg["metadata"]))
    kept, dropped = filter_examples(df, data_cfg["min_duration_s"], data_cfg["max_duration_s"])
    splits = split_dataset(
        kept,
        test_speakers=split_cfg["test_speakers"],
        test_hours_per_speaker=split_cfg["test_hours_per_speaker"],
        validation_hours=split_cfg["validation_hours"],
        test_seen_speakers_hours=split_cfg["test_seen_speakers_hours"],
        seed=split_cfg["seed"],
    )

    # LF line endings everywhere, so the hash is identical on Windows and Linux.
    test_csv = splits["test"][MANIFEST_COLUMNS].to_csv(index=False, lineterminator="\n").encode("utf-8")
    lock_path = out / "test.lock.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock["sha256"] != _sha256(test_csv) and not args.allow_test_change:
            raise SystemExit(
                f"Test set is locked ({lock_path}) and this config would change it. "
                "Revert the config or pass --allow-test-change and log the reason."
            )

    (out / "test.csv").write_bytes(test_csv)
    for name in ("validation", "test_seen_speakers"):
        splits[name][MANIFEST_COLUMNS].to_csv(
            out / f"{name}.csv", index=False, encoding="utf-8", lineterminator="\n"
        )
    # train.parquet is too large for git; it is rebuilt from the config and checked by hash.
    splits["train"][MANIFEST_COLUMNS].to_parquet(out / "train.parquet", index=False)
    train_ids_sha256 = _sha256("\n".join(splits["train"]["utt_id"]).encode("utf-8"))

    report = {
        "dataset": data_cfg["dataset"],
        "revision": data_cfg["revision"],
        "split_option": split_cfg["option"],
        "config": cfg,
        "input_utterances": len(df),
        "dropped": dropped,
        "splits": split_summary(splits),
        "test_csv_sha256": _sha256(test_csv),
        "train_utt_ids_sha256": train_ids_sha256,
        "test_locked": bool(split_cfg["lock_test"]) or lock_path.exists(),
    }
    (out / "split_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if split_cfg["lock_test"] and (not lock_path.exists() or args.allow_test_change):
        lock = {"sha256": _sha256(test_csv), "config": args.config, "locked_on": str(date.today())}
        lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
        print(f"Locked test set: {lock_path}")

    for name, s in report["splits"].items():
        print(f"{name:20} {s['utterances']:>8} utts {s['hours']:>8} h {s['speakers']:>3} speakers")
    print(f"Dropped: {dropped}")


if __name__ == "__main__":
    main()
