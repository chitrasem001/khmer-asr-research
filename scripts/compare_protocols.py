"""Rescore every saved prediction under normalization protocols v1 and v2 (no inference, no GPU).

Writes experiments/normalization_v2/:
  comparison.csv  CER under v1 and v2 (with 95% bootstrap intervals) for every run and test set
  paired.csv      the thesis's key paired comparisons recomputed under both protocols
  summary.json    split check (do v2 sentence keys leak test sentences into training?)

    python scripts/compare_protocols.py
"""
import json
from pathlib import Path

import pandas as pd

from khmer_asr.evaluation.stats import bootstrap_ci, clip_errors, paired_diff_ci
from khmer_asr.preprocessing.text import normalize_khmer_text, remove_spaces

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
OUT = EXP / "normalization_v2"
SETS = {"test": ("m0_baseline", "m1_eval"), "fleurs": ("m0_fleurs", "m1_eval_fleurs"),
        "seen": ("m1_eval_seen", "m1_eval_seen")}
# (label, run a, run b): CER(a) - CER(b) on the same clips
PAIRS = [
    ("M2 vs new adapter, 10 h", "m2_mms_khm_init_10h", "m1_mms"),
    ("M2 vs new adapter, 100 h", "m2_mms_khm_init_100h", "s_mms_100h"),
    ("M2 100 h vs zero-shot MMS", "m2_mms_khm_init_100h", "mms-1b-all"),
    ("Whisper 872 h vs M2 100 h", "f_whisper_small_872h", "m2_mms_khm_init_100h"),
    ("Whisper 100 h vs MMS new adapter 100 h", "s_whisper_small_100h", "s_mms_100h"),
    ("Whisper 50 h vs MMS new adapter 50 h", "s_whisper_small_50h", "s_mms_50h"),
]


def prediction_dirs() -> dict[tuple[str, str], Path]:
    """(run, set) -> folder with predictions.csv, for every saved evaluation."""
    found = {}
    for split, folders in SETS.items():
        for folder in set(folders):
            for f in sorted((EXP / folder).glob("*/predictions.csv")):
                found[(f.parent.name, split)] = f.parent
    return found


def load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path / "predictions.csv").sort_values("utt_id").reset_index(drop=True)


def split_check() -> dict:
    train = pd.read_parquet(ROOT / "data" / "manifests" / "train.parquet", columns=["text"])
    out = {}
    for protocol in ("v1", "v2"):
        def key(t: str, p: str = protocol) -> str:
            return remove_spaces(normalize_khmer_text(t, protocol=p))
        train_keys = set(train["text"].map(key))
        out[protocol] = {"distinct_train_sentences": len(train_keys)}
        for name in ("test", "test_seen_speakers", "validation"):
            df = pd.read_csv(ROOT / "data" / "manifests" / f"{name}.csv")
            out[protocol][f"{name}_clips_with_sentence_in_train"] = int(df["text"].map(key).isin(train_keys).sum())
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dirs = prediction_dirs()
    errs = {}
    rows = []
    for (run, split), path in sorted(dirs.items()):
        pred = load(path)
        rec = {"run": run, "set": split, "clips": len(pred)}
        for protocol in ("v1", "v2"):
            e = clip_errors(pred, protocol=protocol)
            errs[(run, split, protocol)] = e
            lo, hi = bootstrap_ci(e)
            rec[f"cer_{protocol}"] = e["errors"].sum() / e["ref_chars"].sum()
            rec[f"ci_{protocol}"] = f"{lo * 100:.2f}–{hi * 100:.2f}"
        rec["relative_change"] = rec["cer_v2"] / rec["cer_v1"] - 1
        rows.append(rec)
        print(f"{run:30s} {split:6s} v1 {rec['cer_v1'] * 100:6.2f}%  v2 {rec['cer_v2'] * 100:6.2f}%  "
              f"({rec['relative_change'] * 100:+.1f}%)", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "comparison.csv", index=False, encoding="utf-8")

    pair_rows = []
    for label, a, b in PAIRS:
        for split in SETS:
            if (a, split, "v1") not in errs or (b, split, "v1") not in errs:
                continue
            if len(errs[(a, split, "v1")]) != len(errs[(b, split, "v1")]):
                continue  # zero-shot Whisper is scored on a sample, not the same clips
            rec = {"comparison": label, "set": split}
            for protocol in ("v1", "v2"):
                d, lo, hi = paired_diff_ci(errs[(a, split, protocol)], errs[(b, split, protocol)])
                rec[f"diff_{protocol}"] = f"{d * 100:+.2f} [{lo * 100:+.2f}, {hi * 100:+.2f}]"
                rec[f"significant_{protocol}"] = lo > 0 or hi < 0
            pair_rows.append(rec)
    pd.DataFrame(pair_rows).to_csv(OUT / "paired.csv", index=False, encoding="utf-8")

    summary = {"split_check": split_check(), "runs_rescored": len(rows)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["split_check"], indent=1))
    print(pd.DataFrame(pair_rows).to_string(index=False))


if __name__ == "__main__":
    main()
