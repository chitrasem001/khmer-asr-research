"""Character-level error analysis of saved predictions (no inference, no GPU).

Writes to experiments/error_analysis/:
  class_rates.csv   error rate per Khmer character class, per model and evaluation set
  confusions.csv    most frequent substitutions (reference -> hypothesis) per model and set
  subsets.json      CER on clips with digits / Latin words / transcripts too long for Whisper,
                    looping hypotheses, CER per speaker and per gender, and CER with spelling
                    variants merged (subscript da/ta, qoo forms) as a sensitivity check

    python scripts/error_analysis.py
"""
import json
from pathlib import Path

import pandas as pd
from transformers import WhisperTokenizer

from khmer_asr.evaluation.error_analysis import (
    char_class,
    char_errors,
    is_looping,
    merge_spelling_variants,
)
from khmer_asr.evaluation.metrics import cer
from khmer_asr.preprocessing.text import normalize_khmer_text
from khmer_asr.training.data import whisper_label_lengths

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
OUT = EXP / "error_analysis"
WHISPER_MAX_LABEL_TOKENS = 448
TOP_CONFUSIONS = 25

# run -> label; prediction folders per evaluation set
MODELS = {
    "mms-1b-all": "MMS-1B zero-shot",
    "s_mms_100h": "MMS-1B new adapter 100 h",
    "m2_mms_khm_init_100h": "MMS-1B khm init (M2) 100 h",
    "s_whisper_small_100h": "Whisper small 100 h",
    "f_whisper_small_872h": "Whisper small 872 h",
}
SETS = {"test": "m1_eval", "fleurs": "m1_eval_fleurs"}
ZERO_SHOT_DIRS = {"test": EXP / "m0_baseline", "fleurs": EXP / "m0_fleurs"}


def predictions(run: str, split: str) -> pd.DataFrame:
    base = ZERO_SHOT_DIRS[split] if run == "mms-1b-all" else EXP / SETS[split]
    df = pd.read_csv(base / run / "predictions.csv")
    df["hypothesis"] = df["hypothesis"].fillna("")
    norm = df["reference"].map(normalize_khmer_text)
    df["has_digit"] = norm.map(lambda t: any(char_class(c) == "digit" for c in t))
    df["has_latin"] = norm.map(lambda t: any(char_class(c) == "latin" for c in t))
    return df


def latin_chars(errs) -> set[str]:
    seen = set(errs.hits) | set(errs.deletions) | {r for r, _ in errs.substitutions}
    return {c for c in seen if char_class(c) == "latin"}


def subset_cer(df: pd.DataFrame, mask: pd.Series) -> dict:
    part = df[mask]
    if part.empty:
        return {"clips": 0, "cer": None}
    return {"clips": len(part), "cer": cer(part["reference"].tolist(), part["hypothesis"].tolist())}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tok = WhisperTokenizer.from_pretrained(EXP / "s_whisper_small_100h" / "best")
    tok.set_prefix_tokens(language="khmer", task="transcribe")
    test_refs = pd.read_csv(ROOT / "data" / "manifests" / "test.csv")
    lengths = dict(zip(test_refs["utt_id"],
                       whisper_label_lengths(tok, test_refs["text"].tolist()), strict=True))
    too_long = {u for u, n in lengths.items() if n > WHISPER_MAX_LABEL_TOKENS}

    class_rows, conf_rows, subsets = [], [], {}
    for run, label in MODELS.items():
        for split in SETS:
            df = predictions(run, split)
            refs, hyps = df["reference"].tolist(), df["hypothesis"].tolist()
            errs = char_errors(refs, hyps)
            for cls, v in sorted(errs.by_class().items()):
                class_rows.append({"model": label, "set": split, "class": cls, **v})
            total_errors = sum(errs.substitutions.values()) + sum(errs.deletions.values()) \
                + sum(errs.insertions.values())
            for (r, h), n in errs.substitutions.most_common(TOP_CONFUSIONS):
                conf_rows.append({"model": label, "set": split, "reference": r, "hypothesis": h,
                                  "count": n, "share_of_errors": n / total_errors})
            for kind, counter in (("deleted", errs.deletions), ("inserted", errs.insertions)):
                for c, n in counter.most_common(5):
                    conf_rows.append({"model": label, "set": split, "reference": c if kind == "deleted" else "",
                                      "hypothesis": c if kind == "inserted" else "", "count": n,
                                      "share_of_errors": n / total_errors})
            speakers = {s: cer(g["reference"].tolist(), g["hypothesis"].tolist())
                        for s, g in df.groupby("speaker_id")}
            genders = {g: cer(p["reference"].tolist(), p["hypothesis"].tolist())
                       for g, p in df.groupby(df["speaker_id"].str[0])}
            entry = {
                "cer": cer(refs, hyps),
                "cer_spelling_variants_merged": cer([merge_spelling_variants(t) for t in refs],
                                                    [merge_spelling_variants(t) for t in hyps]),
                "with_digits": subset_cer(df, df["has_digit"]),
                "without_digits": subset_cer(df, ~df["has_digit"]),
                "with_latin": subset_cer(df, df["has_latin"]),
                "without_latin": subset_cer(df, ~df["has_latin"]),
                "latin_reference_chars": sum(errs.reference_count(c) for c in latin_chars(errs)),
                "latin_written_as_other_script": sum(
                    n for (r, h), n in errs.substitutions.items()
                    if char_class(r) == "latin" and char_class(h) != "latin"),
                "looping_clips": int(sum(is_looping(r, h) for r, h in zip(refs, hyps, strict=True))),
                "empty_hypotheses": int((df["hypothesis"].str.strip() == "").sum()),
                "cer_by_speaker": speakers,
                "cer_by_gender": genders,
            }
            if split == "test":
                long_mask = df["utt_id"].isin(too_long)
                entry["too_long_for_whisper"] = subset_cer(df, long_mask)
                entry["fits_whisper"] = subset_cer(df, ~long_mask)
            subsets[f"{run}|{split}"] = {"label": label, **entry}
            print(f"{label:28s} {split:6s} CER {entry['cer'] * 100:6.2f}%  "
                  f"digits {entry['with_digits']['cer'] * 100:6.2f}%  "
                  f"latin {entry['with_latin']['cer'] * 100 if entry['with_latin']['cer'] else 0:6.2f}%  "
                  f"loops {entry['looping_clips']}", flush=True)

    pd.DataFrame(class_rows).to_csv(OUT / "class_rates.csv", index=False, encoding="utf-8")
    pd.DataFrame(conf_rows).to_csv(OUT / "confusions.csv", index=False, encoding="utf-8")
    (OUT / "subsets.json").write_text(json.dumps(
        {"too_long_test_clips": len(too_long), "models": subsets}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
