"""Audit the Khmer speech dataset (schema, durations, speakers, duplicates) without audio download."""
import argparse
import json
import time
from pathlib import Path

import pandas as pd

from khmer_asr.data.audit import (
    char_inventory,
    check_audio_sample,
    list_parquet_files,
    load_metadata,
    resolve_revision,
    speaker_table,
    summarize,
    text_issues,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="Digital-Divide-Data/khmer-speech-dataset")
    parser.add_argument("--revision", default=None, help="Commit hash; default = latest")
    parser.add_argument("--max-files", type=int, default=None, help="Limit for a quick run")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", default="data/manifests/audit_report.json")
    parser.add_argument("--metadata-out", default="data/processed/metadata.parquet")
    parser.add_argument(
        "--metadata-in", default=None, help="Summarize a saved metadata parquet instead of reading remotely"
    )
    args = parser.parse_args()

    start = time.time()
    revision = resolve_revision(args.dataset, args.revision)
    paths = list_parquet_files(args.dataset, revision)
    files_total = len(paths)
    if args.max_files:
        paths = paths[: args.max_files]

    if args.metadata_in:
        print(f"Summarizing {args.metadata_in} ({args.dataset}@{revision})")
        df = pd.read_parquet(args.metadata_in)
        read_files = set(df["source_file"])
        paths = sorted(p for p in paths if p.rsplit("/", 1)[-1] in read_files)
    else:
        print(f"{args.dataset}@{revision}: reading {len(paths)}/{files_total} files")
        df = load_metadata(paths, workers=args.workers)
    speakers = speaker_table(df)
    sentences = df["transcript"].dropna().drop_duplicates()
    inventory = char_inventory(sentences)
    report = {
        "dataset": args.dataset,
        "revision": revision,
        "files_read": len(paths),
        "files_total": files_total,
        "summary": summarize(df),
        "speakers": speakers.to_dict(orient="records"),
        "unique_raw_transcripts": len(sentences),
        "text_issues_in_sentences": text_issues(sentences),
        "distinct_chars": len(inventory),
        "audio_check_first_file": check_audio_sample(paths[0]),
        "elapsed_s": round(time.time() - start, 1),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    speakers.to_csv(out.with_name("audit_speakers.csv"), index=False)
    inventory.to_csv(out.with_name("char_inventory.csv"), index=False, encoding="utf-8")
    print(f"Wrote {out}, audit_speakers.csv and char_inventory.csv")
    if not args.metadata_in:
        meta = Path(args.metadata_out)
        meta.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(meta, index=False)
        print(f"Wrote {meta}")


if __name__ == "__main__":
    main()
