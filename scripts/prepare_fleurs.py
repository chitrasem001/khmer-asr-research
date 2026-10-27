"""Prepare FLEURS Khmer (km_kh) as an out-of-domain test set.

Writes 16 kHz FLAC clips and a manifest in the same format as data/manifests/test.csv, so
scripts/evaluate.py scores it with the same normalization and metric. FLEURS is CC BY 4.0.

    python scripts/prepare_fleurs.py --split test
"""
import argparse
import io
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download

from khmer_asr.data.audio import TARGET_SR, clip_path, to_target_rate
from khmer_asr.preprocessing.text import normalize_khmer_text

REPO = "google/fleurs"
REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
GENDER = {0: "male", 1: "female"}  # FLEURS gender codes; other values kept as "other"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--split", default="test", choices=["test", "validation", "train"])
    parser.add_argument("--audio-dir", default="data/processed/fleurs_audio")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    path = hf_hub_download(
        REPO, f"parquet-data/km_kh/{args.split}-00000-of-00001.parquet",
        repo_type="dataset", revision=REVISION,
    )
    table = pq.read_table(path, columns=["id", "audio", "raw_transcription", "gender"])
    out_dir = Path(args.audio_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, rates = [], {}
    for i, rec in enumerate(table.to_pylist()):
        utt_id = f"fleurs-km-{args.split}:{i}"
        data, sr = sf.read(io.BytesIO(rec["audio"]["bytes"]))
        rates[sr] = rates.get(sr, 0) + 1
        audio = to_target_rate(data, sr)
        target = clip_path(out_dir, utt_id)
        if not target.exists():
            sf.write(target, audio, TARGET_SR, format="FLAC")
        rows.append({
            "utt_id": utt_id,
            "source_file": f"fleurs/km_kh/{args.split}",
            "row_in_file": i,
            # FLEURS has no speaker IDs; gender is the only speaker attribute.
            "speaker_id": GENDER.get(rec["gender"], "other"),
            "duration": len(audio) / TARGET_SR,
            "text": normalize_khmer_text(rec["raw_transcription"]),
            "fleurs_sentence_id": rec["id"],
        })

    manifest = pd.DataFrame(rows)
    output = Path(args.output or f"data/manifests/fleurs_km_{args.split}.csv")
    manifest.to_csv(output, index=False, encoding="utf-8")
    print(
        f"{len(manifest)} clips, {manifest['duration'].sum() / 3600:.2f} h, "
        f"{manifest['fleurs_sentence_id'].nunique()} sentences, sample rates {rates}, "
        f"speakers {manifest['speaker_id'].value_counts().to_dict()} -> {output}"
    )


if __name__ == "__main__":
    main()
