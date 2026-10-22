"""Extract the audio of a manifest's clips to local FLAC files (resumable)."""
import argparse
import time

import pandas as pd

from khmer_asr.data.audio import clip_path, extract_clips
from khmer_asr.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--manifest", required=True, help="e.g. data/manifests/test.csv")
    parser.add_argument("--audio-dir", default="data/processed/audio")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    data_cfg = load_config(args.config)["data"]
    manifest = pd.read_csv(args.manifest)
    start = time.time()
    written = extract_clips(
        manifest, data_cfg["dataset"], data_cfg["revision"], args.audio_dir, args.workers
    )
    missing = [u for u in manifest["utt_id"] if not clip_path(args.audio_dir, u).exists()]
    print(f"Wrote {written} clips in {time.time() - start:.0f} s; missing: {len(missing)}")
    if missing:
        raise SystemExit(f"{len(missing)} clips missing, e.g. {missing[:3]}")


if __name__ == "__main__":
    main()
