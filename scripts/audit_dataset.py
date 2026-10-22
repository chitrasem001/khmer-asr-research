"""Audit the Khmer speech dataset (schema, durations, duplicates) via streaming."""
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="Digital-Divide-Data/khmer-speech-dataset")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--output", default="data/manifests/audit_report.json")
    parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
