"""Transcribe a single Khmer audio file."""
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", default="openai/whisper-small")
    parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
