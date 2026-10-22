"""Clean, deduplicate and split the dataset; write manifests to data/manifests/."""
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
