# Experiment Plan

| ID | Description | Variable changed | Status |
|---|---|---|---|
| M0 | Pretrained baseline, no fine-tuning | — | planned |
| M1 | Khmer fine-tuning | training on Khmer subset | planned |
| M2 | Customized strategy | one of: normalization / augmentation / FT strategy / HPO | planned |
| M3 | Second pretrained architecture (optional) | architecture | planned |
| S  | Data scaling (e.g. 10/50/100 h, if available) | training hours | planned |

## Fixed across experiments
- Locked test set: `data/manifests/test.csv`
- Text normalization protocol (see `docs/decisions.md`)
- Evaluation script and metric implementation

## Pilot
