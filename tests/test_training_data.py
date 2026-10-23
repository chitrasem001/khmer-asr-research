import pandas as pd

from khmer_asr.data.subsets import select_train_hours
from khmer_asr.training.data import build_ctc_vocab


def _train() -> pd.DataFrame:
    # 10 files x 4 clips x 900 s = 1 h per file.
    return pd.DataFrame(
        {
            "source_file": [f"f{i}.parquet" for i in range(10) for _ in range(4)],
            "duration": [900.0] * 40,
        }
    )


def test_select_train_hours_takes_whole_files_up_to_target():
    sub = select_train_hours(_train(), 3, seed=0)
    assert sub["source_file"].nunique() == 3
    assert sub.groupby("source_file").size().eq(4).all()


def test_subsets_are_nested():
    small = set(select_train_hours(_train(), 2, seed=1)["source_file"])
    large = set(select_train_hours(_train(), 5, seed=1)["source_file"])
    assert small < large


def test_none_returns_everything():
    assert len(select_train_hours(_train(), None, seed=0)) == 40


def test_ctc_vocab_maps_space_to_delimiter():
    vocab = build_ctc_vocab(pd.Series(["ក ខ", "៤"]))
    assert " " not in vocab
    assert {"ក", "ខ", "៤", "|", "[UNK]", "[PAD]"} == set(vocab)
    assert vocab["[PAD]"] == len(vocab) - 1
