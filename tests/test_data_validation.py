from khmer_asr.data.validation import find_split_overlap, is_valid_example


def test_rejects_empty_transcript():
    assert not is_valid_example("   ", 3.0)


def test_rejects_out_of_range_duration():
    assert not is_valid_example("សួស្តី", 0.1)
    assert not is_valid_example("សួស្តី", 45.0)


def test_accepts_valid_example():
    assert is_valid_example("សួស្តី", 3.0)


def test_detects_split_overlap():
    assert find_split_overlap(["a", "b"], ["c"], ["b", "d"]) == {"b"}
    assert find_split_overlap(["a"], ["b"], ["c"]) == set()
