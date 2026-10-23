import numpy as np

from khmer_asr.data.audio import TARGET_SR, clip_path, to_target_rate


def test_resamples_48k_to_16k_keeping_duration_and_pitch():
    sr = 48000
    t = np.arange(sr * 2) / sr  # 2 s, 440 Hz tone
    out = to_target_rate(np.sin(2 * np.pi * 440 * t), sr)
    assert len(out) == 2 * TARGET_SR
    peak_hz = np.argmax(np.abs(np.fft.rfft(out))) * TARGET_SR / len(out)
    assert abs(peak_hz - 440) < 1


def test_16k_mono_is_unchanged():
    x = np.random.default_rng(0).standard_normal(TARGET_SR)
    assert np.array_equal(to_target_rate(x, TARGET_SR), x)


def test_downmixes_stereo():
    x = np.ones((100, 2))
    assert to_target_rate(x, TARGET_SR).shape == (100,)


def test_clip_path_has_no_colon():
    assert ":" not in clip_path("audio", "train-00000-of-00653:12").name
