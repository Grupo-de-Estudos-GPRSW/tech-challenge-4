from io import BytesIO
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from clinical_audio.audio import decode_upload, validate_upload, windows
from clinical_audio.config import Settings


def encoded(samples, rate=16000, format="WAV", subtype=None):
    buffer = BytesIO()
    sf.write(buffer, samples, rate, format=format, subtype=subtype)
    return buffer.getvalue()


@pytest.mark.parametrize(
    "name,data",
    [
        ("a.exe", b"x"),
        ("a.mp3", b"x"),
        ("a.m4a", b"x"),
        ("a.ogg", b"x"),
        ("a.wav", b""),
        ("a.wav", b"x" * 1048577),
    ],
)
def test_invalid_uploads(name, data):
    with pytest.raises(ValueError):
        validate_upload(name, data, Settings(max_upload_mb=1))


def test_sliding_windows_cover_tail_and_short_audio():
    clips = list(windows(np.ones(23), 10, 1.2, 0.6))
    assert [(a, b) for a, b, _ in clips] == [(0, 1.2), (0.6, 1.8), (1.2, 2.3)]
    assert len(list(windows(np.ones(2), 10, 1.2, 0.6))) == 1
    with pytest.raises(ValueError):
        list(windows(np.ones(20), 10, 0.1, 0.5))


@pytest.mark.parametrize("format,suffix", [("WAV", ".WAV"), ("FLAC", ".flac")])
@pytest.mark.parametrize("rate", [8000, 16000, 44100, 48000])
def test_decode_real_stereo_formats_without_external_process(format, suffix, rate):
    time = np.arange(rate // 2) / rate
    tone = 0.5 * np.sin(2 * np.pi * 440 * time)
    stereo = np.column_stack([tone, tone * 0.5])
    payload = encoded(stereo, rate, format)
    with patch("subprocess.run", side_effect=AssertionError("Não pode chamar executável externo")):
        samples = decode_upload("../../consulta" + suffix, payload, Settings())
    assert samples.dtype == np.float32
    assert samples.flags.c_contiguous
    assert samples.shape == (8000,)
    expected = 0.375 * np.sin(2 * np.pi * 440 * np.arange(8000) / 16000)
    np.testing.assert_allclose(samples[100:-100], expected[100:-100], atol=0.003)


def test_downsampling_filters_frequencies_above_nyquist():
    rate = 48000
    tone = 0.8 * np.sin(2 * np.pi * 12000 * np.arange(rate) / rate)
    samples = decode_upload("a.wav", encoded(tone, rate), Settings())
    assert np.sqrt(np.mean(samples[100:-100] ** 2)) < 0.01


@pytest.mark.parametrize("suffix,format", [(".flac", "WAV"), (".wav", "FLAC"), (".wav", "OGG")])
def test_disguised_formats_rejected(suffix, format):
    with pytest.raises(ValueError, match="conteúdo"):
        decode_upload("a" + suffix, encoded(np.ones(1600), format=format), Settings())


def test_corrupt_file_rejected():
    with pytest.raises(ValueError, match="WAV ou FLAC válido"):
        decode_upload("a.wav", b"not audio", Settings())


@pytest.mark.parametrize(
    "samples,rate,subtype",
    [
        (np.array([]), 16000, None),
        (np.array([np.nan]), 16000, "FLOAT"),
        (np.zeros(16001), 16000, None),
    ],
)
def test_empty_nonfinite_and_overlong_audio_rejected(samples, rate, subtype):
    with pytest.raises(ValueError):
        decode_upload("a.wav", encoded(samples, rate, subtype=subtype), Settings(max_duration_s=1))


@pytest.mark.parametrize("samples,rate", [(np.zeros((100, 9)), 16000), (np.zeros(100), 384000)])
def test_resource_limits_rejected(samples, rate):
    with pytest.raises(ValueError, match="8 canais"):
        decode_upload("a.wav", encoded(samples, rate), Settings())


def test_exact_duration_limit_is_accepted():
    samples = decode_upload("a.wav", encoded(np.zeros(16000)), Settings(max_duration_s=1))
    assert len(samples) == 16000
