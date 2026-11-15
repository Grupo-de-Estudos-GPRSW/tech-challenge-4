import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from clinical_audio.backends import LocalModels
from clinical_audio.config import Settings


def test_whisper_preserves_original_ids_and_rounded_timestamps():
    models = LocalModels(Settings(language="auto"))
    models.device = "cpu"
    models.whisper = Mock()
    models.whisper.transcribe.return_value = {
        "segments": [
            {"start": 0, "end": 2, "text": " olá "},
            {"start": 0, "end": 0.5, "text": " "},
            {"start": 2, "end": 3, "text": "fora"},
        ]
    }
    result = models.transcribe(np.zeros(16000, dtype=np.float32))
    assert len(result) == 2 and result[0].end == 2
    assert result[1].id == 3
    assert result[0].text == "olá"
    assert models.whisper.transcribe.call_args.kwargs["language"] is None


def test_ast_requests_notebook_top_eight():
    models = LocalModels(Settings())
    models.ast = Mock(return_value=[{"label": "Cough", "score": 0.5}])
    assert models.classify_events(np.ones(100))[0]["label"] == "Cough"
    assert models.ast.call_args.kwargs == {"top_k": 8}


@pytest.mark.parametrize("length", [20, 16000, 400000])
def test_voice_full_segment_padding_rounding_and_pitch(monkeypatch, length):
    pyin = Mock(return_value=(np.array([np.nan, 100, 200]), None, None))
    monkeypatch.setitem(
        sys.modules,
        "librosa",
        SimpleNamespace(pyin=pyin, note_to_hz=lambda x: {"C2": 65.406, "C6": 1046.502}[x]),
    )
    models = LocalModels(Settings())
    models.ser = Mock(
        return_value=[{"label": "sad", "score": 0.6004}, {"label": "neutral", "score": 0.2998}]
    )
    output = models.voice_features(np.ones(length, dtype=np.float32))
    assert len(pyin.call_args.args[0]) == max(length, 3200)
    assert models.ser.call_args.kwargs == {"top_k": 3}
    assert output["voice_emotion"] == "Tristeza / Abatimento"
    assert output["voice_score"] == 0.6
    assert output["secondary_score"] == 0.3
    assert output["pitch_mean"] == 150 and output["pitch_std"] == 50
    assert pyin.call_args.kwargs["fmax"] == 1046.502
    pyin.return_value = (None, None, None)
    models.ser.return_value = [{"label": "custom", "score": 0.7}]
    fallback = models.voice_features(np.zeros(16000))
    assert fallback["pitch_mean"] == 0 and fallback["secondary_emotion"] == "Neutro"
    assert fallback["secondary_score"] == 0


def test_model_factories_are_lazy_and_cached(monkeypatch):
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False), device=lambda value: value
    )
    whisper = SimpleNamespace(load_model=Mock(return_value="whisper"))
    transformers = SimpleNamespace(
        pipeline=Mock(return_value="classifier"),
        AutoModelForAudioClassification=SimpleNamespace(
            from_pretrained=Mock(
                return_value=("model", {"missing_keys": [], "mismatched_keys": []})
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "whisper", whisper)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    models = LocalModels(Settings())
    assert models.device == "cpu"
    assert models.whisper == models.whisper == "whisper"
    assert whisper.load_model.call_count == 1
    assert models.ast == models.ser == "classifier"
    assert transformers.pipeline.call_count == 2
    assert (
        transformers.pipeline.call_args_list[0].kwargs["model"]
        == "MIT/ast-finetuned-audioset-10-10-0.4593"
    )
    assert (
        transformers.pipeline.call_args_list[1].kwargs["model"]
        == "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    )
    assert models.settings.language == "en" and models.settings.whisper_model == "base"


def test_pyannote_memory_input_and_community_output(monkeypatch):
    tensor = Mock()
    torch = SimpleNamespace(from_numpy=Mock(return_value=tensor), device=lambda value: value)
    factory = Mock()
    pipeline = Mock()
    factory.from_pretrained.return_value.to.return_value = pipeline
    pipeline.return_value = SimpleNamespace(
        speaker_diarization=[(SimpleNamespace(start=0, end=1), "A")]
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "pyannote", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "pyannote.audio", SimpleNamespace(Pipeline=factory))
    models = LocalModels(Settings(hf_token="fake"))
    models.device = "cpu"
    result = models.diarize(np.ones(16000, dtype=np.float32))
    assert result[0].speaker == "A"
    assert pipeline.call_args.args[0]["sample_rate"] == 16000
    assert factory.from_pretrained.call_args.kwargs["token"] == "fake"
    with pytest.raises(ValueError, match="HF_TOKEN"):
        assert LocalModels(Settings()).diarizer is not None
