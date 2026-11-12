import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from clinical_audio.backends import LocalModels
from clinical_audio.config import Settings


def test_whisper_filters_empty_and_invalid_segments_and_clips_end():
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
    assert len(result) == 1 and result[0].end == 1
    assert result[0].text == "olá"
    assert models.whisper.transcribe.call_args.kwargs["language"] is None


def test_ast_explicitly_requests_multilabel_sigmoid_and_all_labels():
    models = LocalModels(Settings())
    models.ast = Mock(return_value=[{"label": "Cough", "score": 0.5}])
    assert models.classify_events(np.ones(500))[0]["label"] == "Cough"
    assert models.ast.call_args.kwargs == {"top_k": None, "function_to_apply": "sigmoid"}


def test_voice_short_and_silent_clips_are_indeterminate(monkeypatch):
    monkeypatch.setitem(sys.modules, "librosa", SimpleNamespace())
    models = LocalModels(Settings())
    assert models.voice_features(np.zeros(16000))["voice_emotion"] == "Indeterminada"
    assert models.voice_features(np.ones(20))["voice_score"] == 0


def test_voice_features_handle_unvoiced_pitch(monkeypatch):
    pyin = Mock(return_value=(np.array([np.nan, 100, 200]), None, None))
    monkeypatch.setitem(sys.modules, "librosa", SimpleNamespace(pyin=pyin))
    models = LocalModels(Settings())
    models.ser = Mock(
        return_value=[{"label": "sad", "score": 0.6}, {"label": "neutral", "score": 0.3}]
    )
    output = models.voice_features(np.ones(400000, dtype=np.float32))
    assert output["pitch_mean"] == 150
    assert output["pitch_std"] == 50
    assert output["voice_emotion"] == "Tristeza"
    assert len(pyin.call_args.args[0]) == 320000
    pyin.return_value = (np.array([np.nan]), None, None)
    models.ser.return_value = [{"label": "custom", "score": 0.7}]
    assert models.voice_features(np.ones(16000))["pitch_mean"] == 0


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


@pytest.mark.parametrize(
    "loading",
    [
        {"missing_keys": ["classifier.weight"]},
        {"missing_keys": ["projector.bias"]},
        {"mismatched_keys": ["classifier.weight"]},
    ],
)
def test_ser_rejects_untrained_classification_weights(monkeypatch, loading):
    transformers = SimpleNamespace(
        pipeline=Mock(),
        AutoModelForAudioClassification=SimpleNamespace(
            from_pretrained=Mock(return_value=("model", loading))
        ),
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    with pytest.raises(ValueError, match="classificação treinada"):
        assert LocalModels(Settings()).ser is not None
    transformers.pipeline.assert_not_called()


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
