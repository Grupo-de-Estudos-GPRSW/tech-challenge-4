from unittest.mock import Mock

import numpy as np
import pytest

from clinical_audio.config import Settings
from clinical_audio.models import Segment, TextEmotion, TextEmotions, Turn
from clinical_audio.pipeline import AnalysisPipeline


class FakeAudio:
    def transcribe(self, samples):
        return [
            Segment(id=1, start=0, end=0.5, text="Bom dia, como tem se sentido?"),
            Segment(id=2, start=0.5, end=1, text="Estou tranquilo."),
        ]

    def diarize(self, samples):
        return [Turn(start=0, end=0.5, speaker="A"), Turn(start=0.5, end=1, speaker="B")]

    def classify_events(self, clip):
        return [
            {"label": "Speech", "score": 0.8},
            {"label": "Cough", "score": 0.6},
            {"label": "Sneeze", "score": 0.5},
        ]

    def voice_features(self, clip):
        return {"voice_emotion": "Medo / Ansiedade", "voice_score": 0.8, "rms": 0.1}


def test_notebook_order_patient_filter_and_neutral_fallback():
    backend = FakeAudio()
    backend.voice_features = Mock(wraps=backend.voice_features)
    stages = []
    result = AnalysisPipeline(Settings(), backend).analyze(np.ones(32000), progress=stages.append)
    assert [s.id for s in result.segments] == [2]
    assert result.segments[0].speaker == "B"
    backend.voice_features.assert_called_once()
    assert result.text_analysis.segments[0].emotion == "Neutro"
    assert result.text_analysis.segments[0].discordance == "Alta"
    assert stages[:3] == [
        "Detectando eventos acústicos",
        "Transcrevendo com Whisper",
        "Separando falantes e selecionando paciente",
    ]
    assert len(result.events) == 1 and result.events[0].end == 1.8
    assert result.events[0].speech_score == 0.8
    assert result.events[0].label_en == "Cough"
    assert result.metadata["patient_speaker"] == "B"


def test_missing_text_id_defaults_to_neutral_first_duplicate_wins():
    semantic = Mock()
    semantic.analyze.return_value = TextEmotions(segments=[TextEmotion(id=99, emotion="Raiva")])
    result = AnalysisPipeline(Settings(), FakeAudio()).analyze(np.ones(16000), semantic=semantic)
    assert [s.id for s in semantic.analyze.call_args.args[0]] == [2]
    assert result.text_analysis.segments[0].emotion == "Neutro"


@pytest.mark.parametrize("stage", ["diarize", "voice_features"])
def test_model_failure_stops_pipeline(stage):
    backend = FakeAudio()
    setattr(backend, stage, Mock(side_effect=RuntimeError("unavailable")))
    with pytest.raises(RuntimeError):
        AnalysisPipeline(Settings(), backend).analyze(np.ones(16000))


def test_api_failure_stops_pipeline():
    semantic = Mock()
    semantic.analyze.side_effect = TimeoutError("failed")
    with pytest.raises(TimeoutError):
        AnalysisPipeline(Settings(), FakeAudio()).analyze(np.ones(16000), semantic=semantic)


def test_empty_patient_result_and_disabled_diarization():
    backend = FakeAudio()
    backend.diarize = lambda samples: []
    result = AnalysisPipeline(Settings(), backend).analyze(np.zeros(16000))
    assert not result.segments and not result.events
    with pytest.raises(ValueError, match="requer diarização"):
        AnalysisPipeline(Settings(), backend).analyze(np.ones(16000), diarize=False)


@pytest.mark.parametrize(
    "samples", [np.array([]), np.array([np.nan]), np.ones((2, 3)), np.ones(16001)]
)
def test_invalid_audio(samples):
    with pytest.raises(ValueError):
        AnalysisPipeline(Settings(max_duration_s=1), FakeAudio()).analyze(samples)


def test_metadata_records_actual_semantic_model():
    semantic = Mock(model="gpt-4o")
    semantic.analyze.return_value = TextEmotions(segments=[TextEmotion(id=2, emotion="Calma")])
    result = AnalysisPipeline(Settings(), FakeAudio()).analyze(np.ones(16000), semantic=semantic)
    assert result.metadata["text_model"] == "gpt-4o"


@pytest.mark.parametrize(
    "items,expected",
    [
        ([(2, "Medo"), (2, "Calma")], "Medo"),
        ([(99, "Raiva"), (2, "Calma")], "Calma"),
        ([], "Neutro"),
    ],
)
def test_text_join_preserves_notebook_duplicate_and_missing_id_policy(items, expected):
    semantic = Mock()
    semantic.analyze.return_value = TextEmotions(
        segments=[TextEmotion(id=i, emotion=e) for i, e in items]
    )
    result = AnalysisPipeline(Settings(), FakeAudio()).analyze(np.ones(16000), semantic=semantic)
    assert result.text_analysis.segments[0].emotion == expected
