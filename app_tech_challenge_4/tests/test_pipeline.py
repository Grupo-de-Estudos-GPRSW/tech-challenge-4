import numpy as np
import pytest

from clinical_audio.config import Settings
from clinical_audio.models import Segment, TextAnalysis, TextSegment, Turn
from clinical_audio.pipeline import AnalysisPipeline


class FakeAudio:
    def transcribe(self, samples):
        return [Segment(id=1, start=0, end=1, text="Tenho falta de ar.")]

    def classify_events(self, clip):
        return [
            {"label": "Speech", "score": 0.9},
            {"label": "Cough", "score": 0.6},
            {"label": "Sneeze", "score": 0.01},
        ]

    def voice_features(self, clip):
        return {"voice_emotion": "Medo", "voice_score": 0.8, "rms": 0.1}

    def diarize(self, samples):
        return [Turn(start=0, end=1, speaker="SPEAKER_01")]


class FakeSemantic:
    def analyze(self, segments, events):
        return TextAnalysis(
            summary="Exemplo",
            observations=[],
            segments=[
                TextSegment(
                    id=1, emotion="Preocupação", discordance="Nenhuma", explanation="Convergente"
                )
            ],
        )


def test_pipeline_fuses_real_contracts_and_preserves_local_results():
    pipeline = AnalysisPipeline(Settings(), FakeAudio())
    result = pipeline.analyze(
        np.ones(32000, dtype=np.float32), diarize=True, semantic=FakeSemantic()
    )
    assert result.mode == "real"
    assert len(result.events) == 1
    assert result.events[0].end == 2
    assert result.segments[0].speaker == "SPEAKER_01"
    assert result.segments[0].nearby_events == ["Tosse"]
    assert result.text_analysis.segments[0].discordance == "Nenhuma"
    assert result.alerts[0].term == "falta de ar"


def test_semantic_failure_never_erases_local_data_or_leaks_exception():
    class BrokenSemantic:
        def analyze(self, segments, events):
            raise RuntimeError("sk-secret patient-private-data")

    result = AnalysisPipeline(Settings(), FakeAudio()).analyze(
        np.ones(16000), semantic=BrokenSemantic()
    )
    assert result.segments
    assert result.text_analysis is None
    assert "sk-secret" not in result.model_dump_json()
    assert any("preservados" in warning for warning in result.warnings)


def test_no_speech_no_api_call_and_silence_skips_ast():
    class Silent(FakeAudio):
        def transcribe(self, samples):
            return []

        def classify_events(self, clip):
            pytest.fail("Silêncio não deve chamar AST")

    result = AnalysisPipeline(Settings(), Silent()).analyze(np.zeros(16000))
    assert not result.events and not result.segments
    assert any("Nenhum segmento" in warning for warning in result.warnings)


def test_diarization_access_failure_preserves_analysis_and_unknown_speakers():
    class Gated(FakeAudio):
        def diarize(self, samples):
            raise OSError("403 gated repo private-token")

    result = AnalysisPipeline(Settings(), Gated()).analyze(np.ones(16000), diarize=True)
    assert result.segments[0].speaker == "Não identificado"
    assert result.events
    assert result.metadata["diarization"] == "indisponível"
    assert any("Acesso ao modelo negado" in warning for warning in result.warnings)
    assert "private-token" not in result.model_dump_json()


@pytest.mark.parametrize(
    "samples", [np.array([]), np.array([np.nan]), np.ones((2, 3)), np.ones(16001)]
)
def test_pipeline_rejects_invalid_audio(samples):
    with pytest.raises(ValueError):
        AnalysisPipeline(Settings(max_duration_s=1), FakeAudio()).analyze(samples)
