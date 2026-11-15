"""Oráculo: executa somente funções puras de comparação, sem rede ou pesos."""

import ast
import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel, Field

from clinical_audio.config import Settings
from clinical_audio.models import Segment, TextEmotion, TextEmotions, Turn
from clinical_audio.notebook_v2 import filter_patient_segments, get_patient_speaker_id_acoustically
from clinical_audio.pipeline import AnalysisPipeline
from clinical_audio.reports import markdown_report, notebook_csv_bytes, notebook_segment_rows
from clinical_audio.rules import SOUND_MAP, calculate_discordance_locally


@pytest.fixture
def reference():
    reference_path = Path(__file__).parents[1] / "scripts" / "fixtures" / "analysis_reference.json"
    cells = json.loads(reference_path.read_text(encoding="utf-8"))["cells"]
    names = {
        "calculate_discordance_locally",
        "filter_patient_segments",
        "get_patient_speaker_id_acoustically",
        "detect_clinical_non_verbal_sounds",
        "generate_clinical_discordance_report",
        "extract_prosodic_features",
        "analyze_voice_emotions",
        "run_text_and_discordance_analysis",
        "transcribe_clinical_audio",
        "analyze_text_emotion_with_openai",
    }
    nodes = []
    constants, maps = {}, {}
    classes = []
    for cell in cells:
        source = "".join(cell["source"])
        if cell["cell_type"] != "code" or source.startswith("!"):
            continue
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in {
                        "OPENAI_MODEL",
                        "WHISPER_MODEL_SIZE",
                        "SAMPLE_RATE",
                        "SER_MODEL_NAME",
                    }:
                        constants[target.id] = ast.literal_eval(node.value)
            if (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id in {"CLINICAL_SOUND_MAP", "EMOTION_TRANSLATION_MAP"}
            ):
                maps[node.target.id] = ast.literal_eval(node.value)
            if isinstance(node, ast.ClassDef) and node.name in {
                "SegmentTextEmotion",
                "ConsultationTextEmotions",
            }:
                classes.append(node)
        nodes.extend(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
        )
    namespace = {
        "List": list,
        "Dict": dict,
        "Any": Any,
        "Path": Path,
        "np": np,
        "pd": pd,
        **constants,
        **maps,
        "notebook_constants": constants,
        "notebook_maps": maps,
        "BaseModel": BaseModel,
        "Field": Field,
        "json": json,
    }
    exec(
        compile(ast.Module(body=classes + nodes, type_ignores=[]), str(reference_path), "exec"),
        namespace,
    )
    return namespace


@pytest.mark.parametrize("text", ["Calma", "Neutro", "Medo", "Surpresa"])
@pytest.mark.parametrize("voice", ["Raiva / Irritação", "Neutro", "Indeterminada"])
@pytest.mark.parametrize("score", [0, 0.6, 0.601])
@pytest.mark.parametrize("events", [[], ["Tosse"]])
def test_discordance_and_explanation_equal_reference(reference, text, voice, score, events):
    expected = reference["calculate_discordance_locally"](
        [
            {
                "emocao_discurso_texto": text,
                "emocao_voz_primaria": voice,
                "confianca_voz_primaria": score,
                "eventos_nao_verbais_proximos": events,
            }
        ]
    )[0]
    actual = calculate_discordance_locally(
        [
            Segment(
                id=1,
                start=0,
                end=1,
                text="Teste",
                voice_emotion=voice,
                voice_score=score,
                nearby_events=events,
            )
        ],
        TextEmotions(segments=[TextEmotion(id=1, emotion=text)]),
    ).segments[0]
    assert actual.discordance == expected["nivel_discordancia"]
    assert actual.explanation == expected["explicacao_clinica"]


@pytest.mark.parametrize("texts", [["bom dia", "estou bem"], ["hello", "fine"], ["doctor", "yes"]])
def test_patient_heuristic_matches_notebook(reference, texts):
    turns = [
        {"inicio_s": 0, "fim_s": 2, "speaker_raw": "A"},
        {"inicio_s": 2, "fim_s": 3, "speaker_raw": "B"},
    ]
    segments = [
        {"segment_id": i + 1, "inicio_s": i * 2, "fim_s": i * 2 + 1, "transcricao": t}
        for i, t in enumerate(texts)
    ]
    assert get_patient_speaker_id_acoustically(turns, segments) == reference[
        "get_patient_speaker_id_acoustically"
    ](turns, segments)
    assert filter_patient_segments(turns, segments) == reference["filter_patient_segments"](
        turns, segments
    )


def test_detector_merging_and_tail_equal_reference(reference):
    samples = np.arange(37, dtype=np.float32) / 100 + 0.1

    def classify(clip):
        index = int(round((float(clip[0]) - 0.1) * 100))
        label = "Sneeze" if index == 6 else "Cough"
        return [
            {"label": "Speech", "score": 0.7},
            {"label": label, "score": 0.45678},
            {"label": "Sigh", "score": 0.4},
        ]

    reference["librosa"] = type(
        "Librosa", (), {"load": staticmethod(lambda *a, **k: (samples, 10))}
    )
    reference["ast_classifier"] = lambda data, top_k: classify(data["raw"])
    reference["CLINICAL_SOUND_MAP"] = SOUND_MAP
    expected = reference["detect_clinical_non_verbal_sounds"](Path("fake.wav"))
    backend = Mock()
    backend.classify_events.side_effect = classify
    backend.transcribe.return_value = [Segment(id=1, start=0, end=1, text="fine")]
    backend.diarize.return_value = [Turn(start=0, end=1, speaker="A")]
    backend.voice_features.return_value = {"voice_emotion": "Neutro", "voice_score": 0.7}
    actual = AnalysisPipeline(Settings(sample_rate=10), backend).analyze(samples)
    assert [(e.start, e.end, e.label, e.score, e.rms, e.speech_score) for e in actual.events] == [
        (r.inicio_s, r.fim_s, r.evento_clinico, r.confianca, r.energia_rms, r.prob_fala_simultanea)
        for r in expected.itertuples()
    ]


def test_report_identical_to_notebook(reference, tmp_path, monkeypatch):
    from clinical_audio.demo import demo_result

    result = demo_result()
    full = pd.DataFrame(notebook_segment_rows(result))
    events = pd.DataFrame(
        [
            {"inicio_s": e.start, "fim_s": e.end, "evento_clinico": e.label, "confianca": e.score}
            for e in result.events
        ]
    )
    monkeypatch.chdir(tmp_path)
    expected = reference["generate_clinical_discordance_report"](
        full, events, tmp_path / "reference.md"
    )
    assert markdown_report(result) == expected
    exported = pd.read_csv(tmp_path / "analise_segmentos_clinicos.csv")
    assert list(exported.columns) == list(full.columns)
    assert notebook_csv_bytes(result) == (tmp_path / "analise_segmentos_clinicos.csv").read_bytes()


@pytest.mark.parametrize("length", [20, 16000, 400000])
def test_prosody_matches_notebook_after_padding(reference, monkeypatch, length):
    import sys
    from types import SimpleNamespace

    from clinical_audio.backends import LocalModels

    fake = SimpleNamespace(
        pyin=lambda clip, **kwargs: (np.array([100.0, np.nan, 200.0]), None, None),
        note_to_hz=lambda note: {"C2": 65.406391, "C6": 1046.502261}[note],
    )
    reference["librosa"] = fake
    monkeypatch.setitem(sys.modules, "librosa", fake)
    clip = np.ones(length, dtype=np.float32) * 0.123456
    models = LocalModels(Settings())
    models.ser = Mock(return_value=[{"label": "neutral", "score": 0.6004}])
    actual = models.voice_features(clip)
    padded = np.pad(clip, (0, max(0, 3200 - length)))
    expected = reference["extract_prosodic_features"](padded)
    assert (actual["pitch_mean"], actual["pitch_std"], actual["rms"]) == (
        expected["pitch_medio_hz"],
        expected["pitch_std_hz"],
        expected["energia_rms"],
    )


def test_voice_enrichment_with_multiple_events_matches_notebook(reference, monkeypatch):
    import sys
    from types import SimpleNamespace

    from clinical_audio.backends import LocalModels
    from clinical_audio.models import Event
    from clinical_audio.rules import EMOTION_MAP, nearby_events

    samples = np.ones(16000, dtype=np.float32) * 0.123456
    fake = SimpleNamespace(
        load=lambda *args, **kwargs: (samples, 16000),
        pyin=lambda *args, **kwargs: (np.array([100.0, np.nan, 200.0]), None, None),
        note_to_hz=lambda note: {"C2": 65.406391, "C6": 1046.502261}[note],
    )
    monkeypatch.setitem(sys.modules, "librosa", fake)
    predictions = [{"label": "neutral", "score": 0.7004}, {"label": "sad", "score": 0.2004}]
    reference["librosa"] = fake
    reference["ser_pipeline"] = lambda *args, **kwargs: predictions
    reference["EMOTION_TRANSLATION_MAP"] = EMOTION_MAP
    rows = [{"segment_id": 1, "inicio_s": 0.0, "fim_s": 1.0, "transcricao": "Estou bem."}]
    events = [
        Event(start=0, end=0.1, label="Tosse", score=0.5, rms=0.1),
        Event(start=0.2, end=0.3, label="Espirro", score=0.5, rms=0.1),
        Event(start=0.4, end=0.5, label="Tosse", score=0.5, rms=0.1),
    ]
    frame = pd.DataFrame(
        [{"inicio_s": e.start, "fim_s": e.end, "evento_clinico": e.label} for e in events]
    )
    expected = reference["analyze_voice_emotions"](Path("fake.wav"), rows, frame)[0]
    backend = LocalModels(Settings())
    backend.ser = Mock(return_value=predictions)
    segment = Segment(id=1, start=0, end=1, text="Estou bem.", **backend.voice_features(samples))
    segment.nearby_events = nearby_events(segment, events)
    result = calculate_discordance_locally(
        [segment], TextEmotions(segments=[TextEmotion(id=1, emotion="Neutro")])
    )
    from clinical_audio.models import AnalysisResult
    from clinical_audio.reports import notebook_segment_rows

    actual = notebook_segment_rows(
        AnalysisResult(
            mode="real", duration_s=1, segments=[segment], events=events, text_analysis=result
        )
    )[0]
    for key, value in expected.items():
        assert actual[key] == value
    expected_discordance = reference["calculate_discordance_locally"](
        [{**expected, "emocao_discurso_texto": "Neutro"}]
    )[0]
    assert actual["explicacao_clinica"] == expected_discordance["explicacao_clinica"]
    assert actual["eventos_nao_verbais_proximos"] == ["Tosse", "Espirro"]


def test_complete_patient_table_matches_reference_pipeline(reference, monkeypatch):
    import sys
    from types import SimpleNamespace

    from clinical_audio.backends import LocalModels
    from clinical_audio.models import ConsultationTextEmotions, SegmentTextEmotion
    from clinical_audio.reports import notebook_event_rows

    samples = np.ones(64000, dtype=np.float32) * 0.1
    fake = SimpleNamespace(
        load=lambda *args, **kwargs: (samples, 16000),
        pyin=lambda *args, **kwargs: (np.array([100.0, 200.0]), None, None),
        note_to_hz=lambda note: {"C2": 65.406391, "C6": 1046.502261}[note],
    )
    monkeypatch.setitem(sys.modules, "librosa", fake)
    reference["librosa"] = fake
    raw = {
        "segments": [
            {"start": 0.0, "end": 0.8, "text": "Bom dia"},
            {"start": 0.8, "end": 1.6, "text": "Estou bem"},
            {"start": 1.6, "end": 3.0, "text": "Sinto medo"},
        ]
    }
    whisper = Mock()
    whisper.transcribe.return_value = raw
    predictions = [{"label": "neutral", "score": 0.7004}, {"label": "sad", "score": 0.2004}]

    def classifier(*args, **kwargs):
        return [{"label": "Cough", "score": 0.45}, {"label": "Speech", "score": 0.4}]

    reference["whisper_model"] = whisper
    reference["torch"] = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    reference["ast_classifier"] = classifier
    reference["ser_pipeline"] = lambda *args, **kwargs: predictions
    from clinical_audio.rules import EMOTION_MAP

    reference["EMOTION_TRANSLATION_MAP"] = EMOTION_MAP
    reference["CLINICAL_SOUND_MAP"] = SOUND_MAP
    emotions = ConsultationTextEmotions(
        segmentos_analisados=[
            SegmentTextEmotion(segment_id=3, emocao_discurso_texto="Medo"),
            SegmentTextEmotion(segment_id=2, emocao_discurso_texto="Calma"),
        ]
    )
    reference["analyze_text_emotion_with_openai"] = lambda segments: emotions
    turns = [
        {"inicio_s": 0.0, "fim_s": 0.8, "speaker_raw": "A"},
        {"inicio_s": 0.8, "fim_s": 3.0, "speaker_raw": "B"},
    ]
    transcription = reference["transcribe_clinical_audio"](Path("fake.wav"))
    events = reference["detect_clinical_non_verbal_sounds"](Path("fake.wav"))
    selected = reference["filter_patient_segments"](turns, transcription)
    voice = reference["analyze_voice_emotions"](Path("fake.wav"), selected, events)
    expected = reference["run_text_and_discordance_analysis"](voice)
    backend = LocalModels(Settings())
    backend.device = "cpu"
    backend.whisper = whisper
    backend.ast = classifier
    backend.ser = Mock(return_value=predictions)
    backend.diarize = lambda samples: [
        Turn(start=t["inicio_s"], end=t["fim_s"], speaker=t["speaker_raw"]) for t in turns
    ]
    semantic = Mock()
    semantic.analyze.return_value = TextEmotions(
        segments=[
            TextEmotion(id=e.segment_id, emotion=e.emocao_discurso_texto)
            for e in emotions.segmentos_analisados
        ]
    )
    actual = AnalysisPipeline(Settings(), backend).analyze(samples, semantic=semantic)
    pd.testing.assert_frame_equal(pd.DataFrame(notebook_segment_rows(actual)), expected)
    pd.testing.assert_frame_equal(
        pd.DataFrame(notebook_event_rows(actual))[list(events.columns)], events
    )


def test_notebook_model_defaults_and_maps_are_independent_oracle(reference):
    from clinical_audio.config import SER_MODEL_NAME
    from clinical_audio.rules import EMOTION_MAP

    settings = Settings()
    assert reference["notebook_constants"] == {
        "OPENAI_MODEL": settings.openai_model,
        "WHISPER_MODEL_SIZE": settings.whisper_model,
        "SAMPLE_RATE": settings.sample_rate,
        "SER_MODEL_NAME": SER_MODEL_NAME,
    }
    assert reference["notebook_maps"]["CLINICAL_SOUND_MAP"] == SOUND_MAP
    assert reference["notebook_maps"]["EMOTION_TRANSLATION_MAP"] == EMOTION_MAP


@pytest.mark.parametrize("count", [0, 1, 17])
def test_api_request_prompt_schema_and_single_call_equal_notebook(reference, count):
    from types import SimpleNamespace

    from clinical_audio.models import ConsultationTextEmotions
    from clinical_audio.semantic import OpenAISemantic

    parsed = ConsultationTextEmotions(segmentos_analisados=[])
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))])
    client = Mock()
    client.beta.chat.completions.parse.return_value = response
    # Cada caso recebe as funções e o namespace originais, sem mocks de outro teste.
    previous = reference.get("client")
    reference["client"] = client
    segments = [
        Segment(
            id=i + 1, start=i, end=i + 1, text="Estou bem.", voice_emotion="Raiva", voice_score=0.9
        )
        for i in range(count)
    ]
    try:
        reference["analyze_text_emotion_with_openai"](
            [{"segment_id": s.id, "transcricao": s.text} for s in segments]
        )
    finally:
        reference["client"] = previous
    expected = client.beta.chat.completions.parse.call_args.kwargs
    client.reset_mock()
    OpenAISemantic(Settings(openai_key="fake"), client).analyze(segments)
    client.beta.chat.completions.parse.assert_called_once()
    actual = client.beta.chat.completions.parse.call_args.kwargs
    for key in ["model", "messages", "temperature"]:
        assert actual[key] == expected[key]
    assert (
        actual["response_format"].model_json_schema()
        == expected["response_format"].model_json_schema()
    )
