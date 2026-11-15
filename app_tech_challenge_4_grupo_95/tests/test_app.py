from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture(autouse=True)
def isolate_cached_models():
    import streamlit as st

    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def button(app, label):
    return next(b for b in app.button if b.label == label)


def test_app_demo_and_clear_without_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN", "fake")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=30)
    assert not app.exception
    assert button(app, "Analisar arquivo").disabled
    button(app, "Explorar demonstração sem modelos ou credenciais").click().run(timeout=30)
    assert not app.exception
    assert any("DEMONSTRAÇÃO" in warning.value for warning in app.warning)
    assert len(app.metric) == 4
    assert len(app.tabs) == 4
    next(s for s in app.selectbox if s.label.startswith("Paciente identificado")).select(
        "SPEAKER_01"
    ).run()
    assert not app.exception
    button(app, "Limpar resultados").click().run()
    assert not app.exception
    assert len(app.metric) == 0


@pytest.mark.parametrize("missing", [True, False])
def test_dependency_errors_show_actionable_install_command(monkeypatch, missing):
    import sys
    from types import SimpleNamespace

    import streamlit as st

    from clinical_audio import dependencies

    def fail(*args):
        if missing:
            raise dependencies.MissingDependenciesError(["openai-whisper"], True)
        raise ImportError("Incompatible package", name="transformers")

    monkeypatch.setattr(dependencies, "ensure_dependencies", fail)
    monkeypatch.setattr(
        st,
        "file_uploader",
        lambda *args, **kwargs: SimpleNamespace(name="a.wav", getvalue=lambda: b"unused"),
    )
    monkeypatch.setenv("HF_TOKEN", "fake")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=30)
    next(c for c in app.checkbox if "Confirmo autorização" in c.label).check().run()
    button(app, "Analisar arquivo").click().run(timeout=30)
    assert not app.exception
    assert any(("openai-whisper" if missing else "transformers") in e.value for e in app.error)
    assert any(
        sys.executable in code.value and ".[ml,diarization]" in code.value for code in app.code
    )
    assert not app.metric


def test_upload_ui_runs_pipeline_and_invalidates_stale_results(monkeypatch):
    from io import BytesIO
    from types import SimpleNamespace

    import numpy as np
    import soundfile as sf
    import streamlit as st

    from clinical_audio import backends, dependencies
    from clinical_audio.models import Segment

    class AudioDouble:
        def __init__(self, settings):
            pass

        def transcribe(self, samples):
            return [Segment(id=1, start=0, end=1, text="Olá")]

        def diarize(self, samples):
            from clinical_audio.models import Turn

            return [Turn(start=0, end=1, speaker="SPEAKER_01")]

        def classify_events(self, clip):
            return []

        def voice_features(self, clip):
            return {"voice_emotion": "Neutro", "voice_score": 0.7}

    buffer = BytesIO()
    sf.write(buffer, np.ones(16000, dtype=np.float32) * 0.1, 16000, format="WAV")

    def uploader(*args, **kwargs):
        assert kwargs["type"] == ["wav", "flac"]
        return SimpleNamespace(name="test.wav", getvalue=buffer.getvalue)

    monkeypatch.setattr(st, "file_uploader", uploader)
    monkeypatch.setattr(backends, "LocalModels", AudioDouble)
    monkeypatch.setattr(dependencies, "ensure_dependencies", lambda *args: None)
    monkeypatch.setenv("HF_TOKEN", "fake")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=30)
    next(c for c in app.checkbox if "Confirmo autorização" in c.label).check().run()
    button(app, "Analisar arquivo").click().run(timeout=30)
    assert not app.exception
    assert len(app.metric) == 4
    next(s for s in app.selectbox if s.label == "Idioma da fala").select("pt").run()
    assert len(app.metric) == 0


def test_v2_ui_openai_flow_uses_local_rules_and_filter_preserves_full_result(monkeypatch):
    from io import BytesIO
    from types import SimpleNamespace
    from unittest.mock import Mock

    import numpy as np
    import soundfile as sf
    import streamlit as st

    from clinical_audio import backends, dependencies, semantic
    from clinical_audio.models import Segment, TextEmotion, TextEmotions, Turn

    class AudioDouble:
        def __init__(self, settings):
            pass

        def transcribe(self, samples):
            return [
                Segment(id=1, start=0, end=0.5, text="Estou tranquilo."),
                Segment(id=2, start=0.5, end=1, text="Estou com medo."),
            ]

        def classify_events(self, clip):
            return []

        def voice_features(self, clip):
            return {"voice_emotion": "Tristeza", "voice_score": 0.8}

        def diarize(self, samples):
            return [Turn(start=0, end=0.5, speaker="A"), Turn(start=0.5, end=1, speaker="B")]

    fake_semantic = Mock()
    fake_semantic.analyze.return_value = TextEmotions(
        segments=[TextEmotion(id=2, emotion="Medo"), TextEmotion(id=1, emotion="Calma")]
    )
    factory = Mock(return_value=fake_semantic)
    buffer = BytesIO()
    sf.write(buffer, np.ones(16000, dtype=np.float32) * 0.1, 16000, format="WAV")
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    monkeypatch.setenv("HF_TOKEN", "fake")
    monkeypatch.setattr(
        st,
        "file_uploader",
        lambda *args, **kwargs: SimpleNamespace(name="v2.wav", getvalue=buffer.getvalue),
    )
    monkeypatch.setattr(backends, "LocalModels", AudioDouble)
    monkeypatch.setattr(dependencies, "ensure_dependencies", lambda *args: None)
    monkeypatch.setattr(semantic, "OpenAISemantic", factory)
    monkeypatch.setenv("HF_TOKEN", "fake")
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=30)
    for label in ["Extrair emoção do texto com OpenAI"]:
        next(c for c in app.checkbox if c.label == label).check()
    next(c for c in app.checkbox if "Confirmo autorização" in c.label).check()
    app.run()
    button(app, "Analisar arquivo").click().run(timeout=30)
    assert not app.exception
    assert not app.error
    result = app.session_state["result"]
    assert [s.discordance for s in result.text_analysis.segments] == ["Alta"]
    assert len(app.dataframe[0].value) == 1
    original_json = result.model_dump_json()
    next(s for s in app.selectbox if s.label.startswith("Paciente identificado")).select("A").run()
    assert not app.exception
    assert len(app.dataframe[0].value) == 1
    assert app.session_state["result"].model_dump_json() == original_json
    fake_semantic.analyze.assert_called_once()
    assert len(fake_semantic.analyze.call_args.args) == 1
