from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


def button(app, label):
    return next(b for b in app.button if b.label == label)


def test_app_demo_and_clear_without_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=30)
    assert not app.exception
    assert button(app, "Analisar arquivo").disabled
    button(app, "Explorar demonstração sem modelos ou credenciais").click().run(timeout=30)
    assert not app.exception
    assert any("DEMONSTRAÇÃO" in warning.value for warning in app.warning)
    assert len(app.metric) == 4
    assert len(app.tabs) == 4
    next(s for s in app.selectbox if s.label.startswith("Filtrar falante")).select(
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
            raise dependencies.MissingDependenciesError(["openai-whisper"], False)
        raise ImportError("Incompatible package", name="transformers")

    monkeypatch.setattr(dependencies, "ensure_dependencies", fail)
    monkeypatch.setattr(
        st,
        "file_uploader",
        lambda *args, **kwargs: SimpleNamespace(name="a.wav", getvalue=lambda: b"unused"),
    )
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=30)
    next(c for c in app.checkbox if "Confirmo autorização" in c.label).check().run()
    button(app, "Analisar arquivo").click().run(timeout=30)
    assert not app.exception
    assert any(("openai-whisper" if missing else "transformers") in e.value for e in app.error)
    assert any(sys.executable in code.value and ".[ml]" in code.value for code in app.code)
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
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=30)
    next(c for c in app.checkbox if "Confirmo autorização" in c.label).check().run()
    button(app, "Analisar arquivo").click().run(timeout=30)
    assert not app.exception
    assert len(app.metric) == 4
    next(s for s in app.selectbox if s.label == "Idioma da fala").select("en").run()
    assert len(app.metric) == 0
