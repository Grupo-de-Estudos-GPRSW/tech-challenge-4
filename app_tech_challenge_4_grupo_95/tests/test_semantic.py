import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from clinical_audio.config import Settings
from clinical_audio.models import ConsultationTextEmotions, Segment, SegmentTextEmotion
from clinical_audio.semantic import OpenAISemantic


def test_notebook_single_call_temperature_and_text_only():
    client = Mock()
    client.beta.chat.completions.parse.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    parsed=ConsultationTextEmotions(
                        segmentos_analisados=[
                            SegmentTextEmotion(segment_id=1, emocao_discurso_texto="Calma")
                        ]
                    )
                )
            )
        ]
    )
    segments = [
        Segment(id=i, start=i, end=i + 1, text="Estou bem", voice_emotion="Raiva", voice_score=0.9)
        for i in range(1, 18)
    ]
    result = OpenAISemantic(Settings(openai_key="fake"), client).analyze(segments)
    assert result.segments[0].emotion == "Calma"
    client.beta.chat.completions.parse.assert_called_once()
    kwargs = client.beta.chat.completions.parse.call_args.kwargs
    assert kwargs["temperature"] == 0.0
    assert kwargs["response_format"] is ConsultationTextEmotions
    payload = json.loads(kwargs["messages"][1]["content"].split("\n", 1)[1])
    assert len(payload) == 17
    assert payload[0] == {"segment_id": 1, "transcricao": "Estou bem"}


def test_refusal_and_errors_propagate():
    client = Mock()
    service = OpenAISemantic(Settings(openai_key="fake"), client)
    client.beta.chat.completions.parse.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(parsed=None))]
    )
    with pytest.raises(ValueError, match="estruturada"):
        service.analyze([])
    client.beta.chat.completions.parse.side_effect = TimeoutError("timeout")
    with pytest.raises(TimeoutError):
        service.analyze([])
