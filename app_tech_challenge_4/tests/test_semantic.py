from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from clinical_audio.config import Settings
from clinical_audio.models import Segment, TextAnalysis, TextSegment
from clinical_audio.semantic import OpenAISemantic


def test_key_required():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAISemantic(Settings())


def test_structured_output_uses_bounded_batches_and_does_not_store():
    client = Mock()

    def parse(**kwargs):
        import json

        payload = json.loads(kwargs["input"][1]["content"])
        assert kwargs["store"] is False
        assert len(payload["segments"]) <= 15
        return SimpleNamespace(
            output_parsed=TextAnalysis(
                summary="Resumo",
                observations=["Revisar"],
                segments=[
                    TextSegment(
                        id=s["id"], emotion="Neutro", discordance="Nenhuma", explanation="Exemplo"
                    )
                    for s in payload["segments"]
                ],
            )
        )

    client.responses.parse.side_effect = parse
    service = OpenAISemantic(Settings(openai_key="fake"), client)
    segments = [Segment(id=i, start=i, end=i + 1, text="Teste") for i in range(1, 18)]
    result = service.analyze(segments, [])
    assert len(result.segments) == 17
    assert result.observations == ["Revisar"]
    assert client.responses.parse.call_count == 2


def test_refusal_raises_without_fake_analysis():
    client = Mock()
    client.responses.parse.return_value = SimpleNamespace(output_parsed=None)
    service = OpenAISemantic(Settings(openai_key="fake"), client)
    with pytest.raises(ValueError, match="estruturada"):
        service.analyze([Segment(id=1, start=0, end=1, text="Olá")], [])
