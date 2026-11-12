import pytest
from pydantic import ValidationError

from clinical_audio.models import Event, Segment, TextAnalysis, TextSegment, Turn
from clinical_audio.rules import (
    assign_speakers,
    keyword_alerts,
    merge_events,
    validate_text_alignment,
)


def segment(**kwargs):
    return Segment(**{"id": 1, "start": 0, "end": 4, "text": "Olá", **kwargs})


def test_speaker_uses_total_overlap_instead_of_largest_single_turn():
    turns = [
        Turn(start=0, end=1.5, speaker="A"),
        Turn(start=2, end=3.5, speaker="A"),
        Turn(start=0, end=2, speaker="B"),
    ]
    assert assign_speakers([segment()], turns)[0].speaker == "A"


def test_speaker_ties_and_no_overlap_are_explicit():
    turns = [Turn(start=0, end=2, speaker="A"), Turn(start=2, end=4, speaker="B")]
    assert assign_speakers([segment()], turns)[0].speaker == "Ambíguo"
    assert assign_speakers([segment(start=5, end=6)], turns)[0].speaker == "Não identificado"


def test_merge_does_not_lose_interleaved_labels_or_mutate_input():
    events = [
        Event(start=0, end=1, label="Tosse", score=0.3, rms=0.1),
        Event(start=0.5, end=1.2, label="Pigarro", score=0.7, rms=0.2),
        Event(start=1.1, end=2, label="Tosse", score=0.8, rms=0.3),
    ]
    merged = merge_events(events)
    assert len(merged) == 2
    assert merged[0].end == 2
    assert merged[0].score == 0.8
    assert events[0].end == 1


def test_critical_terms_preserve_negated_mentions_for_human_review():
    alerts = keyword_alerts([segment(text="Não tenho FALTA DE AR nem dor no peito.")])
    assert {a.term for a in alerts} == {"falta de ar", "dor no peito"}
    assert "negação" in alerts[0].message
    assert not keyword_alerts([segment(text="desmaiosidade")])


@pytest.mark.parametrize("start,end", [(2, 1), (0, 0), (-1, 2), (0, float("inf"))])
def test_invalid_intervals_rejected(start, end):
    with pytest.raises(ValidationError):
        segment(start=start, end=end)


@pytest.mark.parametrize("ids", [[2], [1, 1], []])
def test_semantic_ids_must_match_exactly(ids):
    analysis = TextAnalysis(
        summary="",
        observations=[],
        segments=[
            TextSegment(id=i, emotion="Neutra", discordance="Nenhuma", explanation="") for i in ids
        ],
    )
    with pytest.raises(ValueError, match="IDs"):
        validate_text_alignment(analysis, [segment()])
