"""Regras temporais testáveis, sem modelos nem acesso à rede."""

import re
import unicodedata
from collections import defaultdict

from clinical_audio.models import (
    Alert,
    Event,
    Segment,
    TextAnalysis,
    TextEmotions,
    TextSegment,
    Turn,
)

SOUND_MAP = {
    "Cough": "Tosse",
    "Throat clearing": "Limpeza de Garganta (Pigarro)",
    "Sneeze": "Espirro",
    "Sniff": "Fungada / Coriza",
    "Sigh": "Suspiro",
    "Gasp": "Arquejo / Falta de Ar",
    "Pant": "Respiração Ofegante",
    "Wheeze": "Sibilo / Chiado Respiratório",
    "Groan": "Gemido de Dor/Desconforto",
    "Grunt": "Grunhido de Esforço/Dor",
    "Crying, sobbing": "Choro / Soluço",
    "Whimper": "Queixume / Choro Contido",
    "Snort": "Ronco / Obstrução Nasal",
}
EMOTION_MAP = {
    "angry": "Raiva / Irritação",
    "calm": "Calma",
    "disgust": "Desconforto / Aversão",
    "fearful": "Medo / Ansiedade",
    "happy": "Alegria / Entusiasmo",
    "neutral": "Neutro",
    "sad": "Tristeza / Abatimento",
    "surprised": "Surpresa / Espanto",
    "ang": "Raiva / Irritação",
    "hap": "Alegria / Entusiasmo",
    "neu": "Neutro",
}


def merge_events(events: list[Event], gap: float = 0.35) -> list[Event]:
    groups: dict[str, list[Event]] = defaultdict(list)
    for event in sorted(events, key=lambda e: e.start):
        group = groups[event.label]
        if group and event.start <= group[-1].end + gap:
            previous = group[-1]
            group[-1] = previous.model_copy(
                update={
                    "end": max(previous.end, event.end),
                    "score": max(previous.score, event.score),
                    "rms": max(previous.rms, event.rms),
                }
            )
        else:
            group.append(event.model_copy())
    return sorted((e for group in groups.values() for e in group), key=lambda e: e.start)


def assign_speakers(segments: list[Segment], turns: list[Turn]) -> list[Segment]:
    result = []
    for segment in segments:
        overlaps: dict[str, float] = defaultdict(float)
        for turn in turns:
            overlaps[turn.speaker] += max(
                0, min(segment.end, turn.end) - max(segment.start, turn.start)
            )
        ranked = sorted(overlaps.items(), key=lambda item: (-item[1], item[0]))
        speaker = "Não identificado"
        if ranked and ranked[0][1] > 0:
            if len(ranked) > 1 and abs(ranked[0][1] - ranked[1][1]) < 0.001:
                speaker = "Ambíguo"
            else:
                speaker = ranked[0][0]
        result.append(segment.model_copy(update={"speaker": speaker}))
    return result


def nearby_events(segment: Segment, events: list[Event]) -> list[str]:
    # Preserva a primeira ocorrência, sem ordenar rótulos.
    return list(
        dict.fromkeys(
            e.label for e in events if e.end >= segment.start - 2.5 and e.start <= segment.end + 2.5
        )
    )


def keyword_alerts(segments: list[Segment]) -> list[Alert]:
    """Ocorrências textuais para revisão; não interpreta negação ou urgência."""
    terms = [
        "falta de ar",
        "dor no peito",
        "desmaio",
        "dificuldade para respirar",
        "shortness of breath",
        "chest pain",
    ]
    alerts = []
    for segment in segments:
        text = "".join(
            c
            for c in unicodedata.normalize("NFD", segment.text.lower())
            if not unicodedata.combining(c)
        )
        for term in terms:
            if re.search(r"\b" + re.escape(term) + r"\b", text):
                alerts.append(
                    Alert(
                        start=segment.start,
                        end=segment.end,
                        segment_id=segment.id,
                        term=term,
                        message="Termo mencionado; revisar contexto, negação e transcrição.",
                    )
                )
    return alerts


def validate_text_alignment(
    analysis: TextAnalysis | TextEmotions, segments: list[Segment]
) -> TextAnalysis | TextEmotions:
    ids = [s.id for s in analysis.segments]
    if len(ids) != len(set(ids)) or set(ids) != {s.id for s in segments}:
        raise ValueError("A análise textual retornou IDs ausentes, duplicados ou desconhecidos.")
    return analysis


POSITIVE_EMOTIONS = (
    "alegria",
    "calma",
    "neutro",
    "tranquilidade",
    "otimismo",
    "bem-estar",
    "positivo",
)
NEGATIVE_EMOTIONS = (
    "tristeza",
    "raiva",
    "medo",
    "ansiedade",
    "desconforto",
    "aversão",
    "abatimento",
    "irritação",
    "negativo",
)


def calculate_discordance_locally(segments: list[Segment], emotions: TextEmotions) -> TextAnalysis:
    from clinical_audio.notebook_v2 import calculate_discordance_locally as reference

    # Usa a primeira emoção com o ID; ausentes recebem Neutro.
    by_id = {}
    for item in emotions.segments:
        by_id.setdefault(item.id, item.emotion)
    rows = reference(
        [
            {
                "segment_id": s.id,
                "emocao_discurso_texto": by_id.get(s.id, "Neutro"),
                "emocao_voz_primaria": s.voice_emotion,
                "confianca_voz_primaria": s.voice_score,
                "eventos_nao_verbais_proximos": s.nearby_events,
            }
            for s in segments
        ]
    )
    items = [
        TextSegment(
            id=r["segment_id"],
            emotion=r["emocao_discurso_texto"],
            discordance=r["nivel_discordancia"],
            explanation=r["explicacao_clinica"],
        )
        for r in rows
    ]
    count = sum(r["ha_discordancia"] for r in rows)
    return TextAnalysis(
        summary=f"Total de segmentos: {len(rows)}. Discordâncias: {count} ({count / max(1, len(rows)) * 100:.1f}%).",
        segments=items,
        observations=["Comparação calculada localmente; resultados para revisão humana."],
    )
