"""Cenário fixo para avaliar a interface sem rede; nunca analisa um upload."""

import io
import wave

import numpy as np

from clinical_audio.models import AnalysisResult, Event, Segment, TextAnalysis, TextSegment
from clinical_audio.rules import keyword_alerts


def demo_result() -> AnalysisResult:
    segments = [
        Segment(
            id=1,
            start=0,
            end=3,
            text="Como você está se sentindo?",
            speaker="SPEAKER_00",
            voice_emotion="Neutro",
            voice_score=0.8,
            rms=0.12,
        ),
        Segment(
            id=2,
            start=3.5,
            end=7,
            text="Estou tranquilo, mas tive falta de ar à noite.",
            speaker="SPEAKER_01",
            voice_emotion="Medo",
            voice_score=0.65,
            nearby_events=["Tosse"],
            rms=0.18,
            pitch_mean=180,
            pitch_std=24,
        ),
    ]
    return AnalysisResult(
        mode="demo",
        duration_s=8,
        segments=segments,
        events=[Event(start=7, end=7.6, label="Tosse", score=0.7, rms=0.2)],
        text_analysis=TextAnalysis(
            summary="Cenário fictício: consulta com menção a falta de ar e possível diferença entre voz e texto.",
            segments=[
                TextSegment(
                    id=1, emotion="Interesse", discordance="Nenhuma", explanation="Exemplo fixo."
                ),
                TextSegment(
                    id=2,
                    emotion="Tranquilidade declarada",
                    discordance="Leve",
                    explanation="Hipótese fictícia de diferença entre texto e voz; requer revisão.",
                ),
            ],
            observations=["Dados inventados exclusivamente para demonstrar a interface."],
        ),
        alerts=keyword_alerts(segments),
        warnings=[
            "DEMONSTRAÇÃO: transcrição, eventos e emoções são fixtures, sem inferência de IA.",
            "O som reproduzido é um tom sintético; não contém as falas ou a tosse exibidas.",
        ],
        metadata={"source": "fixture acadêmica fixa"},
    )


def demo_audio() -> bytes:
    sr = 16000
    samples = (0.12 * np.sin(2 * np.pi * 220 * np.arange(sr * 8) / sr)).astype(np.float32)
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(sr)
            output.writeframes((samples * 32767).astype("<i2").tobytes())
        return buffer.getvalue()
