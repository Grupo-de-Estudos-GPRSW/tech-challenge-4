"""Contratos entre transcrição, áudio, texto e apresentação."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Interval(BaseModel):
    start: float = Field(ge=0, allow_inf_nan=False)
    end: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def ordered(self) -> "Interval":
        if self.end <= self.start:
            raise ValueError("O fim deve ser posterior ao início.")
        return self


class Turn(Interval):
    speaker: str


class Event(Interval):
    label: str
    score: float = Field(ge=0, le=1, allow_inf_nan=False)
    rms: float = Field(ge=0, allow_inf_nan=False)


class Segment(Interval):
    id: int = Field(gt=0)
    text: str
    speaker: str = "Não identificado"
    voice_emotion: str = "Não avaliada"
    voice_score: float = Field(default=0, ge=0, le=1, allow_inf_nan=False)
    secondary_emotion: str = "Não avaliada"
    pitch_mean: float = 0
    pitch_std: float = 0
    rms: float = 0
    nearby_events: list[str] = Field(default_factory=list)


class TextSegment(BaseModel):
    id: int
    emotion: str
    discordance: Literal["Não avaliada", "Nenhuma", "Leve", "Moderada", "Alta"]
    explanation: str


class TextAnalysis(BaseModel):
    summary: str
    segments: list[TextSegment]
    observations: list[str]


class Alert(Interval):
    segment_id: int
    term: str
    message: str


class AnalysisResult(BaseModel):
    mode: Literal["real", "demo"]
    duration_s: float
    segments: list[Segment]
    events: list[Event]
    text_analysis: TextAnalysis | None = None
    alerts: list[Alert] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)
