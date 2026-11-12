"""Orquestração independente do Streamlit; dependências podem ser substituídas em testes."""

from collections.abc import Callable
from threading import Lock
from typing import Protocol

import numpy as np

from clinical_audio.audio import windows
from clinical_audio.config import SER_MODEL_NAME, Settings
from clinical_audio.errors import processing_error_message
from clinical_audio.models import AnalysisResult, Event, Segment, TextAnalysis, Turn
from clinical_audio.rules import (
    SOUND_MAP,
    assign_speakers,
    keyword_alerts,
    merge_events,
    nearby_events,
    validate_text_alignment,
)


class AudioBackend(Protocol):
    def transcribe(self, samples: np.ndarray) -> list[Segment]: ...
    def classify_events(self, clip: np.ndarray) -> list[dict]: ...
    def voice_features(self, clip: np.ndarray) -> dict: ...
    def diarize(self, samples: np.ndarray) -> list[Turn]: ...


class SemanticBackend(Protocol):
    def analyze(self, segments: list[Segment], events: list[Event]) -> TextAnalysis: ...


class AnalysisPipeline:
    def __init__(self, settings: Settings, backend: AudioBackend):
        self.settings, self.backend = settings, backend
        self._lock = Lock()

    def analyze(
        self,
        samples: np.ndarray,
        *,
        diarize: bool = False,
        semantic: SemanticBackend | None = None,
        progress: Callable[[str], None] = lambda _: None,
    ) -> AnalysisResult:
        # Modelos cacheados são compartilhados; nenhum resultado é mantido no serviço.
        with self._lock:
            return self._analyze(samples, diarize=diarize, semantic=semantic, progress=progress)

    def _analyze(self, samples, *, diarize, semantic, progress):
        sr = self.settings.sample_rate
        if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
            raise ValueError("Áudio deve ser mono, finito e não vazio.")
        if len(samples) / sr > self.settings.max_duration_s:
            raise ValueError("Duração máxima excedida.")
        progress("Transcrevendo com Whisper")
        segments = self.backend.transcribe(samples)
        warnings = [
            "Protótipo acadêmico: resultados são hipóteses para revisão humana, sem diagnóstico.",
            "Modelo SER treinado em inglês, com quatro categorias; desempenho em português não validado.",
            "SER e prosódia usam no máximo os primeiros 20 segundos de cada segmento.",
        ]
        diarization_status = "desabilitada"
        if diarize:
            progress("Separando falantes")
            try:
                segments = assign_speakers(segments, self.backend.diarize(samples))
                diarization_status = "community-1"
            except Exception as exc:
                diarization_status = "indisponível"
                warning = processing_error_message(exc, "Separando falantes")
                warning += " A análise continua com falantes não identificados; nenhum filtro de paciente será aplicado."
                warnings.append(warning)
                progress(warning)
        else:
            warnings.append(
                "Diarização desabilitada: falantes não identificados; nenhum filtro de paciente aplicado."
            )
        progress("Detectando eventos acústicos")
        events = []
        for start, end, clip in windows(samples, sr, self.settings.window_s, self.settings.hop_s):
            rms = float(np.sqrt(np.mean(clip**2)))
            if rms < self.settings.rms_threshold:
                continue
            for prediction in self.backend.classify_events(clip):
                if (
                    prediction["label"] in SOUND_MAP
                    and prediction["score"] >= self.settings.event_threshold
                ):
                    events.append(
                        Event(
                            start=start,
                            end=end,
                            label=SOUND_MAP[prediction["label"]],
                            score=float(prediction["score"]),
                            rms=rms,
                        )
                    )
        events = merge_events(events)
        progress("Analisando voz e prosódia")
        enriched = []
        for segment in segments:
            clip = samples[int(segment.start * sr) : int(segment.end * sr)]
            features = self.backend.voice_features(clip)
            # Revalida valores oriundos dos adaptadores, inclusive scores e NaN.
            enriched.append(
                Segment(
                    **{
                        **segment.model_dump(),
                        **features,
                        "nearby_events": nearby_events(segment, events),
                    }
                )
            )
        text = None
        if semantic is not None and enriched:
            progress("Analisando texto pela OpenAI")
            try:
                text = validate_text_alignment(semantic.analyze(enriched, events), enriched)
            except Exception:
                # Preserva resultados locais sem expor transcrição/chaves em erros do provedor.
                warnings.append(
                    "Análise textual indisponível. Confira chave, modelo, quota e conexão; dados locais preservados."
                )
        elif semantic is None:
            warnings.append("Análise textual desabilitada; discordâncias não foram avaliadas.")
        if not enriched:
            warnings.append(
                "Nenhum segmento de fala reconhecido. Confira idioma e qualidade do áudio."
            )
        return AnalysisResult(
            mode="real",
            duration_s=len(samples) / sr,
            segments=enriched,
            events=events,
            text_analysis=text,
            alerts=keyword_alerts(enriched),
            warnings=warnings,
            metadata={
                "whisper": self.settings.whisper_model,
                "language": self.settings.language,
                "ast": "MIT/ast-finetuned-audioset-10-10-0.4593",
                "ser": SER_MODEL_NAME,
                "text_model": self.settings.openai_model
                if text is not None
                else "desabilitado/indisponível",
                "diarization": diarization_status,
            },
        )
