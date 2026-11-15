"""Orquestração independente do Streamlit; dependências podem ser substituídas em testes."""

from collections.abc import Callable
from threading import Lock
from typing import Protocol

import numpy as np

from clinical_audio.config import SER_MODEL_NAME, Settings
from clinical_audio.models import AnalysisResult, Event, Segment, TextEmotions, Turn
from clinical_audio.rules import (
    SOUND_MAP,
    calculate_discordance_locally,
    keyword_alerts,
    nearby_events,
)


class AudioBackend(Protocol):
    def transcribe(self, samples: np.ndarray) -> list[Segment]: ...
    def classify_events(self, clip: np.ndarray) -> list[dict]: ...
    def voice_features(self, clip: np.ndarray) -> dict: ...
    def diarize(self, samples: np.ndarray) -> list[Turn]: ...


class SemanticBackend(Protocol):
    def analyze(self, segments: list[Segment]) -> TextEmotions: ...


class AnalysisPipeline:
    def __init__(self, settings: Settings, backend: AudioBackend):
        self.settings, self.backend = settings, backend
        self._lock = Lock()

    def analyze(
        self,
        samples: np.ndarray,
        *,
        diarize: bool = True,
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
        if not diarize:
            raise ValueError("O fluxo requer diarização e seleção automática do paciente.")
        warnings = ["Papéis e discordâncias são heurísticas, não diagnóstico."]
        diarization_status = "community-1"
        progress("Detectando eventos acústicos")
        events = []
        size, hop = int(self.settings.window_s * sr), int(self.settings.hop_s * sr)
        for offset in range(0, max(1, len(samples) - size + 1), hop):
            clip = samples[offset : min(len(samples), offset + size)]
            start, end = round(offset / sr, 2), round(min(len(samples), offset + size) / sr, 2)
            rms = float(np.sqrt(np.mean(clip**2)))
            if rms < self.settings.rms_threshold:
                continue
            predictions = self.backend.classify_events(clip)
            speech_score = sum(
                p["score"]
                for p in predictions
                if p["label"] in ("Speech", "Conversation", "Narration, monologue")
            )
            for prediction in predictions:
                if (
                    prediction["label"] in SOUND_MAP
                    and prediction["score"] >= self.settings.event_threshold
                ):
                    events.append(
                        Event(
                            start=start,
                            end=end,
                            label=SOUND_MAP[prediction["label"]],
                            score=round(float(prediction["score"]), 3),
                            rms=round(rms, 4),
                            label_en=prediction["label"],
                            speech_score=round(speech_score, 3),
                        )
                    )
                    break
        merged = []
        for event in events:
            if merged and event.label == merged[-1].label and event.start <= merged[-1].end + 0.35:
                previous = merged[-1]
                merged[-1] = previous.model_copy(
                    update={
                        "end": max(previous.end, event.end),
                        "score": max(previous.score, event.score),
                        "rms": max(previous.rms, event.rms),
                    }
                )
            else:
                merged.append(event)
        events = merged
        progress("Transcrevendo com Whisper")
        transcription = self.backend.transcribe(samples)
        progress("Separando falantes e selecionando paciente")
        turns = self.backend.diarize(samples)
        from clinical_audio.notebook_v2 import (
            filter_patient_segments,
            get_patient_speaker_id_acoustically,
        )

        raw_turns = [{"inicio_s": t.start, "fim_s": t.end, "speaker_raw": t.speaker} for t in turns]
        raw_transcription = [
            {"segment_id": s.id, "inicio_s": s.start, "fim_s": s.end, "transcricao": s.text}
            for s in transcription
        ]
        patient = get_patient_speaker_id_acoustically(raw_turns, raw_transcription)
        selected_ids = {
            r["segment_id"] for r in filter_patient_segments(raw_turns, raw_transcription)
        }
        segments = [
            s.model_copy(update={"speaker": patient}) for s in transcription if s.id in selected_ids
        ]
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
        progress("Extraindo emoções textuais e calculando discordâncias")
        if semantic is None:
            emotions = TextEmotions(segments=[])
            warnings.append("Sem cliente OpenAI: emoção textual Neutro por fallback.")
        else:
            emotions = semantic.analyze(enriched)
        text = calculate_discordance_locally(enriched, emotions)
        if not enriched:
            warnings.append("Nenhum segmento selecionado como paciente pela heurística automática.")
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
                "text_model": (
                    semantic.model
                    if isinstance(getattr(semantic, "model", None), str)
                    else self.settings.openai_model
                )
                if semantic is not None
                else "fallback Neutro",
                "diarization": diarization_status,
                "patient_speaker": patient,
                "reference": "fluxo de análise clínica",
                "discordance": "regras locais",
                "text_payload": "somente ID e transcrição",
            },
        )
