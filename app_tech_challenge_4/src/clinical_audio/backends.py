"""Adaptadores de modelos locais. Imports pesados ocorrem apenas no modo real."""

from functools import cached_property

import numpy as np

from clinical_audio.config import SER_MODEL_NAME, Settings
from clinical_audio.models import Segment, Turn
from clinical_audio.rules import EMOTION_MAP


class LocalModels:
    ast_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
    ser_name = SER_MODEL_NAME

    def __init__(self, settings: Settings):
        self.settings = settings

    @cached_property
    def device(self):
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    @cached_property
    def whisper(self):
        import whisper

        return whisper.load_model(self.settings.whisper_model, device=self.device)

    @cached_property
    def ast(self):
        from transformers import pipeline

        return pipeline("audio-classification", model=self.ast_name, device=self.device)

    @cached_property
    def ser(self):
        from transformers import AutoModelForAudioClassification, pipeline

        model, loading = AutoModelForAudioClassification.from_pretrained(
            self.ser_name,
            output_loading_info=True,
        )
        missing_head = any(
            key.startswith(("classifier.", "projector.", "layer_weights"))
            for key in loading.get("missing_keys", [])
        )
        if missing_head or loading.get("mismatched_keys"):
            raise ValueError(
                "O checkpoint SER não carregou a classificação treinada. A análise de emoções foi interrompida para evitar previsões com pesos novos."
            )
        return pipeline(
            "audio-classification",
            model=model,
            feature_extractor=self.ser_name,
            device=self.device,
        )

    def transcribe(self, samples: np.ndarray) -> list[Segment]:
        output = self.whisper.transcribe(
            samples,
            language=None if self.settings.language == "auto" else self.settings.language,
            verbose=False,
            fp16=self.device == "cuda",
        )
        duration = len(samples) / self.settings.sample_rate
        segments = []
        for raw in output.get("segments", []):
            start, end = max(0, float(raw["start"])), min(duration, float(raw["end"]))
            if raw.get("text", "").strip() and end > start:
                segments.append(
                    Segment(id=len(segments) + 1, start=start, end=end, text=raw["text"].strip())
                )
        return segments

    def classify_events(self, clip: np.ndarray) -> list[dict]:
        # AudioSet é multirrótulo: sigmoid, sem descartar classes clínicas pelo top-k.
        return self.ast(
            {"raw": clip, "sampling_rate": self.settings.sample_rate},
            top_k=None,
            function_to_apply="sigmoid",
        )

    def voice_features(self, clip: np.ndarray) -> dict:
        import librosa

        sr = self.settings.sample_rate
        rms = float(np.sqrt(np.mean(clip**2))) if len(clip) else 0.0
        if len(clip) < int(0.2 * sr) or rms < 0.001:
            return {"voice_emotion": "Indeterminada", "voice_score": 0, "rms": rms}
        # Limita o custo do SER e do pYIN por segmento; texto/intervalo são preservados.
        clip = clip[: sr * 20]
        predictions = self.ser({"raw": clip, "sampling_rate": sr}, top_k=2)
        primary = predictions[0]
        secondary = predictions[1] if len(predictions) > 1 else {"label": "Não avaliada"}
        f0, _, _ = librosa.pyin(clip, fmin=65, fmax=1000, sr=sr)
        valid = f0[np.isfinite(f0)]
        return {
            "voice_emotion": EMOTION_MAP.get(primary["label"], primary["label"]),
            "voice_score": float(primary["score"]),
            "secondary_emotion": EMOTION_MAP.get(secondary["label"], secondary["label"]),
            "pitch_mean": round(float(np.mean(valid)), 1) if len(valid) else 0,
            "pitch_std": round(float(np.std(valid)), 1) if len(valid) else 0,
            "rms": rms,
        }

    @cached_property
    def diarizer(self):
        import torch
        from pyannote.audio import Pipeline

        if not self.settings.hf_token.get_secret_value():
            raise ValueError("Configure HF_TOKEN e aceite os termos do modelo para diarizar.")
        model = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=self.settings.hf_token.get_secret_value(),
        )
        return model.to(torch.device(self.device))

    def diarize(self, samples: np.ndarray) -> list[Turn]:
        import torch

        output = self.diarizer(
            {
                "waveform": torch.from_numpy(samples.copy()).unsqueeze(0),
                "sample_rate": self.settings.sample_rate,
            }
        )
        return [
            Turn(start=turn.start, end=turn.end, speaker=speaker)
            for turn, speaker in output.speaker_diarization
        ]
