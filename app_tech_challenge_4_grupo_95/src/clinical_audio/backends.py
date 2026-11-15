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
        from transformers import pipeline

        return pipeline("audio-classification", model=self.ser_name, device=self.device)

    def transcribe(self, samples: np.ndarray) -> list[Segment]:
        output = self.whisper.transcribe(
            samples,
            language=None if self.settings.language == "auto" else self.settings.language,
            verbose=False,
            fp16=self.device == "cuda",
        )
        segments = []
        for index, raw in enumerate(output.get("segments", [])):
            start, end = float(raw["start"]), float(raw["end"])
            if raw.get("text", "").strip():
                segments.append(
                    Segment(
                        id=index + 1,
                        start=round(start, 2),
                        end=round(end, 2),
                        text=raw["text"].strip(),
                    )
                )
        return segments

    def classify_events(self, clip: np.ndarray) -> list[dict]:
        return self.ast({"raw": clip, "sampling_rate": self.settings.sample_rate}, top_k=8)

    def voice_features(self, clip: np.ndarray) -> dict:
        import librosa

        sr = self.settings.sample_rate
        if len(clip) < int(0.2 * sr):
            clip = np.pad(clip, (0, int(0.2 * sr) - len(clip)))
        predictions = self.ser({"raw": clip, "sampling_rate": sr}, top_k=3)
        primary = predictions[0]
        secondary = predictions[1] if len(predictions) > 1 else {"label": "neutral", "score": 0.0}
        rms = float(np.sqrt(np.mean(clip**2)))
        f0, _, _ = librosa.pyin(
            clip, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"), sr=sr
        )
        valid = f0[~np.isnan(f0)] if f0 is not None else np.array([])
        return {
            "voice_emotion": EMOTION_MAP.get(primary["label"], primary["label"]),
            "voice_score": round(float(primary["score"]), 3),
            "secondary_emotion": EMOTION_MAP.get(secondary["label"], secondary["label"]),
            "secondary_score": round(float(secondary["score"]), 3),
            "pitch_mean": round(float(np.mean(valid)), 1) if len(valid) else 0,
            "pitch_std": round(float(np.std(valid)), 1) if len(valid) else 0,
            "rms": round(rms, 4),
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
            Turn(start=round(turn.start, 2), end=round(turn.end, 2), speaker=speaker)
            for turn, speaker in output.speaker_diarization
        ]
