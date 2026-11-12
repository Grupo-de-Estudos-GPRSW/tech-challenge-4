"""Configuração validada; segredos não entram nos relatórios."""

import os

from pydantic import BaseModel, Field, SecretStr

SER_MODEL_NAME = "superb/wav2vec2-base-superb-er"


class Settings(BaseModel):
    sample_rate: int = 16000
    max_upload_mb: int = Field(default=25, ge=1, le=25)
    max_duration_s: float = Field(default=600, gt=0, le=600)
    window_s: float = Field(default=1.2, gt=0)
    hop_s: float = Field(default=0.6, gt=0, le=1.2)
    event_threshold: float = Field(default=0.12, ge=0, le=1)
    rms_threshold: float = Field(default=0.015, ge=0)
    whisper_model: str = "base"
    language: str = "pt"
    openai_model: str = "gpt-4o-mini"
    openai_key: SecretStr = SecretStr("")
    hf_token: SecretStr = SecretStr("")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            whisper_model=os.getenv("WHISPER_MODEL", "base"),
            language=os.getenv("AUDIO_LANGUAGE", "pt"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            openai_key=SecretStr(os.getenv("OPENAI_API_KEY", "")),
            hf_token=SecretStr(os.getenv("HF_TOKEN", "")),
        )
