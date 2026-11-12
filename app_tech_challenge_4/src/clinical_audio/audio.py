"""Leitura em memória de WAV/FLAC e reamostragem, sem executáveis externos."""

from io import BytesIO
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from clinical_audio.config import Settings

ALLOWED_EXTENSIONS = {".wav", ".flac"}


def validate_upload(name: str, data: bytes, settings: Settings) -> str:
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("Formato inválido. Use apenas WAV ou FLAC.")
    if not data:
        raise ValueError("O arquivo está vazio.")
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise ValueError(f"O arquivo excede {settings.max_upload_mb} MB.")
    return suffix


def decode_upload(name: str, data: bytes, settings: Settings) -> np.ndarray:
    suffix = validate_upload(name, data, settings)
    try:
        with sf.SoundFile(BytesIO(data)) as source:
            expected = {".wav": {"WAV", "WAVEX", "RF64"}, ".flac": {"FLAC"}}
            if source.format not in expected[suffix]:
                raise ValueError(
                    "O conteúdo do arquivo não corresponde ao formato WAV/FLAC informado."
                )
            if source.frames <= 0 or source.samplerate <= 0:
                raise ValueError("O arquivo não contém áudio válido.")
            if source.frames / source.samplerate > settings.max_duration_s:
                raise ValueError(f"O áudio excede {settings.max_duration_s:g} segundos.")
            if source.channels > 8 or source.samplerate > 192000:
                raise ValueError("Use áudio com até 8 canais e taxa de amostragem de até 192 kHz.")
            # Leitura em blocos limita a alocação para arquivos multicanal.
            blocks = []
            for block in source.blocks(blocksize=65536, dtype="float32", always_2d=True):
                if not np.isfinite(block).all():
                    raise ValueError("O arquivo não contém áudio válido.")
                blocks.append(block.mean(axis=1))
            original_rate = source.samplerate
    except (sf.LibsndfileError, sf.SoundFileError, RuntimeError) as exc:
        raise ValueError("Não foi possível ler o áudio. Envie um WAV ou FLAC válido.") from exc
    samples = np.concatenate(blocks) if blocks else np.array([], dtype=np.float32)
    if not len(samples) or not np.isfinite(samples).all():
        raise ValueError("O arquivo não contém áudio válido.")
    if original_rate != settings.sample_rate:
        divisor = gcd(original_rate, settings.sample_rate)
        samples = resample_poly(samples, settings.sample_rate // divisor, original_rate // divisor)
    return np.ascontiguousarray(samples, dtype=np.float32)


def windows(samples: np.ndarray, sr: int, window_s: float, hop_s: float):
    """Inclui o final do áudio e evita uma cauda redundante já coberta."""
    size, hop = max(1, int(window_s * sr)), max(1, int(hop_s * sr))
    if hop > size:
        raise ValueError("O passo não pode ser maior que a janela.")
    for start in range(0, len(samples), hop):
        end = min(start + size, len(samples))
        yield start / sr, end / sr, samples[start:end]
        if end == len(samples):
            break
