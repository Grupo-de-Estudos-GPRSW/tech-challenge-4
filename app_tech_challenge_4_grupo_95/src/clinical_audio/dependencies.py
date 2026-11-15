"""Diagnóstico das dependências opcionais sem carregar modelos ou credenciais."""

import importlib.util
import os
import sys

ML_MODULES = {
    "whisper": "openai-whisper",
    "torch": "torch",
    "torchaudio": "torchaudio",
    "transformers": "transformers",
    "librosa": "librosa",
}


class MissingDependenciesError(ImportError):
    def __init__(self, packages: list[str], diarize: bool):
        self.packages = packages
        self.diarize = diarize
        super().__init__("Pacotes necessários não instalados: " + ", ".join(packages))


def install_command(diarize: bool = False) -> str:
    extra = "ml,diarization" if diarize else "ml"
    prefix = "& " if os.name == "nt" else ""
    return f'{prefix}"{sys.executable}" -m pip install -e ".[{extra}]"'


def ensure_dependencies(diarize: bool = False) -> None:
    modules = dict(ML_MODULES)
    if diarize:
        modules["pyannote.audio"] = "pyannote.audio"
    missing = []
    for module, package in modules.items():
        try:
            found = importlib.util.find_spec(module) is not None
        except (ModuleNotFoundError, ValueError):
            found = False
        if not found:
            missing.append(package)
    if missing:
        raise MissingDependenciesError(missing, diarize)
