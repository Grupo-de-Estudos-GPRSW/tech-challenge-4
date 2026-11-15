import sys
from unittest.mock import patch

import pytest

from clinical_audio.dependencies import (
    MissingDependenciesError,
    ensure_dependencies,
    install_command,
)


def test_missing_packages_list_distribution_names_and_skip_optional_diarization():
    with patch("clinical_audio.dependencies.importlib.util.find_spec", return_value=None):
        with pytest.raises(MissingDependenciesError) as raised:
            ensure_dependencies()
    assert "openai-whisper" in raised.value.packages
    assert "pyannote.audio" not in raised.value.packages
    assert not raised.value.diarize


def test_missing_parent_namespace_is_reported_for_diarization():
    def find_spec(module):
        if module == "pyannote.audio":
            raise ModuleNotFoundError("pyannote")
        return object()

    with patch("clinical_audio.dependencies.importlib.util.find_spec", side_effect=find_spec):
        with pytest.raises(MissingDependenciesError) as raised:
            ensure_dependencies(True)
    assert raised.value.packages == ["pyannote.audio"]


def test_available_modules_do_not_load_models():
    with patch("clinical_audio.dependencies.importlib.util.find_spec", return_value=object()):
        ensure_dependencies(True)


def test_install_command_uses_running_interpreter_and_selected_extra():
    assert sys.executable in install_command()
    assert '".[ml]"' in install_command()
    assert '".[ml,diarization]"' in install_command(True)
