import pytest

from clinical_audio.errors import processing_error_message


@pytest.mark.parametrize(
    "error,expected",
    [
        (MemoryError("private"), "Memória insuficiente"),
        (RuntimeError("CUDA out of memory private"), "Memória insuficiente"),
        (OSError("401 private token"), "Acesso ao modelo negado"),
        (OSError("certificate verify failed private"), "certificado HTTPS"),
        (OSError("couldn't connect to huggingface.co private"), "acesso HTTPS"),
        (PermissionError("private path"), "Sem permissão"),
        (RuntimeError("private transcript"), "compatibilidade"),
    ],
)
def test_safe_stage_specific_errors(error, expected):
    message = processing_error_message(error, "Detectando eventos acústicos")
    assert expected in message
    assert "Detectando eventos acústicos" in message
    assert "private" not in message


def test_exception_chain_and_cycles_are_supported():
    error = OSError("wrapper")
    error.__cause__ = RuntimeError("SSL private token")
    error.__cause__.__context__ = error
    assert "certificado HTTPS" in processing_error_message(error, "Modelo")
