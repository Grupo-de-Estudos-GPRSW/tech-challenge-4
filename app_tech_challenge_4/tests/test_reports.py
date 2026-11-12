import csv
import io

from clinical_audio.demo import demo_audio, demo_result
from clinical_audio.reports import csv_bytes, markdown_report, segment_rows


def test_demo_is_explicit_and_export_roundtrip_preserves_accents():
    result = demo_result()
    assert result.mode == "demo"
    assert demo_audio().startswith(b"RIFF")
    rows = list(csv.DictReader(io.StringIO(csv_bytes(result).decode("utf-8-sig"))))
    assert len(rows) == 2
    assert "à noite" in rows[1]["text"]
    assert "Modo: demo" in markdown_report(result)
    assert "Modelos e configuração" in markdown_report(result)


def test_csv_escapes_formulas_and_handles_quotes_and_no_text_analysis():
    result = demo_result()
    result.segments[0].text = '=HYPERLINK("x","ação")'
    result.text_analysis = None
    rows = list(csv.DictReader(io.StringIO(csv_bytes(result).decode("utf-8-sig"))))
    assert rows[0]["text"].startswith("'=")
    assert rows[0]["discordance"] == "Não avaliada"
    assert "não disponível" in markdown_report(result)
    result.segments = []
    assert segment_rows(result) == []
    assert csv_bytes(result).decode("utf-8-sig").startswith("id,text,discordance")
