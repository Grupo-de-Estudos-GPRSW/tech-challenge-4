import csv
import io

from clinical_audio.demo import demo_audio, demo_result
from clinical_audio.reports import (
    csv_bytes,
    markdown_report,
    notebook_csv_bytes,
    notebook_segment_rows,
    segment_rows,
)


def test_reference_columns_and_downloads():
    result = demo_result()
    assert demo_audio().startswith(b"RIFF")
    rows = list(csv.DictReader(io.StringIO(notebook_csv_bytes(result).decode("utf-8-sig"))))
    assert len(rows) == 2
    assert rows[1]["interlocutor"] == "Paciente"
    assert rows[1]["nivel_discordancia"] == "Alta"
    assert rows[1]["ha_discordancia"] == "True"
    assert "confianca_voz_secundaria" in rows[0]
    assert "## 1. Resumo Quantitativo da Consulta" in markdown_report(result)
    assert "Total de discordâncias detectadas:** 1 (50.0%)" in markdown_report(result)
    assert notebook_segment_rows(result)[1]["eventos_nao_verbais_proximos"] == ["Tosse"]


def test_complementary_export_and_empty_report():
    result = demo_result()
    result.segments[0].text = '=HYPERLINK("x","ação")'
    assert "'=HYPERLINK" in csv_bytes(result).decode("utf-8-sig")
    result.text_analysis = None
    assert segment_rows(result)[0]["discordance"] == "Não avaliada"
    result.segments = []
    assert segment_rows(result) == []
    assert "0 (0.0%)" in markdown_report(result)
