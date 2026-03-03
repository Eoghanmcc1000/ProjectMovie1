from __future__ import annotations

from scripts.ingest import extract_year, parse_json_column, safe_float, safe_int


def test_parse_json_column_valid() -> None:
    raw = '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]'
    result = parse_json_column(raw)
    assert len(result) == 2
    assert result[0]["name"] == "Action"


def test_parse_json_column_empty() -> None:
    assert parse_json_column("") == []
    assert parse_json_column(None) == []


def test_parse_json_column_malformed() -> None:
    assert parse_json_column("not json at all") == []
    assert parse_json_column("[{broken}]") == []


def test_extract_year() -> None:
    assert extract_year("2010-07-16") == 2010
    assert extract_year("1999-03-31") == 1999
    assert extract_year("") is None
    assert extract_year(None) is None
    assert extract_year("bad") is None


def test_safe_int() -> None:
    assert safe_int("42") == 42
    assert safe_int("116.0") == 116
    assert safe_int("") == 0
    assert safe_int(None) == 0
    assert safe_int("bad") == 0
    assert safe_int("0") == 0


def test_safe_float() -> None:
    assert safe_float("7.5") == 7.5
    assert safe_float("") == 0.0
    assert safe_float(None) == 0.0
    assert safe_float("bad") == 0.0
