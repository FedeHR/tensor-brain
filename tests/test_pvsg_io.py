import hashlib

from experiments.pvsg.io import (
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)


def test_io_reads_writes_appends_and_hashes(tmp_path) -> None:
    document_path = tmp_path / "nested" / "document.json"
    rows_path = tmp_path / "rows.jsonl"

    write_json(document_path, {"value": 3}, sort_keys=True)
    write_jsonl(rows_path, ({"row": 1},))
    write_jsonl(rows_path, ({"row": 2},), append=True)

    assert read_json(document_path) == {"value": 3}
    assert read_jsonl(rows_path) == [{"row": 1}, {"row": 2}]
    assert sha256_file(document_path) == hashlib.sha256(document_path.read_bytes()).hexdigest()
