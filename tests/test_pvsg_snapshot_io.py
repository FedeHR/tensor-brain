import hashlib
import json

from experiments.pvsg.snapshot_io import read_json, read_jsonl, sha256_file


def test_snapshot_io_reads_json_and_jsonl_and_hashes_incrementally(tmp_path) -> None:
    document_path = tmp_path / "document.json"
    document_path.write_text(json.dumps({"value": 3}), encoding="utf-8")
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text('{"row":1}\n\n{"row":2}\n', encoding="utf-8")

    assert read_json(document_path) == {"value": 3}
    assert read_jsonl(rows_path) == [{"row": 1}, {"row": 2}]
    assert sha256_file(document_path) == hashlib.sha256(document_path.read_bytes()).hexdigest()
