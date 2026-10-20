import json
import os
import time

import pytest

from eqpred.safe_io import file_lock, read_json, write_json


def test_write_keeps_a_backup_and_leaves_no_temp_file(tmp_path):
    path = tmp_path / "ledger.json"
    write_json(path, {"entries": [1]}, backup=True)
    write_json(path, {"entries": [1, 2]}, backup=True)
    assert json.loads(path.read_text(encoding="utf-8")) == {"entries": [1, 2]}
    assert json.loads((tmp_path / "ledger.json.bak").read_text(encoding="utf-8")) == {"entries": [1]}
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]


def test_a_damaged_file_names_its_backup(tmp_path):
    path = tmp_path / "ledger.json"
    path.write_text('{"entries": [', encoding="utf-8")
    (tmp_path / "ledger.json.bak").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="ledger.json.bak"):
        read_json(path)


def test_the_lock_waits_and_a_stale_lock_is_cleared(tmp_path):
    target = tmp_path / "ledger.json"
    lock = tmp_path / "ledger.json.lock"
    with file_lock(target):
        assert lock.exists()
        with pytest.raises(TimeoutError):
            with file_lock(target, timeout=0.3):
                pass
    assert not lock.exists()
    lock.write_text("12345", encoding="utf-8")
    old = time.time() - 600
    os.utime(lock, (old, old))
    with file_lock(target, timeout=1.0):
        pass
    assert not lock.exists()
