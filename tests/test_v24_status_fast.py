from __future__ import annotations

import json

from profit_taker import v24_contract_runtime_v4 as runtime


def test_status_is_read_only_by_default(tmp_path, monkeypatch, capsys):
    db = tmp_path / "raw.sqlite"

    def forbidden_refresh(*args, **kwargs):
        raise AssertionError("ordinary status must not rebuild historical pretraining targets")

    monkeypatch.setattr(runtime.runtime, "_refresh", forbidden_refresh)
    monkeypatch.setattr(
        runtime.pretraining_status,
        "status_snapshot",
        lambda *args, **kwargs: {"available": True, "targets_refreshed": False},
    )
    monkeypatch.setattr(runtime.v24, "status", lambda *args, **kwargs: {"model_status": "ok"})

    assert runtime.main(["status", "--db", str(db)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["model_status"] == "ok"
    assert payload["pretraining_contract"]["mode"] == "read_only_snapshot"
    assert payload["pretraining_contract"]["targets_refreshed"] is False


def test_status_full_refresh_is_explicit_opt_in(tmp_path, monkeypatch, capsys):
    db = tmp_path / "raw.sqlite"
    calls: list[str] = []

    def fake_refresh(path, cfg):
        calls.append(str(path))
        return {"targets": {"written": 7}, "counterfactual_friction": {"updated": 0}}

    monkeypatch.setattr(runtime.runtime, "_refresh", fake_refresh)
    monkeypatch.setattr(
        runtime.pretraining_status,
        "status_snapshot",
        lambda *args, **kwargs: {"available": True, "targets_refreshed": False},
    )
    monkeypatch.setattr(runtime.v24, "status", lambda *args, **kwargs: {"model_status": "ok"})

    assert runtime.main(["status", "--db", str(db), "--refresh-pretraining"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert calls == [str(db)]
    assert payload["pretraining_contract"]["mode"] == "full_refresh"
    assert payload["pretraining_contract"]["targets"]["written"] == 7
