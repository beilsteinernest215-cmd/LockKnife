from __future__ import annotations

import concurrent.futures
import json
import threading
from types import SimpleNamespace

from lockknife.core.case import create_case_workspace
from lockknife_headless_cli import tui_callback as cb


def test_concurrent_callbacks_finalize_their_own_case_job(tmp_path, monkeypatch):
    barrier = threading.Barrier(2)

    def dispatch(app, action, params):
        barrier.wait(timeout=5)
        return cb._ok({"case": params["case_dir"]}, "done")

    registry = SimpleNamespace(dispatch=dispatch, catalog_json=lambda: "{}")
    monkeypatch.setattr(cb, "build_action_registry", lambda: registry)
    callback = cb.build_tui_callback(None)
    cases = [tmp_path / "first", tmp_path / "second"]
    for case in cases:
        create_case_workspace(case_dir=case, case_id=case.name, examiner="EXAMPLE", title="EXAMPLE")

    def run(case):
        result = callback("forensics.sqlite", {"case_dir": str(case), "path": "EXAMPLE.db"})
        assert result["ok"]
        job = json.loads(result["job_json"])
        assert job["case_dir"] == str(case)
        return job

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, cases))
    assert len(results) == 2
    assert cb._JOB_TRACKER_STACK.get() == ()
