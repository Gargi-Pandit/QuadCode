"""The Streamlit app, driven headlessly with AppTest over mock runs and a mock bench.

Viewing must make no API call and change no byte of any run; agent text must stay inert.
"""

from __future__ import annotations

import copy
import re
import time
from pathlib import Path

import pytest
import yaml

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

from quadcode.evaluation.conditions import Condition  # noqa: E402
from quadcode.evaluation.outputs import write_outputs  # noqa: E402

from .bench_helpers import make_harness  # noqa: E402
from .orch_helpers import TEST_CONFIG  # noqa: E402
from .ui_helpers import APP, digest, mock_run  # noqa: E402

TIMEOUT = 120


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("ui-runs")
    mock_run(root, "run")
    mock_run(root, "hostile", hostile=True, rounds=1)
    harness = make_harness(root, "b1", ["lis_length"], [Condition.FULL], rounds=1)
    harness.run()
    write_outputs(harness.bench_dir)
    (root / "cache").mkdir(exist_ok=True)
    return root


@pytest.fixture
def app(runs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(copy.deepcopy(TEST_CONFIG)))
    monkeypatch.setenv("QUADCODE_RUNS_DIR", str(runs))
    monkeypatch.setenv("QUADCODE_CONFIG", str(cfg))
    monkeypatch.setenv("QUADCODE_ENV_FILE", str(tmp_path / "none.env"))
    return AppTest.from_file(str(APP), default_timeout=TIMEOUT)


def _ok(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]


def _pick_run(at: AppTest, run_id: str) -> None:
    box = at.sidebar.selectbox(key="pick-solve")
    labels = [str(o) for o in box.options]
    box.select_index(next(i for i, label in enumerate(labels) if label.startswith(f"{run_id} "))).run()


def test_run_view_renders_every_tab_without_writing(app: AppTest, runs: Path) -> None:
    before = digest(runs)
    app.run()
    _ok(app)
    _pick_run(app, "run")
    _ok(app)
    assert [t.value for t in app.title][0] == "Run run"
    assert [t.label for t in app.tabs] == ["Overview", "Report", "Communication timeline", "Calls & tokens",
                                           "Versions & diffs", "Modifications", "Tests & benchmark"]
    assert any(m.label == "Status" and m.value == "COMPLETED" for m in app.metric)
    headers = [h.value for h in app.header]
    for title in ("1. Problem analysis", "6. Testing and benchmarking", "10. Reproduction instructions"):
        assert title.replace(".", "\\.") in headers or title in headers
    assert any("Mock run" in i.value for i in app.info)
    app.selectbox(key="run-section").select("7. Complete modification history").run()
    app.multiselect(key="run-tl-from").select("validator").run()
    app.selectbox(key="run-v-old").select("V0").run()
    app.selectbox(key="run-v-new").select("V1").run()
    app.selectbox(key="run-mod-pick").select("M-002").run()
    _ok(app)
    assert any(c.language == "diff" and "+++ V1" in c.value for c in app.code)
    assert any("rejected" in m.value for m in app.markdown)
    assert digest(runs) == before  # nothing written, so no call was made and no record changed


def test_hostile_agent_text_stays_inert(app: AppTest) -> None:
    app.run()
    _pick_run(app, "hostile")
    _ok(app)
    texts = [m.value for m in app.markdown] + [w.value for w in app.warning] + [i.value for i in app.info]
    joined = "\n".join(texts)
    assert "script" in joined  # the hostile review is shown ...
    assert not re.search(r"(?<!\\)<(script|img|/code|/pre)", joined)  # ... with every '<' escaped
    assert not re.search(r"(?<!\\)\]\(javascript", joined)
    cells = "\n".join(str(v) for d in app.dataframe for v in d.value.to_numpy().ravel())
    assert "costs $x$ and :red[y]" in cells  # tables show the text as plain cells, unchanged


def test_bench_view_and_sub_run(app: AppTest, runs: Path) -> None:
    before = digest(runs)
    app.run()
    app.sidebar.radio(key="view").set_value("Benches").run()
    _ok(app)
    assert [t.value for t in app.title][0] == "Bench b1"
    assert any("Mock bench" in i.value for i in app.info)
    tables = [d.value for d in app.dataframe]
    conditions = next(t for t in tables if "condition" in t.columns and "problems scored" in t.columns)
    assert "full" in list(conditions["condition"]) and "oracle" in list(conditions["condition"])
    assert len(app.get("image")) >= 1  # the chart PNGs written by bench-report
    app.selectbox(key="bench-b1-sub").select("lis_length--full").run()
    _ok(app)
    assert "Run lis\\_length--full" in [t.value for t in app.title]  # escaped Markdown, shown as lis_length
    assert digest(runs) == before


def test_launcher_defaults_to_mock_and_gates_real_runs(app: AppTest, runs: Path) -> None:
    app.run()
    app.sidebar.radio(key="view").set_value("New run").run()
    _ok(app)
    assert app.radio(key="solve-mode").value.startswith("mock")
    commands = [c.value for c in app.code if c.language == "bash"]
    assert commands and "--mock" in commands[0] and commands[0].startswith("quadcode --config")
    app.radio(key="solve-mode").set_value("real providers").run()
    _ok(app)
    assert any("confirm" in e.value for e in app.error)
    assert not [b for b in app.button if b.key == "solve-start"]  # nothing can be started yet
    app.checkbox(key="solve-confirm").check().run()
    commands = [c.value for c in app.code if c.language == "bash"]
    assert commands and "--mock" not in commands[0] and "--budget-calls 40" in commands[0]
    assert not (runs / ".ui-logs").exists()  # nothing was started


def test_empty_runs_dir_explains_how_to_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QUADCODE_RUNS_DIR", str(tmp_path / "none"))
    monkeypatch.setenv("QUADCODE_CONFIG", str(tmp_path / "missing.yaml"))
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT)
    at.run()
    _ok(at)
    assert any("No runs" in i.value for i in at.info)
    assert any("config not loaded" in w.value for w in at.sidebar.warning)


def test_start_button_launches_the_cli_and_follows_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = copy.deepcopy(TEST_CONFIG)
    data["run"] = {"rounds": 1, "cache_dir": str(tmp_path / "cache"), "runs_dir": str(tmp_path / "runs")}
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(data))
    monkeypatch.setenv("QUADCODE_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("QUADCODE_CONFIG", str(cfg))
    monkeypatch.setenv("QUADCODE_ENV_FILE", str(tmp_path / "none.env"))
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT)
    at.run()
    at.sidebar.radio(key="view").set_value("New run").run()
    at.slider(key="solve-rounds").set_value(1).run()
    at.text_input(key="solve-id").set_value("ui-app-e2e").run()
    shown = [c.value for c in at.code if c.language == "bash"][0]
    assert "--run-id ui-app-e2e" in shown and "--mock" in shown
    at.button(key="solve-start").click().run()
    _ok(at)
    summary = tmp_path / "runs" / "ui-app-e2e" / "summary.json"
    deadline = time.monotonic() + 300
    while not summary.is_file() and time.monotonic() < deadline:
        time.sleep(0.5)
    assert summary.is_file(), (tmp_path / "runs" / ".ui-logs" / "ui-app-e2e.log").read_text()
    log = tmp_path / "runs" / ".ui-logs" / "ui-app-e2e.log"
    assert log.read_text().startswith(f"$ {shown}")  # exactly the command that was shown was run
    for _ in range(100):  # summary.json is written before the report; wait for the process to exit
        at.run()
        if any("exited with code 0" in m.value for m in at.markdown):
            break
        time.sleep(0.5)
    _ok(at)
    assert any("ui-app-e2e" in m.value and "exited with code 0" in m.value and "completed" in m.value
               for m in at.markdown)
    assert at.text_input(key="solve-id").value != "ui-app-e2e"  # a fresh id is offered for the next run
