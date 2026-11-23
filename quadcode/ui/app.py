"""QuadCode dashboard: ``streamlit run quadcode/ui/app.py -- [--runs-dir runs] [--config config.yaml]``.

A thin view over saved runs and benches. Viewing makes no API call and writes nothing; new runs are
started only through the regular command line (see ``launch.py``). Options can also come from the
environment: QUADCODE_RUNS_DIR, QUADCODE_CONFIG, QUADCODE_ENV_FILE.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import streamlit as st

from quadcode.config import AppConfig, ConfigError, load_config
from quadcode.ui import catalog
from quadcode.ui.launch import CommonOptions
from quadcode.ui.page_bench import show_bench
from quadcode.ui.page_launch import show_launcher
from quadcode.ui.page_run import load_run_cached, show_run
from quadcode.ui.render import safe_inline, safe_text

VIEWS = ["Runs", "Benches", "New run"]


def options(argv: list[str]) -> tuple[Path | None, Path, Path]:
    """(runs dir, config path, env file) from ``-- --runs-dir ...`` arguments, else the environment.
    The runs dir is None when neither gives one (the config's ``run.runs_dir`` is used then)."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--runs-dir", default=None)
    p.add_argument("--config", default=os.environ.get("QUADCODE_CONFIG", "config.yaml"))
    p.add_argument("--env-file", default=os.environ.get("QUADCODE_ENV_FILE", ".env"))
    args, _ = p.parse_known_args(argv)
    config_path = Path(args.config)
    runs = args.runs_dir or os.environ.get("QUADCODE_RUNS_DIR")
    return (Path(runs) if runs else None), config_path, Path(args.env_file)


def _config(path: Path) -> tuple[AppConfig | None, str | None]:
    """config.yaml for prices, runs_dir and problems_dir. Only the YAML is read; keys are not loaded."""
    try:
        return load_config(path), None
    except (ConfigError, OSError) as exc:
        return None, f"{path}: {exc}"


def main() -> None:
    st.set_page_config(page_title="QuadCode runs", layout="wide")
    runs_arg, config_path, env_file = options(sys.argv[1:])
    config, config_error = _config(config_path)
    runs_dir = runs_arg or Path(config.run.runs_dir if config else "runs")

    st.sidebar.title("QuadCode")
    view = st.sidebar.radio("View", VIEWS, key="view")
    st.sidebar.caption(safe_inline(f"runs directory: {runs_dir}"))
    if config_error:
        st.sidebar.warning(safe_text(f"config not loaded ({config_error}); costs show as not configured"))
    if st.sidebar.button("Refresh list"):
        st.rerun()
    entries = catalog.scan(runs_dir)

    if view == "New run":
        show_launcher(CommonOptions(config=config_path, env_file=env_file, runs_dir=runs_dir), config, entries)
        return
    kind = "bench" if view == "Benches" else "solve"
    shown = [e for e in entries if e.kind == kind or (kind == "solve" and e.kind == "other")]
    if not shown:
        st.info(f"No {'benches' if kind == 'bench' else 'runs'} in {runs_dir}. Start one from 'New run', or with "
                f"`quadcode {'bench --mock' if kind == 'bench' else 'solve samples/two_sum/problem.md --mock'}`.")
        return
    entry = st.sidebar.selectbox(kind.capitalize() if kind == "bench" else "Run", shown,
                                 format_func=lambda e: e.label, key=f"pick-{kind}")
    if entry.title:
        st.sidebar.caption(safe_inline(entry.title))
    if entry.error:
        st.error(safe_text(f"{entry.id}: {entry.error}"))
    if entry.kind == "bench":
        show_bench(entry.path, config)
    elif entry.kind == "other":
        st.caption(safe_inline(f"{entry.id}: {entry.status}. This directory has no pipeline run log; it holds "
                               "versions from `quadcode apply-diff` or provider calls from `quadcode smoke`."))
        show_run(load_run_cached(entry.path), config, key=entry.id)
    else:
        show_run(load_run_cached(entry.path), config, key=entry.id)


main()
