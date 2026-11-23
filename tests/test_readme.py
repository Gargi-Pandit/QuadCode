"""The README stays true: its commands parse, the paths it names exist, its line-up matches config.yaml."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest
import yaml

from quadcode.cli import build_parser

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
RUN_FILES = {"final_solution.py"}  # written into run directories, not part of the repository


def bash_lines() -> list[str]:
    blocks = re.findall(r"```bash\n(.*?)```", README, flags=re.DOTALL)
    return [line.strip() for block in blocks for line in block.splitlines() if line.strip()]


def quadcode_commands() -> list[str]:
    return [line for line in bash_lines() if line.startswith("quadcode ")]


def test_readme_has_commands() -> None:
    assert len(quadcode_commands()) >= 10


@pytest.mark.parametrize("command", quadcode_commands())
def test_every_quadcode_command_parses(command: str) -> None:
    try:
        args = build_parser().parse_args(shlex.split(command)[1:])
    except SystemExit as exc:  # argparse exits on an unknown command or flag
        pytest.fail(f"README command does not parse ({exc.code}): {command}")
    assert callable(args.func)


def test_other_commands_point_at_real_files() -> None:
    for line in bash_lines():
        words = shlex.split(line)
        if words[:2] == ["streamlit", "run"]:
            assert (ROOT / words[2]).is_file(), line
        if words[0] == "pytest":
            for w in words[1:]:
                if not w.startswith("-"):
                    assert (ROOT / w).exists(), line
        if words[0] == "cp":
            assert (ROOT / words[1]).is_file(), line


def test_named_paths_exist() -> None:
    names = set(re.findall(r"`([^`\s<>*]+)`", README))
    package = {p.name for p in (ROOT / "quadcode").rglob("*.py")}
    top = {"quadcode", "samples", "benchmarks", "tests", ".streamlit"}  # model ids like `openai/x` are not paths
    missing: list[str] = []
    for name in sorted(names):
        clean = name.rstrip("/")
        if "/" in clean and clean.split("/")[0] in top:
            if not (ROOT / clean).exists():
                missing.append(name)
        elif re.fullmatch(r"\w+\.py", clean) and clean not in package and clean not in RUN_FILES:
            missing.append(name)
    for doc in re.findall(r"\]\(([\w.\-/]+)\)", README):
        if not (ROOT / doc).exists():
            missing.append(doc)
    assert not missing, f"README names paths that do not exist: {missing}"


def test_line_up_table_matches_config_yaml() -> None:
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    rows = re.findall(r"^\| (Generator|Reviewer|Optimizer|Validator) \| `([^`]+)` \| `([^`]+)` \|", README,
                      flags=re.MULTILINE)
    assert len(rows) == 4
    for role, first, fallback in rows:
        agent = config["agents"][role.lower()]
        assert agent["model"] == first, role
        assert agent["fallbacks"][0]["model"] == fallback, role
    assert config["allow_duplicate_models"] is True  # the README explains why two slots share a model
