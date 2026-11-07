"""Docs must not rot: runbook covers every alert, links resolve, README only mentions real make targets."""
import re
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _md_files(): return [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")), *sorted((ROOT / "deploy").glob("*.md"))]


def test_runbook_covers_every_alert():
    alerts = {r["alert"] for g in yaml.safe_load(open(ROOT / "deploy/prometheus/alerts.yml"))["groups"] for r in g["rules"]}
    runbook = (ROOT / "docs/RUNBOOK.md").read_text()
    headings = set(re.findall(r"^### (\w+)", runbook, re.M))
    assert alerts <= headings, f"alerts without a runbook entry: {sorted(alerts - headings)}"
    assert headings - alerts == set(), f"runbook entries for alerts that no longer exist: {sorted(headings - alerts)}"


def test_relative_links_resolve():
    for f in _md_files():
        for target in re.findall(r"\]\((?!https?://|#|mailto:)([^)#\s]+)", f.read_text()):
            assert (f.parent / target).resolve().exists(), f"{f.relative_to(ROOT)} links to missing {target}"


def test_readme_only_mentions_real_make_targets():
    make = (ROOT / "Makefile").read_text()
    defined = set(re.findall(r"^([a-zA-Z_-]+):", make, re.M))
    mentioned = set(re.findall(r"`make ([a-z-]+)", (ROOT / "README.md").read_text()))
    mentioned |= set(re.findall(r"^make ([a-z-]+)", (ROOT / "README.md").read_text(), re.M))
    assert mentioned <= defined, f"README mentions unknown targets: {sorted(mentioned - defined)}"


def test_every_make_target_is_documented_in_help():
    make = (ROOT / "Makefile").read_text()
    phony = set(make.split(".PHONY:")[1].splitlines()[0].split())
    helped = set(re.findall(r"^([a-zA-Z_-]+):.*## ", make, re.M))
    assert phony <= helped, f"targets without ## help text: {sorted(phony - helped)}"


def test_design_doc_states_limitations():
    d = (ROOT / "docs/DESIGN.md").read_text()
    assert "## 6. Limitations" in d and "Synthetic data" in d and "Not executed in my build sandbox" in d
