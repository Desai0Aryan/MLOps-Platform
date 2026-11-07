"""scripts/deploy.sh drives `docker compose`. Here `docker` is a fake that records every call and
simulates healthy / unhealthy releases, so the rollback logic is tested for real."""
import os, shutil, subprocess
from pathlib import Path
import pytest, yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "deploy.sh"

FAKE_DOCKER = """#!/usr/bin/env bash
echo "${MLOPS_VERSION:-?} $*" >> "$SHIM_LOG"
case "$*" in *" up "*) echo "${MLOPS_VERSION:-?}" > "$SHIM_STATE" ;; esac
exit 0
"""


@pytest.fixture()
def host(tmp_path):
    (tmp_path / "bin").mkdir(); fake = tmp_path / "bin" / "docker"; fake.write_text(FAKE_DOCKER); fake.chmod(0o755)
    d = tmp_path / "srv"; d.mkdir()
    for f in ("docker-compose.yml", "docker-compose.prod.yml"): (d / f).write_text("services: {}\n")
    env = {**os.environ, "COMPOSE_DIR": str(d), "DOCKER": str(fake), "SHIM_LOG": str(tmp_path / "calls.log"), "SHIM_STATE": str(tmp_path / "state"),
           "HEALTH_TIMEOUT": "1", "POLL_SECONDS": "0",
           "HEALTH_CMD": 'case "$(cat "$SHIM_STATE" 2>/dev/null)" in bad*) exit 1;; "") exit 1;; *) exit 0;; esac'}   # any tag starting with "bad" is unhealthy
    class H:
        dir, log, state = d, tmp_path / "calls.log", tmp_path / "state"
        def run(self, *args): return subprocess.run([str(SCRIPT), *args], env=env, capture_output=True, text=True)
        def tag(self, n): p = d / n; return p.read_text().strip() if p.exists() else None
        def ups(self): return [l.split()[0] for l in self.log.read_text().splitlines() if " up " in l]
    return H()


def test_first_deploy(host):
    r = host.run("sha-1")
    assert r.returncode == 0 and host.tag(".deployed_tag") == "sha-1" and host.tag(".previous_tag") is None


def test_images_are_pulled_before_anything_is_restarted(host):
    host.run("sha-1")
    verbs = [("pull" if " pull " in l else "up") for l in host.log.read_text().splitlines() if " pull " in l or " up " in l]
    assert verbs == ["pull", "up"]


def test_second_deploy_remembers_previous(host):
    host.run("sha-1"); r = host.run("sha-2")
    assert r.returncode == 0 and host.tag(".deployed_tag") == "sha-2" and host.tag(".previous_tag") == "sha-1"


def test_unhealthy_release_rolls_back_automatically(host):
    host.run("sha-1")
    r = host.run("bad-2")
    assert r.returncode == 1 and "rolling back to sha-1" in r.stdout and "ROLLED BACK" in r.stdout
    assert host.ups() == ["sha-1", "bad-2", "sha-1"]                       # old -> bad -> old again
    assert host.tag(".deployed_tag") == "sha-1" and host.state.read_text().strip() == "sha-1"


def test_first_ever_deploy_failing_has_nothing_to_roll_back_to(host):
    r = host.run("bad-1")
    assert r.returncode == 1 and "no previous release" in r.stdout
    assert host.ups() == ["bad-1"] and host.tag(".deployed_tag") is None


def test_manual_rollback(host):
    host.run("sha-1"); host.run("sha-2")
    r = host.run("--rollback")
    assert r.returncode == 0 and host.tag(".deployed_tag") == "sha-1" and host.ups()[-1] == "sha-1"
    assert host.tag(".previous_tag") == "sha-2"                           # so you can roll forward again


def test_rollback_without_history_fails_cleanly(host):
    assert host.run("--rollback").returncode == 1
    host.run("sha-1"); assert host.run("--rollback").returncode == 1


def test_usage_error(host):
    assert host.run().returncode == 2 and host.run("a", "b").returncode == 2


@pytest.mark.skipif(not shutil.which("shellcheck"), reason="shellcheck not installed")
def test_shellcheck_clean():
    assert subprocess.run(["shellcheck", str(SCRIPT)], capture_output=True, text=True).returncode == 0


# ------------------------------------------------------------------- production compose override
def _load(path):
    class L(yaml.SafeLoader): pass
    L.add_multi_constructor("!", lambda loader, suffix, node: None)       # compose-specific tags (!reset / !override)
    return yaml.load(open(path), Loader=L)


def test_prod_override_never_builds_and_never_exposes_publicly():
    prod = _load(ROOT / "docker-compose.prod.yml")["services"]
    for svc in ("mlflow", "bootstrap", "api", "worker"):
        assert prod[svc]["pull_policy"] == "always" and "build" in prod[svc], svc      # build is reset to null
    text = (ROOT / "docker-compose.prod.yml").read_text()
    assert text.count('"127.0.0.1:') >= 2 and ":?" in text                              # localhost-only ports; required secrets
    assert "change-me" not in text and ":-mlops" not in text                            # no dev defaults leak into prod


@pytest.mark.skipif(not (shutil.which("docker-compose") or shutil.which("docker")), reason="docker compose CLI not installed")
def test_real_compose_merges_the_prod_override():
    exe = [shutil.which("docker-compose")] if shutil.which("docker-compose") else ["docker", "compose"]
    env = {**os.environ, "MLOPS_VERSION": "sha-abc", "IMAGE_REPO": "acme/x", "POSTGRES_USER": "u", "POSTGRES_PASSWORD": "p", "ADMIN_TOKEN": "t", "GRAFANA_ADMIN_PASSWORD": "g"}
    r = subprocess.run([*exe, "-f", "docker-compose.yml", "-f", "docker-compose.prod.yml", "config", "--format", "json"], cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    import json
    s = json.loads(r.stdout)["services"]
    assert s["api"]["image"] == "ghcr.io/acme/x:sha-abc" and "build" not in s["api"]
    assert all(p["host_ip"] == "127.0.0.1" for p in s["api"]["ports"] + s["grafana"]["ports"])
    env.pop("ADMIN_TOKEN")
    bad = subprocess.run([*exe, "-f", "docker-compose.yml", "-f", "docker-compose.prod.yml", "config", "-q"], cwd=ROOT, env=env, capture_output=True, text=True)
    assert bad.returncode != 0 and "ADMIN_TOKEN" in bad.stderr                           # fails fast without secrets
