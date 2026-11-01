"""Static checks on the container stack, runnable anywhere (no Docker needed)."""
import json, re, shutil, subprocess
from pathlib import Path
import pytest, yaml

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = yaml.safe_load(open(ROOT / "docker-compose.yml"))
SERVICES = COMPOSE["services"]


def test_depends_on_references_real_services_and_conditions():
    for name, s in SERVICES.items():
        deps = s.get("depends_on", {})
        for dep, spec in (deps.items() if isinstance(deps, dict) else [(d, {}) for d in deps]):
            assert dep in SERVICES, f"{name} depends on unknown service {dep}"
            cond = spec.get("condition")
            if cond == "service_healthy":
                assert "healthcheck" in SERVICES[dep] or dep in ("api",), f"{dep} has no healthcheck for {name} to wait on"
                assert not SERVICES[dep].get("healthcheck", {}).get("disable"), f"{name} waits on {dep}, whose healthcheck is disabled"
            if cond == "service_completed_successfully":
                assert SERVICES[dep].get("restart") == "no", f"{dep} is a one-shot job and must not restart"


def test_startup_order_is_a_dag_with_the_intended_shape():
    deps = {n: set(s.get("depends_on", {})) for n, s in SERVICES.items()}
    def closure(n, seen=()):
        assert n not in seen, f"dependency cycle through {n}"
        return set().union(*[{d} | closure(d, seen + (n,)) for d in deps[n]]) if deps[n] else set()
    assert {"postgres", "mlflow", "bootstrap"} <= closure("api")        # api never starts without a model + databases
    assert "api" in closure("worker") and "api" in closure("prometheus")


def test_application_services_share_one_image_and_consistent_env():
    app = [n for n, s in SERVICES.items() if s.get("image", "").startswith("mlops-platform")]
    assert {"mlflow", "bootstrap", "api", "worker", "traffic"} <= set(app)
    for n in app:
        if n == "mlflow":
            continue                 # the tracking server IS the target of MLFLOW_TRACKING_URI; it needs no client settings
        env = SERVICES[n]["environment"]
        assert env["MLFLOW_TRACKING_URI"] == "http://mlflow:5000" and "@postgres:5432/mlops" in env["DATABASE_URL"] and "ADMIN_TOKEN" in env


def test_mlflow_is_configured_for_container_networking():
    cmd = " ".join(SERVICES["mlflow"]["command"])
    assert "--serve-artifacts" in cmd and "--allowed-hosts" in cmd and "mlflow:5000" in cmd and "/mlflow/artifacts" in cmd
    assert "postgresql+psycopg2://" in cmd and cmd.count("@postgres:5432/mlflow") == 1       # its OWN database, not the serving one


def test_every_mounted_file_exists_and_every_volume_is_declared():
    declared = set(COMPOSE["volumes"])
    for n, s in SERVICES.items():
        for v in s.get("volumes", []):
            src = v.split(":")[0]
            if src.startswith("./"):
                assert (ROOT / src).exists(), f"{n}: missing mount source {src}"
            else:
                assert src in declared, f"{n}: undeclared volume {src}"
    assert (ROOT / "deploy/postgres/init-databases.sh").stat().st_mode & 0o111, "init script must be executable"


def test_databases_and_admin_ports_are_not_exposed_to_the_network():
    for n in ("postgres", "mlflow", "prometheus"):
        for p in SERVICES[n]["ports"]:
            assert p.startswith("127.0.0.1:"), f"{n} publishes {p} on all interfaces"


def test_dockerfile_hardening():
    d = (ROOT / "Dockerfile").read_text()
    assert re.search(r"^USER mlops", d, re.M) and "HEALTHCHECK" in d and "AS builder" in d and "AS runtime" in d
    assert "requirements.txt" in d and "requirements-dev" not in d                   # test tooling never ships
    assert "--factory" in d


def test_env_example_covers_every_variable_the_compose_file_uses():
    used = set(re.findall(r"\$\{([A-Z_]+)", (ROOT / "docker-compose.yml").read_text()))
    defined = {l.split("=")[0] for l in (ROOT / ".env.example").read_text().splitlines() if "=" in l and not l.startswith("#")}
    assert used <= defined, f"undocumented variables: {sorted(used - defined)}"


@pytest.mark.skipif(not (shutil.which("docker-compose") or shutil.which("docker")), reason="docker compose CLI not installed")
def test_real_compose_cli_accepts_the_file():
    exe = [shutil.which("docker-compose")] if shutil.which("docker-compose") else ["docker", "compose"]
    for prof in ([], ["--profile", "demo"]):
        r = subprocess.run([*exe, "--env-file", ".env.example", *prof, "config", "-q"], cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


@pytest.mark.skipif(not shutil.which("hadolint"), reason="hadolint not installed")
def test_dockerfile_lints_clean():
    assert subprocess.run(["hadolint", str(ROOT / "Dockerfile")], capture_output=True, text=True).returncode == 0
