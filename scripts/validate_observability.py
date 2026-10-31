"""Machine-validates everything under deploy/ with the real `promtool` (used by CI in Part 8).

  * prometheus.yml parses and its rule files load
  * every alert rule parses (13 rules)
  * every PromQL expression in every Grafana dashboard + annotation parses

    python scripts/validate_observability.py [--require]     # --require: fail if promtool is missing
"""
import glob, json, re, shutil, subprocess, sys, tempfile
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    promtool = shutil.which("promtool")
    if not promtool:
        print("promtool not found: skipping" + (" (FAIL: --require)" if "--require" in sys.argv else ""))
        return 1 if "--require" in sys.argv else 0
    ok = True
    for args in (["check", "config", str(ROOT / "deploy/prometheus/prometheus.yml")],
                 ["check", "rules", str(ROOT / "deploy/prometheus/alerts.yml")]):
        r = subprocess.run([promtool, *args], capture_output=True, text=True)
        print(("✔" if r.returncode == 0 else "✘"), " ".join(args[:2]), args[2].split("/")[-1]); ok &= r.returncode == 0
        if r.returncode: print(r.stdout, r.stderr)
    exprs = []
    for f in sorted(glob.glob(str(ROOT / "deploy/grafana/dashboards/*.json"))):
        d = json.load(open(f))
        exprs += [t["expr"] for p in d["panels"] for t in p.get("targets", [])] + [a["expr"] for a in d["annotations"]["list"] if "expr" in a]
    rules = [{"record": f"t{i}", "expr": e.replace("$__rate_interval", "5m").replace("$__interval", "1m")} for i, e in enumerate(exprs)]
    with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False) as fh:
        yaml.safe_dump({"groups": [{"name": "dashboards", "rules": rules}]}, fh)
    r = subprocess.run([promtool, "check", "rules", fh.name], capture_output=True, text=True)
    print(("✔" if r.returncode == 0 else "✘"), f"{len(exprs)} dashboard queries parse"); ok &= r.returncode == 0
    if r.returncode: print(r.stdout, r.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
