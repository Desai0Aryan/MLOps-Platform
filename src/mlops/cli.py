"""`python -m mlops.cli ...` : a fancy terminal front-end for the platform."""
from __future__ import annotations
import argparse, sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from .config import load_config, ROOT
from .data import generate_dataset, SCENARIOS, validate_dataframe
from .training import run_training_pipeline, DataValidationError, QualityGateError

console = Console()
BANNER = r"""[bold cyan]
  __  __ _     ___           ____  _       _    __
 |  \/  | |   / _ \ _ __  __|  _ \| | __ _| |_ / _| ___  _ __ _ __ ___
 | |\/| | |  | | | | '_ \/ __| |_) | |/ _` | __| |_ / _ \| '__| '_ ` _ \
 | |  | | |__| |_| | |_) \__ \  __/| | (_| | |_|  _| (_) | |  | | | | | |
 |_|  |_|_____\___/| .__/|___/_|   |_|\__,_|\__|_|  \___/|_|  |_| |_| |_|
                   |_|[/bold cyan]  [dim]end-to-end MLOps with drift detection[/dim]
"""


def _validation_table(report) -> Table:
    t = Table(title="Data validation", header_style="bold magenta")
    for c in ("Check", "Column", "Status", "Detail"):
        t.add_column(c)
    for c in report.checks:
        ok = "[green]PASS[/green]" if c.passed else ("[red]FAIL[/red]" if c.severity == "error" else "[yellow]WARN[/yellow]")
        t.add_row(c.name, c.column or "-", ok, c.detail)
    return t


def cmd_generate(a, cfg):
    df = generate_dataset(a.rows or cfg["data"]["n_train"], a.scenario, seed=a.seed, with_target=True)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    console.print(f"[green]✔[/green] wrote {len(df):,} rows ({a.scenario}) -> {out}  churn rate={df['churned'].mean():.1%}")


def cmd_validate(a, cfg):
    import pandas as pd
    rep = validate_dataframe(pd.read_csv(a.data), cfg)
    console.print(_validation_table(rep) if a.verbose or not rep.passed else "")
    console.print("[green]VALID[/green]" if rep.passed else "[red]INVALID[/red]",
                  f"- {len(rep.checks) - len(rep.failures)}/{len(rep.checks)} checks passed")
    return 0 if rep.passed else 1


def cmd_train(a, cfg):
    import pandas as pd
    df = pd.read_csv(a.data) if a.data else generate_dataset(a.rows or cfg["data"]["n_train"], a.scenario, seed=cfg["project"]["seed"])
    console.print(BANNER)
    console.print(Panel(f"Scenario [bold]{a.scenario}[/bold] · rows [bold]{len(df):,}[/bold] · "
                        f"tracking [bold]{cfg['mlflow']['tracking_uri']}[/bold]", title="Training pipeline"))
    with Progress(TextColumn("[bold blue]{task.description:<28}"), BarColumn(bar_width=None),
                  TextColumn("{task.percentage:>3.0f}%"), TimeElapsedColumn(), console=console) as prog:
        task = prog.add_task("starting", total=100)
        def cb(stage, frac): prog.update(task, description=stage, completed=frac * 100)
        try:
            res = run_training_pipeline(df, cfg, tags={"scenario": a.scenario}, progress=cb)
        except DataValidationError as e:
            prog.stop(); console.print(_validation_table(e.report))
            console.print(Panel(str(e), title="[red]PIPELINE REJECTED: bad data[/red]", border_style="red")); return 2
        except QualityGateError as e:
            prog.stop(); console.print(Panel(str(e), title="[red]PIPELINE REJECTED: quality gate[/red]", border_style="red")); return 3

    t = Table(title="Candidate leaderboard", header_style="bold cyan")
    for c in ("Model", "CV AUC", "Test AUC", "PR AUC", "F1", "Brier"):
        t.add_column(c, justify="left" if c == "Model" else "right")
    for n, m in sorted(res.candidates.items(), key=lambda kv: -kv[1]["cv_auc"]):
        star = " 🏆" if n == res.best_model else ""
        t.add_row(n + star, f"{m['cv_auc']:.4f}", f"{m['roc_auc']:.4f}", f"{m['pr_auc']:.4f}", f"{m['f1']:.4f}", f"{m['brier']:.4f}")
    console.print(t)
    console.print(Panel(f"run_id  [bold]{res.run_id}[/bold]\nmodel   {res.model_uri}\ndata    sha:{res.data_hash}\n"
                        f"time    {res.duration_s:.1f}s\n\nView experiments:  [cyan]mlflow ui --backend-store-uri {cfg['mlflow']['tracking_uri']}[/cyan]",
                        title="[green]✔ Pipeline succeeded[/green]", border_style="green"))
    if a.register or a.promote:
        from .registry import ModelRegistry
        from .cli_registry import decision_panel, registry_table
        reg = ModelRegistry(cfg); mv = reg.register_run(res.run_id)
        console.print(f"[green]✔[/green] registered as [bold]v{mv.version}[/bold] (challenger)")
        if a.promote:
            d = reg.promote_if_better(mv.version); console.print(decision_panel(d, mv.version))
        console.print(registry_table(reg))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="mlops", description=__doc__)
    p.add_argument("--config", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="create a synthetic dataset")
    g.add_argument("--scenario", choices=SCENARIOS, default="baseline"); g.add_argument("--rows", type=int)
    g.add_argument("--seed", type=int, default=42); g.add_argument("--out", default="data/train.csv")
    v = sub.add_parser("validate", help="validate a CSV against the data contract")
    v.add_argument("data"); v.add_argument("-v", "--verbose", action="store_true")
    t = sub.add_parser("train", help="run the full training pipeline")
    t.add_argument("--data"); t.add_argument("--scenario", choices=SCENARIOS, default="baseline")
    t.add_argument("--rows", type=int, help="override training rows (generated data only)")
    t.add_argument("--register", action="store_true", help="register the model after training")
    t.add_argument("--promote", action="store_true", help="run the promotion gate after registering")
    r = sub.add_parser("registry", help="inspect and manage the model registry")
    rs = r.add_subparsers(dest="rcmd", required=True)
    rs.add_parser("list")
    rg = rs.add_parser("register"); rg.add_argument("--run-id")
    pr = rs.add_parser("promote"); pr.add_argument("version", nargs="?", type=int); pr.add_argument("--force", action="store_true")
    rb = rs.add_parser("rollback"); rb.add_argument("--reason", default="manual rollback")
    sv = sub.add_parser("serve", help="run the FastAPI serving layer")
    sv.add_argument("--host", default="0.0.0.0"); sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--reload", action="store_true")
    sm = sub.add_parser("simulate", help="replay synthetic traffic (optionally drifted) against a running API")
    sm.add_argument("--url", default="http://localhost:8000"); sm.add_argument("--scenario", choices=SCENARIOS, default="baseline")
    sm.add_argument("--n", type=int, default=500); sm.add_argument("--label-rate", type=float, default=0.7)
    sm.add_argument("--seed", type=int, default=123)
    sub.add_parser("demo", help="scripted end-to-end registry demo (v1 -> v2 promoted -> v3 rejected -> rollback)")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    if a.cmd == "registry":
        from .cli_registry import run_registry_command
        return run_registry_command(a, cfg)
    if a.cmd == "serve":
        import uvicorn
        uvicorn.run("mlops.serving.app:create_app", factory=True, host=a.host, port=a.port, reload=a.reload)
        return 0
    if a.cmd == "simulate":
        import httpx
        from .simulate import simulate
        with httpx.Client(base_url=a.url, timeout=60) as client, Progress(
                TextColumn("[bold blue]{task.description:<30}"), BarColumn(bar_width=None),
                TextColumn("{task.percentage:>3.0f}%"), console=console) as prog:
            task = prog.add_task("starting", total=100)
            sm = simulate(client, scenario=a.scenario, n=a.n, label_rate=a.label_rate, seed=a.seed,
                          progress=lambda st, f: prog.update(task, description=st, completed=f * 100))
        console.print(Panel(f"scenario [bold]{a.scenario}[/bold] · sent {sm.sent} · labeled {sm.labeled} · errors {sm.errors}\n"
                            f"served by versions {sm.versions}\nmean predicted P(churn) {sm.mean_probability:.3f}  vs  actual churn rate {sm.actual_churn_rate:.3f}",
                            title="Simulation complete", border_style="green"))
        return 0
    if a.cmd == "demo":
        from .demo import run_registry_demo
        return run_registry_demo(cfg)
    return {"generate": cmd_generate, "validate": cmd_validate, "train": cmd_train}[a.cmd](a, cfg) or 0


if __name__ == "__main__":
    sys.exit(main())
