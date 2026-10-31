"""Scripted story that shows the registry doing its job (fresh database each run).

 v1  trained on a tiny dataset           -> first model, becomes champion
 v2  trained on 8x more data             -> beats v1, PROMOTED
 v3  trained on concept-drifted data     -> worse on golden set, REJECTED by the gate
 ...then an emergency ROLLBACK of v2 -> v1.
"""
from __future__ import annotations
import copy, os, tempfile
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from .data import generate_dataset
from .registry import ModelRegistry
from .training import run_training_pipeline
from .cli_registry import decision_panel, registry_table

console = Console()


def _train(cfg, rows, scenario, seed, label):
    df = generate_dataset(rows, scenario, seed=seed)
    return run_training_pipeline(df, cfg, run_name=label, tags={"scenario": scenario, "label": label})


def run_registry_demo(cfg) -> int:
    cfg = copy.deepcopy(cfg)
    db = os.path.join(tempfile.mkdtemp(), "demo_mlflow.db")
    cfg["mlflow"]["tracking_uri"] = f"sqlite:///{db}"
    cfg["training"]["candidates"] = {k: v for k, v in cfg["training"]["candidates"].items()
                                     if k in ("logistic_regression", "hist_gradient_boosting")}
    reg = ModelRegistry(cfg)
    steps = [("v1 · small dataset (1,500 rows)", 1500, "baseline", 1),
             ("v2 · full dataset (12,000 rows)", 12000, "baseline", 2),
             ("v3 · concept-drifted training data", 6000, "concept_drift", 3)]
    for label, rows, scen, seed in steps:
        console.print(Rule(f"[bold cyan]{label}"))
        with console.status(f"training {label} ..."):
            res = _train(cfg, rows, scen, seed, label)
        mv = reg.register_run(res.run_id)
        console.print(f"trained [bold]{res.best_model}[/bold] (own test AUC {res.metrics['roc_auc']:.4f}) → registered v{mv.version}")
        d = reg.promote_if_better(mv.version)
        console.print(decision_panel(d, mv.version))
    console.print(Rule("[bold magenta]Incident: v2 misbehaves in production"))
    old, new = reg.rollback("simulated production incident")
    console.print(Panel(f"champion rolled back  [red]v{old}[/red] → [green]v{new}[/green]", title="⏪ Rollback", border_style="magenta"))
    console.print(registry_table(reg))
    console.print(f"[dim]demo database: {db}[/dim]")
    return 0


# =====================================================================================
def run_deploy_demo(cfg) -> int:
    """Production story:  healthy -> drift hits -> retrain on fresh data -> SHADOW -> A/B -> auto-promote."""
    from fastapi.testclient import TestClient
    from rich.progress import BarColumn, Progress, TextColumn
    from .cli_deploy import render_report, render_state
    from .serving import create_app
    from .simulate import simulate

    cfg = copy.deepcopy(cfg)
    cfg["mlflow"]["tracking_uri"] = f"sqlite:///{os.path.join(tempfile.mkdtemp(), 'deploy_demo.db')}"
    cfg["training"]["candidates"] = {"logistic_regression": cfg["training"]["candidates"]["logistic_regression"]}
    cfg["serving"]["refresh_seconds"] = 0; cfg["deployment"]["state_cache_seconds"] = 0
    cfg["deployment"]["experiment"]["min_labeled_per_arm"] = 300
    reg = ModelRegistry(cfg)
    stages = ["Train + deploy v1", "Healthy traffic", "Drift hits", "Retrain v2", "Shadow v2", "A/B test v2", "Auto-promote"]
    prog = Progress(TextColumn("[bold cyan]{task.description:<22}"), BarColumn(bar_width=30), TextColumn("{task.completed}/{task.total}"), console=console)
    task = prog.add_task(stages[0], total=len(stages), completed=0)

    def step(i):
        prog.update(task, description=stages[i], completed=i)
        console.print(Rule(f"[bold cyan]Stage {i + 1}/{len(stages)} · {stages[i]}")); console.print(prog.get_renderable())

    def sim(cl, scenario, n, seed, label=1.0):
        return simulate(cl, scenario=scenario, n=n, label_rate=label, seed=seed)

    step(0)
    with console.status("training v1 on baseline data"):
        v1 = reg.register_run(_train(cfg, 8000, "baseline", 1, "v1").run_id)
    reg.promote_version(v1.version, "initial")
    console.print(f"[green]✔[/green] v{v1.version} is champion")

    with TestClient(create_app(cfg, database_url="sqlite://")) as cl:
        step(1); s = sim(cl, "baseline", 600, 10)
        q = cl.get("/stats").json()["live_quality"]
        console.print(f"live accuracy [green]{q['accuracy']:.1%}[/green] · predicted churn {q['mean_predicted']:.1%} vs actual {q['actual_rate']:.1%}  (calibrated)")

        step(2); sim(cl, "concept_drift", 800, 11)
        q = cl.get("/stats", params={"window": 800}).json()["live_quality"]
        console.print(Panel(f"A competitor launches. Same inputs, different behaviour (concept drift).\n"
                            f"live accuracy [red]{q['accuracy']:.1%}[/red] · predicted churn {q['mean_predicted']:.1%} vs actual [red]{q['actual_rate']:.1%}[/red]\n"
                            "[dim]AUC barely moves, but the model is now systematically under-predicting churn.[/dim]", title="⚠ production degradation", border_style="red"))

        step(3)
        with console.status("retraining on fresh, recent data"):
            v2 = reg.register_run(_train(cfg, 8000, "concept_drift", 2, "v2").run_id)
        ok, auc = reg.safety_check(v2.version)
        console.print(f"v{v2.version} registered as challenger · golden-set safety check AUC {auc:.3f} → {'[green]allowed live[/green]' if ok else '[red]blocked[/red]'}")
        console.print("[dim]Note: the OLD golden set can't tell us if v2 is better on today's world. Only live evidence can.[/dim]")

        step(4)
        cl.post("/deployment", json={"mode": "shadow"}); console.print(render_state(cl.get("/deployment").json()))
        s = sim(cl, "concept_drift", 700, 12)
        console.print(f"callers still received only: {s.variants} (v{list(s.versions)[0]})   ← zero user risk")
        sh = cl.get("/experiment/analysis").json(); console.print(render_report(sh))

        step(5)
        cl.post("/deployment", json={"mode": "ab", "traffic_split": 0.3}); console.print(render_state(cl.get("/deployment").json()))
        s = sim(cl, "concept_drift", 1800, 13)
        console.print(f"traffic served: {s.variants}")
        ab = cl.get("/experiment/analysis").json(); console.print(render_report(ab))

        step(6)
        res = cl.post("/experiment/conclude", json={"action": "auto"}).json()
        console.print(Panel(f"applied: [bold green]{res['applied']}[/bold green] · champion is now v{res['champion_version']} · mode {res['mode']}",
                            title="Auto-decision", border_style="green"))
        s = sim(cl, "concept_drift", 600, 14)
        q = cl.get("/stats", params={"window": 600}).json()["live_quality"]
        console.print(f"post-promotion traffic served by {s.versions} · live accuracy [green]{q['accuracy']:.1%}[/green] · predicted {q['mean_predicted']:.1%} vs actual {q['actual_rate']:.1%}")
        prog.update(task, description="Complete", completed=len(stages)); console.print(prog.get_renderable())
    from .cli_registry import registry_table
    console.print(registry_table(reg))
    return 0


# =====================================================================================
def run_lifecycle_demo(cfg, html_path: str | None = None) -> int:
    """Fully autonomous model lifecycle. A human only watches."""
    from fastapi.testclient import TestClient
    from rich.progress import BarColumn, Progress, TextColumn
    from .cli_monitor import events_table, history_table, render_monitoring
    from .orchestrator import Orchestrator
    from .serving import create_app
    from .simulate import simulate

    cfg = copy.deepcopy(cfg)
    cfg["mlflow"]["tracking_uri"] = f"sqlite:///{os.path.join(tempfile.mkdtemp(), 'lifecycle.db')}"
    cfg["training"]["candidates"] = {"logistic_regression": cfg["training"]["candidates"]["logistic_regression"]}
    cfg["serving"]["refresh_seconds"] = 0; cfg["deployment"]["state_cache_seconds"] = 0
    cfg["deployment"]["experiment"]["min_labeled_per_arm"] = 300
    cfg["retraining"].update(cooldown_hours=0, max_rows=2400)
    reg = ModelRegistry(cfg)
    stages = ["Deploy v1", "Healthy day", "Market shifts (no labels yet)", "Labels arrive → auto-retrain",
              "Shadow verdict → A/B", "A/B verdict → promote", "Recovered"]
    prog = Progress(TextColumn("[bold cyan]{task.description:<32}"), BarColumn(bar_width=28), TextColumn("{task.completed}/{task.total}"), console=console)
    task = prog.add_task(stages[0], total=len(stages))

    def step(i):
        prog.update(task, description=stages[i], completed=i)
        console.print(Rule(f"[bold cyan]Stage {i + 1}/{len(stages)} · {stages[i]}")); console.print(prog.get_renderable())

    step(0)
    with console.status("training v1"):
        v1 = reg.register_run(_train(cfg, 8000, "baseline", 1, "v1").run_id)
    reg.promote_version(v1.version, "initial"); console.print(f"[green]✔[/green] v{v1.version} is champion")

    with TestClient(create_app(cfg, database_url="sqlite://")) as cl:
        orch = Orchestrator(cfg, cl.app.state.engine, cl, reg)

        def tick(show_report=True):
            t = orch.tick()
            if show_report:
                console.print(render_monitoring(cl.app.state.monitor.latest()))
            console.print(Panel(f"[bold]{t.get('action')}[/bold]  {t.get('reason', '')}"
                                + (f"\n{t['advance']}" if "advance" in t else ""), title="🤖 orchestrator decision", border_style="cyan"))
            return t

        step(1); simulate(cl, scenario="baseline", n=800, label_rate=1.0, seed=10); tick()
        step(2)
        console.print("[dim]A new customer mix + inflation + a competitor. Outcomes take weeks to arrive, so no labels yet.[/dim]")
        simulate(cl, scenario="market_shift", n=1200, label_rate=0.0, seed=11); tick()
        step(3); simulate(cl, scenario="market_shift", n=2200, label_rate=1.0, seed=12)
        with console.status("orchestrator retraining on production data..."):
            t = tick(show_report=False)
        step(4); simulate(cl, scenario="market_shift", n=900, label_rate=1.0, seed=13)
        console.print(Panel(str(cl.get("/experiment/analysis").json()["recommendation"]), title="shadow analysis")); tick(show_report=False)
        step(5); simulate(cl, scenario="market_shift", n=2500, label_rate=1.0, seed=14); tick(show_report=False)
        step(6); simulate(cl, scenario="market_shift", n=1200, label_rate=1.0, seed=15); tick()
        prog.update(task, description="Complete", completed=len(stages)); console.print(prog.get_renderable())
        console.print(history_table(cl.app.state.monitor.history()))
        console.print(events_table(cl.get("/events").json()))
        if html_path:
            from .monitoring import render_html
            open(html_path, "w").write(render_html(cl.app.state.monitor.latest())); console.print(f"[green]✔[/green] wrote {html_path}")
    from .cli_registry import registry_table
    console.print(registry_table(reg))
    return 0
