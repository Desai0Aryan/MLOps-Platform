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
