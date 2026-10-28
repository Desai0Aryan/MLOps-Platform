"""Rich rendering + commands for the model registry."""
from __future__ import annotations
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .registry import ModelRegistry, RegistryError, PromotionDecision

console = Console()
STATUS_STYLE = {"champion": "bold green", "candidate": "yellow", "archived": "dim",
                "rejected": "red", "rolled_back": "magenta"}


def decision_panel(d: PromotionDecision, version) -> Panel:
    t = Table.grid(padding=(0, 2))
    for r in d.rules:
        t.add_row("[green]✔[/green]" if r.passed else "[red]✘[/red]", f"[bold]{r.name}[/bold]", r.detail)
    head = f"challenger v{version}  AUC {d.challenger['roc_auc']:.4f}"
    if d.champion:
        head += f"   vs champion AUC {d.champion['roc_auc']:.4f}"
    verdict = "[bold green]PROMOTED[/bold green]" if d.promote else "[bold red]REJECTED[/bold red]"
    return Panel(t, title=f"{head}   →   {verdict}", subtitle=d.reason,
                 border_style="green" if d.promote else "red")


def registry_table(reg: ModelRegistry) -> Table:
    t = Table(title=f"Model registry · {reg.name}", header_style="bold cyan")
    for c in ("Ver", "Alias", "Status", "Algo", "AUC", "Note"):
        t.add_column(c, overflow="fold")
    for v in reg.list_versions():
        tg = v.tags; st = tg.get("status", "?")
        note = tg.get("rejection_reason") or tg.get("rollback_reason") or \
            (f"replaced v{tg['replaced_version']}" if "replaced_version" in tg else "")
        t.add_row(f"v{v.version}", ", ".join(v.aliases) or "-", f"[{STATUS_STYLE.get(st, '')}]{st}[/]",
                  tg.get("algorithm", "?").replace("logistic_regression", "logreg").replace("hist_gradient_boosting", "hgb").replace("random_forest", "rf"),
                  tg.get("test_roc_auc", "?"), note)
    return t


def run_registry_command(a, cfg) -> int:
    reg = ModelRegistry(cfg)
    try:
        if a.rcmd == "list":
            console.print(registry_table(reg))
        elif a.rcmd == "register":
            run_id = a.run_id or _latest_successful_run(cfg)
            mv = reg.register_run(run_id)
            console.print(f"[green]✔[/green] registered run {run_id[:8]} as [bold]v{mv.version}[/bold] (alias: challenger)")
        elif a.rcmd == "promote":
            v = a.version or (reg.challenger().version if reg.challenger() else None)
            if v is None:
                raise RegistryError("no challenger; pass a version")
            d = reg.promote_if_better(v, force=a.force)
            console.print(decision_panel(d, v)); console.print(registry_table(reg))
            return 0 if d.promote or a.force else 4
        elif a.rcmd == "rollback":
            old, new = reg.rollback(a.reason)
            console.print(Panel(f"champion rolled back  [red]v{old}[/red] → [green]v{new}[/green]\nreason: {a.reason}",
                                title="⏪ Rollback", border_style="magenta")); console.print(registry_table(reg))
    except RegistryError as e:
        console.print(f"[red]registry error:[/red] {e}"); return 1
    return 0


def _latest_successful_run(cfg) -> str:
    import mlflow
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    exp = mlflow.get_experiment_by_name(cfg["mlflow"]["experiment_name"])
    runs = mlflow.search_runs([exp.experiment_id], filter_string="tags.pipeline_status = 'success' and tags.stage = 'training'",
                              order_by=["start_time DESC"], max_results=1) if exp else []
    if len(runs) == 0:
        raise RegistryError("no successful training runs found; run `train` first")
    return runs.iloc[0]["run_id"]
