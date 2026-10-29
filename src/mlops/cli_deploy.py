"""CLI for rollouts + rich rendering of experiment reports."""
from __future__ import annotations
import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
VERDICT_STYLE = {"promote": "bold green", "challenger_better": "bold green", "continue": "yellow", "no_difference": "cyan",
                 "stop": "bold red", "challenger_worse": "bold red", "unhealthy": "bold red", "invalid_srm": "bold red", "hold": "magenta"}
ICON = {"pass": "[green]✔[/green]", "fail": "[red]✘[/red]", "info": "[blue]ℹ[/blue]", "wait": "[yellow]…[/yellow]"}


def render_report(d: dict) -> Panel:
    t = Table.grid(padding=(0, 1))
    for c in d["checks"]:
        t.add_row(ICON.get(c["status"], "?"), f"[bold]{c['name']}[/bold]", c["detail"])
    prog = d["stats"].get("progress")
    bar = ""
    if prog is not None:
        filled = int(prog * 30)
        bar = f"\nEvidence collected  [cyan]{'█' * filled}{'░' * (30 - filled)}[/cyan] {prog:.0%}"
    style = VERDICT_STYLE.get(d["verdict"], "white")
    return Panel(t, title=f"{d['kind'].upper()} experiment → [{style}]{d['verdict'].upper()}[/]",
                 subtitle=d["recommendation"] + bar, border_style=style.replace("bold ", ""))


def render_state(s: dict) -> Panel:
    split = f" · challenger share {s['traffic_split']:.0%}" if s["mode"] == "ab" else ""
    return Panel(f"mode [bold]{s['mode']}[/bold]{split}\nchampion v{s['champion_version']} · challenger "
                 f"{('v' + str(s['challenger_version'])) if s['challenger_version'] else '-'}",
                 title="Rollout state", border_style="cyan")


def run_deploy_command(a, cfg) -> int:
    import os
    headers = {"X-Admin-Token": os.environ["ADMIN_TOKEN"]} if os.getenv("ADMIN_TOKEN") else {}
    with httpx.Client(base_url=a.url, timeout=60, headers=headers) as c:
        def show(r):
            if r.status_code >= 400:
                console.print(f"[red]{r.status_code}[/red] {r.json().get('detail', r.text)}"); return 1
            return 0
        if a.dcmd == "status":
            r = c.get("/deployment"); console.print(render_state(r.json())); return show(r)
        if a.dcmd in ("shadow", "ab", "stop"):
            body = {"mode": {"stop": "champion_only"}.get(a.dcmd, a.dcmd)}
            if a.dcmd == "ab": body["traffic_split"] = a.split
            r = c.post("/deployment", json=body)
            if r.status_code < 400: console.print(render_state(r.json()))
            return show(r)
        if a.dcmd == "analyze":
            r = c.get("/experiment/analysis")
            if r.status_code < 400: console.print(render_report(r.json()))
            return show(r)
        if a.dcmd == "conclude":
            r = c.post("/experiment/conclude", json={"action": a.action})
            if r.status_code < 400: console.print(Panel(str(r.json()), title="Conclusion applied", border_style="green"))
            return show(r)
    return 0
