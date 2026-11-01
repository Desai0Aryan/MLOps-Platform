"""Rich rendering for monitoring + CLI commands (monitor / worker / events)."""
from __future__ import annotations
import os, time
import httpx
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()
SEV_STYLE = {"ok": "green", "warning": "yellow", "critical": "bold red", "insufficient_data": "dim"}


def _bar(v: float | None, width=14, full=0.5) -> Text:
    if v is None:
        return Text("n/a", style="dim")
    n = int(min(v / full, 1) * width)
    style = "red" if v >= 0.25 else ("yellow" if v >= 0.10 else "green")
    return Text("█" * n + "░" * (width - n), style=style) + Text(f" {v:.2f}")


def render_monitoring(rep: dict) -> Panel:
    parts = []
    d = rep.get("drift")
    if d:
        t = Table(title=f"Data drift · {d['summary'][:90]}", header_style="bold magenta", title_justify="left")
        for c in ("Feature", "PSI", "Severity", "Note"):
            t.add_column(c)
        feats = ([d["score"]] if d.get("score") else []) + sorted(d["features"], key=lambda f: -(f["psi"] or 0))
        for f in feats[:10]:
            t.add_row(f["name"], _bar(f["psi"]), Text(f["severity"], style=SEV_STYLE[f["severity"]]), f["note"][:48])
        parts.append(t)
    else:
        parts.append(Text("Data drift: not enough live rows yet", style="dim"))
    p = rep["performance"]
    pt = Table(title=f"Live performance · {p['summary']}", header_style="bold cyan", title_justify="left")
    for c in ("Check", "Severity", "Detail"):
        pt.add_column(c)
    for c in p.get("checks", []):
        pt.add_row(c["name"], Text(c["severity"], style=SEV_STYLE[c["severity"]]), c["detail"])
    parts.append(pt if p.get("checks") else Text("Live performance: " + p["summary"], style="dim"))
    style = SEV_STYLE.get(rep["status"], "white")
    return Panel(Group(*parts), title=f"Monitoring · model v{rep['model_version']} → [{style}]{rep['status'].upper()}[/]",
                 subtitle="⟳ retraining recommended" if rep["trigger_retrain"] else "", border_style=style.replace("bold ", ""))


def history_table(rows: list[dict]) -> Table:
    t = Table(title="Monitoring history", header_style="bold cyan")
    for c in ("#", "time", "model", "overall", "drift", "performance", "max PSI", "rows", "labeled"):
        t.add_column(c)
    for r in reversed(rows):
        t.add_row(str(r["id"]), r["created_at"][11:19], f"v{r['model_version']}", Text(r["status"], style=SEV_STYLE[r["status"]]),
                  Text(r["drift_status"], style=SEV_STYLE.get(r["drift_status"], "")), Text(r["perf_status"], style=SEV_STYLE.get(r["perf_status"], "")),
                  "-" if r["max_psi"] is None else f"{r['max_psi']:.2f}", str(r["n_rows"]), str(r["n_labeled"]))
    return t


def events_table(rows: list[dict]) -> Table:
    t = Table(title="Lifecycle timeline (what the platform decided, and why)", header_style="bold cyan")
    for c in ("time", "event", "model", "detail"):
        t.add_column(c, overflow="fold")
    icon = {"retrain_started": "🔁", "retrain_completed": "✅", "shadow_started": "👻", "ab_started": "🧪", "challenger_promoted": "🏆",
            "challenger_rejected": "🚫", "retrain_skipped": "⏳", "retrain_rejected_bad_data": "🛑", "retrain_rejected_quality": "🛑"}
    for e in reversed(rows):
        d = {k: v for k, v in e["detail"].items() if k not in ("trigger",)}
        t.add_row(e["created_at"][11:19], f"{icon.get(e['kind'], '•')} {e['kind']}", "-" if e["model_version"] is None else f"v{e['model_version']}",
                  ", ".join(f"{k}={v}" for k, v in d.items())[:90])
    return t


def _client(url):
    h = {"X-Admin-Token": os.environ["ADMIN_TOKEN"]} if os.getenv("ADMIN_TOKEN") else {}
    return httpx.Client(base_url=url, timeout=120, headers=h)


def run_monitor_command(a, cfg) -> int:
    with _client(a.url) as c:
        if a.history:
            console.print(history_table(c.get("/monitoring/history").json())); return 0
        r = c.post("/monitoring/run")
        if r.status_code >= 400:
            console.print(f"[red]{r.status_code}[/red] {r.text}"); return 1
        rep = r.json(); console.print(render_monitoring(rep))
        if a.html:
            from .monitoring import render_html
            open(a.html, "w").write(render_html(rep)); console.print(f"[green]✔[/green] HTML report -> {a.html}")
    return 0


def run_events_command(a, cfg) -> int:
    with _client(a.url) as c:
        console.print(events_table(c.get("/events", params={"limit": a.limit}).json()))
    return 0


def run_worker_command(a, cfg) -> int:
    from .orchestrator import Orchestrator
    from .serving.db import init_db
    url = os.getenv("DATABASE_URL") or cfg["serving"]["database_url"]
    with _client(a.url) as api:
        orch = Orchestrator(cfg, init_db(url)[0], api)
        console.print(Panel(f"api [bold]{a.url}[/bold] · interval {a.interval}s · retraining "
                            f"{'on' if cfg['retraining']['enabled'] else 'off'}", title="⚙ MLOps worker", border_style="cyan"))
        while True:
            try:
                t = orch.tick()
                console.print(f"[dim]{time.strftime('%H:%M:%S')}[/dim] v{t.get('model_version')} "
                              f"[{SEV_STYLE.get(t.get('status', ''), '')}]{t.get('status')}[/] → {t.get('action')} {t.get('reason', '')}")
            except Exception as e:     # a worker must survive transient failures
                console.print(f"[red]tick failed:[/red] {type(e).__name__}: {e}")
            if a.once:
                return 0
            time.sleep(a.interval)
