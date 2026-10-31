"""Self-contained HTML drift report (no external assets; light + dark mode)."""
from __future__ import annotations
from html import escape

COLORS = {"ok": "var(--ok)", "warning": "var(--warn)", "critical": "var(--crit)", "insufficient_data": "var(--mute)"}


def _bars(f: dict) -> str:
    n = len(f["labels"]);
    if not n: return ""
    w, h, pad = 300, 90, 14
    mx = max(max(f["ref_dist"]), max(f["live_dist"]), 1e-6)
    bw = (w - 2 * pad) / n
    out = [f'<svg viewBox="0 0 {w} {h + 22}" class="chart" role="img" aria-label="distribution of {escape(f["name"])}">']
    for i in range(n):
        x = pad + i * bw
        for j, (key, cls) in enumerate((("ref_dist", "ref"), ("live_dist", "live"))):
            v = f[key][i]; bh = v / mx * h
            out.append(f'<rect class="{cls}" x="{x + j * bw * 0.45:.1f}" y="{h - bh:.1f}" width="{bw * 0.42:.1f}" height="{bh:.1f}"><title>{escape(str(f["labels"][i]))}: {cls} {v:.1%}</title></rect>')
        if n <= 6:
            out.append(f'<text x="{x + bw * 0.4:.1f}" y="{h + 14}" text-anchor="middle" class="lbl">{escape(str(f["labels"][i])[:11])}</text>')
    out.append("</svg>")
    return "".join(out)


def render_html(rep: dict, title: str = "Model monitoring report") -> str:
    st = rep["status"]; drift = rep.get("drift"); perf = rep["performance"]
    feats = ([drift["score"]] if drift and drift.get("score") else []) + (drift["features"] if drift else [])
    feats = sorted([f for f in feats if f], key=lambda f: -(f["psi"] or 0))
    rows, charts = [], []
    for f in feats:
        psi = f["psi"] or 0; width = min(psi / 0.5, 1) * 100
        rows.append(f'<tr><td><b>{escape(f["name"])}</b><div class="sub">{escape(f["kind"])}</div></td>'
                    f'<td><div class="bar"><span style="width:{width:.0f}%;background:{COLORS[f["severity"]]}"></span></div>{psi:.3f}</td>'
                    f'<td><span class="pill" style="background:{COLORS[f["severity"]]}">{f["severity"]}</span></td>'
                    f'<td>{"" if f["p_value"] is None else f"{f["p_value"]:.2g}"}</td><td class="sub">{escape(f["note"])}</td></tr>')
        if f["severity"] != "ok":
            charts.append(f'<div class="card"><h4>{escape(f["name"])} <span class="sub">PSI {psi:.2f}</span></h4>{_bars(f)}'
                          f'<div class="legend"><i class="ref"></i> training <i class="live"></i> live</div></div>')
    perf_rows = "".join(f'<tr><td>{escape(c["name"])}</td><td><span class="pill" style="background:{COLORS[c["severity"]]}">{c["severity"]}</span></td><td>{escape(c["detail"])}</td></tr>'
                        for c in perf.get("checks", [])) or '<tr><td colspan=3 class="sub">' + escape(perf.get("summary", "")) + "</td></tr>"
    reasons = "".join(f"<li>{escape(r)}</li>" for r in rep["reasons"])
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title><style>
:root{{--bg:#f6f7f9;--card:#fff;--fg:#14171f;--mute:#6b7280;--line:#e5e7eb;--ok:#16a34a;--warn:#d97706;--crit:#dc2626;--ref:#94a3b8;--live:#6366f1}}
@media(prefers-color-scheme:dark){{:root{{--bg:#0f1117;--card:#181b24;--fg:#e8eaf0;--mute:#9aa3b2;--line:#2a2f3b;--ref:#64748b;--live:#818cf8}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;padding:28px;max-width:1100px;margin:auto}}
h1{{margin:0 0 4px}}.sub{{color:var(--mute);font-size:12px}}.top{{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-bottom:18px}}
.badge{{padding:6px 16px;border-radius:99px;color:#fff;font-weight:700;text-transform:uppercase;letter-spacing:.05em}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:14px 0}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}}.k{{font-size:24px;font-weight:700}}
table{{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}}
th,td{{text-align:left;padding:9px 12px;border-bottom:1px solid var(--line);vertical-align:top}}th{{font-size:12px;color:var(--mute);text-transform:uppercase}}
.pill{{color:#fff;border-radius:99px;padding:2px 10px;font-size:12px}}.bar{{height:8px;background:var(--line);border-radius:4px;width:120px;margin-bottom:3px}}.bar span{{display:block;height:100%;border-radius:4px}}
.charts{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:12px;margin-top:12px}}h2{{margin:26px 0 10px}}h4{{margin:0 0 6px}}
.chart{{width:100%}}.ref{{fill:var(--ref)}}.live{{fill:var(--live)}}.lbl{{font-size:9px;fill:var(--mute)}}.legend{{font-size:12px;color:var(--mute)}}
i.ref,i.live{{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px 0 8px}}i.ref{{background:var(--ref)}}i.live{{background:var(--live)}}
</style></head><body>
<div class="top"><div><h1>{escape(title)}</h1><div class="sub">model v{rep["model_version"]} · {escape(rep["created_at"][:19])} UTC</div></div>
<span class="badge" style="background:{COLORS.get(st, 'var(--mute)')}">{escape(st.replace("_", " "))}</span>
{'<span class="badge" style="background:var(--crit)">retrain recommended</span>' if rep["trigger_retrain"] else ""}</div>
<div class="grid"><div class="card"><div class="sub">live rows analysed</div><div class="k">{rep["n_rows"]:,}</div></div>
<div class="card"><div class="sub">labeled outcomes</div><div class="k">{rep["n_labeled"]:,}</div></div>
<div class="card"><div class="sub">live AUC</div><div class="k">{perf.get("metrics", {}).get("roc_auc", float("nan")):.3f}</div></div>
<div class="card"><div class="sub">calibration gap</div><div class="k">{perf.get("metrics", {}).get("calibration_gap", float("nan")):+.1%}</div></div></div>
<div class="card"><b>Why</b><ul>{reasons}</ul></div>
<h2>Feature drift (vs. training reference)</h2>
<table><tr><th>Feature</th><th>PSI</th><th>Severity</th><th>p-value</th><th>Note</th></tr>{"".join(rows) or '<tr><td colspan=5 class="sub">not enough live rows yet</td></tr>'}</table>
<div class="charts">{"".join(charts)}</div>
<h2>Live performance (delayed labels)</h2>
<table><tr><th>Check</th><th>Severity</th><th>Detail</th></tr>{perf_rows}</table>
</body></html>"""
