#!/usr/bin/env python3
"""Assemble self-contained report.html from report_data.json (inline SVG charts, no runtime fetch)."""
import csv, json, os, html

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = json.load(open(os.path.join(OUT, "report_data.json")))

def svg(name):
    p = os.path.join(OUT, "assets", name)
    s = open(p).read()
    # inline CSS variables fallback for standalone SVG rendering in <img>-less inline use
    return s

def esc(x): return html.escape(str(x))

def fmt_pct(x, signed=True):
    if x is None: return "—"
    return f"{x*100:+.2f}%" if signed else f"{x*100:.2f}%"

# ---------- chart builders (inline SVG strings) ----------

def ch_equity():
    series=D["equity_curves"]
    panels=[("baselines",["LMart_sectors_1","LRank_sectors_1"],"Baseline window (2024-11-11 to 2025-12-30)"),
            ("test",["LMart_sectors_7","LRank_sectors_14"],"Illustrative test-window runs (2025-01-24 to 2025-12-30)")]
    cols={"LMart_sectors_1":"#8a6d3b","LRank_sectors_1":"#7a4a8a","LMart_sectors_7":"#c98a3a","LRank_sectors_14":"#4c7a4c"}
    names={"LMart_sectors_1":"LMart_1 (tree, two-tail)","LRank_sectors_1":"LRank_1 (neural, two-tail)",
           "LMart_sectors_7":"LMart_7 (tree, two-tail)","LRank_sectors_14":"LRank_14 (neural, long-only)"}
    W,H,P=960,300,54
    out=[]
    for _,keys,title in panels:
        # common window = intersection of series dates; rebased to 0%
        n=min(len(series[k]["equity"]) for k in keys)
        rets={}
        for k in keys:
            e=series[k]["equity"][:n]
            rets[k]=[(v/e[0]-1)*100 for v in e]
        spy=[(v/series[keys[0]]["spy"][0]-1)*100 for v in series[keys[0]]["spy"][:n]]
        allv=[v for k in keys for v in rets[k]]+spy
        ymin,ymax=min(allv),max(allv)
        def px(i,n=n): return P+(W-2*P)*i/(n-1)
        def py(v): return H-P-(H-2*P)*(v-ymin)/(ymax-ymin)
        parts=[f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Return since common window start, percent scale, with drawdown" class="chart">']
        parts.append(f'<text x="{P}" y="20" font-size="13" font-weight="600">{title}</text>')
        parts.append(f'<line x1="{P}" y1="{py(0)}" x2="{W-P}" y2="{py(0)}" stroke="#999" stroke-dasharray="3 3"/>')
        # drawdown shading for each strategy (light red under curve)
        for k in keys:
            r=rets[k]; peak=-1e9; dd=[]
            for v in r: peak=max(peak,v); dd.append(v-peak)
            dpts="M"+" L".join(f"{px(i):.1f},{py(r[i]):.1f}" for i in range(n))
            parts.append(f'<path d="{dpts}" fill="none" stroke="{cols[k]}" stroke-width="2"/>')
        d="M"+" L".join(f"{px(i):.1f},{py(v):.1f}" for i,v in enumerate(spy))
        parts.append(f'<path d="{d}" fill="none" stroke="#888" stroke-width="2.4" stroke-dasharray="7 4"/>')
        parts.append(f'<text x="{px(n-1)-26}" y="{py(spy[-1])-8}" font-size="12" fill="#666">SPY</text>')
        for k in keys:
            parts.append(f'<rect x="{P}" y="{H-P+18}" width="16" height="4" fill="{cols[k]}"/><text x="{P+22}" y="{H-P+24}" font-size="12">{names[k]}</text>')
            P_ = P+22+len(names[k])*7.2
        parts.append(f'<text x="{W-P}" y="{H-P+24}" font-size="11" fill="#888" text-anchor="end">grey dashed = SPY (external benchmark, not risk-matched)</text>')
        t0=series[keys[0]]["t"][0][:7]; t1=series[keys[0]]["t"][n-1][:7]
        parts.append(f'<text x="{P}" y="{H-6}" font-size="11" fill="#777">{t0}</text>')
        parts.append(f'<text x="{W-P}" y="{H-6}" font-size="11" fill="#777" text-anchor="end">{t1}</text>')
        # y labels
        parts.append(f'<text x="{P-6}" y="{py(ymax)+4}" font-size="10.5" text-anchor="end" fill="#777">{ymax:.0f}%</text>')
        parts.append(f'<text x="{P-6}" y="{py(ymin)+4}" font-size="10.5" text-anchor="end" fill="#777">{ymin:.0f}%</text>')
        parts.append('</svg>')
        out.append("".join(parts))
    table='<table class="fallback"><caption>Rebased returns over each panel\'s common window (realized closed lots only; terminal open lots excluded)</caption><tr><th>Run</th><th>Window</th><th>Total return</th><th>vs SPY</th><th>Max drawdown</th></tr>'
    for k in ["LMart_sectors_1","LRank_sectors_1","LMart_sectors_7","LRank_sectors_14"]:
        r=D.get("baselines",{}).get(k) or D["illustrative_runs"][k]
        table+=f'<tr><td>{k} (illustrative)</td><td>{r.get("window","2025-01-24 to 2025-12-30")}</td><td>{fmt_pct(r["total_return"])}</td><td>{fmt_pct(r.get("excess_vs_spy"))}</td><td>{fmt_pct(r.get("max_dd"))}</td></tr>'
    table+='</table><p class="src">SPY is an external benchmark — not risk-, beta-, exposure-, or cash-matched. Realized P&amp;L includes closed lots only.</p>'
    return "".join(out)+f'<details class="drawer"><summary>Data table</summary>{table}</details>'

def ch_timeline():
    items=[
      ("2026-08-22","Technical","Reproducible iteration packages + MLflow logging (runs 1–6)"),
      ("2026-08-22","Concept","Symmetric long/short ranking metrics beyond NDCG"),
      ("2026-08-22","Concept","Smooth percentile relevance + elite label weighting"),
      ("2026-08-22 → 23","Concept","Explicit tail-mode experiments (two-tail vs long-only)"),
      ("2026-08-22 → 23","Technical","Modern activation / normalization / optimizer controls"),
      ("2026-08-23","Concept","LightGBM vs XGBoost as a model-family knob (mixed-backend study)"),
      ("2026-08-23","Method","Purged expanding folds, 2 seeds, stability penalty, untouched test (all three studies)"),
      ("not evidenced","Diagnostic","Strategy 2 ATR trailing exits — designed, no completed package"),
    ]
    rows=""
    for i,(when,kind,label) in enumerate(items):
        color="#4c7a4c" if kind=="Concept" else ("#8a6d3b" if kind=="Technical" else "#3a5a8a")
        y=34+i*44
        rows+=f'<circle cx="150" cy="{y}" r="6" fill="{color}"/><text x="140" y="{y+4}" text-anchor="end" font-size="12" fill="#777">{esc(when)}</text><text x="168" y="{y+4}" font-size="13" fill="currentColor">{esc(label)}</text>'
    return f'''<svg viewBox="0 0 900 {34+len(items)*44}" role="img" aria-label="Experiment timeline distinguishing technical fixes from conceptual experiments" class="chart">
      <line x1="150" y1="24" x2="150" y2="{20+len(items)*44}" stroke="#bbb"/>
      <g font-family="inherit"><rect x="600" y="8" width="12" height="12" rx="3" fill="#4c7a4c"/><text x="618" y="18" font-size="12" fill="currentColor">Conceptual experiment</text>
      <rect x="600" y="28" width="12" height="12" rx="3" fill="#8a6d3b"/><text x="618" y="38" font-size="12" fill="currentColor">Technical fix / capability</text>
      <rect x="600" y="48" width="12" height="12" rx="3" fill="#3a5a8a"/><text x="618" y="58" font-size="12" fill="currentColor">Post-selection diagnostic</text></g>
      {rows}</svg>'''

def ch_fold_box():
    box=D["fold_box"]
    fams=["Neural","Legacy LightGBM","Mixed tree"]
    W,H,P=960,400,60
    ymin=-4.0; ymax=4.0
    def py(v): return H-P-(H-2*P)*(v-ymin)/(ymax-ymin)
    bw=70; group_w=(W-2*P)/3
    parts=[f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Fold-level Sharpe distributions by family and tail mode" class="chart">',
           f'<line x1="{P}" y1="{py(0)}" x2="{W-P}" y2="{py(0)}" stroke="#999" stroke-dasharray="3 3"/><text x="{W-P-70}" y="{py(0)-6}" font-size="11" fill="#777">Sharpe 0</text>']
    for gi,fam in enumerate(fams):
        gx=P+gi*group_w
        for ti,tail in enumerate(["long_only","two_tail"]):
            b=[x for x in box if x["family"]==fam and x["tail"]==tail][0]
            cx=gx+group_w/2+(bw+26)*(ti-0.5)
            col="#4c7a4c" if tail=="long_only" else "#a05252"
            q1,q3,med=b["q1"],b["q3"],b["med"]
            parts.append(f'<line x1="{cx}" y1="{py(b["max"])}" x2="{cx}" y2="{py(b["min"])}" stroke="{col}"/>')
            parts.append(f'<line x1="{cx-bw/3}" y1="{py(b["max"])}" x2="{cx+bw/3}" y2="{py(b["max"])}" stroke="{col}"/>')
            parts.append(f'<line x1="{cx-bw/3}" y1="{py(b["min"])}" x2="{cx+bw/3}" y2="{py(b["min"])}" stroke="{col}"/>')
            parts.append(f'<rect x="{cx-bw/2}" y="{py(q3)}" width="{bw}" height="{py(q1)-py(q3)}" fill="{col}" opacity=".25" stroke="{col}"/>')
            parts.append(f'<line x1="{cx-bw/2}" y1="{py(med)}" x2="{cx+bw/2}" y2="{py(med)}" stroke="{col}" stroke-width="2.4"/>')
            parts.append(f'<text x="{cx}" y="{py(b["max"])-6}" font-size="10.5" text-anchor="middle" fill="#777">n={b["n"]}</text>')
        parts.append(f'<text x="{gx+group_w/2}" y="{H-P+22}" font-size="13" text-anchor="middle" font-weight="600" fill="currentColor">{fam}</text>')
        parts.append(f'<text x="{gx+group_w/2-bw/2-13}" y="{H-P+40}" font-size="11" text-anchor="middle" fill="#4c7a4c">long-only</text>')
        parts.append(f'<text x="{gx+group_w/2+bw/2+13}" y="{H-P+40}" font-size="11" text-anchor="middle" fill="#a05252">two-tail</text>')
    parts.append('</svg>')
    table='<table class="fallback"><caption>Fold-level walk-forward Sharpe quartiles</caption><tr><th>Family</th><th>Tail</th><th>n</th><th>Min</th><th>Q1</th><th>Median</th><th>Q3</th><th>Max</th><th>Mean</th></tr>'
    for b in box:
        table+=f'<tr><td>{b["family"]}</td><td>{b["tail"]}</td><td>{b["n"]}</td><td>{b["min"]}</td><td>{b["q1"]}</td><td>{b["med"]}</td><td>{b["q3"]}</td><td>{b["max"]}</td><td>{b["mean"]}</td></tr>'
    table+='</table>'
    return "".join(parts)+f'<details class="drawer"><summary>Data table</summary>{table}</details>'

def _cluster_rows():
    studies = [
        ("Neural LambdaRank", "LRank_sectors_both_broad_smooth_1", "#7a4a8a"),
        ("Legacy LightGBM", "LMart_sectors_both_architecture_1", "#8a6d3b"),
        ("Mixed tree", "TreeRank_sectors_both_architecture_1", "#3a5a8a"),
    ]
    rows=[]
    for label, folder, color in studies:
        base=os.path.join(ROOT,"artifacts","optuna",folder)
        selected=json.load(open(os.path.join(base,"optuna_selection.json")))["selected_cluster"]
        with open(os.path.join(base,"optuna_clusters.csv"),newline="") as handle:
            for raw in csv.DictReader(handle):
                keys=("tail_mode","activation","normalization","optimizer") if folder.startswith("LRank_") else (("tail_mode","max_depth") if folder.startswith("LMart_") else ("tree_backend","tail_mode","max_depth"))
                chosen=all(str(raw.get(k)).lower()==str(selected.get(k)).lower() for k in keys)
                definition=" · ".join(f'{k.replace("_"," ")}={raw.get(k)}' for k in keys)
                rows.append({"study":label,"folder":folder,"color":color,"definition":definition,"selected":chosen,
                             "tail":raw["tail_mode"],"trials":int(float(raw["trials"])),
                             "mean_sharpe":float(raw["mean_sharpe"]),"mean_fold_std":float(raw["mean_fold_std"]),
                             "across_trial_std":float(raw["across_trial_std"]),"stability_score":float(raw["stability_score"])})
    return rows

def ch_trial_map():
    pts=_cluster_rows()
    W,H,P=960,500,64
    xmax=max(p["mean_fold_std"] for p in pts)*1.08
    ymin=min(-1.7,min(p["mean_sharpe"] for p in pts)-.1); ymax=max(1.3,max(p["mean_sharpe"] for p in pts)+.1)
    def px(v): return P+(W-2*P)*v/xmax
    def py(v): return H-P-(H-2*P)*(v-ymin)/(ymax-ymin)
    parts=[f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Actual hyperparameter clusters plotted by raw mean Sharpe and mean fold dispersion; bubble area represents completed trials" class="chart">',
           f'<line x1="{P}" y1="{py(0)}" x2="{W-P}" y2="{py(0)}" stroke="#999" stroke-dasharray="3 3"/>']
    for p in pts:
        cx,cy=px(p["mean_fold_std"]),py(p["mean_sharpe"]); radius=4+2.0*(p["trials"]**.5)
        fill=p["color"] if p["tail"]=="long_only" else "none"
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}" fill="{fill}" fill-opacity=".55" stroke="{p["color"]}" stroke-width="1.8"/>')
        if p["selected"]:
            parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius+5:.1f}" fill="none" stroke="#111" stroke-width="2.5"/>')
    parts.append(f'<text x="{W/2}" y="{H-16}" font-size="12" text-anchor="middle" fill="#555">mean fold dispersion within cluster → lower is steadier</text>')
    parts.append(f'<text x="18" y="{H/2}" font-size="12" fill="#555" transform="rotate(-90 20 {H/2})">cluster mean walk-forward Sharpe →</text>')
    lx,ly=W-270,28
    parts.append(f'<rect x="{lx-16}" y="{ly-20}" width="260" height="116" fill="#fff" opacity=".82" rx="6"/>')
    for i,(label,color) in enumerate((("Neural LambdaRank","#7a4a8a"),("Legacy LightGBM","#8a6d3b"),("Mixed tree","#3a5a8a"))):
        y=ly+i*22; parts.append(f'<circle cx="{lx}" cy="{y}" r="6" fill="{color}" fill-opacity=".55" stroke="{color}"/><text x="{lx+16}" y="{y+4}" font-size="11.5">{label}</text>')
    parts.append(f'<circle cx="{lx}" cy="{ly+68}" r="7" fill="none" stroke="#555"/><text x="{lx+16}" y="{ly+72}" font-size="11.5">hollow = two-tail</text>')
    parts.append(f'<circle cx="{lx+145}" cy="{ly+68}" r="9" fill="none" stroke="#111" stroke-width="2.5"/><text x="{lx+161}" y="{ly+72}" font-size="11.5">ring = selected cluster</text>')
    parts.append('</svg>')
    selected_table='<table class="fallback"><caption>Selected clusters shown by the black rings</caption><tr><th>Study</th><th>Cluster definition</th><th>Trials</th><th>Mean Sharpe</th><th>Mean fold std</th><th>Across-trial std</th><th>Stability score</th></tr>'
    for p in (row for row in pts if row["selected"]):
        selected_table+=f'<tr><td>{esc(p["study"])}</td><td>{esc(p["definition"])}</td><td>{p["trials"]}</td><td>{p["mean_sharpe"]:.3f}</td><td>{p["mean_fold_std"]:.3f}</td><td>{p["across_trial_std"]:.3f}</td><td>{p["stability_score"]:.3f}</td></tr>'
    selected_table+='</table>'
    table='<table class="fallback"><caption>Hyperparameter clusters; one row is one actual cluster, not one trial</caption><tr><th>Study</th><th>Cluster definition</th><th>Tail</th><th>Trials</th><th>Mean Sharpe</th><th>Mean fold std</th><th>Across-trial std</th><th>Stability score</th><th>Selected</th></tr>'
    for p in sorted(pts,key=lambda q:(q["study"],-q["stability_score"])):
        table+=f'<tr><td>{esc(p["study"])}</td><td>{esc(p["definition"])}</td><td>{p["tail"]}</td><td>{p["trials"]}</td><td>{p["mean_sharpe"]:.3f}</td><td>{p["mean_fold_std"]:.3f}</td><td>{p["across_trial_std"]:.3f}</td><td>{p["stability_score"]:.3f}</td><td>{"yes" if p["selected"] else ""}</td></tr>'
    table+='</table>'
    return "".join(parts)+selected_table+f'<details class="drawer"><summary>All cluster definitions and values</summary>{table}</details>'

def ch_dumbbell():
    rows=D["stability_raw"]
    best_by_study={p["study"]:p for p in D["trial_scatter"] if p["best_single"]}
    study_ids={"Neural LambdaRank":"LRank_sectors_both_broad_smooth_1","Legacy LightGBM":"LMart_sectors_both_architecture_1","Mixed tree":"TreeRank_sectors_both_architecture_1"}
    for r in rows:
        best=best_by_study[study_ids[r["study"]]]
        r["best_mean_sharpe"],r["best_fold_std"]=best["sharpe"],best["fold_std"]
    W,H,P=960,330,190
    vals=[v for r in rows for v in (r["cluster_mean_sharpe"],r["stable_mean_sharpe"],r["best_mean_sharpe"])]
    vmin,vmax=min(vals)-0.15,max(vals)+0.15
    def px(v): return P+(W-P-40)*(v-vmin)/(vmax-vmin)
    parts=[f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Raw mean walk-forward Sharpe for the selected cluster, stable trial, and best single trial; all points use the same unit" class="chart">']
    for i,r in enumerate(rows):
        y=62+i*78
        parts.append(f'<text x="{P-12}" y="{y+4}" text-anchor="end" font-size="13" font-weight="600">{esc(r["study"])}</text>')
        lo=min(r["cluster_mean_sharpe"],r["stable_mean_sharpe"],r["best_mean_sharpe"]); hi=max(r["cluster_mean_sharpe"],r["stable_mean_sharpe"],r["best_mean_sharpe"])
        parts.append(f'<line x1="{px(lo)}" y1="{y}" x2="{px(hi)}" y2="{y}" stroke="#bbb" stroke-width="2"/>')
        parts.append(f'<circle cx="{px(r["cluster_mean_sharpe"])}" cy="{y}" r="6" fill="#999"/><text x="{px(r["cluster_mean_sharpe"])}" y="{y+22}" font-size="10.5" text-anchor="middle">cluster {r["cluster_mean_sharpe"]:.3f}</text>')
        sx=px(r["stable_mean_sharpe"])
        parts.append(f'<circle cx="{sx}" cy="{y}" r="8" fill="#fff" stroke="#4c7a4c" stroke-width="3"/>')
        bx=px(r["best_mean_sharpe"]); star=f'M{bx},{y-9} l2.4 5.4 5.9.3 -4.6 3.9 1.5 5.8 -5.2-3.2 -5.2 3.2 1.5-5.8 -4.6-3.9 5.9-.3 z'
        parts.append(f'<path d="{star}" fill="#c98a3a" stroke="#8a5c20" stroke-width="1.3"/>')
        if abs(r["stable_mean_sharpe"]-r["best_mean_sharpe"]) < .001:
            parts.append(f'<text x="{sx}" y="{y-16}" font-size="10.5" text-anchor="middle" font-weight="700">stable = best {r["stable_mean_sharpe"]:.3f}</text>')
        else:
            parts.append(f'<text x="{sx}" y="{y-14}" font-size="10.5" text-anchor="middle" fill="#355d35">stable {r["stable_mean_sharpe"]:.3f}</text><text x="{bx}" y="{y-14}" font-size="10.5" text-anchor="middle" fill="#8a5c20">best {r["best_mean_sharpe"]:.3f}</text>')
    parts.append(f'<text x="{P}" y="{H-22}" font-size="11.5" fill="#555">Horizontal axis: raw mean walk-forward Sharpe only. Higher is better. Objectives are deliberately excluded from this axis.</text>')
    parts.append('</svg>')
    table='<table class="fallback"><caption>Stability adjustment shown separately from raw Sharpe: objective = mean Sharpe − 0.25 × fold std</caption><tr><th>Study</th><th>Candidate</th><th>Raw mean Sharpe</th><th>Fold std</th><th>Penalty (0.25×std)</th><th>Objective</th></tr>'
    for r in rows:
        if abs(r["stable_mean_sharpe"]-r["best_mean_sharpe"]) < .001 and abs(r["penalized_objective"]-r["best_single_objective"]) < .001:
            table+=f'<tr><td>{esc(r["study"])}</td><td>Stable = best single trial {r["stable_trial"]}</td><td>{r["stable_mean_sharpe"]:.3f}</td><td>{r["stable_fold_std"]:.3f}</td><td>{r["stable_mean_sharpe"]-r["penalized_objective"]:.3f}</td><td>{r["penalized_objective"]:.3f}</td></tr>'
        else:
            table+=f'<tr><td>{esc(r["study"])}</td><td>Stable trial {r["stable_trial"]}</td><td>{r["stable_mean_sharpe"]:.3f}</td><td>{r["stable_fold_std"]:.3f}</td><td>{r["stable_mean_sharpe"]-r["penalized_objective"]:.3f}</td><td>{r["penalized_objective"]:.3f}</td></tr>'
            table+=f'<tr><td>{esc(r["study"])}</td><td>Best single trial</td><td>{r["best_mean_sharpe"]:.3f}</td><td>{r["best_fold_std"]:.3f}</td><td>{r["best_mean_sharpe"]-r["best_single_objective"]:.3f}</td><td>{r["best_single_objective"]:.3f}</td></tr>'
    table+='</table>'
    return "".join(parts)+table+'<p class="src"><strong>mean_sharpe</strong>: raw mean across one trial\'s fold×seed Sharpe observations. <strong>fold_std</strong>: their dispersion. <strong>penalty</strong>: 0.25 × fold_std. <strong>objective</strong>: mean_sharpe − penalty; it is a selection score, not Sharpe.</p>'

def ch_ndcg_sharpe():
    pts=D["packaged_runs_ndcg_sharpe"]
    W,H,P=960,480,58
    xs=[p["ndcg"] for p in pts]; ys=[p["sharpe"] for p in pts]
    xmin,xmax=min(xs)-.01,max(xs)+.01; ymin,ymax=min(ys)-.2,max(ys)+.2
    def px(v): return P+(W-2*P)*(v-xmin)/(xmax-xmin)
    def py(v): return H-P-(H-2*P)*(v-ymin)/(ymax-ymin)
    parts=[f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Test NDCG versus realized Sharpe for fifteen packaged runs sharing the 2025 test window" class="chart">']
    spy=D["illustrative_runs"]["LRank_sectors_14"]["spy_sharpe"]
    parts.append(f'<line x1="{P}" y1="{py(spy)}" x2="{W-P}" y2="{py(spy)}" stroke="#888" stroke-dasharray="7 4"/><text x="{W-P-120}" y="{py(spy)-6}" font-size="11.5" fill="#666">SPY Sharpe {spy:.3f} (no NDCG coordinate)</text>')
    parts.append(f'<line x1="{P}" y1="{py(0)}" x2="{W-P}" y2="{py(0)}" stroke="#bbb" stroke-dasharray="3 3"/>')
    for p in pts:
        cx,cy=px(p["ndcg"]),py(p["sharpe"])
        col={"lambdamart":"#c98a3a","lambdarank":"#7a4a8a"}[p["family"]]
        ret=p["ret"]; size=7+min(abs(ret),0.06)*500
        op=.9 if p["tail"]=="long_only" else .45
        if p["family"]=="lambdamart":
            parts.append(f'<rect x="{cx-5}" y="{cy-5}" width="10" height="10" fill="{col}" opacity="{op}"/>')
        else:
            parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{size**.5:.1f}" fill="{col}" opacity="{op}"/>')
    parts.append(f'<text x="{W/2}" y="{H-14}" font-size="12" text-anchor="middle" fill="#555">packaged test NDCG@2 (long-only runs: long NDCG; two-tail runs: macro-tail NDCG) →</text>')
    parts.append(f'<text x="18" y="{H/2}" font-size="12" fill="#555" transform="rotate(-90 20 {H/2})">realized Sharpe →</text>')
    parts.append(f'<rect x="{W-330}" y="20" width="310" height="86" fill="#fff" opacity=".78" rx="6"/>')
    parts.append(f'<rect x="{W-315}" y="32" width="12" height="12" fill="#c98a3a"/><text x="{W-296}" y="42" font-size="12" fill="currentColor">LightGBM (square)</text>')
    parts.append(f'<circle cx="{W-309}" cy="62" r="7" fill="#7a4a8a"/><text x="{W-296}" y="66" font-size="12" fill="currentColor">Neural (circle; area ∝ |return|)</text>')
    parts.append(f'<text x="{W-315}" y="92" font-size="11" fill="#777">opaque = long-only · faded = two-tail</text>')
    parts.append('</svg>')
    table='<table class="fallback"><caption>Packaged runs: NDCG vs realized outcome (identical 2025 test window and exit policy)</caption><tr><th>Run</th><th>Family</th><th>Tail</th><th>NDCG@2</th><th>Realized Sharpe</th><th>Return</th></tr>'
    for p in sorted(pts,key=lambda q:q["sharpe"]):
        table+=f'<tr><td>{p["pkg"]}</td><td>{"LightGBM" if p["family"]=="lambdamart" else "Neural"}</td><td>{p["tail"]}</td><td>{p["ndcg"]:.4f}</td><td>{p["sharpe"]:+.3f}</td><td>{fmt_pct(p["ret"])}</td></tr>'
    table+='</table>'
    return "".join(parts)+f'<details class="drawer"><summary>Data table</summary>{table}</details>'

def ch_long_short():
    runs=D["long_short_two_tail"]
    W,H,P=960,360,150
    vals=[r["long"] for r in runs]+[abs(r["short"]) for r in runs]
    vmax=max(vals)*1.25
    bw=26
    parts=[f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Long versus short closed-lot expected return per comparable two-tail run" class="chart">']
    gw=(W-P-40)/len(runs)
    for i,r in enumerate(runs):
        gx=P+i*gw
        hl=(r["long"]/vmax)*(H-2*P); sl=(abs(r["short"])/vmax)*(H-2*P)
        parts.append(f'<rect x="{gx}" y="{H-P-hl}" width="{bw}" height="{hl}" fill="#4c7a4c" opacity=".85"/>')
        parts.append(f'<rect x="{gx+bw+8}" y="{H-P-sl}" width="{bw}" height="{sl}" fill="#a05252" opacity=".85"/>')
        parts.append(f'<text x="{gx+bw/2}" y="{H-P-hl-6}" font-size="10.5" text-anchor="middle" fill="#4c7a4c">{r["long"]*100:+.2f}%</text>')
        parts.append(f'<text x="{gx+bw*1.5+8}" y="{H-P-sl-6}" font-size="10.5" text-anchor="middle" fill="#a05252">{r["short"]*100:+.2f}%</text>')
        parts.append(f'<text x="{gx+bw+4}" y="{H-P+18}" font-size="10" text-anchor="middle" fill="currentColor">{esc(r["pkg"].replace("LMart_","L").replace("LRank_","N"))}</text>')
        parts.append(f'<text x="{gx+bw+4}" y="{H-P+32}" font-size="9.5" text-anchor="middle" fill="#888">{"tree" if r["family"]=="lambdamart" else "neural"}</text>')
    parts.append(f'<rect x="{P}" y="18" width="12" height="12" fill="#4c7a4c"/><text x="{P+20}" y="28" font-size="12" fill="currentColor">long closed-lot expected return</text>')
    parts.append(f'<rect x="{P+240}" y="18" width="12" height="12" fill="#a05252"/><text x="{P+260}" y="28" font-size="12" fill="currentColor">short closed-lot expected return (negative)</text>')
    parts.append('</svg>')
    table='<table class="fallback"><caption>Comparable two-tail packaged runs: side-level expected returns</caption><tr><th>Run</th><th>Family</th><th>Long</th><th>Short</th><th>Total return</th><th>Realized Sharpe</th></tr>'
    for r in runs:
        table+=f'<tr><td>{r["pkg"]}</td><td>{"tree" if r["family"]=="lambdamart" else "neural"}</td><td>{r["long"]*100:+.2f}%</td><td>{r["short"]*100:+.2f}%</td><td>{fmt_pct(r["total"])}</td><td>{r["sharpe"]:+.2f}</td></tr>'
    table+='</table>'
    return "".join(parts)+f'<details class="drawer"><summary>Data table</summary>{table}</details>'

def ch_stops():
    st=D["stop_concentration_LMart7"]
    shorts={k:v for k,v in st.items() if k.endswith("Short") and v["stops"]>0}
    shorts=sorted(shorts.items(),key=lambda kv:-kv[1]["pnl"])
    W,H,P=960,380,90
    vmax=max(v["pnl"] for _,v in shorts)
    bw=(W-P-60)/len(shorts)
    parts=[f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Stop-loss counts and dollar losses by asset for short positions in the two-tail tree run" class="chart">']
    for i,(k,v) in enumerate(shorts):
        x=P+i*bw
        h=(-v["pnl"]/vmax)*(H-2*P)
        asset=k.split("-")[0]
        hi = asset in ("XLK",)
        parts.append(f'<rect x="{x+4}" y="{H-P-h}" width="{bw-8}" height="{h}" fill="#a05252" opacity="{.95 if hi else .6}"/>')
        parts.append(f'<text x="{x+bw/2}" y="{H-P+16}" font-size="10" text-anchor="middle" fill="currentColor">{asset}</text>')
        parts.append(f'<text x="{x+bw/2}" y="{H-P+30}" font-size="9" text-anchor="middle" fill="#888">n={v["stops"]}</text>')
    xlk=[k for k,_ in shorts if k.startswith("XLK")]
    parts.append(f'<text x="{P}" y="26" font-size="12" fill="#a05252" font-weight="600">Dark bar: XLK — a high-momentum tech sector where short stops hit hardest</text>')
    parts.append(f'<text x="{W/2}" y="{H-8}" font-size="12" text-anchor="middle" fill="#555">stop-loss dollars by asset, short side, LMart_sectors_7 (counts under bars)</text>')
    parts.append('</svg>')
    tot=sum(v["pnl"] for _,v in shorts)
    table='<table class="fallback"><caption>Short-side stop concentration (LMart_sectors_7)</caption><tr><th>Asset</th><th>Stop count</th><th>Stop P&amp;L ($)</th></tr>'
    for k,v in shorts:
        table+=f'<tr><td>{k.split("-")[0]}</td><td>{v["stops"]}</td><td>${v["pnl"]:,.0f}</td></tr>'
    table+=f'<tr><td><strong>Total short stops</strong></td><td>{sum(v["stops"] for _,v in shorts)}</td><td><strong>${tot:,.0f}</strong></td></tr></table>'
    return "".join(parts)+f'<details class="drawer"><summary>Data table</summary>{table}</details>'

def ch_speed():
    sp=D["trade_speed"]
    rows=[]
    for run,groups in sp.items():
        for g,v in groups.items():
            rows.append((run,g,v))
    W,H,P=960,340,170
    vmax=max(abs(v["mean_roc_per_bar"]) for _,_,v in rows)*1.2
    rh=(H-2*P)/len(rows)
    parts=[f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Median bars held and per-bar drift by side and outcome" class="chart">']
    for i,(run,g,v) in enumerate(rows):
        y=P+i*rh+rh/2
        w=v["mean_roc_per_bar"]/vmax*(W-P-40)
        col="#4c7a4c" if "profit" in g else "#a05252"
        parts.append(f'<line x1="{P}" y1="{y}" x2="{P+w}" y2="{y}" stroke="{col}" stroke-width="16" opacity=".8"/>')
        parts.append(f'<text x="{P-10}" y="{y+4}" text-anchor="end" font-size="11.5" fill="currentColor">{run.split("_")[0]} · {g.replace("_"," ")}</text>')
        parts.append(f'<text x="{P+w+8}" y="{y+4}" font-size="11" fill="#555">{v["mean_roc_per_bar"]*100:+.2f}%/bar · median {int(v["median_bars"])} bars · n={v["n"]}</text>')
    parts.append('</svg>')
    table='<table class="fallback"><caption>Trade speed by side and outcome</caption><tr><th>Run</th><th>Group</th><th>n</th><th>Median bars held</th><th>Mean signed ROC/bar</th></tr>'
    for run,g,v in rows:
        table+=f'<tr><td>{run}</td><td>{g}</td><td>{v["n"]}</td><td>{int(v["median_bars"])}</td><td>{v["mean_roc_per_bar"]*100:+.2f}%</td></tr>'
    table+='</table>'
    return "".join(parts)+f'<details class="drawer"><summary>Data table</summary>{table}</details>'

FEATS=["atr_14","ret_20","volatility_20","vwap_dev_20","volume_z_20","ret_1","range_1","close_location","volume_change_1","poh_baz_vol_normalized_macd"]
def ch_feat_ranks():
    fr=D["feature_rank_trajectories"]
    iters=["LMart_1","LMart_2","LMart_3","LMart_4","LMart_5","LMart_6","LMart_7","LRank_10","LRank_13","LRank_14"]
    n=len(FEATS)
    W,H,P=960,430,110
    def px(i): return P+(W-P-30)*i/(len(iters)-1)
    def py(r): return P+(H-2*P)*(r-1)/(n-1)
    colors=["#a05252","#4c7a4c","#c98a3a","#7a4a8a","#3a5a8a","#8a6d3b","#5c8a8a","#b06a9a","#6a7a4a","#333"]
    parts=[f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Feature importance rank trajectories across iterations" class="chart">']
    for fi,f in enumerate(FEATS):
        pts=[]
        for ii,it in enumerate(iters):
            r=fr.get(it,{}).get(f)
            if r: pts.append((ii,r))
        d="M"+" L".join(f"{px(i):.1f},{py(r):.1f}" for i,r in pts)
        emph = f in ("atr_14","volatility_20") or "macd" in f
        sw="3" if emph else "1.4"
        op="1" if emph else ".45"
        dash=' stroke-dasharray="5 4"' if "macd" in f else ''
        parts.append(f'<path d="{d}" fill="none" stroke="{colors[fi]}" stroke-width="{sw}" opacity="{op}"{dash}/>')
        last=pts[-1]
        anchor="end" if False else "start"
        parts.append(f'<text x="{px(last[0])+8}" y="{py(last[1])+4}" font-size="11" fill="{colors[fi]}" font-weight="{700 if emph else 400}">{esc(f)}</text>')
    for ii,it in enumerate(iters):
        parts.append(f'<text x="{px(ii)}" y="{H-P+22}" font-size="10.5" text-anchor="middle" fill="#777">{it}</text>')
    parts.append(f'<text x="20" y="{P-24}" font-size="12" fill="#555">rank 1 (top) ↓</text>')
    for r in [1,5,10]:
        parts.append(f'<line x1="{P-6}" y1="{py(r)}" x2="{W-24}" y2="{py(r)}" stroke="#eee"/><text x="{P-14}" y="{py(r)+4}" font-size="10" text-anchor="end" fill="#aaa">{r}</text>')
    parts.append(f'<line x1="{P-6}" y1="{py(1)}" x2="{P-6}" y2="{py(n)}" stroke="#ccc"/>')
    parts.append('</svg>')
    table='<table class="fallback"><caption>Importance ranks by feature and iteration</caption><tr><th>Feature</th>'+"".join(f"<th>{i}</th>" for i in iters)+'</tr>'
    for f in FEATS:
        cells="".join(f"<td>{fr.get(it,{}).get(f,'—')}</td>" for it in iters)
        table+=f'<tr><td>{esc(f)}</td>{cells}</tr>'
    table+='</table>'
    return "".join(parts)+f'<details class="drawer"><summary>Data table</summary>{table}</details>'

def param_table(study, extra_rows=None):
    p=study["stable_params"]
    rows=[("Tail mode",p["tail_mode"]),("MACD included","No" if not p["use_macd"] else "Yes")]
    if "hidden_dim" in p:
        rows+= [("Hidden layers",f'{p["hidden_layers"]} (requested space allowed up to 3)'),
                ("Hidden width",f'{p["hidden_dim"]}'),
                ("Activation",f'LeakyReLU, slope ≈ {p["leaky_slope"]}'),
                ("Normalization",f'{p["normalization"]}'),
                ("Dropout",f'requested {p["dropout_requested"]}; EFFECTIVE: none — see methods note'),
                ("Optimizer",f'{p["optimizer"]}'),
                ("Learning rate",f'≈ {p["lr"]:.2e}'),("Weight decay",f'≈ {p["weight_decay"]:.2e}'),
                ("Loss sigma",p["sigma"]),("Grad clipping",p["grad_clip"])]
    else:
        rows+= [("Backend",p.get("backend","LightGBM")),
                ("Depth",p["max_depth"]),
                ("Leaves requested",p["num_leaves_requested"]),
                ("Leaves effective",f'{p["effective_leaves"]} (depth cap)'),
                ("Min child samples",p.get("min_child_samples")),
                ("Learning rate",f'≈ {p["lr"]}'),("Feature fraction",p["feature_fraction"]),
                ("Bagging fraction",p["bagging_fraction"]),("Max bin",p["max_bin"]),
                ("L1",f'≈ {p["l1"]}'),("L2",f'≈ {p["l2"]}'),
                ("LambdaRank truncation",p["truncation"])]
    h='<table class="params"><caption>Requested search space versus effective selected configuration</caption><tr><th>Knob</th><th>Selected value</th></tr>'
    for k,v in rows: h+=f'<tr><td>{k}</td><td>{esc(v)}</td></tr>'
    return h+'</table>'

# ---------- page assembly ----------
S={s["id"]:s for s in D["studies"]}
lm_svg=open(os.path.join(OUT,"assets/model-flow-lambdamart.svg")).read()
nn_svg=open(os.path.join(OUT,"assets/model-flow-lambdarank.svg")).read()

html_out=f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Learning to Rank Sector Returns: What Improved, What Broke, and What the Search Actually Taught Us</title>
<style>
:root{{--ink:#26231d;--sub:#5c564a;--paper:#faf8f4;--card:#fff;--blk:#f4f1ea;--acc:#e7efe4;--warn:#fdf6ec;--rule:#e3ded2}}
@media (prefers-color-scheme:dark){{:root{{--ink:#e8e4da;--sub:#a89f8d;--paper:#191713;--card:#211e19;--blk:#26231c;--acc:#232d22;--warn:#2b2317;--rule:#37332a}}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);font:17px/1.65 Georgia,'Times New Roman',serif}}
main{{max-width:820px;margin:0 auto;padding:0 20px 90px}}
header{{padding:64px 0 8px}}
h1{{font-size:clamp(30px,5vw,44px);line-height:1.15;margin:.2em 0;font-weight:700;letter-spacing:-.01em}}
.scope{{color:var(--sub);font-style:italic}}
.banner{{border:1.5px solid #a05252;background:var(--warn);padding:10px 14px;border-radius:8px;font-family:ui-sans-serif,system-ui,sans-serif;font-size:14px;margin-top:14px}}
h2{{font-size:26px;margin-top:2.4em;border-bottom:1px solid var(--rule);padding-bottom:.3em}}
h3{{font-size:19px;margin-top:1.8em}}
nav#toc{{position:sticky;top:0;z-index:9;background:var(--paper);border-bottom:1px solid var(--rule);padding:8px 0;font-family:ui-sans-serif,system-ui,sans-serif;font-size:12.5px}}
nav#toc a{{color:var(--sub);text-decoration:none;margin-right:12px;white-space:nowrap}}
nav#toc a:hover{{color:var(--ink)}}
figure{{margin:2em 0;padding:1em;background:var(--card);border:1px solid var(--rule);border-radius:10px}}
figure svg{{width:100%;height:auto;display:block}}
figcaption{{font-size:13.5px;color:var(--sub);margin-top:.8em;line-height:1.5}}
.badge{{display:inline-block;font:700 10.5px ui-sans-serif,system-ui,sans-serif;letter-spacing:.06em;text-transform:uppercase;padding:3px 9px;border-radius:999px;margin-right:6px;vertical-align:middle}}
.badge.test{{background:#e7efe4;color:#336033}} .badge.sel{{background:#e8ecf6;color:#3a4f8a}} .badge.post{{background:#fdf0dc;color:#8a5f1e}} .badge.val{{background:#f1e9f4;color:#6a3a82}}
@media(prefers-color-scheme:dark){{.badge.test{{background:#233523;color:#9fce9f}}.badge.sel{{background:#232c42;color:#a8bcf0}}.badge.post{{background:#3a2d15;color:#e8bd76}}.badge.val{{background:#33243a;color:#cbaade}}}}
table{{border-collapse:collapse;width:100%;font-family:ui-sans-serif,system-ui,sans-serif;font-size:13px;margin:1em 0}}
caption{{text-align:left;color:var(--sub);padding-bottom:6px;font-style:italic}}
th,td{{border-bottom:1px solid var(--rule);padding:6px 9px;text-align:left}}
th{{font-weight:600;color:var(--sub)}}
figure svg{{color:var(--ink)}}
.drawer{{margin:.6em 0}} .drawer summary{{cursor:pointer;color:var(--sub);font-family:ui-sans-serif,system-ui,sans-serif;font-size:13px}}
.src{{font-family:ui-sans-serif,system-ui,sans-serif;font-size:12px;color:var(--sub);border-left:3px solid var(--rule);padding-left:10px;margin-top:1em}}
.callout{{border-left:4px solid var(--rule);padding:.2em 1em;margin:1.4em 0;color:var(--sub)}}
.kicker{{font:700 11px ui-sans-serif,system-ui,sans-serif;letter-spacing:.12em;text-transform:uppercase;color:var(--sub)}}
ul li{{margin:.35em 0}}
@media print{{nav#toc{{position:static}} figure{{break-inside:avoid}} body{{font-size:11pt}} .drawer{{display:block}} details{{open:true}}}}
@media(max-width:720px){{body{{font-size:16px}} nav#toc{{overflow-x:auto;display:flex}}}}
:focus-visible{{outline:3px solid #4c7a4c;outline-offset:2px}}
</style>
</head>
<body>
<main>
<header>
<div class="kicker">Post-study journey report · Learning2Rank sectors</div>
<p class="banner" role="note"><strong>Evidence status:</strong> Optuna-selected configurations have not yet been rerun as frozen, untouched-test strategy packages. Packaged-run charts below are <em>illustrative packaged runs</em>, not the selected winners.</p>
<h1>Learning to Rank Sector Returns:<br>What Improved, What Broke, and What the Search Actually Taught Us</h1>
<p class="scope">11 U.S. sector ETFs · daily data · SPY benchmark · 2018–2025 source period · untouched test window Jan–Dec 2025</p>
</header>

<nav id="toc" aria-label="Table of contents">
<a href="#exec">Summary</a><a href="#start">Starting point</a><a href="#discipline">Discipline</a><a href="#tail">One tail</a><a href="#stability">Stability</a><a href="#models">Models</a><a href="#diverge">Divergence</a><a href="#shortside">Short side</a><a href="#vol">Volatility</a><a href="#ridge">Ridge</a><a href="#limits">Limits</a><a href="#next">Next steps</a><a href="#questions">Questions</a>
</nav>

<section id="exec">
<h2>Executive summary</h2>
<p><strong>The central finding:</strong> in this small cross-sectional sector study, strategy design — especially one-tail versus two-tail selection — dominated model and hyperparameter variation. The experiments produced useful positive results, but also exposed short-side fragility, regime dependence, and the difference between ranking quality and realized strategy outcomes. Model winners are conditional results, not universal conclusions.</p>
<ul>
<li>The experiment became more informative when it shifted from isolated runs to purged walk-forward studies with stability-based selection across three searches (500 attempted, <strong>203 completed</strong>, 297 pruned).</li>
<li>Tail choice dominated model tuning: long-only configurations won every completed search, and the short sleeve — especially under the fixed exit rule — was the persistent source of fragility in this sample. Regime explanations (e.g., low-volatility conditions) remain hypotheses from curve inspection until backed by a defined regime variable and stratified table.</li>
<li>Simpler effective capacity won: one neural hidden layer and depth-2 trees capped at 3 effective leaves — despite larger requested knobs in many trials.</li>
<li>The strongest lesson is methodological: ranking metrics, realized Sharpe, and absolute return answer different questions and must stay separated.</li>
</ul>
</section>

<section id="start">
<h2>1 · The starting point was plausible but unstable</h2>
<p>The earliest comparable packages — a LightGBM LambdaMART and a neural LambdaRank, both effectively two-tail (top-2 long, bottom-2 short) over the same late-2024-through-2025 window — established that the full pipeline worked: features were built, models trained, dates grouped, trades executed and logged. Neither produced a credible investment result. The tree baseline returned {fmt_pct(D['baselines']['LMart_sectors_1']['total_return'])} against SPY's {fmt_pct(D['baselines']['LMart_sectors_1']['spy_return'],signed=False)}; the neural baseline returned {fmt_pct(D['baselines']['LRank_sectors_1']['total_return'])}. Both trailed SPY by roughly 17–21 percentage points.</p>
<figure>{ch_equity()}<figcaption><strong>Takeaway.</strong> Both baselines sit near flat while SPY compounds upward — the pipeline functioned, but nothing here justified capital yet. Read each line left to right on identical date scales; the dashed grey line is always SPY. <em>Caveat:</em> early packages predate realized-Sharpe logging, so this panel compares returns, not risk-adjusted paths.</figcaption>
<p class="src">Sources: LMart_sectors_1 &amp; LRank_sectors_1 backtest summaries and equity curves; SPY file kept separate from the ranked universe. Evidence class: untouched test (early windows).</p></figure>
</section>

<section id="discipline">
<h2>2 · Discipline changed the quality of the evidence</h2>
<p>The progression ran through eight identifiable steps, dates derived from MLflow run start times and artifact modification times (August 2026): reproducible iteration packages with MLflow logging; symmetric long/short ranking diagnostics beyond NDCG; smooth percentile relevance grades with elite-tail weighting; explicit tail-mode experiments; modern neural controls (activation, normalization, optimizer families); LightGBM-versus-XGBoost as an explicit model-family knob; and finally purged expanding walk-forward folds with two seeds, a stability penalty, and a quarantined test set. ATR trailing exits arrived last — deliberately labeled post-selection.</p>
<figure>{ch_timeline()}<figcaption><strong>Reading guide.</strong> Green milestones changed what was being learned; brown ones made results reproducible; blue marks work explicitly outside model selection. The decisive conceptual step was treating <em>selection robustness</em>, not peak performance, as the target.</figcaption>
<p class="src">Sources: MLflow run chronology (23 finished runs; one run relogged three times and deduplicated by UUID) plus artifact creation times.</p></figure>
</section>

<section id="tail">
<h2>3 · The search consistently preferred one tail</h2>
<p>Across all three completed Optuna studies — 500 trials attempted, of which 203 completed (67 neural, 49 legacy LightGBM, 87 mixed tree; the rest pruned) — fold-level walk-forward Sharpe separated cleanly by tail mode. Long-only folds averaged between +0.94 and +1.02 depending on family; two-tail folds averaged negative in every family. Parameter-importance analysis makes the same point brutally: tail mode carries {D['param_importance']['neural']['tail_mode']*100:.0f}% of importance in the neural study and {D['param_importance']['tree_mixed']['tail_mode']*100:.0f}% in the mixed-tree study — more than every architecture knob combined.</p>
<figure>{ch_fold_box()}<figcaption><strong>Takeaway.</strong> Every green box sits above zero at the median; every red box straddles or sits below it. Sample counts are printed above each whisker. <em>Caveat:</em> even long-only dispersion remains wide (fold SD 0.76–0.94), so these are distributions, not guarantees.</figcaption>
<p class="src">Sources: optuna_fold_scores.csv joined to trial tail_mode across the three completed studies. Walk-forward Sharpe, not return.</p></figure>
<h3>Where the actual hyperparameter clusters sit</h3>
<figure>{ch_trial_map()}<figcaption><strong>How to read it.</strong> Each bubble is one cluster definition from <code>optuna_clusters.csv</code>, not one trial. Its vertical position is the cluster's raw mean walk-forward Sharpe; horizontal position is mean fold dispersion; area represents completed trials. Filled bubbles are long-only, hollow bubbles are two-tail, and the outer black ring marks the cluster selected by the study's stability rule. The useful pattern is that selected clusters occupy the high-Sharpe region without requiring the single most extreme trial.</figcaption>
<p class="src">Sources: optuna_clusters.csv and optuna_selection.json for all three studies. Cluster sizes are unique completed trials; fold×seed observations are not counted as clusters.</p></figure>
<p><em>Caveat:</em> tail-mode distributions reflect adaptive, imbalanced search allocation rather than a balanced randomized comparison (187 of 203 completed trials are long-only). TPE concentrating trials in a promising region is expected behavior, but it weakens naive distribution comparisons.</p>
<p>This pattern does not prove shorts are universally unprofitable — one small universe over one regime cannot carry that claim. It says that within this sample, the short book was where the losses concentrated.</p>
</section>

<section id="stability">
<h2>4 · Stable selection traded a little peak Sharpe for a repeatable region</h2>
<p>The chart now uses one unit only: raw mean walk-forward Sharpe. It compares the selected cluster mean, the stable trial chosen from that cluster, and the globally best single trial. The table then performs the separate stability calculation. This separation matters: a penalized objective must never appear as if it were another Sharpe observation.</p>
<figure>{ch_dumbbell()}<figcaption><strong>Takeaway.</strong> Neural trial 99 and legacy-tree trial 194 gave up some raw mean Sharpe relative to their best isolated trials, but came from the highest-scoring multi-trial stability regions. Mixed-tree trial 159 was both the stable choice and the best single trial. The visible table shows exactly how fold dispersion becomes the penalty and then the objective. <em>Caveat:</em> this selection rule was chosen after inspecting validation behavior and remains a research decision, not proof of out-of-sample superiority.</figcaption>
<p class="src">Sources: optuna_trials_clean.csv, optuna_clusters.csv, and optuna_selection.json per study. All chart points are raw mean walk-forward Sharpe; objectives appear only in the decomposition table.</p></figure>
</section>

<section id="models">
<h2>5 · The winning models were simpler than the search space</h2>
<p>Both selected configurations were simple relative to what the searches were allowed to build. Simplicity was <em>selected</em>, not proven causal — tail mode dominates the importance analysis and may confound any capacity interpretation. The neural winner used one hidden layer of 128 units — when up to three were available — with no normalization layer. The tree winner asked for 31 leaves but its depth-2 cap meant any tree could grow at most 3 effective leaves. Requested capacity and effective capacity diverged, and the divergence is the finding.</p>
<h3>Tree ranker (LambdaMART / mixed-backend winner)</h3>
<figure>{lm_svg}<figcaption><strong>How to read.</strong> Follow the daily cross-section through standardization and relevance grading into leaf-wise boosted trees; the knob panel states the exact selected values. Note the requested-versus-effective leaves cap — the central senior-level lesson. XGBoost was tested as a backend knob (native top-k pairs, no LightGBM truncation parameter). Under this adaptive search, no XGBoost configuration displaced the selected LightGBM region — only 7 of 87 complete mixed-backend trials chose it, averaging −0.67 versus +0.73. The backend allocation was too imbalanced for a controlled superiority claim.</figcaption>
<p class="src">Source: TreeRank_sectors_both_architecture_1/optuna_selection.json (stable trial 159, also the best single trial).</p></figure>
{param_table(S['TreeRank_sectors_both_architecture_1'])}
<h3>Neural ranker (LambdaRank winner)</h3>
<figure>{nn_svg}<figcaption><strong>How to read.</strong> Ten signals flow through cross-sectional standardization into a single 128-unit hidden layer with LeakyReLU slope ≈ 0.0604. Normalization is crossed out because none exists in the selected graph. Dropout 0.05 was requested but never instantiated — the builder inserts dropout only between hidden layers, and there is just one. The loss forms pairs within each date only, NDCG-weighted with sigma 1.25.</figcaption>
<p class="src">Source: LRank_sectors_both_broad_smooth_1/optuna_selection.json (stable trial 99).</p>
<details class="drawer"><summary>Methods note — effective dropout</summary>
<p><strong>Effective dropout: none.</strong> The search recorded a dropout value, but this architecture had only one hidden layer and the implementation applies dropout only between hidden layers. There is no next hidden layer, so the condition that creates the dropout module is never satisfied. The sampled value was therefore inactive and must not be credited for the result.</p></details></figure>
{param_table(S['LRank_sectors_both_broad_smooth_1'])}
</section>

<section id="diverge">
<h2>6 · Ranking quality and strategy outcomes diverged</h2>
<p><em>Runs below are illustrative packaged runs; none matches the frozen selected-trial fingerprint.</em></p>
<p>If ranking metrics predicted money, the chart below would slope upward. Instead NDCG separates almost perfectly by tail mode while realized Sharpe spans −1.36 to +1.19 within each band. A model can rank somewhat better without producing a better realized path — particularly when exit policy and the short book dominate P&amp;L.</p>
<figure>{ch_ndcg_sharpe()}<figcaption><strong>Takeaway.</strong> All long-only runs (opaque) score NDCG above ~0.50; all two-tail runs (faded) below ~0.42. Yet Sharpe varies enormously inside both groups. SPY's Sharpe (dashed) has no NDCG coordinate — it benchmarks the vertical axis only. <em>Caveat:</em> only packaged runs sharing the same 2025 test window and exit policy are plotted; incompatible assumptions would make this comparison meaningless.</figcaption>
<p class="src">Sources: backtest_summary.json + ranking_metrics.json for 15 packaged runs; evidence class: untouched test.</p></figure>
</section>

<section id="shortside">
<h2>7 · The short side carried disproportionate fragility</h2>
<p>In every comparable two-tail packaged run, the long sleeve earned positive expected return per closed lot while the short sleeve lost — and the spread widened as iterations progressed. Failures attribute to <em>short selections under the fixed exit rule</em>; a 100% stop rate can reflect signal direction, entry timing, threshold geometry, volatility scale, or their interaction. This is an association observed in closed-lot records, not a causal claim.</p>
<figure>{ch_long_short()}<figcaption><strong>Takeaway.</strong> Green bars (longs) stay positive across all six runs; red bars (shorts) are negative in all six, reaching −1.85% per lot in the worst case. <em>Caveat:</em> limited to comparable two-tail runs; long-only runs have no short leg to compare.</figcaption>
<p class="src">Sources: backtest_summary.json long/short closed-lot fields for six two-tail packages. Evidence class: untouched test.</p></figure>
<h3>Where the stops landed</h3>
<figure>{ch_stops()}<figcaption><strong>Takeaway.</strong> Short-side stop-outs concentrate in a handful of names — XLB shorts alone booked 49 stops costing $2,721, and XLK shorts lost $607 on just 11 stops. Counts and dollars both shown; percentages alone would hide the concentration. <em>Caveat:</em> one representative two-tail run; other runs differ in detail but agree in direction.</figcaption>
<p class="src">Source: LMart_sectors_7 strategy_trade_summary_by_asset.csv. Post-trade accounting, closed lots only.</p></figure>
<h3>Trade speed</h3>
<figure>{ch_speed()}<figcaption><strong>Takeaway.</strong> Losing trades held longer with weaker per-bar drift than winning ones in both books. Under fixed exit barriers, duration and exit type are mechanically related, so this is diagnostic of short selections <em>under the fixed exit rule</em> — not evidence that short ranking quality is poor or that stops arrived late.</figcaption>
<p class="src">Sources: trade_speed_calculations.csv for LMart_sectors_7 and LRank_sectors_14.</p></figure>
</section>

<section id="vol">
<h2>8 · Volatility became both a signal and an execution question</h2>
<p>Tracking feature ranks across iterations (ranks, not raw importances, because gain and permutation scales are incomparable across model families) shows volatility-family features — ATR, trailing volatility, range — persistently near the top regardless of model family, while the volume-normalized MACD variant fell out of favor as no-MACD configurations began winning selection — an associational observation from an adaptive search with correlated features, not a controlled feature ablation.</p>
<figure>{ch_feat_ranks()}<figcaption><strong>Takeaway.</strong> ATR holds rank 1–3 in every iteration; the MACD variant (dashed) drops from rank 2–3 to bottom-half once excluded configurations dominate. <em>Caveat:</em> feature importance ranks reflect model reliance, not causal attribution.</figcaption>
<p class="src">Sources: feature_importance.csv across LMart_sectors_1–7 (gain ranks) and LRank_sectors_10/13/14 (permutation-importance ranks).</p></figure>
<div class="callout">
<span class="badge post">Post-selection diagnostic</span>
<strong>Strategy 2 (ATR trailing exits): no completed evaluation exists.</strong>
The Director's Script asks for a fixed-stop versus ATR-trailing comparison on the exact selected model. At build time no Strategy 2 package matching the stable selected configurations was found among the artifacts, so no numbers are presented. Any such result must be treated as post-selection until evaluated once, unchanged, under a predeclared protocol.
<p class="src">Build-time discovery: artifacts searched for ATR/trailing/strategy-2 packages; none matched the selected stable configurations. Recorded as unresolved in the build log.</p>
</div>
</section>

<section id="ridge">
<h2>Designed but not yet evidenced: the Ridge baseline</h2>
<p>Ridge is a <em>linear sanity-check baseline</em>, designed alongside the rankers. Features are cross-sectionally standardized, while the forward-return target is centered and scaled within each date before fitting pooled observations. Date standardization removes each timestamp's target location and scale; Ridge then applies global L2 coefficient shrinkage. The shrinkage stabilizes coefficients when related signals such as returns, volatility, ATR, ranges, and volume measures are multicollinear.</p>
<p>This is <strong>date-standardized targeting plus global L2 regularization</strong> — not "date-wise Ridge regularization." Ridge is a baseline/control: it is not a layer in LambdaRank and not a regularizer for LambdaMART. <strong>No completed Ridge artifact exists at build time, so no result is reported</strong> — the design is recorded as designed but not yet evidenced. When run, it should report validation/test ranking metrics, realized strategy metrics, coefficient signs and magnitudes, and coefficient stability across time splits.</p>
<p class="src">Status: designed but not yet evidenced. No result is inferred from configuration defaults.</p>
</section>

<section id="limits">
<h2>9 · What the search did not establish</h2>
<ul>
<li>One small universe over one historical period cannot establish broad efficacy.</li>
<li>Repeated experimentation raises selection risk despite walk-forward folds — the studies themselves were iterated.</li>
<li>SPY is a benchmark, not a risk-matched portfolio; comparing a 2-position trading rule to a diversified index flatters neither.</li>
<li>No transaction costs, slippage, borrow costs, or capacity model appear unless artifacts explicitly contain them — they do not.</li>
<li>Long-only superiority may simply reflect the sample's market regime.</li>
<li>Strategy 2 remains post-selection until run once under a predeclared protocol.</li>
<li>Feature importance is not causal attribution.</li>
<li>The selected trials require exact untouched-test reruns before final claims.</li>
</ul>
</section>

<section id="next">
<h2>10 · Recommended next steps</h2>
<ol>
<li>Freeze the selected stable neural (trial 99 configuration) and tree (trial 159 configuration) setups.</li>
<li>Run Strategy 1 and Strategy 2 once each on identical untouched-test data, with no further tuning.</li>
<li>Add costs and a risk-matched long-only benchmark before making publication claims.</li>
<li>Repeat the frozen protocol on a separately named dataset before changing features or labels again.</li>
</ol>
</section>

<section id="questions">
<h2>11 · Further questions worth asking</h2>
<ul>
<li>Does the long-only result survive a different asset universe or period?</li>
<li>Does a wider short ATR trail improve robustness, or merely raise loss tolerance?</li>
<li>Are top-decile label weights helping rank quality and strategy outcomes, or only validation selection?</li>
<li>Does the no-MACD selection persist under a matched feature-on/feature-off experiment and on new data?</li>
</ul>
</section>

<footer style="margin-top:4em;border-top:1px solid var(--rule);padding-top:1em">
<p class="src">Every quantitative statement resolves to report_sources.json via the source ids cited beneath each figure. Charts degrade to semantic tables (each drawer) when graphics are unavailable. No external resources; renders offline and prints cleanly. Build log: report_build_log.md.</p>
</footer>
</main>
</body>
</html>"""

open(os.path.join(OUT, "report.html"), "w").write(html_out)
print("report.html bytes:", os.path.getsize(os.path.join(OUT, "report.html")))
