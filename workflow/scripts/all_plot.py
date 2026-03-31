#!/usr/bin/env python3
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import Optional
import click
import pandas as pd
import plotly.graph_objects as go


def load_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t", dtype=str).fillna("")


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    d = df.copy()
    for c in cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def write_no_data_html(path: Path, message: str = "No data to plot.") -> None:
    html = f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>plot</title></head>
<body><p><em>{message}</em></p></body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def _prepare_description_col(d: pd.DataFrame) -> pd.DataFrame:
    if "description" not in d.columns:
        d["description"] = "N/A"
    else:
        d["description"] = d["description"].astype(str).replace("", "N/A").replace("nan", "N/A")

    for src, dst in [
        ("chain_i_description", "chain_i_desc"),
        ("chain_j_description", "chain_j_desc"),
    ]:
        if src in d.columns:
            d[dst] = d[src].astype(str).replace("", "N/A").replace("nan", "N/A")
        else:
            d[dst] = "N/A"

    return d


_GT_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
]


def _gt_color(idx: int) -> str:
    return _GT_PALETTE[idx % len(_GT_PALETTE)]


# ═══════════════════════════════════════════════════════════════════════════
# TM-score distribution
#   • ground-truth multi-select checklist (preserved from reference)
#   • baseline (all data) always shown
#   • NEW: strip / CDF toggle
# ═══════════════════════════════════════════════════════════════════════════
def plot_tm_score_distribution(
    df_pred: pd.DataFrame,
    out_html: Path,
    title: str = "Distribution of TM scores across all predictions",
) -> bool:
    required = {"sample_id", "prediction_id", "TM1", "TM2"}
    if df_pred.empty or not required.issubset(df_pred.columns):
        write_no_data_html(out_html, "No TM score data available.")
        return False

    d = df_pred.copy()
    d["sample_id"] = d["sample_id"].astype(str)
    d["prediction_id"] = d["prediction_id"].astype(str)

    for c in ["TM1", "TM2"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d["name"] = d["sample_id"].str.split("_seed-").str[0].astype(str).replace("nan", "N/A")

    if "seed" not in d.columns or d["seed"].astype(str).eq("").all():
        d["seed"] = d["sample_id"].str.extract(r"_seed-(\d+)", expand=False).fillna("N/A")
    else:
        d["seed"] = d["seed"].astype(str).replace("", "N/A").replace("nan", "N/A")

    if "sample" not in d.columns or d["sample"].astype(str).eq("").all():
        d["sample"] = d["sample_id"].str.extract(r"_sample-(\d+)", expand=False).fillna("N/A")
    else:
        d["sample"] = d["sample"].astype(str).replace("", "N/A").replace("nan", "N/A")

    d["tm_score"] = pd.to_numeric(d["TM2"], errors="coerce")
    d = d[d["tm_score"].notna()].copy()
    if d.empty:
        write_no_data_html(out_html, "No valid TM scores available.")
        return False

    for c in ["ranking_score", "ptm", "iptm", "mean_plddt_total", "fraction_disordered"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    if "is_top" in d.columns:
        d["is_top"] = d["is_top"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        d["is_top"] = False

    d = _prepare_description_col(d)

    if "ground_truth_id" not in d.columns:
        d["ground_truth_id"] = "default"
    else:
        d["ground_truth_id"] = d["ground_truth_id"].astype(str).replace("", "default").replace("nan", "default")

    meta_cols = [
        "name", "sample", "seed", "ranking_score", "ptm", "iptm",
        "mean_plddt_total", "fraction_disordered", "has_clash", "description",
        "ground_truth_id",
    ]
    for c in meta_cols:
        if c not in d.columns:
            d[c] = "N/A"

    d["is_top_str"] = d["is_top"].map({True: "True", False: "False"})

    gt_ids = sorted(d["ground_truth_id"].unique().tolist())
    gt_color_map = {gt: _gt_color(i) for i, gt in enumerate(gt_ids)}

    n_total = len(d)
    default_mode = "cdf" if n_total > 100 else "strip"

    # ── Build BOTH strip and CDF data for baseline + each GT ──
    jitter_val = 0.01
    d["jitter"] = np.random.uniform(-jitter_val, jitter_val, size=len(d))

    def _build_both(df_all, df_top_src):
        """Return dict with strip_* and cdf_* keys for a subset."""
        # strip — all
        s_all_x = (df_all["tm_score"] + df_all["jitter"]).tolist()
        s_all_cd = df_all[meta_cols + ["is_top_str"]].values.tolist()

        # strip — top
        df_top = df_top_src.copy()
        if not df_top.empty:
            df_top["jitter"] = np.random.uniform(-jitter_val, jitter_val, size=len(df_top))
            s_top_x = (df_top["tm_score"] + df_top["jitter"]).tolist()
            s_top_cd = df_top[meta_cols + ["is_top_str"]].values.tolist()
        else:
            s_top_x, s_top_cd = [], []

        # cdf — all
        df_s = df_all.sort_values("tm_score").reset_index(drop=True)
        n = len(df_s)
        c_all_x = df_s["tm_score"].tolist()
        c_all_y = [(i + 1) / n for i in range(n)]
        c_all_cd = df_s[meta_cols + ["is_top_str"]].values.tolist()

        # cdf — top
        if not df_top.empty:
            dt_s = df_top.sort_values("tm_score").reset_index(drop=True)
            nt = len(dt_s)
            c_top_x = dt_s["tm_score"].tolist()
            c_top_y = [(i + 1) / nt for i in range(nt)]
            c_top_cd = dt_s[meta_cols + ["is_top_str"]].values.tolist()
        else:
            c_top_x, c_top_y, c_top_cd = [], [], []

        return {
            "strip_all_x": s_all_x, "strip_all_cd": s_all_cd,
            "strip_top_x": s_top_x, "strip_top_cd": s_top_cd,
            "cdf_all_x": c_all_x, "cdf_all_y": c_all_y, "cdf_all_cd": c_all_cd,
            "cdf_top_x": c_top_x, "cdf_top_y": c_top_y, "cdf_top_cd": c_top_cd,
        }

    # Baseline (all data combined)
    baseline_data = _build_both(d, d[d["is_top"]])

    # Per ground-truth
    gt_data_list = []
    for gt_id in gt_ids:
        dg = d[d["ground_truth_id"] == gt_id].copy()
        if dg.empty:
            continue
        color = gt_color_map[gt_id]
        entry = _build_both(dg, dg[dg["is_top"]])
        entry["gt_id"] = gt_id
        entry["color"] = color
        gt_data_list.append(entry)

    baseline_json = json.dumps(baseline_data)
    gt_data_json = json.dumps(gt_data_list)

    hover_strip = (
        "<b>name:</b> %{customdata[0]}<br>"
        "<b>description:</b> %{customdata[9]}<br>"
        "<b>ground truth:</b> %{customdata[10]}<br>"
        "<b>sample:</b> %{customdata[1]}<br>"
        "<b>seed:</b> %{customdata[2]}<br>"
        "<b>ranking score:</b> %{customdata[3]}<br>"
        "<b>ptm:</b> %{customdata[4]}<br>"
        "<b>iptm:</b> %{customdata[5]}<br>"
        "<b>mean pLDDT:</b> %{customdata[6]}<br>"
        "<b>fraction disordered:</b> %{customdata[7]}<br>"
        "<b>has clash:</b> %{customdata[8]}<br>"
        "<b>is top:</b> %{customdata[11]}<br>"
        "<b>TM score:</b> %{x:.3f}<br>"
        "<extra></extra>"
    )
    hover_cdf = (
        "<b>name:</b> %{customdata[0]}<br>"
        "<b>description:</b> %{customdata[9]}<br>"
        "<b>ground truth:</b> %{customdata[10]}<br>"
        "<b>sample:</b> %{customdata[1]}<br>"
        "<b>seed:</b> %{customdata[2]}<br>"
        "<b>ranking score:</b> %{customdata[3]}<br>"
        "<b>ptm:</b> %{customdata[4]}<br>"
        "<b>iptm:</b> %{customdata[5]}<br>"
        "<b>mean pLDDT:</b> %{customdata[6]}<br>"
        "<b>fraction disordered:</b> %{customdata[7]}<br>"
        "<b>has clash:</b> %{customdata[8]}<br>"
        "<b>is top:</b> %{customdata[11]}<br>"
        "<b>TM score \u2264</b> %{x:.3f}<br>"
        "<b>Fraction:</b> %{y:.3f}<br>"
        "<extra></extra>"
    )

    n_visible = min(max(len(gt_ids), 2), 8)
    options_html = "\n".join(
        f'          <option value="{gt}" selected '
        f'style="padding:3px 6px;">'
        f'&#9632; {gt}</option>'
        for gt in gt_ids
    )

    strip_active = "active" if default_mode == "strip" else ""
    cdf_active = "active" if default_mode == "cdf" else ""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>TM Score Distribution</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  body {{ font-family: Arial, sans-serif; margin: 20px; }}
  .container {{ max-width: 1400px; margin: 0 auto; }}

  h2.plot-title {{
      text-align: center; font-size: 18px; margin: 0 0 14px 0;
      color: #333;
  }}

  .controls {{
      margin-bottom: 12px; padding: 10px 14px;
      background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px;
  }}
  .controls-label {{
      font-size: 13px; font-weight: bold; color: #495057;
      margin-bottom: 6px;
  }}
  .controls-row {{
      display: flex; align-items: flex-start; gap: 10px;
  }}

  #gt-select {{
      font-family: monospace; font-size: 13px;
      border: 1px solid #ced4da; border-radius: 4px;
      padding: 2px; min-width: 220px;
  }}
  #gt-select option {{
      padding: 2px 6px;
  }}
  #gt-select option:checked {{
      background: #e7f1ff;
  }}

  .btn-col {{
      display: flex; flex-direction: column; gap: 4px;
  }}
  .btn-col button {{
      padding: 4px 12px; font-size: 12px; cursor: pointer;
      border: 1px solid #adb5bd; border-radius: 3px;
      background: #fff; color: #495057; white-space: nowrap;
  }}
  .btn-col button:hover {{ background: #e9ecef; }}

  .toggle-group {{
      display: flex; gap: 0; margin-left: 18px; align-self: flex-start;
  }}
  .toggle-group button {{
      padding: 6px 16px; font-size: 13px; cursor: pointer;
      border: 1px solid #adb5bd; background: #fff; color: #495057;
  }}
  .toggle-group button:first-child {{
      border-radius: 4px 0 0 4px;
  }}
  .toggle-group button:last-child {{
      border-radius: 0 4px 4px 0;
      border-left: none;
  }}
  .toggle-group button.active {{
      background: #4C72B0; color: #fff; border-color: #4C72B0;
      font-weight: bold;
  }}

  .hint {{
      font-size: 11px; color: #6c757d; margin-top: 4px;
  }}
  .legend-note {{
      font-size: 11px; color: #6c757d; text-align: center;
      margin-top: 2px;
  }}

  #plot {{ margin-top: 4px; }}
</style>
</head>
<body>
<div class="container">
  <h2 class="plot-title">{title}</h2>

  <div class="controls">
    <div class="controls-label">Highlight ground truth references</div>
    <div class="controls-row">
      <select id="gt-select" multiple size="{n_visible}">
{options_html}
      </select>
      <div class="btn-col">
        <button id="btn-all">Select All</button>
        <button id="btn-none">Clear All</button>
      </div>
      <div class="toggle-group">
        <button id="btn-strip" class="{strip_active}">Strip Chart</button>
        <button id="btn-cdf" class="{cdf_active}">CDF</button>
      </div>
    </div>
    <div class="hint">
      Click to toggle ground truths. Baseline (all data) is always shown.
      Use Strip Chart / CDF to switch view.
    </div>
  </div>

  <div id="plot"></div>
  <div id="legend-note" class="legend-note"></div>
</div>

<script>
var BASELINE    = {baseline_json};
var GT_DATA     = {gt_data_json};
var HOVER_STRIP = {json.dumps(hover_strip)};
var HOVER_CDF   = {json.dumps(hover_cdf)};
var currentMode = {json.dumps(default_mode)};

var sel        = document.getElementById('gt-select');
var btnStrip   = document.getElementById('btn-strip');
var btnCdf     = document.getElementById('btn-cdf');
var legendNote = document.getElementById('legend-note');

// --- Toggle selection on plain click (no ctrl needed) ---
sel.addEventListener('mousedown', function(e) {{
    if (e.target.tagName === 'OPTION') {{
        e.preventDefault();
        e.target.selected = !e.target.selected;
        sel.dispatchEvent(new Event('change'));
    }}
}});

sel.addEventListener('change', rebuildPlot);

document.getElementById('btn-all').addEventListener('click', function() {{
    for (var i = 0; i < sel.options.length; i++) sel.options[i].selected = true;
    rebuildPlot();
}});
document.getElementById('btn-none').addEventListener('click', function() {{
    for (var i = 0; i < sel.options.length; i++) sel.options[i].selected = false;
    rebuildPlot();
}});

btnStrip.addEventListener('click', function() {{
    currentMode = 'strip';
    btnStrip.classList.add('active');
    btnCdf.classList.remove('active');
    rebuildPlot();
}});
btnCdf.addEventListener('click', function() {{
    currentMode = 'cdf';
    btnCdf.classList.add('active');
    btnStrip.classList.remove('active');
    rebuildPlot();
}});

// --- Color the options to match their trace color ---
(function colorOptions() {{
    var gtColors = {{}};
    GT_DATA.forEach(function(gt) {{ gtColors[gt.gt_id] = gt.color; }});
    for (var i = 0; i < sel.options.length; i++) {{
        var c = gtColors[sel.options[i].value] || '#999';
        sel.options[i].style.color = c;
        sel.options[i].style.fontWeight = 'bold';
    }}
}})();

function rebuildPlot() {{
    var selected = new Set();
    for (var i = 0; i < sel.options.length; i++) {{
        if (sel.options[i].selected) selected.add(sel.options[i].value);
    }}

    var traces = [];
    var isStrip = (currentMode === 'strip');
    var HOVER   = isStrip ? HOVER_STRIP : HOVER_CDF;

    // ---- Baseline: always visible ----
    if (isStrip) {{
        if (BASELINE.strip_all_x.length > 0) {{
            traces.push({{
                x: BASELINE.strip_all_x,
                y: BASELINE.strip_all_x.map(function() {{ return 0.5; }}),
                mode: 'markers',
                name: 'All (baseline)',
                customdata: BASELINE.strip_all_cd,
                hovertemplate: HOVER,
                showlegend: false,
                marker: {{
                    color: 'rgba(76,114,176,0.25)', size: 5,
                    line: {{ width: 0.5, color: 'rgba(0,0,0,0.15)' }}
                }}
            }});
        }}
        if (BASELINE.strip_top_x.length > 0) {{
            traces.push({{
                x: BASELINE.strip_top_x,
                y: BASELINE.strip_top_x.map(function() {{ return 0.5; }}),
                mode: 'markers',
                name: 'Top (baseline)',
                customdata: BASELINE.strip_top_cd,
                hovertemplate: HOVER,
                showlegend: false,
                marker: {{
                    color: 'rgba(213,94,0,0.25)', size: 7, symbol: 'diamond',
                    line: {{ width: 0.5, color: 'rgba(0,0,0,0.15)' }}
                }}
            }});
        }}
    }} else {{
        if (BASELINE.cdf_all_x.length > 0) {{
            traces.push({{
                x: BASELINE.cdf_all_x, y: BASELINE.cdf_all_y,
                mode: 'lines',
                name: 'All (baseline)',
                customdata: BASELINE.cdf_all_cd,
                hovertemplate: HOVER,
                showlegend: false,
                line: {{ color: 'rgba(76,114,176,0.3)', width: 2.5 }}
            }});
        }}
        if (BASELINE.cdf_top_x.length > 0) {{
            traces.push({{
                x: BASELINE.cdf_top_x, y: BASELINE.cdf_top_y,
                mode: 'lines',
                name: 'Top (baseline)',
                customdata: BASELINE.cdf_top_cd,
                hovertemplate: HOVER,
                showlegend: false,
                line: {{ color: 'rgba(213,94,0,0.25)', width: 2, dash: 'dot' }}
            }});
        }}
    }}

    // ---- Highlighted ground truths ----
    GT_DATA.forEach(function(gt) {{
        if (!selected.has(gt.gt_id)) return;

        if (isStrip) {{
            if (gt.strip_all_x.length > 0) {{
                traces.push({{
                    x: gt.strip_all_x,
                    y: gt.strip_all_x.map(function() {{ return 0.5; }}),
                    mode: 'markers',
                    name: gt.gt_id,
                    customdata: gt.strip_all_cd,
                    hovertemplate: HOVER,
                    showlegend: true,
                    marker: {{
                        color: gt.color, size: 8, opacity: 0.85,
                        line: {{ width: 1, color: 'black' }}
                    }}
                }});
            }}
            if (gt.strip_top_x.length > 0) {{
                traces.push({{
                    x: gt.strip_top_x,
                    y: gt.strip_top_x.map(function() {{ return 0.5; }}),
                    mode: 'markers',
                    name: gt.gt_id + ' (top)',
                    customdata: gt.strip_top_cd,
                    hovertemplate: HOVER,
                    showlegend: true,
                    marker: {{
                        color: gt.color, size: 10, opacity: 0.95,
                        symbol: 'star',
                        line: {{ width: 1.5, color: 'black' }}
                    }}
                }});
            }}
        }} else {{
            if (gt.cdf_all_x.length > 0) {{
                traces.push({{
                    x: gt.cdf_all_x, y: gt.cdf_all_y,
                    mode: 'lines',
                    name: gt.gt_id,
                    customdata: gt.cdf_all_cd,
                    hovertemplate: HOVER,
                    showlegend: true,
                    line: {{ color: gt.color, width: 3 }}
                }});
            }}
            if (gt.cdf_top_x.length > 0) {{
                traces.push({{
                    x: gt.cdf_top_x, y: gt.cdf_top_y,
                    mode: 'lines',
                    name: gt.gt_id + ' (top)',
                    customdata: gt.cdf_top_cd,
                    hovertemplate: HOVER,
                    showlegend: true,
                    line: {{ color: gt.color, width: 3, dash: 'dash' }}
                }});
            }}
        }}
    }});

    var yaxis, plotHeight;
    if (isStrip) {{
        yaxis = {{ showticklabels: false, title: '', range: [0, 1] }};
        plotHeight = 450;
        legendNote.textContent =
            'Circle = all predictions  |  \u2605 = top predictions  |  Faint = baseline';
    }} else {{
        yaxis = {{
            title: 'Cumulative fraction', range: [0, 1.02],
            tickvals: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        }};
        plotHeight = 700;
        legendNote.textContent =
            'Solid = all predictions  |  Dashed = top predictions  |  Faint = baseline';
    }}

    var layout = {{
        xaxis: {{ title: 'TM score', range: [0, 1],
                  tickvals: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] }},
        yaxis: yaxis,
        template: 'plotly_white',
        hovermode: 'closest',
        height: plotHeight,
        margin: {{ l: 70, r: 30, t: 30, b: 60 }},
        legend: {{
            orientation: 'h', yanchor: 'bottom', y: 1.01,
            xanchor: 'center', x: 0.5, font: {{ size: 11 }},
        }},
    }};

    Plotly.newPlot('plot', traces, layout, {{ responsive: true }});
}}

rebuildPlot();
</script>
</body>
</html>
"""

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    return True


def add_prediction_metadata(df_pair: pd.DataFrame, df_pred: pd.DataFrame) -> pd.DataFrame:
    d = df_pair.copy()
    if d.empty:
        return d

    required = {"sample_id", "prediction_id"}
    if not required.issubset(d.columns):
        return d

    d["sample_id"] = d["sample_id"].astype(str)
    d["prediction_id"] = d["prediction_id"].astype(str)

    if "pair_iptm" in d.columns:
        d["pair_iptm"] = pd.to_numeric(d["pair_iptm"], errors="coerce")

    if "chain_i" in d.columns:
        d["chain_i"] = d["chain_i"].astype(str)
    if "chain_j" in d.columns:
        d["chain_j"] = d["chain_j"].astype(str)

    if "is_diagonal" in d.columns:
        diag = d["is_diagonal"].astype(str).str.lower().isin(["true", "1", "yes"])
        d = d[~diag].copy()

    if "is_top" in d.columns:
        d["is_top"] = d["is_top"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        d["is_top"] = False

    if "seed" in d.columns:
        d["seed"] = d["seed"].astype(str).replace("", "N/A").replace("nan", "N/A")
    if "sample" in d.columns:
        d["sample"] = d["sample"].astype(str).replace("", "N/A").replace("nan", "N/A")

    if df_pred.empty or not required.issubset(df_pred.columns):
        return d

    p = df_pred.copy()
    p["sample_id"] = p["sample_id"].astype(str)
    p["prediction_id"] = p["prediction_id"].astype(str)

    keep = [
        "sample_id",
        "prediction_id",
        "ranking_score",
        "iptm",
        "ptm",
        "mean_plddt_total",
        "description",
    ]
    keep = [c for c in keep if c in p.columns]
    extra = [c for c in keep if c not in d.columns or c in ["sample_id", "prediction_id"]]
    p = p[extra].drop_duplicates()

    if len(extra) > 2:
        d = d.merge(p, on=["sample_id", "prediction_id"], how="left")

    for c in ["ranking_score", "iptm", "ptm", "mean_plddt_total"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    return d


# ═══════════════════════════════════════════════════════════════════════════
# chain-pair ipTM distribution
#   • NO ground-truth selector (same as reference)
#   • NEW: strip / CDF toggle
# ═══════════════════════════════════════════════════════════════════════════
def plot_chain_pair_iptm_cumulative(
    df_pair: pd.DataFrame,
    out_html: Path,
    title: str = "Distribution of chain-pair ipTM across all predictions",
) -> bool:
    required = {"sample_id", "prediction_id", "pair_iptm"}
    if df_pair.empty or not required.issubset(df_pair.columns):
        write_no_data_html(out_html, "No chain-pair ipTM data available.")
        return False

    d = df_pair.copy()
    d["sample_id"] = d["sample_id"].astype(str)
    d["prediction_id"] = d["prediction_id"].astype(str)
    d["pair_iptm"] = pd.to_numeric(d["pair_iptm"], errors="coerce")
    d = d[d["pair_iptm"].notna()].copy()

    if d.empty:
        write_no_data_html(out_html, "No valid chain-pair ipTM values available.")
        return False

    d["name"] = d["sample_id"].str.split("_seed-").str[0].astype(str).replace("nan", "N/A")

    if "seed" not in d.columns or d["seed"].astype(str).isin(["", "N/A", "nan"]).all():
        d["seed"] = d["sample_id"].str.extract(r"_seed-(\d+)", expand=False).fillna("N/A")
    else:
        d["seed"] = d["seed"].astype(str).replace("", "N/A").replace("nan", "N/A")

    if "sample" not in d.columns or d["sample"].astype(str).isin(["", "N/A", "nan"]).all():
        d["sample"] = d["sample_id"].str.extract(r"_sample-(\d+)", expand=False).fillna("N/A")
    else:
        d["sample"] = d["sample"].astype(str).replace("", "N/A").replace("nan", "N/A")

    for c in ["chain_i", "chain_j"]:
        if c in d.columns:
            d[c] = d[c].astype(str).replace("nan", "N/A")
        else:
            d[c] = "N/A"

    d = _prepare_description_col(d)

    if "is_top" in d.columns:
        if d["is_top"].dtype == bool:
            pass
        else:
            d["is_top"] = d["is_top"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        d["is_top"] = False

    d["is_top_str"] = d["is_top"].map({True: "True", False: "False"})

    n_predictions = len(d)
    default_mode = "cdf" if n_predictions > 100 else "strip"

    hover_cols = [
        "name", "seed", "sample", "chain_i", "chain_j",
        "chain_i_desc", "chain_j_desc", "description", "is_top_str",
    ]

    # --- Build BOTH strip and CDF data ---
    jitter_val = 0.01
    d["jitter"] = np.random.uniform(-jitter_val, jitter_val, size=len(d))

    strip_all_x = (d["pair_iptm"] + d["jitter"]).tolist()
    strip_all_cd = d[hover_cols].values.tolist()

    d_top = d[d["is_top"]].copy()
    if not d_top.empty:
        d_top["jitter"] = np.random.uniform(-jitter_val, jitter_val, size=len(d_top))
        strip_top_x = (d_top["pair_iptm"] + d_top["jitter"]).tolist()
        strip_top_cd = d_top[hover_cols].values.tolist()
    else:
        strip_top_x, strip_top_cd = [], []

    d_sorted = d.sort_values("pair_iptm").reset_index(drop=True)
    n_s = len(d_sorted)
    cdf_all_x = d_sorted["pair_iptm"].tolist()
    cdf_all_y = [(i + 1) / n_s for i in range(n_s)]
    cdf_all_cd = d_sorted[hover_cols].values.tolist()

    if not d_top.empty:
        dt_s = d_top.sort_values("pair_iptm").reset_index(drop=True)
        n_t = len(dt_s)
        cdf_top_x = dt_s["pair_iptm"].tolist()
        cdf_top_y = [(i + 1) / n_t for i in range(n_t)]
        cdf_top_cd = dt_s[hover_cols].values.tolist()
    else:
        cdf_top_x, cdf_top_y, cdf_top_cd = [], [], []

    plot_data = {
        "strip_all_x": strip_all_x, "strip_all_cd": strip_all_cd,
        "strip_top_x": strip_top_x, "strip_top_cd": strip_top_cd,
        "cdf_all_x": cdf_all_x, "cdf_all_y": cdf_all_y, "cdf_all_cd": cdf_all_cd,
        "cdf_top_x": cdf_top_x, "cdf_top_y": cdf_top_y, "cdf_top_cd": cdf_top_cd,
    }
    plot_data_json = json.dumps(plot_data)

    strip_hover = (
        "<b>name:</b> %{customdata[0]}<br>"
        "<b>description:</b> %{customdata[7]}<br>"
        "<b>pair ipTM:</b> %{x:.3f}<br>"
        "<b>seed:</b> %{customdata[1]}<br>"
        "<b>sample:</b> %{customdata[2]}<br>"
        "<b>chain i:</b> %{customdata[3]} (%{customdata[5]})<br>"
        "<b>chain j:</b> %{customdata[4]} (%{customdata[6]})<br>"
        "<b>is top:</b> %{customdata[8]}<br>"
        "<extra></extra>"
    )
    cdf_hover = (
        "<b>name:</b> %{customdata[0]}<br>"
        "<b>description:</b> %{customdata[7]}<br>"
        "<b>pair ipTM \u2264</b> %{x:.3f}<br>"
        "<b>Fraction:</b> %{y:.3f}<br>"
        "<b>seed:</b> %{customdata[1]}<br>"
        "<b>sample:</b> %{customdata[2]}<br>"
        "<b>chain i:</b> %{customdata[3]} (%{customdata[5]})<br>"
        "<b>chain j:</b> %{customdata[4]} (%{customdata[6]})<br>"
        "<b>is top:</b> %{customdata[8]}<br>"
        "<extra></extra>"
    )

    strip_active = "active" if default_mode == "strip" else ""
    cdf_active = "active" if default_mode == "cdf" else ""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>chain-pair ipTM distribution</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  body {{ font-family: Arial, sans-serif; margin: 20px; }}
  .container {{ max-width: 1400px; margin: 0 auto; }}

  h2.plot-title {{
      text-align: center; font-size: 18px; margin: 0 0 14px 0;
      color: #333;
  }}

  .controls {{
      margin-bottom: 12px; padding: 10px 14px;
      background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px;
      display: flex; align-items: center; gap: 18px;
  }}
  .controls-label {{
      font-size: 13px; font-weight: bold; color: #495057;
  }}

  .toggle-group {{
      display: flex; gap: 0;
  }}
  .toggle-group button {{
      padding: 6px 16px; font-size: 13px; cursor: pointer;
      border: 1px solid #adb5bd; background: #fff; color: #495057;
  }}
  .toggle-group button:first-child {{
      border-radius: 4px 0 0 4px;
  }}
  .toggle-group button:last-child {{
      border-radius: 0 4px 4px 0;
      border-left: none;
  }}
  .toggle-group button.active {{
      background: #4C72B0; color: #fff; border-color: #4C72B0;
      font-weight: bold;
  }}

  .legend-note {{
      font-size: 11px; color: #6c757d; text-align: center;
      margin-top: 2px;
  }}

  #plot {{ margin-top: 4px; }}
</style>
</head>
<body>
<div class="container">
  <h2 class="plot-title">{title}</h2>

  <div class="controls">
    <span class="controls-label">View:</span>
    <div class="toggle-group">
      <button id="btn-strip" class="{strip_active}">Strip Chart</button>
      <button id="btn-cdf" class="{cdf_active}">CDF</button>
    </div>
  </div>

  <div id="plot"></div>
  <div id="legend-note" class="legend-note"></div>
</div>

<script>
var DATA        = {plot_data_json};
var STRIP_HOVER = {json.dumps(strip_hover)};
var CDF_HOVER   = {json.dumps(cdf_hover)};
var currentMode = {json.dumps(default_mode)};

var btnStrip   = document.getElementById('btn-strip');
var btnCdf     = document.getElementById('btn-cdf');
var legendNote = document.getElementById('legend-note');

btnStrip.addEventListener('click', function() {{
    currentMode = 'strip';
    btnStrip.classList.add('active');
    btnCdf.classList.remove('active');
    rebuildPlot();
}});
btnCdf.addEventListener('click', function() {{
    currentMode = 'cdf';
    btnCdf.classList.add('active');
    btnStrip.classList.remove('active');
    rebuildPlot();
}});

function rebuildPlot() {{
    var traces = [];
    var isStrip = (currentMode === 'strip');

    if (isStrip) {{
        if (DATA.strip_all_x.length > 0) {{
            traces.push({{
                x: DATA.strip_all_x,
                y: DATA.strip_all_x.map(function() {{ return 0.5; }}),
                mode: 'markers',
                name: 'All predictions',
                customdata: DATA.strip_all_cd,
                hovertemplate: STRIP_HOVER,
                showlegend: true,
                marker: {{
                    color: '#4C72B0', size: 6, opacity: 0.7,
                    line: {{ width: 0.5, color: 'black' }}
                }}
            }});
        }}
        if (DATA.strip_top_x.length > 0) {{
            traces.push({{
                x: DATA.strip_top_x,
                y: DATA.strip_top_x.map(function() {{ return 0.5; }}),
                mode: 'markers',
                name: 'Top predictions',
                customdata: DATA.strip_top_cd,
                hovertemplate: STRIP_HOVER,
                showlegend: true,
                marker: {{
                    color: '#D55E00', size: 8, opacity: 0.8,
                    line: {{ width: 1.5, color: 'black' }}
                }}
            }});
        }}
    }} else {{
        if (DATA.cdf_all_x.length > 0) {{
            traces.push({{
                x: DATA.cdf_all_x,
                y: DATA.cdf_all_y,
                mode: 'lines',
                name: 'All predictions',
                customdata: DATA.cdf_all_cd,
                hovertemplate: CDF_HOVER,
                showlegend: true,
                line: {{ color: '#4C72B0', width: 2.5 }}
            }});
        }}
        if (DATA.cdf_top_x.length > 0) {{
            traces.push({{
                x: DATA.cdf_top_x,
                y: DATA.cdf_top_y,
                mode: 'lines',
                name: 'Top predictions',
                customdata: DATA.cdf_top_cd,
                hovertemplate: CDF_HOVER,
                showlegend: true,
                line: {{ color: '#D55E00', width: 2.5, dash: 'solid' }}
            }});
        }}
    }}

    var yaxis, plotHeight;
    if (isStrip) {{
        yaxis = {{ showticklabels: false, title: '', range: [0, 1] }};
        plotHeight = 400;
        legendNote.textContent = 'Circle = all predictions  |  Large circle = top predictions';
    }} else {{
        yaxis = {{
            title: 'Cumulative fraction', range: [0, 1.02],
            tickvals: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        }};
        plotHeight = 650;
        legendNote.textContent = 'Blue = all predictions  |  Orange = top predictions';
    }}

    var layout = {{
        title: {{ text: '', x: 0.5, xanchor: 'center' }},
        xaxis: {{
            title: 'pair ipTM', range: [0, 1],
            tickvals: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        }},
        yaxis: yaxis,
        template: 'plotly_white',
        hovermode: 'closest',
        height: plotHeight,
        margin: {{ l: 70, r: 50, t: 40, b: 70 }},
        legend: {{
            orientation: 'h', yanchor: 'bottom', y: 1.02,
            xanchor: 'center', x: 0.5
        }},
    }};

    Plotly.newPlot('plot', traces, layout, {{ responsive: true }});
}}

rebuildPlot();
</script>
</body>
</html>
"""

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    return True


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--pair-tsv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Aggregated cohort chain-pair table, e.g. cohort_chain_pairs.tsv"
)
@click.option(
    "-o", "--out-html",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Output HTML path for ipTM plot."
)
@click.option(
    "--tm-plot",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output HTML path for TM score distribution plot."
)
@click.option(
    "--master-tsv",
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
    default=None,
    help="Optional: Path to all_master.tsv (contains TM1, TM2, description, etc.)."
)
def main(pair_tsv: Path, out_html: Path, tm_plot: Optional[Path], master_tsv: Optional[Path]):
    """
    Create two interactive plots:
    1. Distribution of chain-pair ipTM across all predictions.
    2. Distribution of TM scores across all predictions.
    """
    df_pair = load_tsv(pair_tsv)
    df_pair = coerce_numeric(df_pair, ["pair_iptm", "pair_pae_min"])

    df_master = load_tsv(master_tsv) if master_tsv is not None and master_tsv.exists() else pd.DataFrame()
    df_master = coerce_numeric(df_master, ["ranking_score", "iptm", "ptm", "mean_plddt_total"])

    d = add_prediction_metadata(df_pair, df_master)

    plot_chain_pair_iptm_cumulative(d, out_html)

    if tm_plot is not None:
        if master_tsv is not None and master_tsv.exists():
            tm_source = master_tsv
        else:
            tm_source = Path("reports/all/all_master.tsv")

        if not tm_source.exists():
            click.echo(f"⚠️  {tm_source} not found. Skipping TM score plot.")
        else:
            df_tm = load_tsv(tm_source)
            df_tm = coerce_numeric(df_tm, [
                "TM1", "TM2", "ranking_score", "ptm", "iptm",
                "mean_plddt_total", "fraction_disordered"
            ])
            plot_tm_score_distribution(df_tm, tm_plot)

    click.echo(f"✅ ipTM plot saved to: {out_html}")
    if tm_plot is not None:
        click.echo(f"✅ TM score plot saved to: {tm_plot}")


if __name__ == "__main__":
    main()