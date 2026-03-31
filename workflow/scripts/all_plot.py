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
<head><meta charset="utf-8"><title>chain-pair ipTM cumulative histogram</title></head>
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
    use_cdf = n_total > 100

    # --- Build baseline (all data, ignoring ground_truth_id) ---
    if use_cdf:
        d_sorted = d.sort_values("tm_score").reset_index(drop=True)
        baseline_all_x = d_sorted["tm_score"].tolist()
        baseline_all_y = [(i + 1) / len(d_sorted) for i in range(len(d_sorted))]
        baseline_all_cd = d_sorted[meta_cols + ["is_top_str"]].values.tolist()

        d_top = d[d["is_top"]].sort_values("tm_score").reset_index(drop=True)
        if not d_top.empty:
            baseline_top_x = d_top["tm_score"].tolist()
            baseline_top_y = [(i + 1) / len(d_top) for i in range(len(d_top))]
            baseline_top_cd = d_top[meta_cols + ["is_top_str"]].values.tolist()
        else:
            baseline_top_x, baseline_top_y, baseline_top_cd = [], [], []
    else:
        jitter = 0.01
        d["jitter"] = np.random.uniform(-jitter, jitter, size=len(d))
        baseline_all_x = (d["tm_score"] + d["jitter"]).tolist()
        baseline_all_y = [0.5] * len(d)
        baseline_all_cd = d[meta_cols + ["is_top_str"]].values.tolist()

        d_top = d[d["is_top"]].copy()
        if not d_top.empty:
            d_top["jitter"] = np.random.uniform(-jitter, jitter, size=len(d_top))
            baseline_top_x = (d_top["tm_score"] + d_top["jitter"]).tolist()
            baseline_top_y = [0.5] * len(d_top)
            baseline_top_cd = d_top[meta_cols + ["is_top_str"]].values.tolist()
        else:
            baseline_top_x, baseline_top_y, baseline_top_cd = [], [], []

    baseline_data = {
        "all_x": baseline_all_x, "all_y": baseline_all_y, "all_cd": baseline_all_cd,
        "top_x": baseline_top_x, "top_y": baseline_top_y, "top_cd": baseline_top_cd,
    }

    # --- Build per-ground-truth overlay data ---
    gt_data_list = []
    for gt_id in gt_ids:
        dg = d[d["ground_truth_id"] == gt_id].copy()
        if dg.empty:
            continue
        color = gt_color_map[gt_id]

        if use_cdf:
            dg_sorted = dg.sort_values("tm_score").reset_index(drop=True)
            all_x = dg_sorted["tm_score"].tolist()
            all_y = [(i + 1) / len(dg_sorted) for i in range(len(dg_sorted))]
            all_cd = dg_sorted[meta_cols + ["is_top_str"]].values.tolist()

            dg_top = dg[dg["is_top"]].sort_values("tm_score").reset_index(drop=True)
            if not dg_top.empty:
                top_x = dg_top["tm_score"].tolist()
                top_y = [(i + 1) / len(dg_top) for i in range(len(dg_top))]
                top_cd = dg_top[meta_cols + ["is_top_str"]].values.tolist()
            else:
                top_x, top_y, top_cd = [], [], []
        else:
            dg["jitter"] = np.random.uniform(-jitter, jitter, size=len(dg))
            all_x = (dg["tm_score"] + dg["jitter"]).tolist()
            all_y = [0.5] * len(dg)
            all_cd = dg[meta_cols + ["is_top_str"]].values.tolist()

            dg_top = dg[dg["is_top"]].copy()
            if not dg_top.empty:
                dg_top["jitter"] = np.random.uniform(-jitter, jitter, size=len(dg_top))
                top_x = (dg_top["tm_score"] + dg_top["jitter"]).tolist()
                top_y = [0.5] * len(dg_top)
                top_cd = dg_top[meta_cols + ["is_top_str"]].values.tolist()
            else:
                top_x, top_y, top_cd = [], [], []

        gt_data_list.append({
            "gt_id": gt_id, "color": color,
            "all_x": all_x, "all_y": all_y, "all_cd": all_cd,
            "top_x": top_x, "top_y": top_y, "top_cd": top_cd,
        })

    baseline_json = json.dumps(baseline_data)
    gt_data_json = json.dumps(gt_data_list)
    gt_colors_json = json.dumps(gt_color_map)

    if use_cdf:
        hover_tpl = (
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
            "<b>TM score ≤</b> %{x:.3f}<br>"
            "<b>Fraction:</b> %{y:.3f}<br>"
            "<extra></extra>"
        )
        mode_all = "lines"
        mode_top = "lines"
        yaxis_cfg = {"title": "Cumulative fraction", "range": [0, 1.02],
                     "tickvals": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]}
        plot_height = 700
    else:
        hover_tpl = (
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
        mode_all = "markers"
        mode_top = "markers"
        yaxis_cfg = {"showticklabels": False, "title": "", "range": [0, 1]}
        plot_height = 450

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>TM Score Distribution</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/jquery@3.7.1/dist/jquery.min.js"></script>
<link href="https://cdn.jsdelivr.net/npm/select2@4.1.0-rc.0/dist/css/select2.min.css" rel="stylesheet" />
<script src="https://cdn.jsdelivr.net/npm/select2@4.1.0-rc.0/dist/js/select2.min.js"></script>
<style>
  body {{ font-family: Arial, sans-serif; margin: 20px; }}
  .container {{ max-width: 1400px; margin: 0 auto; }}

  .controls {{
      margin-bottom: 16px; padding: 14px 16px;
      background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px;
  }}
  .controls h3 {{ margin: 0 0 10px 0; font-size: 14px; color: #495057; }}
  .controls-row {{
      display: flex; align-items: flex-start; gap: 12px; flex-wrap: wrap;
  }}
  .select-wrapper {{ flex: 1; min-width: 300px; max-width: 750px; }}
  .btn-group {{ display: flex; gap: 6px; padding-top: 4px; }}
  .btn-group button {{
      padding: 6px 14px; font-size: 13px; cursor: pointer;
      border: 1px solid #adb5bd; border-radius: 4px;
      background: #fff; color: #495057; white-space: nowrap;
  }}
  .btn-group button:hover {{ background: #e9ecef; }}

  /* Color swatches in Select2 dropdown and tags */
  .gt-swatch {{
      display: inline-block; width: 10px; height: 10px;
      border-radius: 2px; margin-right: 6px; vertical-align: middle;
      border: 1px solid rgba(0,0,0,0.15);
  }}
  /* Style selected tags with ground-truth color */
  .select2-container--default .select2-selection--multiple .select2-selection__choice {{
      border: none; color: #fff; font-size: 12px; padding: 3px 8px;
      border-radius: 3px;
  }}
  .select2-container--default .select2-selection--multiple .select2-selection__choice__remove {{
      color: rgba(255,255,255,0.7); margin-right: 4px;
  }}
  .select2-container--default .select2-selection--multiple .select2-selection__choice__remove:hover {{
      color: #fff;
  }}
  .select2-container {{ min-width: 100%; }}

  .hint {{ font-size: 12px; color: #6c757d; margin-top: 6px; }}

  #plot {{ margin-top: 12px; }}
</style>
</head>
<body>
<div class="container">
  <div class="controls">
    <h3>Highlight ground truth references</h3>
    <div class="controls-row">
      <div class="select-wrapper">
        <select id="gt-select" multiple="multiple" style="width:100%">
        </select>
        <div class="hint">
          Baseline (all data) is always shown. Select references to highlight them in color.
        </div>
      </div>
      <div class="btn-group">
        <button id="btn-all" title="Highlight all references">All</button>
        <button id="btn-none" title="Remove all highlights">None</button>
      </div>
    </div>
  </div>
  <div id="plot"></div>
</div>

<script>
const BASELINE   = {baseline_json};
const GT_DATA    = {gt_data_json};
const GT_COLORS  = {gt_colors_json};
const USE_CDF    = {'true' if use_cdf else 'false'};
const HOVER_TPL  = {json.dumps(hover_tpl)};
const MODE_ALL   = {json.dumps(mode_all)};
const MODE_TOP   = {json.dumps(mode_top)};
const YAXIS_CFG  = {json.dumps(yaxis_cfg)};
const PLOT_HEIGHT = {plot_height};
const TITLE      = {json.dumps(title)};

// Populate Select2 options
var $sel = $('#gt-select');
GT_DATA.forEach(function(gt) {{
    $sel.append(new Option(gt.gt_id, gt.gt_id, false, false));
}});

// Custom rendering for Select2 dropdown items and selected tags
function formatGtOption(opt) {{
    if (!opt.id) return opt.text;
    var color = GT_COLORS[opt.id] || '#999';
    return $('<span><span class="gt-swatch" style="background-color:' + color + '"></span>' + opt.text + '</span>');
}}

$sel.select2({{
    placeholder: 'Select references to highlight...',
    allowClear: true,
    closeOnSelect: false,
    templateResult: formatGtOption,
    templateSelection: formatGtOption,
}});

// Color the selected tags
function colorTags() {{
    $sel.next('.select2-container').find('.select2-selection__choice').each(function() {{
        // Select2 stores the value in data or in the title/text
        var text = $(this).attr('title') || $(this).text().trim();
        // Strip the "×" remove button text
        text = text.replace(/^×\\s*/, '');
        var color = GT_COLORS[text];
        if (color) {{
            $(this).css({{
                'background-color': color,
                'border-color': color,
            }});
        }}
    }});
}}

$sel.on('change', function() {{
    colorTags();
    rebuildPlot();
}});

// Select All / Clear All
$('#btn-all').on('click', function() {{
    var allVals = GT_DATA.map(function(gt) {{ return gt.gt_id; }});
    $sel.val(allVals).trigger('change');
}});
$('#btn-none').on('click', function() {{
    $sel.val([]).trigger('change');
}});

function rebuildPlot() {{
    var selected = new Set($sel.val() || []);
    var traces = [];

    // ---- Baseline: always visible ----
    if (BASELINE.all_x.length > 0) {{
        var baseAll = {{
            x: BASELINE.all_x, y: BASELINE.all_y,
            mode: MODE_ALL, name: 'All predictions (baseline)',
            customdata: BASELINE.all_cd, hovertemplate: HOVER_TPL,
            showlegend: true, opacity: 0.35,
        }};
        if (MODE_ALL === 'lines') {{
            baseAll.line = {{ color: '#4C72B0', width: 2.5 }};
        }} else {{
            baseAll.marker = {{
                color: '#4C72B0', size: 5, opacity: 0.3,
                line: {{ width: 0.5, color: '#999' }}
            }};
        }}
        traces.push(baseAll);
    }}

    if (BASELINE.top_x.length > 0) {{
        var baseTop = {{
            x: BASELINE.top_x, y: BASELINE.top_y,
            mode: MODE_TOP, name: 'Top predictions (baseline)',
            customdata: BASELINE.top_cd, hovertemplate: HOVER_TPL,
            showlegend: true, opacity: 0.35,
        }};
        if (MODE_TOP === 'lines') {{
            baseTop.line = {{ color: '#D55E00', width: 2.5, dash: 'dot' }};
        }} else {{
            baseTop.marker = {{
                color: '#D55E00', size: 7, opacity: 0.35,
                symbol: 'diamond',
                line: {{ width: 0.5, color: '#999' }}
            }};
        }}
        traces.push(baseTop);
    }}

    // ---- Highlighted ground truths ----
    GT_DATA.forEach(function(gt) {{
        if (!selected.has(gt.gt_id)) return;

        if (gt.all_x.length > 0) {{
            var trAll = {{
                x: gt.all_x, y: gt.all_y,
                mode: MODE_ALL, name: gt.gt_id + ' (all)',
                customdata: gt.all_cd, hovertemplate: HOVER_TPL,
                showlegend: true,
            }};
            if (MODE_ALL === 'lines') {{
                trAll.line = {{ color: gt.color, width: 3 }};
            }} else {{
                trAll.marker = {{
                    color: gt.color, size: 8, opacity: 0.85,
                    line: {{ width: 1, color: 'black' }}
                }};
            }}
            traces.push(trAll);
        }}

        if (gt.top_x.length > 0) {{
            var trTop = {{
                x: gt.top_x, y: gt.top_y,
                mode: MODE_TOP, name: gt.gt_id + ' (top)',
                customdata: gt.top_cd, hovertemplate: HOVER_TPL,
                showlegend: true,
            }};
            if (MODE_TOP === 'lines') {{
                trTop.line = {{ color: gt.color, width: 3, dash: 'dash' }};
            }} else {{
                trTop.marker = {{
                    color: gt.color, size: 10, opacity: 0.95,
                    symbol: 'star',
                    line: {{ width: 1.5, color: 'black' }}
                }};
            }}
            traces.push(trTop);
        }}
    }});

    var layout = {{
        title: {{ text: TITLE, x: 0.5, xanchor: 'center' }},
        xaxis: {{ title: 'TM score', range: [0, 1],
                  tickvals: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] }},
        yaxis: YAXIS_CFG,
        template: 'plotly_white',
        hovermode: 'closest',
        height: PLOT_HEIGHT,
        margin: {{ l: 70, r: 50, t: 80, b: 70 }},
        legend: {{ orientation: 'h', yanchor: 'bottom', y: 1.02,
                   xanchor: 'center', x: 0.5 }},
    }};

    Plotly.newPlot('plot', traces, layout, {{ responsive: true }});
}}

// Initial render (no highlights, just baseline)
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
    fig = go.Figure()

    hover_cols = [
        "name", "seed", "sample", "chain_i", "chain_j",
        "chain_i_desc", "chain_j_desc", "description", "is_top_str",
    ]

    if n_predictions <= 100:
        jitter = 0.01
        d["jitter"] = np.random.uniform(-jitter, jitter, size=len(d))

        fig.add_trace(go.Scatter(
            x=d["pair_iptm"] + d["jitter"],
            y=[0.5] * len(d),
            mode='markers',
            name="All predictions",
            marker=dict(
                color="#4C72B0", size=6, opacity=0.7,
                line=dict(width=0.5, color="black")
            ),
            hovertemplate=(
                "<b>name:</b> %{customdata[0]}<br>"
                "<b>description:</b> %{customdata[7]}<br>"
                "<b>pair ipTM:</b> %{x:.3f}<br>"
                "<b>seed:</b> %{customdata[1]}<br>"
                "<b>sample:</b> %{customdata[2]}<br>"
                "<b>chain i:</b> %{customdata[3]} (%{customdata[5]})<br>"
                "<b>chain j:</b> %{customdata[4]} (%{customdata[6]})<br>"
                "<b>is top:</b> %{customdata[8]}<br>"
                "<extra></extra>"
            ),
            customdata=d[hover_cols].values,
            showlegend=True,
        ))

        d_top = d[d["is_top"]].copy()
        if not d_top.empty:
            d_top["jitter"] = np.random.uniform(-jitter, jitter, size=len(d_top))
            fig.add_trace(go.Scatter(
                x=d_top["pair_iptm"] + d_top["jitter"],
                y=[0.5] * len(d_top),
                mode='markers',
                name="Top predictions",
                marker=dict(
                    color="#D55E00", size=8, opacity=0.8,
                    line=dict(width=1.5, color="black")
                ),
                hovertemplate=(
                    "<b>name:</b> %{customdata[0]}<br>"
                    "<b>description:</b> %{customdata[7]}<br>"
                    "<b>pair ipTM:</b> %{x:.3f}<br>"
                    "<b>seed:</b> %{customdata[1]}<br>"
                    "<b>sample:</b> %{customdata[2]}<br>"
                    "<b>chain i:</b> %{customdata[3]} (%{customdata[5]})<br>"
                    "<b>chain j:</b> %{customdata[4]} (%{customdata[6]})<br>"
                    "<b>is top:</b> %{customdata[8]}<br>"
                    "<extra></extra>"
                ),
                customdata=d_top[hover_cols].values,
                showlegend=True,
            ))

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            xaxis=dict(
                title="pair ipTM",
                range=[0, 1],
                tickvals=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            ),
            yaxis=dict(showticklabels=False, title="", range=[0, 1]),
            template="plotly_white",
            hovermode="closest",
            height=400,
            margin=dict(l=70, r=50, t=80, b=70),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="center", x=0.5
            ),
        )

    else:
        d_sorted = d.sort_values("pair_iptm").reset_index(drop=True)
        all_iptm = d_sorted["pair_iptm"]
        if len(all_iptm) > 0:
            y_all = [(i + 1) / len(all_iptm) for i in range(len(all_iptm))]
            fig.add_trace(go.Scatter(
                x=all_iptm, y=y_all,
                mode='lines', name="All predictions",
                line=dict(color="#4C72B0", width=2.5),
                hovertemplate=(
                    "<b>name:</b> %{customdata[0]}<br>"
                    "<b>description:</b> %{customdata[7]}<br>"
                    "<b>pair ipTM ≤</b> %{x:.2f}<br>"
                    "<b>Fraction:</b> %{y:.3f}<br>"
                    "<b>seed:</b> %{customdata[1]}<br>"
                    "<b>sample:</b> %{customdata[2]}<br>"
                    "<b>chain i:</b> %{customdata[3]} (%{customdata[5]})<br>"
                    "<b>chain j:</b> %{customdata[4]} (%{customdata[6]})<br>"
                    "<b>is top:</b> %{customdata[8]}<br>"
                    "<extra></extra>"
                ),
                customdata=d_sorted[hover_cols].values,
                showlegend=True,
            ))

        d_top = d[d["is_top"]].copy()
        if not d_top.empty:
            d_top_sorted = d_top.sort_values("pair_iptm").reset_index(drop=True)
            top_iptm = d_top_sorted["pair_iptm"]
            if len(top_iptm) > 0:
                y_top = [(i + 1) / len(top_iptm) for i in range(len(top_iptm))]
                fig.add_trace(go.Scatter(
                    x=top_iptm, y=y_top,
                    mode='lines', name="Top predictions",
                    line=dict(color="#D55E00", width=2.5, dash="solid"),
                    hovertemplate=(
                        "<b>name:</b> %{customdata[0]}<br>"
                        "<b>description:</b> %{customdata[7]}<br>"
                        "<b>pair ipTM ≤</b> %{x:.2f}<br>"
                        "<b>Fraction:</b> %{y:.3f}<br>"
                        "<b>seed:</b> %{customdata[1]}<br>"
                        "<b>sample:</b> %{customdata[2]}<br>"
                        "<b>chain i:</b> %{customdata[3]} (%{customdata[5]})<br>"
                        "<b>chain j:</b> %{customdata[4]} (%{customdata[6]})<br>"
                        "<b>is top:</b> %{customdata[8]}<br>"
                        "<extra></extra>"
                    ),
                    customdata=d_top_sorted[hover_cols].values,
                    showlegend=True,
                ))

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            xaxis=dict(
                title="pair ipTM",
                range=[0, 1],
                tickvals=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            ),
            yaxis=dict(
                title="Cumulative fraction",
                range=[0, 1.02],
                tickvals=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            ),
            template="plotly_white",
            hovermode="closest",
            height=650,
            margin=dict(l=70, r=50, t=80, b=70),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="center", x=0.5
            ),
        )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn", full_html=True)
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