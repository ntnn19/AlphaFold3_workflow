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

    # --- Baseline data (all ground truths combined) ---
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
        jitter_val = 0.01
        d["jitter"] = np.random.uniform(-jitter_val, jitter_val, size=len(d))
        baseline_all_x = (d["tm_score"] + d["jitter"]).tolist()
        baseline_all_y = [0.5] * len(d)
        baseline_all_cd = d[meta_cols + ["is_top_str"]].values.tolist()

        d_top = d[d["is_top"]].copy()
        if not d_top.empty:
            d_top["jitter"] = np.random.uniform(-jitter_val, jitter_val, size=len(d_top))
            baseline_top_x = (d_top["tm_score"] + d_top["jitter"]).tolist()
            baseline_top_y = [0.5] * len(d_top)
            baseline_top_cd = d_top[meta_cols + ["is_top_str"]].values.tolist()
        else:
            baseline_top_x, baseline_top_y, baseline_top_cd = [], [], []

    baseline_data = {
        "all_x": baseline_all_x, "all_y": baseline_all_y, "all_cd": baseline_all_cd,
        "top_x": baseline_top_x, "top_y": baseline_top_y, "top_cd": baseline_top_cd,
    }

    # --- Per-ground-truth overlay data ---
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
            dg["jitter"] = np.random.uniform(-jitter_val, jitter_val, size=len(dg))
            all_x = (dg["tm_score"] + dg["jitter"]).tolist()
            all_y = [0.5] * len(dg)
            all_cd = dg[meta_cols + ["is_top_str"]].values.tolist()

            dg_top = dg[dg["is_top"]].copy()
            if not dg_top.empty:
                dg_top["jitter"] = np.random.uniform(-jitter_val, jitter_val, size=len(dg_top))
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

    # Build <option> tags for the native select
    n_visible = min(max(len(gt_ids), 2), 8)
    options_html = "\n".join(
        f'          <option value="{gt}" selected '
        f'style="padding:3px 6px;">'
        f'&#9632; {gt}</option>'
        for gt in gt_ids
    )

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
    </div>
    <div class="hint">
      Click to toggle. Baseline (all data) is always shown.
    </div>
  </div>

  <div id="plot"></div>
  <div class="legend-note">
    Solid = all predictions &nbsp;|&nbsp; {"Dashed" if use_cdf else "★"} = top predictions
  </div>
</div>

<script>
var BASELINE   = {baseline_json};
var GT_DATA    = {gt_data_json};
var USE_CDF    = {'true' if use_cdf else 'false'};
var HOVER_TPL  = {json.dumps(hover_tpl)};
var MODE_ALL   = {json.dumps(mode_all)};
var MODE_TOP   = {json.dumps(mode_top)};
var YAXIS_CFG  = {json.dumps(yaxis_cfg)};
var PLOT_HEIGHT = {plot_height};

// --- Toggle selection on plain click (no ctrl needed) ---
var sel = document.getElementById('gt-select');
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

    // ---- Baseline: always visible ----
    if (BASELINE.all_x.length > 0) {{
        var baseAll = {{
            x: BASELINE.all_x, y: BASELINE.all_y,
            mode: MODE_ALL,
            name: 'All (baseline)',
            customdata: BASELINE.all_cd,
            hovertemplate: HOVER_TPL,
            showlegend: false,
        }};
        if (MODE_ALL === 'lines') {{
            baseAll.line = {{ color: 'rgba(76,114,176,0.3)', width: 2.5 }};
        }} else {{
            baseAll.marker = {{
                color: 'rgba(76,114,176,0.25)', size: 5,
                line: {{ width: 0.5, color: 'rgba(0,0,0,0.15)' }}
            }};
        }}
        traces.push(baseAll);
    }}
    if (BASELINE.top_x.length > 0) {{
        var baseTop = {{
            x: BASELINE.top_x, y: BASELINE.top_y,
            mode: MODE_TOP,
            name: 'Top (baseline)',
            customdata: BASELINE.top_cd,
            hovertemplate: HOVER_TPL,
            showlegend: false,
        }};
        if (MODE_TOP === 'lines') {{
            baseTop.line = {{ color: 'rgba(213,94,0,0.25)', width: 2, dash: 'dot' }};
        }} else {{
            baseTop.marker = {{
                color: 'rgba(213,94,0,0.25)', size: 7, symbol: 'diamond',
                line: {{ width: 0.5, color: 'rgba(0,0,0,0.15)' }}
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
                mode: MODE_ALL,
                name: gt.gt_id,
                customdata: gt.all_cd,
                hovertemplate: HOVER_TPL,
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
                mode: MODE_TOP,
                name: gt.gt_id + ' (top)',
                customdata: gt.top_cd,
                hovertemplate: HOVER_TPL,
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
        xaxis: {{ title: 'TM score', range: [0, 1],
                  tickvals: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] }},
        yaxis: YAXIS_CFG,
        template: 'plotly_white',
        hovermode: 'closest',
        height: PLOT_HEIGHT,
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