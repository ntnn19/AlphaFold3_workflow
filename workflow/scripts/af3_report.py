#!/usr/bin/env python3
from __future__ import annotations
import math
import json
import re
from pathlib import Path

import click
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless for Snakemake/HPC
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple, Any
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.colors import LinearSegmentedColormap

BLUE_WHITE_CMAP = LinearSegmentedColormap.from_list(
    "blue_white_good",
    [
        "#08519c",  # dark blue
        "#3182bd",
        "#6baed6",
        "#bdd7e7",
        "#eff3ff",
        "#ffffff",  # white
    ]
)

SEED_SAMPLE_RE = re.compile(r"seed-(\d+)_sample-(\d+)")

def mark_and_filter_top(df_pred: pd.DataFrame,
                        df_chain: pd.DataFrame,
                        df_pair: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # choose best among non-"top" if present, else best overall
    d = df_pred.copy()
    cand = d[(d["prediction_id"] != "top") & d["ranking_score"].notna()].copy()
    if cand.empty:
        cand = d[d["ranking_score"].notna()].copy()

    top_pid = None
    if not cand.empty:
        top_pid = str(cand.sort_values("ranking_score", ascending=False).iloc[0]["prediction_id"])

    # add boolean label columns
    d["is_top"] = False
    if top_pid is not None:
        d.loc[d["prediction_id"].astype(str) == top_pid, "is_top"] = True

    # filter out literal "top" everywhere
    d = d[d["prediction_id"] != "top"].copy()
    df_chain2 = df_chain[df_chain["prediction_id"] != "top"].copy()
    df_pair2 = df_pair[df_pair["prediction_id"] != "top"].copy()

    # propagate is_top to chain/pair tables
    df_chain2["is_top"] = df_chain2["prediction_id"].astype(str) == str(top_pid) if top_pid is not None else False
    df_pair2["is_top"] = df_pair2["prediction_id"].astype(str) == str(top_pid) if top_pid is not None else False

    return d, df_chain2, df_pair2

def get_sample_id_from_output_dir(output_dir: Path) -> str:
    """
    Use the AF3 output directory name as sample identifier.
    Example: reports/7wr6_template_based_afdb_seed-1  ->  7wr6_template_based_afdb_seed-1
    """
    return output_dir.name

def plot_plddt_with_chain_breaks(
    conf: Dict[str, Any],
    outpath: Path,
    title: str = "pLDDT per token (with chain breaks)"
) -> bool:
    """
    Plot pLDDT values as a line plot over token index.
    Chain breaks are shown as vertical dashed lines.

    Args:
        conf: Dictionary containing confidence data, including 'atom_plddts' and 'token_chain_ids'.
        outpath: Path to save the output PNG file.
        title: Title for the plot.

    Returns:
        True if the plot was successfully saved, False otherwise.
    """
    plddt = conf.get("atom_plddts")
    tchains = conf.get("token_chain_ids")

    if plddt is None or tchains is None:
        return False

    plddt_array = np.asarray(plddt, dtype=float)
    chain_ids = np.asarray(tchains).astype(str)

    # Find boundaries where chain ID changes
    change_idx = np.nonzero(chain_ids[:-1] != chain_ids[1:])[0] + 1
    boundaries = [0, *change_idx.tolist(), len(chain_ids)]

    # Create x-axis (token indices)
    x = np.arange(len(plddt_array))

    plt.figure(figsize=(10, 5))
    ax = plt.gca()

    # Plot pLDDT as line
    ax.plot(x, plddt_array, color="blue", linewidth=1.0, alpha=0.8, label="pLDDT")

    # Draw vertical dashed lines at chain breaks
    for x_break in boundaries[1:-1]:
        ax.axvline(x_break, color="red", linestyle="--", linewidth=1.0, alpha=0.7)

    # Add chain labels on x-axis
    chain_ids_labels = []
    mids = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        chain_ids_labels.append(chain_ids[a])
        mids.append((a + b - 1) / 2)

    ax.set_xticks(mids)
    ax.set_xticklabels(chain_ids_labels, rotation=45, ha="right")

    ax.set_xlabel("Token index")
    ax.set_ylabel("pLDDT")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend()

    # Ensure layout fits
    plt.tight_layout()

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=180)
    plt.close()

    return True

def plot_plddt_combined_interactive(
    df_pred: pd.DataFrame,
    output_dir: Path,
    max_rows: int = 200
) -> str:
    """
    Generate a single interactive Plotly plot showing pLDDT vs. token index
    for all predictions, with chain breaks and color by prediction ID.

    Returns:
        Relative path to saved HTML file.
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    html_path = plots_dir / "plddt_combined.html"

    # Collect data for all predictions
    traces = []
    prediction_ids = []
    chain_breaks_all = set()

    # Group by prediction_id
    for pred_id, group in df_pred.groupby("prediction_id"):
        conf_path = group["confidences_path"].iloc[0]
        if not conf_path or not Path(conf_path).exists():
            continue

        conf = load_json(Path(conf_path))
        plddt = np.asarray(conf.get("atom_plddts", []), dtype=float)
        tchains = np.asarray(conf.get("token_chain_ids", []), dtype=str)

        if len(plddt) == 0:
            continue

        # Find chain breaks
        change_idx = np.nonzero(tchains[:-1] != tchains[1:])[0] + 1
        breaks = set(change_idx.tolist())
        chain_breaks_all.update(breaks)

        # Create x-axis
        x = np.arange(len(plddt))

        # Add trace
        traces.append({
            "x": x,
            "y": plddt,
            "name": pred_id,
            "mode": "lines",
            "line": {"width": 1.5},
            "hovertemplate": f"<b>{pred_id}</b><br>Token: %{{x}}<br>pLDDT: %{{y:.2f}}<extra></extra>",
            "showlegend": True
        })

        prediction_ids.append(pred_id)

    # If no data, return empty
    if not traces:
        html_path.write_text("<p><em>No data to plot.</em></p>", encoding="utf-8")
        return str(html_path.relative_to(output_dir))

    # Create Plotly figure
    fig = go.Figure()

    # Add all traces
    for trace in traces:
        fig.add_trace(go.Scatter(**trace))

    # Add chain break lines
    for x_break in sorted(chain_breaks_all):
        fig.add_shape(
            type="line",
            x0=x_break,
            y0=0,
            x1=x_break,
            y1=100,
            line=dict(color="red", width=1, dash="dash"),
            xref="x",
            yref="paper",
            opacity=0.6
        )

    # Update layout
    fig.update_layout(
        title={
            "text": "pLDDT per token (all predictions)",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 16}
        },
        xaxis=dict(
            title="Token index",
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray"
        ),
        yaxis=dict(
            title="pLDDT",
            range=[0, 100],
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray"
        ),
        hovermode="x unified",
        legend=dict(
            title="Prediction",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=50, r=50, t=80, b=50),
        height=600,
        template="plotly_white"
    )

    # Save as HTML
    fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)

    return str(html_path.relative_to(output_dir))

def plot_pae_interactive(df_pred: pd.DataFrame, output_dir: Path) -> str:
    """
    Lightweight "interactive" PAE viewer: dropdown switches between per-prediction PNGs.
    Expects PNGs named: plots/pae_{prediction_id}.png
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    html_path = plots_dir / "pae_interactive.html"

    pred_ids = sorted(df_pred["prediction_id"].dropna().astype(str).unique().tolist())

    # Keep only predictions that actually have a PNG
    items = []
    for pid in pred_ids:
        png = plots_dir / f"pae_{pid}.png"
        if png.exists():
            items.append((pid, f"pae_{pid}.png"))

    if not items:
        html_path.write_text("<p><em>No PAE PNGs available.</em></p>", encoding="utf-8")
        return str(html_path.relative_to(output_dir))

    first_pid, first_png = items[0]

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>PAE (image switcher)</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    .dropdown {{ margin-bottom: 16px; padding: 8px; font-size: 16px; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #eee; padding: 2px; }}
  </style>
</head>
<body>
  <div class="container">
    <h2>PAE matrices (PNG switcher)</h2>
    <select id="prediction-select" class="dropdown">
      {''.join(f'<option value="{png}">{pid}</option>' for pid, png in items)}
    </select>
    <div>
      <img id="pae-img" src="{first_png}" alt="PAE PNG">
    </div>
  </div>

  <script>
    const sel = document.getElementById('prediction-select');
    const img = document.getElementById('pae-img');
    sel.addEventListener('change', function() {{
      img.src = this.value;
    }});
  </script>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
    return str(html_path.relative_to(output_dir))

def plot_iptm_interactive(
    df_pair: pd.DataFrame,
    df_pred: pd.DataFrame,
    output_dir: Path
) -> str:
    """
    Generate an interactive ipTM matrix with dropdown to select prediction.

    Keeps the simple Plotly heatmap style of the old version, but:
    - builds matrices correctly from long-form df_pair
    - uses white->blue colormap (high = good = blue)
    - labels the top-ranked prediction
    - shows ipTM, pTM, fraction_disordered, has_clash, and ranking_score below the plot using df_pred
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    html_path = plots_dir / "iptm_interactive.html"

    required_pair = {"prediction_id", "chain_i", "chain_j", "pair_iptm"}
    if df_pair.empty or not required_pair.issubset(df_pair.columns):
        html_path.write_text("<p><em>No ipTM data available.</em></p>", encoding="utf-8")
        return str(html_path.relative_to(output_dir))

    # ---- metadata from df_pred ----
    pred_meta = {}
    top_pred_id = None

    if df_pred is not None and not df_pred.empty and "prediction_id" in df_pred.columns:
        dmeta = df_pred.copy()
        dmeta["prediction_id"] = dmeta["prediction_id"].astype(str)

        if "ranking_score" in dmeta.columns:
            dmeta["ranking_score_num"] = pd.to_numeric(dmeta["ranking_score"], errors="coerce")
            top_rows = dmeta.dropna(subset=["ranking_score_num"]).sort_values(
                ["ranking_score_num", "prediction_id"],
                ascending=[False, True]
            )
            if not top_rows.empty:
                top_pred_id = str(top_rows.iloc[0]["prediction_id"])
        else:
            dmeta["ranking_score_num"] = np.nan

        iptm_col = None
        for col in ["iptm", "ipTM"]:
            if col in dmeta.columns:
                iptm_col = col
                break

        ptm_col = None
        for col in ["ptm", "pTM"]:
            if col in dmeta.columns:
                ptm_col = col
                break

        for _, row in dmeta.iterrows():
            pid = str(row["prediction_id"])

            ranking_score = None
            if pd.notna(row.get("ranking_score_num")):
                ranking_score = float(row["ranking_score_num"])

            iptm = None
            if iptm_col is not None:
                val = pd.to_numeric(pd.Series([row.get(iptm_col)]), errors="coerce").iloc[0]
                if pd.notna(val):
                    iptm = float(val)

            ptm = None
            if ptm_col is not None:
                val = pd.to_numeric(pd.Series([row.get(ptm_col)]), errors="coerce").iloc[0]
                if pd.notna(val):
                    ptm = float(val)

            fraction_disordered = None
            if "fraction_disordered" in dmeta.columns:
                val = pd.to_numeric(pd.Series([row.get("fraction_disordered")]), errors="coerce").iloc[0]
                if pd.notna(val):
                    fraction_disordered = float(val)

            has_clash = None
            if "has_clash" in dmeta.columns:
                val = row.get("has_clash")
                if pd.notna(val):
                    has_clash = bool(val)

            pred_meta[pid] = {
                "ranking_score": ranking_score,
                "iptm": iptm,
                "ptm": ptm,
                "fraction_disordered": fraction_disordered,
                "has_clash": has_clash,
            }

    # ---- collect matrices ----
    data = {}
    prediction_ids = []

    for pred_id, group in df_pair.groupby("prediction_id"):
        pred_id = str(pred_id)

        piv = group.pivot(index="chain_i", columns="chain_j", values="pair_iptm")
        chains = sorted(set(piv.index.astype(str)).union(set(piv.columns.astype(str))))
        piv = piv.reindex(index=chains, columns=chains)

        mat = piv.to_numpy(dtype=float)
        if mat.size == 0 or np.isnan(mat).all():
            continue

        meta = pred_meta.get(pred_id, {})
        data[pred_id] = {
            "iptm": mat.tolist(),
            "chain_ids": chains,
            "score_iptm": meta.get("iptm"),
            "ptm": meta.get("ptm"),
            "fraction_disordered": meta.get("fraction_disordered"),
            "has_clash": meta.get("has_clash"),
            "ranking_score": meta.get("ranking_score"),
            "is_top": pred_id == top_pred_id,
        }
        prediction_ids.append(pred_id)

    if not data:
        html_path.write_text("<p><em>No ipTM data available.</em></p>", encoding="utf-8")
        return str(html_path.relative_to(output_dir))

    # sort with top first, then ranking_score desc, then prediction_id
    def _sort_key(pid: str):
        d = data[pid]
        is_top = d.get("is_top", False)
        rs = d.get("ranking_score")
        rs_sort = float("-inf") if rs is None else float(rs)
        return (0 if is_top else 1, -rs_sort, pid)

    prediction_ids = sorted(prediction_ids, key=_sort_key)

    options_html = "".join(
        f'<option value="{pid}">{"TOP: " if data[pid]["is_top"] else ""}{pid}</option>'
        for pid in prediction_ids
    )

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Interactive ipTM Matrices</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 24px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .dropdown {{ margin-bottom: 20px; padding: 8px; font-size: 16px; }}
            .plot {{ margin-top: 20px; }}
            .meta {{ margin-top: 10px; font-size: 15px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Interactive ipTM Matrices</h2>
            <select id="prediction-select" class="dropdown">
                {options_html}
            </select>
            <div id="plot" class="plot"></div>
            <div id="meta" class="meta"></div>
        </div>

        <script>
        const data = {json.dumps(data)};

        function updatePlot(predId) {{
            const entry = data[predId];
            const iptm = entry.iptm;
            const chain_ids = entry.chain_ids;
            const isTop = entry.is_top;
            const scoreIptm = entry.score_iptm;
            const ptm = entry.ptm;
            const fractionDisordered = entry.fraction_disordered;
            const hasClash = entry.has_clash;
            const rankingScore = entry.ranking_score;

            const trace = {{
                z: iptm,
                x: chain_ids,
                y: chain_ids,
                type: 'heatmap',
                colorscale: [
                    [0.00, '#ffffff'],
                    [0.20, '#eff3ff'],
                    [0.40, '#bdd7e7'],
                    [0.60, '#6baed6'],
                    [0.80, '#3182bd'],
                    [1.00, '#08519c']
                ],
                zmin: 0,
                zmax: 1,
                colorbar: {{ title: "ipTM" }},
                hovertemplate:
                    'Chain i: %{{y}}<br>' +
                    'Chain j: %{{x}}<br>' +
                    'pair ipTM: %{{z:.3f}}<extra></extra>'
            }};

            const titleText = isTop
                ? `ipTM Matrix - TOP: ${{predId}}`
                : `ipTM Matrix - ${{predId}}`;

            const layout = {{
                title: titleText,
                xaxis: {{ title: "Chain" }},
                yaxis: {{ title: "Chain" }},
                margin: {{ l: 50, r: 50, t: 50, b: 50 }}
            }};

            Plotly.newPlot('plot', [trace], layout, {{responsive: true}});

            const parts = [];
            if (scoreIptm !== null && scoreIptm !== undefined && !Number.isNaN(scoreIptm)) {{
                parts.push(`<strong>ipTM:</strong> ${{Number(scoreIptm).toFixed(3)}}`);
            }} else {{
                parts.push(`<strong>ipTM:</strong> n/a`);
            }}

            if (ptm !== null && ptm !== undefined && !Number.isNaN(ptm)) {{
                parts.push(`<strong>pTM:</strong> ${{Number(ptm).toFixed(3)}}`);
            }} else {{
                parts.push(`<strong>pTM:</strong> n/a`);
            }}

            if (fractionDisordered !== null && fractionDisordered !== undefined && !Number.isNaN(fractionDisordered)) {{
                parts.push(`<strong>fraction_disordered:</strong> ${{Number(fractionDisordered).toFixed(3)}}`);
            }} else {{
                parts.push(`<strong>fraction_disordered:</strong> n/a`);
            }}

            if (hasClash !== null && hasClash !== undefined) {{
                parts.push(`<strong>has_clash:</strong> ${{hasClash ? "true" : "false"}}`);
            }} else {{
                parts.push(`<strong>has_clash:</strong> n/a`);
            }}

            if (rankingScore !== null && rankingScore !== undefined && !Number.isNaN(rankingScore)) {{
                parts.push(`<strong>Ranking score:</strong> ${{Number(rankingScore).toFixed(3)}}`);
            }}

            if (isTop) {{
                parts.push(`<strong>Top-ranked sample</strong>`);
            }}

            document.getElementById('meta').innerHTML = parts.join(" &nbsp;&nbsp; ");
        }}

        document.getElementById('prediction-select').addEventListener('change', function() {{
            updatePlot(this.value);
        }});

        updatePlot('{prediction_ids[0]}');
        </script>
    </body>
    </html>
    """

    html_path.write_text(html, encoding="utf-8")
    return str(html_path.relative_to(output_dir))

def _chain_boundaries_from_token_chain_ids(tchains: np.ndarray) -> list[int]:
    if tchains.size == 0:
        return [0]
    change_idx = np.nonzero(tchains[:-1] != tchains[1:])[0] + 1
    return [0, *change_idx.tolist(), int(tchains.size)]

def plot_pae_multipanel_best_labeled(
    df_pred: pd.DataFrame,
    outpath: Path,
    ncols: int = 3,
    max_panels: int | None = None,
    vmax_percentile: float = 99.0,
) -> bool:
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from pathlib import Path

    if df_pred.empty:
        return False

    BLUE_WHITE_CMAP = LinearSegmentedColormap.from_list(
        "blue_white_good",
        [
            "#08519c",  # dark blue = good
            "#3182bd",
            "#6baed6",
            "#bdd7e7",
            "#eff3ff",
            "#ffffff",  # white = bad
        ]
    )

    d_plot = df_pred.copy()  # already filtered to exclude literal "top"

    if max_panels is not None and len(d_plot) > max_panels:
        d_plot = d_plot.sort_values(
            ["ranking_score", "prediction_id"], ascending=[False, True]
        ).head(max_panels)

    d_plot = d_plot.sort_values(
        ["is_top", "ranking_score", "prediction_id"],
        ascending=[False, False, True]
    )

    entries: list[tuple[str, bool, np.ndarray, list[int]]] = []
    all_vals = []

    for _, row in d_plot.iterrows():
        pid = str(row["prediction_id"])
        is_top = bool(row.get("is_top", False))
        conf_path = row.get("confidences_path", None)
        if not conf_path:
            continue
        p = Path(conf_path)
        if not p.exists():
            continue

        conf = load_json(p)
        pae = np.asarray(conf.get("pae", []), dtype=float)
        tchains = np.asarray(conf.get("token_chain_ids", []), dtype=str)
        if pae.size == 0 or tchains.size == 0:
            continue

        bounds = _chain_boundaries_from_token_chain_ids(tchains)
        entries.append((pid, is_top, pae, bounds))
        all_vals.append(pae[np.isfinite(pae)])

    if not entries:
        return False

    all_vals = np.concatenate(all_vals) if all_vals else np.asarray([], dtype=float)
    vmax = float(np.nanpercentile(all_vals, vmax_percentile)) if all_vals.size else None

    n = len(entries)
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(n / ncols))

    fig_w = 4.0 * ncols
    fig_h = 4.0 * nrows
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), squeeze=False)

    last_im = None
    for k, (pid, is_top, pae, bounds) in enumerate(entries):
        r, c = divmod(k, ncols)
        ax = axes[r][c]

        # low PAE = good = blue ; high PAE = bad = white
        last_im = ax.imshow(
            pae,
            cmap=BLUE_WHITE_CMAP,
            vmin=0,
            vmax=vmax,
            origin="upper",
            interpolation="nearest"
        )

        # draw black chain borders
        for x in bounds[1:-1]:
            ax.axhline(x - 0.5, color="black", lw=1.2)
            ax.axvline(x - 0.5, color="black", lw=1.2)

        # outer border
        nres = pae.shape[0]
        ax.axhline(-0.5, color="black", lw=1.2)
        ax.axhline(nres - 0.5, color="black", lw=1.2)
        ax.axvline(-0.5, color="black", lw=1.2)
        ax.axvline(nres - 0.5, color="black", lw=1.2)

        title = f"TOP: {pid}" if is_top else pid
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    for k in range(n, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r][c].axis("off")

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
        cbar.set_label("PAE (Å)  (blue=good, white=bad)")

    fig.subplots_adjust(wspace=0.15, hspace=0.25)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True

def plot_pae_with_chain_breaks(conf: dict, outpath: Path, title: str = "PAE") -> bool:
    pae = conf.get("pae", None)
    tchains = conf.get("token_chain_ids", None)
    if pae is None or tchains is None:
        return False

    pae = np.asarray(pae, dtype=float)
    tchains = np.asarray(tchains).astype(str)

    # boundaries where chain id changes
    change_idx = np.nonzero(tchains[:-1] != tchains[1:])[0] + 1
    boundaries = [0, *change_idx.tolist(), len(tchains)]

    # compute chain midpoints for tick labels
    chain_ids = []
    mids = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        chain_ids.append(tchains[a])
        mids.append((a + b - 1) / 2)

    plt.figure(figsize=(7.5, 6.5))
    ax = sns.heatmap(pae, cmap="magma", vmin=0, vmax=np.nanpercentile(pae, 99),
                     square=True, cbar_kws={"label": "PAE (Å)"})

    # draw chain break lines
    for x in boundaries[1:-1]:
        ax.axhline(x, color="white", lw=1.0)
        ax.axvline(x, color="white", lw=1.0)

    ax.set_title(title)
    ax.set_xlabel("token index")
    ax.set_ylabel("token index")

    # optional: chain labels on axes (can get crowded for many chains)
    ax.set_xticks(mids)
    ax.set_xticklabels(chain_ids, rotation=0)
    ax.set_yticks(mids)
    ax.set_yticklabels(chain_ids, rotation=0)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()
    return True

def load_json(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def pick_conf_for_plot(df_pred: pd.DataFrame) -> Path | None:
    if df_pred.empty:
        return None
    # prefer top if present
    d = df_pred.copy()
    if (d["prediction_id"] == "top").any():
        row = d[d["prediction_id"] == "top"].iloc[0]
    else:
        row = d.iloc[0]
    p = row.get("confidences_path", None)
    return Path(p) if isinstance(p, str) and p else None

def parse_prediction_id(summary_path: Path) -> tuple[int | None, int | None, str]:
    seed = sample = None
    pred_id = "top"
    for part in summary_path.parts:
        m = SEED_SAMPLE_RE.fullmatch(part)
        if m:
            seed, sample = int(m.group(1)), int(m.group(2))
            pred_id = part
            break
    return seed, sample, pred_id

def make_sample_pred_id(sample_id: str, prediction_id: str, sample: int | None) -> str:
    """
    Create a globally unique per-prediction identifier without duplicating seed info.
      - top                -> <sample_id>_sample-top
      - seed-*_sample-N    -> <sample_id>_sample-N  (N from parsed sample index)
    """
    if prediction_id == "top":
        return f"{sample_id}_sample-top"
    if sample is None:
        # Fallback (shouldn't normally happen for non-top predictions)
        return f"{sample_id}_sample-unknown"
    return f"{sample_id}_sample-{int(sample)}"

def resolve_confidences_path(summary_path: Path, layout: str) -> Path | None:
    """
    Given a summary_confidences.json path, find the matching confidences.json.
    """
    parent = summary_path.parent

    if layout == "nagarnat":  # your default layout
        # per-sample: seed-*_sample-*/summary_confidences.json -> confidences.json
        if summary_path.name == "summary_confidences.json":
            p = parent / "confidences.json"
            return p if p.exists() else None

        # top-level: <job>_summary_confidences.json -> <job>_confidences.json
        if summary_path.name.endswith("_summary_confidences.json"):
            p = parent / summary_path.name.replace("_summary_confidences.json", "_confidences.json")
            return p if p.exists() else None

        return None

    if layout == "dm":  # DeepMind canonical naming
        # per-sample and top-level both use replace convention
        if summary_path.name.endswith("_summary_confidences.json"):
            p = parent / summary_path.name.replace("_summary_confidences.json", "_confidences.json")
            return p if p.exists() else None
        return None

    raise ValueError(f"Unknown layout: {layout}")


def mean_plddt_total(conf: dict) -> float | None:
    arr = conf.get("atom_plddts")
    if arr is None:
        return None
    a = np.asarray(arr, dtype=float)
    return float(a.mean()) if a.size else None


def mean_plddt_by_chain(conf: dict) -> dict[str, float]:
    p = conf.get("atom_plddts")
    c = conf.get("atom_chain_ids")
    if p is None or c is None:
        return {}
    p = np.asarray(p, dtype=float)
    c = np.asarray(c)
    out: dict[str, float] = {}
    for chain in np.unique(c):
        mask = (c == chain)
        if mask.any():
            out[str(chain)] = float(p[mask].mean())
    return out

def find_summary_files(output_dir: Path, layout: str) -> list[Path]:
    if layout == "nagarnat":
        # includes top-level <job>_summary_confidences.json and per-sample summary_confidences.json
        a = list(output_dir.rglob("*_summary_confidences.json"))
        b = list(output_dir.rglob("summary_confidences.json"))
        return sorted(set(a + b))
    if layout == "dm":
        return sorted(output_dir.rglob("*_summary_confidences.json"))
    raise ValueError(f"Unknown layout: {layout}")


def summarize_job(output_dir: Path, layout: str):

    sample_id = output_dir.name  # e.g. 7wr6_template_based_afdb_seed-1

    summary_files = find_summary_files(output_dir, layout)

    pred_rows = []
    chain_rows = []
    pair_rows = []

    for sp in summary_files:
        seed, sample, pred_id = parse_prediction_id(sp)
        sample_pred_id = make_sample_pred_id(sample_id, pred_id, sample)
        summ = load_json(sp)

        cp = resolve_confidences_path(sp, layout)
        conf = load_json(cp) if cp and cp.exists() else {}

        # ---- complex-wide ----
        pred_rows.append({
            "sample_id": sample_id,
            "sample_pred_id": sample_pred_id,
            "prediction_id": pred_id,
            "seed": seed,
            "sample": sample,
            "ranking_score": summ.get("ranking_score"),
            "ptm": summ.get("ptm"),
            "iptm": summ.get("iptm"),
            "fraction_disordered": summ.get("fraction_disordered"),
            "has_clash": summ.get("has_clash"),
            "mean_plddt_total": mean_plddt_total(conf),
            "summary_path": str(sp),
            "confidences_path": str(cp) if cp else None,
        })

        # ---- per-chain ----
        chain_ptm = summ.get("chain_ptm", [])
        chain_iptm = summ.get("chain_iptm", [])

        chain_ids = sorted({str(x) for x in conf.get("atom_chain_ids", [])})
        n = max(len(chain_ptm), len(chain_iptm), len(chain_ids))
        if not chain_ids and n:
            chain_ids = [str(i) for i in range(n)]

        chain_mean_plddt = mean_plddt_by_chain(conf)

        for i, cid in enumerate(chain_ids):
            chain_rows.append({
                "sample_id": sample_id,
                "sample_pred_id": sample_pred_id,
                "prediction_id": pred_id,
                "seed": seed,
                "sample": sample,
                "chain_id": cid,
                "chain_ptm": chain_ptm[i] if i < len(chain_ptm) else None,
                "chain_iptm": chain_iptm[i] if i < len(chain_iptm) else None,
                "mean_plddt_chain": chain_mean_plddt.get(cid),
            })

        # ---- per chain-pair ----
        cp_iptm = summ.get("chain_pair_iptm")
        cp_pae_min = summ.get("chain_pair_pae_min")
        if isinstance(cp_iptm, list):
            mat_iptm = np.asarray(cp_iptm, dtype=float)
            mat_pae = np.asarray(cp_pae_min, dtype=float) if cp_pae_min is not None else None

            if len(chain_ids) != mat_iptm.shape[0]:
                chain_ids_pair = [str(i) for i in range(mat_iptm.shape[0])]
            else:
                chain_ids_pair = chain_ids

            for i, ci in enumerate(chain_ids_pair):
                for j, cj in enumerate(chain_ids_pair):
                    pair_rows.append({
                        "sample_id": sample_id,
                        "sample_pred_id": sample_pred_id,
                        "prediction_id": pred_id,
                        "seed": seed,
                        "sample": sample,
                        "chain_i": ci,
                        "chain_j": cj,
                        "pair_iptm": float(mat_iptm[i, j]),
                        "pair_pae_min": float(mat_pae[i, j]) if mat_pae is not None else None,
                        "is_diagonal": (i == j)
                    })

    return pd.DataFrame(pred_rows), pd.DataFrame(chain_rows), pd.DataFrame(pair_rows)

def save_fig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_complex_overview(df_pred: pd.DataFrame, outdir: Path) -> dict[str, str]:
    """Return dict: plot_name -> relative path."""
    plots = {}

    d = df_pred.copy()
    d["seed"] = d["seed"].astype("Int64")

    # ranking vs mean plddt
    plt.figure(figsize=(max(6, 0.35 * len(d)), 4.2))
    sns.stripplot(data=d, x="prediction_id", y="ranking_score", dodge=True)
    plt.xticks(rotation=60, ha="right")
    plt.xlabel("prediction")
    plt.ylabel("ranking_score")
    p = outdir / "plots" / "ranking_by_prediction.png"
    save_fig(p)
    plots["ranking_by_prediction"] = str(p.relative_to(outdir))

    # mean plddt per prediction id
    plt.figure(figsize=(max(6, 0.35 * len(d)), 4.2))
    sns.stripplot(data=d, x="prediction_id", y="mean_plddt_total", dodge=True)
    plt.xticks(rotation=60, ha="right")
    plt.xlabel("prediction")
    plt.ylabel("mean pLDDT (atom mean)")
    p = outdir / "plots" / "plddt_by_prediction.png"
    save_fig(p)
    plots["plddt_by_prediction"] = str(p.relative_to(outdir))

    return plots


def plot_chain_bars_per_prediction(
    df_chain: pd.DataFrame,
    outdir: Path,
    ncols: int = 3,
    max_panels: int | None = None
) -> str:
    """
    Generate a multipanel per-chain mean pLDDT bar plot, one panel per prediction.
    Returns relative path to saved plot.
    """
    if df_chain.empty:
        return ""

    d = df_chain.copy()

    # infer top prediction if available from ranking_score-like ordering is not possible here
    # so we only use prediction order as present unless df_chain has is_top
    if "is_top" not in d.columns:
        d["is_top"] = False

    # one row per prediction for sorting
    pred_order_df = d[["prediction_id", "is_top"]].drop_duplicates().copy()

    if max_panels is not None and len(pred_order_df) > max_panels:
        pred_order_df = pred_order_df.head(max_panels)

    pred_order_df = pred_order_df.sort_values(
        ["is_top", "prediction_id"],
        ascending=[False, True]
    )

    prediction_ids = pred_order_df["prediction_id"].astype(str).tolist()
    if not prediction_ids:
        return ""

    n = len(prediction_ids)
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(n / ncols))

    fig_w = 4.2 * ncols
    fig_h = 3.8 * nrows
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), squeeze=False)

    ymax = pd.to_numeric(d["mean_plddt_chain"], errors="coerce").max()
    if pd.isna(ymax):
        ymax = 100
    ymax = max(100, float(ymax))

    for k, pred_id in enumerate(prediction_ids):
        r, c = divmod(k, ncols)
        ax = axes[r][c]

        sub = d[d["prediction_id"].astype(str) == str(pred_id)].copy()
        if sub.empty:
            ax.axis("off")
            continue

        sub["chain_id"] = sub["chain_id"].astype(str)
        sub["mean_plddt_chain"] = pd.to_numeric(sub["mean_plddt_chain"], errors="coerce")

        sns.barplot(
            data=sub,
            x="chain_id",
            y="mean_plddt_chain",
            color="#4C72B0",
            ax=ax
        )

        is_top = bool(sub["is_top"].fillna(False).any())
        title = f"TOP: {pred_id}" if is_top else str(pred_id)

        ax.set_title(title, fontsize=9)
        ax.set_xlabel("chain")
        ax.set_ylabel("mean pLDDT")
        ax.set_ylim(0, ymax)

        # rotate labels if needed
        ax.tick_params(axis="x", rotation=45)

        # optional horizontal confidence guide lines
        ax.axhline(50, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.axhline(70, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.axhline(90, color="gray", lw=0.8, ls="--", alpha=0.5)

    for k in range(n, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r][c].axis("off")

    fig.suptitle("Per-chain mean pLDDT by prediction", fontsize=14)
    fig.subplots_adjust(wspace=0.28, hspace=0.42, top=0.90)

    p = outdir / "plots" / "chain_plddt_multipanel.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return str(p.relative_to(outdir))


def plot_pair_heatmap_per_prediction(
    df_pair: pd.DataFrame,
    outdir: Path,
    pred_id: str,
    title_suffix: str = ""
) -> str:
    """
    Generate a pair iptm heatmap for a single prediction.
    Returns relative path to saved plot.
    """
    d = df_pair[df_pair["prediction_id"] == pred_id].copy()
    if d.empty:
        return ""

    piv = d.pivot(index="chain_i", columns="chain_j", values="pair_iptm")

    plt.figure(figsize=(6, 5))
    sns.heatmap(piv, vmin=0, vmax=1, cmap="viridis", square=True, cbar_kws={"label": "pair ipTM"})
    plt.title(f"chain_pair_iptm ({pred_id}) {title_suffix}".strip())

    # Save with prediction ID in filename
    fname = f"pair_iptm_heatmap_{pred_id}.png"
    p = outdir / "plots" / fname
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(p, dpi=180)
    plt.close()

    return str(p.relative_to(outdir))


def df_to_html_table(df: pd.DataFrame, max_rows: int, table_id: str = None) -> str:
    """
    Convert a DataFrame to HTML with DataTables for interactive filtering, sorting, and pagination.
    If the table is too large, it will be truncated with a note.

    Args:
        df: DataFrame to convert.
        max_rows: Maximum rows to show (truncates if exceeded).
        table_id: Optional unique ID for the table (e.g., "chains_table").

    Returns:
        HTML string with DataTables integration.
    """
    if df.empty:
        return "<p><em>No data.</em></p>"

    d = df.copy()
    if len(d) > max_rows:
        d = d.head(max_rows)
        note = f"""
        <p><em>Showing first {max_rows} of {len(df)} rows.</em></p>
        """
    else:
        note = ""

    # Generate unique ID if not provided
    table_id = table_id or f"table_{hash(str(df.head(10))) % 1000000}"

    # Convert to HTML with DataTables
    html = f"""
    {note}
    <table id="{table_id}" class="table display" style="width:100%">
        {d.to_html(index=False, escape=True, border=0, classes="table")}
    </table>
    <script>
    $(document).ready(function() {{
        $('#{table_id}').DataTable({{
            "pageLength": 50,
            "lengthMenu": [25, 50, 100, 200],
            "searching": true,
            "ordering": true,
            "info": true,
            "autoWidth": false,
            "responsive": true,
            "language": {{
                "search": "🔍 Search:",
                "lengthMenu": "Show _MENU_ rows",
                "info": "Showing _START_ to _END_ of _TOTAL_ entries"
            }}
        }});
    }});
    </script>
    """
    return html



@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("af3_output_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("-o", "--outdir", type=click.Path(file_okay=False, path_type=Path), required=True,
              help="Output directory for report + tables.")
@click.option("--html-name", default="report.html", show_default=True, help="HTML report filename.")
@click.option("--max-rows", default=200, show_default=True, type=int, help="Max rows shown per table in HTML.")
@click.option("--write-csv/--no-write-csv", default=True, show_default=True, help="Write CSV tables.")
@click.option("--layout",type=click.Choice(["nagarnat", "dm"], case_sensitive=False), default="nagarnat", show_default=True, help="Output directory layout / AlphaFold version naming convention.")
def main(af3_output_dir: Path, outdir: Path, html_name: str, max_rows: int, write_csv: bool, layout: str):
    """
    Generate an HTML report + tables from an AlphaFold 3 output directory.

    Uses *_summary_confidences.json for metrics and *_confidences.json to compute
    mean pLDDT from Full array output atom_plddts (total + per-chain).
    """
    outdir.mkdir(parents=True, exist_ok=True)

    df_pred, df_chain, df_pair = summarize_job(af3_output_dir, layout=layout.lower())
    df_pred, df_chain, df_pair = mark_and_filter_top(df_pred, df_chain, df_pair)

    if write_csv:
        df_pred.to_csv(outdir / "predictions.tsv", sep="\t", index=False)
        df_chain.to_csv(outdir / "chains.tsv", sep="\t", index=False)
        df_pair.to_csv(outdir / "chain_pairs.tsv", sep="\t", index=False)

    # --- Generate all plots per prediction ---
    plots = {}

    # 1. Complex-wide plots (already per-prediction)
    plots.update(plot_complex_overview(df_pred, outdir))

    # 2. Per-prediction chain bar plots
    chain_multi_path = plot_chain_bars_per_prediction(df_chain, outdir, ncols=3)
    if chain_multi_path:
        plots["chain_plddt_multipanel"] = chain_multi_path



    # 4. Per-prediction PAE plots
    pae_multi = outdir / "plots" / "pae_multipanel.png"
    if plot_pae_multipanel_best_labeled(df_pred, pae_multi, ncols=3):
        plots["pae_multipanel"] = str(pae_multi.relative_to(outdir))


    combined_plot_path = plot_plddt_combined_interactive(df_pred, outdir, max_rows=max_rows)
    plots["plddt_combined"] = combined_plot_path


    # 7. Generate **interactive ipTM matrix** (dropdown)
    iptm_interactive_path = plot_iptm_interactive(df_pair, df_pred, outdir)
    plots["iptm_interactive"] = iptm_interactive_path
    print("Plots=",plots)



if __name__ == "__main__":
    main()

# chain borders for iptm and pae matrices should be clearly visualized using black lines.
# color scales for pae and iptm should be red to blue (blue is good, red is bad)
