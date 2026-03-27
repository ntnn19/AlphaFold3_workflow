#!/usr/bin/env python3
from __future__ import annotations

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

SEED_SAMPLE_RE = re.compile(r"seed-(\d+)_sample-(\d+)")

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

def plot_pae_interactive(
    df_pred: pd.DataFrame,
    output_dir: Path
) -> str:
    """
    Generate an interactive PAE matrix with dropdown to select prediction.
    Uses Plotly for interactivity.

    Returns:
        Relative path to saved HTML file.
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    html_path = plots_dir / "pae_interactive.html"

    # Collect data for all predictions
    data = {}
    prediction_ids = []

    for pred_id, group in df_pred.groupby("prediction_id"):
        conf_path = group["confidences_path"].iloc[0]
        if not conf_path or not Path(conf_path).exists():
            continue

        conf = load_json(Path(conf_path))
        pae = np.asarray(conf.get("pae", []), dtype=float)
        tchains = np.asarray(conf.get("token_chain_ids", []), dtype=str)

        if pae.size == 0 or len(tchains) == 0:
            continue

        # Compute chain boundaries
        change_idx = np.nonzero(tchains[:-1] != tchains[1:])[0] + 1
        boundaries = [0, *change_idx.tolist(), len(tchains)]

        # Get chain IDs
        chain_ids = []
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            chain_ids.append(tchains[a])

        # Store data
        data[pred_id] = {
            "pae": pae,
            "chain_ids": chain_ids
        }
        prediction_ids.append(pred_id)

    if not data:
        html_path.write_text("<p><em>No PAE data available.</em></p>", encoding="utf-8")
        return str(html_path.relative_to(output_dir))

    # Sort prediction IDs
    prediction_ids.sort()

    # Create HTML with dropdown and Plotly
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Interactive PAE Matrices</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 24px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .dropdown {{ margin-bottom: 20px; padding: 8px; font-size: 16px; }}
            .plot {{ margin-top: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Interactive PAE Matrices</h2>
            <select id="prediction-select" class="dropdown">
                {"".join(f'<option value="{pid}">{pid}</option>' for pid in prediction_ids)}
            </select>
            <div id="plot" class="plot"></div>
        </div>

        <script>
        // Data
        const data = {json.dumps(data, default=str)};

        // Initial render
        function updatePlot(predId) {{
            const pae = data[predId].pae;
            const chain_ids = data[predId].chain_ids;

            // Create heatmap
            const trace = {{
                z: pae,
                x: chain_ids,
                y: chain_ids,
                type: 'heatmap',
                colorscale: 'magma',
                zmin: 0,
                zmax: Math.max(...pae.flat()),
                colorbar: {{ title: "PAE (Å)" }}
            }};

            const layout = {{
                title: `PAE Matrix - ${{predId}}`,
                xaxis: {{ title: "Token index" }},
                yaxis: {{ title: "Token index" }},
                margin: {{ l: 50, r: 50, t: 50, b: 50 }}
            }};

            Plotly.newPlot('plot', [trace], layout);
        }}

        // On load
        document.getElementById('prediction-select').addEventListener('change', function() {{
            updatePlot(this.value);
        }});

        // Initial render
        updatePlot('{prediction_ids[0]}');
        </script>
    </body>
    </html>
    """

    html_path.write_text(html, encoding="utf-8")
    return str(html_path.relative_to(output_dir))

def plot_iptm_interactive(
    df_pair: pd.DataFrame,
    output_dir: Path
) -> str:
    """
    Generate an interactive ipTM matrix with dropdown to select prediction.
    Uses Plotly for interactivity.

    Returns:
        Relative path to saved HTML file.
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    html_path = plots_dir / "iptm_interactive.html"

    # Collect data for all predictions
    data = {}
    prediction_ids = []

    for pred_id, group in df_pair.groupby("prediction_id"):
        # Get the first row (or any) for this prediction
        row = group.iloc[0]
        if pd.isna(row["pair_iptm"]) or row["pair_iptm"] is None:
            continue

        # Get the full matrix
        cp_iptm = row["pair_iptm"]
        if isinstance(cp_iptm, list):
            mat = np.asarray(cp_iptm, dtype=float)
        else:
            continue

        # Get chain IDs
        chain_ids = sorted(set(row["chain_i"].astype(str) + row["chain_j"].astype(str)))

        # Store data
        data[pred_id] = {
            "iptm": mat,
            "chain_ids": chain_ids
        }
        prediction_ids.append(pred_id)

    if not data:
        html_path.write_text("<p><em>No ipTM data available.</em></p>", encoding="utf-8")
        return str(html_path.relative_to(output_dir))

    # Sort prediction IDs
    prediction_ids.sort()

    # Create HTML with dropdown and Plotly
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
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Interactive ipTM Matrices</h2>
            <select id="prediction-select" class="dropdown">
                {"".join(f'<option value="{pid}">{pid}</option>' for pid in prediction_ids)}
            </select>
            <div id="plot" class="plot"></div>
        </div>

        <script>
        // Data
        const data = {json.dumps(data, default=str)};

        // Initial render
        function updatePlot(predId) {{
            const iptm = data[predId].iptm;
            const chain_ids = data[predId].chain_ids;

            // Create heatmap
            const trace = {{
                z: iptm,
                x: chain_ids,
                y: chain_ids,
                type: 'heatmap',
                colorscale: 'viridis',
                zmin: 0,
                zmax: 1,
                colorbar: {{ title: "ipTM" }}
            }};

            const layout = {{
                title: `ipTM Matrix - ${{predId}}`,
                xaxis: {{ title: "Chain" }},
                yaxis: {{ title: "Chain" }},
                margin: {{ l: 50, r: 50, t: 50, b: 50 }}
            }};

            Plotly.newPlot('plot', [trace], layout);
        }}

        // On load
        document.getElementById('prediction-select').addEventListener('change', function() {{
            updatePlot(this.value);
        }});

        // Initial render
        updatePlot('{prediction_ids[0]}');
        </script>
    </body>
    </html>
    """

    html_path.write_text(html, encoding="utf-8")
    return str(html_path.relative_to(output_dir))

def plot_pae_with_chain_breaks(conf: dict, outpath: Path, pred_id: str, title: str = "PAE") -> bool:
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

    # Use full pred_id in title
    ax.set_title(f"PAE (with chain breaks) - {pred_id}")

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

def parse_prediction_id(summary_path: Path) -> tuple[int | None, int | None, str, str]:
    """
    Returns: (seed, sample, prediction_id, sample_id)
    """
    seed = sample = None
    sample_id = summary_path.parent.name  # e.g., "7wr6_template_based_afdb"

    # Try to extract seed/sample from path
    for part in summary_path.parts:
        m = SEED_SAMPLE_RE.fullmatch(part)
        if m:
            seed, sample = int(m.group(1)), int(m.group(2))
            # Create full prediction_id: seed-<seed>_sample-<sample>_<sample_id>
            pred_id = f"seed-{seed}_sample-{sample}_{sample_id}"
            return seed, sample, pred_id, sample_id

    # If no seed/sample found, it's top-level
    # Use "top" + seed/sample from parent dir if possible
    # But we don't have seed/sample — so we can't.
    # So use: top_<sample_id>
    pred_id = f"top_{sample_id}"
    return None, None, pred_id, sample_id



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

    summary_files = find_summary_files(output_dir, layout)

    pred_rows = []
    chain_rows = []
    pair_rows = []

    for sp in summary_files:
        seed, sample, pred_id, sample_id = parse_prediction_id(sp)
        summ = load_json(sp)

        cp = resolve_confidences_path(sp, layout)
        conf = load_json(cp) if cp and cp.exists() else {}

        # ---- complex-wide ----
        pred_rows.append({
            "prediction_id": pred_id,  # e.g., seed-1_sample-0_7wr6_template_based_afdb
            "seed": seed,
            "sample": sample,
            "sample_id": sample_id,  # ← new column
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
                "prediction_id": pred_id,
                "seed": seed,
                "sample": sample,
                "sample_id": sample_id,
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
                        "prediction_id": pred_id,
                        "seed": seed,
                        "sample": sample,
                        "sample_id": sample_id,
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
    pred_id: str
) -> str:
    d = df_chain[df_chain["prediction_id"] == pred_id].copy()
    if d.empty:
        return ""

    plt.figure(figsize=(max(6, 0.6 * len(d)), 4.2))
    sns.barplot(data=d, x="chain_id", y="mean_plddt_chain", color="#4C72B0")
    plt.title(f"Per-chain mean pLDDT ({pred_id})")
    plt.xlabel("chain")
    plt.ylabel("mean pLDDT (atom mean)")

    fname = f"chain_plddt_{pred_id}.png"
    p = outdir / "plots" / fname
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(p, dpi=180)
    plt.close()

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


def write_html_report(out_html: Path,
                      df_pred: pd.DataFrame,
                      df_chain: pd.DataFrame,
                      df_pair: pd.DataFrame,
                      max_rows: int):
    css = """
    <style>
    body { font-family: Arial, sans-serif; margin: 24px; }
    h1,h2 { margin-top: 1.2em; }
    .table { border-collapse: collapse; font-size: 13px; }
    .table th, .table td { border: 1px solid #ddd; padding: 6px 8px; }
    .table th { background: #f5f5f5; }
    img { max-width: 100%; height: auto; border: 1px solid #eee; padding: 2px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    </style>
    """

    def img(tag):
        if tag in plots:
            return f'<img src="{plots[tag]}"/>'
        return "<p><em>Plot not available.</em></p>"

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>AlphaFold3 report</title>{css}</head>
<body>
<h1>AlphaFold 3 summary report</h1>

<h2>Complex-wide metrics (per prediction)</h2>
{df_to_html_table(df_pred.sort_values(["prediction_id"], ascending=True), max_rows=max_rows)}

<h2>Per-chain metrics</h2>
{img("chain_plddt_bar")}
{df_to_html_table(df_chain.sort_values(["prediction_id","chain_id"]), max_rows=max_rows)}

<h2>Per-interface (chain-pair) metrics</h2>
{img("pair_iptm_heatmap")}
{df_to_html_table(df_pair.sort_values(["prediction_id","pair_iptm"], ascending=[True, False]), max_rows=max_rows)}
</body></html>
"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")


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

    if write_csv:
        df_pred.to_csv(outdir / "predictions.csv", index=False)
        df_chain.to_csv(outdir / "chains.csv", index=False)
        df_pair.to_csv(outdir / "chain_pairs.csv", index=False)

    # --- Generate all plots per prediction ---
    # --- Generate all plots per prediction ---
    # --- Generate all plots per prediction ---
    plots = {}

    # 1. Complex-wide plots (already per-prediction)
    plots.update(plot_complex_overview(df_pred, outdir))

    # 2. Per-prediction chain bar plots
    for pred_id in df_pred["prediction_id"].unique():
        path = plot_chain_bars_per_prediction(df_chain, outdir, pred_id)
        if path:
            plots[f"chain_plddt_{pred_id}"] = path

    # 3. Per-prediction pair heatmap plots
    for pred_id in df_pred["prediction_id"].unique():
        path = plot_pair_heatmap_per_prediction(df_pair, outdir, pred_id)
        if path:
            plots[f"pair_iptm_heatmap_{pred_id}"] = path

    # 4. Per-prediction PAE plots
    # 4. Per-prediction PAE plots
    for pred_id in df_pred["prediction_id"].unique():
        conf_path = df_pred[df_pred["prediction_id"] == pred_id]["confidences_path"].iloc[0]
        if not conf_path or not Path(conf_path).exists():
            continue

        conf = load_json(Path(conf_path))

        # PAE matrix
        pae_png = outdir / "plots" / f"pae_{pred_id}.png"
        ok_pae = plot_pae_with_chain_breaks(conf, pae_png, pred_id=pred_id)
        if ok_pae:
            plots[f"pae_{pred_id}"] = str(pae_png.relative_to(outdir))

    # 5. Generate **one interactive combined pLDDT plot** for all predictions
    combined_plot_path = plot_plddt_combined_interactive(df_pred, outdir, max_rows=max_rows)
    plots["plddt_combined"] = combined_plot_path

    # 6. Generate **interactive PAE matrix** (dropdown)
    pae_interactive_path = plot_pae_interactive(df_pred, outdir)
    plots["pae_interactive"] = pae_interactive_path

    # 7. Generate **interactive ipTM matrix** (dropdown)
    iptm_interactive_path = plot_iptm_interactive(df_pair, outdir)
    plots["iptm_interactive"] = iptm_interactive_path
    print("Plots=",plots)
    #out_html = outdir / html_name
    #write_html_report(out_html, df_pred, df_chain, df_pair, max_rows=max_rows)

    #click.echo(str(out_html))


if __name__ == "__main__":
    main()
