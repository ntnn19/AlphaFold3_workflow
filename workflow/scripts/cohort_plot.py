#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Optional
import click
import pandas as pd
import plotly.graph_objects as go


def load_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\s+", dtype=str).fillna("")


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    d = df.copy()
    for c in cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d

def plot_tm_score_distribution(
    df_pred: pd.DataFrame,
    out_html: Path,
    title: str = "Distribution of normalized TM scores across all predictions",
) -> bool:
    required = {"sample_id", "prediction_id", "TM1", "TM2"}
    print(df_pred)
    print(required.issubset(df_pred.columns))
    exit()
    if df_pred.empty or not required.issubset(df_pred.columns):
        write_no_data_html(out_html, "No TM score data available.")
        return False

    d = df_pred.copy()
    d["sample_id"] = d["sample_id"].astype(str)
    d["prediction_id"] = d["prediction_id"].astype(str)

    # Extract name from sample_id
    d["name"] = d["sample_id"].str.split("_seed-").str[0]
    d["name"] = d["name"].astype(str).replace("nan", "N/A")

    # Extract seed and sample
    d["seed"] = d["sample_id"].str.extract(r"_seed-(\d+)", expand=False).fillna("N/A")
    d["sample"] = d["sample_id"].str.extract(r"_sample-(\d+)", expand=False).fillna("N/A")

    # Normalize TM score: use min(TM1, TM2) as the effective TM score
    # This is standard in structural biology: TM score is limited by the shorter chain
    d["tm_score"] = d[["tm1", "tm2"]].min(axis=1)
    d["tm_score"] = pd.to_numeric(d["tm_score"], errors="coerce")

    # Drop invalid values
    d = d[d["tm_score"].notna()].copy()
    if d.empty:
        write_no_data_html(out_html, "No valid TM scores available.")
        return False

    # Add is_top as string
    d["is_top"] = d["is_top"].astype(str).str.title()

    # Add metadata columns
    meta_cols = [
        "sample", "seed", "name", "ranking_score", "ptm", "iptm",
        "mean_plddt_total", "fraction_disordered", "has_clash"
    ]
    available_meta = [c for c in meta_cols if c in d.columns]
    d = d[["tm_score"] + available_meta + ["is_top"]].copy()

    n_predictions = len(d)

    fig = go.Figure()

    if n_predictions <= 100:
        # Strip chart: jitter on x-axis
        jitter = 0.01
        d["jitter"] = np.random.uniform(-jitter, jitter, size=len(d))

        # All predictions
        fig.add_trace(go.Scatter(
            x=d["tm_score"] + d["jitter"],
            y=[0.5] * len(d),
            mode='markers',
            name="All predictions",
            marker=dict(
                color="#4C72B0",
                size=6,
                opacity=0.7,
                line=dict(width=0.5, color="black")
            ),
            hovertemplate=(
                "<b>name:</b> %{customdata[0]}<br>"
                "<b>sample:</b> %{customdata[1]}<br>"
                "<b>seed:</b> %{customdata[2]}<br>"
                "<b>ranking score:</b> %{customdata[3]:.3f}<br>"
                "<b>ptm:</b> %{customdata[4]:.3f}<br>"
                "<b>iptm:</b> %{customdata[5]:.3f}<br>"
                "<b>mean pLDDT:</b> %{customdata[6]:.2f}<br>"
                "<b>fraction disordered:</b> %{customdata[7]:.3f}<br>"
                "<b>has clash:</b> %{customdata[8]}<br>"
                "<b>is top:</b> %{customdata[9]}<br>"
                "<b>normalized TM:</b> %{x:.3f}<br>"
                "<extra></extra>"
            ),
            customdata=d[meta_cols + ["is_top"]].values,
            showlegend=True,
        ))

        # Top predictions
        d_top = d[d["is_top"].str.lower() == "true"]
        if not d_top.empty:
            d_top["jitter"] = np.random.uniform(-jitter, jitter, size=len(d_top))
            fig.add_trace(go.Scatter(
                x=d_top["tm_score"] + d_top["jitter"],
                y=[0.5] * len(d_top),
                mode='markers',
                name="Top predictions",
                marker=dict(
                    color="#D55E00",
                    size=8,
                    opacity=0.8,
                    line=dict(width=1.5, color="black")
                ),
                hovertemplate=(
                    "<b>name:</b> %{customdata[0]}<br>"
                    "<b>sample:</b> %{customdata[1]}<br>"
                    "<b>seed:</b> %{customdata[2]}<br>"
                    "<b>ranking score:</b> %{customdata[3]:.3f}<br>"
                    "<b>ptm:</b> %{customdata[4]:.3f}<br>"
                    "<b>iptm:</b> %{customdata[5]:.3f}<br>"
                    "<b>mean pLDDT:</b> %{customdata[6]:.2f}<br>"
                    "<b>fraction disordered:</b> %{customdata[7]:.3f}<br>"
                    "<b>has clash:</b> %{customdata[8]}<br>"
                    "<b>is top:</b> %{customdata[9]}<br>"
                    "<b>normalized TM:</b> %{x:.3f}<br>"
                    "<extra></extra>"
                ),
                customdata=d_top[meta_cols + ["is_top"]].values,
                showlegend=True,
            ))

        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor="center"
            ),
            xaxis=dict(
                title="Normalized TM score (min(TM1, TM2))",
                range=[0, 1],
                tickvals=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                ticktext=["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
            ),
            yaxis=dict(
                showticklabels=True,
                title="Prediction",
                tickvals=[0.5],
                ticktext=[""],
                range=[0, 1]
            ),
            template="plotly_white",
            hovermode="x unified",
            height=400,
            margin=dict(l=70, r=50, t=80, b=70),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
        )

    else:
        # CDF: smooth line
        all_tm = d["tm_score"].dropna().sort_values()
        if len(all_tm) > 0:
            y_all = [i / len(all_tm) for i in range(1, len(all_tm) + 1)]
            fig.add_trace(go.Scatter(
                x=all_tm,
                y=y_all,
                mode='lines',
                name="All predictions",
                line=dict(color="#4C72B0", width=2.5),
                hovertemplate=(
                    "<b>name:</b> %{customdata[0]}<br>"
                    "<b>sample:</b> %{customdata[1]}<br>"
                    "<b>seed:</b> %{customdata[2]}<br>"
                    "<b>ranking score:</b> %{customdata[3]:.3f}<br>"
                    "<b>ptm:</b> %{customdata[4]:.3f}<br>"
                    "<b>iptm:</b> %{customdata[5]:.3f}<br>"
                    "<b>mean pLDDT:</b> %{customdata[6]:.2f}<br>"
                    "<b>fraction disordered:</b> %{customdata[7]:.3f}<br>"
                    "<b>has clash:</b> %{customdata[8]}<br>"
                    "<b>is top:</b> %{customdata[9]}<br>"
                    "<b>normalized TM ≤</b> %{x:.2f}<br>"
                    "<b>Fraction:</b> %{y:.3f}<br>"
                    "<extra></extra>"
                ),
                customdata=d[meta_cols + ["is_top"]].values,
                showlegend=True,
            ))

        d_top = d[d["is_top"].str.lower() == "true"]
        if not d_top.empty:
            top_tm = d_top["tm_score"].dropna().sort_values()
            if len(top_tm) > 0:
                y_top = [i / len(top_tm) for i in range(1, len(top_tm) + 1)]
                fig.add_trace(go.Scatter(
                    x=top_tm,
                    y=y_top,
                    mode='lines',
                    name="Top predictions",
                    line=dict(color="#D55E00", width=2.5, dash="solid"),
                    hovertemplate=(
                        "<b>name:</b> %{customdata[0]}<br>"
                        "<b>sample:</b> %{customdata[1]}<br>"
                        "<b>seed:</b> %{customdata[2]}<br>"
                        "<b>ranking score:</b> %{customdata[3]:.3f}<br>"
                        "<b>ptm:</b> %{customdata[4]:.3f}<br>"
                        "<b>iptm:</b> %{customdata[5]:.3f}<br>"
                        "<b>mean pLDDT:</b> %{customdata[6]:.2f}<br>"
                        "<b>fraction disordered:</b> %{customdata[7]:.3f}<br>"
                        "<b>has clash:</b> %{customdata[8]}<br>"
                        "<b>is top:</b> %{customdata[9]}<br>"
                        "<b>normalized TM ≤</b> %{x:.2f}<br>"
                        "<b>Fraction:</b> %{y:.3f}<br>"
                        "<extra></extra>"
                    ),
                    customdata=d_top[meta_cols + ["is_top"]].values,
                    showlegend=True,
                ))

        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor="center"
            ),
            xaxis=dict(
                title="Normalized TM score (min(TM1, TM2))",
                range=[0, 1],
                tickvals=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                ticktext=["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
            ),
            yaxis=dict(
                title="Cumulative fraction",
                range=[0, 1.02],
                tickvals=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                ticktext=["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
            ),
            template="plotly_white",
            hovermode="x unified",
            height=650,
            margin=dict(l=70, r=50, t=80, b=70),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
        )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn", full_html=True)
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

    if df_pred.empty or not required.issubset(df_pred.columns):
        return d

    p = df_pred.copy()
    p["sample_id"] = p["sample_id"].astype(str)
    p["prediction_id"] = p["prediction_id"].astype(str)

    keep = [
        "sample_id",
        "prediction_id",
        "is_top",
        "ranking_score",
        "iptm",
        "ptm",
        "mean_plddt_total",
    ]
    keep = [c for c in keep if c in p.columns]
    p = p[keep].drop_duplicates()

    d = d.merge(p, on=["sample_id", "prediction_id"], how="left")

    if "is_top" in d.columns:
        d["is_top"] = d["is_top"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        d["is_top"] = False

    for c in ["ranking_score", "iptm", "ptm", "mean_plddt_total"]:
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


import numpy as np  # Make sure this is at the top of the file

def plot_chain_pair_iptm_cumulative(
    df_pair: pd.DataFrame,
    out_html: Path,
    title: str = "Cumulative distribution of chain-pair ipTM across all predictions",
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

    # Extract name from sample_id: e.g., "8sm3_template_free_afdb_seed-1" → "8sm3_template_free_afdb"
    d["name"] = d["sample_id"].str.split("_seed-").str[0]
    d["name"] = d["name"].astype(str).replace("nan", "N/A")

    # Extract seed and sample (if available)
    d["seed"] = d["sample_id"].str.extract(r"_seed-(\d+)", expand=False).fillna("N/A")
    d["sample"] = d["sample_id"].str.extract(r"_sample-(\d+)", expand=False).fillna("N/A")

    # Ensure chain_i, chain_j are strings
    for c in ["chain_i", "chain_j"]:
        if c in d.columns:
            d[c] = d[c].astype(str).replace("nan", "N/A")
        else:
            d[c] = "N/A"

    # is_top as string
    d["is_top"] = d["is_top"].astype(str).str.title()

    n_predictions = len(d)

    fig = go.Figure()

    if n_predictions <= 100:
        # --- Jittered strip chart for small datasets ---
        jitter = 0.01  # Increased jitter for better visibility
        d["jitter"] = np.random.uniform(-jitter, jitter, size=len(d))

        # All predictions
        fig.add_trace(go.Scatter(
            x=d["pair_iptm"] + d["jitter"],
            y=[0.5] * len(d),
            mode='markers',
            name="All predictions",
            marker=dict(
                color="#4C72B0",
                size=6,
                opacity=0.7,
                line=dict(width=0.5, color="black")
            ),
            hovertemplate=(
                "<b>name:</b> %{customdata[0]}<br>"
                "<b>pair ipTM:</b> %{x:.3f}<br>"
                "<b>seed:</b> %{customdata[1]}<br>"
                "<b>sample:</b> %{customdata[2]}<br>"
                "<b>chain i:</b> %{customdata[3]}<br>"
                "<b>chain j:</b> %{customdata[4]}<br>"
                "<b>is top:</b> %{customdata[5]}<br>"
                "<extra></extra>"
            ),
            customdata=d[["name", "seed", "sample", "chain_i", "chain_j", "is_top"]].values,
            showlegend=True,
        ))

        # Top predictions
        d_top = d[d["is_top"].str.lower() == "true"]
        if not d_top.empty:
            d_top["jitter"] = np.random.uniform(-jitter, jitter, size=len(d_top))
            fig.add_trace(go.Scatter(
                x=d_top["pair_iptm"] + d_top["jitter"],
                y=[0.5] * len(d_top),
                mode='markers',
                name="Top predictions",
                marker=dict(
                    color="#D55E00",
                    size=8,
                    opacity=0.8,
                    line=dict(width=1.5, color="black")
                ),
                hovertemplate=(
                    "<b>name:</b> %{customdata[0]}<br>"
                    "<b>pair ipTM:</b> %{x:.3f}<br>"
                    "<b>seed:</b> %{customdata[1]}<br>"
                    "<b>sample:</b> %{customdata[2]}<br>"
                    "<b>chain i:</b> %{customdata[3]}<br>"
                    "<b>chain j:</b> %{customdata[4]}<br>"
                    "<b>is top:</b> %{customdata[5]}<br>"
                    "<extra></extra>"
                ),
                customdata=d_top[["name", "seed", "sample", "chain_i", "chain_j", "is_top"]].values,
                showlegend=True,
            ))

        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor="center"
            ),
            xaxis=dict(
                title="pair ipTM",
                range=[0, 1],
                tickvals=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                ticktext=["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
            ),
            yaxis=dict(
                showticklabels=False,
                title="",
                range=[0, 1]
            ),
            template="plotly_white",
            hovermode="x unified",
            height=400,
            margin=dict(l=70, r=50, t=80, b=70),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
        )

    else:
        # --- Smooth line-based CDF for large datasets ---
        all_iptm = d["pair_iptm"].dropna().sort_values()
        if len(all_iptm) > 0:
            y_all = [i / len(all_iptm) for i in range(1, len(all_iptm) + 1)]
            fig.add_trace(go.Scatter(
                x=all_iptm,
                y=y_all,
                mode='lines',
                name="All predictions",
                line=dict(color="#4C72B0", width=2.5),
                hovertemplate=(
                    "<b>name:</b> %{customdata[0]}<br>"
                    "<b>pair ipTM ≤</b> %{x:.2f}<br>"
                    "<b>Fraction:</b> %{y:.3f}<br>"
                    "<b>seed:</b> %{customdata[1]}<br>"
                    "<b>sample:</b> %{customdata[2]}<br>"
                    "<b>chain i:</b> %{customdata[3]}<br>"
                    "<b>chain j:</b> %{customdata[4]}<br>"
                    "<b>is top:</b> %{customdata[5]}<br>"
                    "<extra></extra>"
                ),
                customdata=d[["name", "seed", "sample", "chain_i", "chain_j", "is_top"]].values,
                showlegend=True,
            ))

        d_top = d[d["is_top"].str.lower() == "true"]
        if not d_top.empty:
            top_iptm = d_top["pair_iptm"].dropna().sort_values()
            if len(top_iptm) > 0:
                y_top = [i / len(top_iptm) for i in range(1, len(top_iptm) + 1)]
                fig.add_trace(go.Scatter(
                    x=top_iptm,
                    y=y_top,
                    mode='lines',
                    name="Top predictions",
                    line=dict(color="#D55E00", width=2.5, dash="solid"),
                    hovertemplate=(
                        "<b>name:</b> %{customdata[0]}<br>"
                        "<b>pair ipTM ≤</b> %{x:.2f}<br>"
                        "<b>Fraction:</b> %{y:.3f}<br>"
                        "<b>seed:</b> %{customdata[1]}<br>"
                        "<b>sample:</b> %{customdata[2]}<br>"
                        "<b>chain i:</b> %{customdata[3]}<br>"
                        "<b>chain j:</b> %{customdata[4]}<br>"
                        "<b>is top:</b> %{customdata[5]}<br>"
                        "<extra></extra>"
                    ),
                    customdata=d_top[["name", "seed", "sample", "chain_i", "chain_j", "is_top"]].values,
                    showlegend=True,
                ))

        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor="center"
            ),
            xaxis=dict(
                title="pair ipTM",
                range=[0, 1],
                tickvals=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                ticktext=["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
            ),
            yaxis=dict(
                title="Cumulative fraction",
                range=[0, 1.02],
                tickvals=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                ticktext=["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
            ),
            template="plotly_white",
            hovermode="x unified",
            height=650,
            margin=dict(l=70, r=50, t=80, b=70),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
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
    "--pred-tsv",
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
    default=None,
    help="Optional aggregated cohort predictions table, e.g. cohort_predictions.tsv"
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
    help="Optional: Path to cohort_master.tsv (contains tm1, tm2)."
)
def main(pair_tsv: Path, pred_tsv: Optional[Path], out_html: Path, tm_plot: Optional[Path], master_tsv: Optional[Path]):
    """
    Create two interactive plots:
    1. Cumulative distribution of chain-pair ipTM across all predictions.
    2. Distribution of normalized TM scores across all predictions.
    """
    # Load chain-pair data
    df_pair = load_tsv(pair_tsv)
    df_pair = coerce_numeric(df_pair, ["pair_iptm", "pair_pae_min"])

    # Load predictions data (for metadata and ipTM)
    df_pred = load_tsv(pred_tsv) if pred_tsv is not None and pred_tsv.exists() else pd.DataFrame()
    df_pred = coerce_numeric(df_pred, ["ranking_score", "iptm", "ptm", "mean_plddt_total"])

    # Add prediction metadata (is_top, etc.)
    d = add_prediction_metadata(df_pair, df_pred)

    # Plot 1: ipTM
    plot_chain_pair_iptm_cumulative(d, out_html)

    # Plot 2: TM score — use cohort_master.tsv if available
    if tm_plot is not None:
        # Use --master-tsv if provided
        if master_tsv is not None:
            master_path = master_tsv
        else:
            # Fallback: try reports/cohort/cohort_master.tsv
            master_path = Path("reports/cohort/cohort_master.tsv")

        if not master_path.exists():
            click.echo(f"⚠️  {master_path} not found. Skipping TM score plot.")
            return

        df_master = load_tsv(master_path)

        # ✅ Use exact column names as they appear in the file
        df_master = coerce_numeric(df_master, ["TM1", "TM2"])

        required_cols = {
            "sample_id", "prediction_id", "sample", "seed", "is_top",
            "ranking_score", "ptm", "iptm", "mean_plddt_total",
            "fraction_disordered", "has_clash", "TM1", "TM2"
        }
        available_cols = [c for c in required_cols if c in df_master.columns]
        df_tm = df_master[available_cols].copy()

        # Add name from sample_id
        df_tm["name"] = df_tm["sample_id"].str.split("_seed-").str[0]
        df_tm["name"] = df_tm["name"].astype(str).replace("nan", "N/A")

        # ✅ Use TM1, TM2 — not tm1, tm2
        df_tm["tm_score"] = df_tm[["TM1", "TM2"]].min(axis=1)
        df_tm["tm_score"] = pd.to_numeric(df_tm["tm_score"], errors="coerce")
        df_tm = df_tm[df_tm["tm_score"].notna()].copy()

        if df_tm.empty:
            click.echo("⚠️  No valid TM scores found. Skipping TM score plot.")
            return

        # Add is_top as string
        df_tm["is_top"] = df_tm["is_top"].astype(str).str.title()

        # Prepare metadata for hover
        meta_cols = [
            "name", "sample", "seed", "ranking_score", "ptm", "iptm",
            "mean_plddt_total", "fraction_disordered", "has_clash"
        ]
        available_meta = [c for c in meta_cols if c in df_tm.columns]
        df_tm = df_tm[["tm_score"] + available_meta + ["is_top"]].copy()

        # Plot
        plot_tm_score_distribution(df_tm, tm_plot)

    click.echo(f"✅ ipTM plot saved to: {out_html}")
    if tm_plot is not None:
        click.echo(f"✅ TM score plot saved to: {tm_plot}")

if __name__ == "__main__":
    main()