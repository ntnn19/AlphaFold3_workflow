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
    """
    Ensure a 'description' column exists and is suitable for display.
    Also builds 'chain_i_desc' and 'chain_j_desc' display strings if
    chain_i_description / chain_j_description columns are present.
    """
    # For prediction-level description
    if "description" not in d.columns:
        d["description"] = "N/A"
    else:
        d["description"] = d["description"].astype(str).replace("", "N/A").replace("nan", "N/A")

    # For chain-pair-level descriptions
    for src, dst in [
        ("chain_i_description", "chain_i_desc"),
        ("chain_j_description", "chain_j_desc"),
    ]:
        if src in d.columns:
            d[dst] = d[src].astype(str).replace("", "N/A").replace("nan", "N/A")
        else:
            d[dst] = "N/A"

    return d


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

    d["name"] = d["sample_id"].str.split("_seed-").str[0]
    d["name"] = d["name"].astype(str).replace("nan", "N/A")

    if "seed" not in d.columns or d["seed"].astype(str).eq("").all():
        d["seed"] = d["sample_id"].str.extract(r"_seed-(\d+)", expand=False).fillna("N/A")
    else:
        d["seed"] = d["seed"].astype(str).replace("", "N/A").replace("nan", "N/A")

    if "sample" not in d.columns or d["sample"].astype(str).eq("").all():
        d["sample"] = d["sample_id"].str.extract(r"_sample-(\d+)", expand=False).fillna("N/A")
    else:
        d["sample"] = d["sample"].astype(str).replace("", "N/A").replace("nan", "N/A")

    d["tm_score"] = d["TM2"]
    d["tm_score"] = pd.to_numeric(d["tm_score"], errors="coerce")

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

    # Prepare description column
    d = _prepare_description_col(d)

    meta_cols = [
        "name", "sample", "seed", "ranking_score", "ptm", "iptm",
        "mean_plddt_total", "fraction_disordered", "has_clash", "description"
    ]
    for c in meta_cols:
        if c not in d.columns:
            d[c] = "N/A"

    d["is_top_str"] = d["is_top"].map({True: "True", False: "False"})

    n_predictions = len(d)
    fig = go.Figure()

    if n_predictions <= 100:
        jitter = 0.01
        d["jitter"] = np.random.uniform(-jitter, jitter, size=len(d))

        fig.add_trace(go.Scatter(
            x=d["tm_score"] + d["jitter"],
            y=[0.5] * len(d),
            mode='markers',
            name="All predictions",
            marker=dict(
                color="#4C72B0", size=6, opacity=0.7,
                line=dict(width=0.5, color="black")
            ),
            hovertemplate=(
                "<b>name:</b> %{customdata[0]}<br>"
                "<b>description:</b> %{customdata[9]}<br>"
                "<b>sample:</b> %{customdata[1]}<br>"
                "<b>seed:</b> %{customdata[2]}<br>"
                "<b>ranking score:</b> %{customdata[3]}<br>"
                "<b>ptm:</b> %{customdata[4]}<br>"
                "<b>iptm:</b> %{customdata[5]}<br>"
                "<b>mean pLDDT:</b> %{customdata[6]}<br>"
                "<b>fraction disordered:</b> %{customdata[7]}<br>"
                "<b>has clash:</b> %{customdata[8]}<br>"
                "<b>is top:</b> %{customdata[10]}<br>"
                "<b>normalized TM:</b> %{x:.3f}<br>"
                "<extra></extra>"
            ),
            customdata=d[meta_cols + ["is_top_str"]].values,
            showlegend=True,
        ))

        d_top = d[d["is_top"]].copy()
        if not d_top.empty:
            d_top["jitter"] = np.random.uniform(-jitter, jitter, size=len(d_top))
            fig.add_trace(go.Scatter(
                x=d_top["tm_score"] + d_top["jitter"],
                y=[0.5] * len(d_top),
                mode='markers',
                name="Top predictions",
                marker=dict(
                    color="#D55E00", size=8, opacity=0.8,
                    line=dict(width=1.5, color="black")
                ),
                hovertemplate=(
                    "<b>name:</b> %{customdata[0]}<br>"
                    "<b>description:</b> %{customdata[9]}<br>"
                    "<b>sample:</b> %{customdata[1]}<br>"
                    "<b>seed:</b> %{customdata[2]}<br>"
                    "<b>ranking score:</b> %{customdata[3]}<br>"
                    "<b>ptm:</b> %{customdata[4]}<br>"
                    "<b>iptm:</b> %{customdata[5]}<br>"
                    "<b>mean pLDDT:</b> %{customdata[6]}<br>"
                    "<b>fraction disordered:</b> %{customdata[7]}<br>"
                    "<b>has clash:</b> %{customdata[8]}<br>"
                    "<b>is top:</b> %{customdata[10]}<br>"
                    "<b>normalized TM:</b> %{x:.3f}<br>"
                    "<extra></extra>"
                ),
                customdata=d_top[meta_cols + ["is_top_str"]].values,
                showlegend=True,
            ))

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            xaxis=dict(
                title="Normalized TM score (min(TM1, TM2))",
                range=[0, 1],
                tickvals=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            ),
            yaxis=dict(
                showticklabels=False,
                title="",
                range=[0, 1]
            ),
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
        d_sorted = d.sort_values("tm_score").reset_index(drop=True)
        all_tm = d_sorted["tm_score"]
        if len(all_tm) > 0:
            y_all = [(i + 1) / len(all_tm) for i in range(len(all_tm))]
            fig.add_trace(go.Scatter(
                x=all_tm, y=y_all,
                mode='lines', name="All predictions",
                line=dict(color="#4C72B0", width=2.5),
                hovertemplate=(
                    "<b>name:</b> %{customdata[0]}<br>"
                    "<b>description:</b> %{customdata[9]}<br>"
                    "<b>sample:</b> %{customdata[1]}<br>"
                    "<b>seed:</b> %{customdata[2]}<br>"
                    "<b>ranking score:</b> %{customdata[3]}<br>"
                    "<b>ptm:</b> %{customdata[4]}<br>"
                    "<b>iptm:</b> %{customdata[5]}<br>"
                    "<b>mean pLDDT:</b> %{customdata[6]}<br>"
                    "<b>fraction disordered:</b> %{customdata[7]}<br>"
                    "<b>has clash:</b> %{customdata[8]}<br>"
                    "<b>is top:</b> %{customdata[10]}<br>"
                    "<b>normalized TM ≤</b> %{x:.2f}<br>"
                    "<b>Fraction:</b> %{y:.3f}<br>"
                    "<extra></extra>"
                ),
                customdata=d_sorted[meta_cols + ["is_top_str"]].values,
                showlegend=True,
            ))

        d_top = d[d["is_top"]].copy()
        if not d_top.empty:
            d_top_sorted = d_top.sort_values("tm_score").reset_index(drop=True)
            top_tm = d_top_sorted["tm_score"]
            if len(top_tm) > 0:
                y_top = [(i + 1) / len(top_tm) for i in range(len(top_tm))]
                fig.add_trace(go.Scatter(
                    x=top_tm, y=y_top,
                    mode='lines', name="Top predictions",
                    line=dict(color="#D55E00", width=2.5, dash="solid"),
                    hovertemplate=(
                        "<b>name:</b> %{customdata[0]}<br>"
                        "<b>description:</b> %{customdata[9]}<br>"
                        "<b>sample:</b> %{customdata[1]}<br>"
                        "<b>seed:</b> %{customdata[2]}<br>"
                        "<b>ranking score:</b> %{customdata[3]}<br>"
                        "<b>ptm:</b> %{customdata[4]}<br>"
                        "<b>iptm:</b> %{customdata[5]}<br>"
                        "<b>mean pLDDT:</b> %{customdata[6]}<br>"
                        "<b>fraction disordered:</b> %{customdata[7]}<br>"
                        "<b>has clash:</b> %{customdata[8]}<br>"
                        "<b>is top:</b> %{customdata[10]}<br>"
                        "<b>normalized TM ≤</b> %{x:.2f}<br>"
                        "<b>Fraction:</b> %{y:.3f}<br>"
                        "<extra></extra>"
                    ),
                    customdata=d_top_sorted[meta_cols + ["is_top_str"]].values,
                    showlegend=True,
                ))

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            xaxis=dict(
                title="Normalized TM score (min(TM1, TM2))",
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

    # Normalise is_top already present in chain-pair data to boolean
    if "is_top" in d.columns:
        d["is_top"] = d["is_top"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        d["is_top"] = False

    # Normalise seed / sample that already exist in chain-pair data
    if "seed" in d.columns:
        d["seed"] = d["seed"].astype(str).replace("", "N/A").replace("nan", "N/A")
    if "sample" in d.columns:
        d["sample"] = d["sample"].astype(str).replace("", "N/A").replace("nan", "N/A")

    if df_pred.empty or not required.issubset(df_pred.columns):
        return d

    p = df_pred.copy()
    p["sample_id"] = p["sample_id"].astype(str)
    p["prediction_id"] = p["prediction_id"].astype(str)

    # Only pull columns from pred that are NOT already in pair data (except keys)
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
    # Add columns that are missing from pair data
    extra = [c for c in keep if c not in d.columns or c in ["sample_id", "prediction_id"]]
    p = p[extra].drop_duplicates()

    if len(extra) > 2:  # more than just the join keys
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

    # Extract name from sample_id
    d["name"] = d["sample_id"].str.split("_seed-").str[0]
    d["name"] = d["name"].astype(str).replace("nan", "N/A")

    # Use seed/sample columns directly if present, otherwise extract
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

    # Prepare description columns
    d = _prepare_description_col(d)

    # Normalise is_top to boolean then to string for display
    if "is_top" in d.columns:
        if d["is_top"].dtype == bool:
            pass  # already boolean
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
    # Load chain-pair data
    df_pair = load_tsv(pair_tsv)
    df_pair = coerce_numeric(df_pair, ["pair_iptm", "pair_pae_min"])

    # Load master table (for prediction metadata including description)
    df_master = load_tsv(master_tsv) if master_tsv is not None and master_tsv.exists() else pd.DataFrame()
    df_master = coerce_numeric(df_master, ["ranking_score", "iptm", "ptm", "mean_plddt_total"])

    # Add prediction metadata (ranking_score, description, etc.)
    d = add_prediction_metadata(df_pair, df_master)

    # Plot 1: ipTM
    plot_chain_pair_iptm_cumulative(d, out_html)

    # Plot 2: TM score
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