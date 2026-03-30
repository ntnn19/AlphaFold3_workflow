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
    help="Output HTML path."
)
def main(pair_tsv: Path, pred_tsv: Optional[Path], out_html: Path):
    """
    Create an interactive cumulative histogram of chain-pair ipTM across all predictions.
    """
    df_pair = load_tsv(pair_tsv)
    df_pair = coerce_numeric(df_pair, ["pair_iptm", "pair_pae_min"])

    df_pred = load_tsv(pred_tsv) if pred_tsv is not None and pred_tsv.exists() else pd.DataFrame()
    df_pred = coerce_numeric(df_pred, ["ranking_score", "iptm", "ptm", "mean_plddt_total"])

    d = add_prediction_metadata(df_pair, df_pred)
    plot_chain_pair_iptm_cumulative(d, out_html)

    click.echo(str(out_html))


if __name__ == "__main__":
    main()