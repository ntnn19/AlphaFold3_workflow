#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Optional

import click
import numpy as np
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


def build_pair_label(df: pd.DataFrame) -> pd.Series:
    return df["chain_i"].astype(str) + "-" + df["chain_j"].astype(str)


def add_prediction_metadata(df_pair: pd.DataFrame, df_pred: pd.DataFrame) -> pd.DataFrame:
    d = df_pair.copy()

    if d.empty:
        return d

    if "pair_iptm" in d.columns:
        d["pair_iptm"] = pd.to_numeric(d["pair_iptm"], errors="coerce")

    if df_pred.empty or "sample_id" not in df_pred.columns or "prediction_id" not in df_pred.columns:
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

    d = d.merge(
        p,
        on=["sample_id", "prediction_id"],
        how="left"
    )

    if "is_top" in d.columns:
        d["is_top"] = d["is_top"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        d["is_top"] = False

    if "ranking_score" in d.columns:
        d["ranking_score"] = pd.to_numeric(d["ranking_score"], errors="coerce")
    if "iptm" in d.columns:
        d["iptm"] = pd.to_numeric(d["iptm"], errors="coerce")
    if "ptm" in d.columns:
        d["ptm"] = pd.to_numeric(d["ptm"], errors="coerce")
    if "mean_plddt_total" in d.columns:
        d["mean_plddt_total"] = pd.to_numeric(d["mean_plddt_total"], errors="coerce")

    return d


def write_no_data_html(path: Path, message: str = "No data to plot.") -> None:
    html = f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>chain-pair ipTM</title></head>
<body><p><em>{message}</em></p></body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def plot_chain_pair_iptm_interactive(
    df_pair: pd.DataFrame,
    out_html: Path,
    title: str = "Interactive chain-pair ipTM across all predictions",
) -> bool:
    required = {"sample_id", "prediction_id", "chain_i", "chain_j", "pair_iptm"}
    if df_pair.empty or not required.issubset(df_pair.columns):
        write_no_data_html(out_html, "No chain-pair ipTM data available.")
        return False

    d = df_pair.copy()
    d["sample_id"] = d["sample_id"].astype(str)
    d["prediction_id"] = d["prediction_id"].astype(str)
    d["chain_i"] = d["chain_i"].astype(str)
    d["chain_j"] = d["chain_j"].astype(str)
    d["pair_label"] = build_pair_label(d)

    if "is_diagonal" in d.columns:
        diag = d["is_diagonal"].astype(str).str.lower().isin(["true", "1", "yes"])
        d = d[~diag].copy()

    d["sample_prediction"] = d["sample_id"] + " | " + d["prediction_id"]
    d["prediction_label"] = np.where(
        d["is_top"].fillna(False),
        "TOP: " + d["prediction_id"].astype(str),
        d["prediction_id"].astype(str)
    ) if "is_top" in d.columns else d["prediction_id"].astype(str)

    if d.empty:
        write_no_data_html(out_html, "No non-diagonal chain-pair ipTM data available.")
        return False

    fig = go.Figure()

    customdata = np.stack([
        d["sample_id"].to_numpy(dtype=str),
        d["prediction_id"].to_numpy(dtype=str),
        d["chain_i"].to_numpy(dtype=str),
        d["chain_j"].to_numpy(dtype=str),
        d["ranking_score"].to_numpy(dtype=float) if "ranking_score" in d.columns else np.full(len(d), np.nan),
        d["iptm"].to_numpy(dtype=float) if "iptm" in d.columns else np.full(len(d), np.nan),
        d["ptm"].to_numpy(dtype=float) if "ptm" in d.columns else np.full(len(d), np.nan),
        d["mean_plddt_total"].to_numpy(dtype=float) if "mean_plddt_total" in d.columns else np.full(len(d), np.nan),
        d["is_top"].astype(int).to_numpy(dtype=int) if "is_top" in d.columns else np.zeros(len(d), dtype=int),
    ], axis=-1)

    fig.add_trace(go.Scatter(
        x=d["pair_label"],
        y=pd.to_numeric(d["pair_iptm"], errors="coerce"),
        mode="markers",
        marker=dict(
            size=9,
            color=pd.to_numeric(d["pair_iptm"], errors="coerce"),
            colorscale=[
                [0.00, "#ffffff"],
                [0.20, "#eff3ff"],
                [0.40, "#bdd7e7"],
                [0.60, "#6baed6"],
                [0.80, "#3182bd"],
                [1.00, "#08519c"],
            ],
            cmin=0,
            cmax=1,
            colorbar=dict(title="pair ipTM"),
            line=dict(
                color=np.where(d["is_top"].fillna(False), "#D55E00", "rgba(0,0,0,0.35)") if "is_top" in d.columns else "rgba(0,0,0,0.35)",
                width=np.where(d["is_top"].fillna(False), 2.0, 0.5) if "is_top" in d.columns else 0.5,
            ),
            symbol=np.where(d["is_top"].fillna(False), "diamond", "circle") if "is_top" in d.columns else "circle",
            opacity=0.9,
        ),
        customdata=customdata,
        hovertemplate=(
            "<b>Sample:</b> %{customdata[0]}<br>"
            "<b>Prediction:</b> %{customdata[1]}<br>"
            "<b>Chain pair:</b> %{customdata[2]}-%{customdata[3]}<br>"
            "<b>pair ipTM:</b> %{y:.3f}<br>"
            "<b>ranking_score:</b> %{customdata[4]:.3f}<br>"
            "<b>iptm:</b> %{customdata[5]:.3f}<br>"
            "<b>ptm:</b> %{customdata[6]:.3f}<br>"
            "<b>mean pLDDT:</b> %{customdata[7]:.3f}<br>"
            "<b>top prediction:</b> %{customdata[8]}<extra></extra>"
        ),
        text=d["sample_prediction"],
        showlegend=False,
    ))

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center"
        ),
        xaxis=dict(
            title="Chain pair",
            categoryorder="array",
            categoryarray=sorted(d["pair_label"].dropna().unique().tolist())
        ),
        yaxis=dict(
            title="pair ipTM",
            range=[0, 1.05]
        ),
        template="plotly_white",
        height=700,
        margin=dict(l=70, r=70, t=80, b=120),
        hovermode="closest",
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
    Create an interactive chain-pair ipTM plot across all predictions.
    """
    df_pair = load_tsv(pair_tsv)
    df_pair = coerce_numeric(df_pair, ["pair_iptm", "pair_pae_min"])

    df_pred = load_tsv(pred_tsv) if pred_tsv is not None and pred_tsv.exists() else pd.DataFrame()
    df_pred = coerce_numeric(df_pred, ["ranking_score", "iptm", "ptm", "mean_plddt_total"])

    df = add_prediction_metadata(df_pair, df_pred)
    plot_chain_pair_iptm_interactive(df, out_html)

    click.echo(str(out_html))


if __name__ == "__main__":
    main()