#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Optional

import click
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def read_predictions(pred_tsv: Path) -> pd.DataFrame:
    if not pred_tsv.exists():
        return pd.DataFrame()
    return pd.read_csv(pred_tsv, sep="\t", dtype=str).fillna("")


def top_prediction_id(df_pred: pd.DataFrame) -> Optional[str]:
    if df_pred.empty or "prediction_id" not in df_pred.columns or "ranking_score" not in df_pred.columns:
        return None

    d = df_pred.copy()
    d["prediction_id"] = d["prediction_id"].astype(str)
    d["ranking_score"] = pd.to_numeric(d["ranking_score"], errors="coerce")

    cand = d[(d["prediction_id"] != "top") & d["ranking_score"].notna()].copy()
    if cand.empty:
        cand = d[d["ranking_score"].notna()].copy()
    if cand.empty:
        return None

    return str(
        cand.sort_values(["ranking_score", "prediction_id"], ascending=[False, True]).iloc[0]["prediction_id"]
    )


def usalign_id_from_file(path: Path) -> str:
    name = path.name
    if name.endswith(".usalign.tsv"):
        return name[:-len(".usalign.tsv")]
    return path.stem


def parse_one_usalign_file(path: Path) -> pd.DataFrame:
    """
    Parse one US-align output file using whitespace delimiter.
    """
    try:
        df = pd.read_csv(path, sep=r"\s+", engine="python")
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    cols = list(df.columns)
    if cols:
        cols[0] = str(cols[0]).lstrip("#")
        df.columns = cols

    df["usalign_id"] = usalign_id_from_file(path)
    return df.head(1).copy()


def load_usalign_results(report_dir: Path) -> pd.DataFrame:
    files = sorted(report_dir.glob("*.usalign.tsv"))
    parts = [parse_one_usalign_file(p) for p in files]
    parts = [x for x in parts if not x.empty]

    if not parts:
        return pd.DataFrame(
            columns=[
                "usalign_id", "PDBchain1", "PDBchain2",
                "TM1", "TM2", "RMSD", "ID1", "ID2", "IDali", "L1", "L2", "Lali"
            ]
        )

    d = pd.concat(parts, ignore_index=True)

    for col in ["TM1", "TM2", "RMSD", "ID1", "ID2", "IDali", "L1", "L2", "Lali"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    return d


def merge_with_predictions(df_u: pd.DataFrame, df_pred: pd.DataFrame) -> pd.DataFrame:
    if df_u.empty:
        return df_u.copy()

    d = df_u.copy()
    d["prediction_id"] = d["usalign_id"].astype(str)
    return d


def build_interactive_plot(df: pd.DataFrame, html_path: Path) -> Optional[str]:
    if df.empty:
        html_path.write_text("<p><em>No data.</em></p>", encoding="utf-8")
        return None

    d = df.copy()
    d["prediction_id"] = d["prediction_id"].astype(str)
    d["ranking_score"] = pd.to_numeric(d.get("ranking_score"), errors="coerce")

    for col in ["TM1", "TM2", "RMSD", "ID1", "ID2", "IDali", "L1", "L2", "Lali"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    d = d.sort_values(
        ["is_top", "ranking_score", "prediction_id"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    labels = [
        f"TOP: {pid}" if bool(is_top) else pid
        for pid, is_top in zip(d["prediction_id"], d["is_top"])
    ]

    x = np.arange(len(d))
    x_tm1 = x - 0.16
    x_tm2 = x + 0.16

    customdata = np.stack([
        d["ID1"].to_numpy(dtype=float) if "ID1" in d.columns else np.full(len(d), np.nan),
        d["ID2"].to_numpy(dtype=float) if "ID2" in d.columns else np.full(len(d), np.nan),
        d["IDali"].to_numpy(dtype=float) if "IDali" in d.columns else np.full(len(d), np.nan),
        d["L1"].to_numpy(dtype=float) if "L1" in d.columns else np.full(len(d), np.nan),
        d["L2"].to_numpy(dtype=float) if "L2" in d.columns else np.full(len(d), np.nan),
        d["Lali"].to_numpy(dtype=float) if "Lali" in d.columns else np.full(len(d), np.nan),
        d["ranking_score"].to_numpy(dtype=float),
        d["is_top"].astype(int).to_numpy(dtype=int),
    ], axis=-1)

    fig = go.Figure()

    # TM1
    if "TM1" in d.columns and d["TM1"].notna().any():
        fig.add_trace(go.Scatter(
            x=x_tm1,
            y=d["TM1"],
            mode="markers",
            name="TM1",
            marker=dict(color="#1f77b4", size=10),
            customdata=customdata,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Source: TM1<br>"
                "TM-score: %{y:.4f}<br>"
                "ID1: %{customdata[0]:.3f}<br>"
                "ID2: %{customdata[1]:.3f}<br>"
                "IDali: %{customdata[2]:.3f}<br>"
                "L1: %{customdata[3]:.0f}<br>"
                "L2: %{customdata[4]:.0f}<br>"
                "Lali: %{customdata[5]:.0f}<br>"
                "Ranking score: %{customdata[6]:.3f}<br>"
                "<extra></extra>"
            ),
            text=labels,
            yaxis="y1"
        ))

    # TM2
    if "TM2" in d.columns and d["TM2"].notna().any():
        fig.add_trace(go.Scatter(
            x=x_tm2,
            y=d["TM2"],
            mode="markers",
            name="TM2",
            marker=dict(color="#ff7f0e", size=10),
            customdata=customdata,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Source: TM2<br>"
                "TM-score: %{y:.4f}<br>"
                "ID1: %{customdata[0]:.3f}<br>"
                "ID2: %{customdata[1]:.3f}<br>"
                "IDali: %{customdata[2]:.3f}<br>"
                "L1: %{customdata[3]:.0f}<br>"
                "L2: %{customdata[4]:.0f}<br>"
                "Lali: %{customdata[5]:.0f}<br>"
                "Ranking score: %{customdata[6]:.3f}<br>"
                "<extra></extra>"
            ),
            text=labels,
            yaxis="y1"
        ))

    # RMSD on secondary axis
    if "RMSD" in d.columns and d["RMSD"].notna().any():
        fig.add_trace(go.Scatter(
            x=x,
            y=d["RMSD"],
            mode="markers",
            name="RMSD",
            marker=dict(color="#2ca02c", size=9, symbol="diamond"),
            line=dict(color="#2ca02c", width=1),
            customdata=customdata,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Metric: RMSD<br>"
                "RMSD: %{y:.4f}<br>"
                "ID1: %{customdata[0]:.3f}<br>"
                "ID2: %{customdata[1]:.3f}<br>"
                "IDali: %{customdata[2]:.3f}<br>"
                "L1: %{customdata[3]:.0f}<br>"
                "L2: %{customdata[4]:.0f}<br>"
                "Lali: %{customdata[5]:.0f}<br>"
                "Ranking score: %{customdata[6]:.3f}<br>"
                "<extra></extra>"
            ),
            text=labels,
            yaxis="y2"
        ))

    # Top annotations + metadata annotations
    annotations = []
    for i, row in d.iterrows():

        if bool(row.get("is_top", False)):
            y_top = 1.08
            annotations.append(dict(
                x=x[i],
                y=y_top,
                xref="x",
                yref="paper",
                text="<b>TOP</b>",
                showarrow=False,
                xanchor="center",
                yanchor="bottom",
                font=dict(size=12, color="#D55E00")
            ))

    fig.update_layout(
        title=dict(
            text="US-align TM-scores and RMSD by prediction",
            x=0.5,
            xanchor="center"
        ),
        xaxis=dict(
            title="Prediction",
            tickmode="array",
            tickvals=x,
            ticktext=labels
        ),
        yaxis=dict(
            title="TM-score",
            range=[0, 1.05]
        ),
        yaxis2=dict(
            title="RMSD",
            overlaying="y",
            side="right"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        hovermode="closest",
        template="plotly_white",
        height=750,
        margin=dict(l=70, r=70, t=80, b=180),
        annotations=annotations
    )

    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)
    return str(html_path)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("report_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("-o", "--outdir", type=click.Path(file_okay=False, path_type=Path), required=True,
              help="Output directory for US-align report artifacts.")
@click.option("--predictions-tsv", type=click.Path(exists=False, path_type=Path), default=None,
              help="Optional AF3 predictions.tsv used to annotate top-ranked prediction.")
def main(report_dir: Path, outdir: Path, predictions_tsv: Optional[Path]):
    """
    Generate standalone US-align report artifacts from *.usalign.tsv files in REPORT_DIR.

    Outputs:
      - usalign_summary.tsv
      - plots/usalign_tm_rmsd_interactive.html
    """
    outdir.mkdir(parents=True, exist_ok=True)

    if predictions_tsv is None:
        predictions_tsv = report_dir / "predictions.tsv"

    df_pred = read_predictions(predictions_tsv) if predictions_tsv and predictions_tsv.exists() else pd.DataFrame()
    df_u = load_usalign_results(report_dir)
    df_m = merge_with_predictions(df_u, df_pred)

    out_tsv = outdir / "usalign_summary.tsv"
    df_m.to_csv(out_tsv, sep="\t", index=False)

    html_path = outdir / "plots" / "usalign_tm_rmsd_interactive.html"
    build_interactive_plot(df_m, html_path)

    click.echo(str(out_tsv))
    click.echo(str(html_path))


if __name__ == "__main__":
    main()