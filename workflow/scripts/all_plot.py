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
<head><meta charset="utf-8"><title>Plot</title></head>
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


# ---------------------------------------------------------------------------
# TM-score heatmap
# ---------------------------------------------------------------------------

def plot_tm_score_distribution(
    df_pred: pd.DataFrame,
    out_html: Path,
    title: str = "TM scores: predictions vs ground truths",
) -> bool:
    required = {"sample_id", "prediction_id", "TM2"}
    if df_pred.empty or not required.issubset(df_pred.columns):
        write_no_data_html(out_html, "No TM score data available.")
        return False

    d = df_pred.copy()
    d["sample_id"] = d["sample_id"].astype(str)
    d["prediction_id"] = d["prediction_id"].astype(str)
    d["TM2"] = pd.to_numeric(d["TM2"], errors="coerce")
    if "TM1" in d.columns:
        d["TM1"] = pd.to_numeric(d["TM1"], errors="coerce")

    d = d[d["TM2"].notna()].copy()
    if d.empty:
        write_no_data_html(out_html, "No valid TM scores available.")
        return False

    d["name"] = d["sample_id"].str.split("_seed-").str[0].astype(str).replace("nan", "N/A")
    if "seed" not in d.columns or d["seed"].astype(str).eq("").all():
        d["seed"] = d["sample_id"].str.extract(r"_seed-(\d+)", expand=False).fillna("N/A")
    else:
        d["seed"] = d["seed"].astype(str).replace("", "N/A").replace("nan", "N/A")
    if "sample" not in d.columns or d["sample"].astype(str).eq("").all():
        d["sample"] = d["sample_id"].str.extract(r"_sample-(\d+)", expand=False).fillna("N/A")
    else:
        d["sample"] = d["sample"].astype(str).replace("", "N/A").replace("nan", "N/A")

    d = _prepare_description_col(d)

    if "is_top" in d.columns:
        d["is_top"] = d["is_top"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        d["is_top"] = False

    if "ground_truth_id" not in d.columns:
        d["ground_truth_id"] = "default"
    else:
        d["ground_truth_id"] = d["ground_truth_id"].astype(str).replace("", "default").replace("nan", "default")

    for c in ["ranking_score", "ptm", "iptm", "mean_plddt_total", "fraction_disordered"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    def _row_label(row):
        parts = [row["name"]]
        if row.get("seed", "N/A") != "N/A":
            parts.append(f"seed={row['seed']}")
        if row.get("sample", "N/A") != "N/A":
            parts.append(f"sample={row['sample']}")
        top_tag = " ★" if row.get("is_top", False) else ""
        return " | ".join(parts) + top_tag

    d["row_label"] = d.apply(_row_label, axis=1)

    gt_ids = sorted(d["ground_truth_id"].unique().tolist())

    label_counts = d.groupby("row_label")["prediction_id"].nunique()
    dup_labels = set(label_counts[label_counts > 1].index)
    if dup_labels:
        d["row_label"] = d.apply(
            lambda r: (
                f"{r['row_label']} [{r['prediction_id'][-6:]}]"
                if r["row_label"] in dup_labels else r["row_label"]
            ), axis=1,
        )

    pivot = d.pivot_table(index="row_label", columns="ground_truth_id", values="TM2", aggfunc="first")
    pivot = pivot.reindex(columns=gt_ids)
    pivot["_mean"] = pivot[gt_ids].mean(axis=1)
    pivot = pivot.sort_values("_mean", ascending=True).drop(columns=["_mean"])

    pred_labels = pivot.index.tolist()
    z = pivot.values.tolist()

    lookup = {}
    for _, row in d.iterrows():
        lookup[(row["row_label"], row["ground_truth_id"])] = row

    hover_text = []
    for pred in pred_labels:
        row_hover = []
        for gt in gt_ids:
            info = lookup.get((pred, gt))
            if info is not None:
                tm2_val = info["TM2"]
                tm1_val = info.get("TM1", float("nan"))
                lines = [
                    f"<b>Prediction:</b> {pred}",
                    f"<b>Ground truth:</b> {gt}",
                    f"<b>TM2:</b> {tm2_val:.3f}" if pd.notna(tm2_val) else "<b>TM2:</b> N/A",
                    f"<b>TM1:</b> {tm1_val:.3f}" if pd.notna(tm1_val) else "<b>TM1:</b> N/A",
                    f"<b>description:</b> {info.get('description', 'N/A')}",
                    f"<b>seed:</b> {info.get('seed', 'N/A')}",
                    f"<b>sample:</b> {info.get('sample', 'N/A')}",
                    f"<b>ranking score:</b> {info.get('ranking_score', 'N/A')}",
                    f"<b>ptm:</b> {info.get('ptm', 'N/A')}",
                    f"<b>iptm:</b> {info.get('iptm', 'N/A')}",
                    f"<b>mean pLDDT:</b> {info.get('mean_plddt_total', 'N/A')}",
                    f"<b>fraction disordered:</b> {info.get('fraction_disordered', 'N/A')}",
                    f"<b>is top:</b> {info.get('is_top', 'N/A')}",
                ]
                row_hover.append("<br>".join(lines))
            else:
                row_hover.append(
                    f"<b>Prediction:</b> {pred}<br><b>Ground truth:</b> {gt}<br><b>TM2:</b> N/A"
                )
        hover_text.append(row_hover)

    z_clean = [
        [None if (v is None or (isinstance(v, float) and np.isnan(v))) else v for v in row]
        for row in z
    ]

    n_preds = len(pred_labels)
    n_gts = len(gt_ids)
    plot_height = max(400, min(2400, 120 + n_preds * 28))
    plot_width = max(500, min(1800, 250 + n_gts * 90))

    colorscale = [
        [0.0, "#d73027"], [0.25, "#fc8d59"], [0.5, "#ffffbf"],
        [0.75, "#91bfdb"], [1.0, "#4575b4"],
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z_clean, x=gt_ids, y=pred_labels, text=hover_text,
        hoverinfo="text", colorscale=colorscale, zmin=0, zmax=1,
        colorbar=dict(title=dict(text="TM2 score", side="right"),
                      tickvals=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], len=0.8),
        xgap=2, ygap=2,
    ))

    for i, pred in enumerate(pred_labels):
        for j, gt in enumerate(gt_ids):
            val = z_clean[i][j]
            if val is not None:
                text_color = "white" if val < 0.25 or val > 0.85 else "black"
                fig.add_annotation(x=gt, y=pred, text=f"{val:.2f}", showarrow=False,
                                   font=dict(size=11, color=text_color))

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=16)),
        xaxis=dict(title="Ground Truth", side="bottom",
                   tickangle=-45 if n_gts > 6 else 0, tickfont=dict(size=11)),
        yaxis=dict(title="Prediction", autorange="reversed", tickfont=dict(size=10)),
        template="plotly_white", height=plot_height, width=plot_width,
        margin=dict(l=250, r=80, t=80, b=120),
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn", full_html=True)
    return True


# ---------------------------------------------------------------------------
# Chain-pair ipTM – ordered dot plot
# ---------------------------------------------------------------------------

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

    keep = ["sample_id", "prediction_id", "ranking_score", "iptm", "ptm",
            "mean_plddt_total", "description"]
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
    title: str = "Chain-pair ipTM (descending order)",
) -> bool:
    """Ordered dot-plot of chain-pair ipTM values.

    Points are sorted in descending ipTM order.  Pairs belonging to the
    top-ranked prediction (by ranking_score) are drawn with a distinct
    marker so they stand out.
    """
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

    # ---- derive helper columns ----
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

    if "ranking_score" in d.columns:
        d["ranking_score"] = pd.to_numeric(d["ranking_score"], errors="coerce")
    else:
        d["ranking_score"] = np.nan

    # ---- determine "top" pairs ----
    # Top prediction = the prediction_id with the highest ranking_score.
    # If is_top was already set upstream we respect it; otherwise derive it.
    if "is_top" in d.columns:
        if d["is_top"].dtype != bool:
            d["is_top"] = d["is_top"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        d["is_top"] = False

    # If no rows are marked top yet, pick the prediction with highest ranking_score
    if not d["is_top"].any() and d["ranking_score"].notna().any():
        best_pred = (
            d.groupby("prediction_id")["ranking_score"]
            .first()
            .idxmax()
        )
        d["is_top"] = d["prediction_id"] == best_pred

    d["is_top_str"] = d["is_top"].map({True: "True", False: "False"})

    for c in ["iptm", "ptm", "mean_plddt_total"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    # ---- sort descending by pair_iptm ----
    d = d.sort_values("pair_iptm", ascending=False).reset_index(drop=True)
    d["rank"] = d.index + 1  # 1-based rank

    # ---- split into regular vs top ----
    d_regular = d[~d["is_top"]].copy()
    d_top = d[d["is_top"]].copy()

    # ---- hover columns ----
    hover_cols = [
        "name", "seed", "sample", "chain_i", "chain_j",
        "chain_i_desc", "chain_j_desc", "description", "is_top_str",
        "ranking_score", "iptm", "ptm", "mean_plddt_total",
    ]
    for c in hover_cols:
        if c not in d.columns:
            d[c] = "N/A"
            d_regular[c] = "N/A"
            d_top[c] = "N/A"

    hover_tpl = (
        "<b>rank:</b> %{x}<br>"
        "<b>pair ipTM:</b> %{y:.3f}<br>"
        "<b>name:</b> %{customdata[0]}<br>"
        "<b>description:</b> %{customdata[7]}<br>"
        "<b>seed:</b> %{customdata[1]}<br>"
        "<b>sample:</b> %{customdata[2]}<br>"
        "<b>chain i:</b> %{customdata[3]} (%{customdata[5]})<br>"
        "<b>chain j:</b> %{customdata[4]} (%{customdata[6]})<br>"
        "<b>ranking score:</b> %{customdata[9]}<br>"
        "<b>iptm:</b> %{customdata[10]}<br>"
        "<b>ptm:</b> %{customdata[11]}<br>"
        "<b>mean pLDDT:</b> %{customdata[12]}<br>"
        "<b>is top:</b> %{customdata[8]}<br>"
        "<extra></extra>"
    )

    fig = go.Figure()

    # Regular points
    if not d_regular.empty:
        fig.add_trace(go.Scatter(
            x=d_regular["rank"],
            y=d_regular["pair_iptm"],
            mode="markers",
            name="Other predictions",
            marker=dict(color="#4C72B0", size=6, opacity=0.7,
                        line=dict(width=0.5, color="white")),
            customdata=d_regular[hover_cols].values,
            hovertemplate=hover_tpl,
            showlegend=True,
        ))

    # Top-ranked prediction
    if not d_top.empty:
        fig.add_trace(go.Scatter(
            x=d_top["rank"],
            y=d_top["pair_iptm"],
            mode="markers",
            name="Top prediction (by ranking score)",
            marker=dict(color="#D55E00", size=9, opacity=0.95,
                        symbol="star",
                        line=dict(width=1, color="black")),
            customdata=d_top[hover_cols].values,
            hovertemplate=hover_tpl,
            showlegend=True,
        ))

    # Reference lines
    fig.add_hline(y=0.4, line_dash="dot", line_color="#999", line_width=1,
                  annotation_text="0.4", annotation_position="bottom right",
                  annotation_font_size=10, annotation_font_color="#999")
    fig.add_hline(y=0.7, line_dash="dot", line_color="#999", line_width=1,
                  annotation_text="0.7", annotation_position="bottom right",
                  annotation_font_size=10, annotation_font_color="#999")

    n_total = len(d)
    plot_height = max(420, min(700, 350 + n_total))

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=16)),
        xaxis=dict(title="Chain pair (sorted by descending ipTM)", tickfont=dict(size=11)),
        yaxis=dict(title="pair ipTM", range=[-0.02, 1.02],
                   tickvals=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        template="plotly_white",
        hovermode="closest",
        height=plot_height,
        margin=dict(l=70, r=40, t=70, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5, font=dict(size=12)),
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn", full_html=True)
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
    help="Output HTML path for TM score heatmap."
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
    1. Chain-pair ipTM ordered dot plot.
    2. Heatmap of TM scores (predictions × ground truths).
    """
    df_pair = load_tsv(pair_tsv)
    df_pair = coerce_numeric(df_pair, ["pair_iptm", "pair_pae_min"])

    df_master = (
        load_tsv(master_tsv)
        if master_tsv is not None and master_tsv.exists()
        else pd.DataFrame()
    )
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