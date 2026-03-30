#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import click
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


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


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize likely US-align outfmt=2 column names to:
      IDali, L1, L2, Lali, RMSD, TM1, TM2
    """
    d = df.copy()

    rename_map = {}
    for c in d.columns:
        lc = str(c).strip().lower()

        if lc in {"idali", "id_ali", "seqid", "id"}:
            rename_map[c] = "IDali"
        elif lc in {"l1", "len1", "length1"}:
            rename_map[c] = "L1"
        elif lc in {"l2", "len2", "length2"}:
            rename_map[c] = "L2"
        elif lc in {"lali", "ali_len", "aligned_length", "alnlen", "len_ali"}:
            rename_map[c] = "Lali"
        elif lc in {"rmsd"}:
            rename_map[c] = "RMSD"
        elif lc in {"tm1", "tmscore1", "tm-score1", "tm_score_1", "tm-score_1"}:
            rename_map[c] = "TM1"
        elif lc in {"tm2", "tmscore2", "tm-score2", "tm_score_2", "tm-score_2"}:
            rename_map[c] = "TM2"

    d = d.rename(columns=rename_map)
    return d


def parse_one_usalign_file(path: Path) -> pd.DataFrame:
    """
    Read one US-align TSV. Assumes one row of tabular results.
    """
    try:
        df = pd.read_csv(path, sep=r"\s+|\t+", engine="python", comment="#")
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    df = _normalize_columns(df)
    df["usalign_id"] = usalign_id_from_file(path)
    return df.head(1).copy()


def load_usalign_results(report_dir: Path) -> pd.DataFrame:
    files = sorted(report_dir.glob("*.usalign.tsv"))
    parts = [parse_one_usalign_file(p) for p in files]
    parts = [x for x in parts if not x.empty]

    if not parts:
        return pd.DataFrame(columns=["usalign_id", "IDali", "L1", "L2", "Lali", "RMSD", "TM1", "TM2"])

    d = pd.concat(parts, ignore_index=True)

    for col in ["IDali", "L1", "L2", "Lali", "RMSD", "TM1", "TM2"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    return d


def merge_with_predictions(df_u: pd.DataFrame, df_pred: pd.DataFrame) -> pd.DataFrame:
    if df_u.empty:
        return df_u.copy()

    d = df_u.copy()
    d["prediction_id"] = d["usalign_id"].astype(str)

    if df_pred.empty or "prediction_id" not in df_pred.columns:
        d["ranking_score"] = np.nan
        d["is_top"] = False
        return d

    p = df_pred.copy()
    p["prediction_id"] = p["prediction_id"].astype(str)
    p["ranking_score"] = pd.to_numeric(p.get("ranking_score"), errors="coerce")

    top_pid = top_prediction_id(p)

    d = d.merge(
        p[["prediction_id", "ranking_score"]].drop_duplicates(),
        on="prediction_id",
        how="left"
    )
    d["is_top"] = d["prediction_id"].astype(str) == str(top_pid) if top_pid is not None else False
    return d


def make_long_tm(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        pid = str(row["prediction_id"])
        is_top = bool(row.get("is_top", False))
        rank = row.get("ranking_score", np.nan)

        if "TM1" in df.columns and pd.notna(row.get("TM1")):
            rows.append({
                "prediction_id": pid,
                "is_top": is_top,
                "ranking_score": rank,
                "metric": "TM1",
                "tm_score": float(row["TM1"])
            })
        if "TM2" in df.columns and pd.notna(row.get("TM2")):
            rows.append({
                "prediction_id": pid,
                "is_top": is_top,
                "ranking_score": rank,
                "metric": "TM2",
                "tm_score": float(row["TM2"])
            })

    return pd.DataFrame(rows)


def _metadata_text(row: pd.Series) -> str:
    parts = []
    for col in ["IDali", "L1", "L2", "Lali"]:
        val = row.get(col, np.nan)
        if pd.notna(val):
            if col == "IDali":
                parts.append(f"{col}={float(val):.3f}")
            else:
                parts.append(f"{col}={int(val) if float(val).is_integer() else float(val):g}")
    return ", ".join(parts)


def plot_usalign_tm_and_rmsd(
    df: pd.DataFrame,
    outpath: Path,
    sample_id: str
) -> bool:
    """
    Panel 1: paired strip plot for TM1 and TM2
    Panel 2: RMSD bar plot (if RMSD available)

    Metadata (IDali, L1, L2, Lali) are annotated above each prediction.
    """
    if df.empty:
        return False

    d = df.copy()
    d["prediction_id"] = d["prediction_id"].astype(str)
    d["ranking_score"] = pd.to_numeric(d.get("ranking_score"), errors="coerce")
    d["is_top"] = d.get("is_top", False).fillna(False)

    d = d.sort_values(
        ["is_top", "ranking_score", "prediction_id"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    tm_long = make_long_tm(d)
    if tm_long.empty:
        return False

    order = d["prediction_id"].tolist()
    label_map = {
        pid: (f"TOP: {pid}" if bool(is_top) else pid)
        for pid, is_top in zip(d["prediction_id"], d["is_top"])
    }

    has_rmsd = "RMSD" in d.columns and d["RMSD"].notna().any()

    nrows = 2 if has_rmsd else 1
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=(max(8, 0.9 * len(order) + 4), 8 if has_rmsd else 5.5),
        squeeze=False
    )
    ax_tm = axes[0][0]

    # paired strip plot TM1/TM2
    sns.stripplot(
        data=tm_long,
        x="prediction_id",
        y="tm_score",
        hue="metric",
        order=order,
        dodge=True,
        jitter=False,
        size=7,
        palette={"TM1": "#1f77b4", "TM2": "#ff7f0e"},
        ax=ax_tm
    )

    # connect paired points
    pos = {pid: i for i, pid in enumerate(order)}
    offset = {"TM1": -0.18, "TM2": 0.18}
    for _, row in d.iterrows():
        pid = row["prediction_id"]
        vals = []
        if pd.notna(row.get("TM1")):
            vals.append((pos[pid] + offset["TM1"], float(row["TM1"])))
        if pd.notna(row.get("TM2")):
            vals.append((pos[pid] + offset["TM2"], float(row["TM2"])))
        if len(vals) == 2:
            xs, ys = zip(*vals)
            ax_tm.plot(xs, ys, color="gray", alpha=0.6, lw=1)

    ax_tm.set_ylim(0, 1.05)
    ax_tm.set_ylabel("TM-score")
    ax_tm.set_xlabel("prediction")
    ax_tm.set_title(f"US-align TM-scores: {sample_id}")

    ax_tm.set_xticks(np.arange(len(order)))
    ax_tm.set_xticklabels([label_map[pid] for pid in order], rotation=60, ha="right")

    # annotate metadata above each x position
    for i, (_, row) in enumerate(d.iterrows()):
        txt = _metadata_text(row)
        if txt:
            ax_tm.text(
                i, 1.02, txt,
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=8,
                transform=ax_tm.get_xaxis_transform()
            )

    ax_tm.legend(title="Metric", loc="upper left", bbox_to_anchor=(1.01, 1.0))

    if has_rmsd:
        ax_rmsd = axes[1][0]
        colors = ["#D55E00" if x else "#4C72B0" for x in d["is_top"]]
        x = np.arange(len(d))
        y = pd.to_numeric(d["RMSD"], errors="coerce").to_numpy(dtype=float)

        ax_rmsd.bar(x, y, color=colors, alpha=0.9)
        ax_rmsd.set_xticks(x)
        ax_rmsd.set_xticklabels([label_map[pid] for pid in order], rotation=60, ha="right")
        ax_rmsd.set_ylabel("RMSD")
        ax_rmsd.set_xlabel("prediction")
        ax_rmsd.set_title("US-align RMSD")

        ymax = np.nanmax(y) if np.isfinite(y).any() else 1.0
        ax_rmsd.set_ylim(0, ymax * 1.15 if ymax > 0 else 1.0)

    fig.subplots_adjust(hspace=0.45, top=0.88, right=0.82)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def df_to_html_table(df: pd.DataFrame, max_rows: int = 200) -> str:
    if df.empty:
        return "<p><em>No data.</em></p>"
    d = df.copy()
    note = ""
    if len(d) > max_rows:
        d = d.head(max_rows)
        note = f"<p><em>Showing first {max_rows} of {len(df)} rows.</em></p>"
    return note + d.to_html(index=False, escape=True, border=0, classes="table")


def write_html_report(out_html: Path, df: pd.DataFrame, plot_relpath: Optional[str], sample_id: str):
    css = """
    <style>
    body { font-family: Arial, sans-serif; margin: 24px; }
    h1,h2 { margin-top: 1.2em; }
    .table { border-collapse: collapse; font-size: 13px; }
    .table th, .table td { border: 1px solid #ddd; padding: 6px 8px; }
    .table th { background: #f5f5f5; }
    img { max-width: 100%; height: auto; border: 1px solid #eee; padding: 2px; }
    </style>
    """

    img_html = (
        f'<img src="{plot_relpath}" alt="US-align summary plot"/>'
        if plot_relpath else
        "<p><em>Plot not available.</em></p>"
    )

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>US-align report - {sample_id}</title>
{css}
</head>
<body>
<h1>US-align summary report</h1>
<p><strong>Sample:</strong> {sample_id}</p>

<h2>TM-score / RMSD summary</h2>
{img_html}

<h2>Parsed US-align results</h2>
{df_to_html_table(df, max_rows=500)}
</body>
</html>
"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("report_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("-o", "--outdir", type=click.Path(file_okay=False, path_type=Path), required=True,
              help="Output directory for US-align report.")
@click.option("--predictions-tsv", type=click.Path(exists=False, path_type=Path), default=None,
              help="Optional AF3 predictions.tsv used to annotate top-ranked prediction.")
@click.option("--html-name", default="usalign_report.html", show_default=True,
              help="HTML report filename.")
def main(report_dir: Path, outdir: Path, predictions_tsv: Optional[Path], html_name: str):
    """
    Generate a standalone US-align report from *.usalign.tsv files in REPORT_DIR.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    if predictions_tsv is None:
        predictions_tsv = report_dir / "predictions.tsv"

    df_pred = read_predictions(predictions_tsv) if predictions_tsv and predictions_tsv.exists() else pd.DataFrame()
    df_u = load_usalign_results(report_dir)
    df_m = merge_with_predictions(df_u, df_pred)

    out_tsv = outdir / "usalign_summary.tsv"
    df_m.to_csv(out_tsv, sep="\t", index=False)

    sample_id = report_dir.name
    plot_path = outdir / "plots" / "usalign_tm_rmsd.png"
    ok = plot_usalign_tm_and_rmsd(df_m, plot_path, sample_id=sample_id)
    plot_rel = str(plot_path.relative_to(outdir)) if ok else None

    out_html = outdir / html_name
    write_html_report(out_html, df_m, plot_rel, sample_id)

    click.echo(str(out_html))


if __name__ == "__main__":
    main()