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
import seaborn as sns


SEED_SAMPLE_RE = re.compile(r"seed-(\d+)_sample-(\d+)")

def plot_pae_with_chain_breaks(conf: dict, outpath: Path, title: str = "PAE"):
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
        seed, sample, pred_id = parse_prediction_id(sp)
        summ = load_json(sp)

        cp = resolve_confidences_path(sp, layout)
        conf = load_json(cp) if cp and cp.exists() else {}

        # ---- complex-wide ----
        pred_rows.append({
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

        # label chains using atom_chain_ids if present
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

            # fall back if chain labels don't match
            if len(chain_ids) != mat_iptm.shape[0]:
                chain_ids_pair = [str(i) for i in range(mat_iptm.shape[0])]
            else:
                chain_ids_pair = chain_ids

            for i, ci in enumerate(chain_ids_pair):
                for j, cj in enumerate(chain_ids_pair):
                    #if i == j:
                    #    continue
                    pair_rows.append({
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


def plot_chain_bars(df_chain: pd.DataFrame, outdir: Path) -> dict[str, str]:
    plots = {}
    if df_chain.empty:
        return plots

    # take top prediction if present (best model) else first
    if (df_chain["prediction_id"] == "top").any():
        d = df_chain[df_chain["prediction_id"] == "top"].copy()
        title = "Per-chain mean pLDDT (top)"
        fname = "chain_plddt_top.png"
    else:
        d = df_chain[df_chain["prediction_id"] == df_chain["prediction_id"].iloc[0]].copy()
        title = f"Per-chain mean pLDDT ({d['prediction_id'].iloc[0]})"
        fname = "chain_plddt_first.png"

    plt.figure(figsize=(max(6, 0.6 * len(d)), 4.2))
    sns.barplot(data=d, x="chain_id", y="mean_plddt_chain", color="#4C72B0")
    plt.title(title)
    plt.xlabel("chain")
    plt.ylabel("mean pLDDT (atom mean)")
    p = outdir / "plots" / fname
    save_fig(p)
    plots["chain_plddt_bar"] = str(p.relative_to(outdir))
    return plots


def plot_pair_heatmap(df_pair: pd.DataFrame, outdir: Path) -> dict[str, str]:
    plots = {}
    if df_pair.empty:
        return plots

    # heatmap for top prediction if available
    if (df_pair["prediction_id"] == "top").any():
        d = df_pair[df_pair["prediction_id"] == "top"].copy()
        suffix = "top"
    else:
        d = df_pair[df_pair["prediction_id"] == df_pair["prediction_id"].iloc[0]].copy()
        suffix = "first"
    piv = d.pivot(index="chain_i", columns="chain_j", values="pair_iptm")
#    piv = d.pivot_table(index="chain_i", columns="chain_j", values="pair_iptm", aggfunc="mean")

    plt.figure(figsize=(6, 5))
    sns.heatmap(piv, vmin=0, vmax=1, cmap="viridis", square=True, cbar_kws={"label": "pair ipTM"})
    plt.title(f"chain_pair_iptm ({suffix})")
    p = outdir / "plots" / f"pair_iptm_heatmap_{suffix}.png"
    save_fig(p)
    plots["pair_iptm_heatmap"] = str(p.relative_to(outdir))
    return plots


def df_to_html_table(df: pd.DataFrame, max_rows: int) -> str:
    if df.empty:
        return "<p><em>No data.</em></p>"
    d = df.copy()
    if len(d) > max_rows:
        d = d.head(max_rows)
        note = f"<p><em>Showing first {max_rows} rows.</em></p>"
    else:
        note = ""
    return note + d.to_html(index=False, escape=True, classes="table", border=0)


def write_html_report(out_html: Path,
                      df_pred: pd.DataFrame,
                      df_chain: pd.DataFrame,
                      df_pair: pd.DataFrame,
                      plots: dict[str, str],
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
<div class="grid">
  <div>{img("ranking_by_prediction")}</div>
  <div>{img("plddt_by_prediction")}</div>
</div>
{df_to_html_table(df_pred.sort_values(["prediction_id"], ascending=True), max_rows=max_rows)}

<h2>Per-chain metrics</h2>
{img("chain_plddt_bar")}
{df_to_html_table(df_chain.sort_values(["prediction_id","chain_id"]), max_rows=max_rows)}

<h2>Per-interface (chain-pair) metrics</h2>
{img("pair_iptm_heatmap")}
{df_to_html_table(df_pair.sort_values(["prediction_id","pair_iptm"], ascending=[True, False]), max_rows=max_rows)}
<h2>PAE matrix</h2>
{img("pae_with_chain_breaks")}
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

    plots = {}
    plots.update(plot_complex_overview(df_pred, outdir))
    plots.update(plot_chain_bars(df_chain, outdir))
    plots.update(plot_pair_heatmap(df_pair, outdir))
    conf_path = pick_conf_for_plot(df_pred)
    if conf_path and conf_path.exists():
        conf = load_json(conf_path)
        pae_png = outdir / "plots" / "pae_with_chain_breaks.png"
        ok = plot_pae_with_chain_breaks(conf, pae_png, title="PAE (with chain breaks)")
        if ok:
            plots["pae_with_chain_breaks"] = str(pae_png.relative_to(outdir))

    out_html = outdir / html_name
    write_html_report(out_html, df_pred, df_chain, df_pair, plots, max_rows=max_rows)

    click.echo(str(out_html))


if __name__ == "__main__":
    main()
