#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Optional

import click
import pandas as pd


def read_report_samples(samplesheet: Path) -> pd.DataFrame:
    df = pd.read_csv(samplesheet, sep="\t", dtype=str).fillna("")
    required = {"sample_id", "af3_dir"}
    if not required.issubset(df.columns):
        raise ValueError(f"Report samplesheet must contain columns: {sorted(required)}")
    if "ground_truth" not in df.columns:
        df["ground_truth"] = ""
    return df


def load_optional_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t", dtype=str).fillna("")


def add_sample_id_if_missing(df: pd.DataFrame, sample_id: str) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    if "sample_id" not in d.columns:
        d.insert(0, "sample_id", sample_id)
    else:
        d["sample_id"] = d["sample_id"].replace("", sample_id)
    return d


def sample_status_table(df_samples: pd.DataFrame, af3_base: Path, usalign_base: Optional[Path]) -> pd.DataFrame:
    rows = []
    for _, row in df_samples.iterrows():
        sample_id = str(row["sample_id"])
        af3_dir = str(row.get("af3_dir", ""))
        ground_truth = str(row.get("ground_truth", "")).strip()

        af3_sample_dir = af3_base / sample_id
        pred_path = af3_sample_dir / "predictions.tsv"
        chains_path = af3_sample_dir / "chains.tsv"
        pairs_path = af3_sample_dir / "chain_pairs.tsv"

        has_ground_truth = bool(ground_truth)
        usalign_expected = has_ground_truth

        usalign_sample_dir = usalign_base / sample_id if usalign_base is not None else None
        usalign_summary = usalign_sample_dir / "usalign_summary.tsv" if usalign_sample_dir is not None else None

        rows.append({
            "sample_id": sample_id,
            "af3_dir": af3_dir,
            "ground_truth": ground_truth,
            "has_ground_truth": has_ground_truth,
            "usalign_expected": usalign_expected,
            "af3_report_dir": str(af3_sample_dir),
            "af3_predictions_tsv": str(pred_path),
            "af3_chains_tsv": str(chains_path),
            "af3_chain_pairs_tsv": str(pairs_path),
            "af3_predictions_found": pred_path.exists(),
            "af3_chains_found": chains_path.exists(),
            "af3_chain_pairs_found": pairs_path.exists(),
            "usalign_report_dir": str(usalign_sample_dir) if usalign_sample_dir is not None else "",
            "usalign_summary_tsv": str(usalign_summary) if usalign_summary is not None else "",
            "usalign_found": bool(usalign_summary.exists()) if usalign_summary is not None else False,
            "usalign_not_applicable": not usalign_expected,
            "usalign_missing_expected": bool(usalign_expected and (usalign_summary is None or not usalign_summary.exists())),
        })

    return pd.DataFrame(rows)


def add_report_paths(
    df: pd.DataFrame,
    sample_id: str,
    af3_base: Path,
    usalign_base: Optional[Path],
) -> pd.DataFrame:
    if df.empty:
        return df

    d = df.copy()
    d["af3_report_dir"] = str(af3_base / sample_id)

    af3_plots = af3_base / sample_id / "plots"
    d["af3_predictions_tsv"] = str(af3_base / sample_id / "predictions.tsv")
    d["af3_chains_tsv"] = str(af3_base / sample_id / "chains.tsv")
    d["af3_chain_pairs_tsv"] = str(af3_base / sample_id / "chain_pairs.tsv")
    d["af3_plot_plddt_combined"] = str(af3_plots / "plddt_combined.html")
    d["af3_plot_iptm_interactive"] = str(af3_plots / "iptm_interactive.html")
    d["af3_plot_pae_multipanel"] = str(af3_plots / "pae_multipanel.png")
    d["af3_plot_chain_plddt_multipanel"] = str(af3_plots / "chain_plddt_multipanel.png")
    d["af3_plot_plddt_by_prediction"] = str(af3_plots / "plddt_by_prediction.png")
    d["af3_plot_ranking_by_prediction"] = str(af3_plots / "ranking_by_prediction.png")

    if usalign_base is not None:
        udir = usalign_base / sample_id
        uplots = udir / "plots"
        d["usalign_report_dir"] = str(udir)
        d["usalign_summary_tsv"] = str(udir / "usalign_summary.tsv")
        d["usalign_plot_tm_rmsd"] = str(uplots / "usalign_tm_rmsd_interactive.html")

    return d


def collect_af3_predictions(df_samples: pd.DataFrame, af3_base: Path, usalign_base: Optional[Path]) -> pd.DataFrame:
    parts = []
    for _, row in df_samples.iterrows():
        sample_id = str(row["sample_id"])
        p = af3_base / sample_id / "predictions.tsv"
        df = load_optional_tsv(p)
        if df.empty:
            continue

        df = add_sample_id_if_missing(df, sample_id)
        df = add_report_paths(df, sample_id, af3_base, usalign_base)

        df["af3_dir"] = str(row.get("af3_dir", ""))
        df["ground_truth"] = str(row.get("ground_truth", ""))
        df["has_ground_truth"] = bool(str(row.get("ground_truth", "")).strip())
        df["usalign_expected"] = df["has_ground_truth"]

        if usalign_base is not None:
            usalign_summary = usalign_base / sample_id / "usalign_summary.tsv"
            df["usalign_found"] = usalign_summary.exists()
            df["usalign_not_applicable"] = ~df["usalign_expected"]
            df["usalign_missing_expected"] = df["usalign_expected"] & (~df["usalign_found"])
        else:
            df["usalign_found"] = False
            df["usalign_not_applicable"] = ~df["usalign_expected"]
            df["usalign_missing_expected"] = df["usalign_expected"]

        parts.append(df)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def collect_af3_chains(df_samples: pd.DataFrame, af3_base: Path) -> pd.DataFrame:
    parts = []
    for _, row in df_samples.iterrows():
        sample_id = str(row["sample_id"])
        p = af3_base / sample_id / "chains.tsv"
        df = load_optional_tsv(p)
        if df.empty:
            continue

        df = add_sample_id_if_missing(df, sample_id)
        df["af3_dir"] = str(row.get("af3_dir", ""))
        df["ground_truth"] = str(row.get("ground_truth", ""))
        df["has_ground_truth"] = bool(str(row.get("ground_truth", "")).strip())
        df["usalign_expected"] = df["has_ground_truth"]

        parts.append(df)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def collect_af3_chain_pairs(df_samples: pd.DataFrame, af3_base: Path) -> pd.DataFrame:
    parts = []
    for _, row in df_samples.iterrows():
        sample_id = str(row["sample_id"])
        p = af3_base / sample_id / "chain_pairs.tsv"
        df = load_optional_tsv(p)
        if df.empty:
            continue

        df = add_sample_id_if_missing(df, sample_id)
        df["af3_dir"] = str(row.get("af3_dir", ""))
        df["ground_truth"] = str(row.get("ground_truth", ""))
        df["has_ground_truth"] = bool(str(row.get("ground_truth", "")).strip())
        df["usalign_expected"] = df["has_ground_truth"]

        parts.append(df)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def collect_usalign(df_samples: pd.DataFrame, usalign_base: Optional[Path]) -> pd.DataFrame:
    if usalign_base is None or not usalign_base.exists():
        return pd.DataFrame()

    parts = []
    for _, row in df_samples.iterrows():
        sample_id = str(row["sample_id"])
        ground_truth = str(row.get("ground_truth", "")).strip()

        if not ground_truth:
            continue

        p = usalign_base / sample_id / "usalign_summary.tsv"
        df = load_optional_tsv(p)
        if df.empty:
            continue

        df = add_sample_id_if_missing(df, sample_id)
        df["af3_dir"] = str(row.get("af3_dir", ""))
        df["ground_truth"] = ground_truth
        df["has_ground_truth"] = True
        df["usalign_expected"] = True
        df["usalign_found"] = True
        df["usalign_not_applicable"] = False
        df["usalign_missing_expected"] = False
        df["usalign_report_dir"] = str(usalign_base / sample_id)
        df["usalign_plot_tm_rmsd"] = str(usalign_base / sample_id / "plots" / "usalign_tm_rmsd_interactive.html")

        parts.append(df)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--report-samples",
    "report_samples_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="TSV with sample_id, af3_dir, and optional ground_truth."
)
@click.option(
    "--af3-base",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("reports/alphafold3"),
    show_default=True,
    help="Base directory containing per-sample AF3 report outputs."
)
@click.option(
    "--usalign-base",
    type=click.Path(exists=False, file_okay=False, path_type=Path),
    default=Path("reports/usalign"),
    show_default=True,
    help="Base directory containing per-sample US-align report outputs. Optional."
)
@click.option(
    "-o", "--outdir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Output directory for cohort tables."
)
def main(
    report_samples_path: Path,
    af3_base: Path,
    usalign_base: Optional[Path],
    outdir: Path
):
    """
    Aggregate AF3 and optional US-align tables across all sample report directories.

    Outputs:
      - cohort_predictions.tsv
      - cohort_chains.tsv
      - cohort_chain_pairs.tsv
      - optional cohort_usalign.tsv
      - cohort_sample_status.tsv
      - cohort_master.tsv

    cohort_master.tsv is a prediction-level unified table built from predictions.tsv
    plus report paths and optional per-prediction US-align metrics.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    df_samples = read_report_samples(report_samples_path)

    use_usalign = usalign_base is not None and Path(usalign_base).exists()
    usalign_base_eff = Path(usalign_base) if use_usalign else None

    df_status = sample_status_table(df_samples, af3_base=af3_base, usalign_base=usalign_base_eff)
    df_pred = collect_af3_predictions(df_samples, af3_base=af3_base, usalign_base=usalign_base_eff)
    df_chain = collect_af3_chains(df_samples, af3_base=af3_base)
    df_pair = collect_af3_chain_pairs(df_samples, af3_base=af3_base)
    df_usalign = collect_usalign(df_samples, usalign_base=usalign_base_eff)

    df_status.to_csv(outdir / "cohort_sample_status.tsv", sep="\t", index=False)
    df_pred.to_csv(outdir / "cohort_predictions.tsv", sep="\t", index=False)
    df_chain.to_csv(outdir / "cohort_chains.tsv", sep="\t", index=False)
    df_pair.to_csv(outdir / "cohort_chain_pairs.tsv", sep="\t", index=False)

    if not df_usalign.empty:
        df_usalign.to_csv(outdir / "cohort_usalign.tsv", sep="\t", index=False)

    master = df_pred.copy()

    if not master.empty:
        status_keep = [
            "sample_id",
            "has_ground_truth",
            "usalign_expected",
            "usalign_found",
            "usalign_not_applicable",
            "usalign_missing_expected",
        ]
        master = master.merge(
            df_status[status_keep].drop_duplicates(),
            on="sample_id",
            how="left",
            suffixes=("", "_status")
        )

    if not master.empty and not df_usalign.empty:
        d_u = df_usalign.copy()

        if "prediction_id" not in d_u.columns and "usalign_id" in d_u.columns:
            d_u["prediction_id"] = d_u["usalign_id"].astype(str)

        u_keep = [
            "sample_id",
            "prediction_id",
            "TM1",
            "TM2",
            "RMSD",
            "ID1",
            "ID2",
            "IDali",
            "L1",
            "L2",
            "Lali",
            "usalign_report_dir",
            "usalign_plot_tm_rmsd",
        ]
        u_keep = [c for c in u_keep if c in d_u.columns]
        d_u = d_u[u_keep].drop_duplicates()

        master = master.merge(
            d_u,
            on=["sample_id", "prediction_id"],
            how="left"
        )

    preferred = [
        "sample_id",
        "prediction_id",
        "sample_pred_id",
        "seed",
        "sample",
        "is_top",
        "ranking_score",
        "ptm",
        "iptm",
        "fraction_disordered",
        "has_clash",
        "mean_plddt_total",
        "std_plddt_total",
        "has_ground_truth",
        "usalign_expected",
        "usalign_found",
        "usalign_not_applicable",
        "usalign_missing_expected",
        "TM1",
        "TM2",
        "RMSD",
        "ID1",
        "ID2",
        "IDali",
        "L1",
        "L2",
        "Lali",
        "af3_dir",
        "ground_truth",
        "summary_path",
        "confidences_path",
        "af3_report_dir",
        "af3_predictions_tsv",
        "af3_chains_tsv",
        "af3_chain_pairs_tsv",
        "af3_plot_plddt_combined",
        "af3_plot_iptm_interactive",
        "af3_plot_pae_multipanel",
        "af3_plot_chain_plddt_multipanel",
        "af3_plot_plddt_by_prediction",
        "af3_plot_ranking_by_prediction",
        "usalign_report_dir",
        "usalign_summary_tsv",
        "usalign_plot_tm_rmsd",
    ]
    cols = [c for c in preferred if c in master.columns] + [c for c in master.columns if c not in preferred]
    master = master[cols]

    master.to_csv(outdir / "cohort_master.tsv", sep="\t", index=False)

    click.echo(str(outdir / "cohort_master.tsv"))


if __name__ == "__main__":
    main()