#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import click
import pandas as pd


def read_report_samples(samplesheet: Path) -> pd.DataFrame:
    df = pd.read_csv(samplesheet, sep="\t", dtype=str).fillna("")
    required = {"sample_id", "af3_dir"}
    if not required.issubset(df.columns):
        raise ValueError(f"Report samplesheet must contain columns: {sorted(required)}")

    if "ground_truth" not in df.columns:
        df["ground_truth"] = ""

    # Optional explicit US-align report location column
    # e.g. reports/usalign/<sample_id>
    if "usalign_dir" not in df.columns:
        df["usalign_dir"] = ""

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


def sample_status_table(df_samples: pd.DataFrame, af3_base: Path) -> pd.DataFrame:
    """
    Derived sample-level merged/status table.
    US-align expectation is driven by ground_truth.
    US-align location is taken from optional samplesheet column usalign_dir.
    """
    rows = []
    for _, row in df_samples.iterrows():
        sample_id = str(row["sample_id"])
        af3_dir = str(row.get("af3_dir", ""))
        ground_truth = str(row.get("ground_truth", "")).strip()
        usalign_dir = str(row.get("usalign_dir", "")).strip()

        af3_sample_dir = af3_base / sample_id
        pred_path = af3_sample_dir / "predictions.tsv"
        chains_path = af3_sample_dir / "chains.tsv"
        pairs_path = af3_sample_dir / "chain_pairs.tsv"

        has_ground_truth = bool(ground_truth)
        usalign_expected = has_ground_truth

        usalign_summary = Path(usalign_dir) / "usalign_summary.tsv" if usalign_dir else None

        rows.append({
            "sample_id": sample_id,
            "af3_dir": af3_dir,
            "ground_truth": ground_truth,
            "usalign_dir": usalign_dir,
            "has_ground_truth": has_ground_truth,
            "usalign_expected": usalign_expected,
            "af3_report_dir": str(af3_sample_dir),
            "af3_predictions_tsv": str(pred_path),
            "af3_chains_tsv": str(chains_path),
            "af3_chain_pairs_tsv": str(pairs_path),
            "af3_predictions_found": pred_path.exists(),
            "af3_chains_found": chains_path.exists(),
            "af3_chain_pairs_found": pairs_path.exists(),
            "usalign_report_dir": usalign_dir,
            "usalign_summary_tsv": str(usalign_summary) if usalign_summary is not None else "",
            "usalign_found": bool(usalign_summary.exists()) if usalign_summary is not None else False,
            "usalign_not_applicable": not usalign_expected,
            "usalign_missing_expected": bool(usalign_expected and (usalign_summary is None or not usalign_summary.exists())),
        })

    return pd.DataFrame(rows)


def collect_af3_predictions(df_samples: pd.DataFrame, af3_base: Path) -> pd.DataFrame:
    """
    Pure aggregation only. No merge with status or US-align.
    """
    parts = []
    for _, row in df_samples.iterrows():
        sample_id = str(row["sample_id"])
        p = af3_base / sample_id / "predictions.tsv"
        df = load_optional_tsv(p)
        if df.empty:
            continue

        df = add_sample_id_if_missing(df, sample_id)
        parts.append(df)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def collect_af3_chains(df_samples: pd.DataFrame, af3_base: Path) -> pd.DataFrame:
    """
    Pure aggregation only.
    """
    parts = []
    for _, row in df_samples.iterrows():
        sample_id = str(row["sample_id"])
        p = af3_base / sample_id / "chains.tsv"
        df = load_optional_tsv(p)
        if df.empty:
            continue

        df = add_sample_id_if_missing(df, sample_id)
        parts.append(df)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def collect_af3_chain_pairs(df_samples: pd.DataFrame, af3_base: Path) -> pd.DataFrame:
    """
    Pure aggregation only.
    """
    parts = []
    for _, row in df_samples.iterrows():
        sample_id = str(row["sample_id"])
        p = af3_base / sample_id / "chain_pairs.tsv"
        df = load_optional_tsv(p)
        if df.empty:
            continue

        df = add_sample_id_if_missing(df, sample_id)
        parts.append(df)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def collect_usalign(df_samples: pd.DataFrame) -> pd.DataFrame:
    """
    Pure aggregation only. No merge with AF3 predictions.
    Only aggregate samples with non-empty ground_truth and existing usalign_dir/usalign_summary.tsv.
    """
    parts = []
    for _, row in df_samples.iterrows():
        sample_id = str(row["sample_id"])
        ground_truth = str(row.get("ground_truth", "")).strip()
        usalign_dir = str(row.get("usalign_dir", "")).strip()

        if not ground_truth or not usalign_dir:
            continue

        p = Path(usalign_dir) / "usalign_summary.tsv"
        df = load_optional_tsv(p)
        if df.empty:
            continue

        df = add_sample_id_if_missing(df, sample_id)
        parts.append(df)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_master(
    df_pred: pd.DataFrame,
    df_usalign: pd.DataFrame,
    df_status: pd.DataFrame,
    af3_base: Path,
) -> pd.DataFrame:
    """
    cohort_master is the only prediction-level merged table:
      AF3 predictions + optional US-align + sample-level status/path metadata
    """
    master = df_pred.copy()
    if master.empty:
        return master

    # Merge sample metadata / status once
    status_keep = [
        "sample_id",
        "af3_dir",
        "ground_truth",
        "usalign_dir",
        "has_ground_truth",
        "usalign_expected",
        "usalign_found",
        "usalign_not_applicable",
        "usalign_missing_expected",
        "af3_report_dir",
        "af3_predictions_tsv",
        "af3_chains_tsv",
        "af3_chain_pairs_tsv",
        "usalign_report_dir",
        "usalign_summary_tsv",
    ]
    status_keep = [c for c in status_keep if c in df_status.columns]

    master = master.merge(
        df_status[status_keep].drop_duplicates(),
        on="sample_id",
        how="left"
    )

    # Add deterministic AF3 plot paths after merge
    af3_plot_rows = []
    for sample_id in master["sample_id"].astype(str).drop_duplicates():
        plot_dir = af3_base / sample_id / "plots"
        af3_plot_rows.append({
            "sample_id": sample_id,
            "af3_plot_plddt_combined": str(plot_dir / "plddt_combined.html"),
            "af3_plot_iptm_interactive": str(plot_dir / "iptm_interactive.html"),
            "af3_plot_pae_multipanel": str(plot_dir / "pae_multipanel.png"),
            "af3_plot_chain_plddt_multipanel": str(plot_dir / "chain_plddt_multipanel.png"),
            "af3_plot_plddt_by_prediction": str(plot_dir / "plddt_by_prediction.png"),
            "af3_plot_ranking_by_prediction": str(plot_dir / "ranking_by_prediction.png"),
        })
    df_af3_plots = pd.DataFrame(af3_plot_rows)

    master = master.merge(df_af3_plots, on="sample_id", how="left")

    # Merge optional US-align only into master
    if not df_usalign.empty:
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
        ]
        u_keep = [c for c in u_keep if c in d_u.columns]
        d_u = d_u[u_keep].drop_duplicates()

        master = master.merge(
            d_u,
            on=["sample_id", "prediction_id"],
            how="left"
        )

    # Add deterministic US-align plot path from explicit usalign_dir, not from a CLI flag
    if "usalign_dir" in master.columns:
        plot_rows = []
        for sample_id, usalign_dir in (
            master[["sample_id", "usalign_dir"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        ):
            usalign_dir = str(usalign_dir).strip()
            plot_rows.append({
                "sample_id": sample_id,
                "usalign_dir": usalign_dir,
                "usalign_plot_tm_rmsd": str(Path(usalign_dir) / "plots" / "usalign_tm_rmsd_interactive.html") if usalign_dir else "",
            })
        df_usalign_plots = pd.DataFrame(plot_rows)
        master = master.merge(df_usalign_plots, on=["sample_id", "usalign_dir"], how="left")

    return master


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--report-samples",
    "report_samples_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="TSV with sample_id, af3_dir, optional ground_truth, and optional usalign_dir."
)
@click.option(
    "--af3-base",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("reports/alphafold3"),
    show_default=True,
    help="Base directory containing per-sample AF3 report outputs."
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
    outdir: Path
):
    """
    Aggregate AF3 and optional US-align tables across all sample report directories.

    Inputs:
      - report_samples.tsv with columns:
          sample_id
          af3_dir
          optional ground_truth
          optional usalign_dir

    Outputs:
      - cohort_chain_pairs.tsv        (aggregation only)
      - cohort_chains.tsv            (aggregation only)
      - cohort_predictions.tsv       (aggregation only)
      - cohort_usalign.tsv           (aggregation only)
      - cohort_sample_status.tsv     (derived sample-level merged/status table)
      - cohort_master.tsv            (merged prediction-level table)

    Only cohort_master.tsv and cohort_sample_status.tsv are merge/derived tables.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    df_samples = read_report_samples(report_samples_path)

    # Aggregation-only tables
    df_pred = collect_af3_predictions(df_samples, af3_base=af3_base)
    df_chain = collect_af3_chains(df_samples, af3_base=af3_base)
    df_pair = collect_af3_chain_pairs(df_samples, af3_base=af3_base)
    df_usalign = collect_usalign(df_samples)

    # Derived/merged tables
    df_status = sample_status_table(df_samples, af3_base=af3_base)
    master = build_master(
        df_pred=df_pred,
        df_usalign=df_usalign,
        df_status=df_status,
        af3_base=af3_base,
    )

    # Write aggregation-only tables
    df_pair.to_csv(outdir / "cohort_chain_pairs.tsv", sep="\t", index=False)
    df_chain.to_csv(outdir / "cohort_chains.tsv", sep="\t", index=False)
    df_pred.to_csv(outdir / "cohort_predictions.tsv", sep="\t", index=False)
    df_usalign.to_csv(outdir / "cohort_usalign.tsv", sep="\t", index=False)

    # Write merged/derived tables
    df_status.to_csv(outdir / "cohort_sample_status.tsv", sep="\t", index=False)

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
        "usalign_dir",
        "has_ground_truth",
        "usalign_expected",
        "usalign_found",
        "usalign_not_applicable",
        "usalign_missing_expected",
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