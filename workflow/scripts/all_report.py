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


def collect_af3_predictions(df_samples: pd.DataFrame, af3_base: Path) -> pd.DataFrame:
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
        parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_master(
    df_samples: pd.DataFrame,
    df_pred: pd.DataFrame,
    df_usalign: pd.DataFrame,
    df_status: pd.DataFrame,
    af3_base: Path,
    usalign_base: Optional[Path],
) -> pd.DataFrame:
    master = df_pred.copy()
    if master.empty:
        return master

    # --- Drop any pre-existing empty ground_truth_id to avoid _x/_y collisions ---
    if "ground_truth_id" in master.columns:
        non_empty = master["ground_truth_id"].astype(str).replace("nan", "").str.strip()
        if (non_empty == "").all():
            master = master.drop(columns=["ground_truth_id"])

    # Merge sample metadata / status
    status_keep = [
        "sample_id", "af3_dir", "ground_truth", "has_ground_truth",
        "usalign_expected", "usalign_found", "usalign_not_applicable",
        "usalign_missing_expected", "af3_report_dir", "af3_predictions_tsv",
        "af3_chains_tsv", "af3_chain_pairs_tsv", "usalign_report_dir",
        "usalign_summary_tsv",
    ]
    status_keep = [c for c in status_keep if c in df_status.columns]
    master = master.merge(
        df_status[status_keep].drop_duplicates(),
        on="sample_id", how="left"
    )

    # Add deterministic AF3 plot paths
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

    # --- Merge US-align ---
    # US-align is run per sample (one model CIF vs each reference), so the
    # usalign prediction_id equals the sample_id, NOT the per-seed/sample
    # prediction_id.  We need to:
    #   1. Extract ground_truth_id from the usalign_id / ground_truth_id col
    #   2. Extract the actual per-prediction prediction_id from the usalign
    #      prediction_id column (which may encode seed+sample info) OR
    #      merge on sample_id alone and cross-join with all predictions.
    #
    # From the data we see:
    #   usalign prediction_id = "5qj0_rna_mn_modeller_seed-1" (= sample_id)
    #   usalign ground_truth_id = "4WTJ_pdb" (already extracted)
    #   usalign usalign_id = "5qj0_rna_mn_modeller_seed-1_ref-4WTJ_pdb"
    #
    # The usalign was run on the *sample-level* model CIF, so the same TM
    # scores apply to every prediction (seed-X_sample-Y) within that sample.
    # We merge on sample_id only, producing one row per (prediction, gt_ref).

    if not df_usalign.empty:
        d_u = df_usalign.copy()

        # Ensure ground_truth_id exists and is populated
        if "ground_truth_id" not in d_u.columns or d_u["ground_truth_id"].astype(str).replace("nan", "").str.strip().eq("").all():
            # Try extracting from usalign_id
            if "usalign_id" in d_u.columns:
                d_u["ground_truth_id"] = (
                    d_u["usalign_id"].astype(str)
                    .str.extract(r"_ref-(.+)$", expand=False)
                    .fillna("")
                )
            # Fallback: derive from PDBchain2 / ground_truth path column
            if "ground_truth_id" not in d_u.columns or d_u["ground_truth_id"].eq("").all():
                for col_candidate in ["PDBchain2", "ground_truth"]:
                    if col_candidate in d_u.columns:
                        d_u["ground_truth_id"] = (
                            d_u[col_candidate].astype(str)
                            .apply(lambda p: Path(p.split(":")[0]).stem if p and p != "nan" else "")
                        )
                        if not d_u["ground_truth_id"].eq("").all():
                            break

        # Replace remaining blanks
        if "ground_truth_id" in d_u.columns:
            d_u["ground_truth_id"] = (
                d_u["ground_truth_id"].astype(str)
                .replace("nan", "").replace("", "default").str.strip()
            )
            d_u.loc[d_u["ground_truth_id"] == "", "ground_truth_id"] = "default"
        else:
            d_u["ground_truth_id"] = "default"

        # Keep only the columns we need for the merge
        u_keep = [
            "sample_id", "ground_truth_id",
            "TM1", "TM2", "RMSD", "ID1", "ID2", "IDali",
            "L1", "L2", "Lali",
        ]
        u_keep = [c for c in u_keep if c in d_u.columns]
        d_u = d_u[u_keep].drop_duplicates()

        # Merge on sample_id only — each prediction in the sample gets
        # one row per ground-truth reference (cross-join within sample)
        master = master.merge(d_u, on="sample_id", how="left")

    # Final fallback for ground_truth_id
    if "ground_truth_id" not in master.columns:
        master["ground_truth_id"] = "default"
    else:
        master["ground_truth_id"] = (
            master["ground_truth_id"].astype(str)
            .replace("", "default").replace("nan", "default")
        )

    # Add deterministic US-align plot path
    if usalign_base is not None:
        usalign_plot_rows = []
        for sample_id in master["sample_id"].astype(str).drop_duplicates():
            udir = usalign_base / sample_id
            usalign_plot_rows.append({
                "sample_id": sample_id,
                "usalign_plot_tm_rmsd": str(udir / "plots" / "usalign_tm_rmsd_interactive.html"),
            })
        df_usalign_plots = pd.DataFrame(usalign_plot_rows)
        master = master.merge(df_usalign_plots, on="sample_id", how="left")

    return master


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--report-samples", "report_samples_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="TSV with sample_id, af3_dir, and optional ground_truth."
)
@click.option(
    "--af3-base",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("reports/alphafold3"), show_default=True,
    help="Base directory containing per-sample AF3 report outputs."
)
@click.option(
    "--usalign-base",
    type=click.Path(exists=False, file_okay=False, path_type=Path),
    default=Path("reports/usalign"), show_default=True,
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
      - all_chain_pairs.tsv        (aggregation only)
      - all_chains.tsv            (aggregation only)
      - all_predictions.tsv       (aggregation only)
      - all_usalign.tsv           (aggregation only)
      - all_sample_status.tsv     (derived sample-level merged/status table)
      - all_master.tsv            (merged prediction-level table)
    """
    outdir.mkdir(parents=True, exist_ok=True)

    df_samples = read_report_samples(report_samples_path)

    use_usalign = usalign_base is not None and Path(usalign_base).exists()
    usalign_base_eff = Path(usalign_base) if use_usalign else None

    # Aggregation-only tables
    df_pred = collect_af3_predictions(df_samples, af3_base=af3_base)
    df_chain = collect_af3_chains(df_samples, af3_base=af3_base)
    df_pair = collect_af3_chain_pairs(df_samples, af3_base=af3_base)
    df_usalign = collect_usalign(df_samples, usalign_base=usalign_base_eff)

    # Derived/merged tables
    df_status = sample_status_table(df_samples, af3_base=af3_base, usalign_base=usalign_base_eff)
    master = build_master(
        df_samples=df_samples,
        df_pred=df_pred,
        df_usalign=df_usalign,
        df_status=df_status,
        af3_base=af3_base,
        usalign_base=usalign_base_eff,
    )

    # Write aggregation-only tables
    df_pair.to_csv(outdir / "all_chain_pairs.tsv", sep="\t", index=False)
    df_chain.to_csv(outdir / "all_chains.tsv", sep="\t", index=False)
    df_pred.to_csv(outdir / "all_predictions.tsv", sep="\t", index=False)
    df_usalign.to_csv(outdir / "all_usalign.tsv", sep="\t", index=False)

    # Write merged/derived tables
    df_status.to_csv(outdir / "all_sample_status.tsv", sep="\t", index=False)

    preferred = [
        "sample_id", "prediction_id", "sample_pred_id", "description",
        "ground_truth_id", "seed", "sample", "is_top", "ranking_score",
        "ptm", "iptm", "fraction_disordered", "has_clash",
        "mean_plddt_total", "std_plddt_total",
        "TM1", "TM2", "RMSD", "ID1", "ID2", "IDali", "L1", "L2", "Lali",
        "af3_dir", "ground_truth", "has_ground_truth",
        "usalign_expected", "usalign_found", "usalign_not_applicable",
        "usalign_missing_expected", "summary_path", "confidences_path",
        "af3_report_dir", "af3_predictions_tsv", "af3_chains_tsv",
        "af3_chain_pairs_tsv", "af3_plot_plddt_combined",
        "af3_plot_iptm_interactive", "af3_plot_pae_multipanel",
        "af3_plot_chain_plddt_multipanel", "af3_plot_plddt_by_prediction",
        "af3_plot_ranking_by_prediction", "usalign_report_dir",
        "usalign_summary_tsv", "usalign_plot_tm_rmsd",
    ]
    cols = [c for c in preferred if c in master.columns] + [
        c for c in master.columns if c not in preferred
    ]
    master = master[cols]
    master.to_csv(outdir / "all_master.tsv", sep="\t", index=False)

    click.echo(str(outdir / "all_master.tsv"))


if __name__ == "__main__":
    main()