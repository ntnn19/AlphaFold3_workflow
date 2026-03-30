#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Optional

import click
import numpy as np
import pandas as pd


def normalize_job_name(s: str) -> str:
    s = str(s).strip()
    s = s.replace("-", "_")
    return s.lower()


def derive_job_name_from_sample_id(sample_id: str) -> str:
    """
    Example:
      7wr6_template_based_afdb_seed-1 -> 7wr6_template_based_afdb
      8sm3_template_free_afdb_seed-1  -> 8sm3_template_free_afdb
    """
    s = str(sample_id).strip()
    s = s.replace("-", "_")
    s = s.lower()

    if "_seed_" in s:
        s = s.split("_seed_")[0]
    return s


def read_report_samples(samplesheet: Path) -> pd.DataFrame:
    df = pd.read_csv(samplesheet, sep="\t", dtype=str).fillna("")
    required = {"sample_id", "af3_dir"}
    if not required.issubset(df.columns):
        raise ValueError(f"Report samplesheet must contain columns: {sorted(required)}")
    if "ground_truth" not in df.columns:
        df["ground_truth"] = ""
    return df


def read_chain_annotations(samplesheet: Path) -> pd.DataFrame:
    df = pd.read_csv(samplesheet, sep="\t", dtype=str).fillna("")
    required = {"job_name", "id"}
    if not required.issubset(df.columns):
        raise ValueError(f"Chain annotation samplesheet must contain columns: {sorted(required)}")
    return df


def load_optional_tsv(path: Path, sep: str = "\t") -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep=sep, dtype=str).fillna("")


def top_prediction_rows(df_pred: pd.DataFrame) -> pd.DataFrame:
    """
    Keep one representative prediction per sample_id:
    - prefer rows with is_top == True if present
    - otherwise best ranking_score among non-literal 'top'
    - otherwise best available ranking_score
    """
    if df_pred.empty:
        return df_pred.copy()

    d = df_pred.copy()
    if "sample_id" not in d.columns or "prediction_id" not in d.columns:
        return pd.DataFrame()

    d["prediction_id"] = d["prediction_id"].astype(str)

    if "ranking_score" in d.columns:
        d["ranking_score_num"] = pd.to_numeric(d["ranking_score"], errors="coerce")
    else:
        d["ranking_score_num"] = np.nan

    if "is_top" in d.columns:
        d["is_top_bool"] = d["is_top"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        d["is_top_bool"] = False

    out = []
    for sample_id, g in d.groupby("sample_id", dropna=False):
        g = g.copy()

        gt = g[g["is_top_bool"]]
        if not gt.empty:
            out.append(gt.iloc[0])
            continue

        cand = g[(g["prediction_id"] != "top") & g["ranking_score_num"].notna()].copy()
        if cand.empty:
            cand = g[g["ranking_score_num"].notna()].copy()
        if cand.empty:
            cand = g.copy()

        cand = cand.sort_values(
            ["ranking_score_num", "prediction_id"],
            ascending=[False, True]
        )
        out.append(cand.iloc[0])

    return pd.DataFrame(out).reset_index(drop=True)


def aggregate_chain_annotations(df_chain_annot: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate chain annotations per normalized job_name.
    """
    if df_chain_annot.empty:
        return pd.DataFrame(columns=[
            "job_name_norm",
            "chain_ids",
            "n_chains_annot",
            "chain_types",
            "chain_descriptions",
        ])

    d = df_chain_annot.copy()
    d["job_name_norm"] = d["job_name"].map(normalize_job_name)
    d["id"] = d["id"].astype(str)

    if "type" not in d.columns:
        d["type"] = ""
    if "sequence" not in d.columns:
        d["sequence"] = ""

    rows = []
    for job_name_norm, g in d.groupby("job_name_norm", dropna=False):
        g = g.copy().sort_values("id")

        chain_ids = ",".join(g["id"].astype(str).tolist())
        chain_types = ",".join(g["type"].astype(str).tolist())

        desc_parts = []
        for _, row in g.iterrows():
            cid = str(row["id"])
            ctype = str(row.get("type", ""))
            seq = str(row.get("sequence", ""))
            seqlen = len(seq) if seq else 0
            desc_parts.append(f"{cid}:{ctype}:len={seqlen}")

        rows.append({
            "job_name_norm": job_name_norm,
            "chain_ids": chain_ids,
            "n_chains_annot": int(len(g)),
            "chain_types": chain_types,
            "chain_descriptions": " | ".join(desc_parts),
        })

    return pd.DataFrame(rows)


def collect_sample_outputs(df_samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each sample_id, load report artifacts if present:
      reports/<sample_id>/predictions.tsv
      reports/<sample_id>/usalign_summary.tsv   (optional)

    Returns:
      df_pred_all, df_usalign_all
    """
    pred_parts = []
    usalign_parts = []

    for _, row in df_samples.iterrows():
        sample_id = str(row["sample_id"])
        report_dir = Path("reports") / sample_id

        pred_path = report_dir / "predictions.tsv"
        u_path_candidates = [
            report_dir / "usalign_summary.tsv",
            report_dir / "usalign_report" / "usalign_summary.tsv",
        ]

        df_pred = load_optional_tsv(pred_path)
        if not df_pred.empty:
            if "sample_id" not in df_pred.columns:
                df_pred["sample_id"] = sample_id
            pred_parts.append(df_pred)

        df_u = pd.DataFrame()
        for p in u_path_candidates:
            if p.exists():
                df_u = load_optional_tsv(p)
                break

        if not df_u.empty:
            if "sample_id" not in df_u.columns:
                df_u["sample_id"] = sample_id
            usalign_parts.append(df_u)

    df_pred_all = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    df_usalign_all = pd.concat(usalign_parts, ignore_index=True) if usalign_parts else pd.DataFrame()
    return df_pred_all, df_usalign_all


def build_master_table(
    df_samples: pd.DataFrame,
    df_chain_annot: pd.DataFrame,
    df_pred_all: pd.DataFrame,
    df_usalign_all: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one unified master table with one row per sample_id,
    using the top prediction for each sample.
    """
    d_samples = df_samples.copy()
    d_samples["sample_id"] = d_samples["sample_id"].astype(str)
    d_samples["job_name_norm"] = d_samples["sample_id"].map(derive_job_name_from_sample_id)

    chain_agg = aggregate_chain_annotations(df_chain_annot)

    d_top = top_prediction_rows(df_pred_all)
    if not d_top.empty:
        d_top = d_top.copy()
        d_top["sample_id"] = d_top["sample_id"].astype(str)

        pred_keep = [
            "sample_id",
            "prediction_id",
            "sample_pred_id",
            "ranking_score",
            "ptm",
            "iptm",
            "fraction_disordered",
            "has_clash",
            "mean_plddt_total",
            "std_plddt_total",
        ]
        pred_keep = [c for c in pred_keep if c in d_top.columns]
        d_top = d_top[pred_keep].copy()

        rename_pred = {
            "prediction_id": "top_prediction_id",
            "sample_pred_id": "top_sample_pred_id",
            "ranking_score": "top_ranking_score",
            "ptm": "top_ptm",
            "iptm": "top_iptm",
            "fraction_disordered": "top_fraction_disordered",
            "has_clash": "top_has_clash",
            "mean_plddt_total": "top_mean_plddt_total",
            "std_plddt_total": "top_std_plddt_total",
        }
        d_top = d_top.rename(columns=rename_pred)

    if not df_usalign_all.empty:
        d_u = df_usalign_all.copy()
        if "prediction_id" not in d_u.columns and "usalign_id" in d_u.columns:
            d_u["prediction_id"] = d_u["usalign_id"].astype(str)
        if "sample_id" in d_u.columns:
            d_u["sample_id"] = d_u["sample_id"].astype(str)

        # Prefer top-marked usalign row if present, else join on top_prediction_id later
        if "is_top" in d_u.columns:
            d_u["is_top_bool"] = d_u["is_top"].astype(str).str.lower().isin(["true", "1", "yes"])
        else:
            d_u["is_top_bool"] = False
    else:
        d_u = pd.DataFrame()

    master = d_samples.merge(chain_agg, on="job_name_norm", how="left")

    if not d_top.empty:
        master = master.merge(d_top, on="sample_id", how="left")

    if not d_u.empty and "top_prediction_id" in master.columns:
        u_keep = [
            "sample_id", "prediction_id",
            "TM1", "TM2", "RMSD", "ID1", "ID2", "IDali", "L1", "L2", "Lali"
        ]
        u_keep = [c for c in u_keep if c in d_u.columns]
        d_u2 = d_u[u_keep].drop_duplicates().copy()

        d_u2 = d_u2.rename(columns={
            "prediction_id": "top_prediction_id",
            "TM1": "top_TM1",
            "TM2": "top_TM2",
            "RMSD": "top_RMSD",
            "ID1": "top_ID1",
            "ID2": "top_ID2",
            "IDali": "top_IDali",
            "L1": "top_L1",
            "L2": "top_L2",
            "Lali": "top_Lali",
        })

        master = master.merge(
            d_u2,
            on=["sample_id", "top_prediction_id"],
            how="left"
        )

    # Add simple summary counts
    if not df_pred_all.empty and "sample_id" in df_pred_all.columns:
        pred_counts = (
            df_pred_all.copy()
            .assign(sample_id=lambda x: x["sample_id"].astype(str))
            .groupby("sample_id", dropna=False)
            .size()
            .reset_index(name="n_predictions")
        )
        master = master.merge(pred_counts, on="sample_id", how="left")

    # Reorder columns for readability
    preferred = [
        "sample_id",
        "job_name_norm",
        "af3_dir",
        "ground_truth",
        "n_predictions",
        "n_chains_annot",
        "chain_ids",
        "chain_types",
        "chain_descriptions",
        "top_prediction_id",
        "top_sample_pred_id",
        "top_ranking_score",
        "top_ptm",
        "top_iptm",
        "top_fraction_disordered",
        "top_has_clash",
        "top_mean_plddt_total",
        "top_std_plddt_total",
        "top_TM1",
        "top_TM2",
        "top_RMSD",
        "top_ID1",
        "top_ID2",
        "top_IDali",
        "top_L1",
        "top_L2",
        "top_Lali",
    ]
    cols = [c for c in preferred if c in master.columns] + [c for c in master.columns if c not in preferred]
    master = master[cols].copy()

    return master


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--report-samples",
    "report_samples_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="TSV with sample_id, af3_dir, and optional ground_truth."
)
@click.option(
    "--chain-samples",
    "chain_samples_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="TSV with chain annotations (job_name, id, sequence, type, ...)."
)
@click.option(
    "-o", "--outdir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Output directory."
)
def main(report_samples_path: Path, chain_samples_path: Path, outdir: Path):
    """
    Build a cohort-level unified master table across all AF3 sample directories.

    Expected per-sample inputs under reports/<sample_id>/:
      - predictions.tsv
      - optional usalign_summary.tsv
      - or optional usalign_report/usalign_summary.tsv

    Outputs:
      - cohort_master.tsv
    """
    outdir.mkdir(parents=True, exist_ok=True)

    df_samples = read_report_samples(report_samples_path)
    df_chain_annot = read_chain_annotations(chain_samples_path)

    df_pred_all, df_usalign_all = collect_sample_outputs(df_samples)
    master = build_master_table(df_samples, df_chain_annot, df_pred_all, df_usalign_all)

    out_tsv = outdir / "cohort_master.tsv"
    master.to_csv(out_tsv, sep="\t", index=False)

    click.echo(str(out_tsv))


if __name__ == "__main__":
    main()