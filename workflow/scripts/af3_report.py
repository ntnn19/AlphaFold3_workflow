#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

import click
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple, Any
import seaborn as sns
import plotly.graph_objects as go

SEED_SAMPLE_RE = re.compile(r"seed-(\d+)_sample-(\d+)")

# ✅ NEW HELPER
def get_sample_id_from_output_dir(output_dir: Path) -> str:
    return output_dir.name


def load_json(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


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
    parent = summary_path.parent

    if layout == "nagarnat":
        if summary_path.name == "summary_confidences.json":
            p = parent / "confidences.json"
            return p if p.exists() else None

        if summary_path.name.endswith("_summary_confidences.json"):
            p = parent / summary_path.name.replace("_summary_confidences.json", "_confidences.json")
            return p if p.exists() else None

        return None

    if layout == "dm":
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
        a = list(output_dir.rglob("*_summary_confidences.json"))
        b = list(output_dir.rglob("summary_confidences.json"))
        return sorted(set(a + b))
    if layout == "dm":
        return sorted(output_dir.rglob("*_summary_confidences.json"))
    raise ValueError(f"Unknown layout: {layout}")


# ✅ FULLY UPDATED FUNCTION
def summarize_job(output_dir: Path, layout: str):
    sample_id = get_sample_id_from_output_dir(output_dir)

    summary_files = find_summary_files(output_dir, layout)

    pred_rows = []
    chain_rows = []
    pair_rows = []

    for sp in summary_files:
        seed, sample, pred_id = parse_prediction_id(sp)

        # ✅ prefix prediction_id
        pred_id_with_sample = f"{sample_id}__{pred_id}"

        summ = load_json(sp)

        cp = resolve_confidences_path(sp, layout)
        conf = load_json(cp) if cp and cp.exists() else {}

        # ---- complex-wide ----
        pred_rows.append({
            "sample_id": sample_id,
            "prediction_id": pred_id_with_sample,
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

        chain_ids = sorted({str(x) for x in conf.get("atom_chain_ids", [])})
        n = max(len(chain_ptm), len(chain_iptm), len(chain_ids))

        if not chain_ids and n:
            chain_ids = [str(i) for i in range(n)]

        chain_mean_plddt = mean_plddt_by_chain(conf)

        for i, cid in enumerate(chain_ids):
            chain_rows.append({
                "sample_id": sample_id,
                "prediction_id": pred_id_with_sample,
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

            if len(chain_ids) != mat_iptm.shape[0]:
                chain_ids_pair = [str(i) for i in range(mat_iptm.shape[0])]
            else:
                chain_ids_pair = chain_ids

            for i, ci in enumerate(chain_ids_pair):
                for j, cj in enumerate(chain_ids_pair):
                    pair_rows.append({
                        "sample_id": sample_id,
                        "prediction_id": pred_id_with_sample,
                        "seed": seed,
                        "sample": sample,
                        "chain_i": ci,
                        "chain_j": cj,
                        "pair_iptm": float(mat_iptm[i, j]),
                        "pair_pae_min": float(mat_pae[i, j]) if mat_pae is not None else None,
                        "is_diagonal": (i == j)
                    })

    return (
        pd.DataFrame(pred_rows),
        pd.DataFrame(chain_rows),
        pd.DataFrame(pair_rows),
    )


@click.command()
@click.argument("af3_output_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("-o", "--outdir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--layout", type=click.Choice(["nagarnat", "dm"]), default="nagarnat")
def main(af3_output_dir: Path, outdir: Path, layout: str):

    outdir.mkdir(parents=True, exist_ok=True)

    df_pred, df_chain, df_pair = summarize_job(af3_output_dir, layout)

    df_pred.to_csv(outdir / "predictions.csv", index=False)
    df_chain.to_csv(outdir / "chains.csv", index=False)
    df_pair.to_csv(outdir / "chain_pairs.csv", index=False)


if __name__ == "__main__":
    main()