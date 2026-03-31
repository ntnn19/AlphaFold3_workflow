#!/usr/bin/env python3
"""
workflow/scripts/aggregate_contact_results.py

Aggregate all per-pair local_tm.tsv files into a single master table.
"""
from __future__ import annotations
from pathlib import Path

import click
import pandas as pd


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--local-tm-dir", type=click.Path(exists=True, path_type=Path), required=True,
              help="Directory containing per-pair local_tm subdirectories.")
@click.option("--pair-ids", type=str, multiple=True, required=True,
              help="List of pair IDs to aggregate.")
@click.option("-o", "--output", "out_path", type=click.Path(path_type=Path), required=True,
              help="Output master TSV.")
def main(local_tm_dir: Path, pair_ids: tuple[str, ...], out_path: Path):
    """Aggregate local TM results into a master table."""
    parts = []
    for pair_id in pair_ids:
        p = local_tm_dir / pair_id / "local_tm.tsv"
        if not p.exists():
            continue
        df = pd.read_csv(p, sep="\t", dtype=str).fillna("")
        df.insert(0, "pair_id", pair_id)

        # Split pair_id into target_id and ref_id
        if "_vs_" in pair_id:
            target_id, ref_id = pair_id.split("_vs_", 1)
        else:
            target_id, ref_id = pair_id, ""
        df.insert(1, "target_id", target_id)
        df.insert(2, "ref_id", ref_id)

        parts.append(df)

    if parts:
        master = pd.concat(parts, ignore_index=True)
    else:
        master = pd.DataFrame()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(out_path, sep="\t", index=False)

    click.echo(f"✅ Master table with {len(master)} rows written to {out_path}")


if __name__ == "__main__":
    main()