"""Concatenate per-job TSV files into a sorted project-level summary table.

Usage:
    python meta_aggregate.py <filelist> <output_tsv>

<filelist> is a plain-text file with one input TSV path per line.

Sort keys are inferred from the columns present in the concatenated DataFrame:
  - global TSV:          name, seed, sample
  - per_chain TSV:       name, seed, sample, chain_id
  - per_chain_pair TSV:  name, seed, sample, chain_i, chain_j

Memory: the full concatenated DataFrame is held in memory. At ~400k rows and
~10 columns of mixed int/float/str, peak usage is well under 500 MB.
"""

import sys
from pathlib import Path

import pandas as pd


def _sort_keys(columns: list[str]) -> list[str]:
    """Return the appropriate sort key list based on which columns are present."""
    cols = set(columns)
    if "chain_i" in cols and "chain_j" in cols:
        return ["name", "seed", "sample", "chain_i", "chain_j"]
    if "chain_id" in cols:
        return ["name", "seed", "sample", "chain_id"]
    return ["name", "seed", "sample"]


def main(filelist: str, output_tsv: str) -> None:
    paths = [
        line.strip()
        for line in Path(filelist).read_text().splitlines()
        if line.strip()
    ]
    if not paths:
        raise ValueError(f"No input files listed in {filelist}")

    frames = [pd.read_csv(p, sep="\t") for p in paths]
    df = pd.concat(frames, ignore_index=True)

    sort_by = [k for k in _sort_keys(df.columns.tolist()) if k in df.columns]
    if sort_by:
        df = df.sort_values(sort_by).reset_index(drop=True)

    df.to_csv(output_tsv, sep="\t", index=False)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(*sys.argv[1:])