"""Concatenate per-job TSV files into project-level summary tables.

Usage:
    python meta_aggregate.py <filelist> <output_tsv>

<filelist> is a plain-text file with one input TSV path per line.
Reads files one at a time — no shell argument limits, no full dataset
in memory at once.
"""

import sys
from pathlib import Path


def main(filelist: str, output_tsv: str):
    paths = [
        line.strip()
        for line in Path(filelist).read_text().splitlines()
        if line.strip()
    ]
    if not paths:
        raise ValueError(f"No input files listed in {filelist}")

    with open(output_tsv, "w") as out:
        header_written = False
        for path in paths:
            with open(path) as f:
                lines = f.readlines()
            if not lines:
                continue
            if not header_written:
                out.writelines(lines)
                header_written = True
            else:
                out.writelines(lines[1:])  # skip header on subsequent files


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(*sys.argv[1:])
