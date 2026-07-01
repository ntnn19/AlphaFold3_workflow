import copy
import itertools
import json
import os
import re
import string
from typing import Dict, Iterable, List, Optional, Set, Tuple

import click
import pandas as pd

UFM_SEQUENCE = "MSKVSFKITLTSDPRLPYKVLSVPESTPFTAVLKFAAEEFKVPAATSAIITNDGIGINPAQTAGNVFLKHGSELRIIPRDRVG"
UFM_CTERM_RESIDUE = 83  # residue bearing the OXT used for conjugation
LINKER_CCD_CODE = "TME"


# --------------------------------------------------------------------------
# Chain id helpers
# --------------------------------------------------------------------------
def next_chain_id(existing: Iterable[str]) -> str:
    letters = list(string.ascii_uppercase)
    letters += ["".join(p) for p in itertools.product(letters, repeat=2)]

    used: Set[str] = set(existing)

    for cid in letters:
        if cid not in used:
            return cid

    raise ValueError("No available chain ids left")


def collect_chain_ids(sequences: List[dict]) -> Set[str]:
    """Gather every chain id currently used in a `sequences` list (list-valued
    ids, e.g. homo-oligomers, are expanded)."""
    ids: Set[str] = set()
    for entry in sequences:
        for key in entry:
            inner = entry[key]
            if not isinstance(inner, dict):
                continue
            cid = inner.get("id")
            if isinstance(cid, list):
                ids.update(str(c) for c in cid)
            elif cid is not None:
                ids.add(str(cid))
    return ids


# --------------------------------------------------------------------------
# Mutation application
# --------------------------------------------------------------------------
def apply_mutation(sequence: str, mutation: str) -> str:
    """Apply a single mutation to a sequence. Format: <POSITION><NEW_AA>, e.g. 14K."""
    mutation = mutation.strip()
    pos_str = mutation[:-1]
    new_aa = mutation[-1]
    position = int(pos_str)
    seq_list = list(sequence)
    seq_list[position - 1] = new_aa  # 1-based
    return "".join(seq_list)


def apply_mutation_to_a3m(a3m_string: str, mutation: str) -> str:
    """
    Apply a mutation to the first (query) sequence in an a3m-formatted string.
    The first sequence is the query; all other lines are alignment rows.
    """
    if not a3m_string or not a3m_string.strip():
        return a3m_string

    lines = a3m_string.split("\n")

    first_seq_start = None
    first_seq_end = None
    in_first_seq = False

    for i, line in enumerate(lines):
        if line.startswith(">"):
            if in_first_seq:
                first_seq_end = i
                break
            else:
                in_first_seq = True
        elif in_first_seq and first_seq_start is None and line.strip():
            first_seq_start = i

    if first_seq_start is None:
        return a3m_string

    if first_seq_end is None:
        first_seq_end = len(lines)

    query_lines = lines[first_seq_start:first_seq_end]
    query_seq = "".join(query_lines)

    mutated_query = apply_mutation(query_seq, mutation)

    mutated_lines = []
    offset = 0
    for ql in query_lines:
        mutated_lines.append(mutated_query[offset : offset + len(ql)])
        offset += len(ql)

    new_lines = lines[:first_seq_start] + mutated_lines + lines[first_seq_end:]
    return "\n".join(new_lines)


def parse_mutation_entry(entry: str) -> Tuple[str, bool]:
    """
    Parse a single mutation token. A trailing '*' marks the position as a
    UFM/PTM conjugation site, e.g. '14K*' -> ('14K', True); '20R' -> ('20R', False).
    """
    entry = entry.strip()
    is_ufm_site = entry.endswith("*")
    if is_ufm_site:
        entry = entry[:-1].strip()
    return entry, is_ufm_site


def mutation_position(mutation_code: str) -> Optional[int]:
    match = re.match(r"(\d+)", mutation_code)
    return int(match.group(1)) if match else None


def strip_seed(name: str) -> str:
    """Remove _seed-<number> suffix from a sample name."""
    return re.sub(r"_seed-\d+$", "", name)


# --------------------------------------------------------------------------
# Mutation table loading (backward compatible with the original 4-column
# format; supports an optional 'variant' column to group multiple chain
# rows into a single output design)
# --------------------------------------------------------------------------
def load_mutation_table(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path, sep="\t", header=None, comment="#", dtype=str)
    ncols = raw.shape[1]

    if ncols == 5:
        raw.columns = ["sample_id", "variant", "type", "id", "mutation"]
    elif ncols == 4:
        raw.columns = ["sample_id", "type", "id", "mutation"]
        # Original behaviour: every row is its own independent variant.
        raw["variant"] = [f"row{i}" for i in range(len(raw))]
    else:
        raise ValueError(
            f"Unexpected number of columns ({ncols}) in mutation table; "
            "expected 4 (sample_id, type, id, mutation) or "
            "5 (sample_id, variant, type, id, mutation)."
        )

    return raw


# --------------------------------------------------------------------------
# UFM ligation
# --------------------------------------------------------------------------
def add_ufm_site(
    mutated_data: dict,
    chain_ids: Set[str],
    chain_type: str,
    target_chain_id: str,
    position: int,
) -> None:
    """Add a UFM protein chain + linker ligand + the two bonded atom pairs
    connecting target_chain_id/position -> linker -> UFM chain."""
    ufm_chain_id = next_chain_id(chain_ids)
    chain_ids.add(ufm_chain_id)
    ligand_chain_id = next_chain_id(chain_ids)
    chain_ids.add(ligand_chain_id)

    mutated_data["sequences"].append(
        {chain_type: {"id": ufm_chain_id, "sequence": UFM_SEQUENCE}}
    )
    mutated_data["sequences"].append(
        {"ligand": {"id": ligand_chain_id, "ccdCodes": [LINKER_CCD_CODE]}}
    )

    bonded = mutated_data.setdefault("bondedAtomPairs", [])
    bonded.append([[target_chain_id, position, "CB"], [ligand_chain_id, 1, "C1"]])
    bonded.append(
        [[ufm_chain_id, UFM_CTERM_RESIDUE, "OXT"], [ligand_chain_id, 1, "C3"]]
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
@click.command()
@click.argument("input_json", type=click.Path(exists=True))
@click.argument("mutation_list", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "--ptm",
    type=click.Choice(["ufm", "ubq"]),
    help=(
        "If 'ufm', any mutation marked with a trailing '*' (e.g. '14K*') is "
        "treated as a UFM conjugation site: a UFM chain and a linker ligand "
        "are added and bonded to that position. Without --ptm, '*' markers "
        "are ignored and only stripped."
    ),
)
def mutate(input_json, mutation_list, output_dir, ptm):
    """
    Mutate sequences in an AlphaFold3 JSON based on a mutations TSV table.

    Mutation table columns (tab separated):
        sample_id  [variant]  type  id  mutation

    - 'variant' is optional. If present, all rows sharing the same
      (sample_id, variant) are combined into a single output design, which
      lets you mutate several chains (several targets) at once. If absent,
      each row is its own design (original behaviour).
    - 'mutation' is a comma separated list of mutations, e.g. '14K,20R'.
      Append '*' to a mutation to mark it as a UFM ligation site when
      --ptm ufm is given, e.g. '14K*,20R'.
    """
    with open(input_json) as f:
        data = json.load(f)

    full_name = data.get("name", "")
    base_name = strip_seed(full_name)

    click.echo(f"Sample name in JSON  : {full_name}")
    click.echo(f"Base name (seed stripped): {base_name}")

    mut_df = load_mutation_table(mutation_list)
    sample_mutations = mut_df[mut_df["sample_id"] == base_name]

    if sample_mutations.empty:
        click.echo(
            f"No mutations found for base name '{base_name}' "
            f"(full name: '{full_name}'). Exiting."
        )
        return

    os.makedirs(output_dir, exist_ok=True)

    for variant_id, group in sample_mutations.groupby("variant", sort=False):
        mutated_data = copy.deepcopy(data)
        sequences = mutated_data.get("sequences", [])

        chain_mutations: List[Tuple[str, List[str]]] = []  # (chain_id, [applied muts])
        ufm_sites: List[Tuple[str, str, int]] = []  # (chain_type, chain_id, position)

        for _, row in group.iterrows():
            raw_muts = [m.strip() for m in str(row["mutation"]).split(",") if m.strip()]
            if not raw_muts:
                continue

            chain_type = row["type"]
            chain_id = str(row["id"])
            applied: List[str] = []

            for raw_mut in raw_muts:
                mutation_code, is_ufm_site = parse_mutation_entry(raw_mut)

                matched = False
                for seq_entry in sequences:
                    if chain_type not in seq_entry:
                        continue

                    inner = seq_entry[chain_type]
                    ids = inner.get("id")
                    if isinstance(ids, list):
                        if chain_id not in ids:
                            continue
                    elif isinstance(ids, str):
                        if ids != chain_id:
                            continue
                    else:
                        continue

                    if "sequence" not in inner:
                        click.echo(
                            f"Warning: No 'sequence' field for "
                            f"type={chain_type}, id={chain_id}"
                        )
                        continue

                    inner["sequence"] = apply_mutation(inner["sequence"], mutation_code)

                    if "pairedMsa" in inner and inner["pairedMsa"]:
                        inner["pairedMsa"] = apply_mutation_to_a3m(
                            inner["pairedMsa"], mutation_code
                        )
                        click.echo(f"  Mutated pairedMsa for {chain_type} {chain_id}")

                    if "unpairedMsa" in inner and inner["unpairedMsa"]:
                        inner["unpairedMsa"] = apply_mutation_to_a3m(
                            inner["unpairedMsa"], mutation_code
                        )
                        click.echo(f"  Mutated unpairedMsa for {chain_type} {chain_id}")

                    applied.append(mutation_code)
                    matched = True
                    break

                if not matched:
                    click.echo(
                        f"Warning: No matching entry for "
                        f"type={chain_type}, id={chain_id}, mutation={mutation_code}"
                    )
                    continue

                if is_ufm_site:
                    if ptm != "ufm":
                        click.echo(
                            f"Note: '{raw_mut}' marked as a UFM site but "
                            "--ptm ufm was not given; marker ignored."
                        )
                    else:
                        position = mutation_position(mutation_code)
                        if position is None:
                            click.echo(
                                f"Warning: Could not parse position from "
                                f"'{mutation_code}'; skipping UFM site."
                            )
                        else:
                            ufm_sites.append((chain_type, chain_id, position))

            if applied:
                chain_mutations.append((chain_id, applied))

        if not chain_mutations:
            click.echo(f"Warning: No mutations applied for variant '{variant_id}'.")
            continue

        # Build a variant name. Single-chain variants keep the original,
        # simpler naming for backward compatibility.
        if len(chain_mutations) == 1:
            suffix = "_".join(chain_mutations[0][1])
        else:
            suffix = "_".join(
                f"{cid}-" + "_".join(muts) for cid, muts in chain_mutations
            )
        new_name = f"{full_name}_{suffix}"
        mutated_data["name"] = new_name

        if ptm == "ufm" and ufm_sites:
            chain_ids = collect_chain_ids(sequences)
            for chain_type, target_chain_id, position in ufm_sites:
                add_ufm_site(
                    mutated_data, chain_ids, chain_type, target_chain_id, position
                )
                click.echo(
                    f"  Added UFM ligation at {target_chain_id}{position} "
                    f"(chain type '{chain_type}')"
                )

        out_path = os.path.join(output_dir, f"{new_name}.json")
        with open(out_path, "w") as f:
            json.dump(mutated_data, f, indent=2)

        click.echo(f"Written: {out_path}")


if __name__ == "__main__":
    mutate()
