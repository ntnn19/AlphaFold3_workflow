import copy
import itertools
import json
import os
import re
import string
from typing import Iterable, List, Optional, Set, Tuple

import click
import pandas as pd

UFM_SEQUENCE = "MSKVSFKITLTSDPRLPYKVLSVPESTPFTAVLKFAAEEFKVPAATSAIITNDGIGINPAQTAGNVFLKHGSELRIIPRDRVG"
UFM_CTERM_RESIDUE = 83  # residue bearing the OXT used for conjugation
LINKER_CCD_CODE = "TME"

SUPPORTED_PTMS = {"ufm"}  # 'ubq' is a recognized choice but not yet implemented


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


def collect_existing_ufm_chain_ids(sequences: List[dict]) -> List[str]:
    """Find chain ids of any entries already present in the input JSON whose
    sequence matches the UFM sequence, so they can be reused instead of
    adding a duplicate UFM chain."""
    ids: List[str] = []
    for entry in sequences:
        for key in entry:
            inner = entry[key]
            if not isinstance(inner, dict):
                continue
            if inner.get("sequence") != UFM_SEQUENCE:
                print("HERE")
                continue
            cid = inner.get("id")
            if isinstance(cid, list):
                ids.extend(str(c) for c in cid)
            elif cid is not None:
                ids.append(str(cid))
    return ids


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


def mutation_position(mutation_code: str) -> Optional[int]:
    match = re.match(r"(\d+)", mutation_code)
    return int(match.group(1)) if match else None


def strip_seed(name: str) -> str:
    """Remove _seed-<number> suffix from a sample name."""
    return re.sub(r"_seed-\d+$", "", name)


# --------------------------------------------------------------------------
# Mutation table loading
#
# Two accepted formats:
#   1. Legacy, headerless, exactly 4 tab-separated columns:
#        sample_id  type  id  mutation
#      Every row is treated as its own independent output variant, and no
#      PTM/ligation behaviour is available (fully backward compatible).
#
#   2. Header-based, any of these columns, in any order:
#        sample_id (required), type (required), id (required),
#        mutation (required), variant (optional), ptm (optional)
#      - 'variant': rows sharing the same (sample_id, variant) are combined
#        into a single output design, so multiple chains can be mutated
#        together.
#      - 'ptm': if set to 'ufm' for a row, every mutation listed in that
#        row's 'mutation' field becomes a UFM conjugation site (a UFM chain
#        + linker ligand + bondedAtomPairs are generated for it). Leave
#        blank for a plain mutation with no PTM.
# --------------------------------------------------------------------------
REQUIRED_COLUMNS = {"sample_id", "type", "id", "mutation"}
_ALLOWED = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_-.")

def load_mutation_table(path: str) -> pd.DataFrame:
    with open(path) as f:
        first_line = f.readline()
    header_tokens = [
        t.strip().strip('"').lower() for t in first_line.rstrip("\n").split("\t")
    ]
    has_header = "sample_id" in header_tokens

    if has_header:
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        df.columns = [c.strip().lower() for c in df.columns]

        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"Mutation table header is missing required column(s): {sorted(missing)}"
            )

        if "variant" not in df.columns:
            df["variant"] = [f"row{i}" for i in range(len(df))]
        if "ptm" not in df.columns:
            df["ptm"] = ""
    else:
        df = pd.read_csv(
            path, sep="\t", header=None, comment="#", dtype=str, keep_default_na=False
        )
        ncols = df.shape[1]
        if ncols != 4:
            raise ValueError(
                f"Headerless mutation table must have exactly 4 columns "
                f"(sample_id, type, id, mutation); found {ncols}. "
                "Add a header row if you want to use 'variant' and/or 'ptm' columns."
            )
        df.columns = ["sample_id", "type", "id", "mutation"]
        df["variant"] = [f"row{i}" for i in range(len(df))]
        df["ptm"] = ""

    # Track whether the caller gave real variant names (used to disambiguate
    # output filenames) vs. auto-generated 'rowN' placeholders.
    df.attrs["explicit_variant"] = has_header and "variant" in header_tokens
    df["variant"] = df["variant"].fillna("").astype(str)
    df["ptm"] = df["ptm"].fillna("").astype(str).str.strip().str.lower()
    return df


# --------------------------------------------------------------------------
# UFM ligation
# --------------------------------------------------------------------------
def add_ufm_site(
    mutated_data: dict,
    chain_ids: Set[str],
    chain_type: str,
    target_chain_id: str,
    position: int,
    reused_ufm_chain_id: Optional[str] = None,
) -> str:
    """Add a linker ligand + the two bonded atom pairs connecting
    target_chain_id/position -> linker -> UFM chain. If reused_ufm_chain_id
    is given, that pre-existing UFM chain is bonded to instead of adding a
    new (duplicate) UFM protein chain. Returns the UFM chain id used."""
    if reused_ufm_chain_id is not None:
        ufm_chain_id = reused_ufm_chain_id
    else:
        ufm_chain_id = next_chain_id(chain_ids)
        chain_ids.add(ufm_chain_id)
        mutated_data["sequences"].append(
            {chain_type: {"id": ufm_chain_id, "sequence": UFM_SEQUENCE}}
        )

    ligand_chain_id = next_chain_id(chain_ids)
    chain_ids.add(ligand_chain_id)
    mutated_data["sequences"].append(
        {"ligand": {"id": ligand_chain_id, "ccdCodes": [LINKER_CCD_CODE]}}
    )

    bonded = mutated_data.setdefault("bondedAtomPairs", [])
    bonded.append([[target_chain_id, position, "CB"], [ligand_chain_id, 1, "C1"]])
    bonded.append(
        [[ufm_chain_id, UFM_CTERM_RESIDUE, "OXT"], [ligand_chain_id, 1, "C3"]]
    )

    return ufm_chain_id


def sanitise(name: str) -> str:
    """Lowercase, replace spaces, strip non-alphanumeric chars."""
    return "".join(c for c in name.lower().replace(" ", "_") if c in _ALLOWED)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
@click.command()
@click.argument("input_json", type=click.Path(exists=True))
@click.argument("mutation_list", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
def mutate(input_json, mutation_list, output_dir):
    """
    Mutate sequences in an AlphaFold3 JSON based on a mutations TSV table.

    See load_mutation_table() for the two accepted table formats. In short:
    add a header row with an optional 'ptm' column and set it to 'ufm' on
    any row to have that row's mutation(s) generate a UFM conjugation site.
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
            row_ptm = row["ptm"]
            if row_ptm and row_ptm not in SUPPORTED_PTMS:
                click.echo(
                    f"Warning: PTM type '{row_ptm}' requested for "
                    f"type={chain_type}, id={chain_id} is not yet implemented; "
                    "mutation(s) will be applied without PTM ligation."
                )

            applied: List[str] = []

            for mutation_code in raw_muts:
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

            if applied:
                chain_mutations.append((chain_id, applied))

                if row_ptm == "ufm":
                    for mutation_code in applied:
                        position = mutation_position(mutation_code)
                        if position is None:
                            click.echo(
                                f"Warning: Could not parse position from "
                                f"'{mutation_code}'; skipping UFM site."
                            )
                        else:
                            ufm_sites.append((chain_type, chain_id, position))

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

        # If the table explicitly names variants, include the variant name in
        # the filename. This guarantees uniqueness even when two variants
        # apply the same mutations but differ only in PTM status (e.g. one
        # row set has ptm=ufm and another doesn't), which the mutation
        # suffix alone would not distinguish.
        if mut_df.attrs.get("explicit_variant"):
            new_name = f"{full_name}_{variant_id}_{suffix}"
        else:
            new_name = f"{full_name}_{suffix}"
        mutated_data["name"] = sanitise(new_name)

        if ufm_sites:
            chain_ids = collect_chain_ids(sequences)
            available_ufm_ids = collect_existing_ufm_chain_ids(sequences)
            for chain_type, target_chain_id, position in ufm_sites:
                reused_id = available_ufm_ids.pop(0) if available_ufm_ids else None
                used_id = add_ufm_site(
                    mutated_data,
                    chain_ids,
                    chain_type,
                    target_chain_id,
                    position,
                    reused_id,
                )
                if reused_id:
                    click.echo(
                        f"  Ligated {target_chain_id}{position} to existing "
                        f"UFM chain '{used_id}' (no duplicate UFM chain added)"
                    )
                else:
                    click.echo(
                        f"  Added UFM ligation at {target_chain_id}{position} "
                        f"(chain type '{chain_type}', new UFM chain '{used_id}')"
                    )

        out_path = os.path.join(output_dir, f"{sanitise(new_name)}.json")
        with open(out_path, "w") as f:
            json.dump(mutated_data, f, indent=2)

        click.echo(f"Written: {out_path}")


if __name__ == "__main__":
    mutate()
