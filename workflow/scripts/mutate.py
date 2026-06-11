import click
import json
import os
import re
import pandas as pd
import copy


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
    Insertions (lowercase letters) in the query are ignored for position counting
    since a3m query sequences are typically uppercase only, but we handle it safely.
    """
    if not a3m_string or not a3m_string.strip():
        return a3m_string

    lines = a3m_string.split("\n")

    # Find the first sequence block: skip header lines (starting with '>') 
    # and collect the query sequence lines until the next header or end
    first_seq_start = None
    first_seq_end = None
    in_first_seq = False

    for i, line in enumerate(lines):
        if line.startswith(">"):
            if in_first_seq:
                # We've reached the second header — first sequence block ends here
                first_seq_end = i
                break
            else:
                in_first_seq = True  # first header found, sequence follows
        elif in_first_seq and first_seq_start is None and line.strip():
            first_seq_start = i  # first non-empty line after first header

    if first_seq_start is None:
        return a3m_string  # no sequence found, return unchanged

    if first_seq_end is None:
        first_seq_end = len(lines)

    # Collect and join the first sequence (may span multiple lines)
    query_lines = lines[first_seq_start:first_seq_end]
    query_seq = "".join(query_lines)

    # Apply mutation (position is 1-based over the full sequence string)
    mutated_query = apply_mutation(query_seq, mutation)

    # Split back to original line lengths
    mutated_lines = []
    offset = 0
    for ql in query_lines:
        mutated_lines.append(mutated_query[offset:offset + len(ql)])
        offset += len(ql)

    # Reconstruct full a3m string
    new_lines = lines[:first_seq_start] + mutated_lines + lines[first_seq_end:]
    return "\n".join(new_lines)


def mutation_suffix(mutations: list) -> str:
    return "_".join(m.strip() for m in mutations)


def strip_seed(name: str) -> str:
    """Remove _seed-<number> suffix from a sample name."""
    return re.sub(r"_seed-\d+$", "", name)


@click.command()
@click.argument("input_json", type=click.Path(exists=True))
@click.argument("mutation_list", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
def mutate(input_json, mutation_list, output_dir):
    """Mutate sequences in an AlphaFold3 JSON based on a mutations TSV table."""
    with open(input_json) as f:
        data = json.load(f)

    full_name = data.get("name", "")
    base_name = strip_seed(full_name)

    click.echo(f"Sample name in JSON  : {full_name}")
    click.echo(f"Base name (seed stripped): {base_name}")

    mut_df = pd.read_csv(
        mutation_list,
        sep="\t",
        names=["sample_id", "type", "id", "mutation"],
        comment="#",
    )

    sample_mutations = mut_df[mut_df["sample_id"] == base_name]

    if sample_mutations.empty:
        click.echo(
            f"No mutations found for base name '{base_name}' "
            f"(full name: '{full_name}'). Exiting."
        )
        return

    os.makedirs(output_dir, exist_ok=True)
#    wt_path = os.path.join(output_dir, f"{full_name}.json")
#    with open(wt_path, "w") as f:
#        json.dump(data, f, indent=2)
#    click.echo(f"Written WT: {wt_path}")
    for _, row in sample_mutations.iterrows():
        muts = [m.strip() for m in str(row["mutation"]).split(",") if m.strip()]
        if not muts:
            continue

        mutated_data = copy.deepcopy(data)
        sequences = mutated_data.get("sequences", [])
        applied = []

        for mut in muts:
            chain_type = row["type"]
            chain_id   = str(row["id"])

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

                # Mutate the main sequence
                inner["sequence"] = apply_mutation(inner["sequence"], mut)

                # Mutate the query row in pairedMsa if present
                if "pairedMsa" in inner and inner["pairedMsa"]:
                    inner["pairedMsa"] = apply_mutation_to_a3m(
                        inner["pairedMsa"], mut
                    )
                    click.echo(f"  Mutated pairedMsa for {chain_type} {chain_id}")

                # Mutate the query row in unpairedMsa if present
                if "unpairedMsa" in inner and inner["unpairedMsa"]:
                    inner["unpairedMsa"] = apply_mutation_to_a3m(
                        inner["unpairedMsa"], mut
                    )
                    click.echo(f"  Mutated unpairedMsa for {chain_type} {chain_id}")

                applied.append(mut)
                matched = True
                break

            if not matched:
                click.echo(
                    f"Warning: No matching entry for "
                    f"type={chain_type}, id={chain_id}, mutation={mut}"
                )

        if not applied:
            click.echo(f"Warning: No mutations applied for row: {row.to_dict()}")
            continue

        suffix = mutation_suffix(applied)
        new_name = f"{full_name}_{suffix}"
        mutated_data["name"] = new_name

        out_path = os.path.join(output_dir, f"{new_name}.json")
        with open(out_path, "w") as f:
            json.dump(mutated_data, f, indent=2)

        click.echo(f"Written: {out_path}")


if __name__ == "__main__":
    mutate()