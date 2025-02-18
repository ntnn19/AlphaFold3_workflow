import json
import string
import click
import itertools
from copy import deepcopy

def transform_json(data):
    transformed_data = deepcopy(data)

    for entry in transformed_data:
        unique_ids = itertools.cycle(string.ascii_uppercase)  # Reset for each name entry
        new_sequences = []
        ligand_sequences = []  # Collect all ligand sequences

        for sequence in entry["sequences"]:
            if "proteinChain" in sequence:
                sequence["proteinChain"]["ids"] = [next(unique_ids) for _ in range(sequence["proteinChain"]["count"])]
                del sequence["proteinChain"]["count"]
                new_sequences.append(sequence)
            elif "dnaSequence" in sequence:
                ligand_sequences.append(sequence["dnaSequence"]["sequence"])

        if ligand_sequences:
            ligand_entry = {
                "ligand": {
                    "smiles": "COc1c(N2[C@H](c3c(N=C2N2CCN(CC2)c2cc(OC)ccc2)c(ccc3)F)CC(=O)O)cc(cc1)C(F)(F)F",
                    "ids": [next(unique_ids) for _ in range(5)]
                }
            }
            new_sequences.append(ligand_entry)

        entry["sequences"] = new_sequences

    return transformed_data

@click.command()
@click.argument("input_file", type=click.File("r"))
@click.argument("output_file", type=click.File("w"))
def process_json(input_file, output_file):
    """Reads JSON from INPUT_FILE, transforms it, and writes to OUTPUT_FILE."""
    try:
        input_data = json.load(input_file)
        output_data = transform_json(input_data)
        json.dump(output_data, output_file, indent=2)
        click.echo(f"Processed JSON saved to {output_file.name}")
    except json.JSONDecodeError:
        click.echo("Error: Invalid JSON file.", err=True)

if __name__ == "__main__":
    process_json()
