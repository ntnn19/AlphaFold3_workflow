import json
import os
from pathlib import Path
import copy
import click

@click.command()
@click.argument("multimer_file", type=click.Path(exists=True))
@click.argument("monomer_file", type=click.Path(exists=True), nargs=-1)
@click.argument("sample_id", type=str)
@click.argument("output_file", type=click.Path())  # where merged JSON will be saved
def main(multimer_file, monomer_file, sample_id, output_file):
    """
    Merge monomer JSONs into a multimer JSON for a given sample.
    """

    # 1️⃣ Load multimer JSON
    with open(multimer_file, "r") as f:
        multimer_data = json.load(f)

    merged_multimer = copy.deepcopy(multimer_data)

    # 2️⃣ Create a map of target chains in the multimer
    # Key = chain ID ('A', 'B', etc.) -> value = dictionary
    target_map = {list(item.keys())[0]: list(item.values())[0] for item in merged_multimer["sequences"]}

    # 3️⃣ Inject each monomer into the corresponding chain
    for mono_file in monomer_file:
        with open(mono_file, "r") as f:
            monomer_data = json.load(f)
            # Take first sequence from monomer file
            source_content = list(monomer_data["sequences"][0].values())[0]
            target_chain_id = source_content["id"]  # assume ID matches target chain

            if target_chain_id in target_map:
                # Preserve original ID, update content
                original_id = target_map[target_chain_id]["id"]
                target_map[target_chain_id].update(source_content)
                target_map[target_chain_id]["id"] = original_id

    # 4️⃣ For each model seed, write a separate merged JSON
    for s in merged_multimer["modelSeeds"]:
        seed_merged_multimer = copy.deepcopy(merged_multimer)
        seed_merged_multimer["modelSeeds"] = [s]

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(seed_merged_multimer, f, indent=4)

    print(f"Merged JSON written to {output_file}")


if __name__ == "__main__":
    main()
