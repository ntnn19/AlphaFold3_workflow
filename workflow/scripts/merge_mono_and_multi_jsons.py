import pandas as pd
import json
import os
from pathlib import Path
import copy
import click
from loguru import logger

@click.command()
@click.argument("multimer_file", type=click.Path(exists=True))
@click.argument("monomer_file", type=click.Path(exists=True), nargs=-1)
@click.argument("output_file", type=click.Path())  # where merged JSON will be saved
@click.option("--inference-to-data-map", type=click.Path())
def main(multimer_file, monomer_file, output_file, inference_to_data_map):
    """
    Merge monomer JSONs into a multimer JSON for a given sample.

    The {multi} wildcard already encodes exactly one seed (e.g. job1_seed-10),
    so the input multimer JSON contains modelSeeds=[N] for a single seed.
    We write the merged result once to output_file.
    """

    if inference_to_data_map:
        # A2 fix: use the correctly-named variable throughout
        inference_to_data_map_df = pd.read_csv(inference_to_data_map, sep="\t")
        job_map_df = inference_to_data_map_df[
            inference_to_data_map_df.multimer_file.str.contains(Path(multimer_file).stem)
        ]
        input_multimer_file = job_map_df.multimer_file.unique()[0]
        grouped = job_map_df.groupby("multimer_file")
        for input_multimer_file_, group in grouped:
            with open(input_multimer_file, "r") as f:
                multimer_data = json.load(f)

            merged_multimer = copy.deepcopy(multimer_data)

            # Build a reference map: chain_id → inner dict (mutable reference)
            target_map = {}
            for item in merged_multimer["sequences"]:
                inner_dict = list(item.values())[0]
                target_map[inner_dict["id"]] = inner_dict

            # Inject per-chain MSA data from each monomer file
            for _, row in group.iterrows():
                monomer_input_file = row["monomer_file"]
                target_chain_id = row["monomer_chain_id"]

                with open(monomer_input_file, "r") as f:
                    monomer_data = json.load(f)
                    source_content = list(monomer_data["sequences"][0].values())[0]

                    if target_chain_id in target_map:
                        original_id = target_map[target_chain_id]["id"]
                        target_map[target_chain_id].update(source_content)
                        # Restore the original chain ID (source may have a different id)
                        target_map[target_chain_id]["id"] = original_id

    else:
        with open(multimer_file, "r") as f:
            multimer_data = json.load(f)

        merged_multimer = copy.deepcopy(multimer_data)
        logger.debug("merged_multimer: %s", merged_multimer)
        for entry, mf in zip(merged_multimer["sequences"], monomer_file):
            mol_type = next(iter(entry))
            monomer_entry = json.load(open(mf))["sequences"][0][mol_type]
            monomer_entry.pop("id")
            entry[mol_type].update(monomer_entry)

    # A3 fix: the {multi} wildcard encodes exactly one seed (e.g. job1_seed-10),
    # so merged_multimer["modelSeeds"] is already a single-element list.
    # Write the merged JSON once — no seed loop needed.
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(merged_multimer, f, indent=4)
    logger.info("Merged JSON written to %s", output_file)

if __name__ == "__main__":
    main()
