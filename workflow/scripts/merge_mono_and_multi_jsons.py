import copy
import json
import os
from pathlib import Path

import click
import pandas as pd
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
            inference_to_data_map_df.multimer_file.str.contains(
                Path(multimer_file).stem
            )
            inference_to_data_map_df.multimer_file.str.contains(Path(multimer_file).stem)
        ]
        input_multimer_file = job_map_df.multimer_file.unique()[0]
        grouped = job_map_df.groupby("multimer_file")
        for input_multimer_file_, group in grouped:
            with open(input_multimer_file, "r") as f:
                multimer_data = json.load(f)
            merged_multimer = copy.deepcopy(multimer_data)
            # Build a reference map: chain_id -> inner dict (mutable reference)
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
        logger.debug("merged_multimer: {}", merged_multimer)
        POLYMER_TYPES = {"protein", "rna", "dna"}

        # Build chain_id -> (entry, mol_type) for every polymer chain in the
        # multimer. Matching is done by chain id, never by argument order, so
        # monomer_file can be passed in any order without risk of attaching
        # the wrong monomer's data to a chain.
        target_map = {}
        for entry in merged_multimer["sequences"]:
            mol_type = next(iter(entry))
            if mol_type not in POLYMER_TYPES:
                continue
            chain_id = entry[mol_type].get("id")
            target_map[chain_id] = (entry, mol_type)

        if len(monomer_file) != len(target_map):
            raise click.UsageError(
                f"Got {len(monomer_file)} monomer file(s) but the multimer "
                f"JSON has {len(target_map)} polymer chain(s) "
                f"({sorted(target_map)}). Refusing to merge positionally, "
                "since that can silently attach the wrong monomer to a chain."
            )

        used_chain_ids = set()
        for mf in monomer_file:
            with open(mf, "r") as f:
                monomer_data = json.load(f)
            mol_type = next(iter(monomer_data["sequences"][0]))
            monomer_entry = monomer_data["sequences"][0][mol_type]
            source_chain_id = monomer_entry.get("id")

            if source_chain_id is None or source_chain_id not in target_map:
                raise click.UsageError(
                    f"Monomer file '{mf}' has chain id '{source_chain_id}', "
                    f"which does not match any polymer chain id in the "
                    f"multimer JSON ({sorted(target_map)})."
                )
            if source_chain_id in used_chain_ids:
                raise click.UsageError(
                    f"Chain id '{source_chain_id}' is supplied by more than "
                    "one monomer file."
                )
            used_chain_ids.add(source_chain_id)

            entry, target_mol_type = target_map[source_chain_id]
            if target_mol_type != mol_type:
                raise click.UsageError(
                    f"Monomer file '{mf}' has molecule type '{mol_type}' but "
                    f"target chain '{source_chain_id}' is of type "
                    f"'{target_mol_type}'."
                )

            monomer_entry = copy.deepcopy(monomer_entry)
            monomer_entry.pop("id", None)
            entry[target_mol_type].update(monomer_entry)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(merged_multimer, f, indent=4)
    logger.info("Merged JSON written to {}", output_file)


if __name__ == "__main__":
    main()
