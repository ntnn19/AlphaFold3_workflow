import glob
import json
import os
from pathlib import Path
import click

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def write_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def extract_monomer_key(path):
    # e.g. "p1_auto_template_free_data.json" -> "p1_auto_template_free"
    return Path(path).stem.replace("_data", "")

def merge_jsons(multimer_data, monomer_data_list):
    """
    Merge monomer data into the multimer data.
    """
    for multimer_entity_dict in multimer_data["sequences"]:
        for monomer_json in monomer_data_list:
            if "protein" in multimer_entity_dict:
                if multimer_entity_dict["protein"]["sequence"].replace("U","X") == monomer_json["sequences"][0]["protein"]["sequence"]:
                    multimer_entity_dict["protein"]["unpairedMsa"] = monomer_json["sequences"][0]["protein"]["unpairedMsa"]
                    multimer_entity_dict["protein"]["pairedMsa"] = monomer_json["sequences"][0]["protein"]["pairedMsa"]
                    multimer_entity_dict["protein"]["templates"] = monomer_json["sequences"][0]["protein"].get("templates", [])
            if "rna" in multimer_entity_dict:
                if multimer_entity_dict["rna"]["sequence"]  == monomer_json["sequences"][0]["rna"]["sequence"]:
                    multimer_entity_dict["rna"]["unpairedMsa"] = monomer_json["sequences"][0]["rna"]["unpairedMsa"]
                    multimer_entity_dict["rna"]["pairedMsa"] = monomer_json["sequences"][0]["rna"]["unpairedMsa"]
            if "dna" in multimer_entity_dict:
                if multimer_entity_dict["dna"]["sequence"]  == monomer_json["sequences"][0]["dna"]["sequence"]:
                    multimer_entity_dict["dna"]["unpairedMsa"] = monomer_json["sequences"][0]["dna"]["unpairedMsa"]
                    multimer_entity_dict["dna"]["pairedMsa"] = monomer_json["sequences"][0]["dna"]["unpairedMsa"]
    return multimer_data

@click.command()
@click.argument("monomers_dir", type=click.Path())
@click.argument("multimer_file", type=click.Path())
@click.argument("output_dir",  type=click.Path())
def main(monomers_dir,multimer_file,output_dir):
    # Separate monomeric and multimeric files
    monomer_files =  []
    multimer_files =  [multimer_file]
    print("multimer_file=",multimer_files)


    # Merge monomer data into each multimer
    for multimer_path in multimer_files:
        multimer_name = os.path.splitext(os.path.basename(multimer_path))[0]  # e.g. "p1_b1_auto_template_free"
        print("multimer_name=",multimer_name)
        multimer_base = multimer_name + ".json"
        multimer_json = load_json(multimer_path)

        matched_monomers = []
        suffix = "_auto_template_based" if "_auto_template_based" in multimer_name else "_auto_template_free"
        monomer_1 = multimer_name.split(suffix)[0].split("_")[0]
        monomer_2 = multimer_name.split(suffix)[0].split("_")[1]
        monomer_files.append(os.path.join(monomers_dir,f"{monomer_1}{suffix}", f"{monomer_1}{suffix}_data.json"))
        monomer_files.append(os.path.join(monomers_dir,f"{monomer_2}{suffix}", f"{monomer_2}{suffix}_data.json"))

#        monomer_data = {}
        for mf in monomer_files:
            key = extract_monomer_key(mf)
#            monomer_data[key] = load_json(mf)
            matched_monomers.append(load_json(mf))
        click.echo(f"matched_monomers= {[m for m in monomer_files]}")
        click.echo(f"multimer_json= {multimer_json}")
        merged = merge_jsons(multimer_json, matched_monomers)
        sub_output_dir = os.path.join(output_dir,multimer_name)
        os.makedirs(sub_output_dir, exist_ok=True)
        out_path = os.path.join(sub_output_dir,multimer_base.replace(".json","_data.json"))

        write_json(merged, out_path)
        click.echo(f"Wrote merged JSON to {out_path}")

if __name__ == "__main__":
    main()
