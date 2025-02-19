import click

@click.command()
@click.argument("input_json", type=click.Path(exists=True))
@click.argument('output', type=click.Path(), required=True)
@click.argument('dialect', type=str, required=True)
def split_json_and_create_job_list(input_json,output,dialect):
    """
    Splits a JSON file (with a top-level list) into multiple JSON files,
    where each resulting JSON contains a single item wrapped in a list and a corresponding job list file with each new JSON file as an input.

    INPUT_JSON: Path to the input JSON file.
    OUTPUT: Path to the output job list file.
    """

    split_json_output_dir = os.path.join(os.path.dirname(os.path.dirname(input_json)),"inference")
    joblist_output_dir = os.path.dirname(output)

    split_json_output_dir_output_path = Path(split_json_output_dir)
    joblist_output_dir_output_path = Path(joblist_output_dir)

    split_json_output_dir_output_path.mkdir(parents=True, exist_ok=True)
    joblist_output_dir_output_path.mkdir(parents=True, exist_ok=True)
    print(os.path.dirname(joblist_output_dir_output_path))
    predictions_root_dir= os.path.dirname(joblist_output_dir_output_path)
    if dialect not in ["server","local"]:
        raise ValueError(f"The dialect {dialect} must be set to either 'local' or 'sever'.")
    if dialect == "server":
        master_json_path = os.path.join(os.path.dirname(os.path.dirname(input_json)),"combination_0.json")
        print("master_json_path=",master_json_path)
        with open(master_json_path, 'r') as f:
            data = json.load(f)
    else:
        with open(input_json, 'r') as f:
            data = json.load(f)

    # Ensure the top-level of the JSON is a list
    if isinstance(data, list):
        print(f"The top-level of {input_json} is a list.")
    elif isinstance(data, dict):
        print(f"The top-level of {input_json} is a dictionary.")
    else:
        raise ValueError(f"The top-level of {input_json} must be either a dictionary or a list.")

    lines=[]

    for index, item in enumerate(data):
        if dialect=="server":
#        if index==4: # DEBUG
#            break
            item_output_dir = os.path.join(predictions_root_dir,item['name'].lower())
            item_json_file = os.path.join(item_output_dir,f"{item['name'].lower()}_data.json")
#        output_json_file = split_json_output_dir_output_path / f"{os.path.splitext(os.path.basename(input_json))[0]}_{index}.json"
#        with open(output_json_file, 'w') as f:
#            json.dump([item], f, indent=2)  # Wrap item in a list

            cmd=f"python /app/alphafold/run_alphafold.py --json_path={item_json_file} --model_dir=/root/models --output_dir=/root/af_output/{item['name'].lower()} --db_dir=/root/public_databases --run_data_pipeline=false --run_inference=true\n"
        else:
#        if index==4: # DEBUG
#            break
            k="name"
            if item!="name": continue
            print(index,item,k)
            item_output_dir = os.path.join(predictions_root_dir,data[k].lower())
            item_json_file = os.path.join(item_output_dir,f"{data[k].lower()}_data.json")
            print(index,item,k,item_output_dir,item_json_file)
#        output_json_file = split_json_output_dir_output_path / f"{os.path.splitext(os.path.basename(input_json))[0]}_{index}.json"
#        with open(output_json_file, 'w') as f:
#            json.dump([item], f, indent=2)  # Wrap item in a list

            cmd=f"python /app/alphafold/run_alphafold.py --json_path={item_json_file} --model_dir=/root/models --output_dir=/root/af_output/{data[k].lower()} --db_dir=/root/public_databases --run_data_pipeline=false --run_inference=true\n"
        lines.append(cmd)

    with open(output, 'w') as f:
        f.write("\n".join(lines))

#    click.echo(f"Split {len(data)} items into {split_json_output_dir_output_path}")
    click.echo(f"Prepared job list {output}")



#        workflow/scripts/parallel.sh {input}
#        python /app/alphafold/run_alphafold.py --json_path={input} \
#        --model_dir=/root/models \
#        --output_dir=/root/af_output \
#        --db_dir=/root/public_databases \
#        --run_data_pipeline=false \
#        --run_inference=true


if __name__ == '__main__':
    import json
    from pathlib import Path
    import os
    split_json_and_create_job_list()
