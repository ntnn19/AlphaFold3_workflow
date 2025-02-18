import os
import click
@click.command()
@click.argument('input_json', type=click.Path(exist=True), required=True)
@click.argument('output', type=click.Path(), required=True)
def main(input_json, output):
        touch("{dataset}/af_output/done_flags/data_pipeline/combination_{i}_data_pipeline.json.done.txt")
#        "{dataset}/af_input/data_pipeline/combination_{i}.json"

    outdir = os.path.dirname(output)
    os.makedirs(outdir,exist_ok=True)
    with open(os.path.join(outdir,"jobs.txt"), 'a') as f:
        lines=[]
        cmd=f"python /app/alphafold/run_alphafold.py --json_path={input_json} --model_dir=/root/models --output_dir=/root/af_output --db_dir=/root/public_databases --run_data_pipeline=false --run_inference=true"
        lines.append(cmd)

        f.write("\n".join(lines))


#        workflow/scripts/parallel.sh {input}
#        python /app/alphafold/run_alphafold.py --json_path={input} \
#        --model_dir=/root/models \
#        --output_dir=/root/af_output \
#        --db_dir=/root/public_databases \
#        --run_data_pipeline=false \
#        --run_inference=true


if __name__ == '__main__':
    main()
