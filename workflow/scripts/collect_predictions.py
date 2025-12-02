import re
import glob
import os
import shutil
import click

@click.command()
@click.option('--job-list', required=True, type=click.Path(exists=True), help='Path to job list file (.txt)')
@click.option('--source-dir', default='output/AF3_INFERENCE', show_default=True, help='Base directory where job folders reside')
@click.option('--output-dir', default='collected_cifs', show_default=True, help='Directory to copy renamed CIF files into')
def collect_cifs(job_list, source_dir, output_dir):
    """
    Collects .cif files from AF3 job outputs, renames them with seed/sample info, and copies to a single output dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(job_list) as f:
        for line in f:
            match = re.search(r"--json_path=.*?/([^/]+)_data\.json", line)
            if not match:
                continue
            job_name = match.group(1)
            job_path = source_dir
            cif_files = glob.glob(f"{job_path}/**/*.cif", recursive=True)

            for cif_path in cif_files:
                rel_path = cif_path.replace(job_path + "/", "")
                match = re.search(r"(seed-\d+).*?(sample-\d+)", cif_path)
                if match:
                    seed, sample = match.groups()
                    new_filename = f"{job_name}_{sample}_model.cif"
                else:
                    continue

                dest_path = os.path.join(output_dir, new_filename)
                shutil.copyfile(cif_path, dest_path)
                click.echo(f"Copied: {cif_path} → {dest_path}")

if __name__ == '__main__':
    collect_cifs()
