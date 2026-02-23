import click
import yaml
import os

@click.command()
@click.argument("config", type=click.Path(exists=True))
@click.option("-o","--output-dir", type=click.Path(exists=True),default = os.getcwd())
def setup_directories(config,output_dir):
    """Reads CONFIG YAML file and creates necessary directories."""
    
    # Load YAML config
    with open(config, "r") as file:
        config_data = yaml.safe_load(file)
    
    # Extract required paths
    output_dir_ = os.path.join(output_dir,config_data.get("output_dir","results"))
    tmp_dir = os.path.join(output_dir,config_data.get("tmp_dir","tmp"))

    if not output_dir or not tmp_dir:
        click.echo("Error: 'output_dir' or 'tmp_dir' is missing in the config file.", err=True)
        return


    for directory in [output_dir_, tmp_dir]:
        os.makedirs(directory, exist_ok=True)
        click.echo(f"Created or verified: {directory}")

if __name__ == "__main__":
    setup_directories()
