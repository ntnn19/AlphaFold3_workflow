import click

def remove_dna(cif_file):
    """
    Remove DNA residues from CIF files in the specified folder
    and save the resulting structures in PDB format.
    """
    output_file = cif_file.replace(".cif", "_no_dna.pdb")
        
    cmd.load(cif_file)
        
    cmd.remove("resn DA+DC+DG+DT+DI")
        
    cmd.save(output_file)
        
    cmd.delete("all")

    click.echo("DNA removal complete. Modified structures saved as PDB format!")

@click.command()
@click.argument('cif_file', type=click.Path(exists=True))
def main(cif_file):
    """
    Remove DNA residues from a CIF file and save it as PDB files.

    CIF_FILE: Path to a CIF file.
    """
    finish_launching(['pymol', '-c'])
    
    remove_dna(cif_file)

if __name__ == "__main__":
    from pymol import cmd, finish_launching
    main()

