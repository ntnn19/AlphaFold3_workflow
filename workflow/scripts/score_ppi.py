#EXAMPLE CMD:  python workflow/scripts/score_ppi.py /scratch/home/nagarnat/tests/test_af3_workflow/output/stoichio_screen/AF3_INFERENCE/st_run_1_c1/st_run_1_c1_model.cif /scratch/home/nagarnat/tests/test_af3_workflow/output/stoichio_screen/AF3_INFERENCE/st_run_1_c1/st_run_1_c1_confidences.json
import numpy as np
import gemmi
import click
import json

def load_json(json_file):
    with open(json_file,"r") as f:
        return json.load(f)
def extract_cb_coords_from_mmcif(cif_file):
    structure = gemmi.read_structure(cif_file)
    model = structure[0]  # first model

    CB_coords = []

    for chain in model:
        for residue in chain:
            resname = residue.name  # 3-letter code (e.g. GLY)

            # Try CB first
            cb = residue.find_atom("CB", "*")
            if cb is not None:
                pos = cb.pos
                CB_coords.append([pos.x, pos.y, pos.z])
            elif resname == "GLY":
                # Glycine fallback to CA
                ca = residue.find_atom("CA", "*")
                if ca is not None:
                    pos = ca.pos
                    CB_coords.append([pos.x, pos.y, pos.z])

    return np.array(CB_coords)

def score_PPI(CB_coords, plddt, l1):
    """Score the PPI
    """

    #Cβs within 8 Å from each other from different chains are used to define the interface.
    CB_dists = np.sqrt(np.sum((CB_coords[:,None]-CB_coords[None,:])**2,axis=-1))

    #Get contacts
    contact_dists = CB_dists[:l1,l1:] #upper triangular --> first dim = chain 1
    contacts = np.argwhere(contact_dists<=100)

    #Get plddt per chain
    plddt1 = plddt[:l1]
    plddt2 = plddt[l1:]

    if contacts.shape[0]<1:
        pdockq=0
        avg_if_plddt=0
        n_if_contacts=0
    else:
        #Get the average interface plDDT
        avg_if_plddt = np.average(np.concatenate([plddt1[np.unique(contacts[:,0])], plddt2[np.unique(contacts[:,1])]]))
        #Get the number of interface contacts
        n_if_contacts = contacts.shape[0]
        x = avg_if_plddt*np.log10(n_if_contacts)
        pdockq = 0.724 / (1 + np.exp(-0.052*(x-152.611)))+0.018


    return pdockq, avg_if_plddt, n_if_contacts
    
@click.command()
@click.argument('cif_file', type=click.Path(exists=True))
@click.argument('confidence_file', type=click.Path(exists=True))
def main(cif_file,confidence_file):
    CB_coords = extract_cb_coords_from_mmcif(cif_file)
    confidence_scores  = load_json(confidence_file)
    plddt  = confidence_scores["atom_plddts"]
    seq_len = len(confidence_scores["token_res_ids"])
    scores = score_PPI(CB_coords, plddt, seq_len)
    print("CB_coords=", CB_coords[:10, :])  # first 10 atoms, all 3 coordinates
    print("CB_coords=", CB_coords.shape)  # first 10 atoms, all 3 coordinates
    print("pLDDT=", plddt[:10])  # first 10 atoms, all 3 coordinates
    print("pLDDT=", len(plddt))  # first 10 atoms, all 3 coordinates
    print("seq len=", seq_len)  # first 10 atoms, all 3 coordinates
    print("PPI scores=", scores)
if __name__ == "__main__":
    main()
