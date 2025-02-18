import click
#p16724_chopped_p16724_chopped_p16732_p16792_chopped
names = ["p16724_chopped","p16732","p16792_chopped"]
def modify_string_with_counts(substrings, s):
    to_permut = [f"{sub}_{s.count(sub)}" for sub in substrings if sub in s]
    permutations_l = list(permutations(to_permut))
    new_basenames = ["_".join(l) for l in permutations_l]
    return new_basenames

#    return "_".join([f"{sub}_{parts.count(sub)}" for sub in substrings if sub in parts])
def collect_pairs(s):
   if (((f"_{1}_" in s and s.count(f"_{1}_") <=2) and f"_{2}_" not in s) or ((f"_{2}_" in s and s.count(f"_{2}_") ==1) and f"_{1}_" not in s)) and bool(re.match(r'^(?=.*_(1|2)_)(?!.*_[3-9]\d*_).*$', s)):
      return s
   else:
      return []
@click.command()
@click.argument('combfold_fasta_dir')
@click.argument('af3_pdb_dir')
@click.argument('outdir')
@click.argument('log')
def main(combfold_fasta_dir,af3_pdb_dir,outdir,log):

    logging.basicConfig(filename=log, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    os.makedirs(outdir,exist_ok=True)
    af3_pdb_dir_contents =  set(os.listdir(af3_pdb_dir))

    for combfold_fasta_file in os.listdir(combfold_fasta_dir):
        renamed_combfold = modify_string_with_counts(names,combfold_fasta_file)
        matched_files = []
        for combfold_file in renamed_combfold:
            for af3_file in af3_pdb_dir_contents:
                if combfold_file in af3_file and af3_file.endswith("_no_dna.pdb"):
                    matched_files.append(af3_file)
                    exact_matched_files = min(matched_files, key=len)
        for f in [exact_matched_files]:
                shutil.copy(os.path.join(af3_pdb_dir,f), outdir)

    logger.info(f"combfold_fasta_dir={combfold_fasta_dir},outdir={outdir},Total number of pdb_files={len(list(os.listdir(outdir)))}, Total number of fasta_files={len(list(os.listdir(combfold_fasta_dir)))}")        

    pairs = []
    for af3_file in af3_pdb_dir_contents:
        if af3_file.endswith("_no_dna.pdb"):
            potential_pair = collect_pairs(af3_file)
            if potential_pair!=[]:
                shutil.copy(os.path.join(af3_pdb_dir,potential_pair), outdir)

if __name__ == '__main__':
    from itertools import permutations
    from pathlib import Path
    import os
    import shutil
    import logging
    import re
    main()
