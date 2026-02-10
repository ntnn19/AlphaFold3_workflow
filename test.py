import sys
import pandas as pd
from Bio import SeqIO
import string
import os

letters = list(string.ascii_uppercase)
ground_truth_dir = "input/ground_truth"
SIMILARITY_METRIC = "sucos_shape_pocket_qcov"
SIMILARITY_BINS = [0, 20, 30, 40, 50, 60, 70, 80, 100]
bin_data = []
bin_sizes = []
bin_cis = []


SIMILARITY_BIN_LABELS = [
    f"{SIMILARITY_BINS[i]}-{SIMILARITY_BINS[i + 1]}"
    for i in range(len(SIMILARITY_BINS) - 1)
]


annotated_df = pd.read_csv('input/annotations.csv')
annotated_df_clean = annotated_df[
    annotated_df["ligand_is_proper"] & (annotated_df["sucos_shape"].notna())
].reset_index(drop=True)
annotated_df_clean["similarity_bin"] = pd.cut(
    annotated_df_clean[SIMILARITY_METRIC].fillna(0),
    bins=SIMILARITY_BINS,
    labels=SIMILARITY_BIN_LABELS,
    include_lowest=True,
)
dissimilar_systems = annotated_df_clean[annotated_df_clean.similarity_bin.isin(SIMILARITY_BIN_LABELS[:3])]
dissimilar_systems_id = dissimilar_systems["system_id"]
annotated_df = annotated_df[annotated_df.system_id.isin(dissimilar_systems_id)]
seq_d = {sys_id:SeqIO.to_dict(SeqIO.parse(os.path.join(ground_truth_dir,sys_id, "sequences.fasta"),"fasta")) for sys_id in annotated_df['system_id'].unique()}
df_grouped = annotated_df[["system_id", "ligand_ccd_code"]].astype(str).groupby('system_id')
annotated_df["job_name"]= df_grouped["ligand_ccd_code"].transform(lambda members: f"{members.name}-{'_'.join(members)}")
annotated_df["target_chain_ids"] = annotated_df["system_id"].apply(lambda x: x.split("__")[2].split("_"))

annotated_df["combined_ids"] = annotated_df.apply(lambda row: row["target_chain_ids"] + [row["ligand_instance_chain"]], axis=1)

exploded = annotated_df[["job_name", "combined_ids", "system_id"]].explode("combined_ids", ignore_index=True)

exploded["ground_truth_chain_ids"] = exploded["combined_ids"]

multimers_df = exploded[["job_name", "ground_truth_chain_ids","system_id"]].drop_duplicates().reset_index(drop=True)
multimers_df["type"] = multimers_df.apply(lambda row: "protein" if row["ground_truth_chain_ids"] in row["system_id"].split("__")[2].split("_") else "ligand", axis=1)
multimers_df["id"] = multimers_df.groupby("job_name").cumcount().map(lambda x: letters[x])
multimers_df_ligand = multimers_df[multimers_df["type"] == "ligand"]
multimers_df_target = multimers_df[multimers_df["type"] == "protein"]

multimers_df["ligand_ccd_code"] = multimers_df.job_name.str.split("-").str[-1]
multimers_df["seq_d"]=multimers_df.system_id.map(seq_d)
merged_df = multimers_df.merge(
    annotated_df[["system_id", "ligand_ccd_code", "ligand_instance_chain"]],
    how="left",
    left_on=["system_id", "ground_truth_chain_ids"],
    right_on=["system_id", "ligand_instance_chain"]
)
multimers_df["ligand_ccd_code_transformed"] = merged_df["ligand_ccd_code_y"]
multimers_df["sequence"] = multimers_df.apply(lambda row: row["seq_d"].get(row["ground_truth_chain_ids"], annotated_df[annotated_df["ligand_ccd_code"] == row["ligand_ccd_code_transformed"]]["ligand_smiles"].unique()), axis=1)
multimers_df["sequence"] = multimers_df.sequence.apply(lambda x: "".join(x))


monomers_df = annotated_df[["job_name", "target_chain_ids", "system_id"]].explode("target_chain_ids", ignore_index=True)
monomers_df["job_name"] = monomers_df["job_name"].apply(lambda x:  x.split("__")[0]) + "-" + monomers_df.target_chain_ids
monomers_df["seq_d"]=monomers_df.system_id.map(seq_d)

monomers_df["sequence"] = monomers_df.apply(lambda row: row["seq_d"].get(row["target_chain_ids"]),axis=1)
monomers_df["sequence"] = monomers_df.sequence.apply(lambda x: "".join(x))
monomers_df["id"] = "A"
monomers_df["type"] = "protein"
monomers_df["output_dir"] = os.path.join(sys.argv[1], "monomers")
multimers_df["output_dir"] = os.path.join(sys.argv[1], "multimers")
multimers_df["job_components"] = multimers_df["job_name"].str.split("__")
multimers_df["job_name"] = multimers_df["job_components"].apply(lambda x: "_".join([f"{x[0]}-{s}" for s in x[2].split("_")] + [x[-1].split("-")[-1]]))
columns_to_select = ["job_name","type","id","sequence","output_dir"]
final_df = pd.concat([multimers_df[columns_to_select], monomers_df[columns_to_select].drop_duplicates()], ignore_index=True)
#final_df["model_seeds"] = 1000
final_df["model_seeds"] = ",".join(list(map(lambda x: str(x),list(range(1,1001)))))
print(final_df.isna().sum().value_counts())
print(final_df)
print(final_df.job_name.nunique())
final_df.to_csv("test.csv",index=False)
