configfile: "config/config.yaml"
import os
import pandas as pd
INPUT_DF = config["input_csv"]
OUTPUT_DIR = config["output_dir"]
MODE = config.get("mode","default")
MSA_OPTION = config.get("msa_option","auto")
AF3_CONTAINER = config["af3_flags"]["--af3_container"]
def get_af3_flag_value(flag, default_value):
    return config.get('alphafold3_flags', {}).get(flag, default_value)

# Example usage: Getting flag values from YAML with fallbacks

BUCKETS = get_af3_flag_value('--buckets', '256,512,768,1024,1280,1536,2048,2560,3072,3584,4096,4608,5120')
CONFORMER_MAX_ITERATIONS = get_af3_flag_value('--conformer_max_iterations', 100)
FLASH_ATTENTION_IMPLEMENTATION = get_af3_flag_value('--flash_attention_implementation', 'triton')
GPU_DEVICE = get_af3_flag_value('--gpu_device', 0)
HMMALIGN_BINARY_PATH = get_af3_flag_value('--hmmalign_binary_path', '/hmmer/bin/hmmalign')
HMMBUILD_BINARY_PATH = get_af3_flag_value('--hmmbuild_binary_path', '/hmmer/bin/hmmbuild')
HMMSEARCH_BINARY_PATH = get_af3_flag_value('--hmmsearch_binary_path', '/hmmer/bin/hmmsearch')
JACKHMMER_BINARY_PATH = get_af3_flag_value('--jackhmmer_binary_path', '/hmmer/bin/jackhmmer')
JACKHMMER_N_CPU = get_af3_flag_value('--jackhmmer_n_cpu', 8)
JAX_COMPILATION_CACHE_DIR = get_af3_flag_value('--jax_compilation_cache_dir', '/path/to/cache')
MAX_TEMPLATE_DATE = get_af3_flag_value('--max_template_date', '2021-09-30')
MGNIFY_DATABASE_PATH = get_af3_flag_value('--mgnify_database_path', os.path.join('/root/public_databases','mgy_clusters_2022_05.fa'))
NHMMER_BINARY_PATH = get_af3_flag_value('--nhmmer_binary_path', '/hmmer/bin/nhmmer')
NHMMER_N_CPU = get_af3_flag_value('--nhmmer_n_cpu', 8)
NTRNA_DATABASE_PATH = get_af3_flag_value('--ntrna_database_path', os.path.join('/root/public_databases','nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta'))
NUM_DIFFUSION_SAMPLES = get_af3_flag_value('--num_diffusion_samples', 5)
NUM_RECYCLES = get_af3_flag_value('--num_recycles', 10)
if "--num_seeds" in config["af3_flags"]:
    NUM_SEEDS_ARG = f"--num_seeds={config['af3_flags']['--num_seeds']}"
else:
    NUM_SEEDS_ARG = f""

PDB_DATABASE_PATH = get_af3_flag_value('--pdb_database_path', os.path.join('/root/public_databases','mmcif_files'))
RFAM_DATABASE_PATH = get_af3_flag_value('--rfam_database_path', os.path.join('/root/public_databases','rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta'))
RNA_CENTRAL_DATABASE_PATH = get_af3_flag_value('--rna_central_database_path', os.path.join('/root/public_databases','rnacentral_active_seq_id_90_cov_80_linclust.fasta'))
SAVE_EMBEDDINGS = get_af3_flag_value('--save_embeddings', False)
SEQRES_DATABASE_PATH = get_af3_flag_value('--seqres_database_path', os.path.join('/root/public_databases','pdb_seqres_2022_09_28.fasta'))
SMALL_BFD_DATABASE_PATH = get_af3_flag_value('--small_bfd_database_path', os.path.join('/root/public_databases','bfd-first_non_consensus_sequences.fasta'))
UNIPROT_CLUSTER_ANNOT_DATABASE_PATH = get_af3_flag_value('--uniprot_cluster_annot_database_path', os.path.join('/root/public_databases','uniprot_all_2021_04.fa'))
UNIREF90_DATABASE_PATH = get_af3_flag_value('--uniref90_database_path', os.path.join('/root/public_databases','uniref90_2022_05.fa'))




def get_af3_outputs(wildcards):
    PREPROCESSING_DIR = checkpoints.PREPROCESSING.get(**wildcards).output[0]
    JOB_NAMES, = glob_wildcards(os.path.join(PREPROCESSING_DIR, "{i}.json"))
    return  list(expand(
        os.path.join(PREPROCESSING_DIR,"{i}.json")
        ,i=JOB_NAMES))+ list(expand(os.path.join(OUTPUT_DIR,"AF3_INFERENCE","{i}/{i}/{i}_model.cif"),i=JOB_NAMES))+list(expand(
        os.path.join(OUTPUT_DIR,"AF3_DATA","{i}/{i}_data.json"),i=JOB_NAMES))


def prepare_colabfold_search_compatible_table(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file)
    df=  df[df["type"]!="ligand"]
    df = df[["job_name","sequence"]]
    # Rename 'job_name' to 'id'
    df.rename(columns={'job_name': 'id'},inplace=True)
    print(df)
    exit()
    # Group by 'id' and join sequences with ':'
    df_grouped = df.groupby('id')['sequence'].apply(lambda x: ':'.join(x)).reset_index()

    # Sort by sequence length
    df_grouped['seq_length'] = df_grouped['sequence'].apply(len)
    df_sorted = df_grouped.sort_values(by='seq_length').drop(columns=['seq_length'])

    # Save to a new CSV file
    df_sorted.to_csv(output_file,index=False)

rule af3_all:
    input:
        get_af3_outputs


checkpoint PREPROCESSING:
    input:
        INPUT_DF
    output:
        directory(os.path.join(OUTPUT_DIR,"PREPROCESSING")),
        os.path.join(OUTPUT_DIR,"PREPROCESSING","task_table_auto_template_free.csv"),
        os.path.join(OUTPUT_DIR,"PREPROCESSING","task_table_auto_template_based.csv")
    params:
        msa_option = MSA_OPTION,
        mode = MODE
    shell:
        """
        if [[ "{params.msa_option}" == "auto" ]]; then
            python workflow/scripts/create_tasks_from_dataframe.py {input} {output[0]} --msa-option auto_template_free --mode {params.mode}
            python workflow/scripts/create_tasks_from_dataframe.py {input} {output[0]} --msa-option auto_template_based --mode {params.mode}
        else
            python workflow/scripts/create_tasks_from_dataframe.py {input} {output[0]} --mode {params.mode}
        fi
        """

rule PREPARE_COLABFOLD_SEARCH_INPUTS:
    input:
        os.path.join(OUTPUT_DIR,"PREPROCESSING","task_table_auto_template_free.csv"),
         os.path.join(OUTPUT_DIR,"PREPROCESSING","task_table_auto_template_based.csv")
    output:
        os.path.join(OUTPUT_DIR,"PREPARE_COLABFOLD_SEARCH_INPUTS","task_table_auto_template_free_colabfold_search_compatible.csv"),
        os.path.join(OUTPUT_DIR,"PREPARE_COLABFOLD_SEARCH_INPUTS","task_table_auto_template_based_colabfold_search_compatible.csv")
    run:
        prepare_colabfold_search_compatible_table(input[0],output[0])
        prepare_colabfold_search_compatible_table(input[1],output[1])

rule MMSEQS2:
    input:
        jsons = os.path.join(OUTPUT_DIR,"PREPROCESSING","{i}.json"),
        #task_table = os.path.join(OUTPUT_DIR,"PREPARE_COLABFOLD_SEARCH_INPUTS","task_table_auto_template_free_colabfold_search_compatible.csv")
    params:
        "pass"
    output:
        data_pipeline_msa = os.path.join(OUTPUT_DIR,"AF3_DATA","{i}/{i}_data.json"),
    container:
        AF3_CONTAINER
    shell:
        """
        python /app/alphafold/run_alphafold.py --json_path=/root/af_output/PREPROCESSING/{wildcards.i}.json \
        --model_dir=/root/models \
        --output_dir=/root/af_output/AF3_DATA \
        --db_dir=/root/public_databases \
        --run_data_pipeline=true \
        --run_inference=false \
        --buckets={params.buckets} \
        --conformer_max_iterations={params.conformer_max_iterations} \
        --flash_attention_implementation={params.flash_attention_implementation} \
        --gpu_device={params.gpu_device} \
        --hmmalign_binary_path={params.hmmalign_binary_path} \
        --hmmbuild_binary_path={params.hmmbuild_binary_path} \
        --hmmsearch_binary_path={params.hmmsearch_binary_path} \
        --jackhmmer_binary_path={params.jackhmmer_binary_path} \
        --jackhmmer_n_cpu={params.jackhmmer_n_cpu} \
        --jax_compilation_cache_dir={params.jax_compilation_cache_dir} \
        --max_template_date={params.max_template_date} \
        --mgnify_database_path={params.mgnify_database_path} \
        --nhmmer_binary_path={params.nhmmer_binary_path} \
        --nhmmer_n_cpu={params.nhmmer_n_cpu} \
        --ntrna_database_path={params.ntrna_database_path} \
        --num_diffusion_samples={params.num_diffusion_samples} \
        --num_recycles={params.num_recycles} \
        {params.num_seeds_arg} \
        --pdb_database_path={params.pdb_database_path} \
        --rfam_database_path={params.rfam_database_path} \
        --rna_central_database_path={params.rna_central_database_path} \
        --save_embeddings={params.save_embeddings} \
        --seqres_database_path={params.seqres_database_path} \
        --small_bfd_database_path={params.small_bfd_database_path} \
        --uniprot_cluster_annot_database_path={params.uniprot_cluster_annot_database_path} \
        --uniref90_database_path={params.uniref90_database_path}
        """


rule AF3_INFERENCE:
    input:
        data_pipeline_msa= os.path.join(OUTPUT_DIR,"AF3_DATA","{i}/{i}_data.json"),
    params:
        buckets = BUCKETS,
        conformer_max_iterations = CONFORMER_MAX_ITERATIONS,
        flash_attention_implementation = FLASH_ATTENTION_IMPLEMENTATION,
        gpu_device = GPU_DEVICE,
        hmmalign_binary_path = HMMALIGN_BINARY_PATH,
        hmmbuild_binary_path = HMMBUILD_BINARY_PATH,
        hmmsearch_binary_path = HMMSEARCH_BINARY_PATH,
        jackhmmer_binary_path = JACKHMMER_BINARY_PATH,
        jackhmmer_n_cpu = JACKHMMER_N_CPU,
        jax_compilation_cache_dir = JAX_COMPILATION_CACHE_DIR,
        max_template_date = MAX_TEMPLATE_DATE,
        mgnify_database_path = MGNIFY_DATABASE_PATH,
        nhmmer_binary_path = NHMMER_BINARY_PATH,
        nhmmer_n_cpu = NHMMER_N_CPU,
        ntrna_database_path = NTRNA_DATABASE_PATH,
        num_diffusion_samples = NUM_DIFFUSION_SAMPLES,
        num_recycles = NUM_RECYCLES,
        num_seeds_arg = NUM_SEEDS_ARG,
        pdb_database_path = PDB_DATABASE_PATH,
        rfam_database_path = RFAM_DATABASE_PATH,
        rna_central_database_path = RNA_CENTRAL_DATABASE_PATH,
        save_embeddings = SAVE_EMBEDDINGS,
        seqres_database_path = SEQRES_DATABASE_PATH,
        small_bfd_database_path = SMALL_BFD_DATABASE_PATH,
        uniprot_cluster_annot_database_path = UNIPROT_CLUSTER_ANNOT_DATABASE_PATH,
        uniref90_database_path = UNIREF90_DATABASE_PATH
    output:
        os.path.join(OUTPUT_DIR,"AF3_INFERENCE","{i}/{i}/{i}_model.cif"),
    container:
        AF3_CONTAINER
    shell:
        """
        python /app/alphafold/run_alphafold.py --json_path=/root/af_output/AF3_DATA/{wildcards.i}/{wildcards.i}_data.json \
        --model_dir=/root/models \
        --output_dir=/root/af_output/AF3_INFERENCE/{wildcards.i} \
        --db_dir=/root/public_databases \
        --run_data_pipeline=false \
        --run_inference=true \
        --buckets={params.buckets} \
        --conformer_max_iterations={params.conformer_max_iterations} \
        --flash_attention_implementation={params.flash_attention_implementation} \
        --gpu_device={params.gpu_device} \
        --hmmalign_binary_path={params.hmmalign_binary_path} \
        --hmmbuild_binary_path={params.hmmbuild_binary_path} \
        --hmmsearch_binary_path={params.hmmsearch_binary_path} \
        --jackhmmer_binary_path={params.jackhmmer_binary_path} \
        --jackhmmer_n_cpu={params.jackhmmer_n_cpu} \
        --jax_compilation_cache_dir={params.jax_compilation_cache_dir} \
        --max_template_date={params.max_template_date} \
        --mgnify_database_path={params.mgnify_database_path} \
        --nhmmer_binary_path={params.nhmmer_binary_path} \
        --nhmmer_n_cpu={params.nhmmer_n_cpu} \
        --ntrna_database_path={params.ntrna_database_path} \
        --num_diffusion_samples={params.num_diffusion_samples} \
        --num_recycles={params.num_recycles} \
        {params.num_seeds_arg} \
        --pdb_database_path={params.pdb_database_path} \
        --rfam_database_path={params.rfam_database_path} \
        --rna_central_database_path={params.rna_central_database_path} \
        --save_embeddings={params.save_embeddings} \
        --seqres_database_path={params.seqres_database_path} \
        --small_bfd_database_path={params.small_bfd_database_path} \
        --uniprot_cluster_annot_database_path={params.uniprot_cluster_annot_database_path} \
        --uniref90_database_path={params.uniref90_database_path}
        """