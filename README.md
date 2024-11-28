This Workflow allows separate execution of the CPU &  GPU steps. It also distributes the inference runs accross multiple GPU devices using GNU parallel. 
1. Build the singularity container that supports parallel inference runs using the following command:
singularity build alphafold3_parallel.sif docker://ntnn19/alphafold3:latest_parallel_a100_40gb
<number_of_inference_job_lists> should be set to 1 for local runs and n for slurm runs, where n is the number of nodes with GPU
2. Download alphafold3 databases and obtain the weights

The following steps assume that you are located in the project directory.

3. Clone this repo to your project directory. It must follow the following structure after cloning:

![image](https://github.com/user-attachments/assets/18bb634a-fa2d-41a0-b3a9-e55b72c7fb6a)


An example json file can be found in this repo under example/example.json

4. Create & activate  snakemake environment

   Install mamba/micromamba

   mamba create env -p $(pwd)/env -f environment.yml

   mamba activate $(pwd)/env

6. Run the workflow
# Dry local run 
snakemake --use-singularity --config af3_container=<path_to_your_alphafold3_container> --singularity-args '--nv -B <alphafold3_weights_dir>:/root/models -B $(pwd)/<dataset_directory>/af_input:/root/af_input -B $(pwd)/<dataset_directory>/af_output:/root/af_output -B <path_to_alphafold3_db_directory>:/root/public_databases' -c all --set-scatter split=<number_of_inference_job_lists> -n
# Dry run with slurm
snakemake --use-singularity --config af3_container=<path_to_your_alphafold3_container> --singularity-args '--nv -B <alphafold3_weights_dir>:/root/models -B $(pwd)/<dataset_directory>/af_input:/root/af_input -B $(pwd)/<dataset_directory>/af_output:/root/af_output -B <path_to_alphafold3_db_directory>:/root/public_databases' -j 99 --executor slurm --set-scatter split==<number_of_inference_job_lists> -n

# Local run 
snakemake --use-singularity --config af3_container=<path_to_your_alphafold3_container> --singularity-args '--nv -B <alphafold3_weights_dir>:/root/models -B $(pwd)/<dataset_directory>/af_input:/root/af_input -B $(pwd)/<dataset_directory>/af_output:/root/af_output -B <path_to_alphafold3_db_directory>:/root/public_databases' -c all --set-scatter split=<number_of_inference_job_lists>
# Run with slurm
snakemake --use-singularity --config af3_container=<path_to_your_alphafold3_container> --singularity-args '--nv -B <alphafold3_weights_dir>:/root/models -B $(pwd)/<dataset_directory>/af_input:/root/af_input -B $(pwd)/<dataset_directory>/af_output:/root/af_output -B <path_to_alphafold3_db_directory>:/root/public_databases' -j 99 --executor slurm --set-scatter split==<number_of_inference_job_lists>
