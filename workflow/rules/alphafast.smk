
checkpoint ALPHAFAST:
    input:
        data = os.path.join(OUTPUT_DIR, "preprocessing", "batch_{batch}")
    params:
        databases_dir = DB_DIR,
        extra_af3_data_flags = EXTRA_AF3_DATA_FLAGS,
        output_dir = OUTPUT_DIR,
        input_dir = ALPHAFAST_INPUT_DIR,
        batch_size = ALPHAFAST_BATCH_SIZE,
        backend = ALPHAFAST_BACKEND,
        temp_dir = ALPHAFAST_TEMP_DIR,
        rna_mmseqs_db_dir = ALPHAFAST_RNA_MMSEQS_DB_DIR,
        use_nhmmer = ALPHAFAST_USE_NHMMER,
        jax_compilation_cache_dir = ALPHAFAST_JAX_COMPILATION_CACHE_DIR,
        jax_cache_container_dir = ALPHAFAST_JAX_CACHE_CONTAINER_DIR,
        temp_dir_container = ALPHAFAST_TEMP_DIR_CONTAINER,
    output:
        data_pipeline_monomers=directory(os.path.join(OUTPUT_DIR,"rule_ALPHAFAST","batch_{batch}"))
    log:
        os.path.join(OUTPUT_DIR, "logs", "rule_ALPHAFAST", "batch_{batch}.log"),
    benchmark:
        os.path.join(OUTPUT_DIR, "benchmarks", "rule_ALPHAFAST", "batch_{batch}.tsv"),
    container:
        ALPHAFAST_CONTAINER
    shell:
        """
        (
        set -euo pipefail
        set -x
        CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits \
                2>/dev/null | head -n 1 | cut -d'.' -f1 || echo 0)
        if [[ "$CC" -ge 8 ]]; then
            FLASH_ARG=""
        else
            export XLA_FLAGS="--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"
            FLASH_ARG="--flash_attention_implementation=xla"
        fi
        NUM_GPUS=$(nvidia-smi -L | wc -l)
        LOGICAL_GPU_LIST=$(seq -s, 0 $((NUM_GPUS - 1)))
        /app/alphafold/scripts/run_multigpu.sh \
                            {input.data} {params.output_dir}/rule_ALPHAFAST/batch_{wildcards.batch} \
                            {params.output_dir}/rule_ALPHAFAST/batch_{wildcards.batch} \
                            $NUM_GPUS {params.batch_size} $LOGICAL_GPU_LIST $FLASH_ARG \
                            ) > {log} 2>&1
        """

rule batch_done:
    input: get_ipsae_outputs
    output: touch(os.path.join(OUTPUT_DIR, "flags", "alphafast_batch_{batch}.done"))

rule MOCK:
    input:
        expand(os.path.join(OUTPUT_DIR, "flags", "alphafast_batch_{batch}.done"), batch=get_alphafast_inputs)
    output: touch("done.txt")