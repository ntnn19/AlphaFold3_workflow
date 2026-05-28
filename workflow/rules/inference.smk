if EXCLUSIVE_LOCK:
    _EXCLUSIVE_LOCK = r"""
    bash /app/scripts/gpu_lock.sh $PWD/.snakemake/.gpu_locks
    """
else:
    _EXCLUSIVE_LOCK = r""""""

_FLASH_DETECT = r"""
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits \
        2>/dev/null | head -n 1 | cut -d'.' -f1 || echo 0)
if [[ "$CC" -ge 8 ]]; then
    FLASH_ARG=""
else
    export XLA_FLAGS="--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"
    FLASH_ARG="--flash_attention_implementation=xla"
fi
"""

rule AF3_INFERENCE:
    input:
        data = branch(
            lookup(query="sample_id == '{multi}'",within=INFERENCE_READY_DF,cols="file"),
            then=lookup(query="sample_id == '{multi}'",within=INFERENCE_READY_DF,cols="file"),
            otherwise=os.path.join(OUTPUT_DIR,"rule_MERGE_MONOMERS_TO_MULTIMERS","{multi}_data.json")
        ),
    output:
        os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{multi}", "{multi}_model.cif")
    log:
        os.path.join(OUTPUT_DIR, "logs", "rule_AF3_INFERENCE", "{multi}.log"),
    resources:
        mem_mb      = 16000,
        runtime     = 480,
        nvidia_gpu  = 1,   # standard Snakemake GPU resource
    benchmark:
        os.path.join(OUTPUT_DIR, "benchmarks", "rule_AF3_INFERENCE", "{multi}.tsv"),
    params:
        extra_af3_flags = EXTRA_AF3_FLAGS,
        exclusive_lock = "true" if EXCLUSIVE_LOCK else "false",
        models_dir = MODELS_DIR,
        output_dir = lambda w, output: str(Path(output[0]).parents[2]),
        database_dir = DB_DIR,
        shell_preamble = _FLASH_DETECT + _EXCLUSIVE_LOCK,
    container:
        AF3_CONTAINER
    shell:
        """
        {params.shell_preamble}
        python \
            /app/alphafold/run_alphafold.py $FLASH_ARG --json_path={input.data} \
            --model_dir={params.models_dir} \
            --output_dir={params.output_dir}/rule_AF3_INFERENCE \
            --db_dir={params.database_dir} \
            --run_data_pipeline=false \
            --run_inference=true \
            {params.extra_af3_flags} 2>&1 | tee {log}
        """
