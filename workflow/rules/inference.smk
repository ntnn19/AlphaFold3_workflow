rule AF3_INFERENCE:
    input:
        _helper = workflow.source_path("../scripts/gpu_lock.sh"),
        data = branch(
            lookup(query="sample_id == '{mut}'" if MUTATION_DF_PATH is not None else "sample_id == '{multi}'", within=INFERENCE_READY_DF, cols="file"),
            then=lookup(query="sample_id == '{mut}'" if MUTATION_DF_PATH is not None else "sample_id == '{multi}'", within=INFERENCE_READY_DF, cols="file"),
            otherwise=lambda w: (
                os.path.join(
                    OUTPUT_DIR, "rule_MUTATE",
                    re.match(r"(.+_seed-\d+)", w.mut).group(1),
                    f"{w.mut}.json"
                )
                if MUTATION_DF_PATH is not None
                else os.path.join(
                    OUTPUT_DIR, "rule_MERGE_MONOMERS_TO_MULTIMERS", f"{w.multi}_data.json"
                )
            )
        )
    output:
        model=expand(
            os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{{multi}}",
                            "seed-{{seed}}_sample-{sample}",
                            "{{multi}}_seed-{{seed}}_sample-{sample}_model.cif" if AF3_VERSION not in ["v3.0.0", "v3.0.1"] else "model.cif"),
            sample=range(N_SAMPLES)
        ) if MUTATION_DF.empty else
        expand(
            os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{{mut}}",
                            "seed-{{seed}}_sample-{sample}",
                            "{{mut}}_seed-{{seed}}_sample-{sample}_model.cif" if AF3_VERSION not in ["v3.0.0", "v3.0.1"] else "model.cif"),
            sample=range(N_SAMPLES)
        ),
        scores=expand(
            os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{{multi}}",
                            "seed-{{seed}}_sample-{sample}",
                            "{{multi}}_seed-{{seed}}_sample-{sample}_confidences.json" if AF3_VERSION not in ["v3.0.0", "v3.0.1"] else "confidences.json"),
            sample=range(N_SAMPLES)
        ) if MUTATION_DF.empty else
        expand(
            os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{{mut}}",
                            "seed-{{seed}}_sample-{sample}",
                            "{{mut}}_seed-{{seed}}_sample-{sample}_confidences.json" if AF3_VERSION not in ["v3.0.0", "v3.0.1"] else "confidences.json"), 
            sample=range(N_SAMPLES)
        ),
    log:
        os.path.join(OUTPUT_DIR, "logs", "rule_AF3_INFERENCE", "{multi}_seed-{seed}.log") if MUTATION_DF.empty else os.path.join(OUTPUT_DIR, "logs", "rule_AF3_INFERENCE", "{mut}_seed-{seed}.log"),
    benchmark:
        os.path.join(OUTPUT_DIR, "benchmarks", "rule_AF3_INFERENCE", "{multi}_seed-{seed}.tsv") if MUTATION_DF.empty else os.path.join(OUTPUT_DIR, "benchmarks", "rule_AF3_INFERENCE", "{mut}_seed-{seed}.tsv"),
    threads: 16
    resources:
        mem_mb      = 16000,
        runtime     = 480,
    params:
        extra_af3_flags = EXTRA_AF3_FLAGS,
        exclusive_lock = "true" if EXCLUSIVE_LOCK else "false",
        models_dir = MODELS_DIR,
        output_dir = OUTPUT_DIR,
        database_dir = DB_DIR,
        flash_detect = _FLASH_DETECT
    container:
        AF3_CONTAINER
    shell:
        """
        CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits \
                2>/dev/null | head -n 1 | cut -d'.' -f1 || echo 0)
        if [[ "$CC" -ge 8 ]]; then
            FLASH_ARG=""
        else
            export XLA_FLAGS="--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"
            FLASH_ARG="--flash_attention_implementation=xla"
        fi
        echo "FLASH_ARG"
        echo $FLASH_ARG
        if [ "{params.exclusive_lock}" = "true" ]; then
            LOCK_PREFIX="bash {input._helper} $PWD/.snakemake/.gpu_locks"
        else
            LOCK_PREFIX=""
        fi
        $LOCK_PREFIX python /app/alphafold/run_alphafold.py $FLASH_ARG --json_path={input.data} \
        --model_dir={params.models_dir} \
        --output_dir={params.output_dir}/rule_AF3_INFERENCE \
        --db_dir={params.database_dir} \
        --run_data_pipeline=false \
        --run_inference=true \
        {params.extra_af3_flags} 2>&1 | tee {log}
        """
