rule AF3_INFERENCE:
    input:
        data = branch(
            lookup(query="sample_id == '{mut}'" if MUTATION_DF_PATH is not None else "sample_id == '{multi}'", within=INFERENCE_READY_DF, cols="file"),
            then=lookup(query="sample_id == '{mut}'" if MUTATION_DF_PATH is not None else "sample_id == '{multi}'", within=INFERENCE_READY_DF, cols="file"),
            otherwise=lambda w: (
                os.path.join(
                    OUTPUT_DIR, "rule_MUTATE",
                    f"{w.mut}.json"
                )
                if MUTATION_DF_PATH is not None
                else os.path.join(
                    OUTPUT_DIR, "rule_MERGE_MONOMERS_TO_MULTIMERS", f"{w.multi}_data.json"
                )
            )
        )
    # NOTE: output/log/benchmark wildcard selection is gated on
    # `MUTATION_DF_PATH is not None` — the SAME condition used above for
    # `input.data` (and by the MUTATE checkpoint's own output declaration and by
    # _collect_inference_targets' rule_MUTATE glob). Gating these on
    # `MUTATION_DF.empty` instead (as the original code did) diverges from the
    # input gate in exactly one edge case — a mutations sheet that is present but
    # has zero data rows (e.g. header-only): there MUTATION_DF_PATH is not None
    # (input resolves via {mut} -> rule_MUTATE/) while MUTATION_DF.empty is True
    # (output would use {multi}), so the rule's wildcards would be {multi} while
    # input.data references w.mut -> InputFunctionException ('Wildcards' object has
    # no attribute 'mut'). Aligning every gate to MUTATION_DF_PATH keeps input and
    # output on the same wildcard and makes the mutation path consistent for all
    # streams (with mutate.py always writing a WT passthrough, rule_MUTATE/ always
    # has a resolvable file per stem when the path is set).
    output:
        model=expand(
            os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{{multi}}",
                            "seed-{{seed}}_sample-{sample}",
                            "{{multi}}_seed-{{seed}}_sample-{sample}_model.cif" if AF3_VERSION not in ["v3.0.0", "v3.0.1"] else "model.cif"),
            sample=range(N_SAMPLES)
        ) if MUTATION_DF_PATH is None else
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
        ) if MUTATION_DF_PATH is None else
        expand(
            os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{{mut}}",
                            "seed-{{seed}}_sample-{sample}",
                            "{{mut}}_seed-{{seed}}_sample-{sample}_confidences.json" if AF3_VERSION not in ["v3.0.0", "v3.0.1"] else "confidences.json"),
            sample=range(N_SAMPLES)
        ),
    log:
        os.path.join(OUTPUT_DIR, "logs", "rule_AF3_INFERENCE", "{multi}_seed-{seed}.log") if MUTATION_DF_PATH is None else os.path.join(OUTPUT_DIR, "logs", "rule_AF3_INFERENCE", "{mut}_seed-{seed}.log"),
    benchmark:
        os.path.join(OUTPUT_DIR, "benchmarks", "rule_AF3_INFERENCE", "{multi}_seed-{seed}.tsv") if MUTATION_DF_PATH is None else os.path.join(OUTPUT_DIR, "benchmarks", "rule_AF3_INFERENCE", "{mut}_seed-{seed}.tsv"),
    resources:
        cpus_per_task=1,
        mem_mb=1000,
        runtime="30m",
        slurm_partition="vds"
    params:
        extra_af3_flags = EXTRA_AF3_FLAGS,
        exclusive_lock = "true" if EXCLUSIVE_LOCK else "false",
        models_dir = MODELS_DIR,
        output_dir = OUTPUT_DIR,
        database_dir = DB_DIR,
        _helper = f"{WORKFLOW_DIR}/scripts/gpu_lock.sh"
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
        if [ "{params.exclusive_lock}" = "true" ]; then
            LOCK_PREFIX="bash {params._helper} $PWD/.snakemake/.gpu_locks/${{SLURM_JOB_ID:-standalone}}"
        else
            LOCK_PREFIX=""
        fi
        tmp=$(mktemp -d "${{PWD}}/.af3_tmp.XXXXXX")
        $LOCK_PREFIX python /app/alphafold/run_alphafold.py $FLASH_ARG --json_path={input.data} \
        --model_dir={params.models_dir} \
        --output_dir=$tmp \
        --db_dir={params.database_dir} \
        --run_data_pipeline=false \
        --run_inference=true \
        {params.extra_af3_flags} 2>&1 | tee {log}        
        cp -a "$tmp"/. "{params.output_dir}/rule_AF3_INFERENCE/"
        rm -rf "$tmp"
        """
