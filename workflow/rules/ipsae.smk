#SEED=re.search(r'seed-(\d+)', multi).group(1),
rule IPSAE:
    input:
        model = branch(
            lambda w: (w.multi if MUTATION_DF.empty else w.mut) in SCORING_READY_DF["sample_id"].values,
            then=lookup(
                query="sample_id == '{multi}'" if MUTATION_DF.empty else "sample_id == '{mut}'",
                within=SCORING_READY_DF, cols="model"
            ),
            otherwise=lambda w: expand(
                os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{key}",
                             "seed-{seed}_sample-{sample}",
                             "{key}_seed-{seed}_sample-{sample}_model.cif"),
                key=w.multi if MUTATION_DF.empty else w.mut,
                seed=re.search(r'seed-(\d+)', w.multi if MUTATION_DF.empty else w.mut).group(1),
                sample=range(N_SAMPLES)
            ) if AF3_VERSION not in ["v3.0.1", "v3.0.0"] else expand(
                os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{key}",
                             "seed-{seed}_sample-{sample}", "model.cif"),
                key=w.multi if MUTATION_DF.empty else w.mut,
                seed=re.search(r'seed-(\d+)', w.multi if MUTATION_DF.empty else w.mut).group(1),
                sample=range(N_SAMPLES)
            )
        ),
        confidences = branch(
            lambda w: (w.multi if MUTATION_DF.empty else w.mut) in SCORING_READY_DF["sample_id"].values,
            then=lookup(
                query="sample_id == '{multi}'" if MUTATION_DF.empty else "sample_id == '{mut}'",
                within=SCORING_READY_DF, cols="confidence"
            ),
            otherwise=lambda w: expand(
                os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{key}",
                             "seed-{seed}_sample-{sample}",
                             "{key}_seed-{seed}_sample-{sample}_confidences.json"),
                key=w.multi if MUTATION_DF.empty else w.mut,
                seed=re.search(r'seed-(\d+)', w.multi if MUTATION_DF.empty else w.mut).group(1),
                sample=range(N_SAMPLES)
            ) if AF3_VERSION not in ["v3.0.1", "v3.0.0"] else expand(
                os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{key}",
                             "seed-{seed}_sample-{sample}", "confidences.json"),
                key=w.multi if MUTATION_DF.empty else w.mut,
                seed=re.search(r'seed-(\d+)', w.multi if MUTATION_DF.empty else w.mut).group(1),
                sample=range(N_SAMPLES)
            )
        ),
    output:
        ipsae_15_15 = expand(
            os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{{multi}}",
                         "seed-{{seed}}_sample-{sample}",
                         "{{multi}}_seed-{{seed}}_sample-{sample}_model_15_15.txt" if AF3_VERSION not in ["v3.0.0", "v3.0.1"] else "seed-{{seed}}_sample-{sample}_model_15_15.txt"),
            sample=range(N_SAMPLES)
        ) if MUTATION_DF.empty else
        expand(
            os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{{mut}}",
                         "seed-{{seed}}_sample-{sample}",
                         "{{mut}}_seed-{{seed}}_sample-{sample}_model_15_15.txt" if AF3_VERSION not in ["v3.0.0", "v3.0.1"] else "seed-{{seed}}_sample-{sample}_model_15_15.txt"),
            sample=range(N_SAMPLES)
        ),
        ipsae_10_15 = expand(
            os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{{multi}}",
                         "seed-{{seed}}_sample-{sample}",
                         "{{multi}}_seed-{{seed}}_sample-{sample}_model_10_15.txt" if AF3_VERSION not in ["v3.0.0", "v3.0.1"] else "seed-{{seed}}_sample-{sample}_model_10_15.txt"),
            sample=range(N_SAMPLES)
        ) if MUTATION_DF.empty else
        expand(
            os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{{mut}}",
                         "seed-{{seed}}_sample-{sample}",
                         "{{mut}}_seed-{{seed}}_sample-{sample}_model_10_15.txt" if AF3_VERSION not in ["v3.0.0", "v3.0.1"] else "seed-{{seed}}_sample-{sample}_model_10_15.txt"),
            sample=range(N_SAMPLES)
        ),
    log:
        os.path.join(OUTPUT_DIR, "logs", "rule_IPSAE", "{multi}_seed-{seed}.log") if MUTATION_DF.empty else
        os.path.join(OUTPUT_DIR, "logs", "rule_IPSAE", "{mut}_seed-{seed}.log"),
    benchmark:
        os.path.join(OUTPUT_DIR, "benchmarks", "rule_IPSAE", "{multi}_seed-{seed}.tsv") if MUTATION_DF.empty else
        os.path.join(OUTPUT_DIR, "benchmarks", "rule_IPSAE", "{mut}_seed-{seed}.tsv"),
    resources:
        mem_mb  = 1000,
        runtime = 480,
    conda: "../envs/structure_scoring.yaml"
    params:
        inference_dir = os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{multi}") if MUTATION_DF.empty else
                        os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{mut}"),
        script = workflow.source_path("../scripts/ipsae.py"),
    shell:
        """
        for f in {params.inference_dir}/seed-*_sample-*/*confidences.json; do
            if [[ "$f" != *summary* ]]; then
                model="${{f/_confidences.json/_model.cif}}"
                python {params.script} "$f" "$model" 10 15
                python {params.script} "$f" "$model" 15 15
            fi
        done
        """
