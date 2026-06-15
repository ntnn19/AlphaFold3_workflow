#SEED=re.search(r'seed-(\d+)', multi).group(1),
rule IPSAE:
    input:
        model = branch(
            lambda w: w.multi in SCORING_READY_DF["sample_id"].values,
            then=lookup(query="sample_id == '{multi}'", within=SCORING_READY_DF, cols="model"),
            otherwise=lambda w: os.path.join(
                OUTPUT_DIR, "rule_AF3_INFERENCE", f"{w.multi}", f"{w.multi}_model.cif"
            )
        ),
        confidences = branch(
            lambda w: w.multi in SCORING_READY_DF["sample_id"].values,
            then=lookup(query="sample_id == '{multi}'", within=SCORING_READY_DF, cols="confidence"),
            otherwise=lambda w: os.path.join(
                OUTPUT_DIR, "rule_AF3_INFERENCE", f"{w.multi}", f"{w.multi}_confidences.json"
            )
        )
    output:
        expand(os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{{multi}}", "seed-{seed}_sample-{sample}","{{multi}}_seed-{seed}_sample-{sample}_model_15_15.txt"),seed=re.search(r'seed-(\d+)', wildcards.multi).group(1),sample=range(5)) if AF3_VERSION not in [ "v3.0.1", "v3.0.0" ] else expand(os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{multi}", "seed-{seed}_sample-{sample}","model_15_15.txt"),seed=re.search(r'seed-(\d+)', wildcards.multi).group(1),sample=range(5)),
        expand(os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{{multi}}", "seed-{seed}_sample-{sample}","{{multi}}_seed-{seed}_sample-{sample}_model_10_15.txt"),seed=re.search(r'seed-(\d+)', wildcards.multi).group(1),sample=range(5)) if AF3_VERSION not in [ "v3.0.1", "v3.0.0" ] else expand(os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{multi}", "seed-{seed}_sample-{sample}","model_10_15.txt"),seed=re.search(r'seed-(\d+)', wildcards.multi).group(1),sample=range(5)),
    log:
        os.path.join(OUTPUT_DIR, "logs", "rule_IPSAE", "{multi}.log")
    benchmark:
        os.path.join(OUTPUT_DIR, "benchmarks", "rule_IPSAE", "{multi}.tsv")
    resources:
        mem_mb      = 1000,
        runtime     = 480,
    conda: "../envs/structure_scoring.yaml"
    params:
        inference_dir = os.path.join(OUTPUT_DIR, "rule_AF3_INFERENCE", "{multi}")
    shell:
        """
        for f in {params.inference_dir}/{wildcards.multi}/seed-*_sample-*/*confidences.json; do
            if [[ "$f" != *summary* ]]; then
                model="${{f/_confidences.json/_model.cif}}"
                python workflow/scripts/ipsae.py "$f" "$model" 10 15
                python workflow/scripts/ipsae.py "$f" "$model" 15 15
                break
            fi
        done
        """
