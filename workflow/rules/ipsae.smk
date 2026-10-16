rule IPSAE:
    input:
        model = os.path.join(
            OUTPUT_DIR, "rule_ALPHAFAST", "batch_{batch}", "{job_name}",
            "seed-{seed}_sample-{sample}",
            "{job_name}_seed-{seed}_sample-{sample}_model.cif"),
        confidences = os.path.join(
            OUTPUT_DIR, "rule_ALPHAFAST", "batch_{batch}", "{job_name}",
            "seed-{seed}_sample-{sample}",
            "{job_name}_seed-{seed}_sample-{sample}_confidences.json")
    output:
        ipsae_15_15 =os.path.join(
            OUTPUT_DIR, "rule_ALPHAFAST", "batch_{batch}", "{job_name}",
            "seed-{seed}_sample-{sample}",
            "{job_name}_seed-{seed}_sample-{sample}_model_15_15.txt"),
        ipsae_10_15 = os.path.join(
            OUTPUT_DIR, "rule_ALPHAFAST", "batch_{batch}", "{job_name}",
            "seed-{seed}_sample-{sample}",
            "{job_name}_seed-{seed}_sample-{sample}_model_10_15.txt")
    log:
        os.path.join(OUTPUT_DIR, "logs", "rule_IPSAE","batch_{batch}", "{job_name}_seed-{seed}_sample-{sample}.log") if MUTATION_DF.empty else
        os.path.join(OUTPUT_DIR, "logs", "rule_IPSAE", "{mut}_seed-{seed}.log"),
    benchmark:
        os.path.join(OUTPUT_DIR, "benchmarks", "rule_IPSAE","batch_{batch}", "{job_name}_seed-{seed}_sample-{sample}.tsv") if MUTATION_DF.empty else
        os.path.join(OUTPUT_DIR, "benchmarks", "rule_IPSAE", "{mut}_seed-{seed}.tsv"),
    conda: "../envs/structure_scoring.yaml"
    params:
        _helper = workflow.source_path("../scripts/ipsae.py") # added this to allow clean deployment
    shell:
        """
        python {params._helper} {input.confidences} {input.model} 10 15
        python {params._helper} {input.confidences} {input.model} 15 15
        """
