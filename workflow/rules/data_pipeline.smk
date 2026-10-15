checkpoint AF3_DATA_SPEEDY_PIPELINE:
    input:
        data = get_monomeric_json_outputs
    params:
        databases_dir = DB_DIR,
        extra_af3_data_flags = EXTRA_AF3_DATA_FLAGS,
        output_dir = lambda w, output: str(Path(output[0]).parents[2])
    output:
        data_pipeline_monomers=directory(os.path.join(OUTPUT_DIR,"rule_AF3_DATA_PIPELINE"))
    log:
        os.path.join(OUTPUT_DIR, "logs", "rule_AF3_DATA_PIPELINE", "rule_AF3_DATA_PIPELINE.log"),
    benchmark:
        os.path.join(OUTPUT_DIR, "benchmarks", "rule_AF3_DATA_PIPELINE", "rule_AF3_DATA_PIPELINE.tsv"),
    container:
        ALPHAFAST_CONTAINER
    shell:
        """
        python /app/alphafold/run_alphafold.py --json_path={input.data} \
        --output_dir={params.output_dir}/rule_AF3_DATA_PIPELINE \
        --db_dir={params.databases_dir} \
        --run_data_pipeline=true \
        --run_inference=false \
        {params.extra_af3_data_flags} \
        2>&1 | tee {log}
        """
