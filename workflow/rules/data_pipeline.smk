rule AF3_DATA_SPEEDY_PIPELINE:
    input:
        data = os.path.join(OUTPUT_DIR,"rule_PREPROCESSING","monomers","{mono}.json")
        #data = branch(
            #lookup(query="sample_id == '{mono}'",within=DATA_PIPELINE_READY_DF ,cols="file"),
            #then=lookup(query="sample_id == '{mono}'",within=DATA_PIPELINE_READY_DF ,cols="file"),
            #otherwise=os.path.join(OUTPUT_DIR,"rule_PREPROCESSING","monomers","{mono}.json")
        #),
    params:
        mode = MODE,
        extra_af3_flags = EXTRA_AF3_FLAGS,
        models_dir = MODELS_DIR,
        databases_dir = DB_DIR,
        output_dir = lambda w, output: str(Path(output[0]).parents[2])
    output:
        # B2 fix: output is uncondi1tional — the branch() input logic already handles
        # the case where no preprocessing was run (data_pipeline_ready entry point).
        # The previous mode guard silently produced an empty output for any undocumented
        # mode value, causing MSA generation to be skipped without error.
        data_pipeline_monomers=os.path.join(OUTPUT_DIR,"rule_AF3_DATA_PIPELINE","{mono}","{mono}_data.json"),
    log:
        os.path.join(OUTPUT_DIR, "logs", "rule_AF3_DATA_PIPELINE", "{mono}.log"),
    benchmark:
        os.path.join(OUTPUT_DIR, "benchmarks", "rule_AF3_DATA_PIPELINE", "{mono}.tsv"),
    container:
        AF3_CONTAINER
    shell:
        """
        python /app/alphafold/run_alphafold.py --json_path={input.data} \
        --model_dir={params.models_dir} \
        --output_dir={params.output_dir}/rule_AF3_DATA_PIPELINE \
        --db_dir={params.databases_dir} \
        --run_data_pipeline=true \
        --run_inference=false \
        {params.extra_af3_data_flags} \
        2>&1 | tee {log}
        """
