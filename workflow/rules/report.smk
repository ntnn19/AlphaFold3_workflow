rule DATAVZRD_REPORT:
    """Render an interactive HTML report from the three project-level TSVs.

    Produces a self-contained HTML report with three linked views:
      - global        per-sample ranking/confidence metrics
      - per-chain     per-chain pTM, ipTM, mean pLDDT
      - per-chain-pair  PAE min and ipTM between every chain pair

    The datavzrd config is a yte template so table paths are injected at
    render time from snakemake.input rather than being hardcoded.
    """
    input:
        config       = workflow.source_path("../resources/datavzrd.yaml"),
        global_tsv   = os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS", "all_af_global.tsv"),
        per_chain_tsv= os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS", "all_af_per_chain.tsv"),
        per_pair_tsv = os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS", "all_af_per_chain_pair.tsv"),
        ipsae        = os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS", "all_ipsae.tsv")
    output:
        report(
            directory(os.path.join(OUTPUT_DIR, "rule_REPORT")),
            htmlindex="index.html",
            caption="../report/workflow.rst",   # optional but recommended
            category="Results",
        )
    params:
        extra = "",
    log:
        os.path.join(OUTPUT_DIR, "logs", "rule_REPORT", "datavzrd.log"),
    wrapper:
        "v3.13.4/utils/datavzrd"