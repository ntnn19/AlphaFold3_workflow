
import re
from pathlib import Path
from snakemake import __version__ as SNAKEMAKE_VERSION   # near the top of the Snakefile

print("cwd:", os.getcwd(), "| PWD:", os.environ.get("PWD"), "| init:", workflow.workdir_init)

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
        global_tsv   = os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS","all_af_global.tsv"),
        per_chain_tsv= os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS","all_af_per_chain.tsv"),
        per_pair_tsv = os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS","all_af_per_chain_pair.tsv"),
        ipsae        = os.path.join(OUTPUT_DIR, "rule_AGGREGATE_RESULTS","all_ipsae.tsv")
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


rule PROVENANCE_REPORT:
    output:
        provenance=report(
            os.path.join(OUTPUT_DIR, "rule_PROVENANCE", "provenance.txt"),
            category="Provenance",
            caption="../report/provenance.rst",
        ),
        launch_log=report(
            os.path.join(OUTPUT_DIR, "rule_PROVENANCE", "snakemake_cmd.txt"),
            category="Logs",
            caption="../report/log.rst",
        ),
    params:
        ref = deployed_ref(),
        log_dir = os.path.join(os.environ.get("PWD"),"log"),
        snakemake_version = SNAKEMAKE_VERSION,
    log:
        os.path.join(OUTPUT_DIR, "logs", "rule_PROVENANCE_REPORT", "provenance_report.log"),
    shell:
        """
        (
        set -euo pipefail
        set -x
        {{
            echo 'ntnn19/AlphaFold3_workflow @ {params.ref}'
            echo 'snakemake {params.snakemake_version}'
        }} > {output.provenance}
        latest=$(find "{params.log_dir}" -maxdepth 1 -name 'log_*.log' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2- || true)
        if [ -n "$latest" ]; then
            head -n 1 "$latest" > {output.launch_log}
        else
            echo "no Snakemake log found in {params.log_dir}" > {output.launch_log}
        fi
        ) > {log} 2>&1
        """


