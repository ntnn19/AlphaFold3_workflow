AlphaFold 3 Results
===================

This report summarises the confidence metrics produced by the AlphaFold 3
Snakemake workflow for all inference jobs in this project.  Each predicted
structure is characterised by a ranking score, ipTM, pTM, mean pLDDT,
fraction disordered, and clash flag at the global level, and by per-chain and
per-chain-pair PAE/ipTM scores at finer resolution.

Views
-----

Six interactive views are available in the sidebar:

- **global** — flat table of per-sample global metrics (one row per
  name + seed + sample).  Use this as the primary entry point.
- **global (grouped)** — same data grouped by job name; expand a group to
  compare seeds and samples side-by-side.
- **per-chain** — flat table of per-chain metrics (mean pLDDT, chain pTM,
  chain ipTM).
- **per-chain (grouped)** — per-chain metrics grouped by job name.
- **per-chain-pair** — flat table of chain-pair PAE min and ipTM; diagonal
  entries reflect intra-chain confidence.
- **per-chain-pair (grouped)** — chain-pair metrics grouped by job name.

Cross-view links on the **name** column let you jump from a global row
directly to the matching per-chain or per-chain-pair rows.

Configuration Summary
---------------------

::

   {{ snakemake.config | tojson(indent=2) | indent(3) }}
