# Third-party baseline boundary

COVE does not redistribute DASP, PackSELL, Spaden, CSR5, cuSPARSE, or SuiteSparse
matrix payloads. The public repository contains wrappers, a CSR5 compatibility
patch, and measured CSVs. Users must obtain each implementation under its own
license and must not apply the COVE MIT license to third-party code.

## Reference sources

| Baseline | Reference source | License / release boundary |
|---|---|---|
| CSR5 | `https://github.com/weifengliu-ssslab/Benchmark_SpMV_using_CSR5` at `4b06ce35e45c3c61a51e249e12e1c4954af443fa` | MIT; see `csr5/PORT.md` |
| DASP | `https://github.com/SuperScientificSoftwareLaboratory/DASP` at `8ce8363599464c92b141366dd2d87014fc266bea` | AGPL-3.0; not redistributed |
| Spaden | `https://github.com/yuang-chen/Spaden-ICPP24` at `af4f327a755229fd0a23a96581b5a7a029cfc515` | no repository license detected at release time; verify terms before use |
| PackSELL | arXiv `2604.13433` | no public source revision was identified for this release |

The DASP and Spaden revisions above are public reference heads inspected on
2026-08-31. They are not asserted to be byte-identical to the historical local
trees used for the June 2026 measurements. PackSELL's frozen CSV is evidence
from the completed campaign, but the source snapshot is not reproducibly pinned;
therefore the repository does not claim cold reproduction of that baseline.

## Expected local layout

`run_external_baseline_campaign.py` accepts explicit roots and defaults to:

```text
repro/dasp/src/DASP
repro/packsell/src/PackSELL/spmv
repro/spaden/src/Spaden-ICPP24-main
```

These `repro/*/src/` trees are intentionally ignored. Review the upstream
license before cloning or building each baseline.
