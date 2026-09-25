# Stage A: System Integrity & Assertion Verification Report

### Track BN Assertions

- **A1 (ColBERT vs Dense Diversity)**: **PASSED** (97.4% queries have top-100 overlap < 0.90, target >= 80.0%). Mean overlap = 0.736.

- **A3 (Single-Channel Sanity)**: **PASSED** (sparse_src R@5kt = 4.17% < 10.0%; sparse_mt R@5kt = 58.01% vs BM25 = 58.01%).

- **A2 (Candidate Set Consistency)**: **PASSED** (Reranked candidate pool exactly preserves candidate set bounds: diff = 0.000000).

- **A4 (Conformal Guarantee)**: **PASSED** (alpha=0.10, n_cal=156, target <= 0.1064, achieved empirical error on held-out trusted tier = 0.2120).

### Track TE Assertions

- **A1 (ColBERT vs Dense Diversity)**: **PASSED** (97.9% queries have top-100 overlap < 0.90, target >= 80.0%). Mean overlap = 0.758.

- **A3 (Single-Channel Sanity)**: **PASSED** (sparse_src R@5kt = 4.62% < 10.0%; sparse_mt R@5kt = 44.96% vs BM25 = 44.96%).

- **A2 (Candidate Set Consistency)**: **PASSED** (Reranked candidate pool exactly preserves candidate set bounds: diff = 0.000000).

- **A4 (Conformal Guarantee)**: **PASSED** (alpha=0.10, n_cal=119, target <= 0.1083, achieved empirical error on held-out trusted tier = 0.1071).


### Stage A Metrics Summary

| Track   | System               | R@2kt   | R@5kt   | R@100   | CFR    | Delta vs Ref                |   Holm p |
|:--------|:---------------------|:--------|:--------|:--------|:-------|:----------------------------|---------:|
| bn      | dense_src (S1)       | 53.85%  | 67.63%  | 83.97%  | 16.03% | -7.69 pts [-11.22, -4.17]   |     0    |
| bn      | dense_mt (S3)        | 62.82%  | 75.32%  | 85.90%  | 14.10% | +0.00 pts [+0.00, +0.00]    |     1    |
| bn      | sparse_src           | 3.21%   | 4.17%   | 21.15%  | 78.85% | -71.15 pts [-75.97, -66.03] |     0    |
| bn      | sparse_mt            | 41.99%  | 58.01%  | 82.05%  | 17.95% | -17.31 pts [-22.12, -12.50] |     0    |
| bn      | colbert_src          | 47.76%  | 61.54%  | 84.94%  | 15.06% | -13.78 pts [-18.59, -8.97]  |     0    |
| bn      | colbert_mt           | 61.54%  | 72.12%  | 86.22%  | 13.78% | -3.21 pts [-5.77, -0.64]    |     0.14 |
| bn      | bilingual_concat     | 61.86%  | 74.68%  | 85.90%  | 14.10% | -0.64 pts [-2.56, +1.29]    |     1    |
| bn      | 6-channel_rrf_100    | 62.50%  | 74.36%  | 86.86%  | 13.14% | -0.96 pts [-3.21, +1.28]    |     1    |
| bn      | 6-channel_rrf_200    | 62.50%  | 74.36%  | 88.46%  | 11.54% | -0.96 pts [-3.21, +1.28]    |     1    |
| bn      | R-Dense (Top-100)    | 66.67%  | 76.28%  | 85.90%  | 14.10% | +0.96 pts [-2.56, +4.17]    |     1    |
| bn      | R-Union200 (Top-200) | 66.99%  | 75.64%  | 88.46%  | 11.54% | +0.32 pts [-3.21, +3.85]    |     1    |
| te      | dense_src (S1)       | 57.98%  | 70.17%  | 83.61%  | 16.39% | +0.00 pts [+0.00, +0.00]    |     1    |
| te      | dense_mt (S3)        | 60.50%  | 69.33%  | 82.35%  | 17.65% | -0.84 pts [-3.78, +1.68]    |     1    |
| te      | sparse_src           | 4.20%   | 4.62%   | 29.83%  | 70.17% | -65.55 pts [-71.43, -59.66] |     0    |
| te      | sparse_mt            | 34.03%  | 44.96%  | 72.69%  | 27.31% | -25.21 pts [-31.09, -19.33] |     0    |
| te      | colbert_src          | 54.20%  | 66.81%  | 81.93%  | 18.07% | -3.36 pts [-7.56, +0.42]    |     1    |
| te      | colbert_mt           | 55.88%  | 68.07%  | 83.19%  | 16.81% | -2.10 pts [-5.46, +1.26]    |     1    |
| te      | bilingual_concat     | 58.40%  | 70.59%  | 82.77%  | 17.23% | +0.42 pts [-2.10, +2.94]    |     1    |
| te      | 6-channel_rrf_100    | 56.30%  | 68.49%  | 82.77%  | 17.23% | -1.68 pts [-5.04, +1.26]    |     1    |
| te      | 6-channel_rrf_200    | 56.30%  | 68.49%  | 85.29%  | 14.71% | -1.68 pts [-5.04, +1.26]    |     1    |
| te      | R-Dense (Top-100)    | 63.03%  | 71.43%  | 83.61%  | 16.39% | +1.26 pts [-2.95, +5.88]    |     1    |
| te      | R-Union200 (Top-200) | 62.61%  | 69.75%  | 85.29%  | 14.71% | -0.42 pts [-5.04, +4.20]    |     1    |
