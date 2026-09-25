# Stage C: Title Injection Evaluation & Delta Report

### Stage C Metrics Summary (Title-Injected Corpus Representation)

| Track   | System                                 | R@2kt   | R@5kt   | R@100   | CFR    | Delta vs Ref               |   Holm p |
|:--------|:---------------------------------------|:--------|:--------|:--------|:-------|:---------------------------|---------:|
| bn      | S1 (Title Injected)                    | 55.13%  | 66.67%  | 84.94%  | 15.06% | -9.62 pts [-13.78, -5.45]  |    0     |
| bn      | S3 (Title Injected)                    | 63.14%  | 75.00%  | 85.90%  | 14.10% | -1.28 pts [-4.50, +1.92]   |    1     |
| bn      | Bilingual (Title Injected)             | 62.18%  | 74.68%  | 85.90%  | 14.10% | -1.60 pts [-5.13, +1.60]   |    1     |
| bn      | Union-200 (Title Injected)             | 63.14%  | 74.36%  | 88.14%  | 11.86% | -1.92 pts [-5.13, +1.28]   |    1     |
| bn      | R-S3 (q_mt, Title Injected)            | 66.35%  | 76.28%  | 85.90%  | 14.10% | +0.00 pts [+0.00, +0.00]   |    1     |
| bn      | R-S3 (q_src, Title Injected)           | 65.38%  | 75.32%  | 85.90%  | 14.10% | -0.96 pts [-3.85, +1.60]   |    1     |
| bn      | R-S3 (bilingual, Title Injected)       | 66.67%  | 76.92%  | 85.90%  | 14.10% | +0.64 pts [-1.60, +2.88]   |    1     |
| bn      | R-S1 (q_mt, Title Injected)            | 66.03%  | 75.32%  | 84.94%  | 15.06% | -0.96 pts [-2.56, +0.64]   |    1     |
| bn      | R-S1 (q_src, Title Injected)           | 64.42%  | 72.76%  | 84.94%  | 15.06% | -3.53 pts [-6.73, -0.64]   |    0.364 |
| bn      | R-S1 (bilingual, Title Injected)       | 67.31%  | 75.96%  | 84.94%  | 15.06% | -0.32 pts [-2.56, +1.92]   |    1     |
| bn      | R-Union200 (q_mt, Title Injected)      | 65.71%  | 75.64%  | 88.14%  | 11.86% | -0.64 pts [-1.60, +0.00]   |    1     |
| bn      | R-Union200 (q_src, Title Injected)     | 64.10%  | 73.72%  | 88.14%  | 11.86% | -2.56 pts [-5.45, +0.00]   |    0.836 |
| bn      | R-Union200 (bilingual, Title Injected) | 66.35%  | 76.28%  | 88.14%  | 11.86% | +0.00 pts [-2.24, +1.92]   |    1     |
| bn      | mE5-small S1 (Title Injected)          | 49.04%  | 62.82%  | 84.29%  | 15.71% | -13.46 pts [-18.28, -8.65] |    0     |
| bn      | mE5-small S3 (Title Injected)          | 56.41%  | 72.44%  | 86.86%  | 13.14% | -3.85 pts [-7.70, -0.32]   |    0.576 |
| te      | S1 (Title Injected)                    | 57.56%  | 69.75%  | 84.03%  | 15.97% | +0.00 pts [-4.62, +4.62]   |    1     |
| te      | S3 (Title Injected)                    | 60.08%  | 70.17%  | 82.77%  | 17.23% | +0.42 pts [-3.78, +4.62]   |    1     |
| te      | Bilingual (Title Injected)             | 59.24%  | 69.33%  | 82.77%  | 17.23% | -0.42 pts [-4.62, +3.78]   |    1     |
| te      | Union-200 (Title Injected)             | 60.50%  | 70.17%  | 85.29%  | 14.71% | +0.42 pts [-3.78, +4.62]   |    1     |
| te      | R-S3 (q_mt, Title Injected)            | 63.87%  | 70.17%  | 82.77%  | 17.23% | +0.42 pts [-2.52, +3.36]   |    1     |
| te      | R-S3 (q_src, Title Injected)           | 57.98%  | 69.75%  | 82.77%  | 17.23% | +0.00 pts [-1.68, +1.68]   |    1     |
| te      | R-S3 (bilingual, Title Injected)       | 62.61%  | 71.01%  | 82.77%  | 17.23% | +1.26 pts [-1.68, +4.20]   |    1     |
| te      | R-S1 (q_mt, Title Injected)            | 63.87%  | 72.27%  | 84.03%  | 15.97% | +2.52 pts [-0.43, +5.46]   |    1     |
| te      | R-S1 (q_src, Title Injected)           | 57.56%  | 69.75%  | 84.03%  | 15.97% | +0.00 pts [+0.00, +0.00]   |    1     |
| te      | R-S1 (bilingual, Title Injected)       | 63.03%  | 71.43%  | 84.03%  | 15.97% | +1.68 pts [-1.26, +4.62]   |    1     |
| te      | R-Union200 (q_mt, Title Injected)      | 63.45%  | 71.43%  | 85.29%  | 14.71% | +1.68 pts [-1.26, +4.62]   |    1     |
| te      | R-Union200 (q_src, Title Injected)     | 58.40%  | 69.75%  | 85.29%  | 14.71% | +0.00 pts [-1.68, +1.68]   |    1     |
| te      | R-Union200 (bilingual, Title Injected) | 62.18%  | 71.01%  | 85.29%  | 14.71% | +1.26 pts [-1.68, +4.20]   |    1     |
| te      | mE5-small S1 (Title Injected)          | 44.96%  | 59.24%  | 81.09%  | 18.91% | -10.50 pts [-15.97, -5.04] |    0     |
| te      | mE5-small S3 (Title Injected)          | 60.08%  | 67.23%  | 82.35%  | 17.65% | -2.52 pts [-7.14, +2.52]   |    1     |

### Step 2 Neighbor Propagation Decision Note
Neighbor context propagation (Idea 3 Step 2) is **skipped** per protocol because Stage B diagnosis demonstrated that `ENTITY_IN_NEIGHBOUR_ONLY` represents only **1.6% (bn)** and **1.7% (te)** of tail queries, far below the required 15.0% threshold.
