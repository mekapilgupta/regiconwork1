# Empirical Factuality Analysis in Multilingual News Summarization

## 1. LLM Factuality Verdict Distribution (% by System & Language)
0 = Contradiction / Hallucination | 1 = Partial / Minor | 2 = Fully Factual

|                              |   0.0 |   1.0 |   2.0 |
|:-----------------------------|------:|------:|------:|
| ('indicbart', 'bengali')     | 90    | 10    |  0    |
| ('indicbart', 'hindi')       |  7.69 | 82.05 | 10.26 |
| ('lead_baseline', 'bengali') |  0    | 10    | 90    |
| ('lead_baseline', 'english') |  0    | 10    | 90    |
| ('lead_baseline', 'hindi')   |  0    | 20    | 80    |
| ('mt5_xlsum', 'bengali')     | 38.46 | 15.38 | 46.15 |
| ('mt5_xlsum', 'english')     | 37.84 | 32.43 | 29.73 |
| ('mt5_xlsum', 'hindi')       | 56.41 |  2.56 | 41.03 |

## 2. Key Metric Correlations (Surface vs. Factuality)

| condition    | surface_metric   | fact_metric                |    n |   pearson_r |     pearson_p |   spearman_r |    spearman_p |
|:-------------|:-----------------|:---------------------------|-----:|------------:|--------------:|-------------:|--------------:|
| POOLED (All) | rouge1           | person_name_precision      |  475 |     -0.5533 |   1.89298e-39 |      -0.4435 |   2.58594e-24 |
| POOLED (All) | rouge1           | person_name_recall         |  976 |     -0.3732 |   1.27091e-33 |      -0.448  |   2.4254e-49  |
| POOLED (All) | rouge1           | org_name_precision         |  510 |     -0.4351 |   5.64879e-25 |      -0.4148 |   1.25927e-22 |
| POOLED (All) | rouge1           | org_name_recall            |  968 |     -0.3315 |   2.92173e-26 |      -0.4136 |   2.74832e-41 |
| POOLED (All) | rouge1           | numeric_quantity_precision | 1003 |     -0.14   |   8.5436e-06  |      -0.1403 |   8.21001e-06 |
| POOLED (All) | rouge1           | numeric_quantity_recall    | 3452 |     -0.0713 |   2.78849e-05 |      -0.1017 |   2.13261e-09 |
| POOLED (All) | rouge1           | date_precision             |   44 |    nan      | nan           |     nan      | nan           |
| POOLED (All) | rouge1           | date_recall                |  267 |     -0.3003 |   5.73849e-07 |      -0.3339 |   2.25228e-08 |
| POOLED (All) | rouge1           | llm_verdict                |  314 |      0.1137 |   0.0441548   |       0.0581 |   0.304659    |
| POOLED (All) | rouge2           | person_name_precision      |  475 |     -0.5297 |   1.02295e-35 |      -0.4237 |   4.07153e-22 |
| POOLED (All) | rouge2           | person_name_recall         |  976 |     -0.278  |   8.95855e-19 |      -0.3494 |   2.08487e-29 |
| POOLED (All) | rouge2           | org_name_precision         |  510 |     -0.3839 |   2.33031e-19 |      -0.3923 |   3.31794e-20 |
| POOLED (All) | rouge2           | org_name_recall            |  968 |     -0.2526 |   1.47109e-15 |      -0.3325 |   2.04471e-26 |
| POOLED (All) | rouge2           | numeric_quantity_precision | 1003 |     -0.1141 |   0.000291933 |      -0.0978 |   0.00192915  |
| POOLED (All) | rouge2           | numeric_quantity_recall    | 3452 |     -0.0573 |   0.000759196 |      -0.0425 |   0.0125284   |
| POOLED (All) | rouge2           | date_precision             |   44 |    nan      | nan           |     nan      | nan           |
| POOLED (All) | rouge2           | date_recall                |  267 |     -0.1963 |   0.0012613   |      -0.1843 |   0.00250448  |
| POOLED (All) | rouge2           | llm_verdict                |  314 |      0.0522 |   0.356219    |       0.0664 |   0.240931    |
| POOLED (All) | rougeL           | person_name_precision      |  475 |     -0.5567 |   5.26631e-40 |      -0.4739 |   5.70961e-28 |
| POOLED (All) | rougeL           | person_name_recall         |  976 |     -0.3694 |   6.48113e-33 |      -0.4925 |   9.12052e-61 |
| POOLED (All) | rougeL           | org_name_precision         |  510 |     -0.455  |   1.98912e-27 |      -0.4425 |   7.27616e-26 |
| POOLED (All) | rougeL           | org_name_recall            |  968 |     -0.3375 |   3.25911e-27 |      -0.4395 |   5.427e-47   |
| POOLED (All) | rougeL           | numeric_quantity_precision | 1003 |     -0.2442 |   4.38405e-15 |      -0.1939 |   5.89423e-10 |
| POOLED (All) | rougeL           | numeric_quantity_recall    | 3452 |     -0.1008 |   2.95173e-09 |      -0.1431 |   2.89577e-17 |
| POOLED (All) | rougeL           | date_precision             |   44 |    nan      | nan           |     nan      | nan           |
