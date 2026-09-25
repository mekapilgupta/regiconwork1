# Synthesis Benchmark Table: Cross-Lingual IR on 21M DPR Corpus

> *Note: All LLM-augmented variants use leak-free prompted representations (<1% audited gold leakage). Full leak-free audit breakdown available in Appendix.*

---

## Bengali (`bn`, $n=312$) Benchmark Skeleton

| Family | System / Method | R@2kt | R@5kt | R@100 | CFR | $\Delta$ vs Ref [95% CI] | Holm $p$ | Tail Wins / Losses / Ties | Note |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Baselines** | S1: Zero-Shot Dense (BGE-M3) | — | — | — | — | — | — | — | Base Native |
| | S2: Sparse BM25 (Lucene / Anserini) | — | — | — | — | — | — | — | First-Stage Sparse |
| | S3: Machine Translation (MT Dense) | — | — | — | — | Reference (Dense) | — | — | Canonical Reference |
| | S8: Gold Translation Reference | — | — | — | — | — | — | — | Human Upper-Bound |
| **Structure & Controls** | S4: Query Decomposition / Subqueries | — | — | — | — | — | — | — | Sub-question fusion |
| | S5: Facet-Augmented Representation | — | — | — | — | — | — | — | Structured tags |
| | S5r: Distractor Control (Random words) | — | — | — | — | — | — | — | Length control for S5 |
| | S6: Entity-Constraint Multi-Vector | — | — | — | — | — | — | — | Triplet multi-vector |
| | S9: Arbitrary Span Multi-Vector | — | — | — | — | — | — | — | Structure control for S6 |
| **Sentence Bucket (M1)** | View 1: Literal Translation (`trans_literal`) | — | — | — | — | — | — | — | Single view ladder |
| | View 2: Natural Translation (`trans_natural`) | — | — | — | — | — | — | — | Single view ladder |
| | View 3: Alias Expansion (`trans_alias`) | — | — | — | — | — | — | — | Single view ladder |
| | View 4: Declarative Cloze (`cloze_declarative`) | — | — | — | — | — | — | — | Single view ladder |
| | View 5: Native Paraphrase (`src_paraphrase`) | — | — | — | — | — | — | — | Single view ladder |
| | View 6: Romanized Script (`src_romanized`) | — | — | — | — | — | — | — | Single view ladder |
| | M1: 6-View Sentence Bucket (CV $\beta$) | — | — | — | — | — | — | — | Out-of-fold LSE |
| | Oracle 6-View Ceiling | — | — | — | — | — | — | — | Per-query view ceiling |
| | M3: Passage-Shaped HyDE Cloze | — | — | — | — | — | — | — | Standalone cloze |
| | M1 + M3-HyDE (7-View Sentence Bucket) | — | — | — | — | — | — | — | LSE hybrid |
| | CONTROL: K=6 Random Distractor Questions | — | — | — | — | — | — | — | Bucket-size control |
| **Reranked (Cross-Encoder)** | R-S1 (bge-reranker-v2-m3) | — | — | — | — | — | — | — | Rerank S1 top-100 |
| | R-S3 (bge-reranker-v2-m3) | — | — | — | — | Reference (Rerank) | — | — | Rerank S3 top-100 |
| | R-UNION (S1 + S3 Union) | — | — | — | — | — | — | — | Rerank union top-100 |
| | R-M1 (CV $\beta$ Sentence Bucket) | — | — | — | — | — | — | — | Rerank M1 top-100 |
| | R-UNION_ALL ($M1 \cup S1 \cup S3$) | — | — | — | — | — | — | — | Rerank combined pool |
| **Routing Ladder** | Language-Only Router ($bn \to S3, te \to S1$) | — | — | — | — | — | — | — | Zero-param routing |
| | Learned Router (5-Fold CV Logistic Regression)| — | — | — | — | — | — | — | Out-of-fold router |
| | Oracle Routing $\{S1, S3\}$ | — | — | — | — | — | — | — | Theoretical headroom |
| **Encoder Ladder** | mE5-small S1 (Zero-Shot) | — | — | — | — | — | — | — | Small encoder baseline |
| | mE5-small S3 (MT Translation) | — | — | — | — | — | — | — | Small encoder translation |
| | mE5-small M1 (CV $\beta$ Sentence Bucket) | — | — | — | — | — | — | — | Small encoder multi-view |
| | mE5-small Oracle $\{S1, S3\}$ | — | — | — | — | — | — | — | Small encoder ceiling |

---

## Telugu (`te`, $n=238$) Benchmark Skeleton

| Family | System / Method | R@2kt | R@5kt | R@100 | CFR | $\Delta$ vs Ref [95% CI] | Holm $p$ | Tail Wins / Losses / Ties | Note |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Baselines** | S1: Zero-Shot Dense (BGE-M3) | — | — | — | — | Reference (Dense) | — | — | Base Native |
| | S2: Sparse BM25 (Lucene / Anserini) | — | — | — | — | — | — | — | First-Stage Sparse |
| | S3: Machine Translation (MT Dense) | — | — | — | — | — | — | — | Canonical Reference |
| | S8: Gold Translation Reference | — | — | — | — | — | — | — | Human Upper-Bound |
| **Structure & Controls** | S4: Query Decomposition / Subqueries | — | — | — | — | — | — | — | Sub-question fusion |
| | S5: Facet-Augmented Representation | — | — | — | — | — | — | — | Structured tags |
| | S5r: Distractor Control (Random words) | — | — | — | — | — | — | — | Length control for S5 |
| | S6: Entity-Constraint Multi-Vector | — | — | — | — | — | — | — | Triplet multi-vector |
| | S9: Arbitrary Span Multi-Vector | — | — | — | — | — | — | — | Structure control for S6 |
| **Sentence Bucket (M1)** | View 1: Literal Translation (`trans_literal`) | — | — | — | — | — | — | — | Single view ladder |
| | View 2: Natural Translation (`trans_natural`) | — | — | — | — | — | — | — | Single view ladder |
| | View 3: Alias Expansion (`trans_alias`) | — | — | — | — | — | — | — | Single view ladder |
| | View 4: Declarative Cloze (`cloze_declarative`) | — | — | — | — | — | — | — | Single view ladder |
| | View 5: Native Paraphrase (`src_paraphrase`) | — | — | — | — | — | — | — | Single view ladder |
| | View 6: Romanized Script (`src_romanized`) | — | — | — | — | — | — | — | Single view ladder |
| | M1: 6-View Sentence Bucket (CV $\beta$) | — | — | — | — | — | — | — | Out-of-fold LSE |
| | Oracle 6-View Ceiling | — | — | — | — | — | — | — | Per-query view ceiling |
| | M3: Passage-Shaped HyDE Cloze | — | — | — | — | — | — | — | Standalone cloze |
| | M1 + M3-HyDE (7-View Sentence Bucket) | — | — | — | — | — | — | — | LSE hybrid |
| | CONTROL: K=6 Random Distractor Questions | — | — | — | — | — | — | — | Bucket-size control |
| **Reranked (Cross-Encoder)** | R-S1 (bge-reranker-v2-m3) | — | — | — | — | Reference (Rerank) | — | — | Rerank S1 top-100 |
| | R-S3 (bge-reranker-v2-m3) | — | — | — | — | — | — | — | Rerank S3 top-100 |
| | R-UNION (S1 + S3 Union) | — | — | — | — | — | — | — | Rerank union top-100 |
| | R-M1 (CV $\beta$ Sentence Bucket) | — | — | — | — | — | — | — | Rerank M1 top-100 |
| | R-UNION_ALL ($M1 \cup S1 \cup S3$) | — | — | — | — | — | — | — | Rerank combined pool |
| **Routing Ladder** | Language-Only Router ($bn \to S3, te \to S1$) | — | — | — | — | — | — | — | Zero-param routing |
| | Learned Router (5-Fold CV Logistic Regression)| — | — | — | — | — | — | — | Out-of-fold router |
| | Oracle Routing $\{S1, S3\}$ | — | — | — | — | — | — | — | Theoretical headroom |
| **Encoder Ladder** | mE5-small S1 (Zero-Shot) | — | — | — | — | — | — | — | Small encoder baseline |
| | mE5-small S3 (MT Translation) | — | — | — | — | — | — | — | Small encoder translation |
| | mE5-small M1 (CV $\beta$ Sentence Bucket) | — | — | — | — | — | — | — | Small encoder multi-view |
| | mE5-small Oracle $\{S1, S3\}$ | — | — | — | — | — | — | — | Small encoder ceiling |
