# Corpus Validation Gate Report

**Date:** 2026-09-04  
**Benchmark:** XOR-TyDi Bengali (`bn`) & Telugu (`te`) Cross-Lingual Information Retrieval  
**Evaluation Mode:** Shared Candidate-Pool Reranking over Authentic English Wikipedia Passages  
**Status:** **PASSED**

---

## 1. Index & Corpus Specifications
- **Corpus File:** `data/corpus/wiki_passages.jsonl`
- **Total Validated Passages:** 629 passages
- **Document ID Format:** Integer document IDs (`1` to `629`), strictly non-hashed and non-query prefixed.
- **Passage Length Distribution:**
  - **Mean Word Count:** 53.5 words
  - **Minimum Word Count:** 15 words
  - **Maximum Word Count:** 575 words
  - **Content Format:** Authentic multi-sentence Wikipedia prose (lead extracts, topic summaries, and section paragraphs).

---

## 2. Contamination & Leakage Audit
- **Exact Query-Copy Matches:** `0` (0.00% leakage against English development translations)
- **Short Synthetic Question Strings:** `0`
- **Query ID Collisions:** `0` (no `doc_-...` or `doc_10...` IDs present)

---

## 3. Retrieval Smoke Test & Sanity Checks
BM25 Search sanity test performed over the candidate pool:

| Query | Rank 1 Retrieved Document | Score | Expected Factual Match |
| :--- | :--- | :--- | :--- |
| `first university established in Turkey` | **[12] Ankara University** (*"Ankara University is a public research university in Ankara, the capital of Turkey. It was the first higher education institution founded in Turkey..."*) | 20.385 | **PASS** (Contains gold answer `Ankara University`) |
| Competitor 1 | **[487] Sikkim Manipal University** | 17.190 | Authentic distractor |
| Competitor 2 | **[583] Richard McSorley** | 8.314 | Authentic distractor |

---

## 4. Acceptance Gate Decision
- [x] All sampled documents are real Wikipedia prose (tens to hundreds of words).
- [x] Document IDs are integer IDs, not query IDs.
- [x] Document texts do not equal query translations or glosses.
- [x] Query-copy rate == 0.00%.
- [x] BM25 returns relevant evidence on smoke tests.

**Gate Decision:** **APPROVED TO PROCEED TO SYSTEM EXECUTION (S1–S8).**
