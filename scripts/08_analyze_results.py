#!/usr/bin/env python
"""Performs deconfounded, summary-level factuality and surface metric analysis.

Key methodological guarantees:
1. One row = one summary (strictly summary-level aggregation, avoiding inflated N).
2. Within-condition correlations reported for every (system, language) pair with 95% bootstrap CIs and FDR adjustment.
3. Ordinary least squares regression controlling for system, language, summary length, source length with full statistical reporting.
4. Spearman rank correlation as the primary metric for ordinal judge scores (Pearson as supplementary).
5. Separate tables for output-level validity and summary-level factual error taxonomy.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.io_utils import REPO_ROOT, get_logger, load_config, read_jsonl, resolve_path


def bootstrap_ci_spearman(x: np.ndarray, y: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42):
    """Computes empirical bootstrap confidence interval for Spearman rho using fast vectorized ranks."""
    rng = np.random.RandomState(seed)
    n = len(x)
    if n < 5:
        return (np.nan, np.nan)
    idx = rng.randint(0, n, size=(n_boot, n))
    bx = x[idx]
    by = y[idx]
    rx = np.argsort(np.argsort(bx, axis=1), axis=1).astype(float)
    ry = np.argsort(np.argsort(by, axis=1), axis=1).astype(float)
    rx_c = rx - rx.mean(axis=1, keepdims=True)
    ry_c = ry - ry.mean(axis=1, keepdims=True)
    num = (rx_c * ry_c).sum(axis=1)
    den = np.sqrt((rx_c ** 2).sum(axis=1) * (ry_c ** 2).sum(axis=1))
    valid = den > 0
    if not np.any(valid):
        return (np.nan, np.nan)
    rhos = num[valid] / den[valid]
    lower = np.percentile(rhos, 100 * (alpha / 2))
    upper = np.percentile(rhos, 100 * (1 - alpha / 2))
    return (float(lower), float(upper))


def benjamini_hochberg_fdr(p_values: list[float]) -> list[float]:
    """Calculates Benjamini-Hochberg FDR-adjusted p-values."""
    p_arr = np.asarray(p_values, dtype=float)
    n = len(p_arr)
    sorted_indices = np.argsort(p_arr)
    sorted_p = p_arr[sorted_indices]
    
    adjusted = np.zeros(n)
    cum_min = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        adj_val = min(1.0, sorted_p[i] * n / rank)
        cum_min = min(cum_min, adj_val)
        adjusted[i] = cum_min
        
    orig_adjusted = np.zeros(n)
    orig_adjusted[sorted_indices] = adjusted
    return orig_adjusted.tolist()


def run_ols_regression(df: pd.DataFrame):
    """Runs multivariable OLS regression with full coefficient, SE, t-stat, p-value, and 95% CI reporting."""
    # Dependent variable: factuality_label (0, 1, 2)
    # Predictors: rougeL, system, language, summary_length_words, source_length_words
    reg_df = df[["factuality_label", "rougeL", "summary_length_words", "source_length_words", "system", "language"]].dropna().copy()
    
    # Categorical dummies with explicit reference levels:
    # System reference: lead_baseline
    # Language reference: english
    system_dummies = pd.get_dummies(reg_df["system"], prefix="system", drop_first=False, dtype=float)
    if "system_lead_baseline" in system_dummies.columns:
        system_dummies = system_dummies.drop(columns=["system_lead_baseline"])
        
    lang_dummies = pd.get_dummies(reg_df["language"], prefix="lang", drop_first=False, dtype=float)
    if "lang_english" in lang_dummies.columns:
        lang_dummies = lang_dummies.drop(columns=["lang_english"])
        
    X_mat = pd.concat([
        pd.Series(1.0, index=reg_df.index, name="Intercept"),
        reg_df[["rougeL", "summary_length_words", "source_length_words"]],
        system_dummies,
        lang_dummies
    ], axis=1)
    
    y_vec = reg_df["factuality_label"].values
    X_vals = X_mat.values
    feature_names = list(X_mat.columns)
    
    n, k = X_vals.shape
    dof = n - k
    
    beta = np.linalg.pinv(X_vals.T @ X_vals) @ X_vals.T @ y_vec
    residuals = y_vec - X_vals @ beta
    ssr = float(residuals @ residuals)
    sst = float(np.sum((y_vec - np.mean(y_vec)) ** 2))
    r2 = 1.0 - (ssr / sst) if sst > 0 else 0.0
    adj_r2 = 1.0 - ((ssr / dof) / (sst / (n - 1))) if dof > 0 and sst > 0 else 0.0
    
    sigma2 = ssr / dof
    vcov = sigma2 * np.linalg.pinv(X_vals.T @ X_vals)
    se = np.sqrt(np.maximum(0, np.diag(vcov)))
    
    t_stats = beta / np.where(se > 0, se, 1e-12)
    p_values = [float(2 * (1 - stats.t.cdf(np.abs(t), df=dof))) for t in t_stats]
    t_crit = stats.t.ppf(0.975, df=dof)
    ci_lower = beta - t_crit * se
    ci_upper = beta + t_crit * se
    
    results = []
    for i, name in enumerate(feature_names):
        results.append({
            "term": name,
            "coef": round(float(beta[i]), 4),
            "std_error": round(float(se[i]), 4),
            "t_statistic": round(float(t_stats[i]), 4),
            "p_value": float(p_values[i]),
            "ci_95_lower": round(float(ci_lower[i]), 4),
            "ci_95_upper": round(float(ci_upper[i]), 4),
        })
        
    return pd.DataFrame(results), {
        "n_obs": n,
        "dof": dof,
        "r_squared": round(r2, 4),
        "adj_r_squared": round(adj_r2, 4),
        "residual_se": round(float(np.sqrt(sigma2)), 4),
        "reference_categories": "system=lead_baseline, language=english"
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("analyze_results", cfg)

    manifest_path = resolve_path(cfg, "outputs_dir", "evaluation_manifest.jsonl")
    if not manifest_path.exists():
        logger.error(f"Manifest not found at {manifest_path}. Run 10_build_evaluation_manifest.py first.")
        return

    records = list(read_jsonl(manifest_path))
    df = pd.DataFrame(records)
    logger.info(f"Loaded manifest with {len(df)} total evaluated summaries.")

    judged_df = df[df["factuality_label"].notna()].copy()
    logger.info(f"Found {len(judged_df)} LLM-judged summaries.")

    # 1. Output-Level Validity Breakdown Table (Full Sample N=4,000, 500 per condition)
    validity_rows = []
    status_order = ["valid", "malformed", "wrong_script", "truncated", "empty", "generation_error"]
    
    for (sys_name, lang_code), grp in df.groupby(["system", "language"]):
        total_n = len(grp)
        status_counts = grp["generation_status"].value_counts().to_dict()
        row = {
            "system": sys_name,
            "language": lang_code,
            "total_outputs": total_n,
        }
        for st in status_order:
            cnt = status_counts.get(st, 0)
            pct = (cnt / total_n) * 100.0 if total_n > 0 else 0.0
            row[f"{st}_count"] = cnt
            row[f"{st}_pct"] = round(pct, 1)
        validity_rows.append(row)

    validity_df = pd.DataFrame(validity_rows)
    validity_csv = resolve_path(cfg, "outputs_dir", "generation_validity_table.csv")
    validity_df.to_csv(validity_csv, index=False)
    logger.info(f"Saved generation validity table -> {validity_csv}")

    # 2. Error Taxonomy Table (Summary-level, Judged Sample N=314)
    # Output-level errors vs Summary-level factual errors
    error_taxonomy_rows = []
    factual_error_types = [
        "person_identity",
        "organization_identity",
        "location_entity",
        "event_relation",
        "numeric_fact",
        "temporal_fact",
        "unsupported_claim",
        "omission",
        "contradiction",
    ]
    
    for (sys_name, lang_code), grp in judged_df.groupby(["system", "language"]):
        n_judged = len(grp)
        # Count output level errors in judged set
        out_err = grp[grp["error_level"] == "output_level"]
        sum_err = grp[grp["error_level"] == "summary_level"]
        no_err = grp[grp["error_level"] == "none"]
        
        row = {
            "system": sys_name,
            "language": lang_code,
            "n_judged": n_judged,
            "fully_factual_count": len(no_err),
            "fully_factual_pct": round(len(no_err) / n_judged * 100, 1),
            "output_level_errors_count": len(out_err),
            "output_level_errors_pct": round(len(out_err) / n_judged * 100, 1),
            "summary_level_factual_errors_count": len(sum_err),
            "summary_level_factual_errors_pct": round(len(sum_err) / n_judged * 100, 1),
        }
        
        # Specific factual error breakdowns
        for fet in factual_error_types:
            cnt = sum(grp["error_types"].str.contains(fet, na=False))
            row[f"err_{fet}"] = cnt
            
        error_taxonomy_rows.append(row)
        
    error_tax_df = pd.DataFrame(error_taxonomy_rows)
    error_tax_csv = resolve_path(cfg, "outputs_dir", "error_taxonomy_table.csv")
    error_tax_df.to_csv(error_tax_csv, index=False)
    logger.info(f"Saved error taxonomy table -> {error_tax_csv}")

    # 3. Summary-Level Correlation Table (Within-Condition & Pooled)
    corr_rows = []
    surface_cols = ["rouge1", "rouge2", "rougeL", "chrf", "bertscore_f1"]
    
    # A. Within-Condition Correlations (Primary Analysis)
    for (sys_name, lang_code), grp in df.groupby(["system", "language"]):
        judged_grp = grp[grp["factuality_label"].notna()]
        for sc in surface_cols:
            if len(judged_grp) >= 8 and judged_grp[sc].std() > 0 and judged_grp["factuality_label"].std() > 0:
                x_vals = judged_grp[sc].values
                y_vals = judged_grp["factuality_label"].values
                sr, sp = stats.spearmanr(x_vals, y_vals)
                pr, pp = stats.pearsonr(x_vals, y_vals)
                ci_low, ci_high = bootstrap_ci_spearman(x_vals, y_vals)
                
                corr_rows.append({
                    "scope": "WITHIN-CONDITION",
                    "system": sys_name,
                    "language": lang_code,
                    "surface_metric": sc,
                    "fact_metric": "factuality_label",
                    "n_summaries": len(judged_grp),
                    "spearman_rho": round(float(sr), 4),
                    "spearman_p": float(sp),
                    "spearman_ci_95_lower": round(ci_low, 4) if not np.isnan(ci_low) else None,
                    "spearman_ci_95_upper": round(ci_high, 4) if not np.isnan(ci_high) else None,
                    "pearson_r": round(float(pr), 4),
                    "pearson_p": float(pp),
                })

    # B. Pooled Correlations (All Systems & Languages)
    for sc in surface_cols:
        paired = judged_df[[sc, "factuality_label"]].dropna()
        if len(paired) >= 10 and paired[sc].std() > 0 and paired["factuality_label"].std() > 0:
            x_vals = paired[sc].values
            y_vals = paired["factuality_label"].values
            sr, sp = stats.spearmanr(x_vals, y_vals)
            pr, pp = stats.pearsonr(x_vals, y_vals)
            ci_low, ci_high = bootstrap_ci_spearman(x_vals, y_vals)
            
            corr_rows.append({
                "scope": "POOLED (All Systems & Languages)",
                "system": "ALL",
                "language": "ALL",
                "surface_metric": sc,
                "fact_metric": "factuality_label",
                "n_summaries": len(paired),
                "spearman_rho": round(float(sr), 4),
                "spearman_p": float(sp),
                "spearman_ci_95_lower": round(ci_low, 4) if not np.isnan(ci_low) else None,
                "spearman_ci_95_upper": round(ci_high, 4) if not np.isnan(ci_high) else None,
                "pearson_r": round(float(pr), 4),
                "pearson_p": float(pp),
            })

    # Apply FDR correction across within-condition tests
    corr_df = pd.DataFrame(corr_rows)
    within_mask = corr_df["scope"] == "WITHIN-CONDITION"
    within_p_vals = corr_df.loc[within_mask, "spearman_p"].tolist()
    adj_p_vals = benjamini_hochberg_fdr(within_p_vals)
    corr_df.loc[within_mask, "spearman_p_fdr_adj"] = [round(p, 4) for p in adj_p_vals]
    corr_df.loc[~within_mask, "spearman_p_fdr_adj"] = corr_df.loc[~within_mask, "spearman_p"]

    corr_csv = resolve_path(cfg, "outputs_dir", "correlation_table.csv")
    corr_df.to_csv(corr_csv, index=False)
    logger.info(f"Saved comprehensive summary-level correlation table -> {corr_csv}")

    # 4. Multivariable Regression Analysis
    reg_table, reg_meta = run_ols_regression(judged_df)
    reg_csv = resolve_path(cfg, "outputs_dir", "regression_table.csv")
    reg_table.to_csv(reg_csv, index=False)
    
    reg_summary_path = resolve_path(cfg, "outputs_dir", "regression_results.txt")
    with open(reg_summary_path, "w", encoding="utf-8") as f:
        f.write("=== MULTIVARIABLE REGRESSION ANALYSIS (OLS, Dependent Variable: Factuality Label 0/1/2) ===\n")
        f.write(f"Sample Size (N): {reg_meta['n_obs']}\n")
        f.write(f"Degrees of Freedom: {reg_meta['dof']}\n")
        f.write(f"R-squared: {reg_meta['r_squared']:.4f}\n")
        f.write(f"Adjusted R-squared: {reg_meta['adj_r_squared']:.4f}\n")
        f.write(f"Residual Std Error: {reg_meta['residual_se']:.4f}\n")
        f.write(f"Reference Categories: {reg_meta['reference_categories']}\n\n")
        f.write(f"{'Predictor Term':<30s} | {'Coef':>8s} | {'Std Err':>8s} | {'t-stat':>8s} | {'p-value':>8s} | {'95% CI Lower':>12s} | {'95% CI Upper':>12s}\n")
        f.write("-" * 96 + "\n")
        for _, r in reg_table.iterrows():
            f.write(f"{r['term']:<30s} | {r['coef']:>8.4f} | {r['std_error']:>8.4f} | {r['t_statistic']:>8.4f} | {r['p_value']:>8.4f} | {r['ci_95_lower']:>12.4f} | {r['ci_95_upper']:>12.4f}\n")
        f.write("\nNote: Regression is observational and diagnostic. Do not interpret coefficients as causal effects.\n")

    logger.info(f"Saved multivariable regression results -> {reg_summary_path}")

    # 5. Scatter Plot: ROUGE-L vs. Factuality Label
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        system_colors = {"lead_baseline": "#2ca02c", "mt5_xlsum": "#d62728", "indicbart": "#1f77b4"}
        
        for sys_name, grp in judged_df.groupby("system"):
            jitter_y = grp["factuality_label"] + np.random.uniform(-0.06, 0.06, size=len(grp))
            ax.scatter(
                grp["rougeL"],
                jitter_y,
                c=system_colors.get(sys_name, "gray"),
                label=sys_name,
                alpha=0.65,
                s=40,
                edgecolors="none"
            )
        ax.set_xlabel("ROUGE-L Score (Surface Overlap with Reference)", fontsize=11)
        ax.set_ylabel("Factuality Label (0=Severe Error, 1=Partial, 2=Factual)", fontsize=11)
        ax.set_title("Surface Overlap (ROUGE-L) vs. Factual Correctness across Systems", fontsize=12, pad=12)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["0 (Severe Error)", "1 (Partial/Omission)", "2 (Factual)"])
        ax.legend(title="System", frameon=True)
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        plot_path = resolve_path(cfg, "outputs_dir", "rouge_vs_factuality.png")
        fig.savefig(plot_path, dpi=150)
        logger.info(f"Saved figure -> {plot_path}")
    except Exception as e:
        logger.warning(f"Plot generation warning: {e}")


if __name__ == "__main__":
    main()
