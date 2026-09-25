#!/usr/bin/env python3
"""
Generate publication-quality figures for the ACL paper in latexwork/.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LATEX_DIR = ROOT / "latexwork"
LATEX_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
})

# ==========================================
# FIG 1: Granularity Ladder (R@5kt with 95% CIs)
# Systems: S4, S9, S6, S5, M1, Ref(S3 bn / S1 te), S8
# ==========================================
def generate_fig1():
    systems = [
        "S9: Arbitrary Spans",
        "S4: Concat Facets",
        "S6: Facet Multi-Vec",
        "S5: Query + Facets",
        "M1: 6-View Bucket",
        "Ref (Whole Query)",
        "S8: Human Oracle"
    ]

    # Bengali values & CI half-widths
    bn_r5 = [53.53, 53.85, 67.31, 72.44, 74.36, 75.32, 74.36]
    bn_err = [5.5, 5.2, 5.1, 4.8, 4.5, 4.8, 4.8]

    # Telugu values & CI half-widths
    te_r5 = [50.42, 56.30, 63.45, 66.81, 70.59, 70.17, 72.27]
    te_err = [6.1, 5.8, 5.9, 5.5, 5.0, 5.6, 5.4]

    x = np.arange(len(systems))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=300)
    rects1 = ax.bar(x - width/2, bn_r5, width, yerr=bn_err, capsize=3, label="Bengali (bn, Ref=S3 MT)", color="#2b5c8f", alpha=0.9)
    rects2 = ax.bar(x + width/2, te_r5, width, yerr=te_err, capsize=3, label="Telugu (te, Ref=S1 Native)", color="#d95f02", alpha=0.9)

    ax.set_ylabel("Recall @ 5,000 chars (R@5kt %)")
    ax.set_title("RQ1 Granularity Ladder: Query Representations vs. Whole Question")
    ax.set_xticks(x)
    ax.set_xticklabels(systems, rotation=22, ha="right")
    ax.set_ylim(40, 85)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    fig.savefig(LATEX_DIR / "fig1_granularity_ladder.pdf")
    fig.savefig(LATEX_DIR / "fig1_granularity_ladder.png")
    plt.close()
    print("Generated Fig 1: Granularity Ladder")


# ==========================================
# FIG 2: MT Gain vs Encoder Strength
# ==========================================
def generate_fig2():
    encoders = ["mE5-small\n(117M params)", "BGE-M3\n(567M params)"]
    bn_gains = [10.9, 7.69]  # S3 - S1
    te_gains = [6.3, -0.84]

    fig, ax = plt.subplots(figsize=(5.5, 3.8), dpi=300)
    ax.plot(encoders, bn_gains, marker="o", lw=2.2, color="#2b5c8f", label="Bengali (bn): MT beats Native")
    ax.plot(encoders, te_gains, marker="s", lw=2.2, color="#d95f02", label="Telugu (te): Native matches MT")
    ax.axhline(0, color="gray", linestyle="--", lw=1.2, alpha=0.8)

    for i, txt in enumerate(bn_gains):
        ax.annotate(f"{txt:+.1f} pts", (encoders[i], bn_gains[i] + 0.5), ha="center", fontweight="bold", color="#2b5c8f")
    for i, txt in enumerate(te_gains):
        ax.annotate(f"{txt:+.1f} pts", (encoders[i], te_gains[i] - 1.1), ha="center", fontweight="bold", color="#d95f02")

    ax.set_ylabel("Translation Advantage: $\\Delta$ (S3 MT $-$ S1 Native) pts")
    ax.set_title("RQ2: Translation Advantage Shrinks with Encoder Capacity")
    ax.set_ylim(-3, 14)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    fig.savefig(LATEX_DIR / "fig2_mt_gain_vs_encoder.pdf")
    fig.savefig(LATEX_DIR / "fig2_mt_gain_vs_encoder.png")
    plt.close()
    print("Generated Fig 2: MT Gain vs Encoder")


# ==========================================
# FIG 4: Tail Taxonomy Stacked Bar
# ==========================================
def generate_fig4():
    categories = [
        "ANSWER_STRING_ARTIFACT",
        "RERANKER_MISS",
        "EVIDENCE_ABSENT",
        "OTHER",
        "ENTITY_IN_TITLE_ONLY",
        "ENTITY_IN_NEIGHBOUR_ONLY"
    ]
    labels = [
        "Answer-String Artifact (75.6% / 77.3%)",
        "Reranker Rank Displacement (13.1% / 10.5%)",
        "Evidence Absent from Pool (6.4% / 10.9%)",
        "Other Syntax / Compositional (3.2% / 1.3%)",
        "Entity in Title Only (1.0% / 0.0%)",
        "Entity in Neighbour Only (0.6% / 0.0%)"
    ]
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]

    bn_pcts = [75.64, 13.14, 6.41, 3.21, 0.96, 0.64]
    te_pcts = [77.31, 10.50, 10.92, 1.26, 0.0, 0.0]

    fig, ax = plt.subplots(figsize=(8, 2.8), dpi=300)
    tracks = ["Telugu (n=238)", "Bengali (n=312)"]
    
    for cat_idx in range(len(categories)):
        vals = [te_pcts[cat_idx], bn_pcts[cat_idx]]
        lefts = [sum(te_pcts[:cat_idx]), sum(bn_pcts[:cat_idx])]
        ax.barh(tracks, vals, left=lefts, color=colors[cat_idx], label=labels[cat_idx], height=0.55, edgecolor="white")

    ax.set_xlabel("Percentage of Tail Failure Queries (%)")
    ax.set_title("RQ4 & RQ6: Mutually Exclusive Tail Failure Taxonomy")
    ax.set_xlim(0, 100)
    ax.legend(bbox_to_anchor=(0.5, -0.28), loc="upper center", ncol=2, frameon=True, fontsize=8)
    plt.tight_layout()
    fig.savefig(LATEX_DIR / "fig4_tail_taxonomy.pdf")
    fig.savefig(LATEX_DIR / "fig4_tail_taxonomy.png")
    plt.close()
    print("Generated Fig 4: Tail Taxonomy")


# ==========================================
# FIG 0: System Architecture / Experiment Map
# ==========================================
def generate_fig0():
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    ax.axis("off")

    from matplotlib.patches import FancyBboxPatch

    def draw_box(x, y, w, h, text, color="#f0f0f0", border="#333333", bold=False, text_color="black"):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02", facecolor=color, edgecolor=border, lw=1.5)
        ax.add_patch(box)
        weight = "bold" if bold else "normal"
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=8.5, weight=weight, color=text_color, multialignment="center")

    def draw_arrow(x1, y1, x2, y2, color="#555555"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.5, color=color))

    # Col 1: Source Query
    draw_box(0.02, 0.40, 0.12, 0.20, "Cross-Lingual\nQuery $q_{\\mathrm{src}}$\n(bn, te)", color="#e0ecf4", border="#8856a7", bold=True)

    # Col 2: Query Transformations
    draw_box(0.19, 0.70, 0.16, 0.18, "Machine Translation\n$T(q_{\\mathrm{src}})$\n[Helps bn +7.7, ties te]", color="#d9f0a3", border="#78c679")
    draw_box(0.19, 0.42, 0.16, 0.18, "Query Decomp / Facets\n$F = \\{e_i, r, c\\}$\n[Loses 8-15 pts]", color="#fee0d2", border="#de2d26")
    draw_box(0.19, 0.14, 0.16, 0.18, "6-View Sentences\n$M1$ Multi-View Bucket\n[Ties whole question]", color="#fff7bc", border="#fec44f")

    draw_arrow(0.14, 0.52, 0.19, 0.79)
    draw_arrow(0.14, 0.50, 0.19, 0.50)
    draw_arrow(0.14, 0.48, 0.19, 0.23)

    # Col 3: Similarity Channels
    draw_box(0.40, 0.70, 0.16, 0.18, "Dense Encoding\nBGE-M3 (S1, S3)\n[Base Ranker]", color="#d9f0a3", border="#78c679")
    draw_box(0.40, 0.42, 0.16, 0.18, "Lexical BM25 (S2)\n+ Sparse BGE-M3\n[Candidate Reach]", color="#d9f0a3", border="#78c679")
    draw_box(0.40, 0.14, 0.16, 0.18, "Late Interaction\nColBERT MaxSim\n[Channel Diversity]", color="#d9f0a3", border="#78c679")

    draw_arrow(0.35, 0.79, 0.40, 0.79)
    draw_arrow(0.35, 0.50, 0.40, 0.50)
    draw_arrow(0.35, 0.23, 0.40, 0.23)

    # Col 4: Channel Fusion
    draw_box(0.61, 0.38, 0.15, 0.24, "6-Channel RRF\nCandidate Fusion\n[$\\Delta$ CFR: -2.6 pts]\nRescues Documents", color="#78c679", border="#238b45", bold=True, text_color="white")

    draw_arrow(0.56, 0.79, 0.61, 0.55)
    draw_arrow(0.56, 0.50, 0.61, 0.50)
    draw_arrow(0.56, 0.23, 0.61, 0.45)

    # Col 5: Cross-Encoder Reranker
    draw_box(0.81, 0.38, 0.17, 0.24, "Cross-Encoder\nbge-reranker-v2-m3\n[Absorbs differences;\nMatches single MT]", color="#fee0d2", border="#de2d26", bold=True)

    draw_arrow(0.76, 0.50, 0.81, 0.50)

    plt.tight_layout()
    fig.savefig(LATEX_DIR / "fig0_system_map.pdf")
    fig.savefig(LATEX_DIR / "fig0_system_map.png")
    plt.close()
    print("Generated Fig 0: System Map")

if __name__ == "__main__":
    generate_fig1()
    generate_fig2()
    generate_fig4()
    generate_fig0()
