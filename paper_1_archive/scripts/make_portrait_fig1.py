import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set up compact portrait figure designed for single-column width (approx 3.3 in wide, 4.3 in tall)
fig, ax = plt.subplots(figsize=(3.3, 4.4), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 15)
ax.axis('off')

# Color palette - subtle clean tints
c_blue = '#EBF5FB'
c_green = '#E8F8F5'
c_yellow = '#FEF9E7'
c_purple = '#F4ECF7'
c_red = '#FDEDEC'

def draw_step(y, title, desc, tag, facecolor, edgecolor):
    # Box
    ax.add_patch(patches.FancyBboxPatch(
        (0.4, y), 9.2, 1.9, 
        boxstyle='round,pad=0.15', 
        facecolor=facecolor, 
        edgecolor=edgecolor, 
        linewidth=1.0
    ))
    # Step Tag
    ax.text(0.7, y + 1.55, tag, fontsize=6.2, fontweight='bold', color=edgecolor, va='center')
    # Title
    ax.text(0.7, y + 1.15, title, fontsize=7.8, fontweight='bold', color='#1A252F', va='center')
    # Desc
    ax.text(0.7, y + 0.50, desc, fontsize=6.5, color='#2C3E50', va='center', linespacing=1.2)

# Step 1: Query Formulation
draw_step(12.4, 'Query Formulation (RQ1, RQ2)',
          'Complete Question (Native / MT) vs. Facets (S6)\nControls: Random Words (S5r), 3-Spans (S9)',
          'STAGE 1: QUERY FORMULATION', c_blue, '#2980B9')
ax.annotate('', xy=(5.0, 11.85), xytext=(5.0, 12.4),
            arrowprops=dict(arrowstyle='->', lw=1.0, color='#2C3E50'))

# Step 2: Full-Corpus BM25 Pool Generation
draw_step(9.65, '21M English Wikipedia DPR Corpus',
          'Deployable BM25 Pool: 77,446 (bn), 66,714 (te)\nDeployable Ceilings: 91.03% (bn), 86.97% (te)',
          'STAGE 2: CANDIDATE GENERATION', c_green, '#27AE60')
ax.annotate('', xy=(5.0, 9.1), xytext=(5.0, 9.65),
            arrowprops=dict(arrowstyle='->', lw=1.0, color='#2C3E50'))

# Step 3: Heterogeneous Channel Scoring
draw_step(6.9, 'Heterogeneous Channels (RQ3)',
          'Dense BGE-M3 (Native / MT) + Lucene BM25 (MT)\n+ ColBERT MaxSim Late-Interaction (Native / MT)',
          'STAGE 3: FIRST-STAGE SCORING', c_yellow, '#D4AC0D')
ax.annotate('', xy=(5.0, 6.35), xytext=(5.0, 6.9),
            arrowprops=dict(arrowstyle='->', lw=1.0, color='#2C3E50'))

# Step 4: Channel Fusion
draw_step(4.15, '6-Channel Reciprocal Rank Fusion',
          'RRF (k=60) over Top-100 and Top-200 Pools\nCandidate Reach: 88.46% (bn), 85.29% (te)',
          'STAGE 4: CHANNEL FUSION', c_purple, '#8E44AD')
ax.annotate('', xy=(5.0, 3.6), xytext=(5.0, 4.15),
            arrowprops=dict(arrowstyle='->', lw=1.0, color='#2C3E50'))

# Step 5: Cross-Encoder Reranking
draw_step(1.4, 'Cross-Encoder Reranking',
          'bge-reranker-v2-m3 over Top-100 / Top-200\nCharacter Budgets: R@2kc (2k), R@5kc (5k)',
          'STAGE 5: FINAL PASSAGE RANKING', c_red, '#C0392B')

# Footer
ax.text(5.0, 0.45, 'Factorial evaluation over 550 XOR-TyDi queries',
        ha='center', va='center', fontsize=6.4, fontstyle='italic', color='#7F8C8D')

plt.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
plt.savefig('/home/kshitij22256/incon2026/latexwork/fig0_pipeline_portrait.pdf', bbox_inches='tight')
plt.savefig('/home/kshitij22256/incon2026/latexwork/fig0_pipeline_portrait.png', bbox_inches='tight', dpi=300)
print('Generated compact portrait figure successfully.')
