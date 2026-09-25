"""make_figs.py -- the paper's two data figures, straight from the recorded numbers.

Every value below is copied from EXPERIMENT_STATE.md / MANUSCRIPT.md; nothing is
interpolated, smoothed, or invented. Rerun after any number changes.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent

# ---- Fig. 2: ranking transfers, levels do not (three independent measurements) ----
tasks = ["knowledge-gain\ngate", "novelty\n(Mahalanobis)", "error\nprediction"]
auc = [0.72, 0.555, 0.679]                      # Sections 3-4, three measurements
level_name = ["source-calibrated\nconformal", "error-model\nguarantee"]
level_val = [0.0, 22.0]                         # 0/23 ports; 22% of ports
required = 80.0                                 # the guarantee's own target (>=80%)

fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
b = ax[0].bar(tasks, auc, color="#3b6ea5", width=0.55)
ax[0].axhline(0.5, ls="--", lw=0.8, c="grey")
ax[0].text(2.35, 0.515, "chance", fontsize=6, c="grey", ha="right")
ax[0].bar_label(b, fmt="%.3f", fontsize=7, padding=2)
ax[0].set_ylim(0.4, 0.85)
ax[0].set_ylabel("transferring: AUC")
ax[0].set_title("(a) ranking transfers", fontsize=8)
ax[0].tick_params(labelsize=7)

b = ax[1].barh(level_name, level_val, color="#a5503b", height=0.45)
ax[1].axvline(required, ls="--", lw=0.8, c="#333")
ax[1].text(required - 1, -0.45, "required level", fontsize=6, ha="right")
ax[1].bar_label(b, fmt="%.0f%%", fontsize=7, padding=2)
ax[1].set_xlim(0, 100)
ax[1].set_xlabel("ports meeting the promised level (%)")
ax[1].set_title("(b) levels do not", fontsize=8)
ax[1].tick_params(labelsize=7)
fig.tight_layout()
fig.savefig(OUT / "fig2_rank_vs_value.pdf")

# ---- Fig. 3: budget sweep and source-count sweep ----
budget = [5, 10, 20, 50]
delta = [0.0076, 0.0149, 0.0192, 0.0113]        # rank - random, pp/100 (Table IV)
lo = [0.0005, 0.0050, 0.0062, 0.0019]
hi = [0.0156, 0.0254, 0.0328, 0.0209]
recovery = [41.4, 80.5, 106.8, 103.0]           # % of consult-everywhere

n_src = [2, 4, 8, 16, 23]
rank20 = [0.415, 1.048, 2.881, 1.469, 2.211]    # pp, e136b
rand20 = [0.114, 0.109, 0.500, 0.347, 0.459]

fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
err = [[d - l for d, l in zip(delta, lo)], [h - d for d, h in zip(delta, hi)]]
b = ax[0].bar([str(k) for k in budget], [d * 100 for d in delta], yerr=[[e * 100 for e in err[0]], [e * 100 for e in err[1]]],
              capsize=3, color="#3b6ea5", width=0.5)
ax[0].axhline(0, lw=0.8, c="k")
for i, (bb, r) in enumerate(zip(b, recovery)):
    ax[0].text(bb.get_x() + bb.get_width() / 2, 0.15, "%.1f%%" % r, ha="center", fontsize=6.5)
ax[0].set_xlabel("consultation budget $k$ (%)")
ax[0].set_ylabel("ranked $-$ random (pp of BA)")
ax[0].set_title("(a) budget sweep (labels: recovery)", fontsize=8)
ax[0].tick_params(labelsize=7)

x = range(len(n_src))
ax[1].bar([i - 0.19 for i in x], rank20, width=0.36, label="ranked top 20%", color="#3b6ea5")
ax[1].bar([i + 0.19 for i in x], rand20, width=0.36, label="random 20%", color="#b9c6d6")
ax[1].set_xticks(list(x))
ax[1].set_xticklabels(n_src)
ax[1].set_xlabel("number of source ports")
ax[1].set_ylabel("$\\Delta$BA (pp)")
ax[1].legend(fontsize=6.5, frameon=False)
ax[1].set_title("(b) source-count sweep", fontsize=8)
ax[1].tick_params(labelsize=7)
fig.tight_layout()
fig.savefig(OUT / "fig3_budget_and_sources.pdf")

# one runnable check: the figure's numbers must stay the recorded ones
assert auc == [0.72, 0.555, 0.679] and recovery[2] == 106.8 and n_src[-1] == 23
print("figs written: fig2_rank_vs_value.pdf, fig3_budget_and_sources.pdf")
