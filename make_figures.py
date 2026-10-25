"""Regenerates the chart figures of the X-TransUNet manuscript as vector PDFs
from results/results.json.  Usage: python make_figures.py <out_dir>"""
import json, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans",
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
})
# Okabe-Ito colour-blind-safe palette
BLUE, ORANGE, GREEN, VERM, PURPLE = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"
out = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(out, exist_ok=True)
R = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "results.json")))
W, H = 3.5, 2.5

def save(fig, name):
    fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight"); plt.close(fig)

# --- rounds 1..15 (untrained round-0 checkpoint is not plotted) ---
rounds = sorted(int(k) for k in R["fed_by_round"]["dice"])
dice = [R["fed_by_round"]["dice"][str(r)] for r in rounds]
iou = [R["fed_by_round"]["iou"][str(r)] for r in rounds]
fig, ax = plt.subplots(figsize=(W, H))
ax.plot(rounds, dice, "o-", color=BLUE, ms=3.2, lw=1.2, label="Dice")
ax.plot(rounds, iou, "s--", color=ORANGE, ms=3.2, lw=1.2, label="IoU")
ax.set_xlabel("Federated communication round"); ax.set_ylabel("Score (pooled validation set)")
ax.set_ylim(0.78, 1.0); ax.set_xticks([1, 3, 5, 7, 9, 11, 13, 15]); ax.grid(alpha=0.3, lw=0.4)
ax.legend(loc="lower right", frameon=False)
save(fig, "dice_iou_rounds")

# --- multi-seed dot plot ---
s = R["seeds"]; p = R["primary_run_round15"]
fig, axs = plt.subplots(2, 1, figsize=(W, H * 1.25), sharex=True)
labels = ["Seed 0", "Seed 1", "Seed 2", "Primary run"]
for ax, key, mean, sd, ttl in [(axs[0], "dice", s["mean_dice"], s["sd_dice"], "Dice"),
                               (axs[1], "iou", s["mean_iou"], s["sd_iou"], "IoU")]:
    vals = s[key]
    ax.axhspan(mean - sd, mean + sd, color="0.88", lw=0)
    ax.axhline(mean, color="0.4", lw=0.8, ls="--")
    ax.plot(range(3), vals, "o", color=BLUE if key == "dice" else ORANGE, ms=5)
    ax.plot([3], [p[key]], "D", mfc="white", mec="k", ms=5)
    for i, v in enumerate(vals + [p[key]]):
        ax.annotate(f"{v:.4f}", (i, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=6.5)
    ax.set_title(f"{ttl}  (mean {mean:.4f} ± {sd:.4f})", fontsize=7.5)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels); ax.set_xlim(-0.6, 3.6)
    lo, hi = min(vals + [p[key]]) , max(vals + [p[key]]); pad = (hi - lo) * 0.9 + 0.0004
    ax.set_ylim(lo - pad, hi + pad); ax.grid(axis="y", alpha=0.3, lw=0.4)
axs[0].set_ylabel("Round-15 Dice"); axs[1].set_ylabel("Round-15 IoU")
fig.tight_layout(); save(fig, "multiseed")

# --- per-client dot plot ---
pc = R["per_client_round15"]
fig, ax = plt.subplots(figsize=(W * 0.85, H))
for i, (k, lab) in enumerate([("montgomery_n28", "Montgomery\n(Client 0, n = 28)"), ("shenzhen_n114", "Shenzhen\n(Client 1, n = 114)")]):
    for j, (m, c, mk) in enumerate([("dice", BLUE, "o"), ("iou", ORANGE, "s")]):
        v = pc[k][m]
        ax.plot(i, v, mk, color=c, ms=6, label=m.capitalize() if i == 0 else None)
        ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points", xytext=(12, -3), fontsize=7)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Montgomery\n(Client 0, n = 28)", "Shenzhen\n(Client 1, n = 114)"])
ax.set_xlim(-0.5, 1.6); ax.set_ylim(0.90, 0.99); ax.set_ylabel("Round-15 score")
ax.grid(axis="y", alpha=0.3, lw=0.4); ax.legend(frameon=False, loc="upper right", ncol=2)
save(fig, "per_client")

# --- in-mask fractions ---
xr = R["xai_rounds"]
att = R["attention_in_mask"]["values"]; g2 = R["gradcam_dec2_in_mask"]["values"]; g4 = R["gradcam_dec4_in_mask"]["values"]; chance = R["lung_area_fraction_chance_level"]
fig, ax = plt.subplots(figsize=(W, H))
ax.plot(xr, g2, "o-", color=VERM, ms=3.5, lw=1.2, label="Grad-CAM, dec_conv2 (28×28)")
ax.plot(xr, g4, "s-", color=PURPLE, ms=3.5, lw=1.2, label="Grad-CAM, dec_conv4 (112×112)")
ax.set_xlabel("Federated communication round"); ax.set_ylabel("In-mask activation fraction")
ax.set_xticks(xr); ax.set_ylim(0.62, 0.84); ax.grid(alpha=0.3, lw=0.4); ax.legend(frameon=False, loc="center right")
save(fig, "gradcam_layer_comparison")

fig, ax = plt.subplots(figsize=(W, H))
ax.plot(xr, g4, "s-", color=PURPLE, ms=3.5, lw=1.2, label="Grad-CAM, dec_conv4")
ax.plot(xr, att, "o-", color=GREEN, ms=3.5, lw=1.2, label="Transformer attention")
ax.axhline(chance, color="0.4", ls=":", lw=1.0, label=f"Chance level ({chance:.3f})")
ax.set_xlabel("Federated communication round"); ax.set_ylabel("In-mask activation fraction")
ax.set_xticks(xr); ax.set_ylim(0.15, 0.9); ax.grid(alpha=0.3, lw=0.4); ax.legend(frameon=False, loc="center right")
save(fig, "complementarity")
print("done ->", out)
