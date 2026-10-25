"""Redraws the methodology-pipeline figure (Figure 1) as an editable vector PDF.
Usage: python make_fig_pipeline.py <out.pdf>"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
matplotlib.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans"})

fig = plt.figure(figsize=(7.3, 7.1)); ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1075); ax.set_ylim(1050, 0); ax.axis("off")

def box(x0, y0, x1, y1, fc, ec, lw=1.2, r=8):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))
def txt(x, y, s, size=6.8, w="normal", c="k", ha="center", style="normal"):
    ax.text(x, y, s, fontsize=size, fontweight=w, color=c, ha=ha, va="center", fontstyle=style, linespacing=1.35)
def arrow(p, q, c="k", ls="-", lw=1.4):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, color=c, lw=lw, ls=ls, shrinkA=0, shrinkB=0))
def card(x0, y0, x1, y1, title, body=None, fc="white", ec="0.7", ts=6.8, bs=6.3):
    box(x0, y0, x1, y1, fc, ec, 0.8, 5)
    cx = (x0 + x1) / 2
    if body: txt(cx, y0 + 20, title, ts, "bold"); txt(cx, (y0 + y1) / 2 + 12, body, bs)
    else: txt(cx, (y0 + y1) / 2, title, ts, "bold")

# Aggregation server
box(243, 43, 658, 204, "#D8EBD8", "#2E7D32"); txt(450, 62, "GLOBAL PARAMETER AGGREGATION (FedAvg)", 7.2, "bold", "#1B5E20")
box(263, 82, 638, 189, "white", "#9CC59E", 0.8, 5)
txt(450, 100, "Central federated server", 6.8, "bold"); txt(450, 118, "Federated Averaging (FedAvg)", 6.8, "bold")
txt(450, 135, "Sample-weighted mean of client parameters:", 6.3)
txt(450, 153, r"$\theta_{global}=\Sigma_i\,(n_i/N)\,\theta_i,\ i\in\{0,1\}$", 7.0)
txt(450, 173, "min_fit_clients = 2, min_evaluate_clients = 2, 15 rounds", 6.1)
# Checkpoint + evaluation
box(741, 64, 1058, 152, "#D8EBD8", "#2E7D32"); txt(900, 94, "Load saved global checkpoint", 6.8, "bold", "#1B5E20"); txt(900, 120, "global_model_round_r.pth", 6.3)
arrow((658, 108), (741, 108), "#2E7D32"); arrow((900, 152), (900, 188), "#2E7D32")
box(741, 188, 1058, 604, "#EBDDF1", "#7B3FA0"); txt(900, 212, "GLOBAL MODEL EVALUATION", 7.2, "bold", "#4A235A"); txt(900, 232, "Saved checkpoint → evaluation and XAI audit", 6.3, "normal", "#4A235A")
card(757, 264, 1043, 357, "Held-out validation X-rays", "Inference with\nglobal_model_round_r.pth")
card(757, 368, 1043, 460, "XAI audit module", "Attention hook +\nGrad-CAM hook")
card(757, 471, 1043, 589, "Explainable output", "Dice / IoU scores\nAttention map + Grad-CAM\noverlays (rounds 3, 5, 10, 15)")
# Local optimisation
box(10, 304, 229, 596, "#F8DBB8", "#B8651B"); txt(120, 332, "LOCAL CLIENT\nOPTIMIZATION", 7.0, "bold", "#8A4B12"); txt(120, 372, "(TransUNetMini training)", 6.3, "bold", "#8A4B12", style="italic")
txt(120, 470, "Adam, lr = 1e-4\n\n1 local epoch / round\n\nbatch size = 8\n\nLoss: BCE + Dice", 6.5)
# Clients
for x0, name, n, ctry in [(243, "Client 0 (Montgomery)", 138, "Montgomery images (USA)"), (461, "Client 1 (Shenzhen)", 566, "Shenzhen images (China)")]:
    box(x0, 303, x0 + 197, 594, "white", "#2B5FAF", 1.4)
    cx = x0 + 98.5; txt(cx, 322, name, 7.0, "bold", "#1F3F78")
    box(x0 + 12, 340, x0 + 185, 378, "#EFEFEF", "0.6", 0.8, 4); txt(cx, 359, "Initialize local\nTransUNetMini", 6.2, "bold")
    box(x0 + 12, 388, x0 + 185, 450, "#FBF0C0", "#C9A400", 0.8, 4); txt(cx, 405, "Load local dataset", 6.4, "bold", "#6B5300"); txt(cx, 429, f"{n} {ctry}".replace(" images (", " images\n("), 5.8, "normal", "#6B5300")
    box(x0 + 12, 462, x0 + 92, 582, "#EFEFEF", "0.6", 0.8, 4); txt(x0 + 52, 522, "Train loop\n(1 epoch)", 6.2, "bold")
    box(x0 + 102, 462, x0 + 185, 582, "#EFEFEF", "0.6", 0.8, 4); txt(x0 + 143, 522, "Local\nmodel", 6.2, "bold")
arrow((315, 303), (315, 204)); arrow((378, 204), (378, 303), "#1F54B0"); txt(425, 291, "Broadcast", 6.2, "bold", "#1F54B0")
arrow((534, 303), (534, 204)); arrow((596, 204), (596, 303), "#1F54B0"); txt(643, 291, "Broadcast", 6.2, "bold", "#1F54B0")
txt(283, 216, r"Send $\theta_0$", 6.2, "bold"); txt(502, 216, r"Send $\theta_1$", 6.2, "bold")
# Partitioning
box(10, 619, 658, 765, "#CFE4F5", "#2B5FAF"); txt(25, 640, "CLIENT DATASET PARTITIONING (NON-IID SETUP)", 7.0, "bold", "#1F3F78", ha="left")
card(25, 656, 222, 750, "Python script", "MetaData.csv", bs=6.3)
card(232, 656, 508, 750, "Data splitter", "partitions the combined 704 image-mask pairs\ninto two isolated, non-overlapping streams\nby the \"county\" field", bs=6.0)
card(518, 656, 648, 750, "Raw datasets", "Montgomery folder\nShenzhen folder", bs=6.0)
arrow((342, 619), (342, 594)); arrow((560, 619), (560, 594))
# Controls
box(10, 791, 607, 1009, "#F8DBB8", "#B8651B"); txt(25, 812, "VALIDATION CONTROLS", 7.0, "bold", "#8A4B12", ha="left"); txt(25, 833, "run alongside the federated loop", 6.2, "bold", "#8A4B12", ha="left", style="italic")
card(25, 850, 207, 995, "Centralized baseline", "TransUNetMini,\npooled data,\n15 epochs", bs=6.0)
card(217, 850, 399, 995, "No-ViT ablation", "ResNet34UNetBaseline,\nFedAvg, 15 rounds", bs=6.0)
card(409, 850, 592, 995, "Multi-seed repetition", "3 repetitions\n(seeds 0, 1, 2),\nfinal round (15)", bs=6.0)
arrow((307, 791), (307, 765), "#B8651B")
# External validation
box(627, 791, 1043, 1009, "#EBDDF1", "#7B3FA0"); txt(835, 812, "EXTERNAL VALIDATION (JSRT)", 7.0, "bold", "#4A235A"); txt(835, 833, "Inference only, no fine-tuning", 6.2, "bold", "#4A235A", style="italic")
card(642, 850, 827, 995, "JSRT cohort (Japan)", "247 held-out images,\nthird institution", bs=6.0)
card(842, 850, 1028, 995, "Evaluated models", "Federated (round 15)\n+ centralized baseline", bs=6.0)
arrow((934, 791), (934, 604), "#7B3FA0", "--")
fig.savefig(sys.argv[1] if len(sys.argv) > 1 else "pipeline.pdf"); print("ok")
