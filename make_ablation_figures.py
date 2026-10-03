#!/usr/bin/env python3
"""절제 실험·회전 진단 도표. 입력 data/ablation.csv, data/rotation_probe.csv -> fig/*.png"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

NAVY, RED, GREEN, OCHRE, GREY = "#1F3B73", "#A32B2B", "#1E6B52", "#8A6D1F", "#9AA0A6"
INK, GRID = "#1A1A1A", "#E4E6EA"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#444444", "axes.linewidth": 0.8,
                     "axes.labelcolor": INK, "xtick.color": "#444444", "ytick.color": "#444444",
                     "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
                     "axes.axisbelow": True})
NAME = {"noA": "A  grasp-depth bias", "noC": "C  grip latch", "noE": "E  binary gripper",
        "noB": "B  close-height gate", "noD": "D  release-height gate"}


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.6)


def rows():
    return {r["condition"]: r for r in csv.DictReader(open("data/ablation.csv"))}


def fig_ablation(R):
    order = ["base10", "noA", "noE", "noC", "noB", "noD"]
    names = {"base10": "all rules on", **NAME}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw={"width_ratios": [1.25, 1]})
    vals = [100 * int(R[k]["n_success"]) / int(R[k]["n_valid"]) for k in order]
    cols = [GREY, NAVY] + [GREEN if v >= 60 else (OCHRE if v >= 30 else RED) for v in vals[2:]]
    a1.bar(range(len(order)), vals, color=cols, width=0.62)
    for i, k in enumerate(order):
        a1.text(i, vals[i] + 2.5, f"{R[k]['n_success']}/{R[k]['n_valid']}", ha="center",
                fontsize=9, color=INK)
    a1.annotate("reference for C/E/B/D", xy=(1, vals[1]), xytext=(2.3, 99),
                fontsize=7.5, color=NAVY,
                arrowprops=dict(arrowstyle="->", color=NAVY, linewidth=0.9))
    a1.set_xticks(range(len(order)))
    a1.set_xticklabels([names[k] for k in order], rotation=18, ha="right", fontsize=8)
    a1.set_ylim(0, 112); a1.set_ylabel("success rate [%]")
    a1.set_title("Leave-one-out at a fixed object layout", fontsize=10, pad=8)
    clean(a1)

    data, labs = [], []
    for k in ["base10", "noA", "noE", "noC", "noB", "noD"]:
        s_ = R[k]["slide_mm"]
        if s_:
            data.append([float(v) for v in s_.split(";")])
            labs.append("base" if k == "base10" else NAME[k].split()[0])
    bp = a2.boxplot(data, tick_labels=labs, widths=0.55, patch_artist=True,
                    medianprops=dict(color=INK, linewidth=1.1))
    for patch, v in zip(bp["boxes"], data):
        c = GREEN if np.median(v) < 4 else RED
        patch.set_facecolor(c); patch.set_alpha(0.3); patch.set_edgecolor(c)
    for i, v in enumerate(data, start=1):
        a2.scatter(np.random.normal(i, 0.05, len(v)), v, s=14, color=INK, alpha=0.55, zorder=3)
    a2.axhline(4.0, color=INK, linewidth=0.9, linestyle=(0, (4, 3)))
    a2.text(6.45, 4.2, "4 mm", ha="right", fontsize=8, color=INK)
    a2.set_ylabel("rim slide after grasp [mm]")
    a2.set_title("Slide distribution by condition", fontsize=10, pad=8)
    clean(a2)
    fig.tight_layout(); fig.savefig("fig/fig11_ablation.png"); plt.close(fig)


def fig_rotation():
    R = list(csv.DictReader(open("data/rotation_probe.csv")))
    lab = {"abs": "raw output\nas absolute", "plus_state": "output +\ncurrent state",
           "unnorm": "output × std\n+ mean", "hold": "hold current\norientation"}
    v = [float(r["median_error_deg"]) for r in R]
    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    cols = [RED, RED, OCHRE, GREEN]
    ax.bar(range(len(R)), v, color=cols, width=0.55)
    for i, x in enumerate(v):
        ax.text(i, x * 1.25, f"{x:.1f}°", ha="center", fontsize=9, color=INK)
    ax.set_yscale("log")
    ax.set_ylim(0.4, 500)
    ax.set_xticks(range(len(R)))
    ax.set_xticklabels([lab[r["hypothesis"]] for r in R], fontsize=8)
    ax.set_ylabel("median orientation error [deg]  (log)")
    ax.set_title("Decoding the policy's rotation output (12 training frames)", fontsize=10, pad=8)
    ax.axhline(1.11, color=NAVY, linewidth=1.1, linestyle=(0, (4, 3)))
    ax.text(3.45, 1.22, "actual rotation per step  1.11°", ha="right", fontsize=7.5, color=NAVY)
    clean(ax)
    fig.tight_layout(); fig.savefig("fig/fig12_rotation.png"); plt.close(fig)


if __name__ == "__main__":
    np.random.seed(1)
    R = rows()
    fig_ablation(R)
    fig_rotation()
    print("fig/fig11_ablation.png, fig/fig12_rotation.png 생성")
