#!/usr/bin/env python3
"""30판 평가 결과 도표 생성. 입력 data/trials.csv, data/layouts.csv -> fig/*.png"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

NAVY, RED, GREEN, OCHRE, PURPLE, GREY = "#1F3B73", "#A32B2B", "#1E6B52", "#8A6D1F", "#5B4B8A", "#9AA0A6"
INK, GRID = "#1A1A1A", "#E4E6EA"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": "#444444", "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "xtick.color": "#444444", "ytick.color": "#444444",
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight", "axes.axisbelow": True,
})


def load():
    T = []
    for r in csv.DictReader(open("data/trials.csv")):
        d = {k: (float(v) if v not in ("", None) and k not in
                 ("verdict", "stage", "failure") else v) for k, v in r.items()}
        d["set"] = int(float(r["set"]))
        T.append(d)
    L = {int(r["set"]): {k: float(v) for k, v in r.items() if k != "set"}
         for r in csv.DictReader(open("data/layouts.csv"))}
    return T, L


def clean(ax, spines=("top", "right")):
    for s in spines:
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.6)


# --- Fig 2. 성공률 -----------------------------------------------------------
def fig_success(T):
    n = len(T); s = sum(1 for t in T if t["verdict"] == "s")
    blocks = [(f"{a+1}–{a+10}", sum(1 for t in T[a:a+10] if t["verdict"] == "s")) for a in (0, 10, 20)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [1, 1.5]})
    se = np.sqrt((s/n)*(1-s/n)/n) * 100
    a1.bar(["FR5\n(this work)", "FR3\n(reference)"], [100*s/n, 53.3],
           yerr=[1.96*se, 1.96*np.sqrt(0.533*0.467/30)*100], capsize=4,
           color=[NAVY, GREY], width=0.55)
    a1.text(0, 100*s/n + 5, f"{100*s/n:.1f}%\n{s}/{n}", ha="center", fontsize=9, color=INK)
    a1.text(1, 53.3 + 5, "53.3%\n16/30", ha="center", fontsize=9, color="#5A5F66")
    a1.set_ylim(0, 100); a1.set_ylabel("success rate [%]"); clean(a1)
    a1.set_title("Overall success (30 trials each)", fontsize=10, pad=8)
    a2.bar([b for b, _ in blocks], [100*c/10 for _, c in blocks], color=NAVY, width=0.5)
    for i, (_, c) in enumerate(blocks):
        a2.text(i, 100*c/10 + 3, f"{c}/10", ha="center", fontsize=9, color=INK)
    a2.set_ylim(0, 100); a2.set_ylabel("success rate [%]"); a2.set_xlabel("trial block")
    clean(a2); a2.set_title("By block of 10 trials", fontsize=10, pad=8)
    fig.tight_layout(); fig.savefig("fig/fig2_success.png"); plt.close(fig)


# --- Fig 3. 단계 통과 --------------------------------------------------------
def fig_stages(T):
    n = len(T)
    ks = [(1, "S1 target"), (3, "S3 pre-grasp"), (4, "S4 contact"), (5, "S5 grasp"),
          (6, "S6 lift"), (7, "S7 clear"), (8, "S8 transport"), (9, "S9 place")]
    vals = []
    for k, _ in ks:
        vals.append(sum(1 for t in T if t["verdict"] == "s" or int(t["stage"][1:]) >= k))
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    x = np.arange(len(ks))
    ax.bar(x, [100*v/n for v in vals], color=[NAVY]*6 + [RED, RED], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, 100*v/n + 2, f"{v}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in ks], rotation=20, ha="right")
    ax.set_ylim(0, 108); ax.set_ylabel("trials passing [%]")
    ax.set_title("Stage pass rate (n = 30)", fontsize=10, pad=8)
    clean(ax)
    ax.annotate("drop at transport", xy=(6, 100*vals[6]/n), xytext=(4.2, 30),
                fontsize=8.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, linewidth=0.9))
    fig.tight_layout(); fig.savefig("fig/fig3_stages.png"); plt.close(fig)


# --- Fig 4. 실패 유형 --------------------------------------------------------
def fig_failures(T):
    lab = {"empty": "empty grasp", "slip": "dropped in transport", "no_release": "no release"}
    c = {}
    for t in T:
        if t["verdict"] == "f":
            c[t["failure"]] = c.get(t["failure"], 0) + 1
    items = sorted(c.items(), key=lambda kv: -kv[1])
    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    ax.barh([lab[k] for k, _ in items][::-1], [v for _, v in items][::-1],
            color=[RED, OCHRE, PURPLE][:len(items)][::-1], height=0.55)
    for i, (_, v) in enumerate(items[::-1]):
        ax.text(v + 0.15, i, str(v), va="center", fontsize=9, color=INK)
    ax.set_xlabel("trials"); ax.set_xlim(0, max(c.values()) + 1.2)
    ax.set_title(f"Failure modes (n = {sum(c.values())})", fontsize=10, pad=8)
    clean(ax); ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig("fig/fig4_failures.png"); plt.close(fig)


# --- Fig 5. 배치별 결과 지도 -------------------------------------------------
def fig_map(T, L):
    style = {"s": (GREEN, "o", "success"), "empty": (RED, "^", "empty grasp"),
             "slip": (OCHRE, "s", "dropped in transport"), "no_release": (PURPLE, "X", "no release")}
    fig, ax = plt.subplots(figsize=(6.4, 6.6))
    for n, l in L.items():
        ax.plot([l["bowl_x"], l["plate_x"]], [l["bowl_y"], l["plate_y"]],
                color="#D5D8DC", linewidth=0.7, zorder=1)
        ax.scatter([l["plate_x"]], [l["plate_y"]], s=26, marker="D",
                   facecolor="#DDE1E5", edgecolor="white", linewidth=0.5, zorder=2)
    seen = {}
    for t in T:
        l = L[t["set"]]
        key = "s" if t["verdict"] == "s" else t["failure"]
        col, mk, lb = style[key]; seen[key] = (col, mk, lb)
        ax.scatter([l["bowl_x"]], [l["bowl_y"]], s=80, marker=mk, facecolor=col,
                   edgecolor="white", linewidth=0.7, zorder=4)
        ax.annotate(str(t["set"]), (l["bowl_x"], l["bowl_y"]), textcoords="offset points",
                    xytext=(8, 5), fontsize=6.5, color=INK, zorder=5,
                    path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
    ax.set_xlim(0.52, 0.82); ax.set_ylim(0.0, 0.40); ax.set_aspect("equal")
    ax.set_xlabel("x [m]  (away from the robot base)"); ax.set_ylabel("y [m]")
    ax.set_title("Outcome by layout (marker at the target bowl)", fontsize=10, pad=8)
    clean(ax, spines=())
    order = ["s", "empty", "slip", "no_release"]
    ax.legend(handles=[Line2D([], [], marker=seen[k][1], color="none", markerfacecolor=seen[k][0],
                              markeredgecolor="white", markersize=8, label=seen[k][2])
                       for k in order if k in seen], loc="lower left", frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig("fig/fig5_outcome_map.png"); plt.close(fig)


# --- Fig 6. 밀림 -------------------------------------------------------------
def fig_slide(T):
    S = [t["slide_mm"] for t in T if t["verdict"] == "s" and t["slide_mm"] != ""]
    F = [t["slide_mm"] for t in T if t["verdict"] == "f" and t["slide_mm"] != ""]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [1, 1.3]})
    bp = a1.boxplot([S, F], tick_labels=["success", "failure"], widths=0.5, patch_artist=True,
                    medianprops=dict(color=INK, linewidth=1.2))
    for patch, c in zip(bp["boxes"], [GREEN, RED]):
        patch.set_facecolor(c); patch.set_alpha(0.35); patch.set_edgecolor(c)
    for i, D in enumerate((S, F), start=1):
        a1.scatter(np.random.normal(i, 0.055, len(D)), D, s=18,
                   color=[GREEN, RED][i-1], alpha=0.9, zorder=3, edgecolor="white", linewidth=0.4)
    a1.axhline(4.0, color=INK, linewidth=0.9, linestyle=(0, (4, 3)))
    a1.text(2.42, 4.15, "4 mm", fontsize=8, color=INK, ha="right")
    a1.set_ylabel("slide after grasp [mm]"); clean(a1)
    a1.set_title("Rim slide vs outcome", fontsize=10, pad=8)
    W = [(t["grip_width_mm"], t["slide_mm"], t["verdict"]) for t in T if t["slide_mm"] != ""]
    for w, s_, v in W:
        a2.scatter([w], [s_], s=46, color=GREEN if v == "s" else RED,
                   edgecolor="white", linewidth=0.5)
    a2.axhline(4.0, color=INK, linewidth=0.9, linestyle=(0, (4, 3)))
    a2.set_xlabel("grip width at closure [mm]"); a2.set_ylabel("slide [mm]")
    a2.set_title("Width does not separate; slide does", fontsize=10, pad=8)
    clean(a2)
    a2.legend(handles=[Line2D([], [], marker="o", color="none", markerfacecolor=GREEN,
                              markeredgecolor="white", markersize=8, label="success"),
                       Line2D([], [], marker="o", color="none", markerfacecolor=RED,
                              markeredgecolor="white", markersize=8, label="failure")],
              loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig("fig/fig6_slide.png"); plt.close(fig)


# --- Fig 7. 공간 추종 --------------------------------------------------------
def fig_tracking(T, L):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
    for ax, key in zip(axes, ("x", "y")):
        b = np.array([L[t["set"]][f"bowl_{key}"] for t in T])
        g = np.array([t[f"grasp_{key}"] for t in T])
        col = [GREEN if t["verdict"] == "s" else RED for t in T]
        ax.scatter(b, g, s=42, c=col, edgecolor="white", linewidth=0.5, zorder=3)
        A = np.vstack([b, np.ones_like(b)]).T
        m, c = np.linalg.lstsq(A, g, rcond=None)[0]
        xs = np.linspace(b.min() - 0.01, b.max() + 0.01, 10)
        ax.plot(xs, m * xs + c, color=NAVY, linewidth=1.4, zorder=2,
                label=f"fit  slope {m:.2f}  (r = {np.corrcoef(b, g)[0,1]:.2f})")
        off = np.mean(g - b)
        ax.plot(xs, xs + off, color=GREY, linewidth=1.1, linestyle=(0, (4, 3)), zorder=2,
                label="perfect tracking (slope 1)")
        ax.set_xlabel(f"bowl {key} [m]"); ax.set_ylabel(f"grasp {key} [m]")
        ax.legend(frameon=False, fontsize=7.5, loc="upper left")
        clean(ax)
    axes[0].set_title("Spatial tracking — x", fontsize=10, pad=8)
    axes[1].set_title("Spatial tracking — y", fontsize=10, pad=8)
    fig.tight_layout(); fig.savefig("fig/fig7_tracking.png"); plt.close(fig)


# --- Fig 8. 해제 높이 --------------------------------------------------------
def fig_release(T):
    P = [t["place_z"] for t in T if t["verdict"] == "s"]
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    ax.axvspan(0.083, 0.118, color=NAVY, alpha=0.10, zorder=0)
    ax.text(0.1005, 5.6, "demonstration range\n(10–90 %)", ha="center", fontsize=7.5, color=NAVY)
    ax.hist(P, bins=np.arange(0.06, 0.18, 0.01), color=GREEN, alpha=0.85,
            edgecolor="white", linewidth=0.8)
    ax.axvline(np.mean(P), color=GREEN, linewidth=1.4)
    ax.text(np.mean(P) + 0.002, 4.6, f"FR5 mean {np.mean(P):.3f} m", fontsize=8, color=GREEN)
    ax.axvline(0.164, color=RED, linewidth=1.4, linestyle=(0, (4, 3)))
    ax.text(0.1655, 4.6, "FR3 mean 0.164 m", fontsize=8, color=RED)
    ax.set_xlabel("release height  z [m]"); ax.set_ylabel("trials")
    ax.set_title("Release height of successful trials", fontsize=10, pad=8)
    ax.set_ylim(0, 6.4); clean(ax)
    fig.tight_layout(); fig.savefig("fig/fig8_release_height.png"); plt.close(fig)


# --- Fig 9. 판 순서 ----------------------------------------------------------
def fig_sequence(T):
    fig, ax = plt.subplots(figsize=(7.6, 2.4))
    for t in T:
        ok = t["verdict"] == "s"
        ax.bar(t["set"], 1, color=GREEN if ok else RED, width=0.7)
        ax.text(t["set"], 1.06, "S" if ok else "F", ha="center", fontsize=6.5,
                color=GREEN if ok else RED)
    cum = np.cumsum([1 if t["verdict"] == "s" else 0 for t in T]) / np.arange(1, len(T) + 1)
    ax2 = ax.twinx()
    ax2.plot([t["set"] for t in T], cum * 100, color=NAVY, linewidth=1.4, marker="o", markersize=3)
    ax2.set_ylabel("running success rate [%]", color=NAVY)
    ax2.set_ylim(0, 100); ax2.tick_params(axis="y", colors=NAVY)
    ax2.spines["top"].set_visible(False)
    ax.set_yticks([]); ax.set_xlabel("trial (layout set)"); ax.set_xlim(0.3, 30.7)
    ax.set_ylim(0, 1.25)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title("Trial sequence and running success rate", fontsize=10, pad=8)
    fig.tight_layout(); fig.savefig("fig/fig9_sequence.png"); plt.close(fig)


# --- Fig 10. 파지 높이 -------------------------------------------------------
def fig_graspz(T):
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    for t in T:
        ok = t["verdict"] == "s"
        ax.scatter([t["grasp_z"] * 1000], [t["grip_width_mm"]], s=46,
                   color=GREEN if ok else RED, edgecolor="white", linewidth=0.5)
    ax.axvspan(48, 69, color=NAVY, alpha=0.08, zorder=0)
    ax.text(58, 11.3, "demonstration grasp height (10–90 %)", ha="center", fontsize=7.5, color=NAVY)
    ax.set_xlabel("grasp height  z [mm]"); ax.set_ylabel("grip width at closure [mm]")
    ax.set_title("Grasp height vs grip width", fontsize=10, pad=8)
    clean(ax)
    ax.legend(handles=[Line2D([], [], marker="o", color="none", markerfacecolor=GREEN,
                              markeredgecolor="white", markersize=8, label="success"),
                       Line2D([], [], marker="o", color="none", markerfacecolor=RED,
                              markeredgecolor="white", markersize=8, label="failure")],
              loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig("fig/fig10_grasp_height.png"); plt.close(fig)


if __name__ == "__main__":
    np.random.seed(0)
    T, L = load()
    fig_success(T); fig_stages(T); fig_failures(T); fig_map(T, L)
    fig_slide(T); fig_tracking(T, L); fig_release(T); fig_sequence(T); fig_graspz(T)
    print("fig/*.png 생성 완료")
