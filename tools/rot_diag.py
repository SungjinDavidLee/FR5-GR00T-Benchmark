#!/usr/bin/env python3
"""회전 라벨 진단 — 로봇 없이 학습 데이터만으로 돈다.

축각(axis-angle)은 같은 회전을 두 가지로 적을 수 있다. 축 n 으로 θ 와 축 -n 으로 (2π-θ).
θ 가 π 근처면 둘의 크기가 비슷해져 프레임마다 표현이 갈릴 수 있고, 그러면
회전 라벨의 프레임 간 차분이 실제 회전각과 전혀 달라진다. GR00T 는 use_relative_actions=true
로 내부에서 action - state 를 쓰므로, 이 차분이 깨지면 회전 채널 자체가 망가진다.

사용
  python3 rot_diag.py
  python3 rot_diag.py --root ~/.cache/huggingface/lerobot/local/fr5_task6_abs --plot
"""
import argparse, glob, json, math, os, sys
import numpy as np

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas 가 필요하다:  pip install pandas pyarrow")


def aa_to_R(aa):
    aa = np.asarray(aa, dtype=float)
    th = np.linalg.norm(aa)
    if th < 1e-12:
        return np.eye(3)
    k = aa / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def rot_angle(R1, R2):
    c = (np.trace(R1.T @ R2) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def load(root):
    pats = [os.path.join(root, "data", "**", "*.parquet"),
            os.path.join(root, "**", "*.parquet")]
    files = []
    for p in pats:
        files = sorted(glob.glob(os.path.expanduser(p), recursive=True))
        if files:
            break
    if not files:
        sys.exit(f"parquet 을 못 찾았다: {root}")
    eps = []
    for f in files:
        df = pd.read_parquet(f)
        if "observation.state" not in df.columns:
            continue
        if "episode_index" in df.columns:
            for key, g in df.groupby("episode_index", sort=True):   # 판마다 나눈다
                st = np.stack(g["observation.state"].to_numpy())
                ac = np.stack(g["action"].to_numpy()) if "action" in g.columns else None
                eps.append((int(key), st, ac))
        else:
            st = np.stack(df["observation.state"].to_numpy())
            ac = np.stack(df["action"].to_numpy()) if "action" in df.columns else None
            eps.append((len(eps), st, ac))
    return files, eps


def analyze(eps, root=None, plot=False):
    norms, d_aa, d_true, flips, cross = [], [], [], [], 0
    flip_eps = {}
    act_err = []
    for key, st, ac in eps:
        aa = st[:, 3:6]
        n = np.linalg.norm(aa, axis=1)
        norms.append(n)
        cross += int((n > math.pi).any() and (n < math.pi).any())
        Rs = [aa_to_R(v) for v in aa]
        for i in range(len(aa) - 1):
            dd = float(np.linalg.norm(aa[i + 1] - aa[i]))
            tt = rot_angle(Rs[i], Rs[i + 1])
            d_aa.append(dd); d_true.append(tt)
            if dd > math.radians(tt) + 0.5:          # 실제 회전보다 라벨 차분이 훨씬 크다
                flips.append((key, i, dd, tt))
                flip_eps[key] = flip_eps.get(key, 0) + 1
        if ac is not None and len(ac) == len(st):     # action[t] == state[t+1][:7] 확인
            act_err.append(float(np.abs(ac[:-1, :7] - st[1:, :7]).max()))

    norms = np.concatenate(norms); d_aa = np.array(d_aa); d_true = np.array(d_true)
    q = [5, 50, 95, 100]
    print("축각 크기 |aa|  [rad]")
    print("  " + "  ".join(f"p{p}={np.percentile(norms, p):.4f}" for p in q))
    print(f"  π = {math.pi:.4f}   π 보다 큰 프레임 {100*np.mean(norms > math.pi):.1f} %")
    print(f"  한 판 안에서 π 를 넘나드는 판 {cross}/{len(eps)}개\n")

    print("프레임 사이 실제 회전각  [deg]")
    print("  " + "  ".join(f"p{p}={np.percentile(d_true, p):.2f}" for p in q))
    print("라벨 차분 ||Δaa|| 를 각도로  [deg]")
    dd_deg = np.degrees(d_aa)
    print("  " + "  ".join(f"p{p}={np.percentile(dd_deg, p):.2f}" for p in q))
    print(f"\n표현이 갈린 것으로 보이는 프레임 {len(flips)}개 "
          f"({100*len(flips)/max(1,len(d_aa)):.2f} %),  해당 판 {len(flip_eps)}개")
    if flips:
        print("  (판, 프레임, ||Δaa||rad, 실제각deg) 상위 5개")
        for f in sorted(flips, key=lambda x: -x[2])[:5]:
            print(f"    판 {f[0]:3d}  프레임 {f[1]:4d}   {f[2]:.3f} rad = {math.degrees(f[2]):7.1f}°"
                  f"   실제 {f[3]:.2f}°")
    if act_err:
        print(f"\naction[t] 와 state[t+1][:7] 최대 차이  {max(act_err):.6f}  (0 이어야 정상)")

    sp = os.path.join(root, "meta", "stats.json") if root else None
    cands = ([sp, os.path.join(root, "meta", "norm_stats.json"),
              os.path.join(root, "norm_stats.json")] if root else [])
    for cand in cands:
        if os.path.exists(cand):
            s = json.load(open(cand))
            key = "observation.state"
            if key in s and "mean" in s[key]:
                m = np.array(s[key]["mean"]).ravel(); sd = np.array(s[key]["std"]).ravel()
                print(f"\n{os.path.basename(cand)}  observation.state 회전 3개 차원")
                for i in (3, 4, 5):
                    if i < len(m):
                        print(f"  dim{i}  mean {m[i]:+.4f}   std {sd[i]:.4f}")
            if "action" in s and "mean" in s["action"]:
                m = np.array(s["action"]["mean"]).ravel(); sd = np.array(s["action"]["std"]).ravel()
                print(f"{os.path.basename(cand)}  action 회전 3개 차원")
                for i in (3, 4, 5):
                    if i < len(m):
                        print(f"  dim{i}  mean {m[i]:+.4f}   std {sd[i]:.4f}")
            break

    print("\n읽는 법")
    print("  실제 회전각 p50 이 1~2° 인데 라벨 차분 p95 가 100° 를 넘으면 표현이 갈린 것이다.")
    print("  그 경우 변환기의 축각 정규화를 '판 첫 프레임 기준'이 아니라 '직전 프레임 기준'으로")
    print("  바꾸고 다시 변환·학습해야 회전 채널을 쓸 수 있다.")

    if plot:
        try:
            import matplotlib
        except ImportError:
            print("\n(matplotlib 없음 — 그림은 건너뛴다)")
            return
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (x1, x2) = plt.subplots(1, 2, figsize=(9, 3.2), dpi=180)
        x1.hist(norms, bins=60, color="#1F3B73")
        x1.axvline(math.pi, color="#A32B2B", linewidth=1.3)
        x1.set_xlabel("|axis-angle|  [rad]"); x1.set_ylabel("frames")
        x1.set_title("axis-angle magnitude (red = π)", fontsize=10)
        x2.hist(dd_deg, bins=np.arange(0, 200, 2), color="#1F3B73")
        x2.set_xlabel("frame-to-frame ||Δaa||  [deg]"); x2.set_ylabel("frames")
        x2.set_yscale("log"); x2.set_title("label difference per step", fontsize=10)
        for ax in (x1, x2):
            ax.grid(True, color="#E4E6EA", linewidth=0.6); ax.set_axisbelow(True)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
        fig.tight_layout(); fig.savefig("rot_diag.png", bbox_inches="tight")
        print("\nrot_diag.png 저장")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/.cache/huggingface/lerobot/local/fr5_task6_abs")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    root = os.path.expanduser(a.root)
    files, eps = load(root)
    print(f"데이터셋 {root}")
    print(f"  parquet {len(files)}개,  판 {len(eps)}개,  프레임 {sum(len(s) for _, s, _ in eps)}개\n")
    analyze(eps, root=root, plot=a.plot)


if __name__ == "__main__":
    main()
