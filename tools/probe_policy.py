#!/usr/bin/env python3
"""정책 출력과 학습 라벨을 직접 대조한다 — 회전 채널이 어디서 어긋나는지 가린다.

전제  정책 서버가 떠 있어야 한다 (기본 http://127.0.0.1:8200)
사용  cd ~/lerobot && ./.venv/bin/python ~/probe_policy.py
      ./.venv/bin/python ~/probe_policy.py --n 20 --repo local/fr5_task6_abs
"""
import argparse, base64, json, math, sys
import numpy as np
import urllib.request


def aa_to_R(aa):
    aa = np.asarray(aa, float); th = np.linalg.norm(aa)
    if th < 1e-12:
        return np.eye(3)
    k = aa / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def ang(R1, R2):
    c = (np.trace(R1.T @ R2) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="local/fr5_task6_abs")
    ap.add_argument("--url", default="http://127.0.0.1:8200/act")
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(a.repo)
    N = len(ds)
    idx = np.linspace(0, N - 2, a.n).astype(int)
    print(f"{a.repo}  프레임 {N}개 중 {len(idx)}개로 대조\n")

    def u8(x):
        x = x.detach().cpu().numpy()
        if x.ndim == 3 and x.shape[0] in (1, 3):
            x = np.transpose(x, (1, 2, 0))
        if x.dtype != np.uint8:
            x = (np.clip(x, 0, 1) * 255).astype(np.uint8) if x.max() <= 1.001 else x.astype(np.uint8)
        return np.ascontiguousarray(x)

    rows = []
    for i in idx:
        s = ds[int(i)]
        st = s["observation.state"].detach().cpu().numpy().astype(float)
        lab = s["action"].detach().cpu().numpy().astype(float)
        if lab.ndim > 1:
            lab = lab[0]
        img, wri = u8(s["observation.images.image"]), u8(s["observation.images.wrist_image"])
        body = json.dumps({"image": base64.b64encode(img).decode(),
                           "wrist": base64.b64encode(wri).decode(),
                           "state": st.tolist()}).encode()
        req = urllib.request.Request(a.url, data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=60) as r:
            out = json.loads(r.read())
        if "chunk" not in out:
            sys.exit(f"서버 응답 이상: {str(out)[:200]}")
        p = np.array(out["chunk"][0], float)          # 청크 첫 스텝
        rows.append((st, lab, p))

    P = np.array([r[2] for r in rows]); S = np.array([r[0] for r in rows]); L = np.array([r[1] for r in rows])
    dpos = np.linalg.norm(P[:, :3] - L[:, :3], axis=1) * 1000
    print("위치 [mm]")
    print(f"  |예측 - 라벨|   중앙 {np.median(dpos):6.1f}   p95 {np.percentile(dpos,95):6.1f}")
    print(f"  예측 x 범위 {P[:,0].min():.3f}~{P[:,0].max():.3f}   라벨 {L[:,0].min():.3f}~{L[:,0].max():.3f}")

    print("\n회전 — 예측값의 축각 크기")
    n_pred = np.linalg.norm(P[:, 3:6], axis=1)
    n_lab = np.linalg.norm(L[:, 3:6], axis=1)
    print(f"  예측 |aa|  중앙 {np.median(n_pred):.4f}   범위 {n_pred.min():.4f}~{n_pred.max():.4f}")
    print(f"  라벨 |aa|  중앙 {np.median(n_lab):.4f}   범위 {n_lab.min():.4f}~{n_lab.max():.4f}")

    hyp = {}
    hyp["① 그대로 절대값"] = P[:, 3:6]
    hyp["② 상태 회전을 더함 (상대값 가정)"] = P[:, 3:6] + S[:, 3:6]
    try:
        stt = json.load(open(f"/home/user/.cache/huggingface/lerobot/{a.repo}/meta/stats.json"))
        m = np.array(stt["action"]["mean"]).ravel()[3:6]
        sd = np.array(stt["action"]["std"]).ravel()[3:6]
        hyp["③ 역정규화 (x*std+mean)"] = P[:, 3:6] * sd + m
    except Exception:
        pass
    print("\n가설별 회전 오차 [deg]  (라벨 자세와의 각도 차이)")
    for name, A in hyp.items():
        e = [ang(aa_to_R(A[i]), aa_to_R(L[i, 3:6])) for i in range(len(A))]
        print(f"  {name:32s} 중앙 {np.median(e):7.2f}   최소 {min(e):6.2f}")
    e0 = [ang(aa_to_R(S[i, 3:6]), aa_to_R(L[i, 3:6])) for i in range(len(S))]
    print(f"  {'(참고) 회전 고정 = 상태 그대로':32s} 중앙 {np.median(e0):7.2f}")

    print("\n그리퍼  예측 중앙 {:.4f}  라벨 중앙 {:.4f}".format(np.median(P[:, 6]), np.median(L[:, 6])))
    print("\n읽는 법")
    print("  ②가 ①보다 훨씬 작으면 후처리에서 상태를 더해 주지 않은 것이다 (롤아웃에서 더하면 해결).")
    print("  ③이 가장 작으면 역정규화가 빠진 것이다.")
    print("  셋 다 20° 안팎이면 모델이 회전을 제대로 학습하지 못한 것이고, 그때는 재학습이 필요하다.")


if __name__ == "__main__":
    main()
