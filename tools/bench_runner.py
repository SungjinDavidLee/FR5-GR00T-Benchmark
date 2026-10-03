#!/usr/bin/env python3
"""30판 평가 진행기 — 키 하나씩으로 배치 안내 → 초기화 → 롤아웃 → 판정 → 기록.

전제  제어판 실행 중, [조작 시작], /leader [리더 추종 시작] 까지 켜둔 상태
사용  python3 bench_runner.py --tag 6k --from 1 --to 30
      python3 bench_runner.py --tag 12k --from 1 --to 30 --csv ~/fr5cap/layouts.csv

대기 화면 키
  s 이 세트 시작      p 배치 안내(손끝이 자리를 짚어준다)   h 초기 자세
  g 그리퍼 초기화     k 이 세트 건너뛰기                    q 종료
판 진행 중 키
  e 지금 끝내기(정책이 스스로 못 끝낼 때)                   q 종료
판정 키
  s 성공   f 실패   x 무효      실패면 이어서 단계 1~9, 원인 a/b/c/d
"""
import argparse, csv, datetime, json, math, os, queue, select, signal
import socket, subprocess, sys, termios, threading, time, tty
import urllib.request
import numpy as np
sys.path.insert(0, "/home/wonseok/fr5cap/collect")
import fr5cfg as C

PANEL = "http://127.0.0.1:8088"
UDP = ("127.0.0.1", 8097)
ROLLOUT = os.path.expanduser("~/fr5cap/fr5_rollout.py")
PY = "/home/wonseok/miniforge3/envs/grounded_sam2_thor/bin/python"
HOME = [88.2542, -80.6386, -100.9385, -88.8735, 89.9876, -0.7449]
DH = (152.0, -425.0, -395.0, 102.0, 102.0, 100.0)
TOOL_Z = 167.6
HOVER_Z = 0.180
VPATH = 6.0

FAIL_CODE = {"a": "stall", "b": "empty", "c": "slip", "d": "off_target", "e": "no_release"}


# ---- 기구학 ---------------------------------------------------------------
def fk(q):
    d1, a2, a3, d4, d5, d6 = DH
    A, D = [0, a2, a3, 0, 0, 0], [d1, 0, 0, d4, d5, d6]
    AL = [math.pi / 2, 0, 0, math.pi / 2, -math.pi / 2, 0]
    T = np.eye(4)
    for a, al, d, th in zip(A, AL, D, np.radians(q)):
        ca, sa, ct, st = math.cos(al), math.sin(al), math.cos(th), math.sin(th)
        T = T @ np.array([[ct, -st * ca, st * sa, a * ct], [st, ct * ca, -ct * sa, a * st],
                          [0, sa, ca, d], [0, 0, 0, 1]])
    return T[:3, :3], (T @ [0, 0, TOOL_Z, 1])[:3]


def jac(q):
    R0, p0 = fk(q); J = np.zeros((6, 6)); e = 1e-4
    for i in range(6):
        qq = list(q); qq[i] += math.degrees(e)
        R1, p1 = fk(qq); W = R0.T @ R1
        w = R0 @ np.array([W[2, 1] - W[1, 2], W[0, 2] - W[2, 0], W[1, 0] - W[0, 1]]) / 2
        J[:3, i], J[3:, i] = (p1 - p0) / e, w / e
    return J


def ik(q0, p_t, R_t, iters=60):
    q = np.array(q0, dtype=float)
    for _ in range(iters):
        R, p = fk(q); ep = p_t - p
        W = R.T @ R_t
        er = R @ (np.array([W[2, 1] - W[1, 2], W[0, 2] - W[2, 0], W[1, 0] - W[0, 1]]) / 2)
        if np.linalg.norm(ep) < 0.02 and np.linalg.norm(er) < 1e-4:
            break
        J = jac(q); J[3:] *= 100.0
        dq = np.degrees(np.linalg.solve(J.T @ J + 1e-2 * np.eye(6), J.T @
                                        np.concatenate([ep, er * 100.0])))
        q = q + np.clip(dq, -10, 10)
    R, p = fk(q)
    return q, float(np.linalg.norm(p_t - p))


# ---- 제어판 ---------------------------------------------------------------
def get(url, t=3):
    with urllib.request.urlopen(url, timeout=t) as r:
        return json.loads(r.read())


def post(path, obj, t=3):
    req = urllib.request.Request(PANEL + path, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=t) as r:
        return json.loads(r.read())


# ---- 목표 전송 (배치 안내·초기 자세 이동에만 쓴다) --------------------------
tgt = {"q": None, "q0": None, "len": 0.0, "s": 0.0, "grip": 1.0, "run": False,
       "seq": 0, "sid": int(time.time()) % 100000, "on": True, "t_last": time.monotonic()}
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def sender():
    while tgt["on"]:
        now = time.monotonic()
        dt = max(0.0, min(0.2, now - tgt["t_last"])); tgt["t_last"] = now
        if tgt["run"] and tgt["q"] is not None:
            if tgt["len"] > 1e-6:
                tgt["s"] = min(tgt["len"], tgt["s"] + VPATH * dt)
                a = tgt["s"] / tgt["len"]
            else:
                a = 1.0
            q = [p0 + (p1 - p0) * a for p0, p1 in zip(tgt["q0"], tgt["q"])]
            tgt["seq"] += 1
            sock.sendto(json.dumps({"sid": tgt["sid"], "seq": tgt["seq"],
                                    "q": q, "grip": tgt["grip"]}).encode(), UDP)
        time.sleep(0.04)


def move_to(q_to, label=""):
    st = get(PANEL + "/api/state")
    q_from = list(st["cmd"])
    tgt["q0"], tgt["q"] = q_from, list(q_to)
    tgt["len"] = max(abs(a - b) for a, b in zip(q_from, q_to))
    tgt["s"], tgt["t_last"], tgt["run"] = 0.0, time.monotonic(), True
    time.sleep(tgt["len"] / VPATH + 1.5)
    tgt["run"] = False
    p, _, _ = C.tcp_to_task(get(PANEL + "/api/state")["tcp"])
    if label:
        print(f"    {label}  손끝 ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")


def guide(row, hover):
    st = get(PANEL + "/api/state")
    q = list(st["cmd"])
    R_hold, _ = fk(q)
    for key, label in (("bowl", "그릇 B"), ("box", "상자 C"), ("plate", "접시 P")):
        p_task = np.array([float(row[f"{key}_x"]), float(row[f"{key}_y"]), hover]) * 1000.0
        q_t, err = ik(q, C.R_TASK_INV @ p_task, R_hold)
        if err > 3.0:
            print(f"    {label}: 도달 불가 (IK 오차 {err:.1f} mm) — 건너뛴다")
            continue
        move_to(q_t.tolist(), label)
        q = q_t.tolist()
        input("      놓았으면 Enter")
    move_to(HOME, "초기 자세")


# ---- 키 입력 --------------------------------------------------------------
class Keys:
    def __init__(self):
        self.q = queue.Queue()
        self.on = True
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.on:
            if select.select([sys.stdin], [], [], 0.2)[0]:
                c = sys.stdin.read(1)
                if c:
                    self.q.put(c.lower())

    def raw(self):
        tty.setcbreak(self.fd)

    def normal(self):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def get(self, timeout=None):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def flush(self):
        while not self.q.empty():
            self.q.get()


# ---- 한 판 실행 -----------------------------------------------------------
def run_trial(name, keys, env_extra):
    env = dict(os.environ)
    env.update(env_extra)
    env.pop("BENCH", None)                 # 기록은 이 스크립트가 한다
    proc = subprocess.Popen([PY, "-u", ROLLOUT], env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    info = {"grip_width": "", "grasp": None, "place": None, "steps": "", "end": "", "slide": ""}

    def pump():
        for line in proc.stdout:
            line = line.rstrip()
            print("   " + line)
            if "그리퍼 닫음" in line and "폭" in line:
                try:
                    info["grip_width"] = line.split("폭")[1].split("m")[0].strip()
                    info["grasp"] = line.split("로봇 [")[1].split("]")[0]
                except Exception:
                    pass
            if "파지 후 밀림" in line:
                try:
                    info["slide"] = line.split("밀림 ")[-1].split("mm")[0].strip()
                except Exception:
                    pass
            if "놓음. 한 판 종료" in line:
                info["end"] = "released"
                try:
                    info["place"] = line.split("위치 [")[1].split("]")[0]
                    info["steps"] = line.split("(")[-1].split("스텝")[0].strip()
                except Exception:
                    pass

    t = threading.Thread(target=pump, daemon=True)
    t.start()
    t0 = time.time()
    while proc.poll() is None:
        c = keys.get(timeout=0.2)
        if c == "e":
            print("\n   [e] 판을 끝낸다")
            info["end"] = info["end"] or "operator"
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
            break
        if c == "q":
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
            return info, time.time() - t0, True
    t.join(timeout=2)
    return info, time.time() - t0, False


def verdict(keys):
    print("\n   결과  [s]성공  [f]실패  [x]무효")
    v = None
    while v not in ("s", "f", "x"):
        v = keys.get(timeout=300)
    stage, fail = "", ""
    if v == "f":
        print("   마지막으로 통과한 단계  [1]~[9]")
        while True:
            c = keys.get(timeout=300)
            if c and c in "123456789":
                stage = "S" + c
                break
        print("   원인  [a]정체  [b]헛잡음  [c]이송 중 놓침  [d]접시 밖  [e]미해제")
        while True:
            c = keys.get(timeout=300)
            if c in FAIL_CODE:
                fail = FAIL_CODE[c]
                break
    elif v == "s":
        stage = "S9"
        fail = "none"
    return v, stage, fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="6k")
    ap.add_argument("--csv", default=os.path.expanduser("~/fr5cap/layouts.csv"))
    ap.add_argument("--out", default=os.path.expanduser("~/fr5cap/bench_runner.csv"))
    ap.add_argument("--from", dest="a", type=int, default=1)
    ap.add_argument("--to", dest="b", type=int, default=30)
    ap.add_argument("--hover", type=float, default=HOVER_Z)
    ap.add_argument("--force", type=int, default=100)
    ap.add_argument("--force-cycle", dest="fcycle", default="",
                    help="판마다 돌려가며 쓸 힘 값들, 예: 70,60,50")
    ap.add_argument("--vpath", type=float, default=6.0)
    ap.add_argument("--grip-wait", type=float, default=2.0)
    args = ap.parse_args()

    rows = {int(r["set"]): r for r in csv.DictReader(open(args.csv))}
    cycle = [int(v) for v in args.fcycle.split(",") if v.strip()] or [args.force]
    # 밖에서 지정한 값이 있으면 그것을 쓴다 (절제 실험에서 개별 항목을 끄기 위함)
    dflt = {"VPATH": str(args.vpath), "KEXEC": "32", "MAXSTEPS": "80", "MAXJUMP": "40",
            "GRIP_Z": "0.09", "NOLIFT": "0", "GRIP_WAIT": str(args.grip_wait)}
    env_extra = {k: os.environ.get(k, v) for k, v in dflt.items()}
    print("실행 설정:", " ".join(f"{k}={v}" for k, v in env_extra.items()),
          " ".join(f"{k}={os.environ[k]}" for k in
                   ("LATCH_MODE", "GRASP_DZ", "GRIP_BINARY", "ORIENT", "REL_Z", "GRIP_LATCH")
                   if k in os.environ))

    st = get(PANEL + "/api/state")
    ld = get(PANEL + "/api/leader/state")
    print(f"제어판 {st['mode']}   추종 '{ld.get('status')}'   배율은 /leader 에서 x1.0 인지 확인")
    if st["mode"] != "engaged":
        sys.exit("제어판이 조작 상태가 아니다. [조작 시작] → /leader [리더 추종 시작] 후 다시 실행")

    new = not os.path.exists(args.out)
    out = open(args.out, "a", newline="")
    w = csv.writer(out)
    if new:
        w.writerow(["time", "trial", "set", "verdict", "stage", "failure", "sec",
                    "steps", "force", "grip_width", "slide_mm", "grasp", "place", "end_why"])
        out.flush()

    keys = Keys()
    keys.raw()
    threading.Thread(target=sender, daemon=True).start()
    try:
        n = args.a
        while n <= args.b:
            row = rows.get(n)
            if row is None:
                print(f"세트 {n} 없음"); n += 1; continue
            name = f"{args.tag}_S{n:02d}"
            f_now = cycle[(n - args.a) % len(cycle)]
            print(f"\n{'='*58}\n세트 {n}  ({name})  힘 {f_now}   그릇 ({row['bowl_x']}, {row['bowl_y']})"
                  f"  상자 ({row['box_x']}, {row['box_y']})  접시 ({row['plate_x']}, {row['plate_y']})")
            print("  [s]시작  [p]배치안내  [h]초기자세  [g]그리퍼초기화  [k]건너뛰기  [q]종료")
            while True:
                c = keys.get(timeout=600)
                if c == "p":
                    keys.normal()
                    print("  배치 안내 — 손끝 아래에 물체 중심을 맞춘다")
                    guide(row, args.hover)
                    keys.raw(); keys.flush()
                    print("  [s]시작  [p]다시  [h]초기자세  [g]그리퍼  [k]건너뛰기  [q]종료")
                elif c == "h":
                    move_to(HOME, "초기 자세")
                elif c == "g":
                    post("/api/gripper", {"pos": 1000})
                    post("/api/force", {"force": f_now})
                    print(f"    그리퍼 1000 / 힘 {f_now}")
                elif c == "k":
                    print("  건너뛴다"); n += 1; break
                elif c == "q":
                    raise KeyboardInterrupt
                elif c == "s":
                    post("/api/gripper", {"pos": 1000})
                    post("/api/force", {"force": f_now})
                    time.sleep(0.6)
                    print(f"  ── {name} 시작 ──   [e] 끝내기")
                    info, sec, quit_now = run_trial(name, keys, env_extra)
                    if quit_now:
                        raise KeyboardInterrupt
                    v, stage, fail = verdict(keys)
                    w.writerow([datetime.datetime.now().isoformat(timespec="seconds"), name, n,
                                v, stage, fail, round(sec, 1), info["steps"], f_now,
                                info["grip_width"], info["slide"],
                                info["grasp"] or "", info["place"] or "", info["end"]])
                    out.flush()
                    print(f"  기록: {v} {stage} {fail}  -> {args.out}")
                    n += 1
                    break
    except KeyboardInterrupt:
        print("\n종료")
    finally:
        tgt["on"] = False
        keys.on = False
        keys.normal()
        out.close()
        time.sleep(0.2)
        print("끝. /leader [리더 추종 중지] → [조작 종료]")


if __name__ == "__main__":
    main()
