#!/usr/bin/env python3
# FR5 정책 롤아웃 (Thor). 정책이 낸 목표를 '가상 리더'로 제어판에 보낸다.
#   전제   제어판(fr5d.py) 실행 중, camd/수집기는 꺼짐, 추론 서버 실행 중
#   실행   python3 ~/fr5cap/fr5_rollout.py            (--dry 는 로봇에 아무것도 보내지 않음)
#   시작   이 스크립트를 띄운 뒤 /leader 에서 [리더 추종 시작]. 중지도 거기서
import base64, json, math, os, socket, sys, threading, time
import numpy as np
import urllib.request
import pyrealsense2 as rs
from PIL import Image
sys.path.insert(0, "/home/wonseok/fr5cap/collect")
import fr5cfg as C

SERVER = os.environ.get("SERVER", "http://10.115.10.57:8200")
PANEL = "http://127.0.0.1:8088"
UDP = ("127.0.0.1", 8097)
KEXEC = int(os.environ.get("KEXEC", "8"))          # 청크 32 개 중 실행할 개수
RATE = float(os.environ.get("RATE", "3.4"))        # 수집 때 실측 주기
MAXSTEPS = int(os.environ.get("MAXSTEPS", "200"))
MAXJUMP = float(os.environ.get("MAXJUMP", "20"))   # 한 목표에서 관절이 이만큼 넘게 뛰면 정지
GRIP_Z = float(os.environ.get("GRIP_Z", "0.09"))   # 이 높이(m)보다 위에서는 그리퍼를 닫지 않는다
NOLIFT = float(os.environ.get("NOLIFT", "0.008"))  # 청크 안에서 목표가 이만큼 다시 올라가면 거기서 끊는다 (0 이면 끔)
SYNC_MM = float(os.environ.get("SYNC_MM", "0"))    # >0 이면 로봇이 이만큼 가까워질 때까지 기다린 뒤 다음 단계로
SYNC_T = float(os.environ.get("SYNC_T", "0"))      # 한 단계에서 기다릴 최대 시간 (0 이면 3/RATE)
GRIP_WAIT = float(os.environ.get("GRIP_WAIT", "0.8"))  # 그리퍼를 닫으라고 한 뒤 멈춰서 기다리는 시간
prev_g = 1.0
held = False        # 실제로 물체를 쥔 적이 있는가
done_ep = False
grasp_p, grasp_w, place_p, end_why = None, None, None, ""
lost_t = 0.0        # 물체를 놓친 것으로 보이기 시작한 시각
w_min = None        # 쥔 뒤 가장 좁아진 폭 (테두리가 얼마나 밀렸는지)
BENCH = os.environ.get("BENCH", "")            # 값을 주면 한 판 끝에 결과를 물어 CSV 에 남긴다
BENCH_CSV = os.environ.get("BENCH_CSV", "/home/wonseok/fr5cap/bench_log.csv")
t_start = time.time()
SETTLE_S = float(os.environ.get("SETTLE_S", "2.0"))    # 청크 끝에서 로봇이 목표에 닿을 때까지 기다릴 시간
SETTLE_MM = float(os.environ.get("SETTLE_MM", "20"))
VPATH = float(os.environ.get("VPATH", "6.0"))            # 경로를 따라가는 관절 속도 (도/초)   # 이만큼 가까워지면 도착으로 본다
RELEASE_MARGIN = float(os.environ.get("RELEASE_MARGIN", "0.003"))  # 파지 폭보다 이만큼(m) 더 열면 놓는 것으로 본다
REL_OPEN = float(os.environ.get("REL_OPEN", "0.002"))              # 파지 폭 + 이만큼을 계속 열라고 하면
REL_HOLD_S = float(os.environ.get("REL_HOLD_S", "0.4"))            # 이 시간 이상 지속되면 놓는 것으로 본다
LOST_W = float(os.environ.get("LOST_W", "0.0015"))                 # 쥔 뒤 폭이 이보다 작아지면 놓친 것으로 본다
HELD_W = float(os.environ.get("HELD_W", "0.0025"))                 # 닫은 뒤 폭이 이보다 크면 물었다고 본다
GRASP_DZ = float(os.environ.get("GRASP_DZ", "0"))                  # 파지 구간에서 목표를 이만큼(m) 더 낮춘다
REL_Z = float(os.environ.get("REL_Z", "0.16"))                     # 이 높이(m)보다 위에서는 놓지 않는다
Z_FLOOR = float(os.environ.get("Z_FLOOR", "0.050"))                # 그래도 이 높이 아래로는 내리지 않는다
LOST_S = float(os.environ.get("LOST_S", "0.6"))                    # 그 상태가 이 시간 이상 이어질 때
RELEASE_MIN = float(os.environ.get("RELEASE_MIN", "0.20"))         # 놓기 판정의 하한 (0.20 = 폭 8 mm)
GRIP_LATCH = os.environ.get("GRIP_LATCH", "1") == "1"    # 쥔 뒤에는 놓을 때까지 계속 꽉 닫아 둔다
FULL_CLOSE = os.environ.get("FULL_CLOSE", "1") == "1"    # 닫는 명령은 폭에 상관없이 끝까지 닫는다
GRIP_BINARY = os.environ.get("GRIP_BINARY", "1") == "1"  # 그리퍼를 열림/닫힘 둘로만 쓴다 (깔짝임 방지)
GRIP_TH = float(os.environ.get("GRIP_TH", "0.3"))        # 이 값(0.3 = 12 mm) 아래면 닫으라는 뜻
LATCH_MODE = os.environ.get("LATCH_MODE", "hold")        # hold: 문 폭에서 버틴다 / full: 계속 끝까지 조인다
LATCH_BITE = float(os.environ.get("LATCH_BITE", "0.0005"))  # 문 폭보다 이만큼(m) 더 조인 위치를 유지한다
ORIENT = os.environ.get("ORIENT", "hold")          # hold: 시작 자세의 방향을 유지 / policy: 정책이 낸 방향
DRY = "--dry" in sys.argv
EXP_T, EXP_W = int(os.environ.get("EXP_THIRD", "25600")), int(os.environ.get("EXP_WRIST", "17300"))
WB_T, WB_W = int(os.environ.get("WB_THIRD", "3600")), int(os.environ.get("WB_WRIST", "3800"))
DH = (152.0, -425.0, -395.0, 102.0, 102.0, 100.0)
TOOL_Z = 167.6


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
        dq = np.degrees(np.linalg.solve(J.T @ J + 1e-2 * np.eye(6), J.T @ np.concatenate([ep, er * 100.0])))
        q = q + np.clip(dq, -10, 10)
    R, p = fk(q)
    return q, float(np.linalg.norm(p_t - p))


def aa_to_R(aa):
    aa = np.asarray(aa, float); th = float(np.linalg.norm(aa))
    if th < 1e-12:
        return np.eye(3)
    k = aa / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def canon(aa, ref):
    aa = np.asarray(aa, float); n = float(np.linalg.norm(aa))
    if n < 1e-9 or ref is None:
        return aa
    alt = aa * (n - 2.0 * math.pi) / n
    return aa if np.linalg.norm(aa - np.asarray(ref)) <= np.linalg.norm(alt - np.asarray(ref)) else alt


class Cam:
    def __init__(self, sn, exp, wb, name):
        self.pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_device(sn)
        cfg.enable_stream(rs.stream.color, C.IMG_W, C.IMG_H, rs.format.rgb8, C.CAM_FPS)
        s = self.pipe.start(cfg).get_device().query_sensors()[0]
        s.set_option(rs.option.enable_auto_exposure, 0); s.set_option(rs.option.exposure, exp)
        s.set_option(rs.option.enable_auto_white_balance, 0); s.set_option(rs.option.white_balance, wb)
        print(f"  {name}: exp={exp} wb={wb}")

    def grab(self):
        f = self.pipe.wait_for_frames().get_color_frame()
        im = np.asanyarray(f.get_data())
        return np.asarray(Image.fromarray(im).resize((256, 256), Image.BILINEAR), dtype=np.uint8)


def http_get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def http_post(url, obj, timeout=30):
    req = urllib.request.Request(url, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


info = http_get(SERVER + "/info")
REF_AA = info.get("ref_axis_angle")
print(f"추론 서버 {SERVER}\n  ckpt {info['ckpt']}\n  ref_axis_angle {REF_AA}")
third, wrist = Cam(C.CAM_THIRD, EXP_T, WB_T, "third"), Cam(C.CAM_WRIST, EXP_W, WB_W, "wrist")
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
tgt = {"q": None, "q0": None, "t0": 0.0, "dwell": 1.0 / RATE, "grip": None,
       "seq": 0, "sid": int(time.time()) % 100000, "on": True,
       "path": None, "s": 0.0, "len": 0.0, "vpath": VPATH,
       "pause": 0.0, "t_last": time.monotonic(), "last_g": 1.0, "close_evt": 0.0,
       "held": False, "rel_th": RELEASE_MIN, "grasp_g": 0.0, "rel_t": 0.0, "rel_now": False,
       "hold_g": 0.0, "z_now": 9.9}


def seg_len(qa, qb):
    return max(abs(a - b) for a, b in zip(qa, qb))


def path_len(path):
    return sum(seg_len(path[i][0], path[i + 1][0]) for i in range(len(path) - 1))


def sample(path, s):                             # 경로 위 s(도) 지점의 관절과 그리퍼
    acc = 0.0
    for i in range(len(path) - 1):
        (qa, _), (qb, gb) = path[i], path[i + 1]
        L = seg_len(qa, qb)
        if s <= acc + L or i == len(path) - 2:
            r = 1.0 if L < 1e-9 else min(1.0, max(0.0, (s - acc) / L))
            return [a + (b - a) * r for a, b in zip(qa, qb)], gb
        acc += L
    return list(path[-1][0]), path[-1][1]


def sender():                                    # 제어판은 0.3초 안에 새 값이 없으면 정지한다
    while tgt["on"]:
        now = time.monotonic()
        dt = max(0.0, min(0.2, now - tgt["t_last"]))
        tgt["t_last"] = now
        if tgt["path"]:
            if now >= tgt["pause"]:              # 경로를 일정한 속도로 따라간다 (중간에 멈추지 않는다)
                tgt["s"] = min(tgt["len"], tgt["s"] + tgt["vpath"] * dt)
            q, g = sample(tgt["path"], tgt["s"])
            if GRIP_BINARY and not tgt["held"]:
                g = 0.0 if g < GRIP_TH else 1.0  # 접근 중에는 열림/닫힘만
            if g < 0.3 <= tgt["last_g"]:         # 닫는 지점에서는 팔을 세우고 그리퍼를 기다린다
                tgt["pause"] = now + GRIP_WAIT
                tgt["close_evt"] = tgt["pause"]
            tgt["last_g"] = g
            if FULL_CLOSE and not tgt["held"] and g < 0.3:
                g = 0.0                          # 닫으라고 하면 끝까지 닫는다 (정책이 낸 폭에서 멈추지 않게)
            if tgt["held"]:                  # 파지 폭보다 꾸준히 열라고 하면 놓으려는 것으로 본다
                if g > tgt["grasp_g"] + REL_OPEN / C.GRIP_MAX_M:
                    tgt["rel_t"] = tgt["rel_t"] or now
                    if now - tgt["rel_t"] >= REL_HOLD_S:
                        tgt["rel_now"] = True
                else:
                    tgt["rel_t"] = 0.0
            if tgt["held"]:                  # 쥔 뒤에는 '놓아도 되는 조건'이 갖춰져야만 연다
                ok = tgt["rel_now"] and tgt["z_now"] <= REL_Z
                g = 1.0 if ok else (tgt["hold_g"] if GRIP_LATCH else g)
            tgt["q"], tgt["grip"] = q, g
        if tgt["q"] is not None and not DRY:
            tgt["seq"] += 1
            sock.sendto(json.dumps({"sid": tgt["sid"], "seq": tgt["seq"],
                                    "q": list(tgt["q"]), "grip": tgt["grip"]}).encode(), UDP)
        time.sleep(0.04)


def panel():
    return http_get(PANEL + "/api/state", timeout=2)


s0 = panel()
print(f"제어판 {s0['mode']}  관절 {[round(x,2) for x in s0['cmd']]}")
_p0, _aa0, _ = C.tcp_to_task(s0["tcp"])
R_HOLD = aa_to_R(canon(_aa0, REF_AA))              # 시작 시점의 손끝 방향
print(f"방향 처리: {ORIENT}" + ("  (정책의 회전 출력은 무시한다)" if ORIENT != "policy" else ""))
tgt["q"] = tgt["q0"] = list(s0["cmd"]); tgt["t0"] = time.monotonic()
tgt["grip"] = C.grip_to_m(s0["gripper"]["pos"]) / C.GRIP_MAX_M
threading.Thread(target=sender, daemon=True).start()
print("\n현재 자세를 목표로 보내는 중. /leader 에서 [리더 추종 시작] 을 누르면 정책이 시작된다.")
print("중지는 /leader 의 [리더 추종 중지] 또는 Ctrl+C. 5초 뒤 시작...\n")
time.sleep(5)

step = 0
try:
    while step < MAXSTEPS:
        st = panel()
        if not DRY and st["mode"] != "engaged":
            print("  제어판이 조작 상태가 아니다. 대기..."); time.sleep(1); continue
        p, aa, _ = C.tcp_to_task(st["tcp"])
        g = C.grip_to_m(st["gripper"]["pos"])
        state = list(p) + list(canon(aa, REF_AA)) + [g, -g]
        im, wr = third.grab(), wrist.grab()
        t0 = time.time()
        r = http_post(SERVER + "/act", {
            "state": state, "image": base64.b64encode(im.tobytes()).decode(),
            "wrist": base64.b64encode(wr.tobytes()).decode()})
        chunk = np.array(r["chunk"], dtype=float)
        net = time.time() - t0
        z_min = None                           # 청크 전체를 하나의 경로로 만든다
        q_seed = list(panel()["cmd"])
        path = [(list(q_seed), tgt["grip"] if tgt["grip"] is not None else 1.0)]
        prev_g = path[0][1]
        for k in range(min(KEXEC, len(chunk))):
            a = chunk[k]
            if NOLIFT > 0 and z_min is not None and a[2] > z_min + NOLIFT:
                break
            z_min = a[2] if z_min is None else min(z_min, a[2])
            R_task = aa_to_R(a[3:6]) if ORIENT == "policy" else R_HOLD
            p_goal = np.array(a[:3], dtype=float)
            if GRASP_DZ > 0 and p_goal[2] < GRIP_Z:      # 집는 구간만 조금 더 깊게
                p_goal[2] = max(Z_FLOOR, p_goal[2] - GRASP_DZ)
            q_t, err = ik(q_seed, C.R_TASK_INV @ (p_goal * 1000.0), C.R_TASK_INV @ R_task)
            jump = float(np.max(np.abs(q_t - np.array(q_seed))))
            if err > 2.0 or jump > MAXJUMP:
                print(f"  step {step:3d} k{k}: 건너뜀 (IK 오차 {err:.2f} mm, 관절 변화 {jump:.1f}도)")
                continue
            g_cmd = float(min(1.0, max(0.0, a[6] / C.GRIP_MAX_M)))
            if a[6] < 0.02 and float(a[2]) > GRIP_Z and prev_g >= 0.3:
                g_cmd = 1.0                      # 높은 곳에서 새로 닫는 것만 막는다
            path.append((q_t.tolist(), g_cmd))
            q_seed, prev_g = q_t.tolist(), g_cmd
            step += 1
        if len(path) < 2:
            print(f"  step {step:3d}: 쓸 수 있는 목표가 없다"); time.sleep(0.3); continue
        d0 = np.linalg.norm(np.array(chunk[0][:3]) - np.array(p)) * 1000
        print(f"  step {step:3d} {net*1000:4.0f}ms  로봇 {np.round(p,3)} -> 경로 {len(path)-1}점 "
              f"({path_len(path):.1f}도, 첫 목표 {d0:.1f}mm)  grip {chunk[0][6]:.4f} (로봇 {g:.4f})")
        tgt["s"], tgt["len"] = 0.0, path_len(path)
        tgt["last_g"], tgt["close_evt"], tgt["pause"] = path[0][1], 0.0, 0.0
        tgt["t_last"] = time.monotonic()
        tgt["path"] = path
        t_run, lim_run = time.time(), tgt["len"] / max(0.5, VPATH) + 3.0 + GRIP_WAIT
        while time.time() - t_run < lim_run:     # 경로를 다 따라갈 때까지 기다린다
            time.sleep(0.15)
            if tgt["close_evt"] and time.monotonic() > tgt["close_evt"]:
                tgt["close_evt"] = 0.0
                try:
                    stg = panel()
                    pg, _, _ = C.tcp_to_task(stg["tcp"])
                    gw = C.grip_to_m(stg["gripper"]["pos"])
                    if gw > HELD_W and not held:         # 처음 문 순간만 기록한다
                        held = True
                        tgt["held"] = True                # 놓을 때까지 그리퍼를 잠근다
                        tgt["rel_th"] = max(RELEASE_MIN, (gw + RELEASE_MARGIN) / C.GRIP_MAX_M)
                        grasp_p, grasp_w = [round(float(x), 4) for x in pg], round(gw, 4)
                        tgt["grasp_g"] = gw / C.GRIP_MAX_M
                        tgt["hold_g"] = (0.0 if LATCH_MODE == "full"
                                         else max(0.0, (gw - LATCH_BITE) / C.GRIP_MAX_M))
                        tgt["rel_t"], tgt["rel_now"] = 0.0, False
                        print(f"  잡은 뒤 유지 위치 {tgt['hold_g']*1000:.0f} "
                              f"({tgt['hold_g']*C.GRIP_MAX_M*1000:.1f} mm, {LATCH_MODE})")
                        print(f"  파지 폭 {gw:.4f} m -> 놓기 판정 기준 {tgt['rel_th']*C.GRIP_MAX_M:.4f} m")
                    print(f"  step {step:3d} 그리퍼 닫음: 로봇 {np.round(pg,3)}  폭 {gw:.4f} m "
                          f"({'물체 있음' if held else '허공'})")
                except Exception as e:
                    print("  그리퍼 확인 실패:", repr(e)[:60])
            if held and tgt["rel_now"] and tgt["z_now"] <= REL_Z:   # 놓았다 = 한 판 끝
                time.sleep(0.8)
                tgt["path"], tgt["held"], tgt["rel_now"] = None, False, False   # 멈추고 잠금 해제
                try:
                    pf, _, _ = C.tcp_to_task(panel()["tcp"])
                    place_p = [round(float(x), 4) for x in pf]
                    print(f"\n  놓음. 한 판 종료 — 위치 {np.round(pf,3)}  ({step} 스텝)")
                except Exception:
                    print("\n  놓음. 한 판 종료")
                done_ep, end_why = True, "released"
                break
            if held and not done_ep:            # 쥐고 있는데 손가락이 다 닫혔다 = 놓쳤다
                try:
                    st_now = panel()
                    gw_now = C.grip_to_m(st_now["gripper"]["pos"])
                    tgt["z_now"] = float(C.tcp_to_task(st_now["tcp"])[0][2])
                except Exception:
                    gw_now = None
                if gw_now is not None:
                    w_min = gw_now if w_min is None else min(w_min, gw_now)
                if gw_now is not None and gw_now < LOST_W:
                    lost_t = lost_t or time.time()
                    if time.time() - lost_t >= LOST_S:
                        tgt["path"] = None
                        try:
                            pl, _, _ = C.tcp_to_task(panel()["tcp"])
                            place_p = [round(float(x), 4) for x in pl]
                            print(f"\n  물체를 놓쳤다 (폭 {gw_now:.4f} m). 한 판 종료 — 위치 "
                                  f"{np.round(pl,3)}  ({step} 스텝)")
                        except Exception:
                            print(f"\n  물체를 놓쳤다. 한 판 종료  ({step} 스텝)")
                        done_ep, end_why = True, "lost"
                        break
                else:
                    lost_t = 0.0
            if tgt["s"] >= tgt["len"] - 1e-6:
                break
        tgt["path"] = None                       # 마지막 지점을 유지한 채 다음 추론으로
        if done_ep:
            break
        try:                                   # 청크를 끝낸 시점의 실제 상태와 제어판 판정
            st2 = panel()
            p2, _, _ = C.tcp_to_task(st2["tcp"])
            L = http_get(PANEL + "/api/leader/state", 2)
            print(f"      청크 끝: 로봇 z {p2[2]:.3f}  (보낸 최저 목표 z {z_min if z_min is not None else float('nan'):.3f})"
                  f"  제어판 '{L.get('status')}'  하한 {L.get('z_min')}  제어루프 {L.get('loop_hz')} Hz")
        except Exception as e:
            print("      상태 읽기 실패:", repr(e)[:80])
except KeyboardInterrupt:
    end_why = end_why or "interrupt"
    print("\n중지")
finally:
    tgt["on"] = False
    time.sleep(0.2)
    if held and grasp_w and w_min is not None:
        print(f"  파지 후 밀림: {grasp_w*1000:.1f} mm -> 최소 {w_min*1000:.1f} mm "
              f"(밀림 {(grasp_w-w_min)*1000:.1f} mm)")
    if BENCH:
        import csv, datetime
        auto = "놓음" if done_ep else ("쥠" if held else "미파지")
        print(f"\n  [{BENCH}] {step} 스텝, {time.time()-t_start:.0f}초, 자동판정 {auto}")
        try:
            v = input("  결과 [s]성공 / [f]실패 / [x]무효 / 메모: ").strip()
        except Exception:
            v = "?"
        row = [datetime.datetime.now().isoformat(timespec="seconds"), BENCH,
               info.get("ckpt", ""), v[:1], v[1:].strip(),
               step, round(time.time() - t_start, 1), int(held), int(done_ep),
               end_why or ("maxsteps" if step >= MAXSTEPS else "stop"),
               grasp_w if grasp_w is not None else "",
               grasp_p[0] if grasp_p else "", grasp_p[1] if grasp_p else "", grasp_p[2] if grasp_p else "",
               place_p[0] if place_p else "", place_p[1] if place_p else "", place_p[2] if place_p else "",
               ORIENT, VPATH, KEXEC, MAXJUMP, GRIP_Z, NOLIFT, GRIP_WAIT]
        new = not os.path.exists(BENCH_CSV)
        with open(BENCH_CSV, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "trial", "ckpt", "verdict", "note", "steps", "sec",
                            "held", "released", "end_why", "grip_width",
                            "grasp_x", "grasp_y", "grasp_z", "place_x", "place_y", "place_z",
                            "orient", "vpath", "kexec", "maxjump", "grip_z", "nolift", "grip_wait"])
            w.writerow(row)
        print(f"  기록: {BENCH_CSV}")
    print("끝. /leader 에서 [리더 추종 중지] 를 누르고 제어판에서 [조작 종료] 하라.")
