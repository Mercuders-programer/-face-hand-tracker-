"""
avatar_renderer.py — 포즈 랜드마크 기반 애니 캐릭터 오버레이 (방향 C)

포즈 33점으로 애니 스타일 캐릭터를 프레임에 직접 그립니다.
실시간 뷰 + 내보내기 모두 지원.
"""

import cv2
import numpy as np

# ── 스킨 팔레트 (모두 BGR 순서) ───────────────────────────────────────────
# (base, shadow, highlight, outline)
AVATAR_SKINS = {
    "anime":  ((185, 210, 250), (145, 165, 210), (215, 235, 255), (20, 20, 30)),
    "light":  ((190, 215, 240), (150, 170, 205), (215, 238, 255), (20, 20, 30)),
    "medium": ((115, 148, 185), ( 85, 115, 148), (145, 178, 212), (15, 15, 20)),
    "dark":   (( 70,  95, 125), ( 48,  68,  95), ( 90, 118, 152), (10, 10, 15)),
}
AVATAR_LABELS    = ["anime", "light", "medium", "dark"]
AVATAR_LABELS_KO = ["애니",  "밝음",  "중간",   "어두움"]

# ── 아이리스(눈) 색 팔레트 (BGR) ─────────────────────────────────────────
_IRIS_COLORS = {
    "anime":  (200,  90, 120),  # 퍼플-핑크
    "light":  (160, 120,  60),  # 브라운
    "medium": (100, 140,  55),  # 그린
    "dark":   ( 80, 100, 160),  # 딥 블루
}

# ── MediaPipe Pose 33점 인덱스 ────────────────────────────────────────────
_NOSE               = 0
_L_EYE_I, _R_EYE_I = 1, 4
_L_EAR,   _R_EAR   = 7, 8
_L_SHLD,  _R_SHLD  = 11, 12
_L_ELBOW, _R_ELBOW = 13, 14
_L_WRIST, _R_WRIST = 15, 16
_L_HIP,   _R_HIP   = 23, 24
_L_KNEE,  _R_KNEE  = 25, 26
_L_ANKLE, _R_ANKLE = 27, 28

_LIMB_PAIRS = [
    (_L_SHLD, _L_ELBOW), (_L_ELBOW, _L_WRIST),  # 왼팔
    (_R_SHLD, _R_ELBOW), (_R_ELBOW, _R_WRIST),  # 오른팔
    (_L_HIP,  _L_KNEE),  (_L_KNEE,  _L_ANKLE),  # 왼다리
    (_R_HIP,  _R_KNEE),  (_R_KNEE,  _R_ANKLE),  # 오른다리
]

_JOINT_IDXS = [
    _L_SHLD, _R_SHLD,
    _L_ELBOW, _R_ELBOW,
    _L_WRIST, _R_WRIST,
    _L_HIP,  _R_HIP,
    _L_KNEE, _R_KNEE,
    _L_ANKLE, _R_ANKLE,
]


# ── 내부 헬퍼 ─────────────────────────────────────────────────────────────

def _pt(pose, idx, w, h):
    lm = pose[idx]
    return (int(np.clip(lm.x * w, 0, w - 1)),
            int(np.clip(lm.y * h, 0, h - 1)))


def _vis(pose, idx, thr=0.3):
    return (idx < len(pose) and
            getattr(pose[idx], "visibility", 1.0) > thr)


def _draw_limb_pass(overlay, pose, s_i, e_i, w, h, radius, color):
    """팔다리 한 뼈대: 두꺼운 선 + 양 끝 원."""
    if not (_vis(pose, s_i) and _vis(pose, e_i)):
        return
    p1, p2 = _pt(pose, s_i, w, h), _pt(pose, e_i, w, h)
    cv2.line(overlay, p1, p2, color, radius * 2, cv2.LINE_AA)
    cv2.circle(overlay, p1, radius, color, -1, cv2.LINE_AA)
    cv2.circle(overlay, p2, radius, color, -1, cv2.LINE_AA)


# ── 메인 드로잉 함수 ──────────────────────────────────────────────────────

def draw_avatar(overlay, pose_res, w, h, skin="anime"):
    """
    포즈 랜드마크로 애니 스타일 캐릭터를 overlay에 직접 그립니다.

    Parameters
    ----------
    overlay  : BGR ndarray (in-place 수정)
    pose_res : PoseLandmarker 감지 결과 (None이면 아무것도 그리지 않음)
    w, h     : 프레임 폭/높이 (픽셀)
    skin     : 스킨 키 ("anime" | "light" | "medium" | "dark")
    """
    if pose_res is None or not pose_res.pose_landmarks:
        return

    base, shadow, hi_col, outline = AVATAR_SKINS.get(skin, AVATAR_SKINS["anime"])
    iris_col  = _IRIS_COLORS.get(skin, _IRIS_COLORS["anime"])
    hair_col  = (30, 25, 18)   # 짙은 갈색 (BGR)

    for pose in pose_res.pose_landmarks:
        if len(pose) < 29:
            continue

        # ── 스케일 기준값 계산 ────────────────────────────────────────────
        if _vis(pose, _L_SHLD) and _vis(pose, _R_SHLD):
            ls = _pt(pose, _L_SHLD, w, h)
            rs = _pt(pose, _R_SHLD, w, h)
            shld_dist = max(int(np.hypot(ls[0] - rs[0], ls[1] - rs[1])), 40)
        else:
            shld_dist = 80

        limb_r  = max(int(shld_dist * 0.14), 6)   # 팔다리 반경
        joint_r = max(int(shld_dist * 0.11), 5)   # 관절 반경
        head_r  = max(int(shld_dist * 0.55), 22)  # 머리 반경
        ol_r    = limb_r + 4                       # 윤곽선 두께 (반경)

        # ═══════════════════════════════════════════════════════════════
        # Pass 1 — 검정 윤곽선 (몸통 + 팔다리)
        # ═══════════════════════════════════════════════════════════════
        for s_i, e_i in _LIMB_PAIRS:
            _draw_limb_pass(overlay, pose, s_i, e_i, w, h, ol_r, outline)

        if all(_vis(pose, i) for i in [_L_SHLD, _R_SHLD, _L_HIP, _R_HIP]):
            ls = _pt(pose, _L_SHLD, w, h);  rs = _pt(pose, _R_SHLD, w, h)
            lh = _pt(pose, _L_HIP,  w, h);  rh = _pt(pose, _R_HIP,  w, h)
            pad = int(shld_dist * 0.08)
            torso_ol = np.array([
                (ls[0] - pad, ls[1] - pad), (rs[0] + pad, rs[1] - pad),
                (rh[0] + pad, rh[1] + pad), (lh[0] - pad, lh[1] + pad),
            ], dtype=np.int32)
            cv2.fillConvexPoly(overlay, torso_ol, outline)

        # ═══════════════════════════════════════════════════════════════
        # Pass 2 — 피부색 채우기 (그림자 → 베이스)
        # ═══════════════════════════════════════════════════════════════
        for s_i, e_i in _LIMB_PAIRS:
            _draw_limb_pass(overlay, pose, s_i, e_i, w, h, limb_r + 1, shadow)
            _draw_limb_pass(overlay, pose, s_i, e_i, w, h, limb_r,     base)

        if all(_vis(pose, i) for i in [_L_SHLD, _R_SHLD, _L_HIP, _R_HIP]):
            ls = _pt(pose, _L_SHLD, w, h);  rs = _pt(pose, _R_SHLD, w, h)
            lh = _pt(pose, _L_HIP,  w, h);  rh = _pt(pose, _R_HIP,  w, h)
            pad = int(shld_dist * 0.05)
            torso = np.array([
                (ls[0] - pad, ls[1]), (rs[0] + pad, rs[1]),
                (rh[0] + pad, rh[1]), (lh[0] - pad, lh[1]),
            ], dtype=np.int32)
            cv2.fillConvexPoly(overlay, torso, base)
            cv2.polylines(overlay, [torso], True, outline, 2, cv2.LINE_AA)

        # ═══════════════════════════════════════════════════════════════
        # Pass 3 — 관절 원 (그림자 + 베이스 + 하이라이트)
        # ═══════════════════════════════════════════════════════════════
        for j_i in _JOINT_IDXS:
            if _vis(pose, j_i):
                p = _pt(pose, j_i, w, h)
                cv2.circle(overlay, p, joint_r + 2, outline,  -1, cv2.LINE_AA)
                cv2.circle(overlay, p, joint_r + 1, shadow,   -1, cv2.LINE_AA)
                cv2.circle(overlay, p, joint_r,     base,     -1, cv2.LINE_AA)
                cv2.circle(overlay,
                           (p[0] - joint_r // 3, p[1] - joint_r // 3),
                           max(joint_r // 3, 2), hi_col, -1, cv2.LINE_AA)

        # ═══════════════════════════════════════════════════════════════
        # Pass 4 — 머리 (원 + 머리카락 + 눈 + 입 + 볼터치)
        # ═══════════════════════════════════════════════════════════════
        hc = None
        if _vis(pose, _NOSE):
            nx, ny = _pt(pose, _NOSE, w, h)
            hc = (nx, ny - int(head_r * 0.50))
        elif _vis(pose, _L_EAR) and _vis(pose, _R_EAR):
            le = _pt(pose, _L_EAR, w, h)
            re = _pt(pose, _R_EAR, w, h)
            hc = ((le[0] + re[0]) // 2, (le[1] + re[1]) // 2)
        elif _vis(pose, _L_SHLD) and _vis(pose, _R_SHLD):
            ls2 = _pt(pose, _L_SHLD, w, h)
            rs2 = _pt(pose, _R_SHLD, w, h)
            hc = ((ls2[0] + rs2[0]) // 2,
                  min(ls2[1], rs2[1]) - int(head_r * 1.1))

        if hc is not None:
            hx, hy = hc

            # 머리 윤곽 + 피부 베이스
            cv2.circle(overlay, (hx, hy), head_r + 3, outline, -1, cv2.LINE_AA)
            cv2.circle(overlay, (hx, hy), head_r,     base,    -1, cv2.LINE_AA)

            # 머리카락 (위쪽 반원 + 옆으로 약간 더)
            hair_pts = []
            for angle in range(-200, 25, 4):
                rad = np.radians(angle)
                hair_pts.append([
                    int(hx + (head_r + 1) * np.cos(rad)),
                    int(hy + (head_r + 1) * np.sin(rad)),
                ])
            hair_pts.append([hx + int(head_r * 0.7),  hy + int(head_r * 0.1)])
            hair_pts.append([hx - int(head_r * 0.7),  hy + int(head_r * 0.1)])
            if len(hair_pts) >= 3:
                cv2.fillPoly(overlay, [np.array(hair_pts, np.int32)], hair_col)

            # 눈
            eye_y   = int(hy + head_r * 0.08)
            eye_off = int(head_r * 0.33)
            ew = max(int(head_r * 0.22), 4)
            eh = max(int(head_r * 0.28), 5)

            for ex in [hx - eye_off, hx + eye_off]:
                # 흰자
                cv2.ellipse(overlay, (ex, eye_y), (ew + 2, eh + 2),
                            0, 0, 360, (255, 255, 255), -1, cv2.LINE_AA)
                # 아이리스 (컬러)
                cv2.ellipse(overlay, (ex, eye_y),
                            (int(ew * 0.75), int(eh * 0.88)),
                            0, 0, 360, iris_col, -1, cv2.LINE_AA)
                # 동공
                cv2.ellipse(overlay, (ex, eye_y),
                            (max(int(ew * 0.38), 2), max(int(eh * 0.48), 2)),
                            0, 0, 360, (10, 10, 10), -1, cv2.LINE_AA)
                # 하이라이트 (2개)
                cv2.circle(overlay,
                           (ex - ew // 4, eye_y - eh // 4),
                           max(ew // 4, 2), (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(overlay,
                           (ex + ew // 5, eye_y + eh // 5),
                           max(ew // 6, 1), (220, 220, 220), -1, cv2.LINE_AA)
                # 속눈썹 (위 호)
                cv2.ellipse(overlay, (ex, eye_y), (ew + 2, eh + 2),
                            0, 195, 345, hair_col, 2, cv2.LINE_AA)

            # 입 (미소)
            mx = hx
            my = int(hy + head_r * 0.52)
            mw = int(head_r * 0.30)
            mh = max(int(head_r * 0.12), 3)
            cv2.ellipse(overlay, (mx, my), (mw, mh),
                        0, 0, 180, outline, 2, cv2.LINE_AA)

            # 볼터치 (핑크 타원)
            blush_r = max(int(head_r * 0.16), 5)
            blush_off = int(head_r * 0.47)
            blush_y   = int(hy + head_r * 0.30)
            for bx in [hx - blush_off, hx + blush_off]:
                blush = np.zeros_like(overlay)
                cv2.circle(blush, (bx, blush_y), blush_r, (140, 130, 210), -1)
                overlay[:] = cv2.addWeighted(overlay, 1.0, blush, 0.35, 0)
