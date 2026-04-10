"""
video_panel.py — 영상 분석 패널

기능:
  - 영상 파일 로드 + 재생/일시정지
  - 타임라인 스크러버 (클릭/드래그로 탐색)
  - 얼굴/손 랜드마크 오버레이 토글
"""

import tkinter as tk
from tkinter import messagebox, filedialog
import threading
import cv2
import numpy as np
import mediapipe as mp
import os

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode
from mediapipe.tasks.python.vision import drawing_utils as mp_draw
from mediapipe.tasks.python.vision import drawing_styles as mp_styles
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections

_BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACE_MODEL = os.path.join(_BASE, "models", "face_landmarker.task")
HAND_MODEL = os.path.join(_BASE, "models", "hand_landmarker.task")
POSE_MODEL = os.path.join(_BASE, "models", "pose_landmarker_full.task")

try:
    from .tracker import (FrameData, VideoInfo, PersonData,
                          _extract_face, _extract_hand, _extract_pose,
                          _build_persons, MAX_PERSONS, PERSON_COLORS)
    from .exporter import export_json, export_ae_keyframes
except ImportError:
    from tracker import (FrameData, VideoInfo, PersonData,
                         _extract_face, _extract_hand, _extract_pose,
                         _build_persons, MAX_PERSONS, PERSON_COLORS)
    from exporter import export_json, export_ae_keyframes

try:
    from PIL import Image, ImageTk
except ImportError:
    raise ImportError("Pillow가 필요합니다: pip install Pillow")

# ── 테마 색상 ──────────────────────────────────────────────────────────────
BG_DARK  = "#1a1a2e"
BG_PANEL = "#16213e"
ACCENT   = "#4a7fff"
TEXT_W   = "#e0e0ff"
TEXT_G   = "#8888aa"
TL_BG    = "#0f0f1f"
TL_H     = 30

# 얼굴 이미지 워핑에 사용할 랜드마크 인덱스 (6점)
_FACE_IMG_KPT = [33, 263, 4, 168, 61, 291]  # R.Eye.O, L.Eye.O, Nose.T, Nose.B, Mouth.R, Mouth.L

# 만화 손 스타일 뼈대 연결 정의
_HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def _apply_face_img_overlay(overlay, face_res, w, h, face_img, face_img_pts,
                             eye_y_pct=55, size_pct=100):
    """로드된 얼굴 이미지(BGRA)를 감지된 얼굴 위에 합성한다.
    face_img_pts is not None → Homography 정밀 모드 (실제 얼굴 사진)
    face_img_pts is None     → Affine 자동 모드  (일러스트/그림)
    """
    if not face_res.face_landmarks:
        return
    img_h, img_w = face_img.shape[:2]
    for _lf in face_res.face_landmarks:
        if len(_lf) < 264:
            continue
        r_eye = np.array([_lf[33].x * w, _lf[33].y * h], dtype=np.float64)
        l_eye = np.array([_lf[263].x * w, _lf[263].y * h], dtype=np.float64)
        eye_center = (r_eye + l_eye) / 2.0
        angle = float(np.degrees(np.arctan2(l_eye[1] - r_eye[1], l_eye[0] - r_eye[0])))

        if face_img_pts is not None:
            # ── Homography 정밀 모드 ─────────────────────────────────────
            if len(_lf) <= max(_FACE_IMG_KPT):
                continue
            dst_pts = np.float32([[_lf[i].x * w, _lf[i].y * h] for i in _FACE_IMG_KPT])
            M, _ = cv2.findHomography(face_img_pts, dst_pts)
            if M is None:
                continue
            warped = cv2.warpPerspective(face_img, M, (w, h),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(0, 0, 0, 0))
        else:
            # ── Affine 자동 모드 (일러스트) ──────────────────────────────
            ys = [_lf[i].y * h for i in range(len(_lf))]
            face_h_px = max(ys) - min(ys)
            scale = face_h_px * (size_pct / 100.0) / (img_h * 0.8)
            src_cx = img_w / 2.0
            src_cy = img_h * (eye_y_pct / 100.0)
            M = cv2.getRotationMatrix2D((src_cx, src_cy), -angle, scale)
            M[0, 2] += eye_center[0] - src_cx
            M[1, 2] += eye_center[1] - src_cy
            warped = cv2.warpAffine(face_img, M, (w, h),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(0, 0, 0, 0))

        alpha = warped[:, :, 3:4].astype(np.float32) / 255.0
        overlay[:] = np.clip(
            warped[:, :, :3].astype(np.float32) * alpha
            + overlay.astype(np.float32) * (1.0 - alpha),
            0, 255,
        ).astype(np.uint8)


def _apply_face_mosaic(frame, face_res, w, h, block=20):
    """감지된 얼굴 영역에 모자이크(픽셀화) 효과를 적용한다."""
    if not face_res.face_landmarks:
        return
    for _lf in face_res.face_landmarks:
        xs = [_lf[i].x * w for i in range(len(_lf))]
        ys = [_lf[i].y * h for i in range(len(_lf))]
        x1 = max(0,  int(min(xs)) - 15)
        y1 = max(0,  int(min(ys)) - 15)
        x2 = min(w,  int(max(xs)) + 15)
        y2 = min(h,  int(max(ys)) + 15)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        roi = frame[y1:y2, x1:x2]
        rh, rw = roi.shape[:2]
        small = cv2.resize(roi,
                           (max(1, rw // block), max(1, rh // block)),
                           interpolation=cv2.INTER_LINEAR)
        frame[y1:y2, x1:x2] = cv2.resize(small, (rw, rh),
                                           interpolation=cv2.INTER_NEAREST)


def _draw_cartoon_hands_hq(overlay, hand_res, w, h):
    """고품질 만화 손 렌더링.
    팜(손바닥) 볼록껍질 채우기 + 3단 셰이딩 + 손톱 광택 디테일.
    """
    if not hand_res.hand_landmarks:
        return

    # ── 색상 팔레트 (BGR) ──────────────────────────────────────────
    _OUTLINE    = (12,  12,  16)    # 검정 윤곽선
    _PALM_FILL  = (185, 205, 225)   # 손바닥 기본색
    _BONE_BASE  = (195, 212, 232)   # 뼈대 기본색
    _BONE_LIGHT = (222, 234, 250)   # 뼈대 하이라이트 중심선
    _JNT_SHADOW = (160, 180, 205)   # 관절 그림자
    _JNT_MID    = (205, 218, 238)   # 관절 중간색
    _JNT_LIGHT  = (232, 240, 254)   # 관절 하이라이트
    _NAIL_FILL  = (242, 247, 255)   # 손톱
    _NAIL_EDGE  = (110, 135, 170)   # 손톱 테두리
    _NAIL_SHINE = (255, 255, 255)   # 손톱 광택

    for hlms in hand_res.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hlms]
        if len(pts) < 21:
            continue

        # wrist(0) → middle_mcp(9) 거리로 스케일 계산
        _dist = float(np.linalg.norm(np.array(pts[0]) - np.array(pts[9])))
        _dist = max(_dist, 35.0)

        bone_w  = max(11, int(_dist * 0.23))
        joint_r = max(8,  int(_dist * 0.15))
        nail_r  = max(5,  int(_dist * 0.10))
        out_ext = max(6,  int(bone_w * 0.55))   # 윤곽선 추가 두께

        # ── Pass 1: 외곽 윤곽선 (뼈대 + 관절) ──────────────────────
        for s, e in _HAND_BONES:
            cv2.line(overlay, pts[s], pts[e],
                     _OUTLINE, bone_w + out_ext * 2, cv2.LINE_AA)
        for i in range(21):
            cv2.circle(overlay, pts[i], joint_r + out_ext, _OUTLINE, -1)

        # ── Pass 2: 손바닥 볼록껍질 채우기 ──────────────────────────
        palm_arr = np.array([pts[i] for i in [0, 1, 5, 9, 13, 17]], dtype=np.int32)
        hull = cv2.convexHull(palm_arr)
        cv2.fillConvexPoly(overlay, hull, _PALM_FILL, cv2.LINE_AA)
        cv2.polylines(overlay, [hull], True, _OUTLINE, 2, cv2.LINE_AA)

        # ── Pass 3: 뼈대 선 ──────────────────────────────────────────
        for s, e in _HAND_BONES:
            cv2.line(overlay, pts[s], pts[e], _BONE_BASE, bone_w, cv2.LINE_AA)
        # 뼈대 중심 하이라이트 (얇은 밝은 선)
        hl_w = max(2, bone_w // 3)
        for s, e in _HAND_BONES:
            cv2.line(overlay, pts[s], pts[e], _BONE_LIGHT, hl_w, cv2.LINE_AA)

        # ── Pass 4: 관절 구체 3단 셰이딩 ────────────────────────────
        for i in range(21):
            cv2.circle(overlay, pts[i], joint_r, _JNT_SHADOW, -1)
            cv2.circle(overlay, pts[i], int(joint_r * 0.75), _JNT_MID, -1)
            # 하이라이트 (약간 좌상단 오프셋)
            hx = pts[i][0] - max(1, joint_r // 5)
            hy = pts[i][1] - max(1, joint_r // 3)
            cv2.circle(overlay, (hx, hy), max(2, joint_r // 3), _JNT_LIGHT, -1)
        # 관절 테두리
        for i in [1,2,3,5,6,7,9,10,11,13,14,15,17,18,19]:
            cv2.circle(overlay, pts[i], joint_r, _OUTLINE, 1, cv2.LINE_AA)

        # ── Pass 5: 손가락 끝 손톱 ──────────────────────────────────
        for tip in [4, 8, 12, 16, 20]:
            cv2.circle(overlay, pts[tip], nail_r, _NAIL_FILL, -1)
            cv2.circle(overlay, pts[tip], nail_r, _NAIL_EDGE, 1, cv2.LINE_AA)
            # 광택 (좌상단)
            sx = pts[tip][0] - max(1, nail_r // 3)
            sy = pts[tip][1] - max(1, nail_r // 3)
            cv2.circle(overlay, (sx, sy), max(1, nail_r // 3), _NAIL_SHINE, -1)


def _draw_landmark_names(overlay, face_res, hand_res, pose_res,
                          w, h, show_face, show_body, show_hands):
    """랜드마크 포인트 이름을 overlay 이미지에 렌더링한다."""

    def _text(img, label, x, y, color):
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.rectangle(img, (x - 1, y - th - 2), (x + tw + 2, y + 2), (0, 0, 0), -1)
        cv2.putText(img, label, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    # ── 얼굴 주요 12포인트 이름 (감지된 모든 사람)
    if show_face and face_res.face_landmarks:
        _fc = (0, 230, 180)
        for _lf in face_res.face_landmarks:
            for _idx, _lbl in [
                (33,  "R.Eye.O"), (133, "R.Eye.I"), (473, "R.Iris"),
                (362, "L.Eye.I"), (263, "L.Eye.O"), (468, "L.Iris"),
                (168, "Nose.B"),  (4,   "Nose.T"),
                (61,  "Mouth.R"), (13,  "Mouth.U"), (291, "Mouth.L"), (14, "Mouth.D"),
            ]:
                if _idx < len(_lf):
                    _text(overlay, _lbl,
                          int(_lf[_idx].x * w) + 4,
                          int(_lf[_idx].y * h) - 4, _fc)

    # ── 포즈 주요 관절 이름 (감지된 모든 사람)
    if show_body and pose_res and pose_res.pose_landmarks:
        for _pl in pose_res.pose_landmarks:
            for _idx, _lbl in [
                (11, "L.Shldr"), (12, "R.Shldr"),
                (13, "L.Elbow"), (14, "R.Elbow"),
                (15, "L.Wrist"), (16, "R.Wrist"),
                (23, "L.Hip"),   (24, "R.Hip"),
                (25, "L.Knee"),  (26, "R.Knee"),
                (27, "L.Ankle"), (28, "R.Ankle"),
            ]:
                if _idx < len(_pl) and _pl[_idx].visibility > 0.3:
                    _col = (255, 160, 50) if _lbl.startswith("L.") else (50, 160, 255)
                    _text(overlay, _lbl,
                          int(_pl[_idx].x * w) + 7,
                          int(_pl[_idx].y * h) - 7, _col)

    # ── 손 주요 6포인트 이름
    if show_hands and hand_res.hand_landmarks:
        _hc = (255, 220, 100)
        for _hlms in hand_res.hand_landmarks:
            for _idx, _lbl in [
                (0, "Wrist"), (4, "Thumb"),
                (8, "Index"), (12, "Middle"),
                (16, "Ring"), (20, "Pinky"),
            ]:
                if _idx < len(_hlms):
                    _text(overlay, _lbl,
                          int(_hlms[_idx].x * w) + 4,
                          int(_hlms[_idx].y * h) - 4, _hc)


class VideoPanel:
    def __init__(self, parent: tk.Tk, path: str):
        self.win = tk.Toplevel(parent)
        self.win.title(f"PoseTracker — 영상 분석: {os.path.basename(path)}")
        self.win.geometry("1160x800")
        self.win.minsize(820, 720)
        self.win.configure(bg=BG_DARK)
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        self._video_path = path  # 내보내기 시 재사용

        # ── VideoCapture 초기화 ────────────────────────────────────────────
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            messagebox.showerror(
                "오류", f"파일을 열 수 없습니다:\n{path}", parent=parent,
            )
            self.win.destroy()
            return

        self._total_frames  = max(int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        self._fps           = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._vid_w         = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._vid_h         = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._current_frame = 0
        self._playing       = False
        self._dragging      = False
        self._drag_was_playing = False
        self._show_face   = tk.BooleanVar(value=False)
        self._show_body   = tk.BooleanVar(value=False)
        self._show_hands  = tk.BooleanVar(value=False)
        self._show_names  = tk.BooleanVar(value=False)
        self._show_mosaic         = tk.BooleanVar(value=False)
        self._show_cartoon_hands  = tk.BooleanVar(value=False)
        self._ai_quality_var      = tk.StringVar(value="fast")  # fast/balance/quality
        self._smooth_var = tk.IntVar(value=3)
        self._time_var           = tk.StringVar(value="00:00 / 00:00")
        self._export_status_var  = tk.StringVar(value="")
        self._face_det      = None
        self._hand_det      = None
        self._pose_det      = None
        self._after_id      = None
        self._photo         = None  # GC 방지
        self._det_skip      = 0
        self._det_cache     = None  # 재생 중 감지 결과 캐시
        self._face_img      = None  # BGRA numpy array (얼굴 이미지)
        self._face_img_pts  = None  # 소스 키포인트 (None = Affine 자동 모드)
        self._eye_y_var     = tk.IntVar(value=55)   # 눈 위치 Y (%)
        self._img_size_var  = tk.IntVar(value=100)  # 크기 배율 (%)

        self._build_ui()
        self._init_mediapipe()
        for _v in (self._show_face, self._show_body, self._show_hands, self._show_names,
                   self._show_mosaic):
            _v.trace_add("write", lambda *_: self._refresh_frame())
        # 첫 프레임 표시 (레이아웃 완료 후)
        self.win.after(100, lambda: self._seek_to(0))

    # ── UI 구성 ────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── 최상위: 좌우 분할 ─────────────────────────────────────────────
        body = tk.Frame(self.win, bg=BG_DARK)
        body.pack(fill=tk.BOTH, expand=True)

        # 우측 정보 패널 — 스크롤 가능
        _i_outer = tk.Frame(body, bg=BG_PANEL, width=210)
        _i_outer.pack(side=tk.RIGHT, fill=tk.Y)
        _i_outer.pack_propagate(False)

        _i_sb = tk.Scrollbar(_i_outer, orient="vertical")
        _i_sb.pack(side=tk.RIGHT, fill=tk.Y)

        _i_cv = tk.Canvas(
            _i_outer, bg=BG_PANEL,
            yscrollcommand=_i_sb.set,
            highlightthickness=0,
        )
        _i_cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        _i_sb.config(command=_i_cv.yview)

        info_panel = tk.Frame(_i_cv, bg=BG_PANEL)
        _i_win = _i_cv.create_window((0, 0), window=info_panel, anchor="nw")

        info_panel.bind("<Configure>",
                        lambda e: _i_cv.configure(scrollregion=_i_cv.bbox("all")))
        _i_cv.bind("<Configure>",
                   lambda e: _i_cv.itemconfig(_i_win, width=e.width))

        def _i_wheel(e):
            _i_cv.yview_scroll(int(-1 * (e.delta / 120)), "units")
        _i_cv.bind("<Enter>", lambda e: _i_cv.bind_all("<MouseWheel>", _i_wheel))
        _i_cv.bind("<Leave>", lambda e: _i_cv.unbind_all("<MouseWheel>"))

        self._build_info_panel(info_panel)

        # 좌측: 영상 + 타임라인 + 컨트롤
        left = tk.Frame(body, bg=BG_DARK)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 영상 캔버스
        self._canvas = tk.Canvas(
            left, bg="#000011",
            highlightthickness=1, highlightbackground="#333355",
        )
        self._canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 0))

        # 타임라인 캔버스
        self._tl = tk.Canvas(
            left, height=TL_H, bg=TL_BG,
            highlightthickness=0, cursor="sb_h_double_arrow",
        )
        self._tl.pack(fill=tk.X, padx=8, pady=(4, 0))
        self._tl.bind("<ButtonPress-1>",   self._tl_press)
        self._tl.bind("<B1-Motion>",       self._tl_drag)
        self._tl.bind("<ButtonRelease-1>", self._tl_release)
        self._tl.bind("<Configure>",       lambda _e: self._draw_timeline())

        # 컨트롤 바 — 재생 / 시간
        ctrl = tk.Frame(left, bg=BG_DARK)
        ctrl.pack(fill=tk.X, padx=8, pady=(4, 8))

        self._play_btn = tk.Button(
            ctrl, text="▶ 재생",
            font=("Segoe UI", 11, "bold"),
            bg=ACCENT, fg="white",
            activebackground="#3a6fee", activeforeground="white",
            relief=tk.FLAT, cursor="hand2",
            padx=14, pady=4,
            command=self._toggle_play,
        )
        self._play_btn.pack(side=tk.LEFT, padx=(0, 10))

        tk.Label(
            ctrl, textvariable=self._time_var,
            font=("Segoe UI", 11),
            fg=TEXT_W, bg=BG_DARK,
        ).pack(side=tk.LEFT)

    # ── 파일 정보 패널 ─────────────────────────────────────────────────────
    def _build_info_panel(self, parent):
        # 상단 accent 바
        tk.Frame(parent, bg=ACCENT, height=3).pack(fill=tk.X)

        # 헤더
        tk.Label(
            parent, text="파일 정보",
            font=("Segoe UI", 12, "bold"),
            fg=TEXT_W, bg=BG_PANEL, anchor="w",
        ).pack(fill=tk.X, padx=14, pady=(14, 4))

        tk.Frame(parent, bg="#2a2a4a", height=1).pack(fill=tk.X, padx=10, pady=(0, 6))

        def row(label: str, value: str, wrap: bool = False):
            tk.Label(
                parent, text=label,
                font=("Segoe UI", 8),
                fg=TEXT_G, bg=BG_PANEL, anchor="w",
            ).pack(fill=tk.X, padx=14, pady=(8, 0))
            tk.Label(
                parent, text=value,
                font=("Segoe UI", 10, "bold"),
                fg=TEXT_W, bg=BG_PANEL, anchor="w",
                wraplength=178, justify=tk.LEFT,
            ).pack(fill=tk.X, padx=14, pady=(1, 0))
            tk.Frame(parent, bg="#1e1e3a", height=1).pack(fill=tk.X, padx=10, pady=(6, 0))

        # ── 파일명 ──
        fname = os.path.basename(self._video_path)
        row("파일명", fname, wrap=True)

        # ── 경로 ──
        row("경로", self._video_path, wrap=True)

        # ── 용량 ──
        try:
            size = os.path.getsize(self._video_path)
            if size < 1024 ** 2:
                size_str = f"{size / 1024:.1f} KB"
            elif size < 1024 ** 3:
                size_str = f"{size / 1024 ** 2:.1f} MB"
            else:
                size_str = f"{size / 1024 ** 3:.2f} GB"
        except Exception:
            size_str = "—"
        row("용량", size_str)

        # ── 해상도 ──
        res = f"{self._vid_w} × {self._vid_h}" if self._vid_w and self._vid_h else "—"
        row("해상도", res)

        # ── 총 프레임 수 ──
        row("총 프레임", f"{self._total_frames:,} 프레임")

        # ── FPS ──
        row("FPS", f"{self._fps:.2f}")

        # ── 재생 시간 ──
        total_secs = int(self._total_frames / max(self._fps, 1))
        h, rem = divmod(total_secs, 3600)
        m, s   = divmod(rem, 60)
        dur = f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
        row("재생 시간", dur)

        # ── 섹션 구분 ──
        tk.Frame(parent, bg="#2a2a4a", height=1).pack(fill=tk.X, padx=10, pady=(14, 8))

        # ── 오버레이 ──
        tk.Label(
            parent, text="오버레이",
            font=("Segoe UI", 8),
            fg=TEXT_G, bg=BG_PANEL, anchor="w",
        ).pack(fill=tk.X, padx=14, pady=(0, 4))

        for _var, _lbl in [
            (self._show_face,  "얼굴  (눈·코·입)"),
            (self._show_body,  "몸  (몸통·팔·다리)"),
            (self._show_hands, "손  (손가락·손바닥)"),
        ]:
            tk.Checkbutton(
                parent, text=_lbl, variable=_var,
                font=("Segoe UI", 10),
                fg=TEXT_W, bg=BG_PANEL,
                selectcolor="#0f3460",
                activeforeground=TEXT_W, activebackground=BG_PANEL,
                anchor="w",
            ).pack(fill=tk.X, padx=10, pady=(0, 2))
        tk.Checkbutton(
            parent, text="  └ 만화 손 스타일 (내보내기 AI)",
            variable=self._show_cartoon_hands,
            font=("Segoe UI", 10),
            fg="#88ccff", bg=BG_PANEL,
            selectcolor="#0f3460",
            activeforeground="#88ccff", activebackground=BG_PANEL,
            anchor="w",
        ).pack(fill=tk.X, padx=10, pady=(0, 2))
        # ── AI 품질 라디오 버튼 ──
        _q_frame = tk.Frame(parent, bg=BG_PANEL)
        _q_frame.pack(fill=tk.X, padx=20, pady=(0, 6))
        for _lbl, _val in [("⚡ 빠름", "fast"), ("⚖ 균형", "balance"), ("🎨 고품질", "quality")]:
            tk.Radiobutton(
                _q_frame, text=_lbl, variable=self._ai_quality_var, value=_val,
                font=("Segoe UI", 9), fg="#88ccff", bg=BG_PANEL,
                selectcolor="#0f3460",
                activeforeground="#88ccff", activebackground=BG_PANEL,
            ).pack(side=tk.LEFT, padx=(0, 6))
        tk.Checkbutton(
            parent, text="랜드마크 이름",
            variable=self._show_names,
            font=("Segoe UI", 10),
            fg="#ffdd88", bg=BG_PANEL,
            selectcolor="#0f3460",
            activeforeground="#ffdd88", activebackground=BG_PANEL,
            anchor="w",
        ).pack(fill=tk.X, padx=10, pady=(4, 2))
        tk.Checkbutton(
            parent, text="얼굴 모자이크",
            variable=self._show_mosaic,
            font=("Segoe UI", 10),
            fg="#ff8888", bg=BG_PANEL,
            selectcolor="#0f3460",
            activeforeground="#ff8888", activebackground=BG_PANEL,
            anchor="w",
        ).pack(fill=tk.X, padx=10, pady=(2, 2))

        tk.Frame(parent, bg="#2a2a4a", height=1).pack(fill=tk.X, padx=10, pady=(4, 8))

        # ── AE 스무딩 ──
        tk.Label(parent, text="AE 스무딩",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14, pady=(0, 2))
        smooth_row = tk.Frame(parent, bg=BG_PANEL)
        smooth_row.pack(fill=tk.X, padx=10, pady=(0, 4))
        tk.Label(smooth_row, text="0", font=("Segoe UI", 9),
                 fg=TEXT_G, bg=BG_PANEL).pack(side=tk.LEFT)
        tk.Scale(
            smooth_row, from_=0, to=15, orient=tk.HORIZONTAL,
            variable=self._smooth_var, length=120,
            bg=BG_PANEL, fg=TEXT_W, troughcolor="#0f3460",
            highlightthickness=0, showvalue=True,
        ).pack(side=tk.LEFT, padx=2)
        tk.Label(smooth_row, text="15", font=("Segoe UI", 9),
                 fg=TEXT_G, bg=BG_PANEL).pack(side=tk.LEFT)

        tk.Frame(parent, bg="#2a2a4a", height=1).pack(fill=tk.X, padx=10, pady=(4, 8))

        # ── 얼굴 이미지 오버레이 ──
        tk.Label(parent, text="얼굴 이미지",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14, pady=(0, 4))
        self._face_img_btn = tk.Button(
            parent, text="🖼  이미지 로드",
            font=("Segoe UI", 10, "bold"),
            bg="#1e3a5f", fg=TEXT_W,
            activebackground="#2a4f80", activeforeground="white",
            relief=tk.FLAT, cursor="hand2",
            pady=6, anchor="w", padx=12,
            command=self._toggle_face_image,
        )
        self._face_img_btn.pack(fill=tk.X, padx=10, pady=(0, 2))
        self._face_img_lbl = tk.Label(
            parent, text="미선택",
            font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
            wraplength=178,
        )
        self._face_img_lbl.pack(fill=tk.X, padx=14, pady=(0, 2))
        # 그림/일러스트 조정 슬라이더
        tk.Label(parent, text="눈 위치 Y (%)",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        tk.Scale(parent, from_=10, to=90, orient=tk.HORIZONTAL,
                 variable=self._eye_y_var, length=160,
                 bg=BG_PANEL, fg=TEXT_W, troughcolor="#0f3460",
                 highlightthickness=0, showvalue=True,
                 ).pack(padx=10, pady=(0, 2))
        tk.Label(parent, text="크기 (%)",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        tk.Scale(parent, from_=30, to=300, orient=tk.HORIZONTAL,
                 variable=self._img_size_var, length=160,
                 bg=BG_PANEL, fg=TEXT_W, troughcolor="#0f3460",
                 highlightthickness=0, showvalue=True,
                 ).pack(padx=10, pady=(0, 4))

        tk.Frame(parent, bg="#2a2a4a", height=1).pack(fill=tk.X, padx=10, pady=(4, 8))

        # ── 내보내기 ──
        tk.Label(
            parent, text="내보내기",
            font=("Segoe UI", 8),
            fg=TEXT_G, bg=BG_PANEL, anchor="w",
        ).pack(fill=tk.X, padx=14, pady=(0, 6))

        def export_btn(text, bg, hover, cmd):
            b = tk.Button(
                parent, text=text,
                font=("Segoe UI", 10, "bold"),
                bg=bg, fg=TEXT_W,
                activebackground=hover, activeforeground="white",
                relief=tk.FLAT, cursor="hand2",
                pady=6, anchor="w", padx=12,
                command=cmd,
            )
            b.pack(fill=tk.X, padx=10, pady=(0, 4))
            return b

        self._json_btn  = export_btn("⬇  JSON 내보내기",  "#1e3a5f", "#2a4f80", self._export_json)
        self._ae_btn    = export_btn("⬇  AE 내보내기",    "#1e3a5f", "#2a4f80", self._export_ae)
        self._video_btn = export_btn("🎬  영상 저장",      "#2a1f5f", "#3d2e80", self._export_video)

        tk.Label(
            parent, textvariable=self._export_status_var,
            font=("Segoe UI", 9),
            fg="#4aff9e", bg=BG_PANEL,
            wraplength=178, justify=tk.CENTER,
        ).pack(fill=tk.X, padx=10, pady=(4, 0))

    # ── MediaPipe 초기화 ───────────────────────────────────────────────────
    def _init_mediapipe(self):
        try:
            face_opts = mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL),
                running_mode=RunningMode.IMAGE,
                num_faces=MAX_PERSONS,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            hand_opts = mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL),
                running_mode=RunningMode.IMAGE,
                num_hands=MAX_PERSONS * 2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            pose_opts = mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL),
                running_mode=RunningMode.IMAGE,
                num_poses=MAX_PERSONS,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._face_det = mp_vision.FaceLandmarker.create_from_options(face_opts)
            self._hand_det = mp_vision.HandLandmarker.create_from_options(hand_opts)
            self._pose_det = mp_vision.PoseLandmarker.create_from_options(pose_opts)
        except Exception as e:
            print(f"[MediaPipe init error] {e}")
            self._face_det = None
            self._hand_det = None
            self._pose_det = None

    # ── 타임라인 렌더링 ────────────────────────────────────────────────────
    def _draw_timeline(self):
        self._tl.update_idletasks()
        w = self._tl.winfo_width()
        if w <= 1:
            return
        mid = TL_H // 2
        self._tl.delete("all")
        self._tl.create_rectangle(0, 0, w, TL_H, fill=TL_BG, outline="")
        # 트랙 배경선
        self._tl.create_line(8, mid, w - 8, mid, fill="#333355", width=3)
        # 진행 선
        px = self._frame_to_x(self._current_frame, w)
        if px > 8:
            self._tl.create_line(8, mid, px, mid, fill=ACCENT, width=3)
        # 플레이헤드 (흰색 원, 반지름 8)
        self._tl.create_oval(px - 8, mid - 8, px + 8, mid + 8, fill="white", outline="")

    def _frame_to_x(self, frame: int, canvas_w: int) -> int:
        frac = frame / max(self._total_frames - 1, 1)
        return int(8 + frac * (canvas_w - 16))

    def _x_to_frame(self, x: int) -> int:
        w = self._tl.winfo_width()
        frac = (x - 8) / max(w - 16, 1)
        frac = max(0.0, min(1.0, frac))
        return int(frac * (self._total_frames - 1))

    # ── 타임라인 이벤트 ────────────────────────────────────────────────────
    def _tl_press(self, event):
        self._drag_was_playing = self._playing
        if self._playing:
            self._playing = False
            if self._after_id:
                self.win.after_cancel(self._after_id)
                self._after_id = None
            self._play_btn.config(text="▶ 재생")
        self._dragging = True
        self._seek_to(self._x_to_frame(event.x))

    def _tl_drag(self, event):
        self._seek_to(self._x_to_frame(event.x))

    def _tl_release(self, event):
        self._dragging = False
        if self._drag_was_playing:
            self._playing = True
            self._play_btn.config(text="⏸ 일시정지")
            self._schedule_next()

    # ── 재생 제어 ──────────────────────────────────────────────────────────
    def _toggle_play(self):
        if self._playing:
            self._playing = False
            if self._after_id:
                self.win.after_cancel(self._after_id)
                self._after_id = None
            self._play_btn.config(text="▶ 재생")
        else:
            if self._current_frame >= self._total_frames - 1:
                self._seek_to(0)
            self._playing = True
            self._play_btn.config(text="⏸ 일시정지")
            self._schedule_next()

    def _schedule_next(self):
        if not self._playing or self._dragging:
            return
        delay = max(1, int(1000 / max(self._fps, 1)))
        self._after_id = self.win.after(delay, self._next_frame)

    def _next_frame(self):
        if not self._playing or self._dragging:
            return
        ret, frame = self._cap.read()
        if not ret:
            self._playing = False
            self._play_btn.config(text="▶ 재생")
            return
        # CAP_PROP_POS_FRAMES는 다음 읽을 프레임 인덱스 → -1 하면 현재 프레임
        self._current_frame = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        self._display_frame(frame, playback=True)
        self._draw_timeline()
        self._update_time()
        self._schedule_next()

    def _seek_to(self, frame_num: int):
        frame_num = max(0, min(frame_num, self._total_frames - 1))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self._cap.read()
        if ret:
            self._current_frame = frame_num
            self._det_cache = None  # 시크 시 캐시 초기화
            self._display_frame(frame)
            self._draw_timeline()
            self._update_time()

    def _refresh_frame(self):
        """오버레이 토글 시 현재 프레임 재표시"""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._current_frame)
        ret, frame = self._cap.read()
        if ret:
            self._display_frame(frame)

    # ── 프레임 표시 ────────────────────────────────────────────────────────
    def _display_frame(self, bgr, playback=False):
        if (self._show_face.get() or self._show_body.get() or self._show_hands.get()
                or self._face_img is not None or self._show_mosaic.get()):
            bgr = self._apply_overlay(bgr, playback=playback)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        self._canvas.update_idletasks()
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw <= 1:
            cw = 840
        if ch <= 1:
            ch = 480

        img = Image.fromarray(rgb)
        img = img.resize((cw, ch), Image.BILINEAR)
        self._photo = ImageTk.PhotoImage(img)
        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)

    def _apply_overlay(self, bgr, playback=False, ext_hand_res=None):
        """오버레이 렌더링.
        ext_hand_res: 외부에서 제공된 hand 감지 결과 (AI 렌더링 모드).
                      None이면 내부에서 감지. 값이 있으면 만화 손 그리기를 건너뜀.
        """
        if self._face_det is None or self._hand_det is None:
            return bgr
        overlay = bgr.copy()

        # 재생 중 최적화: 2프레임마다 감지, 나머지는 캐시 재사용
        self._det_skip += 1 if playback else 0
        use_cache = playback and (self._det_skip % 2 == 0) and self._det_cache is not None

        if not use_cache:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h_px, w_px = bgr.shape[:2]
            # 추론 해상도 축소 (최대 640px 너비)
            _sc = min(1.0, 640 / max(w_px, 1))
            if _sc < 0.99:
                _iw, _ih = int(w_px * _sc), int(h_px * _sc)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                                  data=cv2.resize(rgb, (_iw, _ih)))
            else:
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            try:
                face_res = self._face_det.detect(mp_img)
                hand_res = self._hand_det.detect(mp_img) if ext_hand_res is None else ext_hand_res
                pose_res = self._pose_det.detect(mp_img) if self._pose_det else None
            except Exception as e:
                print(f"[detect error] {e}")
                return overlay
            if playback:
                self._det_cache = (face_res, hand_res, pose_res)
        else:
            face_res, hand_res, pose_res = self._det_cache
            if ext_hand_res is not None:
                hand_res = ext_hand_res

        _oh, _ow = overlay.shape[:2]

        # ── 얼굴 모자이크 (가장 먼저 적용)
        if self._show_mosaic.get():
            _apply_face_mosaic(overlay, face_res, _ow, _oh)

        # ── 포즈 스켈레톤 (모든 감지된 사람, 얼굴/손 아래 레이어로 먼저 그리기)
        if self._show_body.get() and pose_res and pose_res.pose_landmarks:
            _SKEL = [(11,12),(11,23),(12,24),(23,24),
                     (11,13),(13,15),(12,14),(14,16),
                     (23,25),(25,27),(24,26),(26,28)]
            for _pidx, _pl in enumerate(pose_res.pose_landmarks):
                _pc = PERSON_COLORS[_pidx % len(PERSON_COLORS)]
                for _s, _e in _SKEL:
                    if (_s < len(_pl) and _e < len(_pl)
                            and _pl[_s].visibility > 0.3
                            and _pl[_e].visibility > 0.3):
                        cv2.line(overlay,
                                 (int(_pl[_s].x*_ow), int(_pl[_s].y*_oh)),
                                 (int(_pl[_e].x*_ow), int(_pl[_e].y*_oh)),
                                 _pc, 2)
                for _i in [11,12,13,14,15,16,23,24,25,26,27,28]:
                    if _i < len(_pl) and _pl[_i].visibility > 0.3:
                        cv2.circle(overlay,
                                   (int(_pl[_i].x*_ow), int(_pl[_i].y*_oh)),
                                   6, _pc, -1)
                # 사람 번호 레이블
                if 0 < len(_pl) and _pl[0].visibility > 0.3:
                    _lx = max(5, int(_pl[0].x*_ow) - 12)
                    _ly = max(20, int(_pl[0].y*_oh) - 20)
                    cv2.putText(overlay, f"P{_pidx+1}", (_lx, _ly),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, _pc, 2, cv2.LINE_AA)

        # ── 얼굴 (모든 감지된 사람)
        if self._show_face.get() and face_res.face_landmarks:
            _nc = (0, 230, 180)
            for _lf in face_res.face_landmarks:
                mp_draw.draw_landmarks(
                    overlay, _lf,
                    FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style(),
                )
                for _s, _e in [(168,6),(6,197),(197,195),(195,5),(5,4),
                               (4,1),(1,19),(98,97),(97,2),(2,326),(326,327)]:
                    if _s < len(_lf) and _e < len(_lf):
                        cv2.line(overlay,
                                 (int(_lf[_s].x*_ow), int(_lf[_s].y*_oh)),
                                 (int(_lf[_e].x*_ow), int(_lf[_e].y*_oh)),
                                 _nc, 1)
                for _i in [1,2,4,5,6,19,97,98,168,195,197,326,327]:
                    if _i < len(_lf):
                        cv2.circle(overlay,
                                   (int(_lf[_i].x*_ow), int(_lf[_i].y*_oh)),
                                   2, _nc, -1)

        # ── 손 (만화 손 스타일은 내보내기 시에만 AI로 적용, 실시간 뷰에서는 미표시)
        if hand_res.hand_landmarks:
            if self._show_hands.get():
                for hlms in hand_res.hand_landmarks:
                    mp_draw.draw_landmarks(
                        overlay, hlms,
                        HandLandmarksConnections.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp_styles.get_default_hand_connections_style(),
                    )
        # ── 랜드마크 이름
        if self._show_names.get():
            _draw_landmark_names(overlay, face_res, hand_res, pose_res,
                                 _ow, _oh,
                                 self._show_face.get(),
                                 self._show_body.get(),
                                 self._show_hands.get())

        # ── 얼굴 이미지 오버레이
        _fi = self._face_img
        _fp = self._face_img_pts
        if _fi is not None:
            _apply_face_img_overlay(overlay, face_res, _ow, _oh, _fi, _fp,
                                    eye_y_pct=self._eye_y_var.get(),
                                    size_pct=self._img_size_var.get())

        return overlay

    # ── 시간 업데이트 ──────────────────────────────────────────────────────
    def _update_time(self):
        def fmt(f):
            secs = int(f / max(self._fps, 1))
            return f"{secs // 60:02d}:{secs % 60:02d}"
        self._time_var.set(f"{fmt(self._current_frame)} / {fmt(self._total_frames - 1)}")

    # ── 내보내기 ───────────────────────────────────────────────────────────
    def _export_json(self):
        self._do_export("json")

    def _export_ae(self):
        self._do_export("ae")

    def _do_export(self, mode: str):
        if self._face_det is None or self._hand_det is None:
            messagebox.showerror("오류", "MediaPipe 초기화에 실패했습니다.", parent=self.win)
            return

        if mode == "json":
            save_path = filedialog.asksaveasfilename(
                parent=self.win,
                title="JSON 저장 위치 선택",
                defaultextension=".json",
                filetypes=[("JSON 파일", "*.json"), ("모든 파일", "*.*")],
            )
            if not save_path:
                return
        else:  # ae
            save_path = filedialog.askdirectory(
                parent=self.win,
                title="AE 키프레임 저장 폴더 선택",
            )
            if not save_path:
                return

        # 재생 일시정지
        was_playing = self._playing
        if self._playing:
            self._playing = False
            if self._after_id:
                self.win.after_cancel(self._after_id)
                self._after_id = None
            self._play_btn.config(text="▶ 재생")

        self._set_export_btns(tk.DISABLED)
        self._export_status_var.set("분석 시작...")
        inc_face  = self._show_face.get()
        inc_body  = self._show_body.get()
        inc_hands = self._show_hands.get()
        smooth    = self._smooth_var.get()

        def _run():
            frames_data, info = self._process_all_frames()
            if not frames_data:
                def _fail():
                    self._export_status_var.set("분석 실패 — 프레임 없음")
                    self._set_export_btns(tk.NORMAL)
                self.win.after(0, _fail)
                return

            try:
                if mode == "json":
                    export_json(frames_data, info, save_path,
                                include_face=inc_face, include_body=inc_body, include_hands=inc_hands)
                    msg = f"JSON 저장 완료!\n{save_path}"
                else:
                    export_ae_keyframes(frames_data, info, save_path,
                                        include_face=inc_face, include_body=inc_body, include_hands=inc_hands,
                                        smooth_radius=smooth)
                    msg = f"AE 키프레임 저장 완료!\n{save_path}/"
            except Exception as e:
                msg = None
                err = str(e)
                def _err():
                    self._export_status_var.set("내보내기 오류")
                    self._set_export_btns(tk.NORMAL)
                    messagebox.showerror("내보내기 오류", err, parent=self.win)
                self.win.after(0, _err)
                return

            final_msg = msg
            def _done():
                self._export_status_var.set("저장 완료!")
                self._set_export_btns(tk.NORMAL)
                messagebox.showinfo("완료", final_msg, parent=self.win)
                self._export_status_var.set("")
                if was_playing:
                    self._playing = True
                    self._play_btn.config(text="⏸ 일시정지")
                    self._schedule_next()
            self.win.after(0, _done)

        threading.Thread(target=_run, daemon=True).start()

    def _set_export_btns(self, state):
        self._json_btn.config(state=state)
        self._ae_btn.config(state=state)
        self._video_btn.config(state=state)

    def _export_video(self):
        save_path = filedialog.asksaveasfilename(
            parent=self.win,
            title="영상 저장 위치 선택",
            defaultextension=".mp4",
            filetypes=[("MP4 파일", "*.mp4"), ("모든 파일", "*.*")],
        )
        if not save_path:
            return

        use_ai_hand = self._show_cartoon_hands.get()
        with_overlay = (self._show_face.get() or self._show_body.get()
                        or self._show_hands.get() or use_ai_hand)

        # AI 만화 손 모드: diffusers 확인 + SD 모델 파일 선택
        ai_sd_model = None
        if use_ai_hand:
            try:
                import diffusers  # noqa: F401
            except ImportError:
                messagebox.showerror(
                    "패키지 필요",
                    "AI 만화 손 렌더링에 diffusers 패키지가 필요합니다.\n\n"
                    "pip install diffusers transformers accelerate\n\n"
                    "설치 후 다시 시도하세요.",
                    parent=self.win,
                )
                return

            # SD 모델 파일 선택 (로컬 safetensors/ckpt)
            messagebox.showinfo(
                "SD 모델 파일 선택",
                "Stable Diffusion 모델 파일(.safetensors 또는 .ckpt)을 선택하세요.\n\n"
                "Civitai 등에서 다운받은 만화/애니 스타일 모델을 추천합니다.\n"
                "(예: anything-v4.safetensors, CounterfeitV30.safetensors 등)\n\n"
                "SD 1.5 기반 모델이어야 합니다.",
                parent=self.win,
            )
            ai_sd_model = filedialog.askopenfilename(
                parent=self.win,
                title="SD 모델 파일 선택 (.safetensors / .ckpt)",
                filetypes=[
                    ("Stable Diffusion 모델", "*.safetensors *.ckpt"),
                    ("모든 파일", "*.*"),
                ],
            )
            if not ai_sd_model:
                return  # 취소

            _quality = self._ai_quality_var.get()
            _quality_info = {
                "fast":    "⚡ 빠름 — LCM 4스텝 (~0.25초/프레임, 5배 빠름)",
                "balance": "⚖ 균형 — DPM++ 10스텝 (~0.6초/프레임)",
                "quality": "🎨 고품질 — DPM++ 20스텝 (~1.2초/프레임)",
            }
            ok = messagebox.askyesno(
                "AI 렌더링 확인",
                f"선택한 모델: {os.path.basename(ai_sd_model)}\n"
                f"품질 모드: {_quality_info.get(_quality, _quality)}\n\n"
                "• ControlNet 모델 자동 다운로드 (최초 1회, ~1.4GB)\n"
                "• ⚡ 빠름 선택 시 LCM-LoRA 추가 다운로드 (~200MB)\n"
                "• 렌더링 중 창을 닫지 마세요\n\n"
                "계속하시겠습니까?",
                parent=self.win,
            )
            if not ok:
                return

        # 오버레이 모드인데 MediaPipe 없으면 경고
        if with_overlay and self._face_det is None:
            messagebox.showwarning(
                "경고",
                "MediaPipe 초기화 실패 — 랜드마크 없이 원본 영상으로 저장합니다.",
                parent=self.win,
            )
            with_overlay = False

        was_playing = self._playing
        if self._playing:
            self._playing = False
            if self._after_id:
                self.win.after_cancel(self._after_id)
                self._after_id = None
            self._play_btn.config(text="▶ 재생")

        self._set_export_btns(tk.DISABLED)
        if use_ai_hand:
            label = "AI 만화 손 렌더링 준비 중..."
        elif with_overlay:
            label = "오버레이 렌더링 중..."
        else:
            label = "영상 저장 중..."
        self._export_status_var.set(label)

        def _run():
            try:
                self._save_video_frames(save_path, with_overlay,
                                        use_ai_hand=use_ai_hand,
                                        ai_sd_model=ai_sd_model,
                                        ai_quality=self._ai_quality_var.get())
                msg = f"영상 저장 완료!\n{save_path}"
                def _done():
                    self._export_status_var.set("저장 완료!")
                    self._set_export_btns(tk.NORMAL)
                    messagebox.showinfo("완료", msg, parent=self.win)
                    self._export_status_var.set("")
                    if was_playing:
                        self._playing = True
                        self._play_btn.config(text="⏸ 일시정지")
                        self._schedule_next()
                self.win.after(0, _done)
            except Exception as e:
                err = str(e)
                def _err():
                    self._export_status_var.set("저장 오류")
                    self._set_export_btns(tk.NORMAL)
                    messagebox.showerror("저장 오류", err, parent=self.win)
                self.win.after(0, _err)

        threading.Thread(target=_run, daemon=True).start()

    def _save_video_frames(self, save_path: str, with_overlay: bool,
                           use_ai_hand: bool = False, ai_sd_model: str = None,
                           ai_quality: str = "fast"):
        cap = cv2.VideoCapture(self._video_path)
        if not cap.isOpened():
            raise RuntimeError(f"영상을 열 수 없습니다: {self._video_path}")

        total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"VideoWriter를 열 수 없습니다: {save_path}")

        # ── AI 만화 손 모드 초기화 ──────────────────────────────────────
        ai          = None
        ai_hand_det = None
        if use_ai_hand:
            try:
                from .cartoon_hand_ai import CartoonHandAI
            except ImportError:
                from cartoon_hand_ai import CartoonHandAI

            def _cb(msg):
                self.win.after(0, lambda m=msg: self._export_status_var.set(m))

            _preset   = CartoonHandAI.QUALITY_PRESETS.get(
                            ai_quality, CartoonHandAI.QUALITY_PRESETS["fast"])
            _use_lcm  = _preset["use_lcm"]
            _steps    = _preset["steps"]
            _guidance = _preset["guidance"]

            ai = CartoonHandAI(sd_model=ai_sd_model)
            ai.load(progress_cb=_cb, use_lcm=_use_lcm)

            # 별도 HandLandmarker (풀 해상도 감지)
            _h_opts = mp_python.BaseOptions(model_asset_path=HAND_MODEL)
            _h_run  = mp_vision.HandLandmarkerOptions(
                base_options=_h_opts,
                running_mode=RunningMode.IMAGE,
                num_hands=MAX_PERSONS * 2,
            )
            ai_hand_det = mp_vision.HandLandmarker.create_from_options(_h_run)

        import time
        _t0 = time.time()

        try:
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if with_overlay:
                    if use_ai_hand:
                        # 1) 손 감지 (풀 해상도)
                        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        try:
                            hand_res = ai_hand_det.detect(mp_img)
                        except Exception:
                            hand_res = None

                        # 2) 얼굴/포즈 오버레이 (만화 손 그리기 스킵)
                        frame = self._apply_overlay(frame, ext_hand_res=hand_res)

                        # 3) AI 만화 손 렌더링
                        if hand_res and hand_res.hand_landmarks:
                            frame = ai.render_frame(
                                frame, hand_res,
                                steps=_steps, guidance=_guidance)
                    else:
                        frame = self._apply_overlay(frame)

                writer.write(frame)
                idx += 1

                if idx % 5 == 0:
                    pct      = int(idx / total * 100)
                    elapsed  = time.time() - _t0
                    per_fr   = elapsed / max(idx, 1)
                    if use_ai_hand:
                        label = f"AI 렌더링 중... {idx}/{total} ({per_fr:.1f}초/프레임)"
                    elif with_overlay:
                        label = f"렌더링 중... {pct}%"
                    else:
                        label = f"저장 중... {pct}%"
                    self.win.after(0, lambda l=label: self._export_status_var.set(l))
        finally:
            writer.release()
            cap.release()
            if ai_hand_det is not None:
                try:
                    ai_hand_det.close()
                except Exception:
                    pass

    def _process_all_frames(self):
        """영상 전체 프레임을 MediaPipe로 처리 → (List[FrameData], VideoInfo) 반환"""
        cap = cv2.VideoCapture(self._video_path)
        if not cap.isOpened():
            return [], None

        total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames_data = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            try:
                face_res = self._face_det.detect(mp_img)
                hand_res = self._hand_det.detect(mp_img)
                pose_res = self._pose_det.detect(mp_img) if self._pose_det else None
            except Exception as e:
                print(f"[frame {idx} detect error] {e}")
                idx += 1
                continue

            fd = FrameData(index=idx, timestamp=idx / fps)
            fd.persons = _build_persons(face_res, hand_res, pose_res, w, h)
            frames_data.append(fd)
            idx += 1

            if idx % 15 == 0:
                pct = int(idx / total * 100)
                self.win.after(0, lambda p=pct: self._export_status_var.set(f"분석중... {p}%"))

        cap.release()
        info = VideoInfo(width=w, height=h, fps=fps, total_frames=len(frames_data))
        return frames_data, info

    # ── 얼굴 이미지 로드/제거 ──────────────────────────────────────────────
    def _toggle_face_image(self):
        if self._face_img is not None:
            self._face_img = None
            self._face_img_pts = None
            self._face_img_lbl.config(text="미선택")
            self._face_img_btn.config(text="🖼  이미지 로드")
            self._det_cache = None
            self._refresh_frame()
        else:
            self._load_face_image()

    def _load_face_image(self):
        if self._face_det is None:
            messagebox.showerror("오류", "MediaPipe 초기화에 실패했습니다.", parent=self.win)
            return

        path = filedialog.askopenfilename(
            parent=self.win,
            title="얼굴 이미지 선택",
            filetypes=[("이미지 파일", "*.png *.jpg *.jpeg *.bmp"), ("모든 파일", "*.*")],
        )
        if not path:
            return

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            messagebox.showerror("오류", f"이미지를 열 수 없습니다:\n{path}", parent=self.win)
            return

        h, w = img.shape[:2]
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 3:
            tmp = np.zeros((h, w, 4), dtype=np.uint8)
            tmp[:, :, :3] = img
            tmp[:, :, 3] = 255
            img = tmp

        rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        try:
            mp_img_src = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            face_res = self._face_det.detect(mp_img_src)
        except Exception as e:
            messagebox.showerror("오류", f"얼굴 감지 실패:\n{e}", parent=self.win)
            return

        # 얼굴 감지 성공 → 정밀 모드 (Homography)
        # 감지 실패 (그림/일러스트) → 자동 모드 (Affine)
        if (face_res.face_landmarks
                and len(face_res.face_landmarks[0]) > max(_FACE_IMG_KPT)):
            lf = face_res.face_landmarks[0]
            src_pts = np.float32([[lf[i].x * w, lf[i].y * h] for i in _FACE_IMG_KPT])
            mode_text = " [정밀]"
        else:
            src_pts = None  # Affine 자동 모드
            mode_text = " [자동]"

        self._face_img = img.copy()
        self._face_img_pts = src_pts
        self._det_cache = None
        self._face_img_lbl.config(text=os.path.basename(path) + mode_text)
        self._face_img_btn.config(text="× 이미지 제거")
        self._refresh_frame()

    # ── 종료 ──────────────────────────────────────────────────────────────
    def _on_close(self):
        self._playing = False
        if self._after_id:
            self.win.after_cancel(self._after_id)
            self._after_id = None
        if self._cap:
            self._cap.release()
            self._cap = None
        if self._face_det:
            self._face_det.close()
            self._face_det = None
        if self._hand_det:
            self._hand_det.close()
            self._hand_det = None
        if self._pose_det:
            self._pose_det.close()
            self._pose_det = None
        self.win.destroy()
