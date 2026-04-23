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

try:
    try:
        from .anime_converter import AnimeGANConverter, apply_anime_to_person
    except ImportError:
        from anime_converter import AnimeGANConverter, apply_anime_to_person
    _ANIME_AVAILABLE = True
except Exception:
    _ANIME_AVAILABLE = False
    AnimeGANConverter      = None
    apply_anime_to_person  = None


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

def _adaptive_ema_update(state: dict, key: str, value: float,
                          catch_scale: float = 30.0) -> float:
    """변화량에 따라 alpha를 자동 조정하는 Adaptive EMA.
    정지 시 → base_alpha 유지(강한 보정), 빠른 이동 시 → alpha→1.0(즉각 추적)."""
    base_alpha = state.get('alpha', 0.15)
    prev = state.get(key)
    if prev is None:
        state[key] = value
        return value
    delta = abs(value - prev)
    dynamic_alpha = min(1.0, base_alpha + delta / catch_scale)
    state[key] = dynamic_alpha * value + (1.0 - dynamic_alpha) * prev
    return state[key]


def _compute_mar(face_res, w: int, h: int) -> float:
    """Mouth Aspect Ratio: 윗입술(13)~아랫입술(14) / 눈 간격(33-263)
    입 닫힘 ~0.04, 입 벌림 ~0.15+"""
    if not face_res.face_landmarks:
        return 0.0
    lf = face_res.face_landmarks[0]
    if len(lf) < 292:
        return 0.0
    mouth_gap = abs(lf[14].y - lf[13].y) * h
    eye_dist  = abs(lf[263].x - lf[33].x) * w
    return mouth_gap / eye_dist if eye_dist > 1 else 0.0


def _apply_face_img_overlay(overlay, face_res, w, h, face_img, face_img_pts,
                             eye_y_pct=55, eye_x_pct=50, size_pct=100,
                             ema_state: dict | None = None):
    """로드된 얼굴 이미지(BGRA)를 감지된 얼굴 위에 합성한다.
    face_img_pts is not None → Homography 정밀 모드 (실제 얼굴 사진)
    face_img_pts is None     → Affine 자동 모드  (일러스트/그림)
    ema_state: {'face_h', 'eye_cx', 'eye_cy', 'angle', 'alpha'} — EMA 상태 dict
    """
    if not face_res.face_landmarks:
        if ema_state is not None:
            for _k in ('face_h', 'eye_cx', 'eye_cy', 'angle'):
                ema_state[_k] = None
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
            raw_face_h = max(ys) - min(ys)
            raw_eye_cx, raw_eye_cy = float(eye_center[0]), float(eye_center[1])
            raw_angle = angle

            # ── EMA 평활화 적용 (떨림 제거) ──
            if ema_state is not None:
                face_h_px  = _adaptive_ema_update(ema_state, 'face_h', raw_face_h, catch_scale=30.0)
                _ec_x      = _adaptive_ema_update(ema_state, 'eye_cx', raw_eye_cx, catch_scale=40.0)
                _ec_y      = _adaptive_ema_update(ema_state, 'eye_cy', raw_eye_cy, catch_scale=40.0)
                angle      = _adaptive_ema_update(ema_state, 'angle',  raw_angle,  catch_scale=10.0)
                eye_center = (_ec_x, _ec_y)
            else:
                face_h_px = raw_face_h

            scale = face_h_px * (size_pct / 100.0) / (img_h * 0.8)
            src_cx = img_w * (eye_x_pct / 100.0)
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
        self._show_anime_var   = tk.BooleanVar(value=False)
        self._anime_style_var  = tk.StringVar(value="animegan")
        self._anime_bg_var     = tk.StringVar(value="original")
        self._anime_model_path = ""
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
        self._face_img_open     = None   # BGRA (입 벌림 이미지)
        self._face_img_open_pts = None
        self._mouth_thr_var     = tk.DoubleVar(value=0.12)
        self._eye_y_var     = tk.IntVar(value=55)   # 눈 위치 Y (%)
        self._eye_x_var     = tk.IntVar(value=50)   # 눈 위치 X (%)
        self._img_size_var  = tk.IntVar(value=100)  # 크기 배율 (%)
        self._ema_smooth_var = tk.IntVar(value=85)  # 떨림 보정 강도 (0~95)
        self._face_img_ema: dict = {
            'face_h': None, 'eye_cx': None, 'eye_cy': None,
            'angle': None, 'alpha': 0.15,
        }

        self._build_ui()
        self._on_ema_smooth_change()  # 슬라이더 초기값 → EMA alpha 동기화
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
        # 접기/펴기 섹션 목록 (- 키 토글용)
        self._panel_sections: list = []

        # 상단 accent 바
        tk.Frame(parent, bg=ACCENT, height=3).pack(fill=tk.X)

        # 헤더 (클릭 토글)
        _info_open = tk.BooleanVar(value=True)
        hdr = tk.Frame(parent, bg=BG_PANEL, cursor="hand2")
        hdr.pack(fill=tk.X)
        hdr_lbl = tk.Label(
            hdr, text="▼  파일 정보",
            font=("Segoe UI", 12, "bold"),
            fg=TEXT_W, bg=BG_PANEL, anchor="w",
        )
        hdr_lbl.pack(fill=tk.X, padx=14, pady=(14, 4))

        _sep = tk.Frame(parent, bg="#2a2a4a", height=1)
        _sep.pack(fill=tk.X, padx=10, pady=(0, 6))

        # 접힐 컨테이너
        body = tk.Frame(parent, bg=BG_PANEL)
        body.pack(fill=tk.X)

        def _toggle(_e=None):
            if _info_open.get():
                body.pack_forget()
                hdr_lbl.config(text="▶  파일 정보")
                _info_open.set(False)
            else:
                body.pack(fill=tk.X, after=_sep)   # _sep 바로 아래 원위치 복원
                hdr_lbl.config(text="▼  파일 정보")
                _info_open.set(True)

        hdr.bind("<Button-1>", _toggle)
        hdr_lbl.bind("<Button-1>", _toggle)
        self._panel_sections.append((_info_open, _toggle))

        def row(label: str, value: str, wrap: bool = False):
            tk.Label(
                body, text=label,
                font=("Segoe UI", 8),
                fg=TEXT_G, bg=BG_PANEL, anchor="w",
            ).pack(fill=tk.X, padx=14, pady=(8, 0))
            tk.Label(
                body, text=value,
                font=("Segoe UI", 10, "bold"),
                fg=TEXT_W, bg=BG_PANEL, anchor="w",
                wraplength=178, justify=tk.LEFT,
            ).pack(fill=tk.X, padx=14, pady=(1, 0))
            tk.Frame(body, bg="#1e1e3a", height=1).pack(fill=tk.X, padx=10, pady=(6, 0))

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

        # ── 오버레이 (접기/펴기) ──
        _ov_open = tk.BooleanVar(value=True)
        _ov_hdr = tk.Frame(parent, bg=BG_PANEL, cursor="hand2")
        _ov_hdr.pack(fill=tk.X)
        _ov_lbl = tk.Label(
            _ov_hdr, text="▼  오버레이",
            font=("Segoe UI", 10, "bold"),
            fg=TEXT_G, bg=BG_PANEL, anchor="w",
        )
        _ov_lbl.pack(fill=tk.X, padx=14, pady=(0, 4))
        _ov_sep = tk.Frame(parent, bg="#1e1e3a", height=1)
        _ov_sep.pack(fill=tk.X, padx=10, pady=(0, 4))
        _ov_body = tk.Frame(parent, bg=BG_PANEL)
        _ov_body.pack(fill=tk.X)

        def _toggle_overlay(_e=None):
            if _ov_open.get():
                _ov_body.pack_forget()
                _ov_lbl.config(text="▶  오버레이")
                _ov_open.set(False)
            else:
                _ov_body.pack(fill=tk.X, after=_ov_sep)
                _ov_lbl.config(text="▼  오버레이")
                _ov_open.set(True)

        _ov_hdr.bind("<Button-1>", _toggle_overlay)
        _ov_lbl.bind("<Button-1>", _toggle_overlay)
        self._panel_sections.append((_ov_open, _toggle_overlay))

        for _var, _lbl in [
            (self._show_face,  "얼굴  (눈·코·입)"),
            (self._show_body,  "몸  (몸통·팔·다리)"),
            (self._show_hands, "손  (손가락·손바닥)"),
        ]:
            tk.Checkbutton(
                _ov_body, text=_lbl, variable=_var,
                font=("Segoe UI", 10),
                fg=TEXT_W, bg=BG_PANEL,
                selectcolor="#0f3460",
                activeforeground=TEXT_W, activebackground=BG_PANEL,
                anchor="w",
            ).pack(fill=tk.X, padx=10, pady=(0, 2))
        tk.Checkbutton(
            _ov_body, text="랜드마크 이름",
            variable=self._show_names,
            font=("Segoe UI", 10),
            fg="#ffdd88", bg=BG_PANEL,
            selectcolor="#0f3460",
            activeforeground="#ffdd88", activebackground=BG_PANEL,
            anchor="w",
        ).pack(fill=tk.X, padx=10, pady=(4, 2))
        tk.Checkbutton(
            _ov_body, text="얼굴 모자이크",
            variable=self._show_mosaic,
            font=("Segoe UI", 10),
            fg="#ff8888", bg=BG_PANEL,
            selectcolor="#0f3460",
            activeforeground="#ff8888", activebackground=BG_PANEL,
            anchor="w",
        ).pack(fill=tk.X, padx=10, pady=(2, 2))

        # ── 애니화 (접기/펴기) ───────────────────────────────────────────────
        tk.Frame(parent, bg="#2a2a4a", height=1).pack(fill=tk.X, padx=10, pady=(8, 4))

        _an_open = tk.BooleanVar(value=True)
        _an_hdr = tk.Frame(parent, bg=BG_PANEL, cursor="hand2")
        _an_hdr.pack(fill=tk.X)
        _an_lbl = tk.Label(
            _an_hdr, text="▼  애니화",
            font=("Segoe UI", 10, "bold"),
            fg="#88ddff", bg=BG_PANEL, anchor="w",
        )
        _an_lbl.pack(fill=tk.X, padx=14, pady=(0, 4))
        _an_sep = tk.Frame(parent, bg="#1e1e3a", height=1)
        _an_sep.pack(fill=tk.X, padx=10, pady=(0, 4))
        _an_body = tk.Frame(parent, bg=BG_PANEL)
        _an_body.pack(fill=tk.X)

        def _toggle_anime(_e=None):
            if _an_open.get():
                _an_body.pack_forget()
                _an_lbl.config(text="▶  애니화")
                _an_open.set(False)
            else:
                _an_body.pack(fill=tk.X, after=_an_sep)
                _an_lbl.config(text="▼  애니화")
                _an_open.set(True)

        _an_hdr.bind("<Button-1>", _toggle_anime)
        _an_lbl.bind("<Button-1>", _toggle_anime)
        self._panel_sections.append((_an_open, _toggle_anime))

        tk.Checkbutton(
            _an_body, text="🎨 내보내기 전용",
            variable=self._show_anime_var,
            font=("Segoe UI", 10, "bold"),
            fg="#88ddff", bg=BG_PANEL,
            selectcolor="#0f3460",
            activeforeground="#88ddff", activebackground=BG_PANEL,
            anchor="w",
        ).pack(fill=tk.X, padx=10, pady=(0, 4))

        # 스타일
        tk.Label(_an_body, text="  스타일",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        _sf = tk.Frame(_an_body, bg=BG_PANEL)
        _sf.pack(fill=tk.X, padx=20, pady=(0, 4))
        for _sv, _sl in [("animegan", "AnimeGAN"), ("opencv", "OpenCV")]:
            tk.Radiobutton(
                _sf, text=_sl, variable=self._anime_style_var, value=_sv,
                font=("Segoe UI", 9), fg=TEXT_W, bg=BG_PANEL,
                selectcolor="#0f3460",
                activeforeground=TEXT_W, activebackground=BG_PANEL,
                command=self._on_anime_style_change,
            ).pack(side=tk.LEFT, padx=(0, 8))

        # 배경
        tk.Label(_an_body, text="  배경",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        _bf = tk.Frame(_an_body, bg=BG_PANEL)
        _bf.pack(fill=tk.X, padx=20, pady=(0, 4))
        for _bv, _bl in [("original", "원본"), ("blur", "블러"), ("solid", "단색")]:
            tk.Radiobutton(
                _bf, text=_bl, variable=self._anime_bg_var, value=_bv,
                font=("Segoe UI", 9), fg=TEXT_W, bg=BG_PANEL,
                selectcolor="#0f3460",
                activeforeground=TEXT_W, activebackground=BG_PANEL,
            ).pack(side=tk.LEFT, padx=(0, 4))

        # ONNX 모델 선택 (AnimeGAN용)
        self._anime_model_btn = tk.Button(
            _an_body, text="  ONNX 모델 선택",
            font=("Segoe UI", 9),
            bg="#1e3a5f", fg=TEXT_W,
            activebackground="#2a4f80", activeforeground="white",
            relief=tk.FLAT, cursor="hand2",
            pady=4, anchor="w", padx=12,
            command=self._select_anime_model,
        )
        self._anime_model_btn.pack(fill=tk.X, padx=20, pady=(0, 2))
        self._anime_model_lbl = tk.Label(
            _an_body, text="미선택 (OpenCV로 대체)",
            font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
            wraplength=160,
        )
        self._anime_model_lbl.pack(fill=tk.X, padx=20, pady=(0, 4))

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

        # ── 얼굴 이미지 (접기/펴기) ──
        _fi_open = tk.BooleanVar(value=True)
        _fi_hdr = tk.Frame(parent, bg=BG_PANEL, cursor="hand2")
        _fi_hdr.pack(fill=tk.X)
        _fi_lbl = tk.Label(
            _fi_hdr, text="▼  얼굴 이미지",
            font=("Segoe UI", 10, "bold"),
            fg=TEXT_G, bg=BG_PANEL, anchor="w",
        )
        _fi_lbl.pack(fill=tk.X, padx=14, pady=(0, 4))
        _fi_sep = tk.Frame(parent, bg="#1e1e3a", height=1)
        _fi_sep.pack(fill=tk.X, padx=10, pady=(0, 4))
        _fi_body = tk.Frame(parent, bg=BG_PANEL)
        _fi_body.pack(fill=tk.X)

        def _toggle_face_img(_e=None):
            if _fi_open.get():
                _fi_body.pack_forget()
                _fi_lbl.config(text="▶  얼굴 이미지")
                _fi_open.set(False)
            else:
                _fi_body.pack(fill=tk.X, after=_fi_sep)
                _fi_lbl.config(text="▼  얼굴 이미지")
                _fi_open.set(True)

        _fi_hdr.bind("<Button-1>", _toggle_face_img)
        _fi_lbl.bind("<Button-1>", _toggle_face_img)
        self._panel_sections.append((_fi_open, _toggle_face_img))

        self._face_img_btn = tk.Button(
            _fi_body, text="🖼  이미지 로드",
            font=("Segoe UI", 10, "bold"),
            bg="#1e3a5f", fg=TEXT_W,
            activebackground="#2a4f80", activeforeground="white",
            relief=tk.FLAT, cursor="hand2",
            pady=6, anchor="w", padx=12,
            command=self._toggle_face_image,
        )
        self._face_img_btn.pack(fill=tk.X, padx=10, pady=(0, 2))
        self._face_img_lbl = tk.Label(
            _fi_body, text="미선택",
            font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
            wraplength=178,
        )
        self._face_img_lbl.pack(fill=tk.X, padx=14, pady=(0, 2))
        tk.Label(_fi_body, text="눈 위치 Y (%)",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        tk.Scale(_fi_body, from_=10, to=90, orient=tk.HORIZONTAL,
                 variable=self._eye_y_var, length=160,
                 bg=BG_PANEL, fg=TEXT_W, troughcolor="#0f3460",
                 highlightthickness=0, showvalue=True,
                 ).pack(padx=10, pady=(0, 2))
        tk.Label(_fi_body, text="눈 위치 X (%)",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        tk.Scale(_fi_body, from_=10, to=90, orient=tk.HORIZONTAL,
                 variable=self._eye_x_var, length=160,
                 bg=BG_PANEL, fg=TEXT_W, troughcolor="#0f3460",
                 highlightthickness=0, showvalue=True,
                 ).pack(padx=10, pady=(0, 2))
        tk.Label(_fi_body, text="크기 (%)",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        tk.Scale(_fi_body, from_=30, to=300, orient=tk.HORIZONTAL,
                 variable=self._img_size_var, length=160,
                 bg=BG_PANEL, fg=TEXT_W, troughcolor="#0f3460",
                 highlightthickness=0, showvalue=True,
                 ).pack(padx=10, pady=(0, 2))
        tk.Label(_fi_body, text="떨림 보정 (0=없음  →  95=최대)",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        tk.Scale(_fi_body, from_=0, to=95, orient=tk.HORIZONTAL,
                 variable=self._ema_smooth_var, length=160,
                 bg=BG_PANEL, fg="#88ddff", troughcolor="#0f3460",
                 highlightthickness=0, showvalue=True,
                 command=self._on_ema_smooth_change,
                 ).pack(padx=10, pady=(0, 4))

        tk.Frame(_fi_body, bg="#2a2a4a", height=1).pack(fill=tk.X, padx=10, pady=(2, 6))

        self._face_img_open_btn = tk.Button(
            _fi_body, text="🖼  입 벌림 이미지 로드",
            font=("Segoe UI", 10, "bold"),
            bg="#1e3a5f", fg=TEXT_W,
            activebackground="#2a4f80", activeforeground="white",
            relief=tk.FLAT, cursor="hand2",
            pady=6, anchor="w", padx=12,
            command=self._toggle_face_image_open,
        )
        self._face_img_open_btn.pack(fill=tk.X, padx=10, pady=(0, 2))
        self._face_img_open_lbl = tk.Label(
            _fi_body, text="미선택",
            font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
            wraplength=178,
        )
        self._face_img_open_lbl.pack(fill=tk.X, padx=14, pady=(0, 2))
        tk.Label(_fi_body, text="전환 임계값 (MAR)",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        self._mouth_thr_scale = tk.Scale(
            _fi_body, from_=0.02, to=0.30, resolution=0.01, orient=tk.HORIZONTAL,
            variable=self._mouth_thr_var, length=160,
            bg=BG_PANEL, fg="#ffcc88", troughcolor="#0f3460",
            highlightthickness=0, showvalue=True,
            state=tk.DISABLED,
        )
        self._mouth_thr_scale.pack(padx=10, pady=(0, 4))

        tk.Frame(parent, bg="#2a2a4a", height=1).pack(fill=tk.X, padx=10, pady=(4, 8))

        self.win.bind('-', self._toggle_all_sections)
        self.win.bind('<space>', lambda e: self._toggle_play())

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

    def _apply_overlay(self, bgr, playback=False):
        """오버레이 렌더링."""
        if self._face_det is None or self._hand_det is None:
            return bgr
        overlay = bgr.copy()

        # 재생 중 최적화: 2프레임마다 감지, 나머지는 캐시 재사용
        self._det_skip += 1 if playback else 0
        use_cache = playback and (self._det_skip % 2 == 0) and self._det_cache is not None

        if not use_cache:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h_px, w_px = bgr.shape[:2]
            # 추론 해상도 축소 (최대 변 640px 기준 — 세로 영상 대응)
            _sc = min(1.0, 640 / max(w_px, h_px, 1))
            if _sc < 0.99:
                _iw, _ih = int(w_px * _sc), int(h_px * _sc)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                                  data=cv2.resize(rgb, (_iw, _ih)))
            else:
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            try:
                face_res = self._face_det.detect(mp_img)
                hand_res = self._hand_det.detect(mp_img)
                pose_res = self._pose_det.detect(mp_img) if self._pose_det else None
            except Exception as e:
                print(f"[detect error] {e}")
                return overlay
            if playback:
                self._det_cache = (face_res, hand_res, pose_res)
        else:
            face_res, hand_res, pose_res = self._det_cache

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

        # ── 손 랜드마크 / 만화 손
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
            if self._face_img_open is not None:
                _mar = _compute_mar(face_res, _ow, _oh)
                if _mar >= self._mouth_thr_var.get():
                    _fi = self._face_img_open
                    _fp = self._face_img_open_pts
            _apply_face_img_overlay(overlay, face_res, _ow, _oh, _fi, _fp,
                                    eye_y_pct=self._eye_y_var.get(),
                                    eye_x_pct=self._eye_x_var.get(),
                                    size_pct=self._img_size_var.get(),
                                    ema_state=self._face_img_ema)

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

    def _select_anime_model(self):
        """AnimeGAN ONNX 모델 파일 선택."""
        path = filedialog.askopenfilename(
            parent=self.win,
            title="AnimeGAN ONNX 모델 선택",
            filetypes=[("ONNX 모델", "*.onnx"), ("모든 파일", "*.*")],
        )
        if path:
            self._anime_model_path = path
            self._anime_model_lbl.config(text=os.path.basename(path))

    def _toggle_all_sections(self, _e=None):
        """- 키: 하나라도 열려 있으면 전체 접기, 모두 닫혀 있으면 전체 펴기."""
        if any(s.get() for s, _ in self._panel_sections):
            for s, fn in self._panel_sections:
                if s.get():
                    fn()
        else:
            for s, fn in self._panel_sections:
                if not s.get():
                    fn()

    def _on_anime_style_change(self):
        """스타일 라디오 변경 시 모델 버튼 상태 업데이트."""
        is_gan = self._anime_style_var.get() == "animegan"
        self._anime_model_btn.config(state=tk.NORMAL if is_gan else tk.DISABLED)
        if not is_gan:
            self._anime_model_lbl.config(text="OpenCV 사용 (모델 불필요)")

    def _export_video(self):
        save_path = filedialog.asksaveasfilename(
            parent=self.win,
            title="영상 저장 위치 선택",
            defaultextension=".mp4",
            filetypes=[("MP4 파일", "*.mp4"), ("모든 파일", "*.*")],
        )
        if not save_path:
            return

        with_overlay = (self._show_face.get() or self._show_body.get()
                        or self._show_hands.get()
                        or self._show_mosaic.get() or self._face_img is not None)
        with_anime   = self._show_anime_var.get() and _ANIME_AVAILABLE

        # 오버레이/애니화 모드인데 MediaPipe 없으면 경고
        if (with_overlay or with_anime) and self._face_det is None:
            messagebox.showwarning(
                "경고",
                "MediaPipe 초기화 실패 — 랜드마크 없이 원본 영상으로 저장합니다.",
                parent=self.win,
            )
            with_overlay = False
            with_anime   = False

        # AnimeGAN 선택 시 모델 없으면 OpenCV로 대체 안내
        if with_anime and self._anime_style_var.get() == "animegan" and not self._anime_model_path:
            if not messagebox.askyesno(
                "모델 미선택",
                "AnimeGAN ONNX 모델이 선택되지 않았습니다.\n"
                "OpenCV 방식으로 대체하여 저장하시겠습니까?",
                parent=self.win,
            ):
                return

        was_playing = self._playing
        if self._playing:
            self._playing = False
            if self._after_id:
                self.win.after_cancel(self._after_id)
                self._after_id = None
            self._play_btn.config(text="▶ 재생")

        self._set_export_btns(tk.DISABLED)
        if with_anime:
            self._export_status_var.set("애니화 렌더링 중...")
        elif with_overlay:
            self._export_status_var.set("오버레이 렌더링 중...")
        else:
            self._export_status_var.set("영상 저장 중...")

        def _run():
            try:
                self._save_video_frames(save_path, with_overlay, with_anime)
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
                           with_anime: bool = False):
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

        import time
        _t0 = time.time()

        # ── 애니화 사전 준비 ────────────────────────────────────────────────
        _anime_converter = None
        _anime_bg_mode   = "original"
        _anime_style     = "opencv"
        if with_anime:
            _anime_style   = self._anime_style_var.get()
            _anime_bg_mode = self._anime_bg_var.get()
            if _anime_style == "animegan" and self._anime_model_path:
                try:
                    _anime_converter = AnimeGANConverter()
                    _anime_converter.load(self._anime_model_path)
                except Exception as _e:
                    print(f"[AnimeGAN load error] {_e} — OpenCV로 대체합니다.")
                    _anime_style = "opencv"

        try:
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # ── 애니화 (사람 마스크 + 스타일 변환) ─────────────────────
                if with_anime:
                    _fr = _hr = _pr = None
                    try:
                        _fh, _fw = frame.shape[:2]
                        _asc = min(1.0, 640 / max(_fw, _fh, 1))
                        _rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        if _asc < 0.99:
                            _rgb = cv2.resize(_rgb,
                                              (int(_fw*_asc), int(_fh*_asc)))
                        _mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                                           data=_rgb)
                        if self._face_det:
                            _fr = self._face_det.detect(_mp_img)
                        if self._hand_det:
                            _hr = self._hand_det.detect(_mp_img)
                        if self._pose_det:
                            _pr = self._pose_det.detect(_mp_img)
                    except Exception:
                        pass
                    frame = apply_anime_to_person(
                        frame, _pr, _fr, _hr,
                        style=_anime_style,
                        bg_mode=_anime_bg_mode,
                        converter=_anime_converter,
                    )

                # ── 오버레이 (랜드마크/모자이크 등) ─────────────────────────
                if with_overlay:
                    frame = self._apply_overlay(frame)

                writer.write(frame)
                idx += 1

                if idx % 5 == 0:
                    elapsed = time.time() - _t0
                    fps_est = idx / max(elapsed, 0.001)
                    if with_anime:
                        label = f"애니화 중... {idx}/{total}  ({fps_est:.1f}fps)"
                    elif with_overlay:
                        label = f"렌더링 중... {idx}/{total}"
                    else:
                        label = f"저장 중... {int(idx/total*100)}%"
                    self.win.after(0, lambda l=label: self._export_status_var.set(l))
        finally:
            writer.release()
            cap.release()

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

    # ── EMA 떨림 보정 슬라이더 콜백 ──────────────────────────────────────
    def _on_ema_smooth_change(self, _=None):
        """슬라이더 값(0~95) → alpha(1.00~0.05) 변환 후 EMA 상태에 반영."""
        v = self._ema_smooth_var.get()
        self._face_img_ema['alpha'] = 1.0 - (v / 100.0)

    # ── 얼굴 이미지 로드/제거 ──────────────────────────────────────────────
    def _toggle_face_image(self):
        if self._face_img is not None:
            self._face_img = None
            self._face_img_pts = None
            for _k in ('face_h', 'eye_cx', 'eye_cy', 'angle'):
                self._face_img_ema[_k] = None
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

    # ── 입 벌림 이미지 로드/제거 ──────────────────────────────────────────
    def _toggle_face_image_open(self):
        if self._face_img_open is not None:
            self._face_img_open = None
            self._face_img_open_pts = None
            self._face_img_open_lbl.config(text="미선택")
            self._face_img_open_btn.config(text="🖼  입 벌림 이미지 로드")
            self._mouth_thr_scale.config(state=tk.DISABLED)
            self._det_cache = None
            self._refresh_frame()
        else:
            self._load_face_image_open()

    def _load_face_image_open(self):
        if self._face_det is None:
            messagebox.showerror("오류", "MediaPipe 초기화에 실패했습니다.", parent=self.win)
            return
        path = filedialog.askopenfilename(
            parent=self.win,
            title="입 벌림 이미지 선택",
            filetypes=[("이미지", "*.png *.jpg *.jpeg *.webp *.bmp"), ("모든 파일", "*.*")],
        )
        if not path:
            return
        try:
            from PIL import Image as PilImg
            pil_img = PilImg.open(path).convert("RGBA")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGRA)
        except Exception as e:
            messagebox.showerror("오류", f"이미지 로드 실패:\n{e}", parent=self.win)
            return

        h, w = img.shape[:2]
        if w > 1024:
            scale = 1024 / w
            img = cv2.resize(img, (1024, int(h * scale)), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]

        mask = img[:, :, 3]
        if mask.min() == 255:
            tmp = img.copy()
            white = (tmp[:, :, 0] > 240) & (tmp[:, :, 1] > 240) & (tmp[:, :, 2] > 240)
            tmp[white, 3] = 0
            img = tmp

        rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        try:
            mp_img_src = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            face_res = self._face_det.detect(mp_img_src)
        except Exception as e:
            messagebox.showerror("오류", f"얼굴 감지 실패:\n{e}", parent=self.win)
            return

        if (face_res.face_landmarks
                and len(face_res.face_landmarks[0]) > max(_FACE_IMG_KPT)):
            lf = face_res.face_landmarks[0]
            src_pts = np.float32([[lf[i].x * w, lf[i].y * h] for i in _FACE_IMG_KPT])
            mode_text = " [정밀]"
        else:
            src_pts = None
            mode_text = " [자동]"

        self._face_img_open = img.copy()
        self._face_img_open_pts = src_pts
        self._det_cache = None
        self._face_img_open_lbl.config(text=os.path.basename(path) + mode_text)
        self._face_img_open_btn.config(text="× 입 벌림 제거")
        self._mouth_thr_scale.config(state=tk.NORMAL)
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
