"""
camera_panel.py — 카메라 추적 UI 패널

기능:
  - 웹캠 실시간 피드 + 얼굴/손 오버레이
  - 30 / 60 fps 선택
  - 녹화 → 저장 경로 선택 → 영상 저장 + AE 키프레임 자동 내보내기
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import cv2
import mediapipe as mp
import os
import tempfile
import shutil
import numpy as np

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode
from mediapipe.tasks.python.vision import drawing_utils as mp_draw
from mediapipe.tasks.python.vision import drawing_styles as mp_styles
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections

_BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACE_MODEL  = os.path.join(_BASE, "models", "face_landmarker.task")
HAND_MODEL  = os.path.join(_BASE, "models", "hand_landmarker.task")
POSE_MODEL  = os.path.join(_BASE, "models", "pose_landmarker_full.task")

try:
    from PIL import Image, ImageTk
except ImportError:
    raise ImportError("Pillow가 필요합니다: pip install Pillow")

try:
    from .tracker import (Tracker, FrameData, VideoInfo, PersonData,
                          _extract_face, _extract_hand, _extract_pose,
                          _build_persons, MAX_PERSONS, PERSON_COLORS)
    from .exporter import export_json, export_ae_keyframes
    from . import insightface_detector as _if_det_mod
except ImportError:
    from tracker import (Tracker, FrameData, VideoInfo, PersonData,
                         _extract_face, _extract_hand, _extract_pose,
                         _build_persons, MAX_PERSONS, PERSON_COLORS)
    from exporter import export_json, export_ae_keyframes
    import insightface_detector as _if_det_mod

try:
    try:
        from .puppet_pin import (PuppetPins, SegmentCache,
                                  build_segment_cache, apply_puppet_warp,
                                  pins_degenerate)
    except ImportError:
        from puppet_pin import (PuppetPins, SegmentCache,
                                 build_segment_cache, apply_puppet_warp,
                                 pins_degenerate)
    _PUPPET_AVAILABLE = True
except Exception:
    _PUPPET_AVAILABLE = False
    PuppetPins = SegmentCache = build_segment_cache = apply_puppet_warp = pins_degenerate = None


# ── 테마 색상 ──────────────────────────────────────────────────────────────
BG_DARK   = "#1a1a2e"
BG_PANEL  = "#16213e"
BG_CTRL   = "#0f3460"
ACCENT    = "#4a7fff"
ACCENT_R  = "#e94560"
TEXT_W    = "#e0e0ff"
TEXT_G    = "#8888aa"

CANVAS_W  = 840
CANVAS_H  = 560
CTRL_W    = 230

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
        # 얼굴 없으면 EMA 상태 리셋 (다음 감지 때 cold-start)
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
            # 얼굴 바운딩박스 높이 → 스케일 (이미지의 80%가 얼굴 영역이라 가정)
            if hasattr(_lf, 'bbox'):
                # InsightFace: bbox로 얼굴 높이 계산
                raw_face_h = float(_lf.bbox[3] - _lf.bbox[1])
            else:
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

            if face_h_px <= 0:
                continue
            scale = face_h_px * (size_pct / 100.0) / (img_h * 0.8)
            # 소스 이미지에서 눈 중심 위치
            src_cx = img_w * (eye_x_pct / 100.0)
            src_cy = img_h * (eye_y_pct / 100.0)
            # Affine: 소스 눈 중심 → 화면 눈 중심, 회전 + 스케일
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


def _apply_arm_img_overlay(overlay, pose_res, w, h, arm_img,
                            anchor_y_pct=50, anchor_x_pct=50, size_pct=100,
                            ema_state: dict | None = None,
                            arm_pins=None, arm_seg_cache=None, side='right'):
    """로드된 팔 이미지(BGRA)를 팔꿈치 위에 Affine 합성.
    arm_pins/arm_seg_cache가 있으면 Puppet Pin 모드 (2 or 3-세그먼트),
    없으면 기존 Legacy(단일 Affine) 모드.
    side='right': 랜드마크 12/14/16/20, 'left': 11/13/15/19"""
    if not pose_res.pose_landmarks:
        if ema_state is not None:
            for _k in ('elbow_x', 'elbow_y', 'angle', 'arm_len',
                       'shldr_x', 'shldr_y', 'wrist_x', 'wrist_y',
                       'hand_x', 'hand_y'):
                ema_state[_k] = None
        return
    if side == 'right':
        shldr_idx, elbow_idx, wrist_idx, hand_idx = 12, 14, 16, 20
    else:
        shldr_idx, elbow_idx, wrist_idx, hand_idx = 11, 13, 15, 19
    img_h, img_w = arm_img.shape[:2]
    for _pl in pose_res.pose_landmarks:
        if len(_pl) <= max(shldr_idx, elbow_idx, wrist_idx):
            continue
        shoulder = _pl[shldr_idx]
        elbow    = _pl[elbow_idx]
        wrist    = _pl[wrist_idx]
        if shoulder.visibility < 0.3 or elbow.visibility < 0.3 or wrist.visibility < 0.3:
            continue
        raw_ex   = elbow.x * w
        raw_ey   = elbow.y * h
        raw_sx   = shoulder.x * w
        raw_sy   = shoulder.y * h
        raw_wx   = wrist.x * w
        raw_wy   = wrist.y * h
        raw_ang  = float(np.degrees(np.arctan2(
            wrist.y - shoulder.y, wrist.x - shoulder.x)))
        raw_len  = float(np.hypot(
            (elbow.x - shoulder.x) * w, (elbow.y - shoulder.y) * h))
        if ema_state is not None:
            ex  = _adaptive_ema_update(ema_state, 'elbow_x', raw_ex,  40.0)
            ey  = _adaptive_ema_update(ema_state, 'elbow_y', raw_ey,  40.0)
            sx  = _adaptive_ema_update(ema_state, 'shldr_x', raw_sx,  40.0)
            sy  = _adaptive_ema_update(ema_state, 'shldr_y', raw_sy,  40.0)
            wx  = _adaptive_ema_update(ema_state, 'wrist_x', raw_wx,  40.0)
            wy  = _adaptive_ema_update(ema_state, 'wrist_y', raw_wy,  40.0)
            ang = _adaptive_ema_update(ema_state, 'angle',   raw_ang, 10.0)
            aln = _adaptive_ema_update(ema_state, 'arm_len', raw_len, 30.0)
        else:
            ex, ey = raw_ex, raw_ey
            sx, sy = raw_sx, raw_sy
            wx, wy = raw_wx, raw_wy
            ang, aln = raw_ang, raw_len

        if arm_pins is not None and arm_seg_cache is not None and _PUPPET_AVAILABLE:
            # ── Puppet Pin 모드 (2 or 3-세그먼트) ──
            vid_hand = None
            if arm_pins.img_hand is not None and len(_pl) > hand_idx:
                hand_lm = _pl[hand_idx]
                if hand_lm.visibility >= 0.2:
                    raw_hx = hand_lm.x * w
                    raw_hy = hand_lm.y * h
                    if ema_state is not None:
                        hx = _adaptive_ema_update(ema_state, 'hand_x', raw_hx, 40.0)
                        hy = _adaptive_ema_update(ema_state, 'hand_y', raw_hy, 40.0)
                    else:
                        hx, hy = raw_hx, raw_hy
                    vid_hand = (hx, hy)
            warped = apply_puppet_warp(
                arm_seg_cache, (sx, sy), (ex, ey), (wx, wy), w, h,
                vid_hand=vid_hand, size_pct=float(size_pct))
        else:
            # ── Legacy 모드: 단일 Affine (기존 동작) ──
            scale  = aln * (size_pct / 100.0) / (img_h * 0.8)
            src_cx = img_w * (anchor_x_pct / 100.0)
            src_cy = img_h * (anchor_y_pct / 100.0)
            M = cv2.getRotationMatrix2D((src_cx, src_cy), -ang, scale)
            M[0, 2] += ex - src_cx
            M[1, 2] += ey - src_cy
            warped = cv2.warpAffine(arm_img, M, (w, h),
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
        if hasattr(_lf, 'bbox'):
            # InsightFace: bbox 직접 사용 (픽셀 좌표)
            x1 = max(0, _lf.bbox[0] - 15)
            y1 = max(0, _lf.bbox[1] - 15)
            x2 = min(w, _lf.bbox[2] + 15)
            y2 = min(h, _lf.bbox[3] + 15)
        else:
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


class CameraPanel:
    def __init__(self, parent: tk.Tk):
        self.win = tk.Toplevel(parent)
        self.win.title("PoseTracker — 카메라 추적")
        self.win.geometry(f"{CANVAS_W + CTRL_W + 36}x{CANVAS_H + 24}")
        self.win.minsize(900, 500)
        self.win.configure(bg=BG_DARK)
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── 상태 ──────────────────────────────────────────────────────────
        self._running    = False
        self._recording  = False
        self._fps        = 30          # 캡처 스레드에서 사용할 정수값
        self._fps_var    = tk.IntVar(value=30)
        self._show_face   = tk.BooleanVar(value=True)
        self._show_body   = tk.BooleanVar(value=True)
        self._show_hands  = tk.BooleanVar(value=True)
        self._show_names  = tk.BooleanVar(value=False)
        self._show_mosaic         = tk.BooleanVar(value=False)
        self._smooth_var  = tk.IntVar(value=3)
        self._status_var = tk.StringVar(value="대기중")
        self._frame_q: queue.Queue = queue.Queue(maxsize=2)
        self._frames_data: list[FrameData] = []
        self._cap         = None
        self._cap_thread  = None
        self._is_file_mode = False
        self._writer     = None        # cv2.VideoWriter
        self._tmp_path   = None
        self._after_id   = None
        self._photo      = None        # GC 방지용 PhotoImage 참조
        self._cam_w      = 0
        self._cam_h      = 0
        self._tracker       = Tracker()
        self._face_img      = None   # BGRA numpy array (얼굴 이미지)
        self._face_img_pts  = None   # 소스 키포인트 (None = Affine 자동 모드)
        self._face_img_open     = None   # BGRA (입 벌림 이미지)
        self._face_img_open_pts = None
        self._mouth_thr_var     = tk.DoubleVar(value=0.12)
        self._face_conf_var     = tk.DoubleVar(value=0.5)   # 얼굴 감지 신뢰도 임계값
        # EMA 평활화 상태 (얼굴 이미지 오버레이 떨림 제거)
        self._ema_smooth_var = tk.IntVar(value=85)   # 떨림 보정 강도 (0~95 → α=1.00~0.05)
        self._face_img_ema: dict = {
            'face_h': None, 'eye_cx': None, 'eye_cy': None,
            'angle': None, 'alpha': 0.15,
        }
        self._eye_y_var     = tk.IntVar(value=55)   # 눈 위치 Y (%)
        self._eye_x_var     = tk.IntVar(value=50)   # 눈 위치 X (%)
        self._img_size_var  = tk.IntVar(value=100)  # 크기 배율 (%)

        # ── 오른팔 이미지 오버레이 상태
        self._arm_img        = None   # BGRA numpy array
        self._arm_y_var      = tk.IntVar(value=50)   # 앵커 Y (%)
        self._arm_x_var      = tk.IntVar(value=50)   # 앵커 X (%)
        self._arm_size_var   = tk.IntVar(value=100)  # 크기 배율 (%)
        self._arm_smooth_var = tk.IntVar(value=85)   # 떨림 보정 (0~95)
        self._arm_img_ema    = {
            'elbow_x': None, 'elbow_y': None,
            'shldr_x': None, 'shldr_y': None,
            'wrist_x': None, 'wrist_y': None,
            'hand_x':  None, 'hand_y':  None,
            'angle': None, 'arm_len': None, 'alpha': 0.15,
        }
        # Puppet Pin 상태 (오른팔)
        self._arm_pins       = None   # PuppetPins | None
        self._arm_seg_cache  = None   # SegmentCache | None
        self._arm_pin_btn    = None   # 피벗 설정 버튼 참조
        self._arm_pin_lbl    = None   # 피벗 상태 레이블 참조
        self._pin_popup      = None   # 중복 팝업 방지

        # ── 왼팔 이미지 오버레이 상태
        self._arm_img_l       = None   # BGRA numpy array
        self._arm_img_ema_l   = {
            'elbow_x': None, 'elbow_y': None,
            'shldr_x': None, 'shldr_y': None,
            'wrist_x': None, 'wrist_y': None,
            'hand_x':  None, 'hand_y':  None,
            'angle': None, 'arm_len': None, 'alpha': 0.15,
        }
        self._arm_pins_l      = None
        self._arm_seg_cache_l = None
        self._arm_img_btn_l   = None   # 로드 버튼 참조
        self._arm_img_lbl_l   = None   # 파일명 레이블 참조
        self._arm_pin_btn_l   = None   # 피벗 설정 버튼 참조
        self._arm_pin_lbl_l   = None   # 피벗 상태 레이블 참조

        self._build_ui()
        self._on_ema_smooth_change()   # 슬라이더 초기값 → EMA alpha 동기화
        self._on_arm_smooth_change()   # arm EMA alpha 동기화
        # 패널 열리면 자동으로 카메라 시작
        self.win.after(200, self._start_camera)

    # ── UI 구성 ────────────────────────────────────────────────────────────
    def _build_ui(self):
        # 좌측: 카메라 캔버스
        left = tk.Frame(self.win, bg=BG_DARK)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(12, 6), pady=12)

        self._canvas = tk.Canvas(
            left, bg="#000011",
            highlightthickness=1, highlightbackground="#333355",
        )
        self._canvas.pack(fill=tk.BOTH, expand=True)
        self._draw_placeholder()

        # 우측: 스크롤 가능한 컨트롤 패널
        _r_outer = tk.Frame(self.win, bg=BG_PANEL, width=CTRL_W)
        _r_outer.pack(side=tk.RIGHT, fill=tk.Y, padx=(6, 12), pady=12)
        _r_outer.pack_propagate(False)

        _r_sb = tk.Scrollbar(_r_outer, orient="vertical")
        _r_sb.pack(side=tk.RIGHT, fill=tk.Y)

        _r_cv = tk.Canvas(
            _r_outer, bg=BG_PANEL,
            yscrollcommand=_r_sb.set,
            highlightthickness=0,
        )
        _r_cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        _r_sb.config(command=_r_cv.yview)

        right = tk.Frame(_r_cv, bg=BG_PANEL)
        _r_win = _r_cv.create_window((0, 0), window=right, anchor="nw")

        right.bind("<Configure>",
                   lambda e: _r_cv.configure(scrollregion=_r_cv.bbox("all")))
        _r_cv.bind("<Configure>",
                   lambda e: _r_cv.itemconfig(_r_win, width=e.width))

        def _r_wheel(e):
            _r_cv.yview_scroll(int(-1 * (e.delta / 120)), "units")
        _r_cv.bind("<Enter>", lambda e: _r_cv.bind_all("<MouseWheel>", _r_wheel))
        _r_cv.bind("<Leave>", lambda e: _r_cv.unbind_all("<MouseWheel>"))

        # ── 탭 분리 ───────────────────────────────────────────────
        _nb_style = ttk.Style()
        _nb_style.theme_use('default')
        _nb_style.configure("PH.TNotebook",
                            background=BG_PANEL, borderwidth=0, tabmargins=[0, 0, 0, 0])
        _nb_style.configure("PH.TNotebook.Tab",
                            background="#0d1b38", foreground=TEXT_G,
                            font=("Segoe UI", 9, "bold"), padding=[14, 5])
        _nb_style.map("PH.TNotebook.Tab",
                      background=[("selected", BG_PANEL), ("active", "#1e2f52")],
                      foreground=[("selected", TEXT_W)])
        nb = ttk.Notebook(right, style="PH.TNotebook")
        nb.pack(fill=tk.BOTH, expand=True)
        tab1 = tk.Frame(nb, bg=BG_PANEL)
        tab2 = tk.Frame(nb, bg=BG_PANEL)
        nb.add(tab1, text="  기본  ")
        nb.add(tab2, text="  이미지  ")
        right = tab1

        # ── 프레임 레이트 ──
        self._section_label(right, "프레임 레이트")
        fps_row = tk.Frame(right, bg=BG_PANEL)
        fps_row.pack(pady=(0, 6))
        for val in (30, 60):
            tk.Radiobutton(
                fps_row,
                text=f"{val} fps",
                variable=self._fps_var, value=val,
                fg=TEXT_W, bg=BG_PANEL,
                selectcolor=BG_CTRL,
                activeforeground=TEXT_W, activebackground=BG_PANEL,
                font=("Segoe UI", 11),
                command=self._on_fps_change,
            ).pack(side=tk.LEFT, padx=10)

        self._separator(right)

        # 접기/펴기 섹션 목록 (- 키 토글용)
        self._panel_sections: list = []

        # ── 오버레이 (접기/펴기) ──
        _ov_open = tk.BooleanVar(value=True)
        _ov_hdr = tk.Frame(right, bg=BG_PANEL, cursor="hand2")
        _ov_hdr.pack(fill=tk.X)
        _ov_lbl = tk.Label(
            _ov_hdr, text="▼  오버레이",
            font=("Segoe UI", 10, "bold"),
            fg=TEXT_G, bg=BG_PANEL,
        )
        _ov_lbl.pack(pady=(14, 4))
        _ov_sep = ttk.Separator(right, orient="horizontal")
        _ov_sep.pack(fill=tk.X, pady=(0, 4), padx=12)
        _ov_body = tk.Frame(right, bg=BG_PANEL)
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
                selectcolor=BG_CTRL,
                activeforeground=TEXT_W, activebackground=BG_PANEL,
                anchor=tk.W,
            ).pack(fill=tk.X, padx=8, pady=(0, 2))
        tk.Checkbutton(
            _ov_body, text="랜드마크 이름",
            variable=self._show_names,
            font=("Segoe UI", 10),
            fg="#ffdd88", bg=BG_PANEL,
            selectcolor=BG_CTRL,
            activeforeground="#ffdd88", activebackground=BG_PANEL,
            anchor=tk.W,
        ).pack(fill=tk.X, padx=8, pady=(4, 2))
        tk.Checkbutton(
            _ov_body, text="얼굴 모자이크",
            variable=self._show_mosaic,
            font=("Segoe UI", 10),
            fg="#ff8888", bg=BG_PANEL,
            selectcolor=BG_CTRL,
            activeforeground="#ff8888", activebackground=BG_PANEL,
            anchor=tk.W,
        ).pack(fill=tk.X, padx=8, pady=(2, 2))

        self._separator(right)

        right = tab2  # 이미지 탭

        # ── 얼굴 이미지 (접기/펴기) ──
        _fi_open = tk.BooleanVar(value=True)
        _fi_hdr = tk.Frame(right, bg=BG_PANEL, cursor="hand2")
        _fi_hdr.pack(fill=tk.X)
        _fi_lbl = tk.Label(
            _fi_hdr, text="▼  얼굴 이미지",
            font=("Segoe UI", 10, "bold"),
            fg=TEXT_G, bg=BG_PANEL,
        )
        _fi_lbl.pack(pady=(14, 4))
        _fi_sep = ttk.Separator(right, orient="horizontal")
        _fi_sep.pack(fill=tk.X, pady=(0, 4), padx=12)
        _fi_body = tk.Frame(right, bg=BG_PANEL)
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

        self._face_img_btn = self._make_btn(
            _fi_body, "이미지 로드", BG_CTRL,
            command=self._toggle_face_image,
        )
        self._face_img_lbl = tk.Label(
            _fi_body, text="미선택",
            font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL,
            wraplength=CTRL_W - 20,
        )
        self._face_img_lbl.pack(pady=(0, 2))
        tk.Label(_fi_body, text="눈 위치 Y (%)",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL).pack()
        tk.Scale(_fi_body, from_=10, to=90, orient=tk.HORIZONTAL,
                 variable=self._eye_y_var, length=170,
                 bg=BG_PANEL, fg=TEXT_W, troughcolor=BG_CTRL,
                 highlightthickness=0, showvalue=True,
                 ).pack(pady=(0, 2))
        tk.Label(_fi_body, text="눈 위치 X (%)",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL).pack()
        tk.Scale(_fi_body, from_=10, to=90, orient=tk.HORIZONTAL,
                 variable=self._eye_x_var, length=170,
                 bg=BG_PANEL, fg=TEXT_W, troughcolor=BG_CTRL,
                 highlightthickness=0, showvalue=True,
                 ).pack(pady=(0, 2))
        tk.Label(_fi_body, text="크기 (%)",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL).pack()
        tk.Scale(_fi_body, from_=30, to=300, orient=tk.HORIZONTAL,
                 variable=self._img_size_var, length=170,
                 bg=BG_PANEL, fg=TEXT_W, troughcolor=BG_CTRL,
                 highlightthickness=0, showvalue=True,
                 ).pack(pady=(0, 2))
        tk.Label(_fi_body, text="떨림 보정 (0=없음  →  95=최대)",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL).pack()
        tk.Scale(_fi_body, from_=0, to=95, orient=tk.HORIZONTAL,
                 variable=self._ema_smooth_var, length=170,
                 bg=BG_PANEL, fg="#88ddff", troughcolor=BG_CTRL,
                 highlightthickness=0, showvalue=True,
                 command=self._on_ema_smooth_change,
                 ).pack(pady=(0, 4))

        tk.Frame(_fi_body, bg="#2a2a4a", height=1).pack(fill=tk.X, padx=10, pady=(2, 6))

        self._face_img_open_btn = self._make_btn(
            _fi_body, "🖼  입 벌림 이미지 로드", BG_CTRL,
            command=self._toggle_face_image_open,
        )
        self._face_img_open_lbl = tk.Label(
            _fi_body, text="미선택",
            font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL,
            wraplength=CTRL_W - 20,
        )
        self._face_img_open_lbl.pack(pady=(0, 2))
        tk.Label(_fi_body, text="전환 임계값 (MAR)",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL).pack()
        self._mouth_thr_scale = tk.Scale(
            _fi_body, from_=0.02, to=0.30, resolution=0.01, orient=tk.HORIZONTAL,
            variable=self._mouth_thr_var, length=170,
            bg=BG_PANEL, fg="#ffcc88", troughcolor=BG_CTRL,
            highlightthickness=0, showvalue=True,
            state=tk.DISABLED,
        )
        self._mouth_thr_scale.pack(pady=(0, 4))

        self._separator(right)

        # ── 오른팔 이미지 (접기/펴기) ──
        _arm_open = tk.BooleanVar(value=True)
        _arm_hdr = tk.Frame(right, bg=BG_PANEL, cursor="hand2")
        _arm_hdr.pack(fill=tk.X)
        _arm_lbl = tk.Label(
            _arm_hdr, text="▼  오른팔 이미지",
            font=("Segoe UI", 10, "bold"),
            fg=TEXT_G, bg=BG_PANEL,
        )
        _arm_lbl.pack(pady=(14, 4))
        _arm_sep = ttk.Separator(right, orient="horizontal")
        _arm_sep.pack(fill=tk.X, pady=(0, 4), padx=12)
        _arm_body = tk.Frame(right, bg=BG_PANEL)
        _arm_body.pack(fill=tk.X)

        def _toggle_arm_img(_e=None):
            if _arm_open.get():
                _arm_body.pack_forget()
                _arm_lbl.config(text="▶  오른팔 이미지")
                _arm_open.set(False)
            else:
                _arm_body.pack(fill=tk.X, after=_arm_sep)
                _arm_lbl.config(text="▼  오른팔 이미지")
                _arm_open.set(True)

        _arm_hdr.bind("<Button-1>", _toggle_arm_img)
        _arm_lbl.bind("<Button-1>", _toggle_arm_img)
        self._panel_sections.append((_arm_open, _toggle_arm_img))

        self._arm_img_btn = self._make_btn(
            _arm_body, "🦾  이미지 로드", BG_CTRL,
            command=lambda: self._toggle_arm_image(side='right'),
        )
        self._arm_img_lbl = tk.Label(
            _arm_body, text="미선택",
            font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL,
            wraplength=CTRL_W - 20,
        )
        self._arm_img_lbl.pack(pady=(0, 2))
        tk.Label(_arm_body, text="앵커 Y (%)",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL).pack()
        tk.Scale(_arm_body, from_=10, to=90, orient=tk.HORIZONTAL,
                 variable=self._arm_y_var, length=170,
                 bg=BG_PANEL, fg=TEXT_W, troughcolor=BG_CTRL,
                 highlightthickness=0, showvalue=True,
                 ).pack(pady=(0, 2))
        tk.Label(_arm_body, text="앵커 X (%)",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL).pack()
        tk.Scale(_arm_body, from_=10, to=90, orient=tk.HORIZONTAL,
                 variable=self._arm_x_var, length=170,
                 bg=BG_PANEL, fg=TEXT_W, troughcolor=BG_CTRL,
                 highlightthickness=0, showvalue=True,
                 ).pack(pady=(0, 2))
        tk.Label(_arm_body, text="크기 (%)",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL).pack()
        tk.Scale(_arm_body, from_=30, to=300, orient=tk.HORIZONTAL,
                 variable=self._arm_size_var, length=170,
                 bg=BG_PANEL, fg=TEXT_W, troughcolor=BG_CTRL,
                 highlightthickness=0, showvalue=True,
                 ).pack(pady=(0, 2))
        tk.Label(_arm_body, text="떨림 보정 (0=없음  →  95=최대)",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL).pack()
        tk.Scale(_arm_body, from_=0, to=95, orient=tk.HORIZONTAL,
                 variable=self._arm_smooth_var, length=170,
                 bg=BG_PANEL, fg="#88ddff", troughcolor=BG_CTRL,
                 highlightthickness=0, showvalue=True,
                 command=self._on_arm_smooth_change,
                 ).pack(pady=(0, 4))

        # ── Puppet Pin UI ──
        tk.Frame(_arm_body, bg="#2a2a4a", height=1).pack(fill=tk.X, padx=10, pady=(4, 4))
        self._arm_pin_lbl = tk.Label(
            _arm_body, text="피벗 미설정",
            font=("Segoe UI", 8), fg="#ffaa44", bg=BG_PANEL, anchor="w",
        )
        self._arm_pin_lbl.pack(fill=tk.X, padx=14, pady=(0, 2))
        self._arm_pin_btn = tk.Button(
            _arm_body, text="🎯 피벗 설정",
            font=("Segoe UI", 9, "bold"), bg="#2a3f5f", fg=TEXT_W,
            activebackground="#3a5a80", activeforeground="white",
            relief=tk.FLAT, cursor="hand2", pady=5, padx=10,
            command=lambda: self._open_pin_picker(side='right'), state=tk.DISABLED,
        )
        self._arm_pin_btn.pack(fill=tk.X, padx=10, pady=(0, 4))

        self._separator(right)

        # ── 왼팔 이미지 (접기/펴기) ──
        _arml_open = tk.BooleanVar(value=True)
        _arml_hdr = tk.Frame(right, bg=BG_PANEL, cursor="hand2")
        _arml_hdr.pack(fill=tk.X)
        _arml_lbl = tk.Label(
            _arml_hdr, text="▼  왼팔 이미지",
            font=("Segoe UI", 10, "bold"),
            fg=TEXT_G, bg=BG_PANEL,
        )
        _arml_lbl.pack(pady=(14, 4))
        _arml_sep = ttk.Separator(right, orient="horizontal")
        _arml_sep.pack(fill=tk.X, pady=(0, 4), padx=12)
        _arml_body = tk.Frame(right, bg=BG_PANEL)
        _arml_body.pack(fill=tk.X)

        def _toggle_arml_img(_e=None):
            if _arml_open.get():
                _arml_body.pack_forget()
                _arml_lbl.config(text="▶  왼팔 이미지")
                _arml_open.set(False)
            else:
                _arml_body.pack(fill=tk.X, after=_arml_sep)
                _arml_lbl.config(text="▼  왼팔 이미지")
                _arml_open.set(True)

        _arml_hdr.bind("<Button-1>", _toggle_arml_img)
        _arml_lbl.bind("<Button-1>", _toggle_arml_img)
        self._panel_sections.append((_arml_open, _toggle_arml_img))

        self._arm_img_btn_l = self._make_btn(
            _arml_body, "🦾  이미지 로드", BG_CTRL,
            command=lambda: self._toggle_arm_image(side='left'),
        )
        self._arm_img_lbl_l = tk.Label(
            _arml_body, text="미선택",
            font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL,
            wraplength=CTRL_W - 20,
        )
        self._arm_img_lbl_l.pack(pady=(0, 2))

        # ── Puppet Pin UI (왼팔) ──
        tk.Frame(_arml_body, bg="#2a2a4a", height=1).pack(fill=tk.X, padx=10, pady=(4, 4))
        self._arm_pin_lbl_l = tk.Label(
            _arml_body, text="피벗 미설정",
            font=("Segoe UI", 8), fg="#ffaa44", bg=BG_PANEL, anchor="w",
        )
        self._arm_pin_lbl_l.pack(fill=tk.X, padx=14, pady=(0, 2))
        self._arm_pin_btn_l = tk.Button(
            _arml_body, text="🎯 피벗 설정",
            font=("Segoe UI", 9, "bold"), bg="#2a3f5f", fg=TEXT_W,
            activebackground="#3a5a80", activeforeground="white",
            relief=tk.FLAT, cursor="hand2", pady=5, padx=10,
            command=lambda: self._open_pin_picker(side='left'), state=tk.DISABLED,
        )
        self._arm_pin_btn_l.pack(fill=tk.X, padx=10, pady=(0, 4))

        self._separator(right)

        right = tab1  # 기본 탭으로 복귀

        # ── AE 스무딩 ──
        self._section_label(right, "AE 스무딩")
        smooth_row = tk.Frame(right, bg=BG_PANEL)
        smooth_row.pack(pady=(0, 4))
        tk.Label(smooth_row, text="0", font=("Segoe UI", 9),
                 fg=TEXT_G, bg=BG_PANEL).pack(side=tk.LEFT)
        tk.Scale(
            smooth_row, from_=0, to=15, orient=tk.HORIZONTAL,
            variable=self._smooth_var, length=130,
            bg=BG_PANEL, fg=TEXT_W, troughcolor=BG_CTRL,
            highlightthickness=0, showvalue=True,
        ).pack(side=tk.LEFT, padx=4)
        tk.Label(smooth_row, text="15", font=("Segoe UI", 9),
                 fg=TEXT_G, bg=BG_PANEL).pack(side=tk.LEFT)

        self._separator(right)

        # ── 감지 설정 ──
        self._section_label(right, "얼굴 감지 신뢰도")
        tk.Scale(right, from_=0.1, to=0.9, resolution=0.05, orient=tk.HORIZONTAL,
                 variable=self._face_conf_var, length=170,
                 bg=BG_PANEL, fg="#88bbff", troughcolor=BG_CTRL,
                 highlightthickness=0, showvalue=True,
                 ).pack(pady=(0, 4))

        self._separator(right)

        # ── 카메라 버튼 ──
        self._cam_btn = self._make_btn(
            right, "카메라 시작", ACCENT,
            command=self._toggle_camera,
        )

        self._separator(right)

        # ── 녹화 버튼 ──
        self._rec_btn = self._make_btn(
            right, "녹화 시작", "#333355",
            fg=TEXT_G, state=tk.DISABLED,
            command=self._toggle_record,
        )

        self._separator(right)

        # ── 상태 표시 ──
        self._section_label(right, "상태")
        tk.Label(
            right,
            textvariable=self._status_var,
            font=("Segoe UI", 10),
            fg="#4aff9e", bg=BG_PANEL,
            wraplength=CTRL_W - 20,
            justify=tk.CENTER,
        ).pack(pady=4)

        # REC 깜빡임 표시
        self._rec_ind = tk.Label(
            right, text="",
            font=("Segoe UI", 12, "bold"),
            fg=ACCENT_R, bg=BG_PANEL,
        )
        self._rec_ind.pack(pady=2)

        self.win.bind('-', self._toggle_all_sections)

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

    def _section_label(self, parent, text: str):
        tk.Label(
            parent, text=text,
            font=("Segoe UI", 10, "bold"),
            fg=TEXT_G, bg=BG_PANEL,
        ).pack(pady=(14, 4))

    def _separator(self, parent):
        ttk.Separator(parent, orient="horizontal").pack(
            fill=tk.X, pady=8, padx=12,
        )

    def _make_btn(self, parent, text, bg,
                  fg="white", state=tk.NORMAL, command=None):
        b = tk.Button(
            parent, text=text,
            font=("Segoe UI", 11, "bold"),
            bg=bg, fg=fg,
            activebackground=bg, activeforeground=fg,
            relief=tk.FLAT, cursor="hand2",
            width=18, height=2,
            state=state, command=command,
        )
        b.pack(pady=4)
        return b

    def _draw_placeholder(self):
        self._canvas.update_idletasks()
        w = self._canvas.winfo_width()  or CANVAS_W
        h = self._canvas.winfo_height() or CANVAS_H
        self._canvas.delete("all")
        self._canvas.create_text(
            w // 2, h // 2,
            text="카메라를 시작하세요",
            fill="#444466", font=("Segoe UI", 18),
        )

    # ── FPS 변경 ──────────────────────────────────────────────────────────
    def _on_fps_change(self):
        self._fps = self._fps_var.get()
        if self._running and self._cap:
            self._cap.set(cv2.CAP_PROP_FPS, self._fps)

    # ── 카메라 토글 ────────────────────────────────────────────────────────
    def _toggle_camera(self):
        if not self._running:
            self._start_camera()
        else:
            self._stop_camera()

    def _start_camera(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap.release()
            # 카메라 없음 → 영상 파일로 대체 여부 묻기
            from tkinter import filedialog
            ans = messagebox.askyesno(
                "카메라 없음",
                "웹캠을 찾을 수 없습니다.\n\n"
                "영상 파일로 대신 실행하시겠습니까?\n"
                "(WSL2 환경에서는 영상 파일을 사용하세요)",
                parent=self.win,
            )
            if not ans:
                return
            path = filedialog.askopenfilename(
                parent=self.win,
                title="영상 파일 선택",
                filetypes=[("영상 파일", "*.mp4 *.avi *.mov *.mkv"), ("모든 파일", "*.*")],
            )
            if not path:
                return
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                messagebox.showerror("오류", f"파일을 열 수 없습니다:\n{path}", parent=self.win)
                return
            self._is_file_mode = True
        else:
            self._is_file_mode = False

        self._fps = self._fps_var.get()
        cap.set(cv2.CAP_PROP_FPS, self._fps)
        self._cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._cap     = cap
        self._running = True

        self._cam_btn.config(text="카메라 중지", bg=ACCENT_R)
        self._rec_btn.config(
            state=tk.NORMAL, bg=ACCENT_R, fg="white", text="녹화 시작",
        )
        self._status_var.set("카메라 실행중")

        self._cap_thread = threading.Thread(
            target=self._capture_loop, daemon=True,
        )
        self._cap_thread.start()
        self._schedule_display()

    def _stop_camera(self):
        if self._recording:
            self._stop_record(cancelled=True)

        self._running = False

        if self._after_id:
            self.win.after_cancel(self._after_id)
            self._after_id = None

        if self._cap:
            self._cap.release()
            self._cap = None

        self._cam_btn.config(text="카메라 시작", bg=ACCENT)
        self._rec_btn.config(
            state=tk.DISABLED, bg="#333355", fg=TEXT_G, text="녹화 시작",
        )
        self._status_var.set("대기중")
        self._rec_ind.config(text="")
        self._draw_placeholder()

    # ── 녹화 토글 ──────────────────────────────────────────────────────────
    def _toggle_record(self):
        if not self._recording:
            self._start_record()
        else:
            self._stop_record()

    def _start_record(self):
        self._tmp_path = os.path.join(
            tempfile.gettempdir(), "_posetracker_rec.mp4",
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            self._tmp_path, fourcc, self._fps,
            (self._cam_w, self._cam_h),
        )
        self._frames_data = []
        self._recording = True
        self._rec_btn.config(text="녹화 중지", bg="#cc2222")
        self._status_var.set(f"녹화중  ({self._fps} fps)")
        self._blink()

    def _stop_record(self, cancelled: bool = False):
        self._recording = False
        if self._writer:
            self._writer.release()
            self._writer = None
        self._rec_btn.config(text="녹화 시작", bg=ACCENT_R)
        self._rec_ind.config(text="")

        if cancelled:
            self._cleanup_tmp()
            return

        self._status_var.set("저장 경로 선택...")

        save_path = filedialog.asksaveasfilename(
            parent=self.win,
            title="영상 저장 위치 선택",
            defaultextension=".mp4",
            filetypes=[("MP4 파일", "*.mp4"), ("모든 파일", "*.*")],
        )

        if not save_path:
            self._cleanup_tmp()
            self._status_var.set("카메라 실행중")
            return

        # 임시 파일 → 지정 경로로 이동
        shutil.move(self._tmp_path, save_path)
        self._tmp_path = None

        # AE 내보내기 (별도 스레드)
        out_dir   = os.path.dirname(save_path) or "."
        json_path = os.path.join(out_dir, "tracking_data.json")
        ae_dir    = os.path.join(out_dir, "ae_keyframes")
        fps       = self._fps
        frames    = list(self._frames_data)
        cam_w, cam_h = self._cam_w, self._cam_h
        inc_face   = self._show_face.get()
        inc_body   = self._show_body.get()
        inc_hands  = self._show_hands.get()
        smooth     = self._smooth_var.get()

        self._status_var.set("AE 데이터 내보내는 중...")

        def _export():
            if frames:
                info = VideoInfo(
                    width=cam_w, height=cam_h,
                    fps=fps, total_frames=len(frames),
                )
                export_json(frames, info, json_path,
                            include_face=inc_face, include_body=inc_body, include_hands=inc_hands)
                export_ae_keyframes(frames, info, ae_dir,
                                    include_face=inc_face, include_body=inc_body, include_hands=inc_hands,
                                    smooth_radius=smooth)

            def _done():
                self._status_var.set(f"저장 완료! ({len(frames)} 프레임)")
                messagebox.showinfo(
                    "저장 완료",
                    f"영상:  {save_path}\n"
                    f"JSON:  {json_path}\n"
                    f"AE 키프레임:  {ae_dir}/",
                    parent=self.win,
                )
                self._status_var.set("카메라 실행중")

            self.win.after(0, _done)

        threading.Thread(target=_export, daemon=True).start()

    def _cleanup_tmp(self):
        if self._tmp_path and os.path.exists(self._tmp_path):
            os.remove(self._tmp_path)
        self._tmp_path = None

    def _blink(self):
        if not self._recording:
            return
        cur = self._rec_ind.cget("text")
        self._rec_ind.config(text="" if cur else "● REC")
        self.win.after(500, self._blink)

    # ── EMA 떨림 보정 슬라이더 콜백 ──────────────────────────────────────
    def _on_ema_smooth_change(self, _=None):
        """슬라이더 값(0~95) → alpha(1.00~0.05) 변환 후 EMA 상태에 반영."""
        v = self._ema_smooth_var.get()          # 0=보정없음, 95=최대보정
        alpha = 1.0 - (v / 100.0)              # 0→1.00, 95→0.05
        self._face_img_ema['alpha'] = alpha

    # ── 오른팔 이미지 EMA 콜백 / 로드/제거 ───────────────────────────────
    def _on_arm_smooth_change(self, _=None):
        v = self._arm_smooth_var.get()
        alpha = 1.0 - (v / 100.0)
        self._arm_img_ema['alpha'] = alpha
        self._arm_img_ema_l['alpha'] = alpha

    def _toggle_arm_image(self, side='right'):
        if side == 'right':
            img_attr  = '_arm_img';    pins_attr  = '_arm_pins';    cache_attr  = '_arm_seg_cache'
            ema_attr  = '_arm_img_ema'; lbl = self._arm_img_lbl;   btn = self._arm_img_btn
            pin_lbl   = self._arm_pin_lbl;  pin_btn = self._arm_pin_btn
        else:
            img_attr  = '_arm_img_l';  pins_attr  = '_arm_pins_l';  cache_attr  = '_arm_seg_cache_l'
            ema_attr  = '_arm_img_ema_l'; lbl = self._arm_img_lbl_l; btn = self._arm_img_btn_l
            pin_lbl   = self._arm_pin_lbl_l; pin_btn = self._arm_pin_btn_l

        if getattr(self, img_attr) is not None:
            setattr(self, img_attr, None)
            setattr(self, pins_attr, None)
            setattr(self, cache_attr, None)
            for _k in ('elbow_x', 'elbow_y', 'angle', 'arm_len',
                       'shldr_x', 'shldr_y', 'wrist_x', 'wrist_y'):
                getattr(self, ema_attr)[_k] = None
            lbl.config(text="미선택")
            btn.config(text="🦾  이미지 로드")
            pin_lbl.config(text="피벗 미설정", fg="#ffaa44")
            pin_btn.config(text="🎯 피벗 설정", state=tk.DISABLED)
        else:
            self._load_arm_image(side=side)

    def _load_arm_image(self, side='right'):
        title = "오른팔 이미지 선택" if side == 'right' else "왼팔 이미지 선택"
        path = filedialog.askopenfilename(
            parent=self.win, title=title,
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

        if side == 'right':
            self._arm_img = img.copy()
            self._arm_pins = None
            self._arm_seg_cache = None
            self._arm_img_lbl.config(text=os.path.basename(path))
            self._arm_img_btn.config(text="× 이미지 제거")
            self._arm_pin_lbl.config(text="피벗 미설정", fg="#ffaa44")
            self._arm_pin_btn.config(text="🎯 피벗 설정", state=tk.NORMAL)
        else:
            self._arm_img_l = img.copy()
            self._arm_pins_l = None
            self._arm_seg_cache_l = None
            self._arm_img_lbl_l.config(text=os.path.basename(path))
            self._arm_img_btn_l.config(text="× 이미지 제거")
            self._arm_pin_lbl_l.config(text="피벗 미설정", fg="#ffaa44")
            self._arm_pin_btn_l.config(text="🎯 피벗 설정", state=tk.NORMAL)
        self.win.after(100, lambda: self._open_pin_picker(side=side))

    # ── Puppet Pin 피벗 설정 팝업 ─────────────────────────────────────────
    def _open_pin_picker(self, side='right'):
        if not _PUPPET_AVAILABLE:
            messagebox.showwarning("미지원", "puppet_pin 모듈을 불러올 수 없습니다.", parent=self.win)
            return
        arm_img  = self._arm_img  if side == 'right' else self._arm_img_l
        arm_pins = self._arm_pins if side == 'right' else self._arm_pins_l
        pin_lbl  = self._arm_pin_lbl  if side == 'right' else self._arm_pin_lbl_l
        pin_btn  = self._arm_pin_btn  if side == 'right' else self._arm_pin_btn_l
        if arm_img is None:
            return
        # 중복 방지
        if self._pin_popup is not None:
            try:
                self._pin_popup.lift()
                return
            except Exception:
                self._pin_popup = None

        popup = tk.Toplevel(self.win)
        popup.title(f"피벗 핀 설정 ({'오른팔' if side == 'right' else '왼팔'})")
        popup.resizable(False, False)
        popup.grab_set()
        self._pin_popup = popup

        def _on_close():
            self._pin_popup = None
            popup.destroy()
        popup.protocol("WM_DELETE_WINDOW", _on_close)

        _clicks = []
        _COLORS = ["#ff4444", "#ffdd00", "#44ff88", "#4488ff"]
        _LABELS = ["어깨 (Shoulder)", "팔꿈치 (Elbow)", "손목 (Wrist)", "손가락 끝 (Hand)"]
        _MARKER_R = 6

        status_lbl = tk.Label(popup, text=f"[1/4] {_LABELS[0]} 위치를 클릭하세요",
                               font=("Segoe UI", 9), fg="#aaccff",
                               bg="#1a1a2e", anchor="w", padx=8, pady=4)
        status_lbl.pack(fill=tk.X)

        img_bgra = arm_img
        ih, iw = img_bgra.shape[:2]
        MAX_SIZE = 420
        scale_f = min(MAX_SIZE / iw, MAX_SIZE / ih, 1.0)
        disp_w = max(1, int(iw * scale_f))
        disp_h = max(1, int(ih * scale_f))

        img_rgb = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2RGB)
        pil_img = __import__('PIL').Image.fromarray(img_rgb).resize(
            (disp_w, disp_h), __import__('PIL').Image.LANCZOS)
        _tk_img = ImageTk.PhotoImage(pil_img)

        canvas = tk.Canvas(popup, width=disp_w, height=disp_h,
                            bg="#000011", highlightthickness=1,
                            highlightbackground="#333355")
        canvas.pack(padx=10, pady=6)
        canvas.create_image(0, 0, anchor="nw", image=_tk_img)
        canvas._tk_img_ref = _tk_img

        if arm_pins is not None:
            pts_img = list(arm_pins.arrays())
            for _pi, (px, py) in enumerate(pts_img):
                cx = px * scale_f; cy = py * scale_f
                _clicks.append((px, py))
                canvas.create_oval(cx - _MARKER_R, cy - _MARKER_R,
                                    cx + _MARKER_R, cy + _MARKER_R,
                                    fill=_COLORS[_pi], outline="white", width=1)

        def _update_status():
            n = len(_clicks)
            if n < 3:
                status_lbl.config(text=f"[{n+1}/4] {_LABELS[n]} 위치를 클릭하세요",
                                  fg="#aaccff")
                ok_btn.config(state=tk.DISABLED)
            elif n == 3:
                status_lbl.config(
                    text=f"3점 완료 (4번째: {_LABELS[3]} 선택 가능) — [확인]으로 저장",
                    fg="#ffdd88")
                ok_btn.config(state=tk.NORMAL)
            else:
                status_lbl.config(text="4점 완료 — [확인]으로 저장하세요", fg="#44ff88")
                ok_btn.config(state=tk.NORMAL)

        def _on_canvas_click(event):
            if len(_clicks) >= 4:
                return
            img_x = event.x / scale_f
            img_y = event.y / scale_f
            _clicks.append((img_x, img_y))
            idx = len(_clicks) - 1
            cx, cy = event.x, event.y
            canvas.create_oval(cx - _MARKER_R, cy - _MARKER_R,
                                cx + _MARKER_R, cy + _MARKER_R,
                                fill=_COLORS[idx], outline="white", width=1)
            _update_status()

        canvas.bind("<Button-1>", _on_canvas_click)

        btn_row = tk.Frame(popup, bg="#1a1a2e")
        btn_row.pack(fill=tk.X, padx=10, pady=(0, 8))

        def _reset():
            nonlocal _clicks
            _clicks = []
            canvas.delete("all")
            canvas.create_image(0, 0, anchor="nw", image=_tk_img)
            status_lbl.config(text=f"[1/4] {_LABELS[0]} 위치를 클릭하세요", fg="#aaccff")
            ok_btn.config(state=tk.DISABLED)

        def _confirm():
            if len(_clicks) < 3:
                return
            pins = PuppetPins(
                img_shldr=_clicks[0],
                img_elbow=_clicks[1],
                img_wrist=_clicks[2],
                img_hand=_clicks[3] if len(_clicks) >= 4 else None,
            )
            if pins_degenerate(pins, min_dist=6.0):
                messagebox.showwarning("경고",
                    "핀 간격이 너무 좁습니다. 더 멀리 클릭해주세요.",
                    parent=popup)
                return
            if side == 'right':
                self._arm_pins     = pins
                self._arm_seg_cache = build_segment_cache(self._arm_img, pins)
            else:
                self._arm_pins_l      = pins
                self._arm_seg_cache_l = build_segment_cache(self._arm_img_l, pins)
            if pins.img_hand is not None:
                pin_lbl.config(text="어깨 ○  팔꿈치 ○  손목 ○  손 ○", fg="#44ff88")
            else:
                pin_lbl.config(text="어깨 ○  팔꿈치 ○  손목 ○", fg="#44ff88")
            pin_btn.config(text="🎯 피벗 재설정")
            _on_close()

        tk.Button(btn_row, text="초기화", font=("Segoe UI", 9),
                  bg="#3a2a2a", fg=TEXT_W, relief=tk.FLAT, cursor="hand2",
                  command=_reset).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(btn_row, text="취소", font=("Segoe UI", 9),
                  bg="#2a2a3a", fg=TEXT_W, relief=tk.FLAT, cursor="hand2",
                  command=_on_close).pack(side=tk.LEFT, padx=(0, 6))
        ok_btn = tk.Button(btn_row, text="확인", font=("Segoe UI", 9, "bold"),
                           bg="#1e5f3a", fg=TEXT_W, relief=tk.FLAT, cursor="hand2",
                           state=tk.DISABLED, command=_confirm)
        ok_btn.pack(side=tk.LEFT)

        _update_status()

    # ── 얼굴 이미지 로드/제거 ──────────────────────────────────────────────
    def _toggle_face_image(self):
        if self._face_img is not None:
            self._face_img = None
            self._face_img_pts = None
            for _k in ('face_h', 'eye_cx', 'eye_cy', 'angle'):
                self._face_img_ema[_k] = None
            self._face_img_lbl.config(text="미선택")
            self._face_img_btn.config(text="이미지 로드")
        else:
            self._load_face_image()

    def _load_face_image(self):
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

        # InsightFace로 얼굴 감지 (BGR 직접 사용)
        bgr_src = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        try:
            face_res = _if_det_mod.detect(bgr_src, w=w, h=h)
        except Exception as e:
            messagebox.showerror("오류", f"얼굴 감지 실패:\n{e}", parent=self.win)
            return

        # 얼굴 감지 성공 → 정밀 모드 (Homography)
        # 감지 실패 (그림/일러스트) → 자동 모드 (Affine)
        if face_res.face_landmarks and len(face_res.face_landmarks[0]) > max(_FACE_IMG_KPT):
            lf = face_res.face_landmarks[0]
            src_pts = np.float32([[lf[i].x * w, lf[i].y * h] for i in _FACE_IMG_KPT])
            mode_text = " [정밀]"
        else:
            src_pts = None  # Affine 자동 모드
            mode_text = " [자동]"

        self._face_img = img.copy()
        self._face_img_pts = src_pts
        self._face_img_lbl.config(text=os.path.basename(path) + mode_text)
        self._face_img_btn.config(text="× 이미지 제거")

    # ── 입 벌림 이미지 로드/제거 ──────────────────────────────────────────
    def _toggle_face_image_open(self):
        if self._face_img_open is not None:
            self._face_img_open = None
            self._face_img_open_pts = None
            self._face_img_open_lbl.config(text="미선택")
            self._face_img_open_btn.config(text="🖼  입 벌림 이미지 로드")
            self._mouth_thr_scale.config(state=tk.DISABLED)
        else:
            self._load_face_image_open()

    def _load_face_image_open(self):
        path = filedialog.askopenfilename(
            parent=self.win,
            title="입 벌림 이미지 선택",
            filetypes=[("이미지", "*.png *.jpg *.jpeg *.webp *.bmp"), ("모든 파일", "*.*")],
        )
        if not path:
            return
        try:
            pil_img = Image.open(path).convert("RGBA")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGRA)
        except Exception as e:
            messagebox.showerror("오류", f"이미지 로드 실패:\n{e}", parent=self.win)
            return

        h, w = img.shape[:2]
        if w > 1024:
            scale = 1024 / w
            img = cv2.resize(img, (1024, int(h * scale)), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]

        # 배경 제거 안 된 경우 흰색 → 투명 변환
        mask = img[:, :, 3]
        if mask.min() == 255:
            tmp = img.copy()
            white = (tmp[:, :, 0] > 240) & (tmp[:, :, 1] > 240) & (tmp[:, :, 2] > 240)
            tmp[white, 3] = 0
            img = tmp

        # InsightFace로 얼굴 감지 (BGR 직접 사용)
        bgr_src = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        try:
            face_res = _if_det_mod.detect(bgr_src, w=w, h=h)
        except Exception as e:
            messagebox.showerror("오류", f"얼굴 감지 실패:\n{e}", parent=self.win)
            return

        if face_res.face_landmarks and len(face_res.face_landmarks[0]) > max(_FACE_IMG_KPT):
            lf = face_res.face_landmarks[0]
            src_pts = np.float32([[lf[i].x * w, lf[i].y * h] for i in _FACE_IMG_KPT])
            mode_text = " [정밀]"
        else:
            src_pts = None
            mode_text = " [자동]"

        self._face_img_open = img.copy()
        self._face_img_open_pts = src_pts
        self._face_img_open_lbl.config(text=os.path.basename(path) + mode_text)
        self._face_img_open_btn.config(text="× 입 벌림 제거")
        self._mouth_thr_scale.config(state=tk.NORMAL)

    # ── 캡처 루프 (백그라운드 스레드) ────────────────────────────────────
    def _capture_loop(self):
        # 얼굴 감지: InsightFace 싱글턴 (별도 초기화 불필요)
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

        try:
            hand_det = mp_vision.HandLandmarker.create_from_options(hand_opts)
            pose_det = mp_vision.PoseLandmarker.create_from_options(pose_opts)
        except Exception as e:
            print(f"[MediaPipe init error] {e}")
            self._capture_loop_raw()
            return

        _empty_r = type('R', (), {
            'face_landmarks': [], 'hand_landmarks': [],
            'handedness': [], 'pose_landmarks': []
        })()
        _det_tick = 0
        _last_f = _last_h = _last_p = _empty_r

        try:
            while self._running and self._cap and self._cap.isOpened():
                ret, frame = self._cap.read()
                if not ret:
                    break

                h_px, w_px = frame.shape[:2]
                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                # 최적화: 2프레임마다 감지 (녹화 중에는 매 프레임 감지)
                _det_tick += 1
                _overlay_on = (self._show_face.get() or self._show_body.get()
                               or self._show_hands.get())
                _fi_on     = self._face_img is not None
                _arm_on    = self._arm_img is not None
                _mosaic_on = self._show_mosaic.get()
                _need_det = (_det_tick % 2 == 1) or self._recording

                if _need_det and (_overlay_on or self._recording or _fi_on or _arm_on or _mosaic_on):
                    # 추론 해상도 축소 (최대 640px 너비, 손/포즈용 mp.Image)
                    _sc = min(1.0, 640 / max(w_px, 1))
                    if _sc < 0.99:
                        _iw, _ih = int(w_px * _sc), int(h_px * _sc)
                        _inf = mp.Image(image_format=mp.ImageFormat.SRGB,
                                        data=cv2.resize(rgb, (_iw, _ih)))
                    else:
                        _inf = mp_img
                    try:
                        if self._show_face.get() or self._recording or _fi_on or _mosaic_on:
                            # InsightFace: BGR 원본 프레임 직접 사용
                            _last_f = _if_det_mod.detect(frame, min_conf=self._face_conf_var.get())
                        if self._show_hands.get() or self._recording:
                            _last_h = hand_det.detect(_inf)
                        if self._show_body.get() or self._recording or _arm_on:
                            _last_p = pose_det.detect(_inf)
                    except Exception as e:
                        print(f"[detect error] {e}")

                face_res, hand_res, pose_res = _last_f, _last_h, _last_p

                # 오버레이 그리기
                overlay = frame.copy()
                # ── 얼굴 모자이크 (가장 먼저 적용)
                if _mosaic_on:
                    _apply_face_mosaic(overlay, face_res, w_px, h_px)
                # ── 포즈 스켈레톤 (모든 감지된 사람, 먼저 그려 얼굴/손 위에 덮이지 않게)
                if self._show_body.get() and pose_res.pose_landmarks:
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
                                         (int(_pl[_s].x*w_px), int(_pl[_s].y*h_px)),
                                         (int(_pl[_e].x*w_px), int(_pl[_e].y*h_px)),
                                         _pc, 2)
                        for _i in [11,12,13,14,15,16,23,24,25,26,27,28]:
                            if _i < len(_pl) and _pl[_i].visibility > 0.3:
                                cv2.circle(overlay,
                                           (int(_pl[_i].x*w_px), int(_pl[_i].y*h_px)),
                                           6, _pc, -1)
                        # 사람 번호 레이블
                        if 0 < len(_pl) and _pl[0].visibility > 0.3:
                            _lx = max(5, int(_pl[0].x*w_px) - 12)
                            _ly = max(20, int(_pl[0].y*h_px) - 20)
                            cv2.putText(overlay, f"P{_pidx+1}", (_lx, _ly),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, _pc, 2, cv2.LINE_AA)
                # ── 얼굴 (모든 감지된 사람)
                if (self._show_face.get() or self._show_body.get()) and face_res.face_landmarks:
                    _nc = (0, 230, 180)
                    for _lf in face_res.face_landmarks:
                        if hasattr(_lf, 'bbox'):
                            # InsightFace: 5개 키포인트 원 + bbox 사각형
                            for _i in [33, 263, 4, 61, 291]:
                                cv2.circle(overlay,
                                           (int(_lf[_i].x * w_px), int(_lf[_i].y * h_px)),
                                           5, _nc, -1)
                            cv2.rectangle(overlay,
                                          (_lf.bbox[0], _lf.bbox[1]),
                                          (_lf.bbox[2], _lf.bbox[3]),
                                          _nc, 1)
                        else:
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
                                             (int(_lf[_s].x*w_px), int(_lf[_s].y*h_px)),
                                             (int(_lf[_e].x*w_px), int(_lf[_e].y*h_px)),
                                             _nc, 1)
                            for _i in [1,2,4,5,6,19,97,98,168,195,197,326,327]:
                                if _i < len(_lf):
                                    cv2.circle(overlay,
                                               (int(_lf[_i].x*w_px), int(_lf[_i].y*h_px)),
                                               2, _nc, -1)
                # ── 손
                if hand_res.hand_landmarks and self._show_hands.get():
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
                                         w_px, h_px,
                                         self._show_face.get(),
                                         self._show_body.get(),
                                         self._show_hands.get())

                # ── 얼굴 이미지 오버레이
                _fi = self._face_img
                _fp = self._face_img_pts
                if _fi is not None:
                    if self._face_img_open is not None:
                        _mar = _compute_mar(face_res, w_px, h_px)
                        if _mar >= self._mouth_thr_var.get():
                            _fi = self._face_img_open
                            _fp = self._face_img_open_pts
                    _apply_face_img_overlay(overlay, face_res, w_px, h_px, _fi, _fp,
                                            eye_y_pct=self._eye_y_var.get(),
                                            eye_x_pct=self._eye_x_var.get(),
                                            size_pct=self._img_size_var.get(),
                                            ema_state=self._face_img_ema)

                # ── 오른팔 이미지 오버레이
                if self._arm_img is not None:
                    _apply_arm_img_overlay(overlay, pose_res, w_px, h_px, self._arm_img,
                                           anchor_y_pct=self._arm_y_var.get(),
                                           anchor_x_pct=self._arm_x_var.get(),
                                           size_pct=self._arm_size_var.get(),
                                           ema_state=self._arm_img_ema,
                                           arm_pins=self._arm_pins,
                                           arm_seg_cache=self._arm_seg_cache,
                                           side='right')

                # ── 왼팔 이미지 오버레이
                if self._arm_img_l is not None:
                    _apply_arm_img_overlay(overlay, pose_res, w_px, h_px, self._arm_img_l,
                                           anchor_y_pct=self._arm_y_var.get(),
                                           anchor_x_pct=self._arm_x_var.get(),
                                           size_pct=self._arm_size_var.get(),
                                           ema_state=self._arm_img_ema_l,
                                           arm_pins=self._arm_pins_l,
                                           arm_seg_cache=self._arm_seg_cache_l,
                                           side='left')

                # 녹화 처리
                if self._recording:
                    if self._writer:
                        self._writer.write(overlay)
                    self._collect_frame(face_res, hand_res, pose_res, w_px, h_px)

                # 디스플레이 큐에 넣기 (BGR → RGB)
                try:
                    self._frame_q.put_nowait(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
                except queue.Full:
                    pass
        finally:
            hand_det.close()
            pose_det.close()

    def _capture_loop_raw(self):
        """MediaPipe 없이 원본 영상만 표시"""
        while self._running and self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret:
                break
            if self._recording and self._writer:
                self._writer.write(frame)
            try:
                self._frame_q.put_nowait(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            except queue.Full:
                pass

    def _collect_frame(self, face_res, hand_res, pose_res, w: int, h: int):
        idx = len(self._frames_data)
        fd  = FrameData(index=idx, timestamp=idx / max(self._fps, 1))
        fd.persons = _build_persons(face_res, hand_res, pose_res, w, h)
        self._frames_data.append(fd)

    # ── 디스플레이 갱신 (메인 스레드) ────────────────────────────────────
    def _schedule_display(self):
        if not self._running:
            return
        delay = max(1, 1000 // self._fps)
        self._after_id = self.win.after(delay, self._update_canvas)

    def _update_canvas(self):
        if not self._running:
            return

        try:
            frame_rgb = self._frame_q.get_nowait()
            cw = self._canvas.winfo_width()
            ch = self._canvas.winfo_height()
            if cw <= 1: cw = CANVAS_W
            if ch <= 1: ch = CANVAS_H
            if True:
                img = Image.fromarray(frame_rgb)
                img = img.resize((cw, ch), Image.BILINEAR)
                self._photo = ImageTk.PhotoImage(img)
                self._canvas.delete("all")
                self._canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)

                # 녹화 중 REC 인디케이터
                if self._recording:
                    self._canvas.create_oval(
                        12, 12, 30, 30, fill="red", outline="",
                    )
                    self._canvas.create_text(
                        38, 21, text="REC",
                        fill="red", font=("Segoe UI", 11, "bold"),
                        anchor=tk.W,
                    )
        except queue.Empty:
            pass
        except Exception as e:
            print(f"[Display error] {e}")

        self._schedule_display()

    # ── 종료 ──────────────────────────────────────────────────────────────
    def _on_close(self):
        self._stop_camera()
        self.win.destroy()
