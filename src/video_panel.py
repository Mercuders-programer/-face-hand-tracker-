"""
video_panel.py — 영상 분석 패널

기능:
  - 영상 파일 로드 + 재생/일시정지
  - 타임라인 스크러버 (클릭/드래그로 탐색)
  - 얼굴/손 랜드마크 오버레이 토글
"""

import tkinter as tk
from tkinter import messagebox, filedialog, ttk
import threading
import cv2
import numpy as np
import mediapipe as mp
import os
from dataclasses import dataclass

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
    from .exporter import export_json, export_ae_keyframes, export_tracks_ae
    from . import insightface_detector as _if_det_mod
except ImportError:
    from tracker import (FrameData, VideoInfo, PersonData,
                         _extract_face, _extract_hand, _extract_pose,
                         _build_persons, MAX_PERSONS, PERSON_COLORS)
    from exporter import export_json, export_ae_keyframes, export_tracks_ae
    import insightface_detector as _if_det_mod

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

try:
    try:
        from .sd_cartoon import SDCartoon
    except ImportError:
        from sd_cartoon import SDCartoon
except Exception:
    SDCartoon = None

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
BG_DARK  = "#1a1a2e"
BG_PANEL = "#16213e"
ACCENT   = "#4a7fff"
TEXT_W   = "#e0e0ff"
TEXT_G   = "#8888aa"
TL_BG    = "#0f0f1f"
TL_H     = 30

# 얼굴 이미지 워핑에 사용할 랜드마크 인덱스 (6점)
_FACE_IMG_KPT = [33, 263, 4, 168, 61, 291]  # R.Eye.O, L.Eye.O, Nose.T, Nose.B, Mouth.R, Mouth.L

EYE_FRAC = 0.40  # 감지 실패 이미지: 콘텐츠 상단 → 눈 라인 비율


def _alpha_content_bbox(bgra, thr=16):
    """알파>thr인 불투명 콘텐츠 박스 (x0,y0,x1,y1). 전부 투명/빈 경우 전체 반환."""
    a = bgra[:, :, 3]
    ys, xs = np.where(a > thr)
    if xs.size == 0:
        h, w = bgra.shape[:2]
        return 0, 0, w, h
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _detect_eye_pivot(bgra, bbox, score_min=0.45):
    """콘텐츠 박스 안 어두운 대칭 블롭 쌍(만화 눈) → (pivot_norm, score). 미달 시 None."""
    h, w = bgra.shape[:2]
    x0, y0, x1, y1 = bbox
    fw, fh = x1 - x0, y1 - y0
    if fw < 8 or fh < 8:
        return None
    ry0, ry1 = int(y0 + fh * 0.20), int(y0 + fh * 0.65)   # 눈 예상 세로 구간
    gray = cv2.cvtColor(cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR), cv2.COLOR_BGR2GRAY)
    amask = bgra[ry0:ry1, x0:x1, 3] > 16
    if amask.sum() < 10:
        return None
    inv = 255.0 - gray[ry0:ry1, x0:x1].astype(np.float32)
    inv[~amask] = 0
    thr = np.percentile(inv[amask], 88)
    bw = ((inv >= thr) & amask).astype(np.uint8)
    n, _, stats, cent = cv2.connectedComponentsWithStats(bw, 8)
    min_area = fw * fh * 0.0008
    blobs = [(stats[i, cv2.CC_STAT_AREA], cent[i][0] + x0, cent[i][1] + ry0)
             for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    blobs.sort(reverse=True)
    best = None
    for i in range(len(blobs)):
        for j in range(i + 1, len(blobs)):
            a_i, x_i, y_i = blobs[i]
            a_j, x_j, y_j = blobs[j]
            dx, dy = abs(x_i - x_j), abs(y_i - y_j)
            if not (fw * 0.12 < dx < fw * 0.75 and dy < fh * 0.12):
                continue
            mid_x = (x_i + x_j) / 2.0
            s_sym  = 1.0 - min(abs(mid_x - (x0 + fw / 2.0)) / (fw * 0.25), 1.0)
            s_vert = 1.0 - min(dy / (fh * 0.12), 1.0)
            s_area = min(a_i, a_j) / max(a_i, a_j)
            score = (s_sym + s_vert + s_area) / 3.0
            if best is None or score > best[0]:
                best = (score, mid_x, (y_i + y_j) / 2.0)
    if best is None or best[0] < score_min:
        return None
    return (best[1] / w, best[2] / h), best[0]


@dataclass
class BodyPins:
    """앞모습 몸통 핀 4점 (이미지 좌표계): L.Shoulder / R.Shoulder / R.Hip / L.Hip."""
    img_l_shldr: tuple
    img_r_shldr: tuple
    img_r_hip:   tuple
    img_l_hip:   tuple

    def arrays(self):
        return [np.array(p, np.float64)
                for p in [self.img_l_shldr, self.img_r_shldr, self.img_r_hip, self.img_l_hip]]

    def is_valid(self, min_dist=8.0):
        pts = self.arrays()
        return all(np.linalg.norm(pts[i] - pts[j]) >= min_dist
                   for i, j in [(0, 1), (1, 2), (2, 3), (3, 0)])


@dataclass
class BodySidePins:
    """옆모습 몸통 핀 4점 (이미지 좌표계): 어깨 / 앞가슴 / 앞엉덩이 / 뒤허리."""
    img_shldr:       tuple
    img_front_chest: tuple
    img_front_hip:   tuple
    img_back_waist:  tuple

    def arrays(self):
        return [np.array(p, np.float64)
                for p in [self.img_shldr, self.img_front_chest,
                          self.img_front_hip, self.img_back_waist]]

    def is_valid(self, min_dist=8.0):
        pts = self.arrays()
        return all(np.linalg.norm(pts[i] - pts[j]) >= min_dist
                   for i, j in [(0, 1), (1, 2), (2, 3), (3, 0)])

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


def _similarity_2pt(src1, src2, dst1, dst2):
    """두 대응점으로 유사 변환(scale+rotation+translate) 2×3 행렬 반환."""
    sv = np.asarray(src2, dtype=np.float64) - np.asarray(src1, dtype=np.float64)
    dv = np.asarray(dst2, dtype=np.float64) - np.asarray(dst1, dtype=np.float64)
    sd = np.linalg.norm(sv)
    if sd < 1e-6:
        return None
    scale = np.linalg.norm(dv) / sd
    rot = np.arctan2(float(dv[1]), float(dv[0])) - np.arctan2(float(sv[1]), float(sv[0]))
    cr, sr = np.cos(rot) * scale, np.sin(rot) * scale
    M = np.float32([[cr, -sr, 0.0], [sr, cr, 0.0]])
    s1 = np.asarray(src1, dtype=np.float64)
    d1 = np.asarray(dst1, dtype=np.float64)
    M[0, 2] = float(d1[0]) - (M[0, 0] * float(s1[0]) + M[0, 1] * float(s1[1]))
    M[1, 2] = float(d1[1]) - (M[1, 0] * float(s1[0]) + M[1, 1] * float(s1[1]))
    return M


def _apply_face_img_overlay(overlay, face_res, w, h, face_img, face_img_pts,
                             eye_y_pct=55, eye_x_pct=50, size_pct=100,
                             ema_state: dict | None = None,
                             pivot=None, rotation_offset=0,
                             side_img=None, side_pts=None, side_threshold=45.0,
                             side_anchors=None,
                             side_eye_y_pct=55, side_eye_x_pct=50, side_size_pct=100,
                             side_ema_state: dict | None = None,
                             ref_h=None, side_ref_h=None,
                             feather_px=0, interp=cv2.INTER_LINEAR):
    """로드된 얼굴 이미지(BGRA)를 감지된 얼굴 위에 합성한다.
    side_anchors: {'eye':(nx,ny), 'nose':(nx,ny)} — 옆모습 2-point 유사변환
    """
    if not face_res.face_landmarks:
        if ema_state is not None:
            for _k in ('face_h', 'eye_cx', 'eye_cy', 'angle'):
                ema_state[_k] = None
        if side_ema_state is not None:
            for _k in ('face_h', 'eye_cx', 'eye_cy', 'angle',
                       's_eye_x', 's_eye_y', 's_nose_x', 's_nose_y',
                       's_ear_dist', 's_angle'):
                side_ema_state[_k] = None
        return
    for _lf in face_res.face_landmarks:
        # 옆모습 여부 → 사용할 이미지 선택
        yaw = getattr(_lf, 'yaw', 0.0)
        is_side = (side_img is not None and abs(yaw) > side_threshold)
        if is_side:
            _fi, _fp = side_img, side_pts
            _piv = None
        else:
            _fi, _fp = face_img, face_img_pts
            _piv = pivot
        if _fi is None:
            continue
        img_h, img_w = _fi.shape[:2]
        if len(_lf) < 264:
            continue
        r_eye = np.array([_lf[33].x * w, _lf[33].y * h], dtype=np.float64)
        l_eye = np.array([_lf[263].x * w, _lf[263].y * h], dtype=np.float64)
        eye_center = (r_eye + l_eye) / 2.0
        angle = float(np.degrees(np.arctan2(l_eye[1] - r_eye[1], l_eye[0] - r_eye[0])))

        # ── 옆모습 2-point 유사 변환 (눈+코 앵커 설정 시 우선 적용) ────────
        if is_side and side_anchors and side_anchors.get('eye') and side_anchors.get('nose'):
            _ae, _an = side_anchors['eye'], side_anchors['nose']
            src_eye_px  = np.array([img_w * _ae[0], img_h * _ae[1]], dtype=np.float64)
            src_nose_px = np.array([img_w * _an[0], img_h * _an[1]], dtype=np.float64)
            # yaw>0: 코가 눈 오른쪽 → 얼굴이 오른쪽 → 왼눈(lm[263])이 카메라 쪽
            _ear_idx     = 454 if yaw >= 0 else 234
            ear_px       = np.array([_lf[_ear_idx].x * w, _lf[_ear_idx].y * h], dtype=np.float64)
            raw_dst_nose = np.array([_lf[4].x * w, _lf[4].y * h], dtype=np.float64)
            raw_dst_eye  = (l_eye if yaw >= 0 else r_eye)
            # 귀-코 거리 (안정적 스케일 기준)
            raw_ear_dist = float(np.linalg.norm(ear_px - raw_dst_nose))
            # 눈→코 방향각 (회전 기준, 크기 미사용)
            _dx = float(raw_dst_eye[0]) - float(raw_dst_nose[0])
            _dy = float(raw_dst_eye[1]) - float(raw_dst_nose[1])
            raw_angle_en = float(np.degrees(np.arctan2(_dy, _dx)))
            # EMA 평활화 (옆모습 떨림 보정 슬라이더 적용)
            _se = side_ema_state if side_ema_state is not None else ema_state
            if _se is not None:
                # 1. 위치: 코 EMA
                _nx = _adaptive_ema_update(_se, 's_nose_x', float(raw_dst_nose[0]), catch_scale=25.0)
                _ny = _adaptive_ema_update(_se, 's_nose_y', float(raw_dst_nose[1]), catch_scale=25.0)
                # 2. 스케일: 귀-코 거리 EMA (핵심 안정화)
                s_ear_dist = _adaptive_ema_update(_se, 's_ear_dist', raw_ear_dist, catch_scale=15.0)
                # 3. 회전: 눈→코 각도 EMA
                s_angle_en = _adaptive_ema_update(_se, 's_angle', raw_angle_en, catch_scale=8.0)
                # 4. 역산: stable_scale → stable dst_eye
                img_en_dist  = float(np.linalg.norm(src_eye_px - src_nose_px))
                img_ear_dist = img_en_dist * 1.6  # 귀-코 ≈ 눈-코 × 1.6 (옆모습 해부학)
                stable_scale = s_ear_dist / max(img_ear_dist, 1.0)
                stable_dist  = stable_scale * img_en_dist
                _rad = np.radians(s_angle_en)
                dst_nose = np.array([_nx, _ny], dtype=np.float64)
                dst_eye  = dst_nose + stable_dist * np.array([np.cos(_rad), np.sin(_rad)])
            else:
                dst_eye, dst_nose = raw_dst_eye, raw_dst_nose
            M2 = _similarity_2pt(src_eye_px, src_nose_px, dst_eye, dst_nose)
            if M2 is None:
                continue
            warped = cv2.warpAffine(_fi, M2, (w, h),
                                    flags=interp,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(0, 0, 0, 0))

        elif _fp is not None:
            # ── Homography 정밀 모드 ─────────────────────────────────────
            if len(_lf) <= max(_FACE_IMG_KPT):
                continue
            raw_dst = np.float32([[_lf[i].x * w, _lf[i].y * h] for i in _FACE_IMG_KPT])
            # EMA 평활화: eye_center 오프셋을 전체 dst_pts에 적용
            _ema_h = (side_ema_state if (is_side and side_ema_state is not None)
                      else ema_state)
            if _ema_h is not None:
                raw_cx = float(eye_center[0])
                raw_cy = float(eye_center[1])
                s_cx = _adaptive_ema_update(_ema_h, 'eye_cx', raw_cx, catch_scale=40.0)
                s_cy = _adaptive_ema_update(_ema_h, 'eye_cy', raw_cy, catch_scale=40.0)
                dst_pts = raw_dst + np.float32([s_cx - raw_cx, s_cy - raw_cy])
            else:
                dst_pts = raw_dst
            M, _ = cv2.findHomography(_fp, dst_pts)
            if M is None:
                continue
            warped = cv2.warpPerspective(_fi, M, (w, h),
                                         flags=interp,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(0, 0, 0, 0))
            # Homography 모드: rotation_offset을 눈 중심 기준으로 추가 적용
            if rotation_offset != 0:
                rot_cx = float(eye_center[0])
                rot_cy = float(eye_center[1])
                R = cv2.getRotationMatrix2D((rot_cx, rot_cy), -rotation_offset, 1.0)
                warped = cv2.warpAffine(warped, R, (w, h),
                                        flags=interp,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=(0, 0, 0, 0))
        else:
            # ── Affine 자동 모드 (일러스트) ──────────────────────────────
            # side 여부에 따라 슬라이더/EMA 선택
            _ey_pct  = side_eye_y_pct if is_side else eye_y_pct
            _ex_pct  = side_eye_x_pct if is_side else eye_x_pct
            _sz_pct  = side_size_pct  if is_side else size_pct
            _ema     = (side_ema_state if (is_side and side_ema_state is not None)
                        else ema_state)
            if hasattr(_lf, 'bbox'):
                raw_face_h = float(_lf.bbox[3] - _lf.bbox[1])
            else:
                ys = [_lf[i].y * h for i in range(len(_lf))]
                raw_face_h = max(ys) - min(ys)
            raw_eye_cx, raw_eye_cy = float(eye_center[0]), float(eye_center[1])
            raw_angle = angle

            # ── EMA 평활화 적용 (떨림 제거) ──
            if _ema is not None:
                face_h_px  = _adaptive_ema_update(_ema, 'face_h', raw_face_h, catch_scale=30.0)
                _ec_x      = _adaptive_ema_update(_ema, 'eye_cx', raw_eye_cx, catch_scale=40.0)
                _ec_y      = _adaptive_ema_update(_ema, 'eye_cy', raw_eye_cy, catch_scale=40.0)
                angle      = _adaptive_ema_update(_ema, 'angle',  raw_angle,  catch_scale=10.0)
                eye_center = (_ec_x, _ec_y)
            else:
                face_h_px = raw_face_h

            if face_h_px <= 0:
                continue
            _rh = side_ref_h if is_side else ref_h
            _denom = _rh if _rh else img_h * 0.8  # 미지정 시 기존 동작 유지(하위호환)
            scale = face_h_px * (_sz_pct / 100.0) / max(_denom, 1.0)
            if _piv is not None:
                src_cx = img_w * _piv[0]
                src_cy = img_h * _piv[1]
            else:
                src_cx = img_w * (_ex_pct / 100.0)
                src_cy = img_h * (_ey_pct / 100.0)
            M = cv2.getRotationMatrix2D((src_cx, src_cy), -(angle + rotation_offset), scale)
            M[0, 2] += eye_center[0] - src_cx
            M[1, 2] += eye_center[1] - src_cy
            warped = cv2.warpAffine(_fi, M, (w, h),
                                    flags=interp,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(0, 0, 0, 0))

        _alpha_ch = warped[:, :, 3]
        if feather_px > 0:
            _ksize = int(feather_px) | 1
            _alpha_ch = cv2.GaussianBlur(_alpha_ch, (_ksize, _ksize), 0)
        alpha = _alpha_ch[:, :, np.newaxis].astype(np.float32) / 255.0
        overlay[:] = np.clip(
            warped[:, :, :3].astype(np.float32) * alpha
            + overlay.astype(np.float32) * (1.0 - alpha),
            0, 255,
        ).astype(np.uint8)


def _apply_arm_img_overlay(overlay, pose_res, w, h, arm_img,
                            anchor_y_pct=50, anchor_x_pct=50, size_pct=100,
                            ema_state: dict | None = None,
                            arm_pins=None, arm_seg_cache=None, side='right',
                            feather_px=0, interp=cv2.INTER_LINEAR):
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
                                    flags=interp,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(0, 0, 0, 0))

        _alpha_ch = warped[:, :, 3]
        if feather_px > 0:
            _ksize = int(feather_px) | 1
            _alpha_ch = cv2.GaussianBlur(_alpha_ch, (_ksize, _ksize), 0)
        alpha = _alpha_ch[:, :, np.newaxis].astype(np.float32) / 255.0
        overlay[:] = np.clip(
            warped[:, :, :3].astype(np.float32) * alpha
            + overlay.astype(np.float32) * (1.0 - alpha),
            0, 255,
        ).astype(np.uint8)


def _apply_leg_img_overlay(overlay, pose_res, w, h, leg_img,
                            size_pct=100,
                            ema_state: dict | None = None,
                            leg_pins=None, leg_seg_cache=None, side='right',
                            feather_px=0, interp=cv2.INTER_LINEAR):
    """로드된 다리 이미지(BGRA)를 무릎 위에 Puppet Pin 합성.
    side='right': 랜드마크 24/26/28/32, 'left': 23/25/27/31
    PuppetPins 재사용: img_shldr=엉덩이, img_elbow=무릎, img_wrist=발목, img_hand=발끝"""
    if not pose_res.pose_landmarks:
        if ema_state is not None:
            for _k in ('knee_x', 'knee_y', 'angle', 'leg_len',
                       'hip_x', 'hip_y', 'ankle_x', 'ankle_y',
                       'foot_x', 'foot_y'):
                ema_state[_k] = None
        return
    if side == 'right':
        hip_idx, knee_idx, ankle_idx, foot_idx = 24, 26, 28, 32
    else:
        hip_idx, knee_idx, ankle_idx, foot_idx = 23, 25, 27, 31
    img_h, img_w = leg_img.shape[:2]
    for _pl in pose_res.pose_landmarks:
        if len(_pl) <= max(hip_idx, knee_idx, ankle_idx):
            continue
        hip   = _pl[hip_idx]
        knee  = _pl[knee_idx]
        ankle = _pl[ankle_idx]
        if hip.visibility < 0.3 or knee.visibility < 0.3 or ankle.visibility < 0.3:
            continue
        raw_kx  = knee.x * w
        raw_ky  = knee.y * h
        raw_hx  = hip.x * w
        raw_hy  = hip.y * h
        raw_ax  = ankle.x * w
        raw_ay  = ankle.y * h
        raw_ang = float(np.degrees(np.arctan2(
            ankle.y - hip.y, ankle.x - hip.x)))
        raw_len = float(np.hypot(
            (knee.x - hip.x) * w, (knee.y - hip.y) * h))
        if ema_state is not None:
            kx  = _adaptive_ema_update(ema_state, 'knee_x',  raw_kx,  40.0)
            ky  = _adaptive_ema_update(ema_state, 'knee_y',  raw_ky,  40.0)
            hx  = _adaptive_ema_update(ema_state, 'hip_x',   raw_hx,  40.0)
            hy  = _adaptive_ema_update(ema_state, 'hip_y',   raw_hy,  40.0)
            ax  = _adaptive_ema_update(ema_state, 'ankle_x', raw_ax,  40.0)
            ay  = _adaptive_ema_update(ema_state, 'ankle_y', raw_ay,  40.0)
            ang = _adaptive_ema_update(ema_state, 'angle',   raw_ang, 10.0)
            lln = _adaptive_ema_update(ema_state, 'leg_len', raw_len, 30.0)
        else:
            kx, ky = raw_kx, raw_ky
            hx, hy = raw_hx, raw_hy
            ax, ay = raw_ax, raw_ay
            ang, lln = raw_ang, raw_len

        if leg_pins is not None and leg_seg_cache is not None and _PUPPET_AVAILABLE:
            # ── Puppet Pin 모드 ──
            vid_foot = None
            if leg_pins.img_hand is not None and len(_pl) > foot_idx:
                foot_lm = _pl[foot_idx]
                if foot_lm.visibility >= 0.2:
                    raw_fx = foot_lm.x * w
                    raw_fy = foot_lm.y * h
                    if ema_state is not None:
                        fx = _adaptive_ema_update(ema_state, 'foot_x', raw_fx, 40.0)
                        fy = _adaptive_ema_update(ema_state, 'foot_y', raw_fy, 40.0)
                    else:
                        fx, fy = raw_fx, raw_fy
                    vid_foot = (fx, fy)
            warped = apply_puppet_warp(
                leg_seg_cache, (hx, hy), (kx, ky), (ax, ay), w, h,
                vid_hand=vid_foot, size_pct=float(size_pct))
        else:
            # ── Legacy 모드: 단일 Affine ──
            scale  = lln * (size_pct / 100.0) / (img_h * 0.8)
            src_cx = img_w * 0.5
            src_cy = img_h * 0.2
            M = cv2.getRotationMatrix2D((src_cx, src_cy), -ang, scale)
            M[0, 2] += kx - src_cx
            M[1, 2] += ky - src_cy
            warped = cv2.warpAffine(leg_img, M, (w, h),
                                    flags=interp,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(0, 0, 0, 0))

        _alpha_ch = warped[:, :, 3]
        if feather_px > 0:
            _ksize = int(feather_px) | 1
            _alpha_ch = cv2.GaussianBlur(_alpha_ch, (_ksize, _ksize), 0)
        alpha = _alpha_ch[:, :, np.newaxis].astype(np.float32) / 255.0
        overlay[:] = np.clip(
            warped[:, :, :3].astype(np.float32) * alpha
            + overlay.astype(np.float32) * (1.0 - alpha),
            0, 255,
        ).astype(np.uint8)


def _apply_shoe_img_overlay(overlay, pose_res, w, h, shoe_img,
                             size_pct=100, ema_state=None, side='right',
                             feather_px=0, interp=cv2.INTER_LINEAR):
    """발목(ankle) 위치에 신발 이미지 합성.
    side='right': 26(knee)/28(ankle)/32(foot), 'left': 25/27/31"""
    if not pose_res.pose_landmarks:
        if ema_state is not None:
            for _k in ('ankle_x', 'ankle_y', 'angle', 'shin_len'):
                ema_state[_k] = None
        return
    if side == 'right':
        knee_idx, ankle_idx, foot_idx = 26, 28, 32
    else:
        knee_idx, ankle_idx, foot_idx = 25, 27, 31
    img_h, img_w = shoe_img.shape[:2]
    for _pl in pose_res.pose_landmarks:
        if len(_pl) <= max(knee_idx, ankle_idx):
            continue
        ankle = _pl[ankle_idx]
        knee  = _pl[knee_idx]
        if ankle.visibility < 0.3:
            continue
        raw_ax   = ankle.x * w
        raw_ay   = ankle.y * h
        raw_kx   = knee.x * w
        raw_ky   = knee.y * h
        raw_shin = float(np.hypot((ankle.x - knee.x) * w, (ankle.y - knee.y) * h))
        # 발끝 랜드마크 있으면 방향에 사용, 없으면 종아리 방향으로 추정
        if len(_pl) > foot_idx and _pl[foot_idx].visibility >= 0.2:
            raw_fx = _pl[foot_idx].x * w
            raw_fy = _pl[foot_idx].y * h
        else:
            raw_fx = raw_ax + (raw_ax - raw_kx) * 0.4
            raw_fy = raw_ay + (raw_ay - raw_ky) * 0.4
        raw_ang = float(np.degrees(np.arctan2(raw_fy - raw_ay, raw_fx - raw_ax)))
        if ema_state is not None:
            ax   = _adaptive_ema_update(ema_state, 'ankle_x',  raw_ax,   40.0)
            ay   = _adaptive_ema_update(ema_state, 'ankle_y',  raw_ay,   40.0)
            ang  = _adaptive_ema_update(ema_state, 'angle',    raw_ang,  10.0)
            shin = _adaptive_ema_update(ema_state, 'shin_len', raw_shin, 30.0)
        else:
            ax, ay = raw_ax, raw_ay
            ang, shin = raw_ang, raw_shin
        scale  = shin * (size_pct / 100.0) / img_h
        src_cx = img_w * 0.5
        src_cy = img_h * 0.2   # 이미지 상단 20% = 발목 위치
        M = cv2.getRotationMatrix2D((src_cx, src_cy), -ang, scale)
        M[0, 2] += ax - src_cx
        M[1, 2] += ay - src_cy
        warped = cv2.warpAffine(shoe_img, M, (w, h),
                                flags=interp,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0, 0))
        _alpha_ch = warped[:, :, 3]
        if feather_px > 0:
            _ksize = int(feather_px) | 1
            _alpha_ch = cv2.GaussianBlur(_alpha_ch, (_ksize, _ksize), 0)
        alpha = _alpha_ch[:, :, np.newaxis].astype(np.float32) / 255.0
        overlay[:] = np.clip(
            warped[:, :, :3].astype(np.float32) * alpha
            + overlay.astype(np.float32) * (1.0 - alpha),
            0, 255,
        ).astype(np.uint8)
        break  # 첫 번째 감지된 사람에만 적용


def _apply_glove_img_overlay(overlay, hand_res, w, h, glove_img,
                              size_pct=100, ema_state=None, side='right',
                              feather_px=0, interp=cv2.INTER_LINEAR):
    """손목(lm[0]) + 중지MCP(lm[9]) 방향으로 장갑 이미지 합성.
    src_cy=80% → 손목 위치 기준."""
    if not hand_res.hand_landmarks:
        if ema_state is not None:
            for _k in ('wrist_x', 'wrist_y', 'angle', 'palm_len'):
                ema_state[_k] = None
        return
    img_h, img_w = glove_img.shape[:2]
    for i, _hl in enumerate(hand_res.hand_landmarks):
        if len(_hl) < 10:
            continue
        hand_label = 'Right'
        if hand_res.handedness and i < len(hand_res.handedness) and hand_res.handedness[i]:
            hand_label = hand_res.handedness[i][0].category_name
        if side == 'right' and hand_label != 'Right':
            continue
        if side == 'left' and hand_label != 'Left':
            continue
        wrist = _hl[0]
        mcp   = _hl[9]
        raw_wx  = wrist.x * w
        raw_wy  = wrist.y * h
        raw_mx  = mcp.x * w
        raw_my  = mcp.y * h
        raw_ang = float(np.degrees(np.arctan2(raw_my - raw_wy, raw_mx - raw_wx)))
        raw_pln = float(np.hypot(raw_mx - raw_wx, raw_my - raw_wy))
        if raw_pln < 5:
            continue
        if ema_state is not None:
            wx  = _adaptive_ema_update(ema_state, 'wrist_x',  raw_wx,  40.0)
            wy  = _adaptive_ema_update(ema_state, 'wrist_y',  raw_wy,  40.0)
            ang = _adaptive_ema_update(ema_state, 'angle',    raw_ang, 10.0)
            pln = _adaptive_ema_update(ema_state, 'palm_len', raw_pln, 30.0)
        else:
            wx, wy = raw_wx, raw_wy
            ang, pln = raw_ang, raw_pln
        scale  = pln * (size_pct / 100.0) / (img_h * 0.4)
        src_cx = img_w * 0.5
        src_cy = img_h * 0.8   # 손목 위치 기준
        M = cv2.getRotationMatrix2D((src_cx, src_cy), -ang, scale)
        M[0, 2] += wx - src_cx
        M[1, 2] += wy - src_cy
        warped = cv2.warpAffine(glove_img, M, (w, h),
                                flags=interp,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0, 0))
        _alpha_ch = warped[:, :, 3]
        if feather_px > 0:
            _ksize = int(feather_px) | 1
            _alpha_ch = cv2.GaussianBlur(_alpha_ch, (_ksize, _ksize), 0)
        alpha = _alpha_ch[:, :, np.newaxis].astype(np.float32) / 255.0
        overlay[:] = np.clip(
            warped[:, :, :3].astype(np.float32) * alpha
            + overlay.astype(np.float32) * (1.0 - alpha),
            0, 255,
        ).astype(np.uint8)


def _apply_weapon_img_overlay(overlay, hand_res, w, h, weapon_img,
                               size_pct=100, ema_state=None, hand_side='right',
                               feather_px=0, interp=cv2.INTER_LINEAR):
    """손목(lm[0]) + 중지MCP(lm[9]) 방향으로 무기 이미지 합성.
    src_cy=90% → 손잡이(그립) 위치 기준."""
    if not hand_res.hand_landmarks:
        if ema_state is not None:
            for _k in ('wrist_x', 'wrist_y', 'angle', 'palm_len'):
                ema_state[_k] = None
        return
    img_h, img_w = weapon_img.shape[:2]
    for i, _hl in enumerate(hand_res.hand_landmarks):
        if len(_hl) < 10:
            continue
        hand_label = 'Right'
        if hand_res.handedness and i < len(hand_res.handedness) and hand_res.handedness[i]:
            hand_label = hand_res.handedness[i][0].category_name
        if hand_side == 'right' and hand_label != 'Right':
            continue
        if hand_side == 'left' and hand_label != 'Left':
            continue
        wrist = _hl[0]
        mcp   = _hl[9]
        raw_wx  = wrist.x * w
        raw_wy  = wrist.y * h
        raw_mx  = mcp.x * w
        raw_my  = mcp.y * h
        raw_ang = float(np.degrees(np.arctan2(raw_my - raw_wy, raw_mx - raw_wx)))
        raw_pln = float(np.hypot(raw_mx - raw_wx, raw_my - raw_wy))
        if raw_pln < 5:
            continue
        if ema_state is not None:
            wx  = _adaptive_ema_update(ema_state, 'wrist_x',  raw_wx,  40.0)
            wy  = _adaptive_ema_update(ema_state, 'wrist_y',  raw_wy,  40.0)
            ang = _adaptive_ema_update(ema_state, 'angle',    raw_ang, 10.0)
            pln = _adaptive_ema_update(ema_state, 'palm_len', raw_pln, 30.0)
        else:
            wx, wy = raw_wx, raw_wy
            ang, pln = raw_ang, raw_pln
        scale  = pln * (size_pct / 100.0) / (img_h * 0.17)
        src_cx = img_w * 0.5
        src_cy = img_h * 0.9   # 손잡이 위치 (이미지 하단 90%)
        M = cv2.getRotationMatrix2D((src_cx, src_cy), -ang, scale)
        M[0, 2] += wx - src_cx
        M[1, 2] += wy - src_cy
        warped = cv2.warpAffine(weapon_img, M, (w, h),
                                flags=interp,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0, 0))
        _alpha_ch = warped[:, :, 3]
        if feather_px > 0:
            _ksize = int(feather_px) | 1
            _alpha_ch = cv2.GaussianBlur(_alpha_ch, (_ksize, _ksize), 0)
        alpha = _alpha_ch[:, :, np.newaxis].astype(np.float32) / 255.0
        overlay[:] = np.clip(
            warped[:, :, :3].astype(np.float32) * alpha
            + overlay.astype(np.float32) * (1.0 - alpha),
            0, 255,
        ).astype(np.uint8)
        break  # 첫 번째 매칭 손만 적용


def _apply_body_front_overlay(overlay, pose_res, w, h, body_img,
                               size_pct=100, ema_state=None, body_pins=None,
                               feather_px=0, interp=cv2.INTER_LINEAR):
    """앞모습 몸통 이미지(BGRA)를 L.Shldr/R.Shldr/R.Hip/L.Hip 4점으로 Perspective 합성.
    body_pins 미설정 시 이미지 4코너를 랜드마크에 맞춤."""
    if not pose_res.pose_landmarks:
        return
    for _pl in pose_res.pose_landmarks:
        if len(_pl) <= 24:
            continue
        l_sh = _pl[11]; r_sh = _pl[12]
        l_hp = _pl[23]; r_hp = _pl[24]
        if min(l_sh.visibility, r_sh.visibility,
               l_hp.visibility, r_hp.visibility) < 0.3:
            continue
        raw_vals = [l_sh.x*w, l_sh.y*h, r_sh.x*w, r_sh.y*h,
                    r_hp.x*w, r_hp.y*h, l_hp.x*w, l_hp.y*h]
        _KEYS = ['b_lsx','b_lsy','b_rsx','b_rsy','b_rhx','b_rhy','b_lhx','b_lhy']
        if ema_state is not None:
            sv = [_adaptive_ema_update(ema_state, k, v, 40.0)
                  for k, v in zip(_KEYS, raw_vals)]
        else:
            sv = raw_vals
        vid_pts = np.float32([[sv[0],sv[1]],[sv[2],sv[3]],[sv[4],sv[5]],[sv[6],sv[7]]])
        if size_pct != 100:
            c = vid_pts.mean(axis=0)
            vid_pts = (c + (vid_pts - c) * (size_pct / 100.0)).astype(np.float32)
        if body_pins is not None:
            src_pts = np.float32([body_pins.img_l_shldr, body_pins.img_r_shldr,
                                   body_pins.img_r_hip,   body_pins.img_l_hip])
        else:
            ih, iw = body_img.shape[:2]
            src_pts = np.float32([[0,0],[iw,0],[iw,ih],[0,ih]])
        M, _ = cv2.findHomography(src_pts, vid_pts)
        if M is None:
            continue
        warped = cv2.warpPerspective(body_img, M, (w, h),
                                     flags=interp,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0, 0, 0, 0))
        _alpha_ch = warped[:, :, 3]
        if feather_px > 0:
            _ksize = int(feather_px) | 1
            _alpha_ch = cv2.GaussianBlur(_alpha_ch, (_ksize, _ksize), 0)
        alpha = _alpha_ch[:, :, np.newaxis].astype(np.float32) / 255.0
        overlay[:] = np.clip(
            warped[:, :, :3].astype(np.float32) * alpha
            + overlay.astype(np.float32) * (1.0 - alpha),
            0, 255,
        ).astype(np.uint8)


def _apply_body_side_overlay(overlay, pose_res, w, h, body_img,
                              size_pct=100, depth_pct=40,
                              offset_x=0, offset_y=0,
                              ema_state=None, body_pins=None,
                              feather_px=0, interp=cv2.INTER_LINEAR):
    """옆모습 몸통 이미지(BGRA)를 척추+수직방향 자동계산 4점으로 Perspective 합성.
    body_pins (BodySidePins): 어깨/앞가슴/앞엉덩이/뒤허리 이미지 핀.
    depth_pct: 몸 두께를 어깨너비 대비 % (기본 40).
    offset_x/y: 픽셀 단위 위치 보정.
    코 위치로 앞방향 자동 판별."""
    if not pose_res.pose_landmarks:
        return
    for _pl in pose_res.pose_landmarks:
        if len(_pl) <= 24:
            continue
        l_sh = _pl[11]; r_sh = _pl[12]
        l_hp = _pl[23]; r_hp = _pl[24]
        if (l_sh.visibility + r_sh.visibility) / 2 < 0.2:
            continue
        if (l_hp.visibility + r_hp.visibility) / 2 < 0.2:
            continue
        # 가중 평균으로 어깨/엉덩이 중심
        sw = l_sh.visibility + r_sh.visibility
        scx = (l_sh.x*l_sh.visibility + r_sh.x*r_sh.visibility) / sw * w
        scy = (l_sh.y*l_sh.visibility + r_sh.y*r_sh.visibility) / sw * h
        hw = l_hp.visibility + r_hp.visibility
        hcx = (l_hp.x*l_hp.visibility + r_hp.x*r_hp.visibility) / hw * w
        hcy = (l_hp.y*l_hp.visibility + r_hp.y*r_hp.visibility) / hw * h
        if ema_state is not None:
            scx = _adaptive_ema_update(ema_state, 'b_scx', scx, 40.0)
            scy = _adaptive_ema_update(ema_state, 'b_scy', scy, 40.0)
            hcx = _adaptive_ema_update(ema_state, 'b_hcx', hcx, 40.0)
            hcy = _adaptive_ema_update(ema_state, 'b_hcy', hcy, 40.0)
        spine_dx = hcx - scx; spine_dy = hcy - scy
        spine_len = float(np.hypot(spine_dx, spine_dy))
        if spine_len < 5:
            continue
        # 척추 수직 방향 (CW 회전 → 아래 척추 기준 오른쪽)
        perp_x = spine_dy / spine_len
        perp_y = -spine_dx / spine_len
        # 코 위치로 앞방향 판별
        nose_x = _pl[0].x * w if len(_pl) > 0 and _pl[0].visibility > 0.2 else scx
        avg_shld_x = (l_sh.x*l_sh.visibility + r_sh.x*r_sh.visibility) / sw * w
        facing = 1.0 if nose_x > avg_shld_x else -1.0
        # 몸 두께
        shoulder_width = abs(r_sh.x - l_sh.x) * w
        body_depth = max(shoulder_width, spine_len * 0.2) * (depth_pct / 100.0) * (size_pct / 100.0)
        # 4 비디오 코너: 어깨(뒤), 앞가슴, 앞엉덩이, 뒤허리
        vid_pts = np.float32([
            [scx + offset_x, scy + offset_y],
            [scx + facing*perp_x*body_depth + offset_x, scy + facing*perp_y*body_depth + offset_y],
            [hcx + facing*perp_x*body_depth + offset_x, hcy + facing*perp_y*body_depth + offset_y],
            [hcx + offset_x, hcy + offset_y],
        ])
        if body_pins is not None:
            src_pts = np.float32([body_pins.img_shldr, body_pins.img_front_chest,
                                   body_pins.img_front_hip, body_pins.img_back_waist])
        else:
            ih, iw = body_img.shape[:2]
            src_pts = np.float32([[0,0],[iw,0],[iw,ih],[0,ih]])
        M, _ = cv2.findHomography(src_pts, vid_pts)
        if M is None:
            continue
        warped = cv2.warpPerspective(body_img, M, (w, h),
                                     flags=interp,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0, 0, 0, 0))
        _alpha_ch = warped[:, :, 3]
        if feather_px > 0:
            _ksize = int(feather_px) | 1
            _alpha_ch = cv2.GaussianBlur(_alpha_ch, (_ksize, _ksize), 0)
        alpha = _alpha_ch[:, :, np.newaxis].astype(np.float32) / 255.0
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
        self._img_only_var        = tk.BooleanVar(value=False)
        self._show_anime_var   = tk.BooleanVar(value=False)
        self._anime_style_var  = tk.StringVar(value="whitebox")
        self._anime_bg_var     = tk.StringVar(value="original")
        self._anime_range_var  = tk.StringVar(value="person")
        self._anime_model_path = self._find_default_anime_model()
        self._anime_converter  = None   # AnimeGANConverter 캐시 (지연 로드)
        self._anime_cache      = None   # 재생 중 변환 프레임 캐시
        self._anime_skip       = 0
        self._anime_strength_var = tk.IntVar(value=3)  # bold/sd 강도 1~5
        self._sd_pipe          = None   # SDCartoon 캐시 (지연 로드)
        self._smooth_var = tk.IntVar(value=3)
        self._time_var           = tk.StringVar(value="00:00 / 00:00")
        self._zoom               = 1.0               # 줌 배율 (1.0 = 100%)
        self._zoom_var           = tk.StringVar(value="100%")
        self._pan_x              = 0                 # 패닝 오프셋 (확대 이미지 픽셀 기준)
        self._pan_y              = 0
        self._pan_start          = None              # 중간 버튼 드래그 시작점
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
        self._face_img_ref_h    = None   # Affine 스케일 기준 높이(px), None=img_h*0.8
        self._face_img_open     = None   # BGRA (입 벌림 이미지)
        self._face_img_open_pts = None
        self._face_img_open_ref_h = None
        self._mouth_thr_var     = tk.DoubleVar(value=0.12)
        self._eye_y_var     = tk.IntVar(value=55)   # 눈 위치 Y (%)
        self._eye_x_var     = tk.IntVar(value=50)   # 눈 위치 X (%)
        self._img_size_var  = tk.IntVar(value=100)  # 크기 배율 (%)
        self._ema_smooth_var = tk.IntVar(value=85)  # 떨림 보정 강도 (0~95)
        self._face_conf_var  = tk.DoubleVar(value=0.5)  # 얼굴 감지 신뢰도 임계값
        self._face_img_ema: dict = {
            'face_h': None, 'eye_cx': None, 'eye_cy': None, 'angle': None, 'alpha': 0.15,
        }
        self._face_img_z_var = tk.IntVar(value=6)   # Z 순서 (낮을수록 뒤)
        self._face_pivot     = None                  # (norm_x, norm_y) 피벗, None = 슬라이더 사용
        self._face_rot_var   = tk.IntVar(value=0)    # 추가 회전 오프셋 (°)
        self._face_img_side     = None               # BGRA (옆모습 이미지)
        self._face_img_side_pts = None
        self._face_img_side_ref_h = None
        self._face_img_side_anchors = None           # {'eye':…,'nose':…} 옆모습 2-point 앵커
        self._face_img_side_kps_n = None             # 정규화 kps (옆모습 피벗 피커용)
        self._side_thr_var      = tk.IntVar(value=45)  # 옆모습 전환 yaw 임계값 (°)
        self._side_eye_y_var    = tk.IntVar(value=55)  # 옆모습 눈 위치 Y (%)
        self._side_eye_x_var    = tk.IntVar(value=50)  # 옆모습 눈 위치 X (%)
        self._side_img_size_var = tk.IntVar(value=100) # 옆모습 크기 배율 (%)
        self._side_ema_smooth_var = tk.IntVar(value=85)# 옆모습 떨림 보정 강도
        self._face_img_side_ema: dict = {
            'face_h': None, 'eye_cx': None, 'eye_cy': None, 'angle': None,
            's_eye_x': None, 's_eye_y': None, 's_nose_x': None, 's_nose_y': None,
            's_ear_dist': None, 's_angle': None,
            'alpha': 0.15,
        }
        self._face_img_kps_n    = None               # 정규화 5-kps (피벗 피커 전용)

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
        self._arm_z_var      = tk.IntVar(value=5)   # Z 순서
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
        self._arm_l_z_var     = tk.IntVar(value=4)  # Z 순서

        # ── 오른다리 이미지 오버레이 상태
        self._leg_img_r       = None   # BGRA numpy array
        self._leg_size_var    = tk.IntVar(value=100)  # 크기 배율 (%)
        self._leg_smooth_var  = tk.IntVar(value=85)   # 떨림 보정 (0~95)
        self._leg_img_ema_r   = {
            'knee_x': None, 'knee_y': None,
            'hip_x':  None, 'hip_y':  None,
            'ankle_x': None, 'ankle_y': None,
            'foot_x':  None, 'foot_y':  None,
            'angle': None, 'leg_len': None, 'alpha': 0.15,
        }
        self._leg_pins_r      = None
        self._leg_seg_cache_r = None
        self._leg_img_btn_r   = None
        self._leg_img_lbl_r   = None
        self._leg_pin_btn_r   = None
        self._leg_pin_lbl_r   = None
        self._leg_r_z_var     = tk.IntVar(value=3)  # Z 순서

        # ── 왼다리 이미지 오버레이 상태
        self._leg_img_l       = None
        self._leg_img_ema_l   = {
            'knee_x': None, 'knee_y': None,
            'hip_x':  None, 'hip_y':  None,
            'ankle_x': None, 'ankle_y': None,
            'foot_x':  None, 'foot_y':  None,
            'angle': None, 'leg_len': None, 'alpha': 0.15,
        }
        self._leg_pins_l      = None
        self._leg_seg_cache_l = None
        self._leg_img_btn_l   = None
        self._leg_img_lbl_l   = None
        self._leg_pin_btn_l   = None
        self._leg_pin_lbl_l   = None
        self._leg_l_z_var     = tk.IntVar(value=2)  # Z 순서

        # ── 알파 엣지 페더링 + 고화질 보간 (전역)
        self._feather_var     = tk.IntVar(value=0)
        self._hq_var          = tk.BooleanVar(value=False)

        # ── 신발 이미지 오버레이 상태
        self._shoe_img_r      = None   # BGRA numpy array
        self._shoe_img_l      = None
        self._shoe_size_var   = tk.IntVar(value=100)
        self._shoe_smooth_var = tk.IntVar(value=85)
        self._shoe_img_ema_r  = {
            'ankle_x': None, 'ankle_y': None,
            'angle': None, 'shin_len': None, 'alpha': 0.15,
        }
        self._shoe_img_ema_l  = {
            'ankle_x': None, 'ankle_y': None,
            'angle': None, 'shin_len': None, 'alpha': 0.15,
        }
        self._shoe_img_btn_r  = None
        self._shoe_img_lbl_r  = None
        self._shoe_img_btn_l  = None
        self._shoe_img_lbl_l  = None
        self._shoe_r_z_var    = tk.IntVar(value=3)
        self._shoe_l_z_var    = tk.IntVar(value=3)

        # ── 장갑 이미지 오버레이 상태
        self._glove_img_r      = None
        self._glove_img_l      = None
        self._glove_size_var   = tk.IntVar(value=100)
        self._glove_smooth_var = tk.IntVar(value=85)
        self._glove_img_ema_r  = {
            'wrist_x': None, 'wrist_y': None,
            'angle': None, 'palm_len': None, 'alpha': 0.15,
        }
        self._glove_img_ema_l  = {
            'wrist_x': None, 'wrist_y': None,
            'angle': None, 'palm_len': None, 'alpha': 0.15,
        }
        self._glove_img_btn_r  = None
        self._glove_img_lbl_r  = None
        self._glove_img_btn_l  = None
        self._glove_img_lbl_l  = None
        self._glove_r_z_var    = tk.IntVar(value=5)
        self._glove_l_z_var    = tk.IntVar(value=5)

        # ── 무기 이미지 오버레이 상태
        self._weapon_img       = None
        self._weapon_hand_var  = tk.StringVar(value='right')
        self._weapon_size_var  = tk.IntVar(value=100)
        self._weapon_smooth_var = tk.IntVar(value=85)
        self._weapon_img_ema_r = {
            'wrist_x': None, 'wrist_y': None,
            'angle': None, 'palm_len': None, 'alpha': 0.15,
        }
        self._weapon_img_ema_l = {
            'wrist_x': None, 'wrist_y': None,
            'angle': None, 'palm_len': None, 'alpha': 0.15,
        }
        self._weapon_img_btn   = None
        self._weapon_img_lbl   = None
        self._weapon_z_var     = tk.IntVar(value=6)

        # ── 앞모습 몸통 이미지 오버레이 상태
        self._body_front_img      = None   # BGRA numpy array
        self._body_front_size_var = tk.IntVar(value=100)
        self._body_front_smooth_var = tk.IntVar(value=85)
        self._body_front_ema      = {
            'b_lsx': None, 'b_lsy': None, 'b_rsx': None, 'b_rsy': None,
            'b_rhx': None, 'b_rhy': None, 'b_lhx': None, 'b_lhy': None,
            'alpha': 0.15,
        }
        self._body_front_pins     = None   # BodyPins | None
        self._body_front_img_btn  = None
        self._body_front_img_lbl  = None
        self._body_front_pin_btn  = None
        self._body_front_pin_lbl  = None
        self._body_front_z_var    = tk.IntVar(value=1)  # Z 순서

        # ── 옆모습 몸통 이미지 오버레이 상태
        self._body_side_img       = None
        self._body_side_size_var  = tk.IntVar(value=100)
        self._body_side_depth_var = tk.IntVar(value=40)   # 몸 두께 %
        self._body_side_x_var     = tk.IntVar(value=0)    # X 오프셋 (px)
        self._body_side_y_var     = tk.IntVar(value=0)    # Y 오프셋 (px)
        self._body_side_smooth_var = tk.IntVar(value=85)
        self._body_side_ema       = {
            'b_scx': None, 'b_scy': None, 'b_hcx': None, 'b_hcy': None,
            'alpha': 0.15,
        }
        self._body_side_pins      = None   # BodySidePins | None
        self._body_side_img_btn   = None
        self._body_side_img_lbl   = None
        self._body_side_pin_btn   = None
        self._body_side_pin_lbl   = None
        self._body_side_z_var     = tk.IntVar(value=0)  # Z 순서

        # ── 포인트 트래킹 상태 (LK 광학 흐름) ─────────────────────────────
        self._track_points     = []     # [{"id","color","origin_frame","pos":{idx:(x,y)}}]
        self._track_pick_mode  = False  # '점 추가' 후 캔버스 클릭 대기 상태
        self._track_gray_cache = None   # 지연 생성 grayscale 프레임 리스트 (전 점 공유)
        self._track_next_id    = 1
        self._track_busy       = False  # 추적 워커 진행 중
        self._show_track_var   = tk.BooleanVar(value=True)
        self._track_status_var = tk.StringVar(value="")
        self._track_list_frame = None
        self._track_pick_btn   = None

        self._build_ui()
        self._on_ema_smooth_change()        # 슬라이더 초기값 → EMA alpha 동기화
        self._on_side_ema_smooth_change()   # 옆모습 EMA alpha 동기화
        self._on_arm_smooth_change()        # arm EMA alpha 동기화
        self._on_leg_smooth_change()        # leg EMA alpha 동기화
        self._on_body_smooth_change()       # body EMA alpha 동기화
        self._on_shoe_smooth_change()       # shoe EMA alpha 동기화
        self._on_glove_smooth_change()      # glove EMA alpha 동기화
        self._on_weapon_smooth_change()     # weapon EMA alpha 동기화
        self._init_mediapipe()
        for _v in (self._show_face, self._show_body, self._show_hands, self._show_names,
                   self._show_mosaic, self._img_only_var, self._hq_var,
                   self._show_track_var):
            _v.trace_add("write", lambda *_: self._refresh_frame())
        # 슬라이더 변수 → 값 변경 시 즉시 프레임 갱신
        _slider_vars = (
            # 얼굴 이미지
            self._eye_y_var, self._eye_x_var, self._img_size_var,
            self._face_rot_var, self._face_img_z_var, self._mouth_thr_var,
            self._side_thr_var, self._ema_smooth_var,
            self._side_eye_y_var, self._side_eye_x_var,
            self._side_img_size_var, self._side_ema_smooth_var,
            # 팔 이미지
            self._arm_y_var, self._arm_x_var, self._arm_size_var,
            self._arm_smooth_var, self._arm_z_var, self._arm_l_z_var,
            # 다리 이미지
            self._leg_size_var, self._leg_smooth_var,
            self._leg_r_z_var, self._leg_l_z_var,
            # 몸 앞/옆 이미지
            self._body_front_size_var, self._body_front_smooth_var, self._body_front_z_var,
            self._body_side_size_var, self._body_side_depth_var,
            self._body_side_x_var, self._body_side_y_var,
            self._body_side_smooth_var, self._body_side_z_var,
            # 페더링 (전역)
            self._feather_var,
            # 신발/장갑/무기
            self._shoe_size_var, self._shoe_smooth_var,
            self._shoe_r_z_var, self._shoe_l_z_var,
            self._glove_size_var, self._glove_smooth_var,
            self._glove_r_z_var, self._glove_l_z_var,
            self._weapon_size_var, self._weapon_smooth_var, self._weapon_z_var,
        )
        for _v in _slider_vars:
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
        self._canvas.bind("<Enter>",            self._canvas_wheel_enter)
        self._canvas.bind("<Leave>",            self._canvas_wheel_leave)
        self._canvas.bind("<ButtonPress-2>",    self._pan_start_cb)
        self._canvas.bind("<B2-Motion>",        self._pan_drag_cb)
        self._canvas.bind("<ButtonRelease-2>",  self._pan_end_cb)
        self._canvas.bind("<Button-1>",         self._on_track_click)

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
        tk.Label(ctrl, text="  |", font=("Segoe UI", 11), fg=TEXT_G, bg=BG_DARK).pack(side=tk.LEFT)
        tk.Label(
            ctrl, textvariable=self._zoom_var,
            font=("Segoe UI", 11, "bold"),
            fg=ACCENT, bg=BG_DARK,
        ).pack(side=tk.LEFT, padx=(4, 0))

    # ── 파일 정보 패널 ─────────────────────────────────────────────────────
    def _build_info_panel(self, parent):
        # 접기/펴기 섹션 목록 (- 키 토글용)
        self._panel_sections: list = []

        # 상단 accent 바
        tk.Frame(parent, bg=ACCENT, height=3).pack(fill=tk.X)

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
        nb = ttk.Notebook(parent, style="PH.TNotebook")
        nb.pack(fill=tk.BOTH, expand=True)
        tab1 = tk.Frame(nb, bg=BG_PANEL)
        tab2 = tk.Frame(nb, bg=BG_PANEL)
        nb.add(tab1, text="  기본  ")
        nb.add(tab2, text="  이미지  ")
        parent = tab1

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

        # ── 트래킹 (접기/펴기) ───────────────────────────────────────────────
        tk.Frame(parent, bg="#2a2a4a", height=1).pack(fill=tk.X, padx=10, pady=(8, 4))

        _tk_open = tk.BooleanVar(value=True)
        _tk_hdr = tk.Frame(parent, bg=BG_PANEL, cursor="hand2")
        _tk_hdr.pack(fill=tk.X)
        _tk_lbl = tk.Label(
            _tk_hdr, text="▼  트래킹",
            font=("Segoe UI", 10, "bold"),
            fg="#ffcc66", bg=BG_PANEL, anchor="w",
        )
        _tk_lbl.pack(fill=tk.X, padx=14, pady=(0, 4))
        _tk_sep = tk.Frame(parent, bg="#1e1e3a", height=1)
        _tk_sep.pack(fill=tk.X, padx=10, pady=(0, 4))
        _tk_body = tk.Frame(parent, bg=BG_PANEL)
        _tk_body.pack(fill=tk.X)

        def _toggle_track(_e=None):
            if _tk_open.get():
                _tk_body.pack_forget()
                _tk_lbl.config(text="▶  트래킹")
                _tk_open.set(False)
            else:
                _tk_body.pack(fill=tk.X, after=_tk_sep)
                _tk_lbl.config(text="▼  트래킹")
                _tk_open.set(True)

        _tk_hdr.bind("<Button-1>", _toggle_track)
        _tk_lbl.bind("<Button-1>", _toggle_track)
        self._panel_sections.append((_tk_open, _toggle_track))

        self._track_pick_btn = tk.Button(
            _tk_body, text="🎯 점 추가",
            font=("Segoe UI", 10, "bold"),
            bg="#1e3a5f", fg=TEXT_W,
            activebackground="#2a4f80", activeforeground="white",
            relief=tk.FLAT, cursor="hand2",
            pady=6, anchor="w", padx=12,
            command=self._toggle_track_pick,
        )
        self._track_pick_btn.pack(fill=tk.X, padx=10, pady=(0, 4))

        tk.Checkbutton(
            _tk_body, text="트래킹 표시", variable=self._show_track_var,
            font=("Segoe UI", 10),
            fg=TEXT_W, bg=BG_PANEL,
            selectcolor="#0f3460",
            activeforeground=TEXT_W, activebackground=BG_PANEL,
            anchor="w",
        ).pack(fill=tk.X, padx=10, pady=(0, 2))

        tk.Label(
            _tk_body, textvariable=self._track_status_var,
            font=("Segoe UI", 8), fg="#ffcc66", bg=BG_PANEL, anchor="w",
        ).pack(fill=tk.X, padx=14, pady=(0, 2))

        self._track_list_frame = tk.Frame(_tk_body, bg=BG_PANEL)
        self._track_list_frame.pack(fill=tk.X, padx=10, pady=(0, 2))

        _tk_btns = tk.Frame(_tk_body, bg=BG_PANEL)
        _tk_btns.pack(fill=tk.X, padx=10, pady=(2, 4))
        tk.Button(
            _tk_btns, text="🗑 전체 삭제",
            font=("Segoe UI", 9), bg="#3a1e1e", fg=TEXT_W,
            activebackground="#5a2a2a", activeforeground="white",
            relief=tk.FLAT, cursor="hand2", pady=4,
            command=self._clear_tracks,
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        tk.Button(
            _tk_btns, text="💾 트랙 내보내기",
            font=("Segoe UI", 9, "bold"), bg="#1e3a5f", fg=TEXT_W,
            activebackground="#2a4f80", activeforeground="white",
            relief=tk.FLAT, cursor="hand2", pady=4,
            command=self._export_tracks,
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))

        self._rebuild_track_list()

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
            _an_body, text="🎨 애니화 (미리보기+내보내기)",
            variable=self._show_anime_var,
            command=self._on_anime_toggle,
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
        self._anime_strength_anchor = _sf
        for _sv, _sl in [("whitebox", "카툰(깨끗)"), ("bold", "굵은카툰"),
                         ("sd", "SD(고품질)"), ("animegan", "AnimeGAN"),
                         ("opencv", "OpenCV")]:
            tk.Radiobutton(
                _sf, text=_sl, variable=self._anime_style_var, value=_sv,
                font=("Segoe UI", 9), fg=TEXT_W, bg=BG_PANEL,
                selectcolor="#0f3460",
                activeforeground=TEXT_W, activebackground=BG_PANEL,
                command=self._on_anime_style_change,
            ).pack(side=tk.LEFT, padx=(0, 6))

        # 강도 (굵은카툰/SD 전용)
        self._anime_strength_row = tk.Frame(_an_body, bg=BG_PANEL)
        self._anime_strength_row.pack(fill=tk.X, padx=14, pady=(0, 2))
        tk.Label(self._anime_strength_row, text="강도",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL,
                 ).pack(side=tk.LEFT, padx=(0, 4))
        tk.Scale(
            self._anime_strength_row, from_=1, to=5, orient=tk.HORIZONTAL,
            variable=self._anime_strength_var, length=120,
            bg=BG_PANEL, fg=TEXT_W, troughcolor="#0f3460",
            highlightthickness=0, showvalue=True,
            command=lambda _v: self._on_anime_opt_change(),
        ).pack(side=tk.LEFT)
        if self._anime_style_var.get() not in ("bold", "sd"):
            self._anime_strength_row.pack_forget()

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
                command=self._on_anime_opt_change,
            ).pack(side=tk.LEFT, padx=(0, 4))

        # 범위
        tk.Label(_an_body, text="  범위",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        _rf = tk.Frame(_an_body, bg=BG_PANEL)
        _rf.pack(fill=tk.X, padx=20, pady=(0, 4))
        for _rv, _rl in [("person", "사람만"), ("full", "전체화면")]:
            tk.Radiobutton(
                _rf, text=_rl, variable=self._anime_range_var, value=_rv,
                font=("Segoe UI", 9), fg=TEXT_W, bg=BG_PANEL,
                selectcolor="#0f3460",
                activeforeground=TEXT_W, activebackground=BG_PANEL,
                command=self._on_anime_opt_change,
            ).pack(side=tk.LEFT, padx=(0, 8))

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
        if self._anime_style_var.get() == "whitebox":
            self._anime_model_btn.config(state=tk.DISABLED)
            _model_lbl_text = "화이트박스 카툰 (내장 모델)"
        elif self._anime_model_path:
            _model_lbl_text = os.path.basename(self._anime_model_path)
        else:
            _model_lbl_text = "미선택 (OpenCV로 대체)"
        self._anime_model_lbl = tk.Label(
            _an_body, text=_model_lbl_text,
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

        parent = tab2  # 이미지 탭

        # ── 이미지만 렌더 ──
        tk.Checkbutton(
            parent, text="이미지만 렌더",
            variable=self._img_only_var,
            font=("Segoe UI", 10, "bold"),
            fg="#ffcc44", bg=BG_PANEL,
            selectcolor="#0f3460",
            activeforeground="#ffcc44", activebackground=BG_PANEL,
            anchor="w",
        ).pack(fill=tk.X, padx=10, pady=(4, 2))
        tk.Frame(parent, bg="#2a2a4a", height=1).pack(fill=tk.X, padx=10, pady=(2, 4))

        # ── 알파 엣지 페더링 슬라이더 (전역) ──
        tk.Label(parent, text="경계 부드럽기 (px)",
                 font=("Segoe UI", 8), fg="#aaddff", bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        tk.Scale(parent, from_=0, to=20, orient=tk.HORIZONTAL,
                 variable=self._feather_var, length=160,
                 bg=BG_PANEL, fg="#aaddff", troughcolor="#0f3460",
                 highlightthickness=0, showvalue=True,
                 ).pack(padx=10, pady=(0, 4))
        tk.Checkbutton(
            parent, text="고화질 모드 (LANCZOS4)",
            variable=self._hq_var,
            font=("Segoe UI", 9, "bold"),
            fg="#ffdd88", bg=BG_PANEL,
            selectcolor="#0f3460",
            activeforeground="#ffdd88", activebackground=BG_PANEL,
            anchor="w",
        ).pack(fill=tk.X, padx=10, pady=(0, 4))
        tk.Frame(parent, bg="#2a2a4a", height=1).pack(fill=tk.X, padx=10, pady=(0, 6))

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
        self._face_pivot_btn = tk.Button(
            _fi_body, text="⊕ 피벗 설정",
            font=("Segoe UI", 9),
            bg="#2a2a4a", fg="#88ddff",
            activebackground="#3a3a6a", activeforeground="#88ddff",
            relief=tk.FLAT, cursor="hand2",
            pady=3, anchor="w", padx=12,
            state=tk.DISABLED,
            command=self._open_face_pivot_picker,
        )
        self._face_pivot_btn.pack(fill=tk.X, padx=10, pady=(0, 4))
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
        tk.Label(_fi_body, text="추가 회전 (°)",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        tk.Scale(_fi_body, from_=-180, to=180, orient=tk.HORIZONTAL,
                 variable=self._face_rot_var, length=160,
                 bg=BG_PANEL, fg="#ff9955", troughcolor="#0f3460",
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
        tk.Label(_fi_body, text="Z 순서 (낮을수록 뒤에 렌더)",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        tk.Scale(_fi_body, from_=0, to=10, orient=tk.HORIZONTAL,
                 variable=self._face_img_z_var, length=160,
                 bg=BG_PANEL, fg="#aaaacc", troughcolor="#0f3460",
                 highlightthickness=0, showvalue=True,
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

        tk.Frame(_fi_body, bg="#2a2a4a", height=1).pack(fill=tk.X, padx=10, pady=(2, 6))

        self._face_img_side_btn = tk.Button(
            _fi_body, text="🖼  옆모습 이미지 로드",
            font=("Segoe UI", 10, "bold"),
            bg="#1e3a5f", fg=TEXT_W,
            activebackground="#2a4f80", activeforeground="white",
            relief=tk.FLAT, cursor="hand2",
            pady=6, anchor="w", padx=12,
            command=self._toggle_face_img_side,
        )
        self._face_img_side_btn.pack(fill=tk.X, padx=10, pady=(0, 2))
        self._face_img_side_lbl = tk.Label(
            _fi_body, text="미선택",
            font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
            wraplength=178,
        )
        self._face_img_side_lbl.pack(fill=tk.X, padx=14, pady=(0, 2))
        self._face_side_pivot_btn = tk.Button(
            _fi_body, text="⊕ 앵커 설정 (옆)",
            font=("Segoe UI", 9),
            bg="#2a2a4a", fg="#ffaa55",
            activebackground="#3a3a6a", activeforeground="#ffaa55",
            relief=tk.FLAT, cursor="hand2",
            pady=3, anchor="w", padx=12,
            state=tk.DISABLED,
            command=self._open_face_side_pivot_picker,
        )
        self._face_side_pivot_btn.pack(fill=tk.X, padx=10, pady=(0, 4))
        tk.Label(_fi_body, text="전환 각도 (yaw°)",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        self._side_thr_scale = tk.Scale(
            _fi_body, from_=10, to=80, orient=tk.HORIZONTAL,
            variable=self._side_thr_var, length=160,
            bg=BG_PANEL, fg="#ffaa55", troughcolor="#0f3460",
            highlightthickness=0, showvalue=True,
            state=tk.DISABLED,
        )
        self._side_thr_scale.pack(padx=10, pady=(0, 4))
        tk.Label(_fi_body, text="위치 Y (%)",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        tk.Scale(_fi_body, from_=10, to=90, orient=tk.HORIZONTAL,
                 variable=self._side_eye_y_var, length=160,
                 bg=BG_PANEL, fg=TEXT_W, troughcolor="#0f3460",
                 highlightthickness=0, showvalue=True,
                 ).pack(padx=10, pady=(0, 2))
        tk.Label(_fi_body, text="위치 X (%)",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        tk.Scale(_fi_body, from_=10, to=90, orient=tk.HORIZONTAL,
                 variable=self._side_eye_x_var, length=160,
                 bg=BG_PANEL, fg=TEXT_W, troughcolor="#0f3460",
                 highlightthickness=0, showvalue=True,
                 ).pack(padx=10, pady=(0, 2))
        tk.Label(_fi_body, text="크기 (%)",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        tk.Scale(_fi_body, from_=30, to=300, orient=tk.HORIZONTAL,
                 variable=self._side_img_size_var, length=160,
                 bg=BG_PANEL, fg=TEXT_W, troughcolor="#0f3460",
                 highlightthickness=0, showvalue=True,
                 ).pack(padx=10, pady=(0, 2))
        tk.Label(_fi_body, text="떨림 보정 (0=없음  →  95=최대)",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        tk.Scale(_fi_body, from_=0, to=95, orient=tk.HORIZONTAL,
                 variable=self._side_ema_smooth_var, length=160,
                 bg=BG_PANEL, fg="#88ddff", troughcolor="#0f3460",
                 highlightthickness=0, showvalue=True,
                 command=self._on_side_ema_smooth_change,
                 ).pack(padx=10, pady=(0, 4))

        tk.Frame(parent, bg="#2a2a4a", height=1).pack(fill=tk.X, padx=10, pady=(4, 8))

        # ── 오른팔 이미지 (접기/펴기) ──
        _arm_open = tk.BooleanVar(value=True)
        _arm_hdr = tk.Frame(parent, bg=BG_PANEL, cursor="hand2")
        _arm_hdr.pack(fill=tk.X)
        _arm_lbl = tk.Label(
            _arm_hdr, text="▼  오른팔 이미지",
            font=("Segoe UI", 10, "bold"),
            fg=TEXT_G, bg=BG_PANEL, anchor="w",
        )
        _arm_lbl.pack(fill=tk.X, padx=14, pady=(0, 4))
        _arm_sep = tk.Frame(parent, bg="#1e1e3a", height=1)
        _arm_sep.pack(fill=tk.X, padx=10, pady=(0, 4))
        _arm_body = tk.Frame(parent, bg=BG_PANEL)
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

        self._arm_img_btn = tk.Button(
            _arm_body, text="🦾  이미지 로드",
            font=("Segoe UI", 10, "bold"),
            bg="#1e3a5f", fg=TEXT_W,
            activebackground="#2a4f80", activeforeground="white",
            relief=tk.FLAT, cursor="hand2",
            pady=6, anchor="w", padx=12,
            command=lambda: self._toggle_arm_image(side='right'),
        )
        self._arm_img_btn.pack(fill=tk.X, padx=10, pady=(0, 2))
        self._arm_img_lbl = tk.Label(
            _arm_body, text="미선택",
            font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
            wraplength=178,
        )
        self._arm_img_lbl.pack(fill=tk.X, padx=14, pady=(0, 2))
        tk.Label(_arm_body, text="앵커 Y (%)",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        tk.Scale(_arm_body, from_=10, to=90, orient=tk.HORIZONTAL,
                 variable=self._arm_y_var, length=160,
                 bg=BG_PANEL, fg=TEXT_W, troughcolor="#0f3460",
                 highlightthickness=0, showvalue=True,
                 ).pack(padx=10, pady=(0, 2))
        tk.Label(_arm_body, text="앵커 X (%)",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        tk.Scale(_arm_body, from_=10, to=90, orient=tk.HORIZONTAL,
                 variable=self._arm_x_var, length=160,
                 bg=BG_PANEL, fg=TEXT_W, troughcolor="#0f3460",
                 highlightthickness=0, showvalue=True,
                 ).pack(padx=10, pady=(0, 2))
        tk.Label(_arm_body, text="크기 (%)",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        tk.Scale(_arm_body, from_=30, to=300, orient=tk.HORIZONTAL,
                 variable=self._arm_size_var, length=160,
                 bg=BG_PANEL, fg=TEXT_W, troughcolor="#0f3460",
                 highlightthickness=0, showvalue=True,
                 ).pack(padx=10, pady=(0, 2))
        tk.Label(_arm_body, text="떨림 보정 (0=없음  →  95=최대)",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        tk.Scale(_arm_body, from_=0, to=95, orient=tk.HORIZONTAL,
                 variable=self._arm_smooth_var, length=160,
                 bg=BG_PANEL, fg="#88ddff", troughcolor="#0f3460",
                 highlightthickness=0, showvalue=True,
                 command=self._on_arm_smooth_change,
                 ).pack(padx=10, pady=(0, 4))

        tk.Label(_arm_body, text="Z 순서 (낮을수록 뒤에 렌더)",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        tk.Scale(_arm_body, from_=0, to=10, orient=tk.HORIZONTAL,
                 variable=self._arm_z_var, length=160,
                 bg=BG_PANEL, fg="#aaaacc", troughcolor="#0f3460",
                 highlightthickness=0, showvalue=True,
                 ).pack(padx=10, pady=(0, 4))
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

        tk.Frame(parent, bg="#2a2a4a", height=1).pack(fill=tk.X, padx=10, pady=(4, 8))

        # ── 왼팔 이미지 (접기/펴기) ──
        _arml_open = tk.BooleanVar(value=True)
        _arml_hdr = tk.Frame(parent, bg=BG_PANEL, cursor="hand2")
        _arml_hdr.pack(fill=tk.X)
        _arml_lbl = tk.Label(
            _arml_hdr, text="▼  왼팔 이미지",
            font=("Segoe UI", 10, "bold"),
            fg=TEXT_G, bg=BG_PANEL, anchor="w",
        )
        _arml_lbl.pack(fill=tk.X, padx=14, pady=(0, 4))
        _arml_sep = tk.Frame(parent, bg="#1e1e3a", height=1)
        _arml_sep.pack(fill=tk.X, padx=10, pady=(0, 4))
        _arml_body = tk.Frame(parent, bg=BG_PANEL)
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

        self._arm_img_btn_l = tk.Button(
            _arml_body, text="🦾  이미지 로드",
            font=("Segoe UI", 10, "bold"),
            bg="#1e3a5f", fg=TEXT_W,
            activebackground="#2a4f80", activeforeground="white",
            relief=tk.FLAT, cursor="hand2",
            pady=6, anchor="w", padx=12,
            command=lambda: self._toggle_arm_image(side='left'),
        )
        self._arm_img_btn_l.pack(fill=tk.X, padx=10, pady=(0, 2))
        self._arm_img_lbl_l = tk.Label(
            _arml_body, text="미선택",
            font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
            wraplength=178,
        )
        self._arm_img_lbl_l.pack(fill=tk.X, padx=14, pady=(0, 2))

        tk.Label(_arml_body, text="Z 순서 (낮을수록 뒤에 렌더)",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        tk.Scale(_arml_body, from_=0, to=10, orient=tk.HORIZONTAL,
                 variable=self._arm_l_z_var, length=160,
                 bg=BG_PANEL, fg="#aaaacc", troughcolor="#0f3460",
                 highlightthickness=0, showvalue=True,
                 ).pack(padx=10, pady=(0, 4))
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

        tk.Frame(parent, bg="#2a2a4a", height=1).pack(fill=tk.X, padx=10, pady=(4, 8))

        # ── 오른다리 이미지 (접기/펴기) ──
        _legr_open = tk.BooleanVar(value=True)
        _legr_hdr = tk.Frame(parent, bg=BG_PANEL, cursor="hand2")
        _legr_hdr.pack(fill=tk.X)
        _legr_lbl = tk.Label(
            _legr_hdr, text="▼  오른발 이미지",
            font=("Segoe UI", 10, "bold"),
            fg=TEXT_G, bg=BG_PANEL, anchor="w",
        )
        _legr_lbl.pack(fill=tk.X, padx=14, pady=(0, 4))
        _legr_sep = tk.Frame(parent, bg="#1e1e3a", height=1)
        _legr_sep.pack(fill=tk.X, padx=10, pady=(0, 4))
        _legr_body = tk.Frame(parent, bg=BG_PANEL)
        _legr_body.pack(fill=tk.X)

        def _toggle_legr_img(_e=None):
            if _legr_open.get():
                _legr_body.pack_forget()
                _legr_lbl.config(text="▶  오른발 이미지")
                _legr_open.set(False)
            else:
                _legr_body.pack(fill=tk.X, after=_legr_sep)
                _legr_lbl.config(text="▼  오른발 이미지")
                _legr_open.set(True)

        _legr_hdr.bind("<Button-1>", _toggle_legr_img)
        _legr_lbl.bind("<Button-1>", _toggle_legr_img)
        self._panel_sections.append((_legr_open, _toggle_legr_img))

        self._leg_img_btn_r = tk.Button(
            _legr_body, text="🦵  이미지 로드",
            font=("Segoe UI", 10, "bold"),
            bg="#1e3a5f", fg=TEXT_W,
            activebackground="#2a4f80", activeforeground="white",
            relief=tk.FLAT, cursor="hand2",
            pady=6, anchor="w", padx=12,
            command=lambda: self._toggle_leg_image(side='right'),
        )
        self._leg_img_btn_r.pack(fill=tk.X, padx=10, pady=(0, 2))
        self._leg_img_lbl_r = tk.Label(
            _legr_body, text="미선택",
            font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
            wraplength=178,
        )
        self._leg_img_lbl_r.pack(fill=tk.X, padx=14, pady=(0, 2))
        tk.Label(_legr_body, text="크기 (%)",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        tk.Scale(_legr_body, from_=30, to=300, orient=tk.HORIZONTAL,
                 variable=self._leg_size_var, length=160,
                 bg=BG_PANEL, fg=TEXT_W, troughcolor="#0f3460",
                 highlightthickness=0, showvalue=True,
                 ).pack(padx=10, pady=(0, 2))
        tk.Label(_legr_body, text="떨림 보정 (0=없음  →  95=최대)",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        tk.Scale(_legr_body, from_=0, to=95, orient=tk.HORIZONTAL,
                 variable=self._leg_smooth_var, length=160,
                 bg=BG_PANEL, fg="#88bbff", troughcolor="#0f3460",
                 highlightthickness=0, showvalue=True,
                 command=self._on_leg_smooth_change,
                 ).pack(padx=10, pady=(0, 4))
        tk.Label(_legr_body, text="Z 순서 (낮을수록 뒤에 렌더)",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        tk.Scale(_legr_body, from_=0, to=10, orient=tk.HORIZONTAL,
                 variable=self._leg_r_z_var, length=160,
                 bg=BG_PANEL, fg="#aaaacc", troughcolor="#0f3460",
                 highlightthickness=0, showvalue=True,
                 ).pack(padx=10, pady=(0, 4))
        tk.Frame(_legr_body, bg="#2a2a4a", height=1).pack(fill=tk.X, padx=10, pady=(4, 4))
        self._leg_pin_lbl_r = tk.Label(
            _legr_body, text="피벗 미설정",
            font=("Segoe UI", 8), fg="#ffaa44", bg=BG_PANEL, anchor="w",
        )
        self._leg_pin_lbl_r.pack(fill=tk.X, padx=14, pady=(0, 2))
        self._leg_pin_btn_r = tk.Button(
            _legr_body, text="🎯 피벗 설정",
            font=("Segoe UI", 9, "bold"), bg="#2a3f5f", fg=TEXT_W,
            activebackground="#3a5a80", activeforeground="white",
            relief=tk.FLAT, cursor="hand2", pady=5, padx=10,
            command=lambda: self._open_leg_pin_picker(side='right'), state=tk.DISABLED,
        )
        self._leg_pin_btn_r.pack(fill=tk.X, padx=10, pady=(0, 4))

        tk.Frame(parent, bg="#2a2a4a", height=1).pack(fill=tk.X, padx=10, pady=(4, 8))

        # ── 왼다리 이미지 (접기/펴기) ──
        _legl_open = tk.BooleanVar(value=True)
        _legl_hdr = tk.Frame(parent, bg=BG_PANEL, cursor="hand2")
        _legl_hdr.pack(fill=tk.X)
        _legl_lbl = tk.Label(
            _legl_hdr, text="▼  왼발 이미지",
            font=("Segoe UI", 10, "bold"),
            fg=TEXT_G, bg=BG_PANEL, anchor="w",
        )
        _legl_lbl.pack(fill=tk.X, padx=14, pady=(0, 4))
        _legl_sep = tk.Frame(parent, bg="#1e1e3a", height=1)
        _legl_sep.pack(fill=tk.X, padx=10, pady=(0, 4))
        _legl_body = tk.Frame(parent, bg=BG_PANEL)
        _legl_body.pack(fill=tk.X)

        def _toggle_legl_img(_e=None):
            if _legl_open.get():
                _legl_body.pack_forget()
                _legl_lbl.config(text="▶  왼발 이미지")
                _legl_open.set(False)
            else:
                _legl_body.pack(fill=tk.X, after=_legl_sep)
                _legl_lbl.config(text="▼  왼발 이미지")
                _legl_open.set(True)

        _legl_hdr.bind("<Button-1>", _toggle_legl_img)
        _legl_lbl.bind("<Button-1>", _toggle_legl_img)
        self._panel_sections.append((_legl_open, _toggle_legl_img))

        self._leg_img_btn_l = tk.Button(
            _legl_body, text="🦵  이미지 로드",
            font=("Segoe UI", 10, "bold"),
            bg="#1e3a5f", fg=TEXT_W,
            activebackground="#2a4f80", activeforeground="white",
            relief=tk.FLAT, cursor="hand2",
            pady=6, anchor="w", padx=12,
            command=lambda: self._toggle_leg_image(side='left'),
        )
        self._leg_img_btn_l.pack(fill=tk.X, padx=10, pady=(0, 2))
        self._leg_img_lbl_l = tk.Label(
            _legl_body, text="미선택",
            font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
            wraplength=178,
        )
        self._leg_img_lbl_l.pack(fill=tk.X, padx=14, pady=(0, 2))

        tk.Label(_legl_body, text="Z 순서 (낮을수록 뒤에 렌더)",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        tk.Scale(_legl_body, from_=0, to=10, orient=tk.HORIZONTAL,
                 variable=self._leg_l_z_var, length=160,
                 bg=BG_PANEL, fg="#aaaacc", troughcolor="#0f3460",
                 highlightthickness=0, showvalue=True,
                 ).pack(padx=10, pady=(0, 4))
        # ── Puppet Pin UI (왼다리) ──
        tk.Frame(_legl_body, bg="#2a2a4a", height=1).pack(fill=tk.X, padx=10, pady=(4, 4))
        self._leg_pin_lbl_l = tk.Label(
            _legl_body, text="피벗 미설정",
            font=("Segoe UI", 8), fg="#ffaa44", bg=BG_PANEL, anchor="w",
        )
        self._leg_pin_lbl_l.pack(fill=tk.X, padx=14, pady=(0, 2))
        self._leg_pin_btn_l = tk.Button(
            _legl_body, text="🎯 피벗 설정",
            font=("Segoe UI", 9, "bold"), bg="#2a3f5f", fg=TEXT_W,
            activebackground="#3a5a80", activeforeground="white",
            relief=tk.FLAT, cursor="hand2", pady=5, padx=10,
            command=lambda: self._open_leg_pin_picker(side='left'), state=tk.DISABLED,
        )
        self._leg_pin_btn_l.pack(fill=tk.X, padx=10, pady=(0, 4))

        tk.Frame(parent, bg="#2a2a4a", height=1).pack(fill=tk.X, padx=10, pady=(4, 8))

        # ── 오른발 신발 이미지 (접기/펴기) ──
        _shoer_open = tk.BooleanVar(value=True)
        _shoer_hdr = tk.Frame(parent, bg=BG_PANEL, cursor="hand2")
        _shoer_hdr.pack(fill=tk.X)
        _shoer_lbl = tk.Label(
            _shoer_hdr, text="▼  오른발 신발 이미지",
            font=("Segoe UI", 10, "bold"), fg=TEXT_G, bg=BG_PANEL, anchor="w",
        )
        _shoer_lbl.pack(fill=tk.X, padx=14, pady=(0, 4))
        _shoer_sep = tk.Frame(parent, bg="#1e1e3a", height=1)
        _shoer_sep.pack(fill=tk.X, padx=10, pady=(0, 4))
        _shoer_body = tk.Frame(parent, bg=BG_PANEL)
        _shoer_body.pack(fill=tk.X)

        def _toggle_shoer_img(_e=None):
            if _shoer_open.get():
                _shoer_body.pack_forget()
                _shoer_lbl.config(text="▶  오른발 신발 이미지")
                _shoer_open.set(False)
            else:
                _shoer_body.pack(fill=tk.X, after=_shoer_sep)
                _shoer_lbl.config(text="▼  오른발 신발 이미지")
                _shoer_open.set(True)

        _shoer_hdr.bind("<Button-1>", _toggle_shoer_img)
        _shoer_lbl.bind("<Button-1>", _toggle_shoer_img)
        self._panel_sections.append((_shoer_open, _toggle_shoer_img))

        self._shoe_img_btn_r = tk.Button(
            _shoer_body, text="👟  이미지 로드",
            font=("Segoe UI", 10, "bold"), bg="#1e3a5f", fg=TEXT_W,
            activebackground="#2a4f80", activeforeground="white",
            relief=tk.FLAT, cursor="hand2", pady=6, anchor="w", padx=12,
            command=lambda: self._toggle_shoe_image(side='right'),
        )
        self._shoe_img_btn_r.pack(fill=tk.X, padx=10, pady=(0, 2))
        self._shoe_img_lbl_r = tk.Label(
            _shoer_body, text="미선택",
            font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w", wraplength=178,
        )
        self._shoe_img_lbl_r.pack(fill=tk.X, padx=14, pady=(0, 2))
        tk.Label(_shoer_body, text="크기 (%)",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        tk.Scale(_shoer_body, from_=30, to=300, orient=tk.HORIZONTAL,
                 variable=self._shoe_size_var, length=160,
                 bg=BG_PANEL, fg=TEXT_W, troughcolor="#0f3460",
                 highlightthickness=0, showvalue=True,
                 ).pack(padx=10, pady=(0, 2))
        tk.Label(_shoer_body, text="떨림 보정 (0=없음  →  95=최대)",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        tk.Scale(_shoer_body, from_=0, to=95, orient=tk.HORIZONTAL,
                 variable=self._shoe_smooth_var, length=160,
                 bg=BG_PANEL, fg="#88bbff", troughcolor="#0f3460",
                 highlightthickness=0, showvalue=True,
                 command=self._on_shoe_smooth_change,
                 ).pack(padx=10, pady=(0, 2))
        tk.Label(_shoer_body, text="Z 순서",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        tk.Scale(_shoer_body, from_=0, to=10, orient=tk.HORIZONTAL,
                 variable=self._shoe_r_z_var, length=160,
                 bg=BG_PANEL, fg="#aaaacc", troughcolor="#0f3460",
                 highlightthickness=0, showvalue=True,
                 ).pack(padx=10, pady=(0, 4))

        tk.Frame(parent, bg="#2a2a4a", height=1).pack(fill=tk.X, padx=10, pady=(4, 8))

        # ── 왼발 신발 이미지 (접기/펴기) ──
        _shoel_open = tk.BooleanVar(value=True)
        _shoel_hdr = tk.Frame(parent, bg=BG_PANEL, cursor="hand2")
        _shoel_hdr.pack(fill=tk.X)
        _shoel_lbl = tk.Label(
            _shoel_hdr, text="▼  왼발 신발 이미지",
            font=("Segoe UI", 10, "bold"), fg=TEXT_G, bg=BG_PANEL, anchor="w",
        )
        _shoel_lbl.pack(fill=tk.X, padx=14, pady=(0, 4))
        _shoel_sep = tk.Frame(parent, bg="#1e1e3a", height=1)
        _shoel_sep.pack(fill=tk.X, padx=10, pady=(0, 4))
        _shoel_body = tk.Frame(parent, bg=BG_PANEL)
        _shoel_body.pack(fill=tk.X)

        def _toggle_shoel_img(_e=None):
            if _shoel_open.get():
                _shoel_body.pack_forget()
                _shoel_lbl.config(text="▶  왼발 신발 이미지")
                _shoel_open.set(False)
            else:
                _shoel_body.pack(fill=tk.X, after=_shoel_sep)
                _shoel_lbl.config(text="▼  왼발 신발 이미지")
                _shoel_open.set(True)

        _shoel_hdr.bind("<Button-1>", _toggle_shoel_img)
        _shoel_lbl.bind("<Button-1>", _toggle_shoel_img)
        self._panel_sections.append((_shoel_open, _toggle_shoel_img))

        self._shoe_img_btn_l = tk.Button(
            _shoel_body, text="👟  이미지 로드",
            font=("Segoe UI", 10, "bold"), bg="#1e3a5f", fg=TEXT_W,
            activebackground="#2a4f80", activeforeground="white",
            relief=tk.FLAT, cursor="hand2", pady=6, anchor="w", padx=12,
            command=lambda: self._toggle_shoe_image(side='left'),
        )
        self._shoe_img_btn_l.pack(fill=tk.X, padx=10, pady=(0, 2))
        self._shoe_img_lbl_l = tk.Label(
            _shoel_body, text="미선택",
            font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w", wraplength=178,
        )
        self._shoe_img_lbl_l.pack(fill=tk.X, padx=14, pady=(0, 2))
        tk.Label(_shoel_body, text="Z 순서",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        tk.Scale(_shoel_body, from_=0, to=10, orient=tk.HORIZONTAL,
                 variable=self._shoe_l_z_var, length=160,
                 bg=BG_PANEL, fg="#aaaacc", troughcolor="#0f3460",
                 highlightthickness=0, showvalue=True,
                 ).pack(padx=10, pady=(0, 4))

        tk.Frame(parent, bg="#2a2a4a", height=1).pack(fill=tk.X, padx=10, pady=(4, 8))

        # ── 오른손 장갑 이미지 (접기/펴기) ──
        _glover_open = tk.BooleanVar(value=True)
        _glover_hdr = tk.Frame(parent, bg=BG_PANEL, cursor="hand2")
        _glover_hdr.pack(fill=tk.X)
        _glover_lbl = tk.Label(
            _glover_hdr, text="▼  오른손 장갑 이미지",
            font=("Segoe UI", 10, "bold"), fg=TEXT_G, bg=BG_PANEL, anchor="w",
        )
        _glover_lbl.pack(fill=tk.X, padx=14, pady=(0, 4))
        _glover_sep = tk.Frame(parent, bg="#1e1e3a", height=1)
        _glover_sep.pack(fill=tk.X, padx=10, pady=(0, 4))
        _glover_body = tk.Frame(parent, bg=BG_PANEL)
        _glover_body.pack(fill=tk.X)

        def _toggle_glover_img(_e=None):
            if _glover_open.get():
                _glover_body.pack_forget()
                _glover_lbl.config(text="▶  오른손 장갑 이미지")
                _glover_open.set(False)
            else:
                _glover_body.pack(fill=tk.X, after=_glover_sep)
                _glover_lbl.config(text="▼  오른손 장갑 이미지")
                _glover_open.set(True)

        _glover_hdr.bind("<Button-1>", _toggle_glover_img)
        _glover_lbl.bind("<Button-1>", _toggle_glover_img)
        self._panel_sections.append((_glover_open, _toggle_glover_img))

        self._glove_img_btn_r = tk.Button(
            _glover_body, text="🧤  이미지 로드",
            font=("Segoe UI", 10, "bold"), bg="#1e3a5f", fg=TEXT_W,
            activebackground="#2a4f80", activeforeground="white",
            relief=tk.FLAT, cursor="hand2", pady=6, anchor="w", padx=12,
            command=lambda: self._toggle_glove_image(side='right'),
        )
        self._glove_img_btn_r.pack(fill=tk.X, padx=10, pady=(0, 2))
        self._glove_img_lbl_r = tk.Label(
            _glover_body, text="미선택",
            font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w", wraplength=178,
        )
        self._glove_img_lbl_r.pack(fill=tk.X, padx=14, pady=(0, 2))
        tk.Label(_glover_body, text="크기 (%)",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        tk.Scale(_glover_body, from_=30, to=300, orient=tk.HORIZONTAL,
                 variable=self._glove_size_var, length=160,
                 bg=BG_PANEL, fg=TEXT_W, troughcolor="#0f3460",
                 highlightthickness=0, showvalue=True,
                 ).pack(padx=10, pady=(0, 2))
        tk.Label(_glover_body, text="떨림 보정",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        tk.Scale(_glover_body, from_=0, to=95, orient=tk.HORIZONTAL,
                 variable=self._glove_smooth_var, length=160,
                 bg=BG_PANEL, fg="#88bbff", troughcolor="#0f3460",
                 highlightthickness=0, showvalue=True,
                 command=self._on_glove_smooth_change,
                 ).pack(padx=10, pady=(0, 2))
        tk.Label(_glover_body, text="Z 순서",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        tk.Scale(_glover_body, from_=0, to=10, orient=tk.HORIZONTAL,
                 variable=self._glove_r_z_var, length=160,
                 bg=BG_PANEL, fg="#aaaacc", troughcolor="#0f3460",
                 highlightthickness=0, showvalue=True,
                 ).pack(padx=10, pady=(0, 4))

        tk.Frame(parent, bg="#2a2a4a", height=1).pack(fill=tk.X, padx=10, pady=(4, 8))

        # ── 왼손 장갑 이미지 (접기/펴기) ──
        _glovel_open = tk.BooleanVar(value=True)
        _glovel_hdr = tk.Frame(parent, bg=BG_PANEL, cursor="hand2")
        _glovel_hdr.pack(fill=tk.X)
        _glovel_lbl = tk.Label(
            _glovel_hdr, text="▼  왼손 장갑 이미지",
            font=("Segoe UI", 10, "bold"), fg=TEXT_G, bg=BG_PANEL, anchor="w",
        )
        _glovel_lbl.pack(fill=tk.X, padx=14, pady=(0, 4))
        _glovel_sep = tk.Frame(parent, bg="#1e1e3a", height=1)
        _glovel_sep.pack(fill=tk.X, padx=10, pady=(0, 4))
        _glovel_body = tk.Frame(parent, bg=BG_PANEL)
        _glovel_body.pack(fill=tk.X)

        def _toggle_glovel_img(_e=None):
            if _glovel_open.get():
                _glovel_body.pack_forget()
                _glovel_lbl.config(text="▶  왼손 장갑 이미지")
                _glovel_open.set(False)
            else:
                _glovel_body.pack(fill=tk.X, after=_glovel_sep)
                _glovel_lbl.config(text="▼  왼손 장갑 이미지")
                _glovel_open.set(True)

        _glovel_hdr.bind("<Button-1>", _toggle_glovel_img)
        _glovel_lbl.bind("<Button-1>", _toggle_glovel_img)
        self._panel_sections.append((_glovel_open, _toggle_glovel_img))

        self._glove_img_btn_l = tk.Button(
            _glovel_body, text="🧤  이미지 로드",
            font=("Segoe UI", 10, "bold"), bg="#1e3a5f", fg=TEXT_W,
            activebackground="#2a4f80", activeforeground="white",
            relief=tk.FLAT, cursor="hand2", pady=6, anchor="w", padx=12,
            command=lambda: self._toggle_glove_image(side='left'),
        )
        self._glove_img_btn_l.pack(fill=tk.X, padx=10, pady=(0, 2))
        self._glove_img_lbl_l = tk.Label(
            _glovel_body, text="미선택",
            font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w", wraplength=178,
        )
        self._glove_img_lbl_l.pack(fill=tk.X, padx=14, pady=(0, 2))
        tk.Label(_glovel_body, text="Z 순서",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        tk.Scale(_glovel_body, from_=0, to=10, orient=tk.HORIZONTAL,
                 variable=self._glove_l_z_var, length=160,
                 bg=BG_PANEL, fg="#aaaacc", troughcolor="#0f3460",
                 highlightthickness=0, showvalue=True,
                 ).pack(padx=10, pady=(0, 4))

        tk.Frame(parent, bg="#2a2a4a", height=1).pack(fill=tk.X, padx=10, pady=(4, 8))

        # ── 무기 이미지 (접기/펴기) ──
        _weapon_open = tk.BooleanVar(value=True)
        _weapon_hdr = tk.Frame(parent, bg=BG_PANEL, cursor="hand2")
        _weapon_hdr.pack(fill=tk.X)
        _weapon_lbl = tk.Label(
            _weapon_hdr, text="▼  무기 이미지",
            font=("Segoe UI", 10, "bold"), fg=TEXT_G, bg=BG_PANEL, anchor="w",
        )
        _weapon_lbl.pack(fill=tk.X, padx=14, pady=(0, 4))
        _weapon_sep = tk.Frame(parent, bg="#1e1e3a", height=1)
        _weapon_sep.pack(fill=tk.X, padx=10, pady=(0, 4))
        _weapon_body = tk.Frame(parent, bg=BG_PANEL)
        _weapon_body.pack(fill=tk.X)

        def _toggle_weapon_img(_e=None):
            if _weapon_open.get():
                _weapon_body.pack_forget()
                _weapon_lbl.config(text="▶  무기 이미지")
                _weapon_open.set(False)
            else:
                _weapon_body.pack(fill=tk.X, after=_weapon_sep)
                _weapon_lbl.config(text="▼  무기 이미지")
                _weapon_open.set(True)

        _weapon_hdr.bind("<Button-1>", _toggle_weapon_img)
        _weapon_lbl.bind("<Button-1>", _toggle_weapon_img)
        self._panel_sections.append((_weapon_open, _toggle_weapon_img))

        self._weapon_img_btn = tk.Button(
            _weapon_body, text="⚔  이미지 로드",
            font=("Segoe UI", 10, "bold"), bg="#1e3a5f", fg=TEXT_W,
            activebackground="#2a4f80", activeforeground="white",
            relief=tk.FLAT, cursor="hand2", pady=6, anchor="w", padx=12,
            command=self._toggle_weapon_image,
        )
        self._weapon_img_btn.pack(fill=tk.X, padx=10, pady=(0, 2))
        self._weapon_img_lbl = tk.Label(
            _weapon_body, text="미선택",
            font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w", wraplength=178,
        )
        self._weapon_img_lbl.pack(fill=tk.X, padx=14, pady=(0, 2))
        tk.Label(_weapon_body, text="손",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        _hand_row = tk.Frame(_weapon_body, bg=BG_PANEL)
        _hand_row.pack(fill=tk.X, padx=14, pady=(0, 4))
        for _lbl_txt, _val in [("오른손", "right"), ("왼손", "left"), ("양손", "both")]:
            tk.Radiobutton(
                _hand_row, text=_lbl_txt, variable=self._weapon_hand_var, value=_val,
                bg=BG_PANEL, fg=TEXT_W, selectcolor="#0f3460",
                activebackground=BG_PANEL, activeforeground=TEXT_W,
                font=("Segoe UI", 8),
            ).pack(side=tk.LEFT, padx=(0, 6))
        tk.Label(_weapon_body, text="크기 (%)",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        tk.Scale(_weapon_body, from_=30, to=300, orient=tk.HORIZONTAL,
                 variable=self._weapon_size_var, length=160,
                 bg=BG_PANEL, fg=TEXT_W, troughcolor="#0f3460",
                 highlightthickness=0, showvalue=True,
                 ).pack(padx=10, pady=(0, 2))
        tk.Label(_weapon_body, text="떨림 보정",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        tk.Scale(_weapon_body, from_=0, to=95, orient=tk.HORIZONTAL,
                 variable=self._weapon_smooth_var, length=160,
                 bg=BG_PANEL, fg="#88bbff", troughcolor="#0f3460",
                 highlightthickness=0, showvalue=True,
                 command=self._on_weapon_smooth_change,
                 ).pack(padx=10, pady=(0, 2))
        tk.Label(_weapon_body, text="Z 순서",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        tk.Scale(_weapon_body, from_=0, to=10, orient=tk.HORIZONTAL,
                 variable=self._weapon_z_var, length=160,
                 bg=BG_PANEL, fg="#aaaacc", troughcolor="#0f3460",
                 highlightthickness=0, showvalue=True,
                 ).pack(padx=10, pady=(0, 4))

        tk.Frame(parent, bg="#2a2a4a", height=1).pack(fill=tk.X, padx=10, pady=(4, 8))

        # ── 앞모습 몸통 이미지 (접기/펴기) ──
        _bodyf_open = tk.BooleanVar(value=True)
        _bodyf_hdr = tk.Frame(parent, bg=BG_PANEL, cursor="hand2")
        _bodyf_hdr.pack(fill=tk.X)
        _bodyf_lbl = tk.Label(
            _bodyf_hdr, text="▼  몸통 앞모습 이미지",
            font=("Segoe UI", 10, "bold"), fg=TEXT_G, bg=BG_PANEL, anchor="w",
        )
        _bodyf_lbl.pack(fill=tk.X, padx=14, pady=(0, 4))
        _bodyf_sep = tk.Frame(parent, bg="#1e1e3a", height=1)
        _bodyf_sep.pack(fill=tk.X, padx=10, pady=(0, 4))
        _bodyf_body = tk.Frame(parent, bg=BG_PANEL)
        _bodyf_body.pack(fill=tk.X)

        def _toggle_bodyf_img(_e=None):
            if _bodyf_open.get():
                _bodyf_body.pack_forget()
                _bodyf_lbl.config(text="▶  몸통 앞모습 이미지")
                _bodyf_open.set(False)
            else:
                _bodyf_body.pack(fill=tk.X, after=_bodyf_sep)
                _bodyf_lbl.config(text="▼  몸통 앞모습 이미지")
                _bodyf_open.set(True)

        _bodyf_hdr.bind("<Button-1>", _toggle_bodyf_img)
        _bodyf_lbl.bind("<Button-1>", _toggle_bodyf_img)
        self._panel_sections.append((_bodyf_open, _toggle_bodyf_img))

        self._body_front_img_btn = tk.Button(
            _bodyf_body, text="👕  이미지 로드",
            font=("Segoe UI", 10, "bold"), bg="#1e3a5f", fg=TEXT_W,
            activebackground="#2a4f80", activeforeground="white",
            relief=tk.FLAT, cursor="hand2", pady=6, anchor="w", padx=12,
            command=self._toggle_body_front_image,
        )
        self._body_front_img_btn.pack(fill=tk.X, padx=10, pady=(0, 2))
        self._body_front_img_lbl = tk.Label(
            _bodyf_body, text="미선택",
            font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w", wraplength=178,
        )
        self._body_front_img_lbl.pack(fill=tk.X, padx=14, pady=(0, 2))
        tk.Label(_bodyf_body, text="크기 (%)",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        tk.Scale(_bodyf_body, from_=30, to=300, orient=tk.HORIZONTAL,
                 variable=self._body_front_size_var, length=160,
                 bg=BG_PANEL, fg=TEXT_W, troughcolor="#0f3460",
                 highlightthickness=0, showvalue=True,
                 ).pack(padx=10, pady=(0, 2))
        tk.Label(_bodyf_body, text="떨림 보정 (0=없음  →  95=최대)",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        tk.Scale(_bodyf_body, from_=0, to=95, orient=tk.HORIZONTAL,
                 variable=self._body_front_smooth_var, length=160,
                 bg=BG_PANEL, fg="#88bbff", troughcolor="#0f3460",
                 highlightthickness=0, showvalue=True,
                 command=self._on_body_smooth_change,
                 ).pack(padx=10, pady=(0, 4))
        tk.Label(_bodyf_body, text="Z 순서 (낮을수록 뒤에 렌더)",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        tk.Scale(_bodyf_body, from_=0, to=10, orient=tk.HORIZONTAL,
                 variable=self._body_front_z_var, length=160,
                 bg=BG_PANEL, fg="#aaaacc", troughcolor="#0f3460",
                 highlightthickness=0, showvalue=True,
                 ).pack(padx=10, pady=(0, 4))
        tk.Frame(_bodyf_body, bg="#2a2a4a", height=1).pack(fill=tk.X, padx=10, pady=(4, 4))
        self._body_front_pin_lbl = tk.Label(
            _bodyf_body, text="피벗 미설정",
            font=("Segoe UI", 8), fg="#ffaa44", bg=BG_PANEL, anchor="w",
        )
        self._body_front_pin_lbl.pack(fill=tk.X, padx=14, pady=(0, 2))
        self._body_front_pin_btn = tk.Button(
            _bodyf_body, text="🎯 피벗 설정",
            font=("Segoe UI", 9, "bold"), bg="#2a3f5f", fg=TEXT_W,
            activebackground="#3a5a80", activeforeground="white",
            relief=tk.FLAT, cursor="hand2", pady=5, padx=10,
            command=self._open_body_front_pin_picker, state=tk.DISABLED,
        )
        self._body_front_pin_btn.pack(fill=tk.X, padx=10, pady=(0, 4))

        tk.Frame(parent, bg="#2a2a4a", height=1).pack(fill=tk.X, padx=10, pady=(4, 8))

        # ── 옆모습 몸통 이미지 (접기/펴기) ──
        _bodys_open = tk.BooleanVar(value=True)
        _bodys_hdr = tk.Frame(parent, bg=BG_PANEL, cursor="hand2")
        _bodys_hdr.pack(fill=tk.X)
        _bodys_lbl = tk.Label(
            _bodys_hdr, text="▼  몸통 옆모습 이미지",
            font=("Segoe UI", 10, "bold"), fg=TEXT_G, bg=BG_PANEL, anchor="w",
        )
        _bodys_lbl.pack(fill=tk.X, padx=14, pady=(0, 4))
        _bodys_sep = tk.Frame(parent, bg="#1e1e3a", height=1)
        _bodys_sep.pack(fill=tk.X, padx=10, pady=(0, 4))
        _bodys_body = tk.Frame(parent, bg=BG_PANEL)
        _bodys_body.pack(fill=tk.X)

        def _toggle_bodys_img(_e=None):
            if _bodys_open.get():
                _bodys_body.pack_forget()
                _bodys_lbl.config(text="▶  몸통 옆모습 이미지")
                _bodys_open.set(False)
            else:
                _bodys_body.pack(fill=tk.X, after=_bodys_sep)
                _bodys_lbl.config(text="▼  몸통 옆모습 이미지")
                _bodys_open.set(True)

        _bodys_hdr.bind("<Button-1>", _toggle_bodys_img)
        _bodys_lbl.bind("<Button-1>", _toggle_bodys_img)
        self._panel_sections.append((_bodys_open, _toggle_bodys_img))

        self._body_side_img_btn = tk.Button(
            _bodys_body, text="👘  이미지 로드",
            font=("Segoe UI", 10, "bold"), bg="#1e3a5f", fg=TEXT_W,
            activebackground="#2a4f80", activeforeground="white",
            relief=tk.FLAT, cursor="hand2", pady=6, anchor="w", padx=12,
            command=self._toggle_body_side_image,
        )
        self._body_side_img_btn.pack(fill=tk.X, padx=10, pady=(0, 2))
        self._body_side_img_lbl = tk.Label(
            _bodys_body, text="미선택",
            font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w", wraplength=178,
        )
        self._body_side_img_lbl.pack(fill=tk.X, padx=14, pady=(0, 2))
        tk.Label(_bodys_body, text="크기 (%)",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        tk.Scale(_bodys_body, from_=30, to=300, orient=tk.HORIZONTAL,
                 variable=self._body_side_size_var, length=160,
                 bg=BG_PANEL, fg=TEXT_W, troughcolor="#0f3460",
                 highlightthickness=0, showvalue=True,
                 ).pack(padx=10, pady=(0, 2))
        tk.Label(_bodys_body, text="몸 두께 (어깨너비 대비 %)",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        tk.Scale(_bodys_body, from_=10, to=120, orient=tk.HORIZONTAL,
                 variable=self._body_side_depth_var, length=160,
                 bg=BG_PANEL, fg="#ffcc88", troughcolor="#0f3460",
                 highlightthickness=0, showvalue=True,
                 ).pack(padx=10, pady=(0, 2))
        tk.Label(_bodys_body, text="위치 Y (px)",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        tk.Scale(_bodys_body, from_=-300, to=300, orient=tk.HORIZONTAL,
                 variable=self._body_side_y_var, length=160,
                 bg=BG_PANEL, fg="#ff88cc", troughcolor="#0f3460",
                 highlightthickness=0, showvalue=True,
                 ).pack(padx=10, pady=(0, 2))
        tk.Label(_bodys_body, text="위치 X (px)",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        tk.Scale(_bodys_body, from_=-300, to=300, orient=tk.HORIZONTAL,
                 variable=self._body_side_x_var, length=160,
                 bg=BG_PANEL, fg="#ff88cc", troughcolor="#0f3460",
                 highlightthickness=0, showvalue=True,
                 ).pack(padx=10, pady=(0, 2))
        tk.Label(_bodys_body, text="떨림 보정 (0=없음  →  95=최대)",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        tk.Scale(_bodys_body, from_=0, to=95, orient=tk.HORIZONTAL,
                 variable=self._body_side_smooth_var, length=160,
                 bg=BG_PANEL, fg="#88bbff", troughcolor="#0f3460",
                 highlightthickness=0, showvalue=True,
                 command=self._on_body_smooth_change,
                 ).pack(padx=10, pady=(0, 4))
        tk.Label(_bodys_body, text="Z 순서 (낮을수록 뒤에 렌더)",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14)
        tk.Scale(_bodys_body, from_=0, to=10, orient=tk.HORIZONTAL,
                 variable=self._body_side_z_var, length=160,
                 bg=BG_PANEL, fg="#aaaacc", troughcolor="#0f3460",
                 highlightthickness=0, showvalue=True,
                 ).pack(padx=10, pady=(0, 4))
        tk.Frame(_bodys_body, bg="#2a2a4a", height=1).pack(fill=tk.X, padx=10, pady=(4, 4))
        self._body_side_pin_lbl = tk.Label(
            _bodys_body, text="피벗 미설정",
            font=("Segoe UI", 8), fg="#ffaa44", bg=BG_PANEL, anchor="w",
        )
        self._body_side_pin_lbl.pack(fill=tk.X, padx=14, pady=(0, 2))
        self._body_side_pin_btn = tk.Button(
            _bodys_body, text="🎯 피벗 설정",
            font=("Segoe UI", 9, "bold"), bg="#2a3f5f", fg=TEXT_W,
            activebackground="#3a5a80", activeforeground="white",
            relief=tk.FLAT, cursor="hand2", pady=5, padx=10,
            command=self._open_body_side_pin_picker, state=tk.DISABLED,
        )
        self._body_side_pin_btn.pack(fill=tk.X, padx=10, pady=(0, 4))

        tk.Frame(parent, bg="#2a2a4a", height=1).pack(fill=tk.X, padx=10, pady=(4, 8))

        parent = tab1  # 기본 탭으로 복귀

        # ── 감지 설정 ──
        tk.Label(parent, text="얼굴 감지 신뢰도",
                 font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                 ).pack(fill=tk.X, padx=14, pady=(0, 2))
        tk.Scale(parent, from_=0.1, to=0.9, resolution=0.05, orient=tk.HORIZONTAL,
                 variable=self._face_conf_var, length=160,
                 bg=BG_PANEL, fg="#88bbff", troughcolor="#0f3460",
                 highlightthickness=0, showvalue=True,
                 command=self._on_face_conf_change,
                 ).pack(padx=10, pady=(0, 4))

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

    # ── MediaPipe 초기화 (손/포즈) + InsightFace (얼굴) ───────────────────
    def _init_mediapipe(self):
        # 얼굴 감지: InsightFace 싱글턴 (별도 초기화 불필요 — 첫 detect 호출 시 자동 로드)
        self._face_det = None  # InsightFace 사용 — MediaPipe FaceLandmarker 제거
        try:
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
            self._hand_det = mp_vision.HandLandmarker.create_from_options(hand_opts)
            self._pose_det = mp_vision.PoseLandmarker.create_from_options(pose_opts)
        except Exception as e:
            print(f"[MediaPipe init error] {e}")
            self._hand_det = None
            self._pose_det = None

    def _on_face_conf_change(self, *_):
        self._init_mediapipe()
        self._det_cache = None
        self._refresh_frame()

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

    def _detect_landmarks(self, bgr):
        """애니화 사람 마스크용 얼굴/손/포즈 감지 (640px 축소 추론)."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h_px, w_px = bgr.shape[:2]
        _sc = min(1.0, 640 / max(w_px, h_px, 1))
        rgb_s = (cv2.resize(rgb, (int(w_px * _sc), int(h_px * _sc)))
                 if _sc < 0.99 else rgb)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_s)
        try:
            face_res = _if_det_mod.detect(bgr, min_conf=self._face_conf_var.get())
            hand_res = self._hand_det.detect(mp_img) if self._hand_det else None
            pose_res = self._pose_det.detect(mp_img) if self._pose_det else None
        except Exception as e:
            print(f"[detect error] {e}")
            return None, None, None
        return face_res, hand_res, pose_res

    def _get_sd_pipe(self):
        """SDCartoon 파이프라인을 1회 로드 후 재사용 (지연 로드)."""
        if SDCartoon is None:
            return None
        if self._sd_pipe is None:
            try:
                self._sd_pipe = SDCartoon().load()
            except Exception as e:
                print(f"[SD load error] {e} — OpenCV로 대체합니다.")
                self._sd_pipe = None
        return self._sd_pipe

    def _apply_anime(self, bgr, playback=False):
        """애니화 적용 (미리보기/내보내기 공용). 재생 중에는 N프레임마다 변환."""
        style      = self._anime_style_var.get()
        bg_mode    = self._anime_bg_var.get()
        range_mode = self._anime_range_var.get()
        strength   = self._anime_strength_var.get()

        # SD는 프레임당 수 초 → 재생 중에는 적용하지 않음(직전 결과/원본 표시)
        if style == "sd" and playback:
            return self._anime_cache.copy() if self._anime_cache is not None else bgr

        # base 변환기 선택 (whitebox/bold=White-box ONNX, animegan=선택 모델)
        if style in ("whitebox", "bold"):
            converter = self._get_anime_converter(self._find_whitebox_model())
        elif style == "animegan":
            converter = self._get_anime_converter(self._anime_model_path)
        else:
            converter = None
        sd_pipe = self._get_sd_pipe() if style == "sd" else None

        # 재생 중 스로틀: 3프레임마다 1회만 변환 (나머지는 직전 결과 재사용)
        if playback:
            self._anime_skip += 1
            if self._anime_cache is not None and (self._anime_skip % 3 != 0):
                return self._anime_cache.copy()

        # 사람 마스크가 필요할 때만 감지 (전체화면이면 생략)
        fr = hr = pr = None
        if range_mode == "person":
            if playback and self._det_cache is not None:
                fr, hr, pr = self._det_cache
            else:
                fr, hr, pr = self._detect_landmarks(bgr)
                if playback:
                    self._det_cache = (fr, hr, pr)

        out = apply_anime_to_person(
            bgr, pr, fr, hr,
            style=style, bg_mode=bg_mode,
            converter=converter, range_mode=range_mode,
            strength=strength, sd_pipe=sd_pipe,
        )
        if playback:
            self._anime_cache = out.copy()
        return out

    # ── 프레임 표시 ────────────────────────────────────────────────────────
    def _display_frame(self, bgr, playback=False):
        if _ANIME_AVAILABLE and self._show_anime_var.get():
            bgr = self._apply_anime(bgr, playback=playback)
        if (self._show_face.get() or self._show_body.get() or self._show_hands.get()
                or self._face_img is not None or self._show_mosaic.get()
                or self._arm_img is not None or self._img_only_var.get()
                or self._leg_img_r is not None or self._leg_img_l is not None
                or self._body_front_img is not None or self._body_side_img is not None
                or self._shoe_img_r is not None or self._shoe_img_l is not None
                or self._glove_img_r is not None or self._glove_img_l is not None
                or self._weapon_img is not None):
            bgr = self._apply_overlay(bgr, playback=playback, img_only=self._img_only_var.get())
        if self._show_track_var.get() and self._track_points:
            bgr = self._draw_track_markers(bgr)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        self._canvas.update_idletasks()
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw <= 1:
            cw = 840
        if ch <= 1:
            ch = 480

        img = Image.fromarray(rgb)
        vw, vh = img.size
        base_scale = min(cw / vw, ch / vh)
        zoom = self._zoom

        if zoom <= 1.0:
            # 레터박스/필러박스 (기존 동작)
            scale = base_scale * zoom
            nw, nh = int(vw * scale), int(vh * scale)
            img = img.resize((nw, nh), Image.BILINEAR)
            canvas_img = Image.new("RGB", (cw, ch), (0, 0, 0))
            ox = (cw - nw) // 2
            oy = (ch - nh) // 2
            canvas_img.paste(img, (ox, oy))
        else:
            # zoom > 100%: 확대 후 pan 오프셋 적용 크롭
            scale = base_scale * zoom
            nw, nh = int(vw * scale), int(vh * scale)
            img = img.resize((nw, nh), Image.BILINEAR)
            cx = (nw - cw) // 2 - self._pan_x
            cy = (nh - ch) // 2 - self._pan_y
            cx = max(0, min(max(0, nw - cw), cx))
            cy = max(0, min(max(0, nh - ch), cy))
            x2 = min(nw, cx + cw)
            y2 = min(nh, cy + ch)
            img = img.crop((cx, cy, x2, y2))
            cw_c, ch_c = img.size
            canvas_img = Image.new("RGB", (cw, ch), (0, 0, 0))
            canvas_img.paste(img, ((cw - cw_c) // 2, (ch - ch_c) // 2))
        self._photo = ImageTk.PhotoImage(canvas_img)
        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)

    def _canvas_wheel_enter(self, _event):
        self._canvas.bind_all("<MouseWheel>", self._on_zoom)
        self._canvas.bind_all("<Button-4>",   self._on_zoom)
        self._canvas.bind_all("<Button-5>",   self._on_zoom)

    def _canvas_wheel_leave(self, _event):
        self._canvas.unbind_all("<MouseWheel>")
        self._canvas.unbind_all("<Button-4>")
        self._canvas.unbind_all("<Button-5>")

    def _on_zoom(self, event):
        if event.num == 4:
            delta = +1
        elif event.num == 5:
            delta = -1
        else:
            delta = int(event.delta / 120)
        new_zoom = round(self._zoom + 0.1 * delta, 2)
        new_zoom = max(0.1, min(5.0, new_zoom))
        if new_zoom == self._zoom:
            return
        self._zoom = new_zoom
        if new_zoom <= 1.0:
            self._pan_x = 0
            self._pan_y = 0
        self._zoom_var.set(f"{int(round(new_zoom * 100))}%")
        self._refresh_frame()

    def _reset_zoom(self):
        self._zoom = 1.0
        self._zoom_var.set("100%")
        self._pan_x = 0
        self._pan_y = 0
        self._refresh_frame()

    def _pan_start_cb(self, event):
        if self._zoom > 1.0:
            self._pan_start = (event.x, event.y)
            self._canvas.config(cursor="fleur")

    def _pan_drag_cb(self, event):
        if self._pan_start is None:
            return
        dx = event.x - self._pan_start[0]
        dy = event.y - self._pan_start[1]
        self._pan_start = (event.x, event.y)
        self._pan_x += dx
        self._pan_y += dy
        self._refresh_frame()

    def _pan_end_cb(self, _event):
        self._pan_start = None
        self._canvas.config(cursor="")

    # ── 포인트 트래킹 ────────────────────────────────────────────────────────
    def _canvas_to_frame(self, cx_click, cy_click):
        """캔버스 클릭 좌표 → 원본 프레임 픽셀 좌표. 영상 밖이면 None.
        _display_frame 의 줌/팬 정변환을 역산한다."""
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw <= 1:
            cw = 840
        if ch <= 1:
            ch = 480
        vw, vh = self._vid_w, self._vid_h
        if vw <= 0 or vh <= 0:
            return None
        base_scale = min(cw / vw, ch / vh)
        scale = base_scale * self._zoom
        if scale <= 0:
            return None
        nw, nh = int(vw * scale), int(vh * scale)
        if self._zoom <= 1.0:
            ox = (cw - nw) // 2
            oy = (ch - nh) // 2
            fx = (cx_click - ox) / scale
            fy = (cy_click - oy) / scale
        else:
            crop_x = (nw - cw) // 2 - self._pan_x
            crop_y = (nh - ch) // 2 - self._pan_y
            crop_x = max(0, min(max(0, nw - cw), crop_x))
            crop_y = max(0, min(max(0, nh - ch), crop_y))
            x2 = min(nw, crop_x + cw)
            y2 = min(nh, crop_y + ch)
            pox = (cw - (x2 - crop_x)) // 2
            poy = (ch - (y2 - crop_y)) // 2
            fx = (cx_click - pox + crop_x) / scale
            fy = (cy_click - poy + crop_y) / scale
        if fx < 0 or fy < 0 or fx >= vw or fy >= vh:
            return None
        return (fx, fy)

    def _toggle_track_pick(self):
        if self._track_busy:
            return
        self._track_pick_mode = not self._track_pick_mode
        if self._track_pick_mode:
            self._track_pick_btn.config(text="🎯 클릭하세요…", bg="#2a4f80")
            self._canvas.config(cursor="tcross")
            self._track_status_var.set("영상에서 추적할 지점을 클릭하세요")
        else:
            self._track_pick_btn.config(text="🎯 점 추가", bg="#1e3a5f")
            self._canvas.config(cursor="")
            self._track_status_var.set("")

    def _on_track_click(self, event):
        if not self._track_pick_mode or self._track_busy:
            return
        pt = self._canvas_to_frame(event.x, event.y)
        if pt is None:
            self._track_status_var.set("영상 영역 안을 클릭하세요")
            return
        fx, fy = pt
        origin = self._current_frame
        self._track_pick_mode = False
        self._canvas.config(cursor="")
        self._track_busy = True
        self._track_pick_btn.config(text="🎯 추적 중…", bg="#2a4f80", state=tk.DISABLED)
        self._track_status_var.set("추적 준비 중…")
        threading.Thread(
            target=self._track_worker, args=(fx, fy, origin), daemon=True,
        ).start()

    def _track_color(self, tid):
        import colorsys
        h = ((tid - 1) * 0.61803398875) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
        return (int(b * 255), int(g * 255), int(r * 255))   # BGR

    def _set_track_status(self, text):
        self.win.after(0, lambda: self._track_status_var.set(text))

    def _ensure_gray_cache(self):
        """전 프레임 grayscale 리스트 (지연 생성, 전 점 공유). 워커 스레드에서 호출."""
        if self._track_gray_cache is not None:
            return self._track_gray_cache
        grays = []
        cap = cv2.VideoCapture(self._video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        cap.release()
        self._track_gray_cache = grays
        return grays

    def _track_worker(self, fx, fy, origin):
        try:
            self._set_track_status("프레임 분석 중…")
            grays = self._ensure_gray_cache()
            n = len(grays) if grays else 0
            if n == 0:
                self._set_track_status("프레임을 읽을 수 없습니다")
                self.win.after(0, self._track_finish_ui)
                return
            origin = max(0, min(n - 1, origin))

            # 클릭 지점 주변에서 가장 강한 코너로 스냅 → LK 추적 안정성 향상
            roi_r = 15
            gx0 = max(0, int(fx - roi_r))
            gy0 = max(0, int(fy - roi_r))
            roi = grays[origin][gy0:gy0 + 2 * roi_r, gx0:gx0 + 2 * roi_r]
            if roi.size > 0:
                feats = cv2.goodFeaturesToTrack(
                    roi, maxCorners=1, qualityLevel=0.01, minDistance=5)
                if feats is not None:
                    fx = gx0 + float(feats[0][0][0])
                    fy = gy0 + float(feats[0][0][1])

            lk = dict(
                winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
            pos = {origin: (float(fx), float(fy))}
            total = max(1, n - 1)
            done = 0

            p = np.array([[[fx, fy]]], dtype=np.float32)
            for i in range(origin, n - 1):
                nxt, st, _e = cv2.calcOpticalFlowPyrLK(grays[i], grays[i + 1], p, None, **lk)
                done += 1
                if st is None or int(st[0][0]) == 0:
                    break
                x, y = float(nxt[0][0][0]), float(nxt[0][0][1])
                if not (0 <= x < self._vid_w and 0 <= y < self._vid_h):
                    break
                pos[i + 1] = (x, y)
                p = nxt
                if done % 20 == 0:
                    self._set_track_status(f"추적 중… {int(done * 100 / total)}%")

            p = np.array([[[fx, fy]]], dtype=np.float32)
            for i in range(origin, 0, -1):
                nxt, st, _e = cv2.calcOpticalFlowPyrLK(grays[i], grays[i - 1], p, None, **lk)
                done += 1
                if st is None or int(st[0][0]) == 0:
                    break
                x, y = float(nxt[0][0][0]), float(nxt[0][0][1])
                if not (0 <= x < self._vid_w and 0 <= y < self._vid_h):
                    break
                pos[i - 1] = (x, y)
                p = nxt
                if done % 20 == 0:
                    self._set_track_status(f"추적 중… {int(done * 100 / total)}%")

            entry = {
                "id": self._track_next_id,
                "color": self._track_color(self._track_next_id),
                "origin_frame": origin,
                "pos": pos,
            }
            self._track_next_id += 1
            self._track_points.append(entry)
            self.win.after(0, lambda e=entry: self._on_track_done(e))
        except Exception as exc:                                   # noqa: BLE001
            msg = str(exc)
            self._set_track_status(f"추적 오류: {msg}")
            self.win.after(0, self._track_finish_ui)

    def _on_track_done(self, entry):
        self._track_status_var.set(f"#{entry['id']} 추적 완료 ({len(entry['pos'])} 프레임)")
        self._track_finish_ui()
        self._rebuild_track_list()
        self._refresh_frame()

    def _track_finish_ui(self):
        self._track_busy = False
        if self._track_pick_btn is not None:
            self._track_pick_btn.config(state=tk.NORMAL, text="🎯 점 추가", bg="#1e3a5f")
        self._canvas.config(cursor="")

    def _draw_track_markers(self, bgr):
        out = bgr.copy()
        idx = self._current_frame
        for entry in self._track_points:
            pt = entry["pos"].get(idx)
            if pt is None:
                continue
            x, y = int(round(pt[0])), int(round(pt[1]))
            color = entry["color"]
            cv2.drawMarker(out, (x, y), color, cv2.MARKER_CROSS, 18, 2)
            cv2.circle(out, (x, y), 10, color, 2, cv2.LINE_AA)
            cv2.putText(out, f"#{entry['id']}", (x + 12, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return out

    def _rebuild_track_list(self):
        frame = self._track_list_frame
        if frame is None:
            return
        for w in frame.winfo_children():
            w.destroy()
        if not self._track_points:
            tk.Label(frame, text="추적 점 없음",
                     font=("Segoe UI", 8), fg=TEXT_G, bg=BG_PANEL, anchor="w",
                     ).pack(fill=tk.X, padx=4, pady=2)
            return
        for entry in self._track_points:
            row = tk.Frame(frame, bg=BG_PANEL)
            row.pack(fill=tk.X, pady=1)
            b, g, r = entry["color"]
            hexc = f"#{r:02x}{g:02x}{b:02x}"
            tk.Label(row, text="●", font=("Segoe UI", 11), fg=hexc, bg=BG_PANEL,
                     ).pack(side=tk.LEFT, padx=(2, 4))
            tk.Label(row, text=f"#{entry['id']}  ({len(entry['pos'])}f)",
                     font=("Segoe UI", 9), fg=TEXT_W, bg=BG_PANEL, anchor="w",
                     ).pack(side=tk.LEFT, fill=tk.X, expand=True)
            tk.Button(row, text="✕", font=("Segoe UI", 9),
                      bg=BG_PANEL, fg="#ff8888", relief=tk.FLAT, cursor="hand2",
                      activebackground="#3a1e1e", activeforeground="white",
                      command=lambda tid=entry["id"]: self._remove_track(tid),
                      ).pack(side=tk.RIGHT, padx=2)

    def _remove_track(self, tid):
        if self._track_busy:
            return
        self._track_points = [e for e in self._track_points if e["id"] != tid]
        self._rebuild_track_list()
        self._refresh_frame()

    def _clear_tracks(self):
        if self._track_busy:
            return
        self._track_points = []
        self._track_status_var.set("")
        self._rebuild_track_list()
        self._refresh_frame()

    def _export_tracks(self):
        if self._track_busy:
            messagebox.showinfo("트래킹", "추적이 끝난 뒤 내보내세요.", parent=self.win)
            return
        if not self._track_points:
            messagebox.showinfo("트래킹", "내보낼 추적 점이 없습니다.", parent=self.win)
            return
        out_dir = filedialog.askdirectory(
            title="트랙 내보낼 폴더 선택", parent=self.win,
        )
        if not out_dir:
            return
        info = VideoInfo(
            width=self._vid_w, height=self._vid_h,
            fps=self._fps, total_frames=self._total_frames,
        )
        try:
            files = export_tracks_ae(out_dir, info, self._track_points)
        except Exception as exc:                                   # noqa: BLE001
            messagebox.showerror("트래킹", f"내보내기 실패:\n{exc}", parent=self.win)
            return
        self._track_status_var.set(f"{len(files)}개 트랙 내보냄")
        messagebox.showinfo(
            "트래킹", f"{len(files)}개 트랙을 AE 키프레임으로 저장했습니다:\n{out_dir}",
            parent=self.win,
        )

    def _apply_overlay(self, bgr, playback=False, img_only=False):
        """오버레이 렌더링."""
        if self._hand_det is None:
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
                face_res = _if_det_mod.detect(bgr, min_conf=self._face_conf_var.get())
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

        if img_only:
            overlay[:] = 0

        # ── 얼굴 모자이크 (가장 먼저 적용)
        if self._show_mosaic.get():
            _apply_face_mosaic(overlay, face_res, _ow, _oh)

        # ── 포즈 스켈레톤 (모든 감지된 사람, 얼굴/손 아래 레이어로 먼저 그리기)
        if self._show_body.get() and pose_res and pose_res.pose_landmarks:
            _SKEL = [(11,12),(11,23),(12,24),(23,24),
                     (11,13),(13,15),(12,14),(14,16),
                     (23,25),(25,27),(24,26),(26,28),
                     (27,31),(28,32)]  # ankle → tiptoe
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
                for _i in [11,12,13,14,15,16,23,24,25,26,27,28,31,32]:
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
        if (self._show_face.get() or self._show_body.get()) and face_res.face_landmarks:
            _nc = (0, 230, 180)
            for _lf in face_res.face_landmarks:
                if hasattr(_lf, 'bbox'):
                    # InsightFace: 5개 키포인트 원 + bbox 사각형
                    for _i in [33, 263, 4, 61, 291]:
                        cv2.circle(overlay,
                                   (int(_lf[_i].x * _ow), int(_lf[_i].y * _oh)),
                                   5, _nc, -1)
                    cv2.rectangle(overlay,
                                  (_lf.bbox[0], _lf.bbox[1]),
                                  (_lf.bbox[2], _lf.bbox[3]),
                                  _nc, 1)
                    if abs(getattr(_lf, 'yaw', 0.0)) > self._side_thr_var.get():
                        _cx = (_lf.bbox[0] + _lf.bbox[2]) // 2
                        _cy = (_lf.bbox[1] + _lf.bbox[3]) // 2
                        cv2.putText(overlay, "side view", (_cx - 40, _cy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 200, 255), 1, cv2.LINE_AA)
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

        # ── 이미지 오버레이 — Z 순서 정렬 후 렌더링 ───────────────────────
        _fi = self._face_img
        _fp = self._face_img_pts
        _rh = self._face_img_ref_h
        if _fi is not None and self._face_img_open is not None:
            _mar = _compute_mar(face_res, _ow, _oh)
            if _mar >= self._mouth_thr_var.get():
                _fi = self._face_img_open
                _fp = self._face_img_open_pts
                _rh = self._face_img_open_ref_h

        _img_jobs = []  # (z, callable)

        _si = self._face_img_side
        _sp = self._face_img_side_pts
        if _fi is not None or _si is not None:
            _piv = self._face_pivot
            _rot = self._face_rot_var.get()
            _sthr = float(self._side_thr_var.get())
            _sanch = self._face_img_side_anchors
            _img_jobs.append((self._face_img_z_var.get(), lambda: _apply_face_img_overlay(
                overlay, face_res, _ow, _oh, _fi, _fp,
                eye_y_pct=self._eye_y_var.get(),
                eye_x_pct=self._eye_x_var.get(),
                size_pct=self._img_size_var.get(),
                ema_state=self._face_img_ema,
                pivot=_piv,
                rotation_offset=_rot,
                side_img=_si,
                side_pts=_sp,
                side_threshold=_sthr,
                side_anchors=_sanch,
                side_eye_y_pct=self._side_eye_y_var.get(),
                side_eye_x_pct=self._side_eye_x_var.get(),
                side_size_pct=self._side_img_size_var.get(),
                side_ema_state=self._face_img_side_ema,
                ref_h=_rh,
                side_ref_h=self._face_img_side_ref_h,
                feather_px=self._feather_var.get(),
                interp=cv2.INTER_LANCZOS4 if self._hq_var.get() else cv2.INTER_LINEAR)))

        if self._arm_img is not None and pose_res:
            _img_jobs.append((self._arm_z_var.get(), lambda: _apply_arm_img_overlay(
                overlay, pose_res, _ow, _oh, self._arm_img,
                anchor_y_pct=self._arm_y_var.get(),
                anchor_x_pct=self._arm_x_var.get(),
                size_pct=self._arm_size_var.get(),
                ema_state=self._arm_img_ema,
                arm_pins=self._arm_pins,
                arm_seg_cache=self._arm_seg_cache,
                side='right',
                feather_px=self._feather_var.get(),
                interp=cv2.INTER_LANCZOS4 if self._hq_var.get() else cv2.INTER_LINEAR)))

        if self._arm_img_l is not None and pose_res:
            _img_jobs.append((self._arm_l_z_var.get(), lambda: _apply_arm_img_overlay(
                overlay, pose_res, _ow, _oh, self._arm_img_l,
                anchor_y_pct=self._arm_y_var.get(),
                anchor_x_pct=self._arm_x_var.get(),
                size_pct=self._arm_size_var.get(),
                ema_state=self._arm_img_ema_l,
                arm_pins=self._arm_pins_l,
                arm_seg_cache=self._arm_seg_cache_l,
                side='left',
                feather_px=self._feather_var.get(),
                interp=cv2.INTER_LANCZOS4 if self._hq_var.get() else cv2.INTER_LINEAR)))

        if self._leg_img_r is not None and pose_res:
            _img_jobs.append((self._leg_r_z_var.get(), lambda: _apply_leg_img_overlay(
                overlay, pose_res, _ow, _oh, self._leg_img_r,
                size_pct=self._leg_size_var.get(),
                ema_state=self._leg_img_ema_r,
                leg_pins=self._leg_pins_r,
                leg_seg_cache=self._leg_seg_cache_r,
                side='right',
                feather_px=self._feather_var.get(),
                interp=cv2.INTER_LANCZOS4 if self._hq_var.get() else cv2.INTER_LINEAR)))

        if self._leg_img_l is not None and pose_res:
            _img_jobs.append((self._leg_l_z_var.get(), lambda: _apply_leg_img_overlay(
                overlay, pose_res, _ow, _oh, self._leg_img_l,
                size_pct=self._leg_size_var.get(),
                ema_state=self._leg_img_ema_l,
                leg_pins=self._leg_pins_l,
                leg_seg_cache=self._leg_seg_cache_l,
                side='left',
                feather_px=self._feather_var.get(),
                interp=cv2.INTER_LANCZOS4 if self._hq_var.get() else cv2.INTER_LINEAR)))

        if self._shoe_img_r is not None and pose_res:
            _img_jobs.append((self._shoe_r_z_var.get(), lambda: _apply_shoe_img_overlay(
                overlay, pose_res, _ow, _oh, self._shoe_img_r,
                size_pct=self._shoe_size_var.get(),
                ema_state=self._shoe_img_ema_r,
                side='right',
                feather_px=self._feather_var.get(),
                interp=cv2.INTER_LANCZOS4 if self._hq_var.get() else cv2.INTER_LINEAR)))

        if self._shoe_img_l is not None and pose_res:
            _img_jobs.append((self._shoe_l_z_var.get(), lambda: _apply_shoe_img_overlay(
                overlay, pose_res, _ow, _oh, self._shoe_img_l,
                size_pct=self._shoe_size_var.get(),
                ema_state=self._shoe_img_ema_l,
                side='left',
                feather_px=self._feather_var.get(),
                interp=cv2.INTER_LANCZOS4 if self._hq_var.get() else cv2.INTER_LINEAR)))

        if self._glove_img_r is not None and hand_res.hand_landmarks:
            _img_jobs.append((self._glove_r_z_var.get(), lambda: _apply_glove_img_overlay(
                overlay, hand_res, _ow, _oh, self._glove_img_r,
                size_pct=self._glove_size_var.get(),
                ema_state=self._glove_img_ema_r,
                side='right',
                feather_px=self._feather_var.get(),
                interp=cv2.INTER_LANCZOS4 if self._hq_var.get() else cv2.INTER_LINEAR)))

        if self._glove_img_l is not None and hand_res.hand_landmarks:
            _img_jobs.append((self._glove_l_z_var.get(), lambda: _apply_glove_img_overlay(
                overlay, hand_res, _ow, _oh, self._glove_img_l,
                size_pct=self._glove_size_var.get(),
                ema_state=self._glove_img_ema_l,
                side='left',
                feather_px=self._feather_var.get(),
                interp=cv2.INTER_LANCZOS4 if self._hq_var.get() else cv2.INTER_LINEAR)))

        if self._weapon_img is not None and hand_res.hand_landmarks:
            _hw = self._weapon_hand_var.get()
            if _hw in ('right', 'both'):
                _img_jobs.append((self._weapon_z_var.get(), lambda: _apply_weapon_img_overlay(
                    overlay, hand_res, _ow, _oh, self._weapon_img,
                    size_pct=self._weapon_size_var.get(),
                    ema_state=self._weapon_img_ema_r,
                    hand_side='right',
                    feather_px=self._feather_var.get(),
                    interp=cv2.INTER_LANCZOS4 if self._hq_var.get() else cv2.INTER_LINEAR)))
            if _hw in ('left', 'both'):
                _img_jobs.append((self._weapon_z_var.get(), lambda: _apply_weapon_img_overlay(
                    overlay, hand_res, _ow, _oh, self._weapon_img,
                    size_pct=self._weapon_size_var.get(),
                    ema_state=self._weapon_img_ema_l,
                    hand_side='left',
                    feather_px=self._feather_var.get(),
                    interp=cv2.INTER_LANCZOS4 if self._hq_var.get() else cv2.INTER_LINEAR)))

        if self._body_front_img is not None and pose_res:
            _img_jobs.append((self._body_front_z_var.get(), lambda: _apply_body_front_overlay(
                overlay, pose_res, _ow, _oh, self._body_front_img,
                size_pct=self._body_front_size_var.get(),
                ema_state=self._body_front_ema,
                body_pins=self._body_front_pins,
                feather_px=self._feather_var.get(),
                interp=cv2.INTER_LANCZOS4 if self._hq_var.get() else cv2.INTER_LINEAR)))

        if self._body_side_img is not None and pose_res:
            _img_jobs.append((self._body_side_z_var.get(), lambda: _apply_body_side_overlay(
                overlay, pose_res, _ow, _oh, self._body_side_img,
                size_pct=self._body_side_size_var.get(),
                depth_pct=self._body_side_depth_var.get(),
                offset_x=self._body_side_x_var.get(),
                offset_y=self._body_side_y_var.get(),
                ema_state=self._body_side_ema,
                body_pins=self._body_side_pins,
                feather_px=self._feather_var.get(),
                interp=cv2.INTER_LANCZOS4 if self._hq_var.get() else cv2.INTER_LINEAR)))

        for _, _fn in sorted(_img_jobs, key=lambda x: x[0]):
            _fn()

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
        if self._hand_det is None:
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

    @staticmethod
    def _find_default_anime_model():
        """models/ 폴더에서 White-box를 제외한 첫 AnimeGAN *.onnx 모델 자동 탐지."""
        import glob
        found = sorted(glob.glob(os.path.join(_BASE, "models", "*.onnx")))
        found = [f for f in found if "whitebox" not in os.path.basename(f).lower()]
        return found[0] if found else ""

    @staticmethod
    def _find_whitebox_model():
        """White-box 카툰 내장 ONNX 경로 반환 (없으면 빈 문자열)."""
        import glob
        p = os.path.join(_BASE, "models", "whitebox_cartoon_720.onnx")
        if os.path.exists(p):
            return p
        found = sorted(glob.glob(os.path.join(_BASE, "models", "*whitebox*.onnx")))
        return found[0] if found else ""

    def _get_anime_converter(self, model_path=None):
        """주어진 ONNX 모델로 AnimeGANConverter를 1회 로드 후 재사용."""
        if model_path is None:
            model_path = self._anime_model_path
        if AnimeGANConverter is None or not model_path:
            return None
        if (self._anime_converter is not None
                and getattr(self._anime_converter, "_model_path", None)
                == model_path):
            return self._anime_converter
        try:
            conv = AnimeGANConverter()
            conv.load(model_path)
            conv._model_path = model_path
            self._anime_converter = conv
        except Exception as e:
            print(f"[AnimeGAN load error] {e} — OpenCV로 대체합니다.")
            self._anime_converter = None
        return self._anime_converter

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
            self._anime_converter = None   # 모델 변경 → 캐시 무효화
            self._anime_cache = None

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
        style  = self._anime_style_var.get()
        is_gan = style == "animegan"
        self._anime_model_btn.config(state=tk.NORMAL if is_gan else tk.DISABLED)
        if style == "whitebox":
            self._anime_model_lbl.config(text="화이트박스 카툰 (내장 모델)")
        elif style == "bold":
            self._anime_model_lbl.config(text="굵은 카툰 (White-box + 양자화/외곽선)")
        elif style == "sd":
            self._anime_model_lbl.config(
                text="SD 고품질 — 느림·정지컷 권장 (영상은 떨림)")
        elif style == "opencv":
            self._anime_model_lbl.config(text="OpenCV 사용 (모델 불필요)")
        elif self._anime_model_path:
            self._anime_model_lbl.config(text=os.path.basename(self._anime_model_path))
        else:
            self._anime_model_lbl.config(text="미선택 (OpenCV로 대체)")

        # 강도 슬라이더는 bold/sd 에서만 노출
        if style in ("bold", "sd"):
            self._anime_strength_row.pack(fill=tk.X, padx=14, pady=(0, 2),
                                          after=self._anime_strength_anchor)
        else:
            self._anime_strength_row.pack_forget()
        self._anime_cache = None
        if self._show_anime_var.get() and not self._playing:
            self._refresh_frame()

    def _on_anime_toggle(self):
        """애니화 체크박스 토글 → 미리보기 갱신."""
        self._anime_cache = None
        if not self._playing:
            self._refresh_frame()

    def _on_anime_opt_change(self):
        """배경/범위 옵션 변경 → 미리보기 갱신."""
        self._anime_cache = None
        if self._show_anime_var.get() and not self._playing:
            self._refresh_frame()

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
                        or self._show_mosaic.get() or self._face_img is not None
                        or self._arm_img is not None or self._arm_img_l is not None
                        or self._img_only_var.get())
        with_anime   = self._show_anime_var.get() and _ANIME_AVAILABLE

        # 오버레이/애니화 모드인데 MediaPipe 없으면 경고
        if (with_overlay or with_anime) and self._hand_det is None:
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

        # SD 스타일은 프레임당 수 초 + 영상 떨림 → 사전 경고
        if with_anime and self._anime_style_var.get() == "sd":
            if not messagebox.askyesno(
                "SD 고품질 — 느림 주의",
                "SD 스타일은 프레임당 수 초가 걸려 영상 전체 변환이 매우 오래 걸립니다.\n"
                "또한 프레임마다 그림이 달라져 영상에서 떨림(flicker)이 생깁니다.\n"
                "(짧은 클립/정지컷 권장) 계속하시겠습니까?",
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
        _anime_range     = "person"
        _anime_strength  = 3
        _sd_pipe         = None
        if with_anime:
            _anime_style    = self._anime_style_var.get()
            _anime_bg_mode  = self._anime_bg_var.get()
            _anime_range    = self._anime_range_var.get()
            _anime_strength = self._anime_strength_var.get()
            _model_path = ""
            if _anime_style in ("whitebox", "bold"):
                _model_path = self._find_whitebox_model()
            elif _anime_style == "animegan":
                _model_path = self._anime_model_path
            if _model_path:
                try:
                    _anime_converter = AnimeGANConverter()
                    _anime_converter.load(_model_path)
                except Exception as _e:
                    print(f"[AnimeGAN load error] {_e} — OpenCV로 대체합니다.")
                    if _anime_style != "bold":
                        _anime_style = "opencv"
            if _anime_style == "sd":
                _sd_pipe = self._get_sd_pipe()
                if _sd_pipe is None:
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
                    if _anime_range == "person":
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
                        range_mode=_anime_range,
                        strength=_anime_strength,
                        sd_pipe=_sd_pipe,
                    )

                # ── 오버레이 (랜드마크/모자이크 등) ─────────────────────────
                if with_overlay:
                    frame = self._apply_overlay(frame, img_only=self._img_only_var.get())

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

    def _on_side_ema_smooth_change(self, _=None):
        self._face_img_side_ema['alpha'] = 1.0 - (self._side_ema_smooth_var.get() / 100.0)

    # ── 오른팔 이미지 EMA 콜백 / 로드/제거 ───────────────────────────────
    def _on_arm_smooth_change(self, _=None):
        v = self._arm_smooth_var.get()
        alpha = 1.0 - (v / 100.0)
        self._arm_img_ema['alpha'] = alpha
        self._arm_img_ema_l['alpha'] = alpha

    def _on_leg_smooth_change(self, _=None):
        v = self._leg_smooth_var.get()
        alpha = 1.0 - (v / 100.0)
        self._leg_img_ema_r['alpha'] = alpha
        self._leg_img_ema_l['alpha'] = alpha

    def _on_body_smooth_change(self, _=None):
        # 앞모습/옆모습 각자 슬라이더가 있지만 공통 alpha 사용
        vf = self._body_front_smooth_var.get()
        vs = self._body_side_smooth_var.get()
        self._body_front_ema['alpha'] = 1.0 - (vf / 100.0)
        self._body_side_ema['alpha']  = 1.0 - (vs / 100.0)

    def _on_shoe_smooth_change(self, _=None):
        alpha = 1.0 - (self._shoe_smooth_var.get() / 100.0)
        self._shoe_img_ema_r['alpha'] = alpha
        self._shoe_img_ema_l['alpha'] = alpha

    def _on_glove_smooth_change(self, _=None):
        alpha = 1.0 - (self._glove_smooth_var.get() / 100.0)
        self._glove_img_ema_r['alpha'] = alpha
        self._glove_img_ema_l['alpha'] = alpha

    def _on_weapon_smooth_change(self, _=None):
        alpha = 1.0 - (self._weapon_smooth_var.get() / 100.0)
        self._weapon_img_ema_r['alpha'] = alpha
        self._weapon_img_ema_l['alpha'] = alpha

    # ── 신발 이미지 로드/제거 ────────────────────────────────────────────────
    def _toggle_shoe_image(self, side='right'):
        attr = '_shoe_img_r' if side == 'right' else '_shoe_img_l'
        lbl  = self._shoe_img_lbl_r if side == 'right' else self._shoe_img_lbl_l
        btn  = self._shoe_img_btn_r if side == 'right' else self._shoe_img_btn_l
        ema  = '_shoe_img_ema_r' if side == 'right' else '_shoe_img_ema_l'
        if getattr(self, attr) is not None:
            setattr(self, attr, None)
            for _k in ('ankle_x', 'ankle_y', 'angle', 'shin_len'):
                getattr(self, ema)[_k] = None
            lbl.config(text="미선택")
            btn.config(text="👟  이미지 로드")
        else:
            self._load_shoe_image(side=side)

    def _load_shoe_image(self, side='right'):
        title = "오른발 신발 이미지 선택" if side == 'right' else "왼발 신발 이미지 선택"
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
        h_i, w_i = img.shape[:2]
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 3:
            tmp = np.zeros((h_i, w_i, 4), dtype=np.uint8)
            tmp[:, :, :3] = img; tmp[:, :, 3] = 255
            img = tmp
        lbl = self._shoe_img_lbl_r if side == 'right' else self._shoe_img_lbl_l
        btn = self._shoe_img_btn_r if side == 'right' else self._shoe_img_btn_l
        if side == 'right':
            self._shoe_img_r = img.copy()
        else:
            self._shoe_img_l = img.copy()
        lbl.config(text=os.path.basename(path))
        btn.config(text="× 이미지 제거")

    # ── 장갑 이미지 로드/제거 ────────────────────────────────────────────────
    def _toggle_glove_image(self, side='right'):
        attr = '_glove_img_r' if side == 'right' else '_glove_img_l'
        lbl  = self._glove_img_lbl_r if side == 'right' else self._glove_img_lbl_l
        btn  = self._glove_img_btn_r if side == 'right' else self._glove_img_btn_l
        ema  = '_glove_img_ema_r' if side == 'right' else '_glove_img_ema_l'
        if getattr(self, attr) is not None:
            setattr(self, attr, None)
            for _k in ('wrist_x', 'wrist_y', 'angle', 'palm_len'):
                getattr(self, ema)[_k] = None
            lbl.config(text="미선택")
            btn.config(text="🧤  이미지 로드")
        else:
            self._load_glove_image(side=side)

    def _load_glove_image(self, side='right'):
        title = "오른손 장갑 이미지 선택" if side == 'right' else "왼손 장갑 이미지 선택"
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
        h_i, w_i = img.shape[:2]
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 3:
            tmp = np.zeros((h_i, w_i, 4), dtype=np.uint8)
            tmp[:, :, :3] = img; tmp[:, :, 3] = 255
            img = tmp
        lbl = self._glove_img_lbl_r if side == 'right' else self._glove_img_lbl_l
        btn = self._glove_img_btn_r if side == 'right' else self._glove_img_btn_l
        if side == 'right':
            self._glove_img_r = img.copy()
        else:
            self._glove_img_l = img.copy()
        lbl.config(text=os.path.basename(path))
        btn.config(text="× 이미지 제거")

    # ── 무기 이미지 로드/제거 ────────────────────────────────────────────────
    def _toggle_weapon_image(self):
        if self._weapon_img is not None:
            self._weapon_img = None
            for _k in ('wrist_x', 'wrist_y', 'angle', 'palm_len'):
                self._weapon_img_ema_r[_k] = None
                self._weapon_img_ema_l[_k] = None
            self._weapon_img_lbl.config(text="미선택")
            self._weapon_img_btn.config(text="⚔  이미지 로드")
        else:
            self._load_weapon_image()

    def _load_weapon_image(self):
        path = filedialog.askopenfilename(
            parent=self.win, title="무기 이미지 선택",
            filetypes=[("이미지 파일", "*.png *.jpg *.jpeg *.bmp"), ("모든 파일", "*.*")],
        )
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            messagebox.showerror("오류", f"이미지를 열 수 없습니다:\n{path}", parent=self.win)
            return
        h_i, w_i = img.shape[:2]
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 3:
            tmp = np.zeros((h_i, w_i, 4), dtype=np.uint8)
            tmp[:, :, :3] = img; tmp[:, :, 3] = 255
            img = tmp
        self._weapon_img = img.copy()
        self._weapon_img_lbl.config(text=os.path.basename(path))
        self._weapon_img_btn.config(text="× 이미지 제거")

    # ── 앞모습 몸통 ──────────────────────────────────────────────────────────
    def _toggle_body_front_image(self):
        if self._body_front_img is not None:
            self._body_front_img = None
            self._body_front_pins = None
            for k in ('b_lsx','b_lsy','b_rsx','b_rsy','b_rhx','b_rhy','b_lhx','b_lhy'):
                self._body_front_ema[k] = None
            self._body_front_img_lbl.config(text="미선택")
            self._body_front_img_btn.config(text="👕  이미지 로드")
            self._body_front_pin_lbl.config(text="피벗 미설정", fg="#ffaa44")
            self._body_front_pin_btn.config(text="🎯 피벗 설정", state=tk.DISABLED)
            self._det_cache = None
            self._refresh_frame()
        else:
            self._load_body_image(mode='front')

    def _toggle_body_side_image(self):
        if self._body_side_img is not None:
            self._body_side_img = None
            self._body_side_pins = None
            for k in ('b_scx','b_scy','b_hcx','b_hcy'):
                self._body_side_ema[k] = None
            self._body_side_img_lbl.config(text="미선택")
            self._body_side_img_btn.config(text="👘  이미지 로드")
            self._body_side_pin_lbl.config(text="피벗 미설정", fg="#ffaa44")
            self._body_side_pin_btn.config(text="🎯 피벗 설정", state=tk.DISABLED)
            self._det_cache = None
            self._refresh_frame()
        else:
            self._load_body_image(mode='side')

    def _load_body_image(self, mode='front'):
        title = "몸통 앞모습 이미지 선택" if mode == 'front' else "몸통 옆모습 이미지 선택"
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
        if mode == 'front':
            self._body_front_img = img.copy()
            self._body_front_pins = None
            self._body_front_img_lbl.config(text=os.path.basename(path))
            self._body_front_img_btn.config(text="× 이미지 제거")
            self._body_front_pin_lbl.config(text="피벗 미설정", fg="#ffaa44")
            self._body_front_pin_btn.config(text="🎯 피벗 설정", state=tk.NORMAL)
        else:
            self._body_side_img = img.copy()
            self._body_side_pins = None
            self._body_side_img_lbl.config(text=os.path.basename(path))
            self._body_side_img_btn.config(text="× 이미지 제거")
            self._body_side_pin_lbl.config(text="피벗 미설정", fg="#ffaa44")
            self._body_side_pin_btn.config(text="🎯 피벗 설정", state=tk.NORMAL)
        self._det_cache = None
        self._refresh_frame()
        if mode == 'front':
            self.win.after(100, self._open_body_front_pin_picker)
        else:
            self.win.after(100, self._open_body_side_pin_picker)

    # ── 앞모습 핀 피커 ────────────────────────────────────────────────────────
    def _open_body_front_pin_picker(self):
        if self._body_front_img is None:
            return
        if self._pin_popup is not None:
            try:
                self._pin_popup.lift(); return
            except Exception:
                self._pin_popup = None
        popup = tk.Toplevel(self.win)
        popup.title("피벗 핀 설정 (몸통 앞모습)")
        popup.resizable(False, False)
        popup.grab_set()
        self._pin_popup = popup

        def _on_close():
            self._pin_popup = None
            popup.destroy()
        popup.protocol("WM_DELETE_WINDOW", _on_close)

        _clicks = []
        _COLORS = ["#ff4444", "#ffdd00", "#44ff88", "#4488ff"]
        _LABELS = ["왼어깨 (L.Shoulder)", "오른어깨 (R.Shoulder)",
                   "오른엉덩이 (R.Hip)",  "왼엉덩이 (L.Hip)"]
        _MARKER_R = 6

        status_lbl = tk.Label(popup, text=f"[1/4] {_LABELS[0]} 위치를 클릭하세요",
                               font=("Segoe UI", 9), fg="#aaccff",
                               bg="#1a1a2e", anchor="w", padx=8, pady=4)
        status_lbl.pack(fill=tk.X)

        img_bgra = self._body_front_img
        ih, iw = img_bgra.shape[:2]
        scale_f = min(420 / iw, 420 / ih, 1.0)
        disp_w, disp_h = max(1, int(iw*scale_f)), max(1, int(ih*scale_f))
        img_rgb = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2RGB)
        pil_img = __import__('PIL').Image.fromarray(img_rgb).resize(
            (disp_w, disp_h), __import__('PIL').Image.LANCZOS)
        _tk_img = ImageTk.PhotoImage(pil_img)
        canvas = tk.Canvas(popup, width=disp_w, height=disp_h,
                            bg="#000011", highlightthickness=1, highlightbackground="#333355")
        canvas.pack(padx=10, pady=6)
        canvas.create_image(0, 0, anchor="nw", image=_tk_img)
        canvas._tk_img_ref = _tk_img

        if self._body_front_pins is not None:
            for _pi, pt in enumerate(self._body_front_pins.arrays()):
                cx, cy = pt[0]*scale_f, pt[1]*scale_f
                _clicks.append(tuple(pt))
                canvas.create_oval(cx-_MARKER_R, cy-_MARKER_R, cx+_MARKER_R, cy+_MARKER_R,
                                    fill=_COLORS[_pi], outline="white", width=1)

        def _update_status():
            n = len(_clicks)
            if n < 4:
                status_lbl.config(text=f"[{n+1}/4] {_LABELS[n]} 위치를 클릭하세요", fg="#aaccff")
                ok_btn.config(state=tk.DISABLED)
            else:
                status_lbl.config(text="4점 완료 — [확인]으로 저장하세요", fg="#44ff88")
                ok_btn.config(state=tk.NORMAL)

        def _on_canvas_click(event):
            if len(_clicks) >= 4:
                return
            ix, iy = event.x / scale_f, event.y / scale_f
            _clicks.append((ix, iy))
            idx = len(_clicks) - 1
            canvas.create_oval(event.x-_MARKER_R, event.y-_MARKER_R,
                                event.x+_MARKER_R, event.y+_MARKER_R,
                                fill=_COLORS[idx], outline="white", width=1)
            _update_status()

        canvas.bind("<Button-1>", _on_canvas_click)

        btn_row = tk.Frame(popup, bg="#1a1a2e")
        btn_row.pack(fill=tk.X, padx=10, pady=(0, 8))

        def _reset():
            nonlocal _clicks; _clicks = []
            canvas.delete("all"); canvas.create_image(0, 0, anchor="nw", image=_tk_img)
            status_lbl.config(text=f"[1/4] {_LABELS[0]} 위치를 클릭하세요", fg="#aaccff")
            ok_btn.config(state=tk.DISABLED)

        def _confirm():
            if len(_clicks) < 4:
                return
            pins = BodyPins(img_l_shldr=_clicks[0], img_r_shldr=_clicks[1],
                            img_r_hip=_clicks[2],   img_l_hip=_clicks[3])
            if not pins.is_valid(min_dist=6.0):
                messagebox.showwarning("경고", "핀 간격이 너무 좁습니다.", parent=popup); return
            self._body_front_pins = pins
            self._body_front_pin_lbl.config(
                text="왼어깨 ○  오른어깨 ○  오른엉덩이 ○  왼엉덩이 ○", fg="#44ff88")
            self._body_front_pin_btn.config(text="🎯 피벗 재설정")
            self._det_cache = None; self._refresh_frame(); _on_close()

        tk.Button(btn_row, text="초기화", font=("Segoe UI", 9), bg="#3a2a2a", fg=TEXT_W,
                  relief=tk.FLAT, cursor="hand2", command=_reset).pack(side=tk.LEFT, padx=(0,6))
        tk.Button(btn_row, text="취소", font=("Segoe UI", 9), bg="#2a2a3a", fg=TEXT_W,
                  relief=tk.FLAT, cursor="hand2", command=_on_close).pack(side=tk.LEFT, padx=(0,6))
        ok_btn = tk.Button(btn_row, text="확인", font=("Segoe UI", 9, "bold"), bg="#1e5f3a",
                           fg=TEXT_W, relief=tk.FLAT, cursor="hand2",
                           state=tk.DISABLED, command=_confirm)
        ok_btn.pack(side=tk.LEFT)
        _update_status()

    # ── 옆모습 핀 피커 ────────────────────────────────────────────────────────
    def _open_body_side_pin_picker(self):
        if self._body_side_img is None:
            return
        if self._pin_popup is not None:
            try:
                self._pin_popup.lift(); return
            except Exception:
                self._pin_popup = None
        popup = tk.Toplevel(self.win)
        popup.title("피벗 핀 설정 (몸통 옆모습)")
        popup.resizable(False, False)
        popup.grab_set()
        self._pin_popup = popup

        def _on_close():
            self._pin_popup = None
            popup.destroy()
        popup.protocol("WM_DELETE_WINDOW", _on_close)

        _clicks = []
        _COLORS = ["#ff4444", "#ffdd00", "#44ff88", "#4488ff"]
        _LABELS = ["어깨 (뒤)", "앞가슴 (Front Chest)",
                   "앞엉덩이 (Front Hip)", "뒤허리 (Back Waist)"]
        _MARKER_R = 6

        status_lbl = tk.Label(popup, text=f"[1/4] {_LABELS[0]} 위치를 클릭하세요",
                               font=("Segoe UI", 9), fg="#aaccff",
                               bg="#1a1a2e", anchor="w", padx=8, pady=4)
        status_lbl.pack(fill=tk.X)

        img_bgra = self._body_side_img
        ih, iw = img_bgra.shape[:2]
        scale_f = min(420 / iw, 420 / ih, 1.0)
        disp_w, disp_h = max(1, int(iw*scale_f)), max(1, int(ih*scale_f))
        img_rgb = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2RGB)
        pil_img = __import__('PIL').Image.fromarray(img_rgb).resize(
            (disp_w, disp_h), __import__('PIL').Image.LANCZOS)
        _tk_img = ImageTk.PhotoImage(pil_img)
        canvas = tk.Canvas(popup, width=disp_w, height=disp_h,
                            bg="#000011", highlightthickness=1, highlightbackground="#333355")
        canvas.pack(padx=10, pady=6)
        canvas.create_image(0, 0, anchor="nw", image=_tk_img)
        canvas._tk_img_ref = _tk_img

        if self._body_side_pins is not None:
            for _pi, pt in enumerate(self._body_side_pins.arrays()):
                cx, cy = pt[0]*scale_f, pt[1]*scale_f
                _clicks.append(tuple(pt))
                canvas.create_oval(cx-_MARKER_R, cy-_MARKER_R, cx+_MARKER_R, cy+_MARKER_R,
                                    fill=_COLORS[_pi], outline="white", width=1)

        def _update_status():
            n = len(_clicks)
            if n < 4:
                status_lbl.config(text=f"[{n+1}/4] {_LABELS[n]} 위치를 클릭하세요", fg="#aaccff")
                ok_btn.config(state=tk.DISABLED)
            else:
                status_lbl.config(text="4점 완료 — [확인]으로 저장하세요", fg="#44ff88")
                ok_btn.config(state=tk.NORMAL)

        def _on_canvas_click(event):
            if len(_clicks) >= 4:
                return
            ix, iy = event.x / scale_f, event.y / scale_f
            _clicks.append((ix, iy))
            idx = len(_clicks) - 1
            canvas.create_oval(event.x-_MARKER_R, event.y-_MARKER_R,
                                event.x+_MARKER_R, event.y+_MARKER_R,
                                fill=_COLORS[idx], outline="white", width=1)
            _update_status()

        canvas.bind("<Button-1>", _on_canvas_click)

        btn_row = tk.Frame(popup, bg="#1a1a2e")
        btn_row.pack(fill=tk.X, padx=10, pady=(0, 8))

        def _reset():
            nonlocal _clicks; _clicks = []
            canvas.delete("all"); canvas.create_image(0, 0, anchor="nw", image=_tk_img)
            status_lbl.config(text=f"[1/4] {_LABELS[0]} 위치를 클릭하세요", fg="#aaccff")
            ok_btn.config(state=tk.DISABLED)

        def _confirm():
            if len(_clicks) < 4:
                return
            pins = BodySidePins(img_shldr=_clicks[0],       img_front_chest=_clicks[1],
                                img_front_hip=_clicks[2],   img_back_waist=_clicks[3])
            if not pins.is_valid(min_dist=6.0):
                messagebox.showwarning("경고", "핀 간격이 너무 좁습니다.", parent=popup); return
            self._body_side_pins = pins
            self._body_side_pin_lbl.config(
                text="어깨(뒤) ○  앞가슴 ○  앞엉덩이 ○  뒤허리 ○", fg="#44ff88")
            self._body_side_pin_btn.config(text="🎯 피벗 재설정")
            self._det_cache = None; self._refresh_frame(); _on_close()

        tk.Button(btn_row, text="초기화", font=("Segoe UI", 9), bg="#3a2a2a", fg=TEXT_W,
                  relief=tk.FLAT, cursor="hand2", command=_reset).pack(side=tk.LEFT, padx=(0,6))
        tk.Button(btn_row, text="취소", font=("Segoe UI", 9), bg="#2a2a3a", fg=TEXT_W,
                  relief=tk.FLAT, cursor="hand2", command=_on_close).pack(side=tk.LEFT, padx=(0,6))
        ok_btn = tk.Button(btn_row, text="확인", font=("Segoe UI", 9, "bold"), bg="#1e5f3a",
                           fg=TEXT_W, relief=tk.FLAT, cursor="hand2",
                           state=tk.DISABLED, command=_confirm)
        ok_btn.pack(side=tk.LEFT)
        _update_status()

    def _toggle_leg_image(self, side='right'):
        if side == 'right':
            img_attr  = '_leg_img_r';   pins_attr  = '_leg_pins_r';   cache_attr  = '_leg_seg_cache_r'
            ema_attr  = '_leg_img_ema_r'; lbl = self._leg_img_lbl_r;  btn = self._leg_img_btn_r
            pin_lbl   = self._leg_pin_lbl_r;  pin_btn = self._leg_pin_btn_r
        else:
            img_attr  = '_leg_img_l';   pins_attr  = '_leg_pins_l';   cache_attr  = '_leg_seg_cache_l'
            ema_attr  = '_leg_img_ema_l'; lbl = self._leg_img_lbl_l;  btn = self._leg_img_btn_l
            pin_lbl   = self._leg_pin_lbl_l;  pin_btn = self._leg_pin_btn_l

        if getattr(self, img_attr) is not None:
            setattr(self, img_attr, None)
            setattr(self, pins_attr, None)
            setattr(self, cache_attr, None)
            for _k in ('knee_x', 'knee_y', 'angle', 'leg_len',
                       'hip_x', 'hip_y', 'ankle_x', 'ankle_y',
                       'foot_x', 'foot_y'):
                getattr(self, ema_attr)[_k] = None
            lbl.config(text="미선택")
            btn.config(text="🦵  이미지 로드")
            pin_lbl.config(text="피벗 미설정", fg="#ffaa44")
            pin_btn.config(text="🎯 피벗 설정", state=tk.DISABLED)
            self._det_cache = None
            self._refresh_frame()
        else:
            self._load_leg_image(side=side)

    def _load_leg_image(self, side='right'):
        title = "오른발 이미지 선택" if side == 'right' else "왼발 이미지 선택"
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
            self._leg_img_r = img.copy()
            self._leg_pins_r = None
            self._leg_seg_cache_r = None
            self._leg_img_lbl_r.config(text=os.path.basename(path))
            self._leg_img_btn_r.config(text="× 이미지 제거")
            self._leg_pin_lbl_r.config(text="피벗 미설정", fg="#ffaa44")
            self._leg_pin_btn_r.config(text="🎯 피벗 설정", state=tk.NORMAL)
        else:
            self._leg_img_l = img.copy()
            self._leg_pins_l = None
            self._leg_seg_cache_l = None
            self._leg_img_lbl_l.config(text=os.path.basename(path))
            self._leg_img_btn_l.config(text="× 이미지 제거")
            self._leg_pin_lbl_l.config(text="피벗 미설정", fg="#ffaa44")
            self._leg_pin_btn_l.config(text="🎯 피벗 설정", state=tk.NORMAL)
        self._det_cache = None
        self._refresh_frame()
        self.win.after(100, lambda: self._open_leg_pin_picker(side=side))

    # ── 다리 Puppet Pin 피벗 설정 팝업 ────────────────────────────────────
    def _open_leg_pin_picker(self, side='right'):
        if not _PUPPET_AVAILABLE:
            messagebox.showwarning("미지원", "puppet_pin 모듈을 불러올 수 없습니다.", parent=self.win)
            return
        leg_img  = self._leg_img_r  if side == 'right' else self._leg_img_l
        leg_pins = self._leg_pins_r if side == 'right' else self._leg_pins_l
        pin_lbl  = self._leg_pin_lbl_r  if side == 'right' else self._leg_pin_lbl_l
        pin_btn  = self._leg_pin_btn_r  if side == 'right' else self._leg_pin_btn_l
        if leg_img is None:
            return
        if self._pin_popup is not None:
            try:
                self._pin_popup.lift()
                return
            except Exception:
                self._pin_popup = None

        popup = tk.Toplevel(self.win)
        popup.title(f"피벗 핀 설정 ({'오른발' if side == 'right' else '왼발'})")
        popup.resizable(False, False)
        popup.grab_set()
        self._pin_popup = popup

        def _on_close():
            self._pin_popup = None
            popup.destroy()
        popup.protocol("WM_DELETE_WINDOW", _on_close)

        _clicks = []
        _COLORS = ["#ff4444", "#ffdd00", "#44ff88", "#4488ff"]
        _LABELS = ["엉덩이 (Hip)", "무릎 (Knee)", "발목 (Ankle)", "발끝 (Tiptoe)"]
        _MARKER_R = 6

        status_lbl = tk.Label(popup, text=f"[1/4] {_LABELS[0]} 위치를 클릭하세요",
                               font=("Segoe UI", 9), fg="#aaccff",
                               bg="#1a1a2e", anchor="w", padx=8, pady=4)
        status_lbl.pack(fill=tk.X)

        img_bgra = leg_img
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

        if leg_pins is not None:
            pts_img = list(leg_pins.arrays())
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
                img_shldr=_clicks[0],  # 엉덩이 (Hip)
                img_elbow=_clicks[1],  # 무릎 (Knee)
                img_wrist=_clicks[2],  # 발목 (Ankle)
                img_hand=_clicks[3] if len(_clicks) >= 4 else None,  # 발끝 (Tiptoe)
            )
            if pins_degenerate(pins, min_dist=6.0):
                messagebox.showwarning("경고",
                    "핀 간격이 너무 좁습니다. 더 멀리 클릭해주세요.",
                    parent=popup)
                return
            if side == 'right':
                self._leg_pins_r      = pins
                self._leg_seg_cache_r = build_segment_cache(self._leg_img_r, pins)
            else:
                self._leg_pins_l      = pins
                self._leg_seg_cache_l = build_segment_cache(self._leg_img_l, pins)
            if pins.img_hand is not None:
                pin_lbl.config(text="엉덩이 ○  무릎 ○  발목 ○  발끝 ○", fg="#44ff88")
            else:
                pin_lbl.config(text="엉덩이 ○  무릎 ○  발목 ○", fg="#44ff88")
            pin_btn.config(text="🎯 피벗 재설정")
            self._det_cache = None
            self._refresh_frame()
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
            self._det_cache = None
            self._refresh_frame()
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
        self._det_cache = None
        self._refresh_frame()
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

        # 팝업 내부 상태
        _clicks = []      # [(x_img, y_img), ...]  최대 4개
        _COLORS = ["#ff4444", "#ffdd00", "#44ff88", "#4488ff"]
        _LABELS = ["어깨 (Shoulder)", "팔꿈치 (Elbow)", "손목 (Wrist)", "손가락 끝 (Hand)"]
        _MARKER_R = 6

        # 상태 레이블
        status_lbl = tk.Label(popup, text=f"[1/4] {_LABELS[0]} 위치를 클릭하세요",
                               font=("Segoe UI", 9), fg="#aaccff",
                               bg="#1a1a2e", anchor="w", padx=8, pady=4)
        status_lbl.pack(fill=tk.X)

        # 이미지 캔버스
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
        canvas._tk_img_ref = _tk_img   # GC 방지

        # 기존 핀 미리 표시
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

        # 버튼 행
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
            self._det_cache = None
            self._refresh_frame()
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
            self._face_img_ref_h = None
            self._face_pivot = None
            self._face_rot_var.set(0)
            self._face_img_side = None
            self._face_img_side_pts = None
            self._face_img_side_ref_h = None
            self._face_img_side_anchors = None
            self._face_img_side_kps_n = None
            self._face_img_kps_n = None
            for _k in ('face_h', 'eye_cx', 'eye_cy', 'angle'):
                self._face_img_ema[_k] = None
            for _k in ('face_h', 'eye_cx', 'eye_cy', 'angle',
                       's_eye_x', 's_eye_y', 's_nose_x', 's_nose_y',
                       's_ear_dist', 's_angle'):
                self._face_img_side_ema[_k] = None
            self._face_img_lbl.config(text="미선택")
            self._face_img_btn.config(text="🖼  이미지 로드")
            self._face_pivot_btn.config(text="⊕ 피벗 설정", state=tk.DISABLED)
            self._face_img_side_btn.config(text="🖼  옆모습 이미지 로드")
            self._face_img_side_lbl.config(text="미선택")
            self._face_side_pivot_btn.config(text="⊕ 앵커 설정 (옆)", state=tk.DISABLED)
            self._det_cache = None
            self._refresh_frame()
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

        # 불투명 이미지(JPG 등)는 흰 배경을 투명 처리 → 콘텐츠 박스 자동 추정 가능
        if img[:, :, 3].min() == 255:
            white = (img[:, :, 0] > 240) & (img[:, :, 1] > 240) & (img[:, :, 2] > 240)
            img[white, 3] = 0

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
        # 자동 피벗/위치/크기: 감지=눈 중심+bbox, 실패=콘텐츠 중앙X·40%Y+콘텐츠높이
        piv, ref_h = self._auto_face_pivot(img, face_res)
        self._face_pivot = piv
        self._face_img_ref_h = ref_h
        # 피벗 피커용 kps_n 저장 (오른눈, 왼눈, 코)
        if face_res.face_landmarks:
            lf = face_res.face_landmarks[0]
            self._face_img_kps_n = [(lf[33].x, lf[33].y),
                                    (lf[263].x, lf[263].y),
                                    (lf[4].x, lf[4].y)]
        else:
            self._face_img_kps_n = None
        self._det_cache = None
        self._face_img_lbl.config(text=os.path.basename(path) + mode_text)
        self._face_img_btn.config(text="× 이미지 제거")
        _px, _py = int(piv[0] * 100), int(piv[1] * 100)
        self._face_pivot_btn.config(state=tk.NORMAL, text=f"⊕ 피벗 ({_px}%,{_py}%) 자동")
        self._side_thr_scale.config(state=tk.NORMAL)
        self._refresh_frame()

    def _auto_face_pivot(self, img, face_res):
        """(pivot_norm, ref_h_px) 반환. 감지 성공=눈 중심+bbox 높이,
        실패=다크블롭 눈 탐지(신뢰 시) 또는 콘텐츠 상단40% 폴백. 크기는 콘텐츠 높이."""
        h, w = img.shape[:2]
        if face_res.face_landmarks and len(face_res.face_landmarks[0]) > 263:
            lf = face_res.face_landmarks[0]
            ex = (lf[33].x + lf[263].x) / 2.0
            ey = (lf[33].y + lf[263].y) / 2.0
            if hasattr(lf, 'bbox'):
                ref_h = float(lf.bbox[3] - lf.bbox[1])
            else:
                ys = [p.y * h for p in lf]
                ref_h = max(ys) - min(ys)
            return (ex, ey), ref_h
        bbox = _alpha_content_bbox(img)
        x0, y0, x1, y1 = bbox
        ref_h = float(y1 - y0)                       # 크기 기준: 콘텐츠 높이 유지
        eye = _detect_eye_pivot(img, bbox)           # 다크블롭 눈 탐지
        if eye is not None:
            return eye[0], ref_h                     # 신뢰도 충분 → 눈 중점 피벗
        return ((x0 + x1) / 2.0 / w, (y0 + (y1 - y0) * EYE_FRAC) / h), ref_h  # 폴백

    def _open_face_pivot_picker(self):
        """얼굴 이미지 위에서 눈/코 포인트 또는 자유 클릭으로 피벗 설정."""
        if self._face_img is None:
            return
        img_bgra = self._face_img
        img_h, img_w = img_bgra.shape[:2]

        max_side = 420
        scale = min(max_side / img_w, max_side / img_h, 1.0)
        dw, dh = max(1, int(img_w * scale)), max(1, int(img_h * scale))

        top = tk.Toplevel(self.win)
        top.title("피벗 포인트 설정")
        top.resizable(False, False)
        top.grab_set()

        kps = self._face_img_kps_n  # [(nx,ny)×3] 또는 None
        hint = "눈/코 포인트를 클릭하거나 자유롭게 클릭하세요." if kps else "피벗 포인트를 클릭하세요."
        tk.Label(top, text=hint, font=("Segoe UI", 9)).pack(padx=10, pady=(8, 4))

        canvas = tk.Canvas(top, width=dw, height=dh, cursor="crosshair",
                            highlightthickness=1, highlightbackground="#555")
        canvas.pack(padx=10, pady=(0, 4))

        from PIL import Image as _PI, ImageTk as _PIT
        rgb = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2RGB)
        photo = _PIT.PhotoImage(_PI.fromarray(rgb).resize((dw, dh), _PI.LANCZOS))
        canvas.create_image(0, 0, anchor="nw", image=photo)
        canvas._photo = photo

        pivot_ref = [self._face_pivot]

        # 눈/코 랜드마크 포인트 표시
        _KP_LABELS = [("오른눈", "#55aaff"), ("왼눈", "#55aaff"), ("코", "#55ff88")]
        if kps:
            for i, ((nx, ny), (label, color)) in enumerate(zip(kps, _KP_LABELS)):
                x, y = int(nx * dw), int(ny * dh)
                r = 13
                ov = canvas.create_oval(x-r, y-r, x+r, y+r,
                                        outline=color, fill="#0a0a2a", width=2,
                                        tags=f"kp{i}")
                tx = canvas.create_text(x, y-r-9, text=label, fill=color,
                                        font=("Segoe UI", 8, "bold"), tags=f"kp{i}")
                def _kp_click(event, _nx=nx, _ny=ny):
                    pivot_ref[0] = (_nx, _ny)
                    _draw_marker(_nx, _ny)
                canvas.tag_bind(ov, "<Button-1>", _kp_click)
                canvas.tag_bind(tx, "<Button-1>", _kp_click)

        def _draw_marker(nx, ny):
            canvas.delete("pivot_marker")
            x, y = int(nx * dw), int(ny * dh)
            r = 9
            canvas.create_oval(x-r, y-r, x+r, y+r,
                                outline="#ff4444", width=2, tags="pivot_marker")
            canvas.create_line(x-r-4, y, x+r+4, y,
                                fill="#ff4444", width=2, tags="pivot_marker")
            canvas.create_line(x, y-r-4, x, y+r+4,
                                fill="#ff4444", width=2, tags="pivot_marker")

        if pivot_ref[0] is not None:
            _draw_marker(*pivot_ref[0])

        def _on_click(event):
            nx = max(0.0, min(1.0, event.x / dw))
            ny = max(0.0, min(1.0, event.y / dh))
            pivot_ref[0] = (nx, ny)
            _draw_marker(nx, ny)

        canvas.bind("<Button-1>", _on_click)

        def _ok():
            self._face_pivot = pivot_ref[0]
            if self._face_pivot is not None:
                px = int(self._face_pivot[0] * 100)
                py = int(self._face_pivot[1] * 100)
                self._face_pivot_btn.config(text=f"⊕ 피벗 ({px}%, {py}%)")
            else:
                self._face_pivot_btn.config(text="⊕ 피벗 설정")
            self._det_cache = None
            self._refresh_frame()
            top.destroy()

        def _reset():
            pivot_ref[0] = None
            canvas.delete("pivot_marker")

        btn_row = tk.Frame(top)
        btn_row.pack(pady=(0, 8))
        tk.Button(btn_row, text="리셋", width=7, command=_reset).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_row, text="취소", width=7, command=top.destroy).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_row, text="확인", width=7, command=_ok).pack(side=tk.LEFT, padx=4)

    # ── 입 벌림 이미지 로드/제거 ──────────────────────────────────────────
    def _toggle_face_image_open(self):
        if self._face_img_open is not None:
            self._face_img_open = None
            self._face_img_open_pts = None
            self._face_img_open_ref_h = None
            self._face_img_open_lbl.config(text="미선택")
            self._face_img_open_btn.config(text="🖼  입 벌림 이미지 로드")
            self._mouth_thr_scale.config(state=tk.DISABLED)
            self._det_cache = None
            self._refresh_frame()
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
        self._face_img_open_ref_h = self._auto_face_pivot(img, face_res)[1]
        self._det_cache = None
        self._face_img_open_lbl.config(text=os.path.basename(path) + mode_text)
        self._face_img_open_btn.config(text="× 입 벌림 제거")
        self._mouth_thr_scale.config(state=tk.NORMAL)
        self._refresh_frame()

    def _open_face_side_pivot_picker(self):
        """옆모습 이미지에서 눈/코/입/목 앵커 포인트를 설정."""
        if self._face_img_side is None:
            return
        img_bgra = self._face_img_side
        img_h, img_w = img_bgra.shape[:2]

        max_side = 440
        scale = min(max_side / img_w, max_side / img_h, 1.0)
        dw, dh = max(1, int(img_w * scale)), max(1, int(img_h * scale))

        top = tk.Toplevel(self.win)
        top.title("앵커 포인트 설정 (옆모습)")
        top.resizable(False, False)
        top.grab_set()

        mode_var = tk.StringVar(value="eye")
        mode_row = tk.Frame(top, bg="#1a1a2e")
        mode_row.pack(fill=tk.X, padx=10, pady=(8, 2))
        tk.Label(mode_row, text="설정할 포인트:", font=("Segoe UI", 9),
                 bg="#1a1a2e", fg="#cccccc").pack(side=tk.LEFT, padx=(4, 8))
        tk.Radiobutton(mode_row, text="눈 (Iris)", variable=mode_var, value="eye",
                       font=("Segoe UI", 9, "bold"), fg="#55aaff",
                       bg="#1a1a2e", selectcolor="#0a0a2a",
                       activebackground="#1a1a2e").pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(mode_row, text="코 (Nose)", variable=mode_var, value="nose",
                       font=("Segoe UI", 9, "bold"), fg="#55ff88",
                       bg="#1a1a2e", selectcolor="#0a0a2a",
                       activebackground="#1a1a2e").pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(mode_row, text="입 (Mouth)", variable=mode_var, value="mouth",
                       font=("Segoe UI", 9, "bold"), fg="#ffaa33",
                       bg="#1a1a2e", selectcolor="#0a0a2a",
                       activebackground="#1a1a2e").pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(mode_row, text="목 (Neck)", variable=mode_var, value="neck",
                       font=("Segoe UI", 9, "bold"), fg="#ff55cc",
                       bg="#1a1a2e", selectcolor="#0a0a2a",
                       activebackground="#1a1a2e").pack(side=tk.LEFT, padx=5)

        canvas = tk.Canvas(top, width=dw, height=dh, cursor="crosshair",
                            highlightthickness=1, highlightbackground="#555")
        canvas.pack(padx=10, pady=(4, 4))

        from PIL import Image as _PI, ImageTk as _PIT
        rgb = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2RGB)
        photo = _PIT.PhotoImage(_PI.fromarray(rgb).resize((dw, dh), _PI.LANCZOS))
        canvas.create_image(0, 0, anchor="nw", image=photo)
        canvas._photo = photo

        existing = self._face_img_side_anchors or {}
        anchors_ref = {
            'eye':   existing.get('eye'),
            'nose':  existing.get('nose'),
            'mouth': existing.get('mouth'),
            'neck':  existing.get('neck'),
        }

        kps = self._face_img_side_kps_n
        if kps and len(kps) >= 5:
            _KP_DATA = [
                (kps[0], "오른눈", "#55aaff", "eye"),
                (kps[1], "왼눈",   "#55aaff", "eye"),
                (kps[2], "코",     "#55ff88", "nose"),
                (kps[3], "입",     "#ffaa33", "mouth"),
                (kps[4], "목시작", "#ff55cc", "neck"),
            ]
            for (nx, ny), label, color, atype in _KP_DATA:
                x, y = int(nx * dw), int(ny * dh)
                r = 13
                ov = canvas.create_oval(x-r, y-r, x+r, y+r,
                                        outline=color, fill="#0a0a2a", width=2)
                tx = canvas.create_text(x, y-r-9, text=label, fill=color,
                                        font=("Segoe UI", 8, "bold"))
                def _kp_click(event, _nx=nx, _ny=ny, _t=atype):
                    anchors_ref[_t] = (_nx, _ny)
                    _draw_markers()
                    _update_status()
                canvas.tag_bind(ov, "<Button-1>", _kp_click)
                canvas.tag_bind(tx, "<Button-1>", _kp_click)

        status_lbl = tk.Label(top, font=("Segoe UI", 8), fg="#aaaacc", bg="#1a1a2e")
        status_lbl.pack(fill=tk.X, padx=10, pady=(0, 2))

        def _fmt(pt):
            return f"({int(pt[0]*100)}%,{int(pt[1]*100)}%)" if pt else "-"

        def _update_status():
            status_lbl.config(text=(
                f"눈:{_fmt(anchors_ref['eye'])}  "
                f"코:{_fmt(anchors_ref['nose'])}  "
                f"입:{_fmt(anchors_ref['mouth'])}  "
                f"목:{_fmt(anchors_ref['neck'])}"
            ))

        _MARKER_COLORS = {'eye': "#55aaff", 'nose': "#55ff88", 'mouth': "#ffaa33", 'neck': "#ff55cc"}

        def _draw_one(nx, ny, color):
            x, y = int(nx * dw), int(ny * dh)
            r = 9
            canvas.create_oval(x-r, y-r, x+r, y+r,
                                outline=color, width=2, tags="anchor_marker")
            canvas.create_line(x-r-4, y, x+r+4, y,
                                fill=color, width=2, tags="anchor_marker")
            canvas.create_line(x, y-r-4, x, y+r+4,
                                fill=color, width=2, tags="anchor_marker")

        def _draw_markers():
            canvas.delete("anchor_marker")
            for k, c in _MARKER_COLORS.items():
                if anchors_ref.get(k):
                    _draw_one(*anchors_ref[k], c)

        _draw_markers()
        _update_status()

        def _on_click(event):
            nx = max(0.0, min(1.0, event.x / dw))
            ny = max(0.0, min(1.0, event.y / dh))
            anchors_ref[mode_var.get()] = (nx, ny)
            _draw_markers()
            _update_status()

        canvas.bind("<Button-1>", _on_click)

        def _ok():
            self._face_img_side_anchors = dict(anchors_ref) if any(anchors_ref.values()) else None
            _a = self._face_img_side_anchors
            if _a:
                set_pts = [k for k in ('eye', 'nose', 'mouth', 'neck') if _a.get(k)]
                _labels = {'eye': '눈', 'nose': '코', 'mouth': '입', 'neck': '목'}
                self._face_side_pivot_btn.config(
                    text="⊕ 앵커 [" + "/".join(_labels[k] for k in set_pts) + "]")
            else:
                self._face_side_pivot_btn.config(text="⊕ 앵커 설정 (옆)")
            self._det_cache = None
            self._refresh_frame()
            top.destroy()

        def _reset():
            for k in anchors_ref:
                anchors_ref[k] = None
            canvas.delete("anchor_marker")
            _update_status()

        btn_row = tk.Frame(top)
        btn_row.pack(pady=(4, 8))
        tk.Button(btn_row, text="리셋", width=7, command=_reset).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_row, text="취소", width=7, command=top.destroy).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_row, text="확인", width=7, command=_ok).pack(side=tk.LEFT, padx=4)

    def _toggle_face_img_side(self):
        if self._face_img_side is not None:
            self._face_img_side = None
            self._face_img_side_pts = None
            self._face_img_side_ref_h = None
            self._face_img_side_anchors = None
            self._face_img_side_kps_n = None
            self._face_img_side_lbl.config(text="미선택")
            self._face_img_side_btn.config(text="🖼  옆모습 이미지 로드")
            self._face_side_pivot_btn.config(state=tk.DISABLED, text="⊕ 앵커 설정 (옆)")
            self._det_cache = None
            self._refresh_frame()
        else:
            self._load_face_img_side()

    def _load_face_img_side(self):
        path = filedialog.askopenfilename(
            parent=self.win,
            title="옆모습 이미지 선택",
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

        self._face_img_side = img.copy()
        self._face_img_side_pts = src_pts
        self._face_img_side_ref_h = self._auto_face_pivot(img, face_res)[1]
        self._face_img_side_anchors = None  # 새 이미지 로드 시 앵커 초기화
        # 앵커 피커용 kps_n 저장 (오른눈, 왼눈, 코, 입중심, 목시작)
        if face_res.face_landmarks:
            lf = face_res.face_landmarks[0]
            _mouth_x = (lf[61].x + lf[291].x) / 2
            _mouth_y = (lf[61].y + lf[291].y) / 2
            if hasattr(lf, 'bbox'):
                _neck_x = (lf.bbox[0] + lf.bbox[2]) / 2 / w
                _neck_y = min(1.0, lf.bbox[3] / h + 0.07)
            else:
                _neck_x = lf[152].x
                _neck_y = min(1.0, lf[152].y + 0.07)
            self._face_img_side_kps_n = [(lf[33].x, lf[33].y),
                                         (lf[263].x, lf[263].y),
                                         (lf[4].x, lf[4].y),
                                         (_mouth_x, _mouth_y),
                                         (_neck_x, _neck_y)]
        else:
            self._face_img_side_kps_n = None
        self._det_cache = None
        self._face_img_side_lbl.config(text=os.path.basename(path) + mode_text)
        self._face_img_side_btn.config(text="× 옆모습 제거")
        self._face_side_pivot_btn.config(state=tk.NORMAL, text="⊕ 앵커 설정 (옆)")
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
        if self._hand_det:
            self._hand_det.close()
            self._hand_det = None
        if self._pose_det:
            self._pose_det.close()
            self._pose_det = None
        self.win.destroy()
