"""
tracker.py — MediaPipe Tasks API 기반 얼굴 + 손 추적기 (mediapipe 0.10.x+)

얼굴: FaceLandmarker 478개 랜드마크 (홍채 포함)
손:   HandLandmarker 21개 랜드마크 × 최대 2손
"""

import os
import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Callable

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode

# 모델 파일 경로
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACE_MODEL  = os.path.join(_BASE, "models", "face_landmarker.task")
HAND_MODEL  = os.path.join(_BASE, "models", "hand_landmarker.task")

# ──────────────────────────────────────────────────────────────────────────
# 데이터 구조체
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class Point2D:
    x: float = 0.0
    y: float = 0.0
    confidence: float = 0.0

    def valid(self, threshold: float = 0.05) -> bool:
        return self.confidence >= threshold


# FaceLandmarker 주요 인덱스 (478-point with iris)
class FaceIdx:
    RIGHT_EYE_OUTER  = 33
    RIGHT_EYE_INNER  = 133
    RIGHT_IRIS       = 473
    LEFT_EYE_INNER   = 362
    LEFT_EYE_OUTER   = 263
    LEFT_IRIS        = 468
    NOSE_BRIDGE_TOP  = 168
    NOSE_TIP         = 4
    MOUTH_RIGHT      = 61
    MOUTH_UPPER      = 13
    MOUTH_LEFT       = 291
    MOUTH_LOWER      = 14
    TOTAL            = 478

HAND_LANDMARK_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip",  "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp","middle_pip","middle_dip","middle_tip",
    "ring_mcp",  "ring_pip",  "ring_dip",  "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]


@dataclass
class FaceLandmarks:
    detected: bool = False

    right_eye_outer:  Point2D = field(default_factory=Point2D)
    right_eye_inner:  Point2D = field(default_factory=Point2D)
    right_iris:       Point2D = field(default_factory=Point2D)
    left_eye_inner:   Point2D = field(default_factory=Point2D)
    left_eye_outer:   Point2D = field(default_factory=Point2D)
    left_iris:        Point2D = field(default_factory=Point2D)
    nose_bridge_top:  Point2D = field(default_factory=Point2D)
    nose_tip:         Point2D = field(default_factory=Point2D)
    mouth_right:      Point2D = field(default_factory=Point2D)
    mouth_upper:      Point2D = field(default_factory=Point2D)
    mouth_left:       Point2D = field(default_factory=Point2D)
    mouth_lower:      Point2D = field(default_factory=Point2D)

    all: List[Point2D] = field(default_factory=list)


@dataclass
class HandLandmarks:
    detected: bool = False
    side: str = ""
    landmarks: List[Point2D] = field(default_factory=list)

    @property
    def wrist(self):      return self.landmarks[0]  if self.landmarks else Point2D()
    @property
    def thumb_tip(self):  return self.landmarks[4]  if len(self.landmarks) > 4  else Point2D()
    @property
    def index_tip(self):  return self.landmarks[8]  if len(self.landmarks) > 8  else Point2D()
    @property
    def middle_tip(self): return self.landmarks[12] if len(self.landmarks) > 12 else Point2D()
    @property
    def ring_tip(self):   return self.landmarks[16] if len(self.landmarks) > 16 else Point2D()
    @property
    def pinky_tip(self):  return self.landmarks[20] if len(self.landmarks) > 20 else Point2D()


@dataclass
class FrameData:
    index:      int   = 0
    timestamp:  float = 0.0
    face:       FaceLandmarks = field(default_factory=FaceLandmarks)
    left_hand:  HandLandmarks = field(default_factory=HandLandmarks)
    right_hand: HandLandmarks = field(default_factory=HandLandmarks)


@dataclass
class VideoInfo:
    width:        int   = 0
    height:       int   = 0
    fps:          float = 30.0
    total_frames: int   = 0


# ──────────────────────────────────────────────────────────────────────────
# 내부 유틸
# ──────────────────────────────────────────────────────────────────────────

def _lm_to_point(lm, w: int, h: int, conf: float = 1.0) -> Point2D:
    return Point2D(x=lm.x * w, y=lm.y * h, confidence=conf)


def _extract_face(face_result, w: int, h: int) -> FaceLandmarks:
    f = FaceLandmarks()
    if not face_result.face_landmarks:
        return f
    lms = face_result.face_landmarks[0]  # 첫 번째 얼굴

    def pt(idx): return _lm_to_point(lms[idx], w, h)

    f.all = [_lm_to_point(lm, w, h) for lm in lms]
    f.right_eye_outer = pt(FaceIdx.RIGHT_EYE_OUTER)
    f.right_eye_inner = pt(FaceIdx.RIGHT_EYE_INNER)
    f.left_eye_inner  = pt(FaceIdx.LEFT_EYE_INNER)
    f.left_eye_outer  = pt(FaceIdx.LEFT_EYE_OUTER)
    f.nose_bridge_top = pt(FaceIdx.NOSE_BRIDGE_TOP)
    f.nose_tip        = pt(FaceIdx.NOSE_TIP)
    f.mouth_right     = pt(FaceIdx.MOUTH_RIGHT)
    f.mouth_upper     = pt(FaceIdx.MOUTH_UPPER)
    f.mouth_left      = pt(FaceIdx.MOUTH_LEFT)
    f.mouth_lower     = pt(FaceIdx.MOUTH_LOWER)
    if len(lms) > FaceIdx.LEFT_IRIS:
        f.right_iris = pt(FaceIdx.RIGHT_IRIS)
        f.left_iris  = pt(FaceIdx.LEFT_IRIS)
    f.detected = True
    return f


def _extract_hand(hand_lms, handedness_list, w: int, h: int) -> HandLandmarks:
    h_data = HandLandmarks()
    conf = handedness_list[0].score if handedness_list else 1.0
    h_data.side = handedness_list[0].category_name.lower() if handedness_list else "unknown"
    h_data.landmarks = [_lm_to_point(lm, w, h, conf=conf) for lm in hand_lms]
    h_data.detected = True
    return h_data


# ──────────────────────────────────────────────────────────────────────────
# Tracker (영상 파일 처리용)
# ──────────────────────────────────────────────────────────────────────────

class Tracker:
    """영상 파일 전체를 처리. camera_panel.py 는 직접 Tasks API 를 사용."""

    def __init__(self,
                 min_face_confidence: float = 0.5,
                 min_hand_confidence: float = 0.5):
        self._face_conf = min_face_confidence
        self._hand_conf = min_hand_confidence

    # camera_panel 에서 호출하는 메서드 (구 API 호환 래퍼)
    def _extract_face(self, face_result, w: int, h: int) -> FaceLandmarks:
        return _extract_face(face_result, w, h)

    def _extract_hand(self, hand_lms, handedness, w: int, h: int) -> HandLandmarks:
        return _extract_hand(hand_lms, handedness, w, h)

    def process_video(self,
                      video_path: str,
                      show_preview: bool = True,
                      callback: Optional[Callable[[int, int], None]] = None
                      ) -> tuple[List[FrameData], VideoInfo]:

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"영상 파일을 열 수 없습니다: {video_path}")

        info = VideoInfo(
            width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0,
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
        print(f"[Tracker] 영상: {info.width}×{info.height} @ {info.fps:.2f} fps, "
              f"총 {info.total_frames} 프레임")

        face_opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=self._face_conf,
            min_face_presence_confidence=self._face_conf,
            min_tracking_confidence=0.5,
        )
        hand_opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL),
            running_mode=RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=self._hand_conf,
            min_hand_presence_confidence=self._hand_conf,
            min_tracking_confidence=0.5,
        )

        from mediapipe.tasks.python.vision import drawing_utils as du
        from mediapipe.tasks.python.vision import drawing_styles as ds
        from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections
        from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections

        frames: List[FrameData] = []

        with mp_vision.FaceLandmarker.create_from_options(face_opts) as face_det, \
             mp_vision.HandLandmarker.create_from_options(hand_opts) as hand_det:

            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                h_px, w_px = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                face_res = face_det.detect(mp_img)
                hand_res = hand_det.detect(mp_img)

                fd = FrameData(index=idx, timestamp=idx / info.fps)
                fd.face = _extract_face(face_res, w_px, h_px)

                if hand_res.hand_landmarks:
                    for hlms, hedness in zip(hand_res.hand_landmarks, hand_res.handedness):
                        hd = _extract_hand(hlms, hedness, w_px, h_px)
                        if hd.side == "left":
                            fd.left_hand = hd
                        else:
                            fd.right_hand = hd

                frames.append(fd)

                if show_preview:
                    preview = frame.copy()
                    if face_res.face_landmarks:
                        du.draw_landmarks(
                            preview,
                            face_res.face_landmarks[0],
                            FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=ds.get_default_face_mesh_contours_style(),
                        )
                    if hand_res.hand_landmarks:
                        for hlms in hand_res.hand_landmarks:
                            du.draw_landmarks(
                                preview, hlms,
                                HandLandmarksConnections.HAND_CONNECTIONS,
                                landmark_drawing_spec=ds.get_default_hand_landmarks_style(),
                                connection_drawing_spec=ds.get_default_hand_connections_style(),
                            )
                    cv2.imshow("PoseTracker Preview  (Q: 종료)", preview)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        show_preview = False
                        cv2.destroyAllWindows()

                if callback:
                    callback(idx + 1, info.total_frames)
                elif idx % 60 == 0:
                    pct = int(100 * idx / max(info.total_frames, 1))
                    bar = '#' * (pct // 2) + '-' * (50 - pct // 2)
                    print(f"\r[{bar}] {pct:3d}% ({idx}/{info.total_frames})",
                          end='', flush=True)
                idx += 1

        print(f"\n[Tracker] 완료: {len(frames)} 프레임 처리됨")
        cap.release()
        if show_preview:
            cv2.destroyAllWindows()

        return frames, info
