"""
face_pipeline.py — OpenCV YuNet + FaceLandmarker 2단계 얼굴 감지 파이프라인

배경:
  MediaPipe 내장 BlazeFace 검출기는 정면 편향(yaw ±30°)이라 측면 얼굴(yaw≥45°)에서 실패함.
  OpenCV YuNet(FaceDetectorYN)은 yaw ±70°까지 안정적으로 감지 가능.

파이프라인:
  프레임 → YuNet 검출(bbox) → 각 얼굴 크롭(+30% 여백) 400×400
          → FaceLandmarker(랜드마크만) → 크롭 좌표 → 원본 좌표 역변환

Fallback:
  YuNet 모델 파일 없으면 → 기존 FaceLandmarker 단독 동작 (conf=0.5)
"""

import os
import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACE_MODEL  = os.path.join(_BASE, "models", "face_landmarker.task")
YUNET_MODEL = os.path.join(_BASE, "models", "face_detection_yunet_2023mar.onnx")


# ── Duck-typing 호환 객체 ──────────────────────────────────────────────────

class _Lm:
    """NormalizedLandmark 호환 (x, y 정규화 좌표)"""
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


class _FaceResult:
    """FaceLandmarkerResult 호환 객체 (face_landmarks 리스트만 노출)"""
    __slots__ = ("face_landmarks",)

    def __init__(self, face_landmarks):
        self.face_landmarks = face_landmarks  # list[list[_Lm]]


# ── RobustFaceLandmarker ──────────────────────────────────────────────────

class RobustFaceLandmarker:
    """
    OpenCV YuNet 검출 + FaceLandmarker 랜드마크 2단계 파이프라인.
    YuNet 모델 없으면 기존 FaceLandmarker 단독 동작으로 자동 폴백.
    """

    # 크롭 이미지 크기 (FaceLandmarker 입력)
    CROP_SIZE = 400
    # 크롭 여백 비율 (bbox 기준 상하좌우 확장)
    PAD_RATIO = 0.30

    def __init__(self, face_model_path: str = FACE_MODEL,
                 num_faces: int = 4, fl_conf: float = 0.05,
                 yunet_model_path: str = YUNET_MODEL):
        """
        Args:
            face_model_path:  face_landmarker.task 경로
            num_faces:        최대 감지 얼굴 수
            fl_conf:          FaceLandmarker 크롭용 confidence (낮게 설정)
            yunet_model_path: face_detection_yunet_2023mar.onnx 경로
        """
        self._num_faces = num_faces
        self._fl_conf = fl_conf

        # YuNet 초기화
        self._yunet = None
        if os.path.exists(yunet_model_path):
            try:
                self._yunet = cv2.FaceDetectorYN.create(
                    model=yunet_model_path,
                    config="",
                    input_size=(320, 320),
                    score_threshold=0.6,
                    nms_threshold=0.3,
                    top_k=num_faces,
                    backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
                    target_id=cv2.dnn.DNN_TARGET_CPU,
                )
                print("[FacePipeline] OpenCV YuNet 초기화 완료")
            except Exception as e:
                print(f"[FacePipeline] YuNet 초기화 실패, 폴백 모드: {e}")
                self._yunet = None
        else:
            print(f"[FacePipeline] YuNet 모델 없음({yunet_model_path}), 폴백 모드")

        # FaceLandmarker 초기화
        _conf = fl_conf if self._yunet is not None else 0.5

        face_opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=face_model_path),
            running_mode=RunningMode.IMAGE,
            num_faces=num_faces,
            min_face_detection_confidence=_conf,
            min_face_presence_confidence=_conf,
            min_tracking_confidence=0.5,
        )
        self._fl = mp_vision.FaceLandmarker.create_from_options(face_opts)

    # ── 공개 API ──────────────────────────────────────────────────────────

    def detect(self, mp_image: mp.Image) -> _FaceResult:
        """
        mp.Image를 입력받아 FaceLandmarkerResult 호환 객체를 반환.
        YuNet 사용 가능 시 → 2단계 파이프라인
        사용 불가 시 → FaceLandmarker 단독 (폴백)
        """
        if self._yunet is not None:
            return self._detect_robust(mp_image)
        else:
            return self._detect_fallback(mp_image)

    def close(self):
        if self._fl is not None:
            self._fl.close()
            self._fl = None

    # ── context manager ────────────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # ── 내부 구현 ──────────────────────────────────────────────────────────

    def _detect_robust(self, mp_image: mp.Image) -> _FaceResult:
        """2단계 파이프라인: YuNet bbox → 크롭 → FaceLandmarker"""
        # mp.Image → numpy RGB
        rgb = mp_image.numpy_view()          # H×W×3, RGB
        frame_h, frame_w = rgb.shape[:2]

        # 1단계: YuNet으로 bbox 검출 (BGR 필요)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self._yunet.setInputSize((frame_w, frame_h))
        _, faces = self._yunet.detect(bgr)   # faces: N×15 or None

        if faces is None or len(faces) == 0:
            return _FaceResult(face_landmarks=[])

        all_landmarks: list = []

        for face_row in faces[:self._num_faces]:
            # YuNet 출력: [x, y, w, h, lm0x, lm0y, ... score]
            x, y, w, h = int(face_row[0]), int(face_row[1]), int(face_row[2]), int(face_row[3])
            x2, y2 = x + w, y + h

            # 2단계: 크롭 + 여백
            pad_x = int(w * self.PAD_RATIO)
            pad_y = int(h * self.PAD_RATIO)

            cx1 = max(0, x - pad_x)
            cy1 = max(0, y - pad_y)
            cx2 = min(frame_w, x2 + pad_x)
            cy2 = min(frame_h, y2 + pad_y)

            crop_w = cx2 - cx1
            crop_h = cy2 - cy1
            if crop_w < 4 or crop_h < 4:
                continue

            crop_rgb = rgb[cy1:cy2, cx1:cx2]

            # 400×400 리사이즈
            resized = cv2.resize(crop_rgb, (self.CROP_SIZE, self.CROP_SIZE),
                                 interpolation=cv2.INTER_LINEAR)
            mp_crop = mp.Image(image_format=mp.ImageFormat.SRGB, data=resized)

            # FaceLandmarker 추론
            try:
                crop_res = self._fl.detect(mp_crop)
            except Exception as e:
                print(f"[FacePipeline] FaceLandmarker 크롭 추론 실패: {e}")
                continue

            if not crop_res.face_landmarks:
                continue

            # 크롭 내 좌표 → 원본 프레임 좌표 역변환
            # lm.x / lm.y 는 크롭(CROP_SIZE×CROP_SIZE) 기준 정규화 좌표
            # → 크롭 픽셀 = lm.x * crop_w (원래 크롭 크기 기준)
            # → 원본 픽셀 = cx1 + lm.x * crop_w
            # → 원본 정규화 = 원본 픽셀 / frame_w
            crop_lms = crop_res.face_landmarks[0]
            orig_lms: list = []
            for lm in crop_lms:
                orig_x = (cx1 + lm.x * crop_w) / frame_w
                orig_y = (cy1 + lm.y * crop_h) / frame_h
                orig_lms.append(_Lm(x=orig_x, y=orig_y, z=lm.z))

            all_landmarks.append(orig_lms)

        return _FaceResult(face_landmarks=all_landmarks)

    def _detect_fallback(self, mp_image: mp.Image) -> _FaceResult:
        """폴백: FaceLandmarker 단독 (YuNet 없을 시)"""
        result = self._fl.detect(mp_image)
        all_lms: list = []
        for face_lms in result.face_landmarks:
            wrapped = [_Lm(x=lm.x, y=lm.y, z=lm.z) for lm in face_lms]
            all_lms.append(wrapped)
        return _FaceResult(face_landmarks=all_lms)
