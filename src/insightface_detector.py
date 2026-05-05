"""insightface_detector.py — InsightFace RetinaFace 기반 얼굴 감지 싱글턴

MediaPipe FaceLandmarker 대체:
  - 정면/측면 모두 강인한 RetinaFace (buffalo_sc 모델, ~6MB)
  - MediaPipe face_landmarks 포맷 호환 mock 반환
  - 5개 키포인트: 오른눈, 왼눈, 코, 오른입꼬리, 왼입꼬리

InsightFace kps[5×2] → MediaPipe 인덱스 매핑:
  kps[0] (오른눈)     → idx 33, 133, 473
  kps[1] (왼눈)       → idx 263, 362, 468
  kps[2] (코)         → idx 4, 168
  kps[3] (오른입꼬리) → idx 61
  kps[4] (왼입꼬리)   → idx 291, 13, 14
  나머지              → bbox 중심점 (mosaic bbox 계산용)
"""

import os
import threading
import numpy as np

_lock = threading.Lock()
_app = None

# MediaPipe FaceLandmarker (MAR 보정 전용)
_mp_lock = threading.Lock()
_mp_face_det = None
_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'face_landmarker.task')


def _get_mp_face():
    """MediaPipe FaceLandmarker 싱글턴 반환 (최초 호출 시 초기화)."""
    global _mp_face_det
    if _mp_face_det is None:
        with _mp_lock:
            if _mp_face_det is None:
                try:
                    import mediapipe as mp
                    from mediapipe.tasks import python as mp_python
                    from mediapipe.tasks.python import vision as mp_vision
                    opts = mp_vision.FaceLandmarkerOptions(
                        base_options=mp_python.BaseOptions(model_asset_path=_MODEL_PATH),
                        running_mode=mp_vision.RunningMode.IMAGE,
                        num_faces=4,
                    )
                    _mp_face_det = mp_vision.FaceLandmarker.create_from_options(opts)
                    print("[MediaPipe] FaceLandmarker (MAR용) 로드 완료")
                except Exception as e:
                    print(f"[MediaPipe init error] {e}")
                    _mp_face_det = False  # 실패 표시 (재시도 방지)
    return _mp_face_det if _mp_face_det else None


def _get_mp_lm(frame_bgr):
    """BGR 프레임 → MediaPipe face_landmarks 리스트 반환 (실패 시 빈 리스트)."""
    try:
        import cv2
        import mediapipe as mp
        detector = _get_mp_face()
        if detector is None:
            return []
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_img)
        return result.face_landmarks  # 각 원소: 랜드마크 478개 리스트
    except Exception:
        return []


class _IFLandmark:
    """정규화된 x, y 좌표를 가진 mock 랜드마크."""
    __slots__ = ('x', 'y')

    def __init__(self, x=0.0, y=0.0):
        self.x = float(x)
        self.y = float(y)


class _IFFaceLandmarkList(list):
    """MediaPipe face_landmarks[i] 포맷 호환 mock 리스트 (478 원소).

    bbox 속성: (x1, y1, x2, y2) 픽셀 좌표 (입력 프레임 기준).
    hasattr(_lf, 'bbox')로 InsightFace 결과 여부 확인 가능.
    """

    def __init__(self, kps, bbox, w, h):
        # 기본값: bbox 중심 (모든 478 원소 — mosaic 순회 대응)
        cx = (bbox[0] + bbox[2]) / 2.0 / w
        cy = (bbox[1] + bbox[3]) / 2.0 / h
        lms = [_IFLandmark(cx, cy) for _ in range(478)]

        # kps를 정규화 좌표로 변환 (픽셀 → 0..1)
        kps_n = [(float(kp[0]) / w, float(kp[1]) / h) for kp in kps]

        # kps[0]: 오른눈 → 33, 133, 473
        for idx in (33, 133, 473):
            lms[idx] = _IFLandmark(*kps_n[0])
        # kps[1]: 왼눈 → 263, 362, 468
        for idx in (263, 362, 468):
            lms[idx] = _IFLandmark(*kps_n[1])
        # kps[2]: 코 → 4, 168
        for idx in (4, 168):
            lms[idx] = _IFLandmark(*kps_n[2])
        # kps[3]: 오른입꼬리 → 61
        lms[61] = _IFLandmark(*kps_n[3])
        # kps[4]: 왼입꼬리 → 291, 13, 14
        for idx in (291, 13, 14):
            lms[idx] = _IFLandmark(*kps_n[4])

        super().__init__(lms)
        # bbox는 원본 픽셀 좌표 (mosaic, 드로잉에 직접 사용)
        self.bbox = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))


class InsightFaceResult:
    """MediaPipe FaceLandmarker 결과 포맷 호환 mock."""
    __slots__ = ('face_landmarks',)

    def __init__(self, face_landmarks):
        self.face_landmarks = face_landmarks


def _get_app():
    global _app
    if _app is None:
        with _lock:
            if _app is None:
                try:
                    from insightface.app import FaceAnalysis
                    app = FaceAnalysis(
                        name='buffalo_sc',
                        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                    )
                    app.prepare(ctx_id=0, det_size=(640, 640))
                    _app = app
                    print("[InsightFace] buffalo_sc 모델 로드 완료")
                except Exception as e:
                    print(f"[InsightFace init error] {e}")
                    raise
    return _app


def detect(frame_bgr, w=None, h=None, min_conf=0.3):
    """BGR 프레임에서 얼굴 감지 → InsightFaceResult 반환.

    frame_bgr: BGR numpy array (원본 프레임 권장, InsightFace가 내부 리사이즈)
    w, h: 프레임 크기 (None이면 frame_bgr.shape[:2] 사용)
    min_conf: 최소 감지 신뢰도 (det_score 기준)
    """
    if w is None or h is None:
        h, w = frame_bgr.shape[:2]
    try:
        app = _get_app()
        faces = app.get(frame_bgr)
        landmarks = []
        for face in faces:
            if face.kps is None or face.bbox is None:
                continue
            if hasattr(face, 'det_score') and face.det_score < min_conf:
                continue
            lm_list = _IFFaceLandmarkList(face.kps, face.bbox, w, h)
            landmarks.append(lm_list)

        # MAR 보정: MediaPipe로 입 위(13)/아래(14) 좌표 덮어쓰기
        if landmarks:
            try:
                mp_lms = _get_mp_lm(frame_bgr)
                for if_lm, mp_lm in zip(landmarks, mp_lms):
                    if_lm[13] = _IFLandmark(mp_lm[13].x, mp_lm[13].y)
                    if_lm[14] = _IFLandmark(mp_lm[14].x, mp_lm[14].y)
            except Exception:
                pass  # MediaPipe 실패 시 MAR=0 그대로

        return InsightFaceResult(landmarks)
    except Exception as e:
        print(f"[InsightFace detect error] {e}")
        return InsightFaceResult([])
