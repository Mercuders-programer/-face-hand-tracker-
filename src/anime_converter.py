"""
anime_converter.py — 사람 영역 애니메이션 스타일 변환

기능:
  - build_person_mask()    : 포즈/얼굴/손 랜드마크 → 사람 마스크
  - apply_opencv_anime()   : OpenCV bilateral 셀 셰이딩 필터
  - AnimeGANConverter      : AnimeGAN ONNX 추론기
  - apply_anime_to_person(): 마스크 + 스타일 변환 + 배경 합성
"""

import cv2
import numpy as np


# ── 사람 마스크 생성 ────────────────────────────────────────────────────────

def build_person_mask(w, h, pose_res=None, face_res=None, hand_res=None,
                      dilate_px=50, blur_k=31):
    """
    포즈/얼굴/손 랜드마크로 사람 영역 마스크 생성.

    반환: uint8 ndarray (H×W) — 0=배경, 255=사람
    감지 결과 없으면 전체 화면 마스크 반환.
    """
    pts = []

    # 포즈 33점 (visibility > 0.2 인 점만)
    if pose_res is not None and pose_res.pose_landmarks:
        for pose in pose_res.pose_landmarks:
            for lm in pose:
                if getattr(lm, "visibility", 1.0) > 0.2:
                    pts.append([
                        int(np.clip(lm.x * w, 0, w - 1)),
                        int(np.clip(lm.y * h, 0, h - 1)),
                    ])

    # 얼굴 bbox (478점 → min/max bbox + padding)
    if face_res is not None and face_res.face_landmarks:
        for face in face_res.face_landmarks:
            xs = [int(np.clip(lm.x * w, 0, w - 1)) for lm in face]
            ys = [int(np.clip(lm.y * h, 0, h - 1)) for lm in face]
            if xs and ys:
                pad = 20
                x1, x2 = max(0, min(xs) - pad), min(w - 1, max(xs) + pad)
                y1, y2 = max(0, min(ys) - pad), min(h - 1, max(ys) + pad)
                pts.extend([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

    # 손 21점 × 2
    if hand_res is not None and hand_res.hand_landmarks:
        for hand in hand_res.hand_landmarks:
            for lm in hand:
                pts.append([
                    int(np.clip(lm.x * w, 0, w - 1)),
                    int(np.clip(lm.y * h, 0, h - 1)),
                ])

    # 감지 결과 없음 → 전체 화면 마스크
    if len(pts) < 3:
        return np.full((h, w), 255, dtype=np.uint8)

    mask = np.zeros((h, w), dtype=np.uint8)
    pts_arr = np.array(pts, dtype=np.int32)
    hull = cv2.convexHull(pts_arr)
    cv2.fillConvexPoly(mask, hull, 255)

    # 팽창 (경계 확장)
    if dilate_px > 0:
        k = dilate_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel)

    # 경계 소프트닝
    if blur_k > 0:
        bk = blur_k | 1  # 홀수 보장
        mask = cv2.GaussianBlur(mask, (bk, bk), bk // 3)

    return mask


# ── OpenCV 셀 셰이딩 필터 ──────────────────────────────────────────────────

def apply_opencv_anime(frame, scale=512):
    """
    bilateral filter × 7 + adaptive edge → 셀 애니메이션 스타일.
    처리 해상도: scale px (짧은 변 기준), 출력은 원본 해상도.
    """
    h, w = frame.shape[:2]
    sh = min(h, w)

    if sh > scale:
        sc = scale / sh
        sw, sh2 = int(w * sc), int(h * sc)
        small = cv2.resize(frame, (sw, sh2), interpolation=cv2.INTER_AREA)
    else:
        small = frame.copy()
        sw, sh2 = w, h

    # 표면 평활화
    color = small
    for _ in range(7):
        color = cv2.bilateralFilter(color, d=9, sigmaColor=75, sigmaSpace=75)

    # 엣지 추출
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray  = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=9, C=2,
    )
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # 합성
    anime_small = cv2.bitwise_and(color, edges_bgr)

    # 원본 해상도 복원
    if (sw, sh2) != (w, h):
        anime_small = cv2.resize(anime_small, (w, h), interpolation=cv2.INTER_LINEAR)

    return anime_small


# ── AnimeGAN ONNX 추론기 ───────────────────────────────────────────────────

class AnimeGANConverter:
    """
    AnimeGAN v2/v3 ONNX 모델 추론기.

    사용법:
        conv = AnimeGANConverter()
        conv.load("AnimeGANv3_Hayao.onnx")
        result = conv.convert(bgr_frame)
    """

    def __init__(self):
        self._session  = None
        self._in_name  = None
        self._out_name = None

    @property
    def loaded(self):
        return self._session is not None

    def load(self, model_path):
        """ONNX 모델 로드. GPU(CUDA) 우선, 없으면 CPU."""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime 미설치:\n"
                "  GPU: pip install onnxruntime-gpu\n"
                "  CPU: pip install onnxruntime"
            )

        avail     = ort.get_available_providers()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in avail
            else ["CPUExecutionProvider"]
        )

        self._session  = ort.InferenceSession(model_path, providers=providers)
        self._in_name  = self._session.get_inputs()[0].name
        self._out_name = self._session.get_outputs()[0].name
        return self

    def convert(self, frame, scale=512):
        """
        BGR 입력 → AnimeGAN 변환 → BGR 출력 (원본 해상도).
        scale: 추론 시 리사이즈 기준 (최대 변 px, 32배수로 맞춤).
        """
        if not self.loaded:
            raise RuntimeError("모델이 로드되지 않았습니다.")

        h, w = frame.shape[:2]

        # 32 배수로 리사이즈 (AnimeGAN 모델 요구사항)
        sc    = scale / max(h, w)
        new_w = max(32, int(w * sc) // 32 * 32)
        new_h = max(32, int(h * sc) // 32 * 32)

        small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # BGR → RGB, [-1, 1] 정규화
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb / 127.5) - 1.0
        inp = rgb[np.newaxis]  # (1, H, W, 3)

        out = self._session.run([self._out_name], {self._in_name: inp})[0]

        # [-1, 1] → [0, 255] BGR
        out    = np.clip((out[0] + 1.0) * 127.5, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

        # 원본 해상도 복원
        if (new_w, new_h) != (w, h):
            result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)

        return result


# ── 통합 파이프라인 ────────────────────────────────────────────────────────

def apply_anime_to_person(frame, pose_res, face_res, hand_res,
                           style="animegan", bg_mode="original",
                           converter=None, dilate_px=50, blur_k=31):
    """
    사람 마스크 생성 → 애니 필터 → 배경 합성.

    Parameters
    ----------
    frame     : BGR ndarray
    pose_res  : PoseLandmarker 결과 (None 가능)
    face_res  : FaceLandmarker 결과 (None 가능)
    hand_res  : HandLandmarker 결과 (None 가능)
    style     : "animegan" | "opencv"
    bg_mode   : "original" | "blur" | "solid"
    converter : AnimeGANConverter 인스턴스 (style="animegan" 시 필요)
    """
    h, w = frame.shape[:2]

    # 1. 사람 마스크 (0~255, 255=사람)
    mask = build_person_mask(w, h, pose_res, face_res, hand_res,
                             dilate_px=dilate_px, blur_k=blur_k)

    # 2. 애니 스타일 변환
    if style == "animegan" and converter is not None and converter.loaded:
        try:
            styled = converter.convert(frame)
        except Exception:
            styled = apply_opencv_anime(frame)
    else:
        styled = apply_opencv_anime(frame)

    # 3. 배경 생성
    if bg_mode == "blur":
        bg = cv2.GaussianBlur(frame, (51, 51), 0)
    elif bg_mode == "solid":
        bg = np.full_like(frame, 255)
    else:  # "original"
        bg = frame

    # 4. Alpha blend: 사람=styled, 배경=bg
    alpha  = mask.astype(np.float32) / 255.0
    alpha3 = alpha[:, :, np.newaxis]
    result = (styled.astype(np.float32) * alpha3 +
              bg.astype(np.float32) * (1.0 - alpha3))
    return result.astype(np.uint8)
