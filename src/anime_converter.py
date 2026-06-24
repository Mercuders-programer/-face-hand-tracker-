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

def _quantize_colors(img, k=8):
    """색상을 k색 팔레트로 양자화 (셀 셰이딩 효과)."""
    data = img.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _ret, labels, centers = cv2.kmeans(
        data, k, None, criteria, 1, cv2.KMEANS_PP_CENTERS,
    )
    centers = centers.astype(np.uint8)
    return centers[labels.flatten()].reshape(img.shape)


def apply_opencv_anime(frame, scale=512, k_colors=8):
    """
    bilateral 평활화 + 색상 양자화(k-means) + 어두운 엣지 오버레이 → 셀 셰이딩.
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

    # 표면 평활화 (디테일 보존)
    color = small
    for _ in range(5):
        color = cv2.bilateralFilter(color, d=9, sigmaColor=75, sigmaSpace=75)

    # 색상 양자화 → 평평한 셀 영역
    color = _quantize_colors(color, k=k_colors)

    # 엣지 추출 (검은 윤곽선)
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray  = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=9, C=2,
    )

    # 엣지를 검은 선으로 오버레이 (양자화 색은 그대로 유지)
    anime_small = color.copy()
    anime_small[edges == 0] = (0, 0, 0)

    # 원본 해상도 복원
    if (sw, sh2) != (w, h):
        anime_small = cv2.resize(anime_small, (w, h), interpolation=cv2.INTER_LINEAR)

    return anime_small


def apply_bold_cartoon(frame, base, strength=3):
    """
    base(예: White-box 결과) 위에 색 양자화 + 굵은 검정 외곽선을 얹어
    "확실한 만화" 룩으로 강화. strength 1~5 (클수록 색 적고 선 굵음).
    엣지는 구조가 또렷한 원본(frame)에서 추출.
    """
    strength = int(max(1, min(5, strength)))
    # 강도별 프리셋: (양자화 색 수, adaptiveThreshold C, 선 두께 erode 반경)
    k_map      = {1: 16, 2: 12, 3: 10, 4: 8, 5: 6}
    c_map      = {1: 9,  2: 7,  3: 6,  4: 5, 5: 4}
    dil_map    = {1: 0,  2: 0,  3: 1,  4: 1, 5: 2}
    k          = k_map[strength]
    edge_c     = c_map[strength]
    edge_dil   = dil_map[strength]

    quant = _quantize_colors(base, k=k)

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray  = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, blockSize=9, C=edge_c,
    )
    if edge_dil > 0:
        ksz = edge_dil * 2 + 1
        edges = cv2.erode(edges, np.ones((ksz, ksz), np.uint8))

    out = quant.copy()
    out[edges == 0] = (0, 0, 0)
    return out


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
        self._session    = None
        self._in_name    = None
        self._out_name   = None
        self._layout     = "nhwc"  # 입력 "nhwc" (1,H,W,3) | "nchw" (1,3,H,W)
        self._out_layout = "nhwc"  # 출력 레이아웃 (입력과 다를 수 있음 — White-box)
        self._fixed_hw   = None    # 모델이 고정 입력 크기면 (H, W), 아니면 None

    @property
    def loaded(self):
        return self._session is not None

    def load(self, model_path):
        """ONNX 모델 로드. GPU(CUDA) 우선, 없으면 CPU. 입력 레이아웃 자동 감지."""
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
        inp            = self._session.get_inputs()[0]
        out            = self._session.get_outputs()[0]
        self._in_name  = inp.name
        self._out_name = out.name

        # 입력 shape로 레이아웃/고정크기 감지.
        # NCHW: (N,3,H,W) → shape[1]==3 / NHWC: (N,H,W,3) → shape[3]==3
        shp = list(inp.shape)
        def _dim(v):
            return v if isinstance(v, int) and v > 0 else None
        if len(shp) == 4 and shp[1] == 3:
            self._layout = "nchw"
            self._fixed_hw = (_dim(shp[2]), _dim(shp[3]))
        else:  # 기본 NHWC
            self._layout = "nhwc"
            self._fixed_hw = (_dim(shp[1]), _dim(shp[2]))
        if not all(self._fixed_hw):
            self._fixed_hw = None

        # 출력 레이아웃은 입력과 다를 수 있음 (White-box: NCHW 입력 / NHWC 출력).
        oshp = list(out.shape)
        self._out_layout = "nchw" if (len(oshp) == 4 and oshp[1] == 3) else "nhwc"
        return self

    def convert(self, frame, scale=512):
        """
        BGR 입력 → AnimeGAN 변환 → BGR 출력 (원본 해상도).
        모델 입력이 고정 크기면 그 크기로, 동적이면 scale 기준 32배수로 리사이즈.
        NHWC/NCHW 레이아웃 모두 지원.
        """
        if not self.loaded:
            raise RuntimeError("모델이 로드되지 않았습니다.")

        h, w = frame.shape[:2]

        if self._fixed_hw:
            new_h, new_w = self._fixed_hw
        else:
            sc    = scale / max(h, w)
            new_w = max(32, int(w * sc) // 32 * 32)
            new_h = max(32, int(h * sc) // 32 * 32)

        small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # BGR → RGB, [-1, 1] 정규화
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb / 127.5) - 1.0

        if self._layout == "nchw":
            inp = np.transpose(rgb, (2, 0, 1))[np.newaxis]  # (1,3,H,W)
        else:
            inp = rgb[np.newaxis]                            # (1,H,W,3)

        out = self._session.run([self._out_name], {self._in_name: inp})[0][0]

        if self._out_layout == "nchw":
            out = np.transpose(out, (1, 2, 0))               # (H,W,3)

        # [-1, 1] → [0, 255] BGR
        out    = np.clip((out + 1.0) * 127.5, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

        # 원본 해상도 복원
        if result.shape[:2] != (h, w):
            result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)

        return result


# ── 통합 파이프라인 ────────────────────────────────────────────────────────

def apply_anime_to_person(frame, pose_res, face_res, hand_res,
                           style="animegan", bg_mode="original",
                           converter=None, dilate_px=50, blur_k=31,
                           range_mode="person", strength=3, sd_pipe=None):
    """
    사람 마스크 생성 → 애니 필터 → 배경 합성.

    Parameters
    ----------
    frame      : BGR ndarray
    pose_res   : PoseLandmarker 결과 (None 가능)
    face_res   : FaceLandmarker 결과 (None 가능)
    hand_res   : HandLandmarker 결과 (None 가능)
    style      : "whitebox" | "bold" | "sd" | "animegan" | "opencv"
    bg_mode    : "original" | "blur" | "solid"
    converter  : AnimeGANConverter 인스턴스 (ONNX 스타일 시 필요, 없으면 OpenCV 폴백)
    range_mode : "person" (사람 마스크) | "full" (전체 화면 변환)
    strength   : 1~5 — "bold" 강도, "sd" denoise 강도(내부에서 0~1로 매핑)
    sd_pipe    : SDCartoon 인스턴스 (style="sd" 시 필요)
    """
    h, w = frame.shape[:2]

    # 1. 애니 스타일 변환
    if style == "sd" and sd_pipe is not None:
        try:
            styled = sd_pipe.stylize(frame, strength=strength)
        except Exception:
            styled = apply_opencv_anime(frame)
    elif converter is not None and converter.loaded:
        try:
            styled = converter.convert(frame)
        except Exception:
            styled = apply_opencv_anime(frame)
    else:
        styled = apply_opencv_anime(frame)

    # 1b. "bold" 스타일이면 base 위에 양자화 + 굵은 외곽선 강화
    if style == "bold":
        styled = apply_bold_cartoon(frame, styled, strength=strength)

    # 2. 전체 화면 변환이면 마스크/합성 생략
    if range_mode == "full":
        return styled

    # 3. 사람 마스크 (0~255, 255=사람)
    mask = build_person_mask(w, h, pose_res, face_res, hand_res,
                             dilate_px=dilate_px, blur_k=blur_k)

    # 4. 배경 생성
    if bg_mode == "blur":
        bg = cv2.GaussianBlur(frame, (51, 51), 0)
    elif bg_mode == "solid":
        bg = np.full_like(frame, 255)
    else:  # "original"
        bg = frame

    # 5. Alpha blend: 사람=styled, 배경=bg
    alpha  = mask.astype(np.float32) / 255.0
    alpha3 = alpha[:, :, np.newaxis]
    result = (styled.astype(np.float32) * alpha3 +
              bg.astype(np.float32) * (1.0 - alpha3))
    return result.astype(np.uint8)
