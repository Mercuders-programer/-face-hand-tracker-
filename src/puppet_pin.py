"""puppet_pin.py — 2-세그먼트 Piecewise Affine 팔 이미지 변형

어깨(Shoulder)→팔꿈치(Elbow) / 팔꿈치(Elbow)→손목(Wrist) 구간을
독립적인 Affine 변환으로 처리해 AE Puppet Pin Tool과 유사한 효과를 냄.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class PuppetPins:
    """이미지 좌표계에서의 핀 위치 (픽셀 단위)."""
    img_shldr: Tuple[float, float]
    img_elbow: Tuple[float, float]
    img_wrist: Tuple[float, float]

    def arrays(self):
        return (
            np.array(self.img_shldr, dtype=np.float64),
            np.array(self.img_elbow, dtype=np.float64),
            np.array(self.img_wrist, dtype=np.float64),
        )

    def is_valid(self, min_dist: float = 8.0) -> bool:
        s, e, w = self.arrays()
        return (np.linalg.norm(e - s) >= min_dist and
                np.linalg.norm(w - e) >= min_dist)


@dataclass
class SegmentCache:
    """이미지/핀 조합이 바뀔 때만 재계산되는 사전-마스킹 이미지."""
    img_upper: np.ndarray   # BGRA, 팔꿈치 기준 Shoulder 쪽만 불투명
    img_lower: np.ndarray   # BGRA, 팔꿈치 기준 Wrist 쪽만 불투명
    pins: PuppetPins


def build_segment_cache(arm_img: np.ndarray, pins: PuppetPins) -> SegmentCache:
    """이미지와 핀이 바뀔 때만 호출. 분할 마스크 사전 계산.

    팔 전체 방향(Shoulder→Wrist)의 수직선(Elbow 통과)으로 이미지를 이분.
    upper = Shoulder 쪽, lower = Wrist 쪽.
    """
    h, w = arm_img.shape[:2]
    s, e, wrist = pins.arrays()

    # Shoulder→Wrist 방향 단위 벡터
    arm_dir = _norm(wrist - s)

    # 각 픽셀의 Elbow 기준 팔 방향 투영값
    yy, xx = np.mgrid[0:h, 0:w]
    d = (xx - e[0]) * arm_dir[0] + (yy - e[1]) * arm_dir[1]

    upper_mask = (d <= 0)  # Shoulder 쪽 (d<=0)

    img_up = arm_img.copy()
    img_up[~upper_mask, 3] = 0   # lower 부분 투명화

    img_lo = arm_img.copy()
    img_lo[upper_mask, 3] = 0    # upper 부분 투명화

    return SegmentCache(img_upper=img_up, img_lower=img_lo, pins=pins)


def apply_puppet_warp(
    cache: SegmentCache,
    vid_shldr: Tuple[float, float],
    vid_elbow: Tuple[float, float],
    vid_wrist: Tuple[float, float],
    W: int, H: int,
) -> np.ndarray:
    """프레임마다 호출. 2-세그먼트 warp 후 alpha-over 합성.

    Returns:
        BGRA numpy array (H×W×4)
    """
    s, e, w = cache.pins.arrays()
    vs = np.float64(vid_shldr)
    ve = np.float64(vid_elbow)
    vw = np.float64(vid_wrist)

    M_up = _seg_affine(s, e, vs, ve)
    M_lo = _seg_affine(e, w, ve, vw)

    result = np.zeros((H, W, 4), dtype=np.uint8)
    for M, seg in ((M_up, cache.img_upper), (M_lo, cache.img_lower)):
        if M is None:
            continue
        warped = cv2.warpAffine(
            seg, M, (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        a = warped[:, :, 3:4].astype(np.float32) / 255.0
        result[:, :, :3] = np.clip(
            warped[:, :, :3].astype(np.float32) * a
            + result[:, :, :3].astype(np.float32) * (1.0 - a),
            0, 255,
        ).astype(np.uint8)
        result[:, :, 3:4] = np.maximum(warped[:, :, 3:4], result[:, :, 3:4])

    return result


def pins_degenerate(pins: PuppetPins, min_dist: float = 8.0) -> bool:
    """핀 간격이 너무 좁으면 True (유효하지 않은 핀)."""
    return not pins.is_valid(min_dist)


# ── 내부 헬퍼 ──────────────────────────────────────────────────────────────

def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _rot90(v: np.ndarray) -> np.ndarray:
    return np.array([-v[1], v[0]], dtype=np.float64)


def _seg_affine(
    p0: np.ndarray, p1: np.ndarray,
    q0: np.ndarray, q1: np.ndarray,
) -> Optional[np.ndarray]:
    """3점(src→dst) Affine 행렬 계산. 수직 방향도 스케일 반영."""
    d_i = p1 - p0
    d_v = q1 - q0
    src = np.float32([p0, p1, p0 + _rot90(d_i)])
    dst = np.float32([q0, q1, q0 + _rot90(d_v)])
    try:
        return cv2.getAffineTransform(src, dst)
    except Exception:
        return None
