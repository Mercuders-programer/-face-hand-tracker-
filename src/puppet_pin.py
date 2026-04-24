"""puppet_pin.py — MLS-Affine 역방향 매핑 팔 이미지 변형

Moving Least Squares (Affine variant) 역방향 매핑으로
AE Puppet Pin Tool과 유사한 연속 메시 변형을 구현.

이전 구현(2-세그먼트 독립 Affine)과 API 시그니처 동일 → 호출부 변경 불필요.
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
    """이미지/핀 조합이 바뀔 때만 재계산되는 사전 처리 이미지."""
    img_orig: np.ndarray              # BGRA 원본 (마스킹 없음)
    pins: PuppetPins
    img_upper: Optional[np.ndarray] = None  # 하위호환 (미사용)
    img_lower: Optional[np.ndarray] = None  # 하위호환 (미사용)


def build_segment_cache(arm_img: np.ndarray, pins: PuppetPins,
                        feather_px: int = 0) -> SegmentCache:
    """이미지와 핀이 바뀔 때만 호출. BGRA 원본 저장."""
    if arm_img.shape[2] == 3:
        arm_img = cv2.cvtColor(arm_img, cv2.COLOR_BGR2BGRA)
    return SegmentCache(img_orig=arm_img.copy(), pins=pins)


def apply_puppet_warp(
    cache: SegmentCache,
    vid_shldr: Tuple[float, float],
    vid_elbow: Tuple[float, float],
    vid_wrist: Tuple[float, float],
    W: int, H: int,
) -> np.ndarray:
    """프레임마다 호출. MLS-Affine 역방향 매핑 warp.

    Returns:
        BGRA numpy array (H×W×4)
    """
    src_pts = np.array([cache.pins.img_shldr,
                        cache.pins.img_elbow,
                        cache.pins.img_wrist], dtype=np.float64)
    dst_pts = np.array([vid_shldr, vid_elbow, vid_wrist], dtype=np.float64)

    if not _pts_valid(dst_pts):
        return np.zeros((H, W, 4), dtype=np.uint8)

    map_x, map_y = _mls_affine_map(src_pts, dst_pts, W, H, scale_down=4)
    warped = cv2.remap(cache.img_orig, map_x, map_y,
                       cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT,
                       borderValue=(0, 0, 0, 0))
    # 캔버스-공간 마스크로 팔 영역 외부 투명 처리
    mask = _arm_canvas_mask(dst_pts, W, H)
    warped[:, :, 3] = (warped[:, :, 3].astype(np.float32) * mask).astype(np.uint8)
    return warped


def pins_degenerate(pins: PuppetPins, min_dist: float = 8.0) -> bool:
    """핀 간격이 너무 좁으면 True (유효하지 않은 핀)."""
    return not pins.is_valid(min_dist)


# ── 내부 헬퍼 ──────────────────────────────────────────────────────────────

def _pts_valid(dst_pts: np.ndarray, min_dist: float = 8.0) -> bool:
    """목적지 핀들이 유효한지 검사 (핀 간격 최소 기준)."""
    return (np.linalg.norm(dst_pts[1] - dst_pts[0]) >= min_dist and
            np.linalg.norm(dst_pts[2] - dst_pts[1]) >= min_dist)


def _mls_affine_map(
    src_pts: np.ndarray,   # (3, 2) 소스 이미지 핀
    dst_pts: np.ndarray,   # (3, 2) 출력 캔버스 핀
    W: int, H: int,
    scale_down: int = 4,
    lam: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """MLS-Affine 역방향 매핑 맵 계산 (벡터화).

    출력 캔버스의 각 픽셀 v → 소스 이미지 픽셀 u 를 계산.
    성능을 위해 scale_down 해상도에서 계산 후 cv2.resize 업스케일.

    알고리즘:
        w_i = 1 / max(||v - q_i||^2, 1)
        q* = Σ(w_i·q_i)/Σw_i,   p* = Σ(w_i·p_i)/Σw_i
        q_hat_i = q_i - q*,      p_hat_i = p_i - p*
        M_num[a,b] = Σ w_i · p_hat_i[a] · q_hat_i[b]
        M_den[a,b] = Σ w_i · q_hat_i[a] · q_hat_i[b]  + λ·I
        M = M_num @ inv(M_den)
        u = M @ (v - q*) + p*

    Returns:
        map_x, map_y: (H, W) float32 — cv2.remap 역방향 맵
    """
    Hs = max(H // scale_down, 2)
    Ws = max(W // scale_down, 2)

    # 출력 캔버스 픽셀 그리드
    vx = np.linspace(0.0, float(W - 1), Ws)
    vy = np.linspace(0.0, float(H - 1), Hs)
    vxg, vyg = np.meshgrid(vx, vy)
    v = np.stack([vxg.ravel(), vyg.ravel()], axis=1)  # (N, 2)

    # 가중치: w_i = 1 / max(||v - q_i||^2, 1)
    diff = v[:, None, :] - dst_pts[None, :, :]        # (N, 3, 2)
    dist2 = np.einsum('nid,nid->ni', diff, diff)       # (N, 3)
    w = 1.0 / np.maximum(dist2, 1.0)                   # (N, 3)
    w_sum = w.sum(axis=1, keepdims=True)                # (N, 1)

    # 가중 중심점
    q_star = (w[:, :, None] * dst_pts[None, :, :]).sum(axis=1) / w_sum  # (N, 2)
    p_star = (w[:, :, None] * src_pts[None, :, :]).sum(axis=1) / w_sum  # (N, 2)

    # 중심점 기준 상대 좌표
    q_hat = dst_pts[None, :, :] - q_star[:, None, :]   # (N, 3, 2)
    p_hat = src_pts[None, :, :] - p_star[:, None, :]   # (N, 3, 2)

    # M_num[n,a,b] = Σ_i w[n,i] * p_hat[n,i,a] * q_hat[n,i,b]
    # M_den[n,a,b] = Σ_i w[n,i] * q_hat[n,i,a] * q_hat[n,i,b]  + λI
    M_num = np.einsum('ni,nia,nib->nab', w, p_hat, q_hat)  # (N, 2, 2)
    M_den = np.einsum('ni,nia,nib->nab', w, q_hat, q_hat)  # (N, 2, 2)
    M_den[:, 0, 0] += lam
    M_den[:, 1, 1] += lam

    # 2×2 역행렬 (해석적 공식, numpy.linalg 미사용)
    a = M_den[:, 0, 0];  b = M_den[:, 0, 1]
    c = M_den[:, 1, 0];  d = M_den[:, 1, 1]
    det = a * d - b * c
    det = np.where(np.abs(det) < 1e-12, 1e-12, det)
    inv_den = np.empty_like(M_den)
    inv_den[:, 0, 0] =  d / det
    inv_den[:, 0, 1] = -b / det
    inv_den[:, 1, 0] = -c / det
    inv_den[:, 1, 1] =  a / det

    # M = M_num @ inv_den,  u = M @ (v - q*) + p*
    M = np.einsum('nab,nbc->nac', M_num, inv_den)         # (N, 2, 2)
    v_rel = v - q_star                                      # (N, 2)
    u = np.einsum('nab,nb->na', M, v_rel) + p_star         # (N, 2)

    # 저해상도 맵 → 원본 해상도 업스케일
    map_x_s = u[:, 0].reshape(Hs, Ws).astype(np.float32)
    map_y_s = u[:, 1].reshape(Hs, Ws).astype(np.float32)
    map_x = cv2.resize(map_x_s, (W, H), interpolation=cv2.INTER_LINEAR)
    map_y = cv2.resize(map_y_s, (W, H), interpolation=cv2.INTER_LINEAR)
    return map_x, map_y


def _arm_canvas_mask(dst_pts: np.ndarray, W: int, H: int,
                     sigma_factor: float = 0.4) -> np.ndarray:
    """캔버스 공간에서 팔 스켈레톤까지의 Gaussian 거리 마스크 (H×W, float32 0~1).

    dst_pts: [[sx,sy],[ex,ey],[wx,wy]] — 캔버스 픽셀 좌표
    팔에서 멀수록 투명해져 MLS 외삽으로 인한 배경 번짐을 방지.
    """
    q = dst_pts.astype(np.float64)
    yy, xx = np.mgrid[0:H, 0:W]
    px = xx.astype(np.float64)
    py = yy.astype(np.float64)

    def _seg_dist2(p0, p1):
        d = p1 - p0
        L2 = float(d @ d)
        if L2 < 1e-6:
            return (px - p0[0])**2 + (py - p0[1])**2
        t = np.clip(((px - p0[0])*d[0] + (py - p0[1])*d[1]) / L2, 0.0, 1.0)
        return (px - p0[0] - t*d[0])**2 + (py - p0[1] - t*d[1])**2

    min_d2 = np.minimum(_seg_dist2(q[0], q[1]), _seg_dist2(q[1], q[2]))
    arm_len = np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[1])
    sigma = max(arm_len * sigma_factor, 30.0)   # 최소 30px
    return np.exp(-min_d2 / (2.0 * sigma**2)).astype(np.float32)


def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _rot90(v: np.ndarray) -> np.ndarray:
    return np.array([-v[1], v[0]], dtype=np.float64)
