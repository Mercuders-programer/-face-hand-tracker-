"""puppet_pin.py — Live2D 메시 방식 팔 이미지 변형

이미지 공간에 Strip Mesh를 사전 생성하고,
변형된 dst_verts → cv2.remap으로 메시 밖 픽셀을 완전 차단.

구조 참조: common/live2d/scrap_model.py vertex_info_from_metadata()
  vertex_indices → mesh_tris   (M, 3) int32
  vertex_pos     → mesh_verts  (N, 2) float32
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
    """이미지/핀 조합이 바뀔 때만 재계산되는 메시 캐시.

    Live2D 대응:
      mesh_verts ↔ vertex_pos     (이미지 UV 좌표)
      mesh_tris  ↔ vertex_indices (삼각형 구성 인덱스)
      vert_seg   ↔ 세그먼트 소속 (0=upper, 1=lower)
    """
    img_orig:   np.ndarray    # BGRA 원본 (H, W, 4)
    pins:       PuppetPins
    mesh_verts: np.ndarray    # (N, 2) float32 — 이미지 공간 vertex
    mesh_tris:  np.ndarray    # (M, 3) int32   — triangle 인덱스
    vert_seg:   np.ndarray    # (N,) int8       — 0=upper, 1=lower


def build_segment_cache(arm_img: np.ndarray, pins: PuppetPins) -> SegmentCache:
    """이미지와 핀이 바뀔 때만 호출. 스트립 메시 사전 계산."""
    h, w = arm_img.shape[:2]
    mesh_verts, mesh_tris, vert_seg = _build_arm_mesh(pins, h, w)
    return SegmentCache(
        img_orig=arm_img,
        pins=pins,
        mesh_verts=mesh_verts,
        mesh_tris=mesh_tris,
        vert_seg=vert_seg,
    )


def apply_puppet_warp(
    cache: SegmentCache,
    vid_shldr: Tuple[float, float],
    vid_elbow: Tuple[float, float],
    vid_wrist: Tuple[float, float],
    W: int, H: int,
) -> np.ndarray:
    """프레임마다 호출. 메시 변형 + remap.

    Returns:
        BGRA numpy array (H×W×4) — 메시 밖은 alpha=0
    """
    dst_verts = _deform_mesh(
        cache.mesh_verts, cache.vert_seg, cache.pins,
        vid_shldr, vid_elbow, vid_wrist,
    )
    map_x, map_y = _mesh_to_remap(
        cache.mesh_verts, dst_verts, cache.mesh_tris, W, H,
    )
    warped = cv2.remap(
        cache.img_orig, map_x, map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return warped


def pins_degenerate(pins: PuppetPins, min_dist: float = 8.0) -> bool:
    """핀 간격이 너무 좁으면 True."""
    return not pins.is_valid(min_dist)


# ── 메시 생성 ─────────────────────────────────────────────────────────────────

def _build_arm_mesh(
    pins: PuppetPins,
    img_h: int,
    img_w: int,
    n_segs: int = 8,
    width_px: Optional[float] = None,
):
    """Live2D 스타일 Strip Mesh 생성.

    N = n_segs (per segment)
    총 cross-section: 2N+1 (elbow 공유)
    총 vertex: 2 * (2N+1) = 4N+2  (예: N=8 → 34)
    총 triangle: 4N                (예: N=8 → 32)

    Vertex 레이아웃:
      indices 0  .. 2N   : L side  (left/upper edge)
      indices 2N+1..4N+1 : R side  (right/lower edge)

    Returns:
        mesh_verts (2*(2N+1), 2) float32
        mesh_tris  (4N, 3)       int32
        vert_seg   (2*(2N+1),)   int8
    """
    N = n_segs
    s, e, w = (np.array(pins.img_shldr, np.float64),
               np.array(pins.img_elbow, np.float64),
               np.array(pins.img_wrist, np.float64))

    if width_px is None:
        arm_len = np.linalg.norm(w - s)
        width_px = max(arm_len * 0.35, 10.0)
    half_w = float(width_px)

    perp_up = _rot90(_norm(e - s))
    perp_lo = _rot90(_norm(w - e))

    all_L: list = []
    all_R: list = []
    all_seg: list = []

    # Upper: shoulder(i=0) → elbow(i=N)
    for i in range(N + 1):
        t = i / N
        center = s + t * (e - s)
        all_L.append(center + perp_up * half_w)
        all_R.append(center - perp_up * half_w)
        all_seg.append(0)

    # Lower: elbow+1/N → wrist (i=1..N, elbow 제외 → 공유)
    for i in range(1, N + 1):
        t = i / N
        center = e + t * (w - e)
        all_L.append(center + perp_lo * half_w)
        all_R.append(center - perp_lo * half_w)
        all_seg.append(1)

    M = 2 * N + 1  # total cross-sections

    # [L0..L(M-1), R0..R(M-1)]
    mesh_verts = np.array(all_L + all_R, dtype=np.float32)  # (2M, 2)
    vert_seg   = np.array(all_seg + all_seg, dtype=np.int8)  # (2M,)

    # Triangulate quads: cross-section i & i+1
    #   L[i]=i, L[i+1]=i+1, R[i]=M+i, R[i+1]=M+i+1
    #   Tri1: (i, i+1, M+i)
    #   Tri2: (i+1, M+i+1, M+i)
    tris = []
    for i in range(M - 1):
        li,  li1 = i,     i + 1
        ri,  ri1 = M + i, M + i + 1
        tris.append([li, li1, ri])
        tris.append([li1, ri1, ri])

    mesh_tris = np.array(tris, dtype=np.int32)  # (2*(M-1)=4N, 3)
    return mesh_verts, mesh_tris, vert_seg


# ── 변형 ──────────────────────────────────────────────────────────────────────

def _deform_mesh(
    mesh_verts: np.ndarray,
    vert_seg:   np.ndarray,
    src_pins:   PuppetPins,
    vid_shldr:  Tuple[float, float],
    vid_elbow:  Tuple[float, float],
    vid_wrist:  Tuple[float, float],
) -> np.ndarray:
    """이미지 공간 vertex → 비디오 공간 vertex.

    Seam 없는 이유: elbow vertex는 M_up/M_lo 모두 vid_elbow로 매핑 → C0 연속.

    Returns: dst_verts (N, 2) float32  — 변환 불가 vertex는 (-1, -1)
    """
    s_s, s_e, s_w = src_pins.arrays()
    d_s = np.array(vid_shldr, np.float64)
    d_e = np.array(vid_elbow,  np.float64)
    d_w = np.array(vid_wrist,  np.float64)

    M_up = _seg_affine(s_s, s_e, d_s, d_e)
    M_lo = _seg_affine(s_e, s_w, d_e, d_w)

    n = len(mesh_verts)
    dst_verts = np.full((n, 2), -1.0, dtype=np.float32)

    ones  = np.ones((n, 1), dtype=np.float64)
    pts_h = np.hstack([mesh_verts.astype(np.float64), ones])  # (n, 3)

    if M_up is not None:
        mask = (vert_seg == 0)
        dst_verts[mask] = (pts_h[mask] @ M_up.T).astype(np.float32)

    if M_lo is not None:
        mask = (vert_seg == 1)
        dst_verts[mask] = (pts_h[mask] @ M_lo.T).astype(np.float32)

    return dst_verts


# ── 렌더링 ────────────────────────────────────────────────────────────────────

def _mesh_to_remap(
    mesh_verts: np.ndarray,
    dst_verts:  np.ndarray,
    mesh_tris:  np.ndarray,
    W: int, H: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """삼각형 단위 역방향 affine → (map_x, map_y).

    map=-1인 픽셀 → cv2.remap BORDER_CONSTANT → alpha=0
    (메시 밖 픽셀은 그릴 삼각형이 없으므로 자동으로 렌더 안 됨)
    """
    map_x = np.full((H, W), -1.0, np.float32)
    map_y = np.full((H, W), -1.0, np.float32)

    for tri_idx in mesh_tris:                              # (3,) int32
        src_tri = mesh_verts[tri_idx].astype(np.float32)  # (3, 2) 소스 UV
        dst_tri = dst_verts[tri_idx].astype(np.float32)   # (3, 2) 화면 좌표

        # 화면 bbox
        x_min_f, y_min_f = dst_tri.min(axis=0)
        x_max_f, y_max_f = dst_tri.max(axis=0)
        x0 = max(0,     int(np.floor(x_min_f)))
        y0 = max(0,     int(np.floor(y_min_f)))
        x1 = min(W - 1, int(np.ceil(x_max_f)))
        y1 = min(H - 1, int(np.ceil(y_max_f)))
        if x0 > x1 or y0 > y1:
            continue

        # 삼각형 넓이 체크 (degenerate 방지)
        area = abs(
            (dst_tri[1, 0] - dst_tri[0, 0]) * (dst_tri[2, 1] - dst_tri[0, 1])
            - (dst_tri[2, 0] - dst_tri[0, 0]) * (dst_tri[1, 1] - dst_tri[0, 1])
        )
        if area < 0.5:
            continue

        # 역방향 affine: 화면 좌표 → 소스 UV
        try:
            M_inv = cv2.getAffineTransform(dst_tri, src_tri)  # (2, 3)
        except Exception:
            continue

        # bbox 내 픽셀 그리드
        ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        pts = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float32)  # (K, 2)

        # 삼각형 내부 픽셀만 선택 (barycentric)
        inside = _pts_in_triangle(pts, dst_tri)
        if not inside.any():
            continue

        pts_in = pts[inside]                                   # (P, 2)
        ones   = np.ones((len(pts_in), 1), np.float32)
        pts_h  = np.hstack([pts_in, ones])                    # (P, 3)
        uv     = (pts_h @ M_inv.T).astype(np.float32)        # (P, 2) 소스 좌표

        px = pts_in[:, 0].astype(np.int32)
        py = pts_in[:, 1].astype(np.int32)
        map_x[py, px] = uv[:, 0]
        map_y[py, px] = uv[:, 1]

    return map_x, map_y


def _pts_in_triangle(pts: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Vectorized barycentric point-in-triangle test.

    pts: (N, 2) float32
    tri: (3, 2) float32
    Returns: (N,) bool
    """
    v0 = (tri[1] - tri[0]).astype(np.float64)
    v1 = (tri[2] - tri[0]).astype(np.float64)
    v2 = pts.astype(np.float64) - tri[0].astype(np.float64)  # (N, 2)

    d00 = float(np.dot(v0, v0))
    d01 = float(np.dot(v0, v1))
    d11 = float(np.dot(v1, v1))
    d02 = v2 @ v0  # (N,)
    d12 = v2 @ v1  # (N,)

    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-10:
        return np.zeros(len(pts), dtype=bool)

    inv = 1.0 / denom
    u = (d11 * d02 - d01 * d12) * inv
    v = (d00 * d12 - d01 * d02) * inv

    return (u >= 0) & (v >= 0) & (u + v <= 1.0)


# ── 내부 헬퍼 ─────────────────────────────────────────────────────────────────

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
