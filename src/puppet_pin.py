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
    img_hand:  Optional[Tuple[float, float]] = None  # 4번째 핀 (손가락 끝)

    def arrays(self):
        res = [np.array(self.img_shldr, dtype=np.float64),
               np.array(self.img_elbow, dtype=np.float64),
               np.array(self.img_wrist, dtype=np.float64)]
        if self.img_hand is not None:
            res.append(np.array(self.img_hand, dtype=np.float64))
        return tuple(res)

    def is_valid(self, min_dist: float = 8.0) -> bool:
        arrs = self.arrays()
        s, e, w = arrs[0], arrs[1], arrs[2]
        ok = (np.linalg.norm(e - s) >= min_dist and
              np.linalg.norm(w - e) >= min_dist)
        if ok and self.img_hand is not None:
            ok = np.linalg.norm(arrs[3] - w) >= min_dist
        return ok


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
    vert_blend: np.ndarray    # (N,) float32    — 0=M_up, 1=M_lo (elbow blend)


def build_segment_cache(arm_img: np.ndarray, pins: PuppetPins) -> SegmentCache:
    """이미지와 핀이 바뀔 때만 호출. 스트립 메시 사전 계산."""
    h, w = arm_img.shape[:2]
    mesh_verts, mesh_tris, vert_seg, vert_blend = _build_arm_mesh(pins, h, w)
    return SegmentCache(
        img_orig=arm_img,
        pins=pins,
        mesh_verts=mesh_verts,
        mesh_tris=mesh_tris,
        vert_seg=vert_seg,
        vert_blend=vert_blend,
    )


def apply_puppet_warp(
    cache: SegmentCache,
    vid_shldr: Tuple[float, float],
    vid_elbow: Tuple[float, float],
    vid_wrist: Tuple[float, float],
    W: int, H: int,
    vid_hand: Optional[Tuple[float, float]] = None,
    size_pct: float = 100.0,
) -> np.ndarray:
    """프레임마다 호출. 메시 변형 + remap.

    Returns:
        BGRA numpy array (H×W×4) — 메시 밖은 alpha=0
    """
    dst_verts = _deform_mesh(
        cache.mesh_verts, cache.vert_seg, cache.vert_blend, cache.pins,
        vid_shldr, vid_elbow, vid_wrist, vid_hand=vid_hand,
        size_pct=size_pct,
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

    img_hand=None (2-세그먼트):
        총 cross-section: 2N+1  (elbow 공유)
        vert_blend: [0, 1]  (0=M_up, 1=M_lo)

    img_hand 있음 (3-세그먼트):
        총 cross-section: 3N+1  (elbow/wrist 공유)
        vert_blend: [0, 2]  (0=M_up, 1=M_lo, 2=M_hand)

    Vertex 레이아웃 (M = total cross-sections):
      indices 0  .. M-1  : L side  (left/upper edge)
      indices M  .. 2M-1 : R side  (right/lower edge)

    Returns:
        mesh_verts (2M, 2) float32
        mesh_tris  (2*(M-1), 3) int32
        vert_seg   (2M,)   int8
        vert_blend (2M,)   float32
    """
    N = n_segs
    s, e, w = (np.array(pins.img_shldr, np.float64),
               np.array(pins.img_elbow, np.float64),
               np.array(pins.img_wrist, np.float64))
    has_hand = pins.img_hand is not None
    if has_hand:
        h_pt = np.array(pins.img_hand, np.float64)

    if width_px is None:
        end_pt = h_pt if has_hand else w
        arm_len = np.linalg.norm(end_pt - s)
        width_px = max(arm_len * 0.35, 10.0)
    half_w = float(width_px)

    perp_up = _rot90(_norm(e - s))
    perp_lo = _rot90(_norm(w - e))

    # elbow 방향 블렌딩: 두 perp의 평균 방향
    elbow_perp_raw = perp_up + perp_lo
    elbow_perp_len = np.linalg.norm(elbow_perp_raw)
    if elbow_perp_len < 0.1:   # 대향(≈180°) 예외 처리
        elbow_perp = perp_up
    else:
        elbow_perp = elbow_perp_raw / elbow_perp_len

    n_blend = min(3, max(1, N // 2))  # 혼합 구간 cross-section 수

    all_L: list = []
    all_R: list = []
    all_seg: list = []

    # Upper: shoulder(i=0) → elbow(i=N)
    for i in range(N + 1):
        t = i / N
        center = s + t * (e - s)
        if i >= N - n_blend:
            # elbow 근방: perp_up → elbow_perp 점진 전환
            alpha = (i - (N - n_blend)) / max(n_blend, 1)
            alpha = np.clip(alpha, 0.0, 1.0)
            perp = perp_up * (1.0 - alpha) + elbow_perp * alpha
        else:
            perp = perp_up
        all_L.append(center + perp * half_w)
        all_R.append(center - perp * half_w)
        all_seg.append(0)

    if has_hand:
        perp_hand = _rot90(_norm(h_pt - w))
        # wrist junction 방향: perp_lo와 perp_hand의 평균
        wrist_perp_raw = perp_lo + perp_hand
        wrist_perp_len = np.linalg.norm(wrist_perp_raw)
        if wrist_perp_len < 0.1:
            wrist_perp = perp_lo
        else:
            wrist_perp = wrist_perp_raw / wrist_perp_len

        # Lower: elbow+1/N → wrist (i=1..N), elbow/wrist 양쪽 blend zone
        for i in range(1, N + 1):
            t = i / N
            center = e + t * (w - e)
            if i <= n_blend:
                # elbow 근방: elbow_perp → perp_lo
                alpha = np.clip(i / max(n_blend, 1), 0.0, 1.0)
                perp = elbow_perp * (1.0 - alpha) + perp_lo * alpha
            elif i >= N - n_blend:
                # wrist 근방: perp_lo → wrist_perp
                alpha = np.clip((i - (N - n_blend)) / max(n_blend, 1), 0.0, 1.0)
                perp = perp_lo * (1.0 - alpha) + wrist_perp * alpha
            else:
                perp = perp_lo
            all_L.append(center + perp * half_w)
            all_R.append(center - perp * half_w)
            all_seg.append(1)

        # Hand: wrist+1/N → hand (i=1..N)
        for i in range(1, N + 1):
            t = i / N
            center = w + t * (h_pt - w)
            if i <= n_blend:
                # wrist 근방: wrist_perp → perp_hand
                alpha = np.clip(i / max(n_blend, 1), 0.0, 1.0)
                perp = wrist_perp * (1.0 - alpha) + perp_hand * alpha
            else:
                perp = perp_hand
            all_L.append(center + perp * half_w)
            all_R.append(center - perp * half_w)
            all_seg.append(2)

        M = 3 * N + 1  # total cross-sections

        # vert_blend [0, 2] — 2단계 코사인 전환
        cs_blend = []
        for cs in range(M):
            if cs < N - n_blend:
                b = 0.0                                          # 순수 M_up
            elif cs <= N + n_blend:
                denom = max(2 * n_blend, 1)
                progress = (cs - (N - n_blend)) / denom
                b = 0.5 * (1.0 - np.cos(np.pi * float(np.clip(progress, 0.0, 1.0))))
            elif cs < 2 * N - n_blend:
                b = 1.0                                          # 순수 M_lo
            elif cs <= 2 * N + n_blend:
                denom = max(2 * n_blend, 1)
                progress = (cs - (2 * N - n_blend)) / denom
                b = 1.0 + 0.5 * (1.0 - np.cos(np.pi * float(np.clip(progress, 0.0, 1.0))))
            else:
                b = 2.0                                          # 순수 M_hand
            cs_blend.append(b)

    else:
        # Lower: elbow+1/N → wrist (i=1..N, elbow 제외 → 공유)
        for i in range(1, N + 1):
            t = i / N
            center = e + t * (w - e)
            if i <= n_blend:
                # elbow 근방: elbow_perp → perp_lo 점진 전환
                alpha = np.clip(i / max(n_blend, 1), 0.0, 1.0)
                perp = elbow_perp * (1.0 - alpha) + perp_lo * alpha
            else:
                perp = perp_lo
            all_L.append(center + perp * half_w)
            all_R.append(center - perp * half_w)
            all_seg.append(1)

        M = 2 * N + 1  # total cross-sections

        # vert_blend [0, 1] 코사인 스텝 (기존)
        cs_blend = []
        for cs in range(M):
            if cs <= N - n_blend - 1:
                b = 0.0                                          # 순수 M_up
            elif cs >= N + n_blend + 1:
                b = 1.0                                          # 순수 M_lo
            else:
                denom = max(2 * n_blend, 1)
                progress = (cs - (N - n_blend)) / denom
                b = 0.5 * (1.0 - np.cos(np.pi * float(np.clip(progress, 0.0, 1.0))))
            cs_blend.append(b)

    vert_blend = np.array(cs_blend + cs_blend, dtype=np.float32)  # (2M,)

    # [L0..L(M-1), R0..R(M-1)]
    mesh_verts = np.array(all_L + all_R, dtype=np.float32)  # (2M, 2)
    vert_seg   = np.array(all_seg + all_seg, dtype=np.int8)  # (2M,)

    # Triangulate quads: cross-section i & i+1
    tris = []
    for i in range(M - 1):
        li,  li1 = i,     i + 1
        ri,  ri1 = M + i, M + i + 1
        tris.append([li, li1, ri])
        tris.append([li1, ri1, ri])

    mesh_tris = np.array(tris, dtype=np.int32)
    return mesh_verts, mesh_tris, vert_seg, vert_blend


# ── 변형 ──────────────────────────────────────────────────────────────────────

def _deform_mesh(
    mesh_verts: np.ndarray,
    vert_seg:   np.ndarray,
    vert_blend: np.ndarray,
    src_pins:   PuppetPins,
    vid_shldr:  Tuple[float, float],
    vid_elbow:  Tuple[float, float],
    vid_wrist:  Tuple[float, float],
    vid_hand:   Optional[Tuple[float, float]] = None,
    size_pct:   float = 100.0,
) -> np.ndarray:
    """이미지 공간 vertex → 비디오 공간 vertex.

    vert_blend [0, 1]: 2-세그먼트 (기존)
      b_elb=clip(b,0,1): 0=M_up, 1=M_lo

    vert_blend [0, 2]: 3-세그먼트 (img_hand 있을 때)
      b_elb=clip(b,0,1): upper↔lower 전환
      b_wrt=clip(b-1,0,1): lower↔hand 전환

    Returns: dst_verts (N, 2) float32  — 변환 불가 vertex는 (-1, -1)
    """
    arrs = src_pins.arrays()
    s_s, s_e, s_w = arrs[0], arrs[1], arrs[2]
    d_s = np.array(vid_shldr, np.float64)
    d_e = np.array(vid_elbow,  np.float64)
    d_w = np.array(vid_wrist,  np.float64)

    # Global scale = 전체 호 길이 비율 (세그먼트간 폭 균일화)
    img_total = np.linalg.norm(s_e - s_s) + np.linalg.norm(s_w - s_e)
    vid_total = np.linalg.norm(d_e - d_s) + np.linalg.norm(d_w - d_e)
    if src_pins.img_hand is not None and vid_hand is not None:
        img_total += np.linalg.norm(arrs[3] - s_w)
        vid_total += np.linalg.norm(np.array(vid_hand) - d_w)
    global_scale = vid_total / img_total if img_total > 1e-9 else 1.0
    global_scale *= size_pct / 100.0

    M_up = _seg_affine(s_s, s_e, d_s, d_e, global_scale=global_scale)
    M_lo = _seg_affine(s_e, s_w, d_e, d_w, global_scale=global_scale)

    # 3번째 세그먼트 (손목→손): src와 vid 모두 있을 때만
    M_hand = None
    if src_pins.img_hand is not None and vid_hand is not None:
        s_h = arrs[3]
        d_h = np.array(vid_hand, np.float64)
        M_hand = _seg_affine(s_w, s_h, d_w, d_h, global_scale=global_scale)

    n = len(mesh_verts)
    ones  = np.ones((n, 1), dtype=np.float64)
    pts_h = np.hstack([mesh_verts.astype(np.float64), ones])  # (n, 3)

    if M_up is not None and M_lo is not None:
        dst_up = (pts_h @ M_up.T).astype(np.float32)          # (n, 2)
        dst_lo = (pts_h @ M_lo.T).astype(np.float32)          # (n, 2)
        b      = vert_blend[:, np.newaxis]                     # (n, 1)
        b_elb  = np.clip(b, 0.0, 1.0)                         # 0→1 at elbow
        b_wrt  = np.clip(b - 1.0, 0.0, 1.0)                   # 0→1 at wrist
        step1  = (1.0 - b_elb) * dst_up + b_elb * dst_lo      # upper↔lower
        if M_hand is not None:
            dst_hand  = (pts_h @ M_hand.T).astype(np.float32)
            dst_verts = ((1.0 - b_wrt) * step1 + b_wrt * dst_hand).astype(np.float32)
        else:
            dst_verts = step1.astype(np.float32)
    elif M_up is not None:
        dst_verts = (pts_h @ M_up.T).astype(np.float32)
    elif M_lo is not None:
        dst_verts = (pts_h @ M_lo.T).astype(np.float32)
    else:
        dst_verts = np.full((n, 2), -1.0, dtype=np.float32)

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
    global_scale: Optional[float] = None,
) -> Optional[np.ndarray]:
    """3점(src→dst) Affine 행렬 계산.

    global_scale: 지정 시 수직 방향에 이 스케일 적용 (세그먼트간 폭 균일화)
    None: 기존 동작 (수직 = 축 방향과 동일 스케일)
    """
    d_i = p1 - p0
    d_v = q1 - q0
    src = np.float32([p0, p1, p0 + _rot90(d_i)])
    if global_scale is not None:
        local_len = np.linalg.norm(d_i)
        perp_dst  = q0 + _rot90(_norm(d_v)) * global_scale * local_len
        dst = np.float32([q0, q1, perp_dst])
    else:
        dst = np.float32([q0, q1, q0 + _rot90(d_v)])
    try:
        return cv2.getAffineTransform(src, dst)
    except Exception:
        return None
