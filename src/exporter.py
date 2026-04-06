"""
exporter.py — 추적 데이터를 JSON + After Effects Keyframe Data 로 저장
"""

import json
import os
from typing import List, Callable
try:
    from .tracker import FrameData, VideoInfo, Point2D, HAND_LANDMARK_NAMES, POSE_LANDMARK_NAMES
except ImportError:
    from tracker import FrameData, VideoInfo, Point2D, HAND_LANDMARK_NAMES, POSE_LANDMARK_NAMES


# ──────────────────────────────────────────────────────────────────────────
# JSON 내보내기
# ──────────────────────────────────────────────────────────────────────────

def export_json(frames: List[FrameData], info: VideoInfo, out_path: str) -> bool:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    def pt(p: Point2D) -> dict:
        return {"x": round(p.x, 3), "y": round(p.y, 3), "confidence": round(p.confidence, 4)}

    data = {
        "metadata": {
            "fps":          info.fps,
            "total_frames": info.total_frames,
            "width":        info.width,
            "height":       info.height,
        },
        "landmark_info": {
            "face": {
                "total_points": 478,
                "named": {
                    "right_eye_outer":  33,
                    "right_eye_inner":  133,
                    "right_iris":       473,
                    "left_eye_inner":   362,
                    "left_eye_outer":   263,
                    "left_iris":        468,
                    "nose_bridge_top":  168,
                    "nose_tip":         4,
                    "mouth_right":      61,
                    "mouth_upper":      13,
                    "mouth_left":       291,
                    "mouth_lower":      14,
                }
            },
            "hand": {
                "total_points": 21,
                "names": HAND_LANDMARK_NAMES,
            },
            "pose": {
                "total_points": 33,
                "names": POSE_LANDMARK_NAMES,
                "key_joints": {
                    "left_shoulder": 11, "right_shoulder": 12,
                    "left_elbow":    13, "right_elbow":    14,
                    "left_wrist":    15, "right_wrist":    16,
                    "left_hip":      23, "right_hip":      24,
                    "left_knee":     25, "right_knee":      26,
                    "left_ankle":    27, "right_ankle":     28,
                }
            },
        },
        "frames": []
    }

    for fd in frames:
        jf = {"frame": fd.index, "timestamp": round(fd.timestamp, 4)}

        # 얼굴
        jface = {"detected": fd.face.detected}
        if fd.face.detected:
            jface["named"] = {
                "right_eye_outer":  pt(fd.face.right_eye_outer),
                "right_eye_inner":  pt(fd.face.right_eye_inner),
                "right_iris":       pt(fd.face.right_iris),
                "left_eye_inner":   pt(fd.face.left_eye_inner),
                "left_eye_outer":   pt(fd.face.left_eye_outer),
                "left_iris":        pt(fd.face.left_iris),
                "nose_bridge_top":  pt(fd.face.nose_bridge_top),
                "nose_tip":         pt(fd.face.nose_tip),
                "mouth_right":      pt(fd.face.mouth_right),
                "mouth_upper":      pt(fd.face.mouth_upper),
                "mouth_left":       pt(fd.face.mouth_left),
                "mouth_lower":      pt(fd.face.mouth_lower),
            }
            jface["all_landmarks"] = [pt(p) for p in fd.face.all]
        jf["face"] = jface

        # 손
        def hand_to_dict(h):
            d = {"detected": h.detected, "side": h.side}
            if h.detected:
                d["landmarks"] = [
                    {**pt(lm), "index": i, "name": HAND_LANDMARK_NAMES[i]}
                    for i, lm in enumerate(h.landmarks)
                ]
            return d

        jf["left_hand"]  = hand_to_dict(fd.left_hand)
        jf["right_hand"] = hand_to_dict(fd.right_hand)

        # 포즈
        jpose = {"detected": fd.pose.detected}
        if fd.pose.detected:
            jpose["key_joints"] = {
                "left_shoulder":  pt(fd.pose.left_shoulder),
                "right_shoulder": pt(fd.pose.right_shoulder),
                "left_elbow":     pt(fd.pose.left_elbow),
                "right_elbow":    pt(fd.pose.right_elbow),
                "left_wrist":     pt(fd.pose.left_wrist),
                "right_wrist":    pt(fd.pose.right_wrist),
                "left_hip":       pt(fd.pose.left_hip),
                "right_hip":      pt(fd.pose.right_hip),
                "left_knee":      pt(fd.pose.left_knee),
                "right_knee":     pt(fd.pose.right_knee),
                "left_ankle":     pt(fd.pose.left_ankle),
                "right_ankle":    pt(fd.pose.right_ankle),
            }
            jpose["all_landmarks"] = [pt(p) for p in fd.pose.all]
        jf["pose"] = jpose

        data["frames"].append(jf)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[Exporter] JSON 저장 완료: {out_path}")
    return True


# ──────────────────────────────────────────────────────────────────────────
# AE Keyframe Data 내보내기
# ──────────────────────────────────────────────────────────────────────────

def _write_ae_file(path: str,
                   info: VideoInfo,
                   frames: List[FrameData],
                   getter: Callable[[FrameData], Point2D],
                   threshold: float = 0.05) -> bool:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    lines = []
    lines.append("Adobe After Effects 8.0 Keyframe Data\n")
    lines.append(f"\tUnits Per Second\t{info.fps}\n")
    lines.append(f"\tSource Width\t{info.width}\n")
    lines.append(f"\tSource Height\t{info.height}\n")
    lines.append("\tSource Pixel Aspect Ratio\t1\n")
    lines.append("\tComp Pixel Aspect Ratio\t1\n")
    lines.append("\n")
    lines.append("Transform\tPosition\n")
    lines.append("\tFrame\tX pixels\tY pixels\tZ pixels\t\n")

    count = 0
    for fd in frames:
        p = getter(fd)
        if p.confidence < threshold:
            continue
        lines.append(f"\t{fd.index}\t{p.x:.4f}\t{p.y:.4f}\t0.0000\t\n")
        count += 1

    lines.append("\nEnd of Keyframe Data\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    if count == 0:
        print(f"[Exporter] 경고: 유효 키프레임 없음 → {os.path.basename(path)}")
    return True


def export_ae_keyframes(frames: List[FrameData], info: VideoInfo, out_dir: str) -> bool:
    face_dir  = os.path.join(out_dir, "face")
    lh_dir    = os.path.join(out_dir, "hands", "left")
    rh_dir    = os.path.join(out_dir, "hands", "right")
    lh_all    = os.path.join(lh_dir,  "all")
    rh_all    = os.path.join(rh_dir,  "all")
    pose_dir  = os.path.join(out_dir, "pose")

    ok = True

    # ── 얼굴 12개 포인트 ─────────────────────────────────────────────────
    face_points = [
        ("right_eye_outer",  lambda f: f.face.right_eye_outer),
        ("right_eye_inner",  lambda f: f.face.right_eye_inner),
        ("right_iris",       lambda f: f.face.right_iris),
        ("left_eye_inner",   lambda f: f.face.left_eye_inner),
        ("left_eye_outer",   lambda f: f.face.left_eye_outer),
        ("left_iris",        lambda f: f.face.left_iris),
        ("nose_bridge_top",  lambda f: f.face.nose_bridge_top),
        ("nose_tip",         lambda f: f.face.nose_tip),
        ("mouth_right",      lambda f: f.face.mouth_right),
        ("mouth_upper",      lambda f: f.face.mouth_upper),
        ("mouth_left",       lambda f: f.face.mouth_left),
        ("mouth_lower",      lambda f: f.face.mouth_lower),
    ]

    for name, getter in face_points:
        path = os.path.join(face_dir, f"{name}.txt")
        ok &= _write_ae_file(path, info, frames, getter)

    # ── 손 주요 6포인트 × 2손 ────────────────────────────────────────────
    hand_tips = [
        ("wrist",      lambda h: h.wrist),
        ("thumb_tip",  lambda h: h.thumb_tip),
        ("index_tip",  lambda h: h.index_tip),
        ("middle_tip", lambda h: h.middle_tip),
        ("ring_tip",   lambda h: h.ring_tip),
        ("pinky_tip",  lambda h: h.pinky_tip),
    ]

    for name, hgetter in hand_tips:
        ok &= _write_ae_file(
            os.path.join(lh_dir, f"{name}.txt"), info, frames,
            lambda f, g=hgetter: g(f.left_hand) if f.left_hand.detected else Point2D())
        ok &= _write_ae_file(
            os.path.join(rh_dir, f"{name}.txt"), info, frames,
            lambda f, g=hgetter: g(f.right_hand) if f.right_hand.detected else Point2D())

    # ── 손 전체 21개 랜드마크 ────────────────────────────────────────────
    for i, lm_name in enumerate(HAND_LANDMARK_NAMES):
        ok &= _write_ae_file(
            os.path.join(lh_all, f"{lm_name}.txt"), info, frames,
            lambda f, idx=i: (f.left_hand.landmarks[idx]
                               if f.left_hand.detected and len(f.left_hand.landmarks) > idx
                               else Point2D()))
        ok &= _write_ae_file(
            os.path.join(rh_all, f"{lm_name}.txt"), info, frames,
            lambda f, idx=i: (f.right_hand.landmarks[idx]
                               if f.right_hand.detected and len(f.right_hand.landmarks) > idx
                               else Point2D()))

    # ── 포즈 주요 관절 12개 ──────────────────────────────────────────────
    pose_joints = [
        ("left_shoulder",  lambda f: f.pose.left_shoulder),
        ("right_shoulder", lambda f: f.pose.right_shoulder),
        ("left_elbow",     lambda f: f.pose.left_elbow),
        ("right_elbow",    lambda f: f.pose.right_elbow),
        ("left_wrist",     lambda f: f.pose.left_wrist),
        ("right_wrist",    lambda f: f.pose.right_wrist),
        ("left_hip",       lambda f: f.pose.left_hip),
        ("right_hip",      lambda f: f.pose.right_hip),
        ("left_knee",      lambda f: f.pose.left_knee),
        ("right_knee",     lambda f: f.pose.right_knee),
        ("left_ankle",     lambda f: f.pose.left_ankle),
        ("right_ankle",    lambda f: f.pose.right_ankle),
    ]

    for name, getter in pose_joints:
        path = os.path.join(pose_dir, f"{name}.txt")
        ok &= _write_ae_file(
            path, info, frames,
            lambda f, g=getter: g(f) if f.pose.detected else Point2D(),
        )

    print(f"[Exporter] AE Keyframe Data 저장 완료: {out_dir}/")
    return ok
