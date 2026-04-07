"""
exporter.py — 추적 데이터를 JSON + After Effects Keyframe Data 로 저장
"""

import json
import os
from typing import List, Callable
try:
    from .tracker import FrameData, VideoInfo, Point2D, PersonData, HAND_LANDMARK_NAMES, POSE_LANDMARK_NAMES
except ImportError:
    from tracker import FrameData, VideoInfo, Point2D, PersonData, HAND_LANDMARK_NAMES, POSE_LANDMARK_NAMES


# ──────────────────────────────────────────────────────────────────────────
# JSON 내보내기
# ──────────────────────────────────────────────────────────────────────────

def export_json(frames: List[FrameData], info: VideoInfo, out_path: str,
                include_face: bool = True, include_body: bool = True,
                include_hands: bool = True) -> bool:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    def pt(p: Point2D) -> dict:
        return {"x": round(p.x, 3), "y": round(p.y, 3), "confidence": round(p.confidence, 4)}

    data = {
        "metadata": {
            "fps":          info.fps,
            "total_frames": info.total_frames,
            "width":        info.width,
            "height":       info.height,
            "included":     {"face": include_face, "body": include_body, "hands": include_hands},
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

    def hand_to_dict(h):
        d = {"detected": h.detected, "side": h.side}
        if h.detected:
            d["landmarks"] = [
                {**pt(lm), "index": i, "name": HAND_LANDMARK_NAMES[i]}
                for i, lm in enumerate(h.landmarks)
            ]
        return d

    for fd in frames:
        jf = {"frame": fd.index, "timestamp": round(fd.timestamp, 4), "persons": []}

        for person in fd.persons:
            jp = {"person_id": person.person_id}

            if include_face:
                jface = {"detected": person.face.detected}
                if person.face.detected:
                    jface["named"] = {
                        "right_eye_outer":  pt(person.face.right_eye_outer),
                        "right_eye_inner":  pt(person.face.right_eye_inner),
                        "right_iris":       pt(person.face.right_iris),
                        "left_eye_inner":   pt(person.face.left_eye_inner),
                        "left_eye_outer":   pt(person.face.left_eye_outer),
                        "left_iris":        pt(person.face.left_iris),
                        "nose_bridge_top":  pt(person.face.nose_bridge_top),
                        "nose_tip":         pt(person.face.nose_tip),
                        "mouth_right":      pt(person.face.mouth_right),
                        "mouth_upper":      pt(person.face.mouth_upper),
                        "mouth_left":       pt(person.face.mouth_left),
                        "mouth_lower":      pt(person.face.mouth_lower),
                    }
                    jface["all_landmarks"] = [pt(p) for p in person.face.all]
                jp["face"] = jface

            if include_hands:
                jp["left_hand"]  = hand_to_dict(person.left_hand)
                jp["right_hand"] = hand_to_dict(person.right_hand)

            if include_body:
                jpose = {"detected": person.pose.detected}
                if person.pose.detected:
                    jpose["key_joints"] = {
                        "left_shoulder":  pt(person.pose.left_shoulder),
                        "right_shoulder": pt(person.pose.right_shoulder),
                        "left_elbow":     pt(person.pose.left_elbow),
                        "right_elbow":    pt(person.pose.right_elbow),
                        "left_wrist":     pt(person.pose.left_wrist),
                        "right_wrist":    pt(person.pose.right_wrist),
                        "left_hip":       pt(person.pose.left_hip),
                        "right_hip":      pt(person.pose.right_hip),
                        "left_knee":      pt(person.pose.left_knee),
                        "right_knee":     pt(person.pose.right_knee),
                        "left_ankle":     pt(person.pose.left_ankle),
                        "right_ankle":    pt(person.pose.right_ankle),
                    }
                    jpose["all_landmarks"] = [pt(p) for p in person.pose.all]
                jp["pose"] = jpose

            jf["persons"].append(jp)

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


def export_ae_keyframes(frames: List[FrameData], info: VideoInfo, out_dir: str,
                        include_face: bool = True, include_body: bool = True,
                        include_hands: bool = True) -> bool:
    max_persons = max((len(fd.persons) for fd in frames), default=0)
    if max_persons == 0:
        print(f"[Exporter] 경고: 감지된 사람 없음 — AE 출력 건너뜀")
        return True

    ok = True
    face_pts = [
        ("right_eye_outer",  lambda p: p.face.right_eye_outer),
        ("right_eye_inner",  lambda p: p.face.right_eye_inner),
        ("right_iris",       lambda p: p.face.right_iris),
        ("left_eye_inner",   lambda p: p.face.left_eye_inner),
        ("left_eye_outer",   lambda p: p.face.left_eye_outer),
        ("left_iris",        lambda p: p.face.left_iris),
        ("nose_bridge_top",  lambda p: p.face.nose_bridge_top),
        ("nose_tip",         lambda p: p.face.nose_tip),
        ("mouth_right",      lambda p: p.face.mouth_right),
        ("mouth_upper",      lambda p: p.face.mouth_upper),
        ("mouth_left",       lambda p: p.face.mouth_left),
        ("mouth_lower",      lambda p: p.face.mouth_lower),
    ]
    hand_tips = [
        ("wrist",      lambda h: h.wrist),
        ("thumb_tip",  lambda h: h.thumb_tip),
        ("index_tip",  lambda h: h.index_tip),
        ("middle_tip", lambda h: h.middle_tip),
        ("ring_tip",   lambda h: h.ring_tip),
        ("pinky_tip",  lambda h: h.pinky_tip),
    ]
    pose_joints = [
        ("left_shoulder",  lambda p: p.pose.left_shoulder),
        ("right_shoulder", lambda p: p.pose.right_shoulder),
        ("left_elbow",     lambda p: p.pose.left_elbow),
        ("right_elbow",    lambda p: p.pose.right_elbow),
        ("left_wrist",     lambda p: p.pose.left_wrist),
        ("right_wrist",    lambda p: p.pose.right_wrist),
        ("left_hip",       lambda p: p.pose.left_hip),
        ("right_hip",      lambda p: p.pose.right_hip),
        ("left_knee",      lambda p: p.pose.left_knee),
        ("right_knee",     lambda p: p.pose.right_knee),
        ("left_ankle",     lambda p: p.pose.left_ankle),
        ("right_ankle",    lambda p: p.pose.right_ankle),
    ]

    for pid in range(max_persons):
        p_dir    = os.path.join(out_dir, f"person_{pid}")
        face_dir = os.path.join(p_dir, "face")
        lh_dir   = os.path.join(p_dir, "hands", "left")
        rh_dir   = os.path.join(p_dir, "hands", "right")
        lh_all   = os.path.join(lh_dir, "all")
        rh_all   = os.path.join(rh_dir, "all")
        pose_dir = os.path.join(p_dir, "pose")

        def _person(f, pid=pid):
            return f.persons[pid] if pid < len(f.persons) else None

        # ── 얼굴 12개 포인트
        if include_face:
            for name, getter in face_pts:
                ok &= _write_ae_file(
                    os.path.join(face_dir, f"{name}.txt"), info, frames,
                    lambda f, g=getter, pid=pid: (
                        g(f.persons[pid]) if pid < len(f.persons) and f.persons[pid].face.detected
                        else Point2D()))

        # ── 손 주요 6포인트 × 2손 + 전체 21포인트
        if include_hands:
            for name, hgetter in hand_tips:
                ok &= _write_ae_file(
                    os.path.join(lh_dir, f"{name}.txt"), info, frames,
                    lambda f, g=hgetter, pid=pid: (
                        g(f.persons[pid].left_hand) if pid < len(f.persons) and f.persons[pid].left_hand.detected
                        else Point2D()))
                ok &= _write_ae_file(
                    os.path.join(rh_dir, f"{name}.txt"), info, frames,
                    lambda f, g=hgetter, pid=pid: (
                        g(f.persons[pid].right_hand) if pid < len(f.persons) and f.persons[pid].right_hand.detected
                        else Point2D()))
            for i, lm_name in enumerate(HAND_LANDMARK_NAMES):
                ok &= _write_ae_file(
                    os.path.join(lh_all, f"{lm_name}.txt"), info, frames,
                    lambda f, idx=i, pid=pid: (
                        f.persons[pid].left_hand.landmarks[idx]
                        if pid < len(f.persons) and f.persons[pid].left_hand.detected
                        and len(f.persons[pid].left_hand.landmarks) > idx
                        else Point2D()))
                ok &= _write_ae_file(
                    os.path.join(rh_all, f"{lm_name}.txt"), info, frames,
                    lambda f, idx=i, pid=pid: (
                        f.persons[pid].right_hand.landmarks[idx]
                        if pid < len(f.persons) and f.persons[pid].right_hand.detected
                        and len(f.persons[pid].right_hand.landmarks) > idx
                        else Point2D()))

        # ── 포즈 주요 관절 12개
        if include_body:
            for name, getter in pose_joints:
                ok &= _write_ae_file(
                    os.path.join(pose_dir, f"{name}.txt"), info, frames,
                    lambda f, g=getter, pid=pid: (
                        g(f.persons[pid]) if pid < len(f.persons) and f.persons[pid].pose.detected
                        else Point2D()))

    print(f"[Exporter] AE Keyframe Data 저장 완료: {out_dir}/")
    return ok
