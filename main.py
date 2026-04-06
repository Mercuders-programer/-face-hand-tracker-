#!/usr/bin/env python3
"""
PoseTracker — 얼굴 + 손 추적 → After Effects 내보내기

사용법:
  python main.py --video <영상경로>        영상 파일 처리
  python main.py --webcam                  웹캠 실시간 (테스트용, 내보내기 없음)
  python main.py --webcam --save           웹캠 → 녹화 후 내보내기

옵션:
  --video   <path>    입력 영상 (mp4, avi, mov 등)
  --webcam            웹캠 실시간 미리보기 (내보내기 없음)
  --webcam-id <n>     웹캠 장치 번호 (기본: 0)
  --save              --webcam 녹화 후 자동으로 내보내기까지 수행
  --output  <dir>     출력 폴더 (기본: ./output)
  --no-preview        미리보기 창 비활성화 (영상 모드)
  --conf-face <n>     얼굴 검출 최소 신뢰도 0~1 (기본: 0.5)
  --conf-hand <n>     손 검출 최소 신뢰도 0~1 (기본: 0.5)
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from tracker  import Tracker
from exporter import export_json, export_ae_keyframes


def parse_args():
    p = argparse.ArgumentParser(
        description="얼굴(눈/코/입) + 손가락 추적 → JSON + AE Keyframe Data 내보내기",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--video",   metavar="PATH", help="입력 영상 파일 경로")
    group.add_argument("--webcam",  action="store_true", help="웹캠 실시간 미리보기")

    p.add_argument("--webcam-id",  type=int,   default=0,       help="웹캠 장치 번호 (기본: 0)")
    p.add_argument("--save",       action="store_true",          help="웹캠 모드에서 녹화 후 내보내기")
    p.add_argument("--output",     default="./output",           help="결과 저장 폴더 (기본: ./output)")
    p.add_argument("--no-preview", action="store_true",          help="미리보기 창 비활성화")
    p.add_argument("--conf-face",  type=float, default=0.5,      help="얼굴 검출 신뢰도 임계값")
    p.add_argument("--conf-hand",  type=float, default=0.5,      help="손 검출 신뢰도 임계값")
    return p.parse_args()


def print_progress(current: int, total: int):
    if total <= 0:
        return
    pct  = int(100 * current / total)
    bars = '#' * (pct // 2) + '-' * (50 - pct // 2)
    print(f"\r[{bars}] {pct:3d}% ({current}/{total})", end='', flush=True)


def run_webcam(args):
    """웹캠 실시간 추적 (Q키로 종료)"""
    import cv2
    import mediapipe as mp

    print("=" * 50)
    print("  PoseTracker — 웹캠 실시간 모드")
    print("=" * 50)
    print(f"  웹캠 ID : {args.webcam_id}")
    print(f"  종료    : Q 키")
    if args.save:
        print(f"  종료 후 내보내기: {args.output}/")
    print()

    cap = cv2.VideoCapture(args.webcam_id)
    if not cap.isOpened():
        print(f"오류: 웹캠 {args.webcam_id}을 열 수 없습니다.", file=sys.stderr)
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[웹캠] {w}×{h} @ {fps:.0f} fps")

    mp_face  = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    # --save 시 프레임 기록용
    from tracker import FrameData, VideoInfo
    tracker_obj = Tracker(args.conf_face, args.conf_hand)
    saved_frames = []

    with mp_face.FaceMesh(
            static_image_mode=False, max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=args.conf_face,
            min_tracking_confidence=0.5) as face_mesh, \
         mp_hands.Hands(
            static_image_mode=False, max_num_hands=2,
            min_detection_confidence=args.conf_hand,
            min_tracking_confidence=0.5) as hands_model:

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_res  = face_mesh.process(rgb)
            hands_res = hands_model.process(rgb)

            # 오버레이 그리기
            preview = frame.copy()
            if face_res.multi_face_landmarks:
                mp_draw.draw_landmarks(
                    preview,
                    face_res.multi_face_landmarks[0],
                    mp_face.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style())

            if hands_res.multi_hand_landmarks:
                for hlms in hands_res.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        preview, hlms,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style())

            # 상태 텍스트
            face_ok  = "O" if face_res.multi_face_landmarks else "X"
            hands_ok = f"{len(hands_res.multi_hand_landmarks or [])}손"
            cv2.putText(preview, f"얼굴:{face_ok}  손:{hands_ok}  (Q: 종료)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("PoseTracker — 웹캠 실시간", preview)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # --save 모드: 프레임 데이터 수집
            if args.save:
                fd = FrameData(index=frame_idx, timestamp=frame_idx / fps)
                fd.face = tracker_obj._extract_face(face_res, w, h)
                if hands_res.multi_hand_landmarks:
                    for hlms, hedness in zip(
                            hands_res.multi_hand_landmarks,
                            hands_res.multi_handedness):
                        hd = tracker_obj._extract_hand(hlms, hedness, w, h)
                        if hd.side == "left":
                            fd.left_hand = hd
                        else:
                            fd.right_hand = hd
                saved_frames.append(fd)

            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    # --save: 내보내기
    if args.save and saved_frames:
        print(f"\n[내보내기] {len(saved_frames)} 프레임 처리...")
        info = VideoInfo(width=w, height=h, fps=fps, total_frames=len(saved_frames))
        json_path = os.path.join(args.output, "tracking_data.json")
        ae_dir    = os.path.join(args.output, "ae_keyframes")
        export_json(saved_frames, info, json_path)
        export_ae_keyframes(saved_frames, info, ae_dir)
        print(f"완료: {args.output}/")
    elif args.save:
        print("저장된 프레임이 없습니다.")


def run_video(args):
    """영상 파일 처리"""
    print("=" * 50)
    print("  PoseTracker — 영상 파일 모드")
    print("=" * 50)
    print(f"  입력: {args.video}")
    print(f"  출력: {args.output}")
    print()

    tracker = Tracker(
        min_face_confidence=args.conf_face,
        min_hand_confidence=args.conf_hand,
    )

    try:
        frames, info = tracker.process_video(
            video_path   = args.video,
            show_preview = not args.no_preview,
            callback     = print_progress,
        )
    except FileNotFoundError as e:
        print(f"\n오류: {e}", file=sys.stderr)
        sys.exit(1)

    if not frames:
        print("\n오류: 처리된 프레임이 없습니다.", file=sys.stderr)
        sys.exit(1)

    print()

    json_path = os.path.join(args.output, "tracking_data.json")
    ae_dir    = os.path.join(args.output, "ae_keyframes")

    ok  = export_json(frames, info, json_path)
    ok &= export_ae_keyframes(frames, info, ae_dir)

    if ok:
        n = len(frames)
        face_det = sum(1 for f in frames if f.face.detected)
        lh_det   = sum(1 for f in frames if f.left_hand.detected)
        rh_det   = sum(1 for f in frames if f.right_hand.detected)

        print()
        print("=" * 50)
        print("  완료!")
        print(f"  총 프레임:   {n}")
        print(f"  얼굴 검출:   {face_det} 프레임 ({100*face_det//n}%)")
        print(f"  왼손 검출:   {lh_det} 프레임 ({100*lh_det//n}%)")
        print(f"  오른손 검출: {rh_det} 프레임 ({100*rh_det//n}%)")
        print()
        print(f"  JSON : {json_path}")
        print(f"  AE   : {ae_dir}/")
        print("=" * 50)
        print()
        print("After Effects에서 사용하려면:")
        print("  방법 1 (자동): scripts/import_to_ae.jsx 실행")
        print("         File > Scripts > Run Script File → JSON 선택")
        print("  방법 2 (수동): ae_keyframes/ 안의 .txt 파일을")
        print("         Null Object 선택 후 Edit > Paste")
    else:
        print("\n내보내기 중 오류가 발생했습니다.", file=sys.stderr)
        sys.exit(1)


def main():
    args = parse_args()
    if args.webcam:
        run_webcam(args)
    else:
        run_video(args)


if __name__ == "__main__":
    main()
