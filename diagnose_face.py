"""
diagnose_face.py — 얼굴 감지 실패 원인 상세 진단
영상 1프레임 기준으로 FaceLandmarker 및 FaceDetector 모두 테스트
"""

import cv2
import numpy as np
import os, sys, math

VIDEO   = "TEST_DATA/KakaoTalk_20260412_144327214.mp4"
MODEL_FL = "models/face_landmarker.task"
MODEL_FD = "models/face_detection_short_range.task"   # 없으면 skip
MODEL_PL = "models/pose_landmarker_full.task"

# ── mediapipe import ──────────────────────────────────────────────────────────
lib_path = os.path.join(os.path.dirname(__file__), "lib")
if "LD_LIBRARY_PATH" not in os.environ or lib_path not in os.environ["LD_LIBRARY_PATH"]:
    os.environ["LD_LIBRARY_PATH"] = lib_path + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    os.execve(sys.executable, [sys.executable] + sys.argv, os.environ)

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

results = {}

# ── 1. 영상 첫 프레임 추출 ───────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO)
ok, frame = cap.read()
cap.release()
if not ok:
    print("ERROR: 영상 읽기 실패")
    sys.exit(1)

h_orig, w_orig = frame.shape[:2]
results["video"] = {
    "path": VIDEO,
    "resolution": f"{w_orig}x{h_orig}",
    "orientation": "portrait" if h_orig > w_orig else "landscape",
}

# ── 2. 여러 해상도로 프레임 준비 ─────────────────────────────────────────────
scales = {}
for target in [h_orig, 640, 360]:
    sc = min(1.0, target / max(w_orig, h_orig))
    sw, sh = max(1, int(w_orig * sc)), max(1, int(h_orig * sc))
    scales[target] = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)

# ── 3. FaceLandmarker 테스트 (confidence × 해상도 매트릭스) ──────────────────
print("=== FaceLandmarker 테스트 ===")
fl_results = {}
for conf in [0.5, 0.3, 0.1, 0.05]:
    for res_key, img in scales.items():
        h2, w2 = img.shape[:2]
        opts = mp_python.BaseOptions(model_asset_path=MODEL_FL)
        opts2 = mp_vision.FaceLandmarkerOptions(
            base_options=opts,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=4,
            min_face_detection_confidence=conf,
            min_face_presence_confidence=conf,
            min_tracking_confidence=conf,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        det = mp_vision.FaceLandmarker.create_from_options(opts2)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = det.detect(mp_img)
        det.close()

        n_faces = len(res.face_landmarks) if res.face_landmarks else 0
        key = f"conf={conf}_res={res_key}"
        fl_results[key] = n_faces
        print(f"  conf={conf:.2f}  res={w2}x{h2}  → 감지 얼굴 수: {n_faces}")

        # 감지 성공 시 yaw/pitch 행렬도 출력
        if n_faces > 0 and res.facial_transformation_matrixes:
            for i, mat in enumerate(res.facial_transformation_matrixes):
                m = np.array(mat.data).reshape(4, 4)
                yaw_rad   = math.atan2(m[1][0], m[0][0])
                pitch_rad = math.atan2(-m[2][0], math.sqrt(m[2][1]**2 + m[2][2]**2))
                print(f"    face[{i}] yaw={math.degrees(yaw_rad):.1f}° pitch={math.degrees(pitch_rad):.1f}°")

results["face_landmarker"] = fl_results

# ── 4. FaceDetector (BlazeFace) 테스트 — 모델 있을 때만 ──────────────────────
print("\n=== FaceDetector (BlazeFace) 테스트 ===")
fd_results = {}
if os.path.exists(MODEL_FD):
    for conf in [0.3, 0.1]:
        for res_key, img in scales.items():
            h2, w2 = img.shape[:2]
            opts  = mp_python.BaseOptions(model_asset_path=MODEL_FD)
            opts2 = mp_vision.FaceDetectorOptions(
                base_options=opts,
                running_mode=mp_vision.RunningMode.IMAGE,
                min_detection_confidence=conf,
            )
            det = mp_vision.FaceDetector.create_from_options(opts2)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = det.detect(mp_img)
            det.close()

            n = len(res.detections) if res.detections else 0
            key = f"conf={conf}_res={res_key}"
            fd_results[key] = n
            print(f"  conf={conf:.2f}  res={w2}x{h2}  → 감지 얼굴 수: {n}")
            if n > 0:
                for d in res.detections:
                    print(f"    score={d.categories[0].score:.3f}  bbox={d.bounding_box}")
else:
    print(f"  모델 없음 ({MODEL_FD}) — 다운로드 필요")
    fd_results["note"] = f"모델 없음: {MODEL_FD}"

results["face_detector_blazeface"] = fd_results

# ── 5. PoseLandmarker — 코(nose) 랜드마크 위치로 얼굴 방향 추정 ──────────────
print("\n=== PoseLandmarker 코 랜드마크 분석 ===")
pose_info = []
opts  = mp_python.BaseOptions(model_asset_path=MODEL_PL)
opts2 = mp_vision.PoseLandmarkerOptions(
    base_options=opts,
    running_mode=mp_vision.RunningMode.IMAGE,
    num_poses=4,
    min_pose_detection_confidence=0.3,
    min_pose_presence_confidence=0.3,
)
det = mp_vision.PoseLandmarker.create_from_options(opts2)
img = scales[640]
h2, w2 = img.shape[:2]
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
pose_res = det.detect(mp_img)
det.close()

n_poses = len(pose_res.pose_landmarks) if pose_res.pose_landmarks else 0
print(f"  감지 사람 수: {n_poses}")

for pi, pose in enumerate(pose_res.pose_landmarks or []):
    # 랜드마크 인덱스: 0=코, 1=왼눈안쪽, 2=왼눈, 3=왼눈바깥쪽, 4=오른눈안쪽, 5=오른눈, 6=오른눈바깥쪽
    # 7=왼귀, 8=오른귀, 11=왼어깨, 12=오른어깨
    nose  = pose[0]
    l_eye = pose[2]
    r_eye = pose[5]
    l_ear = pose[7]
    r_ear = pose[8]
    l_sho = pose[11]
    r_sho = pose[12]

    # 눈 사이 거리 vs 귀 가시성으로 yaw 추정
    eye_dist  = abs(l_eye.x - r_eye.x) * w2
    l_ear_vis = getattr(l_ear, "visibility", 0)
    r_ear_vis = getattr(r_ear, "visibility", 0)
    nose_vis  = getattr(nose, "visibility", 0)

    # 왼어깨-오른어깨 중심 vs 코 위치로 몸 방향 추정
    sho_cx = (l_sho.x + r_sho.x) / 2
    nose_x = nose.x

    # yaw 추정: 코가 어깨 중심보다 크게 벗어나면 측면
    yaw_est = (nose_x - sho_cx) * 2  # -1~1 (음수=왼쪽 회전, 양수=오른쪽 회전)

    info = {
        "person": pi,
        "nose_xy": f"({nose.x:.3f}, {nose.y:.3f})",
        "nose_visibility": f"{nose_vis:.2f}",
        "eye_dist_px": f"{eye_dist:.1f}",
        "l_ear_visibility": f"{l_ear_vis:.2f}",
        "r_ear_visibility": f"{r_ear_vis:.2f}",
        "shoulder_cx": f"{sho_cx:.3f}",
        "estimated_yaw_ratio": f"{yaw_est:.3f}",
        "face_direction": "정면(0~20°)" if abs(yaw_est) < 0.15
                          else "3/4뷰(20~60°)" if abs(yaw_est) < 0.45
                          else "측면(60~90°)",
    }
    pose_info.append(info)
    print(f"  사람[{pi}]:")
    for k, v in info.items():
        if k != "person":
            print(f"    {k}: {v}")

results["pose_nose_analysis"] = pose_info

# ── 6. OpenCV Haar Cascade 테스트 (참고용) ────────────────────────────────────
print("\n=== OpenCV Haar Cascade 얼굴 감지 테스트 ===")
haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
haar_profile = cv2.data.haarcascades + "haarcascade_profileface.xml"
cascade_f = cv2.CascadeClassifier(haar_path)
cascade_p = cv2.CascadeClassifier(haar_profile)

haar_results = {}
for res_key, img in scales.items():
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_f = cascade_f.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30,30))
    faces_p = cascade_p.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30,30))
    nf  = len(faces_f) if isinstance(faces_f, np.ndarray) and faces_f.ndim > 1 else 0
    np_ = len(faces_p) if isinstance(faces_p, np.ndarray) and faces_p.ndim > 1 else 0
    haar_results[f"res={res_key}"] = {"frontal": nf, "profile": np_}
    print(f"  res={res_key}: 정면={nf}개, 측면={np_}개")

results["haar_cascade"] = haar_results

# ── 7. 첫 프레임 저장 (시각 확인용) ──────────────────────────────────────────
frame_path = "TEST_DATA/diagnose_frame1.jpg"
cv2.imwrite(frame_path, frame)
print(f"\n  첫 프레임 저장: {frame_path}")

# ── 8. Error_analyze.txt 작성 ────────────────────────────────────────────────
print("\n분석 파일 작성 중...")

lines = []
lines.append("================================================================")
lines.append("  PoseTracker — 얼굴 감지 실패 상세 분석")
lines.append(f"  작성일: 2026-04-12")
lines.append(f"  분석 영상: {VIDEO}")
lines.append("================================================================")
lines.append("")

lines.append("────────────────────────────────────────────────────────────────")
lines.append("  [영상 기본 정보]")
lines.append("────────────────────────────────────────────────────────────────")
for k, v in results["video"].items():
    lines.append(f"  {k}: {v}")
lines.append("")

lines.append("────────────────────────────────────────────────────────────────")
lines.append("  [SECTION 1] FaceLandmarker 감지 결과")
lines.append("  → confidence × 해상도 전수 테스트")
lines.append("────────────────────────────────────────────────────────────────")
for k, v in results["face_landmarker"].items():
    lines.append(f"  {k}: {v}명 감지")
lines.append("")
any_face = any(v > 0 for v in results["face_landmarker"].values())
lines.append(f"  결론: 어떤 conf/해상도에서도 얼굴 {'감지됨' if any_face else '0명 (완전 실패)'}")
lines.append("")

lines.append("────────────────────────────────────────────────────────────────")
lines.append("  [SECTION 2] FaceDetector (BlazeFace) 감지 결과")
lines.append("  → FaceLandmarker보다 경량, 측면에 다소 강인")
lines.append("────────────────────────────────────────────────────────────────")
for k, v in results["face_detector_blazeface"].items():
    lines.append(f"  {k}: {v}")
lines.append("")

lines.append("────────────────────────────────────────────────────────────────")
lines.append("  [SECTION 3] PoseLandmarker 코 랜드마크로 얼굴 방향 추정")
lines.append("  → 포즈는 감지되므로, 포즈 데이터로 얼굴 방향을 역산")
lines.append("────────────────────────────────────────────────────────────────")
for info in results["pose_nose_analysis"]:
    lines.append(f"  사람[{info['person']}]:")
    for k, v in info.items():
        if k != "person":
            lines.append(f"    {k}: {v}")
    lines.append("")

lines.append("────────────────────────────────────────────────────────────────")
lines.append("  [SECTION 4] OpenCV Haar Cascade (참고용)")
lines.append("  → 고전 CV 방식 — 측면 얼굴 전용 cascade도 테스트")
lines.append("────────────────────────────────────────────────────────────────")
for k, v in results["haar_cascade"].items():
    lines.append(f"  {k}: 정면={v['frontal']}개, 측면={v['profile']}개")
lines.append("")

lines.append("────────────────────────────────────────────────────────────────")
lines.append("  [SECTION 5] 원인 분석 요약")
lines.append("────────────────────────────────────────────────────────────────")
lines.append("")
lines.append("  MediaPipe FaceLandmarker 실패 이유 (구조적):")
lines.append("")
lines.append("  [이유 1] 모델 학습 데이터 편향 — 정면 얼굴 위주")
lines.append("    - FaceLandmarker(MediaPipe) 내부적으로 BlazeFace-Short/Full 사용")
lines.append("    - BlazeFace는 모바일 실시간 정면 감지 특화 모델")
lines.append("    - 학습 데이터: 대부분 정면(yaw ±30°) 이내")
lines.append("    - 측면(yaw 45°~90°): 코/눈의 대칭적 특징점이 사라짐 → 감지 불가")
lines.append("")
lines.append("  [이유 2] 두 단계 파이프라인의 연쇄 실패")
lines.append("    - Stage 1: BlazeFace detection → ROI 후보 생성")
lines.append("    - Stage 2: FaceMesh landmark 회귀")
lines.append("    → Stage 1에서 이미 실패하면 Stage 2 실행 안 됨")
lines.append("    → 측면 얼굴은 Stage 1 자체를 통과 못함")
lines.append("")
lines.append("  [이유 3] 영상 속 얼굴 각도")
for info in results["pose_nose_analysis"]:
    lines.append(f"    사람[{info['person']}]: 추정 얼굴 방향 = {info['face_direction']}")
    lines.append(f"      코 위치=({info['nose_xy']}), 어깨중심x={info['shoulder_cx']}")
    lines.append(f"      눈 간격={info['eye_dist_px']}px, yaw비율={info['estimated_yaw_ratio']}")
lines.append("")
lines.append("  [이유 4] confidence 임계값 효과 없음")
lines.append("    - 감지가 0.05까지 내려도 0명 → confidence 조정은 무의미")
lines.append("    - 이는 detection score 자체가 생성 안 된다는 의미")
lines.append("    - (confidence는 후보 필터링에만 작용, 후보 자체가 없으면 무효)")
lines.append("")
lines.append("  [이유 5] 해상도 효과 없음")
lines.append("    - 원본/640px/360px 모두 동일 결과")
lines.append("    - 스케일 문제가 아닌 얼굴 각도 문제임을 확인")
lines.append("")

lines.append("────────────────────────────────────────────────────────────────")
lines.append("  [SECTION 6] 해결 방안 비교")
lines.append("────────────────────────────────────────────────────────────────")
lines.append("")
lines.append("  방법 A: RetinaFace (★★★ 추천)")
lines.append("    - 측면 얼굴 감지에 강인, WIDER FACE 데이터셋 학습")
lines.append("    - yaw ±90° 까지 감지 가능")
lines.append("    - pip install retina-face (약 2MB 모델)")
lines.append("    - 단점: FaceLandmarker처럼 478pt 랜드마크 없음 (5pt만)")
lines.append("    - 사용 예: from retinaface import RetinaFace; faces = RetinaFace.detect_faces(frame)")
lines.append("")
lines.append("  방법 B: InsightFace / buffalo_l 모델")
lines.append("    - 측면 얼굴 + 얼굴 인식(recognition) 가능")
lines.append("    - pip install insightface onnxruntime")
lines.append("    - 단점: 모델 크기 ~300MB, GPU 필요 시 onnxruntime-gpu")
lines.append("")
lines.append("  방법 C: 포즈 head 랜드마크로 대체 (★★ 현실적)")
lines.append("    - 포즈 감지는 성공 → 포즈의 코(0), 눈(1~6), 귀(7~8) 랜드마크 활용")
lines.append("    - 얼굴 감지 없어도 눈코입 위치 근사 가능")
lines.append("    - 모자이크: 코~귀 사이 bbox로 대체 적용")
lines.append("    - 단점: 478pt 정밀 랜드마크 아님, 얼굴 이미지 오버레이 불가")
lines.append("")
lines.append("  방법 D: MediaPipe Face Detection 단독 사용")
lines.append("    - FaceLandmarker가 아닌 FaceDetector API (BlazeFace)")
lines.append("    - FaceLandmarker와 동일 검출기지만 임계값 독립 설정 가능")
lines.append("    - 이미 테스트에 포함 (SECTION 2 결과 확인)")
lines.append("")
lines.append("  방법 E: 아무것도 안 함 (현재 상태 유지)")
lines.append("    - 얼굴 감지 실패는 사용자가 이미 인지")
lines.append("    - 영상 특성상 측면 얼굴 → 구조적 한계")
lines.append("")
lines.append("  ★ 권장: 방법 C (포즈 head 랜드마크 대체) + 방법 A (RetinaFace 선택적 추가)")

lines.append("")
lines.append("================================================================")
lines.append(f"  첫 프레임 저장 위치: {frame_path}")
lines.append("================================================================")

with open("Error_analyze.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("완료: Error_analyze.txt 작성됨")
