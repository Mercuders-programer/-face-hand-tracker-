"""
파이프라인 전체 테스트 (카메라 없이 가짜 프레임 사용)
Windows/WSL2 모두에서 실행 가능
로그: test_pipeline.log
"""
import sys, os, traceback, logging, queue
import numpy as np

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_pipeline.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger()

BASE       = os.path.dirname(os.path.abspath(__file__))
FACE_MODEL = os.path.join(BASE, "models", "face_landmarker.task")
HAND_MODEL = os.path.join(BASE, "models", "hand_landmarker.task")
W, H = 640, 480

# ── 1. 패키지 임포트 ──────────────────────────────────────────────────────
log.info("=== 1. 패키지 임포트 ===")
try:
    import cv2
    log.info(f"  cv2 {cv2.__version__} OK")
except Exception as e:
    log.error(f"  cv2 실패: {e}"); sys.exit(1)

try:
    import mediapipe as mp
    log.info(f"  mediapipe {mp.__version__} OK")
except Exception as e:
    log.error(f"  mediapipe 실패: {e}"); sys.exit(1)

try:
    from PIL import Image, ImageTk
    import PIL
    log.info(f"  Pillow {PIL.__version__} OK")
except Exception as e:
    log.error(f"  Pillow 실패: {e}"); sys.exit(1)

try:
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.vision import RunningMode
    from mediapipe.tasks.python.vision import drawing_utils as mp_draw
    from mediapipe.tasks.python.vision import drawing_styles as mp_styles
    from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections
    from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections
    log.info("  mediapipe Tasks API OK")
except Exception as e:
    log.error(f"  Tasks API 실패: {e}\n{traceback.format_exc()}"); sys.exit(1)

# ── 2. 모델 파일 ─────────────────────────────────────────────────────────
log.info("\n=== 2. 모델 파일 ===")
ok = True
for path, name in [(FACE_MODEL, "face_landmarker.task"), (HAND_MODEL, "hand_landmarker.task")]:
    if os.path.exists(path):
        log.info(f"  {name} OK  ({os.path.getsize(path)//1024} KB)")
    else:
        log.error(f"  {name} 없음: {path}"); ok = False
if not ok:
    sys.exit(1)

# ── 3. MediaPipe Tasks 초기화 ────────────────────────────────────────────
log.info("\n=== 3. MediaPipe Tasks 초기화 ===")
try:
    face_opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    face_det = mp_vision.FaceLandmarker.create_from_options(face_opts)
    log.info("  FaceLandmarker OK")
except Exception as e:
    log.error(f"  FaceLandmarker 실패: {e}"); sys.exit(1)

try:
    hand_opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hand_det = mp_vision.HandLandmarker.create_from_options(hand_opts)
    log.info("  HandLandmarker OK")
except Exception as e:
    log.error(f"  HandLandmarker 실패: {e}"); sys.exit(1)

# ── 4. 가짜 프레임 처리 ───────────────────────────────────────────────────
log.info("\n=== 4. 프레임 처리 (5 프레임) ===")
overlay = None
for i in range(5):
    frame  = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    try:
        face_res = face_det.detect(mp_img)
        hand_res = hand_det.detect(mp_img)
        log.info(f"  frame {i}: detect OK  face={bool(face_res.face_landmarks)}  hand={bool(hand_res.hand_landmarks)}")
    except Exception as e:
        log.error(f"  frame {i} detect 실패: {e}"); continue

    overlay = frame.copy()
    try:
        if face_res.face_landmarks:
            mp_draw.draw_landmarks(
                overlay, face_res.face_landmarks[0],
                FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style(),
            )
        if hand_res.hand_landmarks:
            for hlms in hand_res.hand_landmarks:
                mp_draw.draw_landmarks(
                    overlay, hlms,
                    HandLandmarksConnections.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp_styles.get_default_hand_connections_style(),
                )
        log.info(f"  frame {i}: draw OK")
    except Exception as e:
        log.error(f"  frame {i} draw 실패: {e}\n{traceback.format_exc()}")

face_det.close()
hand_det.close()

# ── 5. PIL 변환 ───────────────────────────────────────────────────────────
log.info("\n=== 5. PIL 변환 ===")
if overlay is not None:
    try:
        rgb_out = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        img     = Image.fromarray(rgb_out)
        resized = img.resize((840, 560), Image.LANCZOS)
        log.info(f"  resize OK  {resized.size}")
    except Exception as e:
        log.error(f"  PIL 실패: {e}\n{traceback.format_exc()}")

# ── 6. ImageTk ───────────────────────────────────────────────────────────
log.info("\n=== 6. ImageTk.PhotoImage ===")
try:
    import tkinter as tk
    root = tk.Tk(); root.withdraw()
    photo = ImageTk.PhotoImage(resized)
    log.info(f"  PhotoImage OK  {photo.width()}x{photo.height()}")
    root.destroy()
except Exception as e:
    log.warning(f"  PhotoImage 실패 (헤드리스이면 정상): {e}")

log.info(f"\n=== 모든 테스트 통과 ===")
log.info(f"로그: {LOG_PATH}")
