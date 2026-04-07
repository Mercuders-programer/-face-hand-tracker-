"""
camera_panel.py — 카메라 추적 UI 패널

기능:
  - 웹캠 실시간 피드 + 얼굴/손 오버레이
  - 30 / 60 fps 선택
  - 녹화 → 저장 경로 선택 → 영상 저장 + AE 키프레임 자동 내보내기
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import cv2
import mediapipe as mp
import os
import tempfile
import shutil

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode
from mediapipe.tasks.python.vision import drawing_utils as mp_draw
from mediapipe.tasks.python.vision import drawing_styles as mp_styles
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections

_BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACE_MODEL  = os.path.join(_BASE, "models", "face_landmarker.task")
HAND_MODEL  = os.path.join(_BASE, "models", "hand_landmarker.task")
POSE_MODEL  = os.path.join(_BASE, "models", "pose_landmarker_full.task")

try:
    from PIL import Image, ImageTk
except ImportError:
    raise ImportError("Pillow가 필요합니다: pip install Pillow")

try:
    from .tracker import (Tracker, FrameData, VideoInfo, PersonData,
                          _extract_face, _extract_hand, _extract_pose,
                          _build_persons, MAX_PERSONS, PERSON_COLORS)
    from .exporter import export_json, export_ae_keyframes
except ImportError:
    from tracker import (Tracker, FrameData, VideoInfo, PersonData,
                         _extract_face, _extract_hand, _extract_pose,
                         _build_persons, MAX_PERSONS, PERSON_COLORS)
    from exporter import export_json, export_ae_keyframes


# ── 테마 색상 ──────────────────────────────────────────────────────────────
BG_DARK   = "#1a1a2e"
BG_PANEL  = "#16213e"
BG_CTRL   = "#0f3460"
ACCENT    = "#4a7fff"
ACCENT_R  = "#e94560"
TEXT_W    = "#e0e0ff"
TEXT_G    = "#8888aa"

CANVAS_W  = 840
CANVAS_H  = 560
CTRL_W    = 230


def _draw_landmark_names(overlay, face_res, hand_res, pose_res,
                          w, h, show_face, show_body, show_hands):
    """랜드마크 포인트 이름을 overlay 이미지에 렌더링한다."""

    def _text(img, label, x, y, color):
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.rectangle(img, (x - 1, y - th - 2), (x + tw + 2, y + 2), (0, 0, 0), -1)
        cv2.putText(img, label, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    # ── 얼굴 주요 12포인트 이름 (감지된 모든 사람)
    if show_face and face_res.face_landmarks:
        _fc = (0, 230, 180)
        for _lf in face_res.face_landmarks:
            for _idx, _lbl in [
                (33,  "R.Eye.O"), (133, "R.Eye.I"), (473, "R.Iris"),
                (362, "L.Eye.I"), (263, "L.Eye.O"), (468, "L.Iris"),
                (168, "Nose.B"),  (4,   "Nose.T"),
                (61,  "Mouth.R"), (13,  "Mouth.U"), (291, "Mouth.L"), (14, "Mouth.D"),
            ]:
                if _idx < len(_lf):
                    _text(overlay, _lbl,
                          int(_lf[_idx].x * w) + 4,
                          int(_lf[_idx].y * h) - 4, _fc)

    # ── 포즈 주요 관절 이름 (감지된 모든 사람)
    if show_body and pose_res and pose_res.pose_landmarks:
        for _pl in pose_res.pose_landmarks:
            for _idx, _lbl in [
                (11, "L.Shldr"), (12, "R.Shldr"),
                (13, "L.Elbow"), (14, "R.Elbow"),
                (15, "L.Wrist"), (16, "R.Wrist"),
                (23, "L.Hip"),   (24, "R.Hip"),
                (25, "L.Knee"),  (26, "R.Knee"),
                (27, "L.Ankle"), (28, "R.Ankle"),
            ]:
                if _idx < len(_pl) and _pl[_idx].visibility > 0.3:
                    _col = (255, 160, 50) if _lbl.startswith("L.") else (50, 160, 255)
                    _text(overlay, _lbl,
                          int(_pl[_idx].x * w) + 7,
                          int(_pl[_idx].y * h) - 7, _col)

    # ── 손 주요 6포인트 이름
    if show_hands and hand_res.hand_landmarks:
        _hc = (255, 220, 100)
        for _hlms in hand_res.hand_landmarks:
            for _idx, _lbl in [
                (0, "Wrist"), (4, "Thumb"),
                (8, "Index"), (12, "Middle"),
                (16, "Ring"), (20, "Pinky"),
            ]:
                if _idx < len(_hlms):
                    _text(overlay, _lbl,
                          int(_hlms[_idx].x * w) + 4,
                          int(_hlms[_idx].y * h) - 4, _hc)


class CameraPanel:
    def __init__(self, parent: tk.Tk):
        self.win = tk.Toplevel(parent)
        self.win.title("PoseTracker — 카메라 추적")
        self.win.geometry(f"{CANVAS_W + CTRL_W + 36}x{CANVAS_H + 24}")
        self.win.minsize(900, 500)
        self.win.configure(bg=BG_DARK)
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── 상태 ──────────────────────────────────────────────────────────
        self._running    = False
        self._recording  = False
        self._fps        = 30          # 캡처 스레드에서 사용할 정수값
        self._fps_var    = tk.IntVar(value=30)
        self._show_face  = tk.BooleanVar(value=True)
        self._show_body  = tk.BooleanVar(value=True)
        self._show_hands = tk.BooleanVar(value=True)
        self._show_names = tk.BooleanVar(value=False)
        self._status_var = tk.StringVar(value="대기중")
        self._frame_q: queue.Queue = queue.Queue(maxsize=2)
        self._frames_data: list[FrameData] = []
        self._cap         = None
        self._cap_thread  = None
        self._is_file_mode = False
        self._writer     = None        # cv2.VideoWriter
        self._tmp_path   = None
        self._after_id   = None
        self._photo      = None        # GC 방지용 PhotoImage 참조
        self._cam_w      = 0
        self._cam_h      = 0
        self._tracker    = Tracker()

        self._build_ui()
        # 패널 열리면 자동으로 카메라 시작
        self.win.after(200, self._start_camera)

    # ── UI 구성 ────────────────────────────────────────────────────────────
    def _build_ui(self):
        # 좌측: 카메라 캔버스
        left = tk.Frame(self.win, bg=BG_DARK)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(12, 6), pady=12)

        self._canvas = tk.Canvas(
            left, bg="#000011",
            highlightthickness=1, highlightbackground="#333355",
        )
        self._canvas.pack(fill=tk.BOTH, expand=True)
        self._draw_placeholder()

        # 우측: 컨트롤 패널
        right = tk.Frame(self.win, bg=BG_PANEL, width=CTRL_W)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(6, 12), pady=12)
        right.pack_propagate(False)

        # ── 프레임 레이트 ──
        self._section_label(right, "프레임 레이트")
        fps_row = tk.Frame(right, bg=BG_PANEL)
        fps_row.pack(pady=(0, 6))
        for val in (30, 60):
            tk.Radiobutton(
                fps_row,
                text=f"{val} fps",
                variable=self._fps_var, value=val,
                fg=TEXT_W, bg=BG_PANEL,
                selectcolor=BG_CTRL,
                activeforeground=TEXT_W, activebackground=BG_PANEL,
                font=("Segoe UI", 11),
                command=self._on_fps_change,
            ).pack(side=tk.LEFT, padx=10)

        self._separator(right)

        # ── 랜드마크 표시 ──
        self._section_label(right, "오버레이")
        for _var, _lbl in [
            (self._show_face,  "얼굴  (눈·코·입)"),
            (self._show_body,  "몸  (몸통·팔·다리)"),
            (self._show_hands, "손  (손가락·손바닥)"),
        ]:
            tk.Checkbutton(
                right, text=_lbl, variable=_var,
                font=("Segoe UI", 10),
                fg=TEXT_W, bg=BG_PANEL,
                selectcolor=BG_CTRL,
                activeforeground=TEXT_W, activebackground=BG_PANEL,
                anchor=tk.W,
            ).pack(fill=tk.X, padx=8, pady=(0, 2))
        tk.Checkbutton(
            right, text="랜드마크 이름",
            variable=self._show_names,
            font=("Segoe UI", 10),
            fg="#ffdd88", bg=BG_PANEL,
            selectcolor=BG_CTRL,
            activeforeground="#ffdd88", activebackground=BG_PANEL,
            anchor=tk.W,
        ).pack(fill=tk.X, padx=8, pady=(4, 2))

        self._separator(right)

        # ── 카메라 버튼 ──
        self._cam_btn = self._make_btn(
            right, "카메라 시작", ACCENT,
            command=self._toggle_camera,
        )

        self._separator(right)

        # ── 녹화 버튼 ──
        self._rec_btn = self._make_btn(
            right, "녹화 시작", "#333355",
            fg=TEXT_G, state=tk.DISABLED,
            command=self._toggle_record,
        )

        self._separator(right)

        # ── 상태 표시 ──
        self._section_label(right, "상태")
        tk.Label(
            right,
            textvariable=self._status_var,
            font=("Segoe UI", 10),
            fg="#4aff9e", bg=BG_PANEL,
            wraplength=CTRL_W - 20,
            justify=tk.CENTER,
        ).pack(pady=4)

        # REC 깜빡임 표시
        self._rec_ind = tk.Label(
            right, text="",
            font=("Segoe UI", 12, "bold"),
            fg=ACCENT_R, bg=BG_PANEL,
        )
        self._rec_ind.pack(pady=2)

    def _section_label(self, parent, text: str):
        tk.Label(
            parent, text=text,
            font=("Segoe UI", 10, "bold"),
            fg=TEXT_G, bg=BG_PANEL,
        ).pack(pady=(14, 4))

    def _separator(self, parent):
        ttk.Separator(parent, orient="horizontal").pack(
            fill=tk.X, pady=8, padx=12,
        )

    def _make_btn(self, parent, text, bg,
                  fg="white", state=tk.NORMAL, command=None):
        b = tk.Button(
            parent, text=text,
            font=("Segoe UI", 11, "bold"),
            bg=bg, fg=fg,
            activebackground=bg, activeforeground=fg,
            relief=tk.FLAT, cursor="hand2",
            width=18, height=2,
            state=state, command=command,
        )
        b.pack(pady=4)
        return b

    def _draw_placeholder(self):
        self._canvas.update_idletasks()
        w = self._canvas.winfo_width()  or CANVAS_W
        h = self._canvas.winfo_height() or CANVAS_H
        self._canvas.delete("all")
        self._canvas.create_text(
            w // 2, h // 2,
            text="카메라를 시작하세요",
            fill="#444466", font=("Segoe UI", 18),
        )

    # ── FPS 변경 ──────────────────────────────────────────────────────────
    def _on_fps_change(self):
        self._fps = self._fps_var.get()
        if self._running and self._cap:
            self._cap.set(cv2.CAP_PROP_FPS, self._fps)

    # ── 카메라 토글 ────────────────────────────────────────────────────────
    def _toggle_camera(self):
        if not self._running:
            self._start_camera()
        else:
            self._stop_camera()

    def _start_camera(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap.release()
            # 카메라 없음 → 영상 파일로 대체 여부 묻기
            from tkinter import filedialog
            ans = messagebox.askyesno(
                "카메라 없음",
                "웹캠을 찾을 수 없습니다.\n\n"
                "영상 파일로 대신 실행하시겠습니까?\n"
                "(WSL2 환경에서는 영상 파일을 사용하세요)",
                parent=self.win,
            )
            if not ans:
                return
            path = filedialog.askopenfilename(
                parent=self.win,
                title="영상 파일 선택",
                filetypes=[("영상 파일", "*.mp4 *.avi *.mov *.mkv"), ("모든 파일", "*.*")],
            )
            if not path:
                return
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                messagebox.showerror("오류", f"파일을 열 수 없습니다:\n{path}", parent=self.win)
                return
            self._is_file_mode = True
        else:
            self._is_file_mode = False

        self._fps = self._fps_var.get()
        cap.set(cv2.CAP_PROP_FPS, self._fps)
        self._cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._cap     = cap
        self._running = True

        self._cam_btn.config(text="카메라 중지", bg=ACCENT_R)
        self._rec_btn.config(
            state=tk.NORMAL, bg=ACCENT_R, fg="white", text="녹화 시작",
        )
        self._status_var.set("카메라 실행중")

        self._cap_thread = threading.Thread(
            target=self._capture_loop, daemon=True,
        )
        self._cap_thread.start()
        self._schedule_display()

    def _stop_camera(self):
        if self._recording:
            self._stop_record(cancelled=True)

        self._running = False

        if self._after_id:
            self.win.after_cancel(self._after_id)
            self._after_id = None

        if self._cap:
            self._cap.release()
            self._cap = None

        self._cam_btn.config(text="카메라 시작", bg=ACCENT)
        self._rec_btn.config(
            state=tk.DISABLED, bg="#333355", fg=TEXT_G, text="녹화 시작",
        )
        self._status_var.set("대기중")
        self._rec_ind.config(text="")
        self._draw_placeholder()

    # ── 녹화 토글 ──────────────────────────────────────────────────────────
    def _toggle_record(self):
        if not self._recording:
            self._start_record()
        else:
            self._stop_record()

    def _start_record(self):
        self._tmp_path = os.path.join(
            tempfile.gettempdir(), "_posetracker_rec.mp4",
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            self._tmp_path, fourcc, self._fps,
            (self._cam_w, self._cam_h),
        )
        self._frames_data = []
        self._recording = True
        self._rec_btn.config(text="녹화 중지", bg="#cc2222")
        self._status_var.set(f"녹화중  ({self._fps} fps)")
        self._blink()

    def _stop_record(self, cancelled: bool = False):
        self._recording = False
        if self._writer:
            self._writer.release()
            self._writer = None
        self._rec_btn.config(text="녹화 시작", bg=ACCENT_R)
        self._rec_ind.config(text="")

        if cancelled:
            self._cleanup_tmp()
            return

        self._status_var.set("저장 경로 선택...")

        save_path = filedialog.asksaveasfilename(
            parent=self.win,
            title="영상 저장 위치 선택",
            defaultextension=".mp4",
            filetypes=[("MP4 파일", "*.mp4"), ("모든 파일", "*.*")],
        )

        if not save_path:
            self._cleanup_tmp()
            self._status_var.set("카메라 실행중")
            return

        # 임시 파일 → 지정 경로로 이동
        shutil.move(self._tmp_path, save_path)
        self._tmp_path = None

        # AE 내보내기 (별도 스레드)
        out_dir   = os.path.dirname(save_path) or "."
        json_path = os.path.join(out_dir, "tracking_data.json")
        ae_dir    = os.path.join(out_dir, "ae_keyframes")
        fps       = self._fps
        frames    = list(self._frames_data)
        cam_w, cam_h = self._cam_w, self._cam_h
        inc_face  = self._show_face.get()
        inc_body  = self._show_body.get()
        inc_hands = self._show_hands.get()

        self._status_var.set("AE 데이터 내보내는 중...")

        def _export():
            if frames:
                info = VideoInfo(
                    width=cam_w, height=cam_h,
                    fps=fps, total_frames=len(frames),
                )
                export_json(frames, info, json_path,
                            include_face=inc_face, include_body=inc_body, include_hands=inc_hands)
                export_ae_keyframes(frames, info, ae_dir,
                                    include_face=inc_face, include_body=inc_body, include_hands=inc_hands)

            def _done():
                self._status_var.set(f"저장 완료! ({len(frames)} 프레임)")
                messagebox.showinfo(
                    "저장 완료",
                    f"영상:  {save_path}\n"
                    f"JSON:  {json_path}\n"
                    f"AE 키프레임:  {ae_dir}/",
                    parent=self.win,
                )
                self._status_var.set("카메라 실행중")

            self.win.after(0, _done)

        threading.Thread(target=_export, daemon=True).start()

    def _cleanup_tmp(self):
        if self._tmp_path and os.path.exists(self._tmp_path):
            os.remove(self._tmp_path)
        self._tmp_path = None

    def _blink(self):
        if not self._recording:
            return
        cur = self._rec_ind.cget("text")
        self._rec_ind.config(text="" if cur else "● REC")
        self.win.after(500, self._blink)

    # ── 캡처 루프 (백그라운드 스레드) ────────────────────────────────────
    def _capture_loop(self):
        face_opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL),
            running_mode=RunningMode.IMAGE,
            num_faces=MAX_PERSONS,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        hand_opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL),
            running_mode=RunningMode.IMAGE,
            num_hands=MAX_PERSONS * 2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        pose_opts = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL),
            running_mode=RunningMode.IMAGE,
            num_poses=MAX_PERSONS,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        try:
            face_det = mp_vision.FaceLandmarker.create_from_options(face_opts)
            hand_det = mp_vision.HandLandmarker.create_from_options(hand_opts)
            pose_det = mp_vision.PoseLandmarker.create_from_options(pose_opts)
        except Exception as e:
            print(f"[MediaPipe init error] {e}")
            self._capture_loop_raw()
            return

        try:
            while self._running and self._cap and self._cap.isOpened():
                ret, frame = self._cap.read()
                if not ret:
                    break

                h_px, w_px = frame.shape[:2]
                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                try:
                    face_res = face_det.detect(mp_img)
                    hand_res = hand_det.detect(mp_img)
                    pose_res = pose_det.detect(mp_img)
                except Exception as e:
                    print(f"[detect error] {e}")
                    _empty = type('R', (), {
                        'face_landmarks': [], 'hand_landmarks': [],
                        'handedness': [], 'pose_landmarks': []
                    })()
                    face_res = hand_res = pose_res = _empty

                # 오버레이 그리기
                overlay = frame.copy()
                # ── 포즈 스켈레톤 (모든 감지된 사람, 먼저 그려 얼굴/손 위에 덮이지 않게)
                if self._show_body.get() and pose_res.pose_landmarks:
                    _SKEL = [(11,12),(11,23),(12,24),(23,24),
                             (11,13),(13,15),(12,14),(14,16),
                             (23,25),(25,27),(24,26),(26,28)]
                    for _pidx, _pl in enumerate(pose_res.pose_landmarks):
                        _pc = PERSON_COLORS[_pidx % len(PERSON_COLORS)]
                        for _s, _e in _SKEL:
                            if (_s < len(_pl) and _e < len(_pl)
                                    and _pl[_s].visibility > 0.3
                                    and _pl[_e].visibility > 0.3):
                                cv2.line(overlay,
                                         (int(_pl[_s].x*w_px), int(_pl[_s].y*h_px)),
                                         (int(_pl[_e].x*w_px), int(_pl[_e].y*h_px)),
                                         _pc, 2)
                        for _i in [11,12,13,14,15,16,23,24,25,26,27,28]:
                            if _i < len(_pl) and _pl[_i].visibility > 0.3:
                                cv2.circle(overlay,
                                           (int(_pl[_i].x*w_px), int(_pl[_i].y*h_px)),
                                           6, _pc, -1)
                        # 사람 번호 레이블
                        if 0 < len(_pl) and _pl[0].visibility > 0.3:
                            _lx = max(5, int(_pl[0].x*w_px) - 12)
                            _ly = max(20, int(_pl[0].y*h_px) - 20)
                            cv2.putText(overlay, f"P{_pidx+1}", (_lx, _ly),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, _pc, 2, cv2.LINE_AA)
                # ── 얼굴 (모든 감지된 사람)
                if self._show_face.get() and face_res.face_landmarks:
                    _nc = (0, 230, 180)
                    for _lf in face_res.face_landmarks:
                        mp_draw.draw_landmarks(
                            overlay, _lf,
                            FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style(),
                        )
                        for _s, _e in [(168,6),(6,197),(197,195),(195,5),(5,4),
                                       (4,1),(1,19),(98,97),(97,2),(2,326),(326,327)]:
                            if _s < len(_lf) and _e < len(_lf):
                                cv2.line(overlay,
                                         (int(_lf[_s].x*w_px), int(_lf[_s].y*h_px)),
                                         (int(_lf[_e].x*w_px), int(_lf[_e].y*h_px)),
                                         _nc, 1)
                        for _i in [1,2,4,5,6,19,97,98,168,195,197,326,327]:
                            if _i < len(_lf):
                                cv2.circle(overlay,
                                           (int(_lf[_i].x*w_px), int(_lf[_i].y*h_px)),
                                           2, _nc, -1)
                # ── 손
                if self._show_hands.get() and hand_res.hand_landmarks:
                    for hlms in hand_res.hand_landmarks:
                        mp_draw.draw_landmarks(
                            overlay, hlms,
                            HandLandmarksConnections.HAND_CONNECTIONS,
                            landmark_drawing_spec=mp_styles.get_default_hand_landmarks_style(),
                            connection_drawing_spec=mp_styles.get_default_hand_connections_style(),
                        )
                # ── 랜드마크 이름
                if self._show_names.get():
                    _draw_landmark_names(overlay, face_res, hand_res, pose_res,
                                         w_px, h_px,
                                         self._show_face.get(),
                                         self._show_body.get(),
                                         self._show_hands.get())

                # 녹화 처리
                if self._recording:
                    if self._writer:
                        self._writer.write(overlay)
                    self._collect_frame(face_res, hand_res, pose_res, w_px, h_px)

                # 디스플레이 큐에 넣기 (BGR → RGB)
                try:
                    self._frame_q.put_nowait(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
                except queue.Full:
                    pass
        finally:
            face_det.close()
            hand_det.close()
            pose_det.close()

    def _capture_loop_raw(self):
        """MediaPipe 없이 원본 영상만 표시"""
        while self._running and self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret:
                break
            if self._recording and self._writer:
                self._writer.write(frame)
            try:
                self._frame_q.put_nowait(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            except queue.Full:
                pass

    def _collect_frame(self, face_res, hand_res, pose_res, w: int, h: int):
        idx = len(self._frames_data)
        fd  = FrameData(index=idx, timestamp=idx / max(self._fps, 1))
        fd.persons = _build_persons(face_res, hand_res, pose_res, w, h)
        self._frames_data.append(fd)

    # ── 디스플레이 갱신 (메인 스레드) ────────────────────────────────────
    def _schedule_display(self):
        if not self._running:
            return
        delay = max(1, 1000 // self._fps)
        self._after_id = self.win.after(delay, self._update_canvas)

    def _update_canvas(self):
        if not self._running:
            return

        try:
            frame_rgb = self._frame_q.get_nowait()
            cw = self._canvas.winfo_width()
            ch = self._canvas.winfo_height()
            if cw <= 1: cw = CANVAS_W
            if ch <= 1: ch = CANVAS_H
            if True:
                img = Image.fromarray(frame_rgb)
                img = img.resize((cw, ch), Image.LANCZOS)
                self._photo = ImageTk.PhotoImage(img)
                self._canvas.delete("all")
                self._canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)

                # 녹화 중 REC 인디케이터
                if self._recording:
                    self._canvas.create_oval(
                        12, 12, 30, 30, fill="red", outline="",
                    )
                    self._canvas.create_text(
                        38, 21, text="REC",
                        fill="red", font=("Segoe UI", 11, "bold"),
                        anchor=tk.W,
                    )
        except queue.Empty:
            pass
        except Exception as e:
            print(f"[Display error] {e}")

        self._schedule_display()

    # ── 종료 ──────────────────────────────────────────────────────────────
    def _on_close(self):
        self._stop_camera()
        self.win.destroy()
