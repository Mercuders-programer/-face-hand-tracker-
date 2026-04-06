#!/usr/bin/env python3
"""
PoseTracker GUI
실행: python app.py
"""

import tkinter as tk
from tkinter import messagebox
import sys
import os

_BASE = os.path.dirname(os.path.abspath(__file__))
_LIB  = os.path.join(_BASE, "lib")

# WSL2: libGLESv2.so.2 자동 확보 — LD_LIBRARY_PATH 미설정 시 재실행
if os.path.isdir(_LIB) and _LIB not in os.environ.get("LD_LIBRARY_PATH", ""):
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = _LIB + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    os.execve(sys.executable, [sys.executable] + sys.argv, env)

sys.path.insert(0, os.path.join(_BASE, "src"))

# ── 테마 색상 ──────────────────────────────────────────────────────────────
BG      = "#1a1a2e"
ACCENT  = "#4a7fff"
TEXT_W  = "#e0e0ff"
TEXT_G  = "#8888aa"
BTN_DIM = "#2a2a4e"


class MainApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("PoseTracker")
        self.root.geometry("540x310")
        self.root.resizable(False, False)
        self.root.configure(bg=BG)
        self._build_ui()

    # ── UI 구성 ────────────────────────────────────────────────────────────
    def _build_ui(self):
        tk.Label(
            self.root, text="PoseTracker",
            font=("Segoe UI", 30, "bold"),
            fg=TEXT_W, bg=BG,
        ).pack(pady=(38, 4))

        tk.Label(
            self.root, text="얼굴 · 손 추적  →  After Effects",
            font=("Segoe UI", 11),
            fg=TEXT_G, bg=BG,
        ).pack(pady=(0, 26))

        # 버튼 + 버전 정보 가로 배치
        row = tk.Frame(self.root, bg=BG)
        row.pack(pady=6)

        # 왼쪽: 버튼 두 개
        btn_col = tk.Frame(row, bg=BG)
        btn_col.pack(side=tk.LEFT, padx=(0, 18))

        tk.Button(
            btn_col, text="  카메라 추적  ",
            font=("Segoe UI", 13, "bold"),
            bg=ACCENT, fg="white",
            activebackground="#3a6fee", activeforeground="white",
            relief=tk.FLAT, cursor="hand2",
            padx=24, pady=10,
            command=self._open_camera,
        ).pack(pady=5)

        tk.Button(
            btn_col, text="  영상 분석  ",
            font=("Segoe UI", 13, "bold"),
            bg=ACCENT, fg="white",
            activebackground="#3a6fee", activeforeground="white",
            relief=tk.FLAT, cursor="hand2",
            padx=24, pady=10,
            command=self._open_video,
        ).pack(pady=5)

        # 구분선
        tk.Frame(row, bg="#2a2a4e", width=1).pack(
            side=tk.LEFT, fill=tk.Y, padx=(0, 18), pady=4,
        )

        # 오른쪽: 버전 정보
        ver_col = tk.Frame(row, bg=BG)
        ver_col.pack(side=tk.LEFT)

        tk.Label(
            ver_col, text="v1.0.0",
            font=("Segoe UI", 22, "bold"),
            fg=ACCENT, bg=BG,
        ).pack(anchor="w")

        tk.Label(
            ver_col, text="PoseTracker",
            font=("Segoe UI", 10),
            fg=TEXT_G, bg=BG,
        ).pack(anchor="w")

    # ── 영상 분석 패널 열기 ────────────────────────────────────────────────
    def _open_video(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            parent=self.root,
            title="분석할 영상 선택",
            filetypes=[("영상 파일", "*.mp4 *.avi *.mov *.mkv"), ("모든 파일", "*.*")],
        )
        if not path:
            return
        try:
            from video_panel import VideoPanel
        except ImportError as e:
            messagebox.showerror("오류", str(e), parent=self.root)
            return
        VideoPanel(self.root, path)

    # ── 카메라 패널 열기 ───────────────────────────────────────────────────
    def _open_camera(self):
        try:
            from camera_panel import CameraPanel
        except ImportError as e:
            messagebox.showerror(
                "패키지 설치 필요",
                f"필요한 패키지가 없습니다:\n{e}\n\n"
                "pip install Pillow 를 실행하세요.",
                parent=self.root,
            )
            return
        CameraPanel(self.root)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    MainApp().run()
