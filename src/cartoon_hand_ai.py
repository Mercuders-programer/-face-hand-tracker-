"""
cartoon_hand_ai.py — ControlNet + Stable Diffusion 기반 만화 손 렌더링

사용 흐름:
  ai = CartoonHandAI()
  ai.load(progress_cb=lambda m: print(m))   # 최초 1회 (모델 다운로드)
  result = ai.render_frame(frame, hand_res)  # 프레임별 호출
"""

import cv2
import numpy as np
import os

# ── OpenPose hand skeleton 색상 (BGR) ──────────────────────────────────────
_FINGER_COLORS_BGR = [
    (0,   0,   255),   # 엄지 (빨강)
    (0,   85,  255),   # 검지 (주황)
    (0,   170, 255),   # 중지 (노랑)
    (0,   255, 255),   # 약지 (연노랑)
    (0,   255, 170),   # 소지 (연초록)
]

# 손목(0)에서 각 손가락 MCP → PIP → DIP → TIP 순서
_FINGER_CHAINS = [
    [0, 1, 2, 3, 4],       # 엄지
    [0, 5, 6, 7, 8],       # 검지
    [0, 9, 10, 11, 12],    # 중지
    [0, 13, 14, 15, 16],   # 약지
    [0, 17, 18, 19, 20],   # 소지
]


class CartoonHandAI:
    """ControlNet + Stable Diffusion으로 만화 손 렌더링 (내보내기 전용)"""

    _pipe          = None   # 클래스 레벨 캐시 (한 번만 로드)
    _pipe_sd_model = None   # 캐시된 sd_model 경로 (다른 모델 선택 시 재로드 판단)
    _pipe_lcm_mode = False  # 캐시된 파이프라인의 LCM 모드 여부

    # 품질 프리셋 (mode → steps, guidance, use_lcm)
    QUALITY_PRESETS = {
        "fast":    dict(steps=4,  guidance=1.5, use_lcm=True),   # ⚡ LCM 4스텝
        "balance": dict(steps=10, guidance=7.5, use_lcm=False),  # ⚖ DPM++ 10스텝
        "quality": dict(steps=20, guidance=7.5, use_lcm=False),  # 🎨 DPM++ 20스텝
    }

    DEFAULT_SD_MODEL         = "andite/anything-v4.0"
    DEFAULT_CONTROLNET_MODEL = "lllyasviel/control_v11p_sd15_openpose"

    PROMPT = (
        "anime cartoon hand, cel shading, clean black outline, "
        "smooth skin, detailed fingers, white background, "
        "high quality illustration, digital art"
    )
    NEG_PROMPT = (
        "realistic, photo, blurry, deformed, extra fingers, "
        "low quality, bad anatomy, ugly, disfigured, watermark"
    )

    def __init__(self, sd_model=None, controlnet_model=None, device="cuda"):
        self.sd_model         = sd_model or self.DEFAULT_SD_MODEL
        self.controlnet_model = controlnet_model or self.DEFAULT_CONTROLNET_MODEL
        self.device           = device

    # ── 모델 로드 ──────────────────────────────────────────────────────────
    def load(self, progress_cb=None, use_lcm=True):
        """파이프라인 초기화 (최초 1회만 실행, 이후 캐시 재사용).

        self.sd_model:
          - 로컬 파일 경로 (.safetensors / .ckpt) → from_single_file() 사용
          - Hugging Face 모델 ID                  → from_pretrained() 사용
        use_lcm:
          - True  → LCM-LoRA + LCMScheduler (4스텝, 5배 빠름)
          - False → DPMSolver++ Karras 스케줄러 (10~20스텝, 고품질)
        """
        need_reload = (
            CartoonHandAI._pipe is None
            or CartoonHandAI._pipe_sd_model != self.sd_model
            or CartoonHandAI._pipe_lcm_mode != use_lcm
        )
        if not need_reload:
            if progress_cb:
                progress_cb("모델 캐시 재사용")
            return
        if CartoonHandAI._pipe is not None:
            self.unload()

        def _cb(msg):
            if progress_cb:
                progress_cb(msg)

        try:
            import torch
            from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
        except ImportError:
            raise ImportError(
                "diffusers 패키지가 필요합니다.\n"
                "pip install diffusers transformers accelerate"
            )

        if not torch.cuda.is_available():
            _cb("⚠ CUDA 미감지 — CPU 모드로 실행 (매우 느림)")
            self.device = "cpu"
            dtype = None
        else:
            dtype = torch.float16

        _cb("ControlNet 모델 로드 중...")
        controlnet = ControlNetModel.from_pretrained(
            self.controlnet_model,
            torch_dtype=dtype,
        )

        _cb(f"SD 모델 로드 중: {os.path.basename(self.sd_model)}")
        pipe_kwargs = dict(
            controlnet=controlnet,
            safety_checker=None,
            requires_safety_checker=False,
        )
        if dtype is not None:
            pipe_kwargs["torch_dtype"] = dtype

        # 로컬 파일(.safetensors/.ckpt)이면 from_single_file, 아니면 from_pretrained
        is_local = os.path.isfile(self.sd_model)
        if is_local:
            pipe = StableDiffusionControlNetPipeline.from_single_file(
                self.sd_model, **pipe_kwargs
            )
        else:
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                self.sd_model, **pipe_kwargs
            )
        pipe = pipe.to(self.device)

        # ── 스케줄러 + LCM-LoRA 설정 ──────────────────────────────────────
        if use_lcm:
            _cb("LCM-LoRA 로드 중... (~200MB, 최초 1회)")
            try:
                from diffusers import LCMScheduler
                pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
                pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
                _cb("LCM-LoRA 적용 완료 (4스텝 모드)")
            except Exception as e:
                _cb(f"LCM-LoRA 실패 ({e}), DPM++ 모드로 대체")
                use_lcm = False
        if not use_lcm:
            try:
                from diffusers import DPMSolverMultistepScheduler
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipe.scheduler.config,
                    use_karras_sigmas=True,
                )
                _cb("DPM++ 2M Karras 스케줄러 적용")
            except Exception:
                pass  # 기본 스케줄러 유지

        # ── 메모리 최적화 ───────────────────────────────────────────────
        try:
            pipe.enable_xformers_memory_efficient_attention()
            _cb("xformers 메모리 최적화 활성화")
        except Exception:
            pass
        pipe.enable_attention_slicing()

        CartoonHandAI._pipe          = pipe
        CartoonHandAI._pipe_sd_model = self.sd_model
        CartoonHandAI._pipe_lcm_mode = use_lcm
        _cb("모델 로드 완료 ✓")

    @classmethod
    def unload(cls):
        """VRAM 해제 (필요 시 호출)"""
        if cls._pipe is not None:
            try:
                import torch
                cls._pipe.to("cpu")
                del cls._pipe
                torch.cuda.empty_cache()
            except Exception:
                pass
            cls._pipe = None

    # ── 스켈레톤 이미지 생성 ───────────────────────────────────────────────
    @staticmethod
    def landmarks_to_skeleton(hand_landmarks_list, frame_w, frame_h,
                               crop_x, crop_y, crop_w, crop_h, out_size=512):
        """손 랜드마크 → OpenPose hand skeleton 이미지 (out_size × out_size)
        ControlNet openpose 모델의 입력 형식과 호환됨.
        """
        img = np.zeros((out_size, out_size, 3), dtype=np.uint8)

        for hlms in hand_landmarks_list:
            def _to_xy(lm):
                px = (lm.x * frame_w - crop_x) / max(crop_w, 1) * out_size
                py = (lm.y * frame_h - crop_y) / max(crop_h, 1) * out_size
                return (int(np.clip(px, 0, out_size - 1)),
                        int(np.clip(py, 0, out_size - 1)))

            pts = [_to_xy(lm) for lm in hlms]
            if len(pts) < 21:
                continue

            # 손가락별 연결선
            for chain, color in zip(_FINGER_CHAINS, _FINGER_COLORS_BGR):
                for i in range(len(chain) - 1):
                    cv2.line(img, pts[chain[i]], pts[chain[i + 1]],
                             color, 3, cv2.LINE_AA)

            # 관절 점
            for i, pt in enumerate(pts):
                fi = next((j for j, c in enumerate(_FINGER_CHAINS) if i in c), 0)
                cv2.circle(img, pt, 6, _FINGER_COLORS_BGR[fi], -1, cv2.LINE_AA)

        return img

    # ── 손 마스크 생성 ─────────────────────────────────────────────────────
    @staticmethod
    def _hand_mask(hand_landmarks_list, frame_w, frame_h,
                   crop_x, crop_y, crop_w, crop_h):
        """크롭 영역 기준 손 마스크 (dilate + Gaussian feather)"""
        mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
        for hlms in hand_landmarks_list:
            pts = np.array([
                [int(np.clip(lm.x * frame_w - crop_x, 0, crop_w - 1)),
                 int(np.clip(lm.y * frame_h - crop_y, 0, crop_h - 1))]
                for lm in hlms
            ], dtype=np.int32)
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(mask, hull, 255)

        k = max(20, int(min(crop_w, crop_h) * 0.07))
        if k % 2 == 0:
            k += 1
        mask = cv2.dilate(mask, np.ones((k, k), np.uint8))
        mask = cv2.GaussianBlur(mask, (k * 2 + 1, k * 2 + 1), 0)
        return mask

    # ── 프레임 렌더링 ──────────────────────────────────────────────────────
    def render_frame(self, frame, hand_res, steps=20, guidance=7.5,
                     controlnet_scale=0.85):
        """손 영역을 AI 만화 스타일로 교체한 프레임 반환.

        Args:
            frame:             BGR numpy 배열 (원본 프레임)
            hand_res:          MediaPipe HandLandmarker 감지 결과
            steps:             SD 추론 스텝 (낮을수록 빠름, 기본 20)
            guidance:          Classifier-free guidance scale (기본 7.5)
            controlnet_scale:  ControlNet 적용 강도 (기본 0.85)
        Returns:
            BGR numpy 배열 (AI 렌더링 적용된 프레임)
        """
        if CartoonHandAI._pipe is None:
            raise RuntimeError("load()를 먼저 호출해야 합니다.")
        if not hand_res.hand_landmarks:
            return frame

        from PIL import Image

        h, w = frame.shape[:2]
        result = frame.copy()

        # ── 모든 손을 하나의 바운딩박스로 합산 ──────────────────────────
        all_x = [lm.x * w for hlms in hand_res.hand_landmarks for lm in hlms]
        all_y = [lm.y * h for hlms in hand_res.hand_landmarks for lm in hlms]
        span_x = max(all_x) - min(all_x)
        span_y = max(all_y) - min(all_y)
        pad = max(span_x * 0.4, span_y * 0.4, 40)

        x1 = max(0, int(min(all_x) - pad))
        y1 = max(0, int(min(all_y) - pad))
        x2 = min(w, int(max(all_x) + pad))
        y2 = min(h, int(max(all_y) + pad))

        # 정사각형화 (SD 입력에 유리)
        side = max(x2 - x1, y2 - y1)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(w, x1 + side)
        y2 = min(h, y1 + side)
        cw, ch = x2 - x1, y2 - y1

        if cw < 30 or ch < 30:
            return result

        # ── OpenPose skeleton 이미지 생성 ────────────────────────────────
        skel = self.landmarks_to_skeleton(
            hand_res.hand_landmarks, w, h, x1, y1, cw, ch, 512)
        skel_pil = Image.fromarray(cv2.cvtColor(skel, cv2.COLOR_BGR2RGB))

        # ── ControlNet + SD 추론 ─────────────────────────────────────────
        out_pil = CartoonHandAI._pipe(
            self.PROMPT,
            image=skel_pil,
            negative_prompt=self.NEG_PROMPT,
            num_inference_steps=steps,
            guidance_scale=guidance,
            controlnet_conditioning_scale=controlnet_scale,
            width=512,
            height=512,
        ).images[0]

        # ── 크롭 크기로 리사이즈 후 합성 ────────────────────────────────
        out_bgr = cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGB2BGR)
        out_bgr = cv2.resize(out_bgr, (cw, ch), interpolation=cv2.INTER_LANCZOS4)

        mask  = self._hand_mask(hand_res.hand_landmarks, w, h, x1, y1, cw, ch)
        alpha = mask.astype(np.float32) / 255.0
        a3    = alpha[:, :, np.newaxis]

        roi     = result[y1:y2, x1:x2].astype(np.float32)
        blended = out_bgr.astype(np.float32) * a3 + roi * (1.0 - a3)
        result[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

        return result
