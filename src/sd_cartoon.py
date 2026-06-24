"""
sd_cartoon.py — Stable Diffusion img2img 카툰 변환 (고품질 / 느림).

필터형(White-box/bold)과 달리 "진짜 그려진" 만화체로 변환한다.
단점: 프레임당 수 초, 프레임마다 디테일이 달라져 영상에선 떨림(flicker)이 있음.
→ 정지컷/짧은 클립용. torch + diffusers 필요(미설치 시 ImportError).
"""

import cv2
import numpy as np

_PROMPT = ("flat 2D cartoon illustration, clean bold outlines, cel shading, "
           "vibrant flat colors, anime style, simple shading")
_NEG = "photo, realistic, 3d render, blurry, noisy, lowres, deformed, extra limbs"

# strength(1~5) → denoise 강도. 낮을수록 원본 보존(정체성↑), 높을수록 변형↑.
_STR_MAP = {1: 0.35, 2: 0.42, 3: 0.5, 4: 0.6, 5: 0.7}


class SDCartoon:
    """Stable Diffusion img2img 카툰 변환기 (지연 로드 + 캐시)."""

    def __init__(self, model="Lykon/dreamshaper-8"):
        self._model = model
        self._pipe  = None

    @property
    def loaded(self):
        return self._pipe is not None

    def load(self):
        if self._pipe is not None:
            return self
        import torch
        from diffusers import AutoPipelineForImage2Image

        use_cuda = torch.cuda.is_available()
        dtype    = torch.float16 if use_cuda else torch.float32
        pipe = AutoPipelineForImage2Image.from_pretrained(
            self._model, torch_dtype=dtype, safety_checker=None,
        )
        pipe = pipe.to("cuda" if use_cuda else "cpu")
        pipe.set_progress_bar_config(disable=True)
        try:
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()
        except Exception:
            pass
        self._pipe = pipe
        return self

    def stylize(self, frame, strength=3, steps=28):
        """BGR 입력 → SD 카툰 BGR 출력(원본 해상도)."""
        if self._pipe is None:
            self.load()
        from PIL import Image

        h, w = frame.shape[:2]
        sc      = 512 / min(h, w)
        nw, nh  = max(8, int(w * sc) // 8 * 8), max(8, int(h * sc) // 8 * 8)
        rgb     = cv2.cvtColor(cv2.resize(frame, (nw, nh)), cv2.COLOR_BGR2RGB)
        img     = Image.fromarray(rgb)

        denoise = _STR_MAP.get(int(max(1, min(5, strength))), 0.5)
        out = self._pipe(
            prompt=_PROMPT, negative_prompt=_NEG, image=img,
            strength=denoise, guidance_scale=7.0, num_inference_steps=steps,
        ).images[0]

        res = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
        if res.shape[:2] != (h, w):
            res = cv2.resize(res, (w, h), interpolation=cv2.INTER_LINEAR)
        return res
