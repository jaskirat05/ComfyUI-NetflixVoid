from typing import Tuple

import torch
import torch.nn.functional as F



def _resize_video_batch(video: torch.Tensor, sample_size: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(video, size=sample_size, mode="area")



def _temporal_padding(video: torch.Tensor, min_length: int, max_length: int) -> torch.Tensor:
    if video.ndim != 5:
        raise ValueError(f"Expected tensor shape [B,C,T,H,W], got {tuple(video.shape)}")
    length = int(video.shape[2])
    if length == 0:
        raise ValueError("Input video has zero frames.")

    # Match upstream VOID temporal padding behavior.
    min_len = (length // 4) * 4 + 1
    if min_len < length:
        min_len += 4
    if (min_len // 4) % 2 == 0:
        min_len += 4
    target_length = min(min_len, max_length)
    target_length = max(min_length, target_length)

    video = video[:, :, :target_length]
    while video.shape[2] < target_length:
        video_flipped = torch.flip(video, [2])
        video = torch.cat([video, video_flipped], dim=2)
        video = video[:, :, :target_length]
    return video


class VoidPQ5EncodeQuadmask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "quadmask": ("IMAGE",),
                "sample_height": ("INT", {"default": 384, "min": 64, "max": 2048, "step": 8}),
                "sample_width": ("INT", {"default": 672, "min": 64, "max": 4096, "step": 8}),
                "max_video_length": ("INT", {"default": 197, "min": 1, "max": 2048, "step": 1}),
                "temporal_window_size": ("INT", {"default": 85, "min": 1, "max": 2048, "step": 1}),
            },
        }

    RETURN_TYPES = ("PQ5_MASK_TENSOR", "IMAGE")
    RETURN_NAMES = ("encoded_quadmask", "mask_preview_video")
    FUNCTION = "encode"
    CATEGORY = "VOID/PQ5"

    def encode(self, quadmask, sample_height, sample_width, max_video_length, temporal_window_size):
        quadmask = quadmask.detach().cpu().float().clamp(0.0, 1.0)
        if quadmask.ndim != 4:
            raise ValueError(f"`quadmask` must be an IMAGE batch [T,H,W,C], got {tuple(quadmask.shape)}.")

        quadmask = quadmask[:max_video_length, ..., 0]
        quadmask = (quadmask * 255.0).round()
        quadmask = torch.where(quadmask <= 31, 0.0, quadmask)
        quadmask = torch.where((quadmask > 31) & (quadmask <= 95), 63.0, quadmask)
        quadmask = torch.where((quadmask > 95) & (quadmask <= 191), 127.0, quadmask)
        quadmask = torch.where(quadmask > 191, 255.0, quadmask)
        quadmask = 255.0 - quadmask

        quadmask = quadmask.unsqueeze(1)
        quadmask = _resize_video_batch(quadmask, (sample_height, sample_width))
        quadmask = quadmask.permute(1, 0, 2, 3).unsqueeze(0)
        quadmask = _temporal_padding(quadmask, min_length=temporal_window_size, max_length=max_video_length)
        quadmask = quadmask / 255.0

        preview_video = quadmask[0, 0].unsqueeze(-1).repeat(1, 1, 1, 3)
        return (quadmask, preview_video)
