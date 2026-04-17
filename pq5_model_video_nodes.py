import os
from pathlib import Path
from typing import Tuple

import folder_paths
import torch
import torch.nn.functional as F

from .pq5.runtime import load_pipeline_bundle_from_checkpoint


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_PQ5_ASSETS = Path(os.environ.get("VOID_PQ5_ASSETS", PACKAGE_DIR / "pq5_assets"))
TEXT_ENCODER_VOID_DIR = Path(folder_paths.get_folder_paths("text_encoders")[0]) / "void"


def _resize_video_batch(video: torch.Tensor, sample_size: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(video, size=sample_size, mode="area")



def _temporal_padding(video: torch.Tensor, min_length: int, max_length: int) -> torch.Tensor:
    if video.ndim != 5:
        raise ValueError(f"Expected tensor shape [B,C,T,H,W], got {tuple(video.shape)}")

    length = int(video.shape[2])
    if length == 0:
        raise ValueError("Input video has zero frames after preprocessing.")

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


class VoidPQ5LoadModel:
    @classmethod
    def INPUT_TYPES(cls):
        checkpoints = folder_paths.get_filename_list("checkpoints")
        if not checkpoints:
            checkpoints = ["<no checkpoints found in ComfyUI/models/checkpoints>"]
        vae_files = folder_paths.get_filename_list("vae")
        if not vae_files:
            vae_files = ["<no vae found in ComfyUI/models/vae>"]
        return {
            "required": {
                "checkpoint": (checkpoints,),
                "vae": (vae_files,),
            },
        }

    RETURN_TYPES = ("PQ5_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "VOID/PQ5"

    def load(self, checkpoint, vae):
        if checkpoint.startswith("<no checkpoints found"):
            raise RuntimeError("No checkpoint files found under ComfyUI/models/checkpoints.")
        if vae.startswith("<no vae found"):
            raise RuntimeError("No VAE files found under ComfyUI/models/vae.")

        checkpoint_path = folder_paths.get_full_path("checkpoints", checkpoint)
        if checkpoint_path is None:
            raise RuntimeError(f"Failed to resolve selected checkpoint `{checkpoint}` from ComfyUI checkpoints.")
        vae_path = folder_paths.get_full_path("vae", vae)
        if vae_path is None:
            raise RuntimeError(f"Failed to resolve selected VAE `{vae}` from ComfyUI VAE registry.")
        if not TEXT_ENCODER_VOID_DIR.exists():
            raise RuntimeError(
                f"Required text encoder directory not found: `{TEXT_ENCODER_VOID_DIR}`. "
                "Please place VOID text encoder files there."
            )

        bundle = load_pipeline_bundle_from_checkpoint(
            base_model=DEFAULT_PQ5_ASSETS,
            transformer_checkpoint=Path(checkpoint_path),
            vae_checkpoint=Path(vae_path),
            text_encoder_checkpoint=TEXT_ENCODER_VOID_DIR,
        )
        model = {
            "bundle": bundle,
            "checkpoint": checkpoint,
            "checkpoint_path": str(checkpoint_path),
            "vae": vae,
            "vae_path": str(vae_path),
            "text_encoder": "models/text_encoders/void",
            "text_encoder_path": str(TEXT_ENCODER_VOID_DIR),
            "base_model": str(DEFAULT_PQ5_ASSETS),
        }
        return (model,)


class VoidPQ5EncodeVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("PQ5_MODEL",),
                "video": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("PQ5_VIDEO_TENSOR", "INT", "IMAGE")
    RETURN_NAMES = ("encoded_video", "original_frame_count", "video_preview")
    FUNCTION = "encode"
    CATEGORY = "VOID/PQ5"

    def encode(self, model, video):
        bundle = model["bundle"]
        config = bundle["config"]
        vae = bundle["vae"]

        sample_size = tuple(int(x) for x in config.data.sample_size)
        max_video_length = int(config.data.max_video_length)
        temporal_window_size = int(config.video_model.temporal_window_size)

        effective_video_length = max_video_length
        if effective_video_length != 1:
            ratio = int(vae.config.temporal_compression_ratio)
            effective_video_length = int((effective_video_length - 1) // ratio * ratio) + 1

        video = video.detach().cpu().float().clamp(0.0, 1.0)
        if video.ndim != 4 or video.shape[-1] != 3:
            raise ValueError(f"`video` must be an IMAGE batch [T,H,W,3], got {tuple(video.shape)}.")

        original_frame_count = min(int(video.shape[0]), effective_video_length)

        encoded_video = video[:effective_video_length]
        encoded_video = encoded_video.permute(0, 3, 1, 2)
        encoded_video = _resize_video_batch(encoded_video, sample_size)
        encoded_video = encoded_video.permute(1, 0, 2, 3).unsqueeze(0)
        encoded_video = _temporal_padding(
            encoded_video,
            min_length=temporal_window_size,
            max_length=effective_video_length,
        )

        preview = encoded_video[0].permute(1, 2, 3, 0).clamp(0.0, 1.0)
        return (encoded_video, original_frame_count, preview)
