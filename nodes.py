import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

import folder_paths


def _to_numpy_uint8_image(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().numpy()
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image


def _to_void_black_mask(mask_tensor: torch.Tensor, threshold: float) -> np.ndarray:
    mask = mask_tensor.detach().cpu().numpy()
    binary = mask > threshold
    # VOID Stage 1 convention: object=0, background=255.
    return np.where(binary, 0, 255).astype(np.uint8)


def _write_lossless_grayscale_video(frames: list[np.ndarray], output_path: str, fps: float) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    temp_avi = output.with_suffix(".avi")
    fourcc = cv2.VideoWriter_fourcc(*"FFV1")
    height, width = frames[0].shape
    writer = cv2.VideoWriter(str(temp_avi), fourcc, fps, (width, height), isColor=False)

    if not writer.isOpened():
        raise RuntimeError(f"Failed to open temporary video writer for {temp_avi}")

    for frame in frames:
        writer.write(frame)
    writer.release()

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(temp_avi),
        "-c:v",
        "libx264",
        "-qp",
        "0",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv444p",
        "-r",
        str(fps),
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    temp_avi.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed while writing {output}: {result.stderr.strip()}")


class VoidExportBlackMask:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "mask": ("MASK",),
                "fps": ("FLOAT", {"default": 12.0, "min": 0.1, "max": 240.0, "step": 0.1}),
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "void/black_mask",
                        "tooltip": "Base output path inside the ComfyUI output directory.",
                    },
                ),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "output_folder": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("mask", "first_frame", "black_mask_path", "first_frame_path")
    FUNCTION = "export"
    CATEGORY = "VOID"
    OUTPUT_NODE = True

    def export(self, images, mask, fps, filename_prefix, threshold, output_folder=""):
        if images.ndim != 4:
            raise ValueError(f"`images` must have shape [T,H,W,C], got {tuple(images.shape)}")
        if mask.ndim != 3:
            raise ValueError(f"`mask` must have shape [T,H,W], got {tuple(mask.shape)}")
        if images.shape[0] != mask.shape[0]:
            raise ValueError(
                f"Frame count mismatch: images has {images.shape[0]} frames, mask has {mask.shape[0]} frames."
            )

        if output_folder:
            base_output_dir = output_folder
            Path(base_output_dir).mkdir(parents=True, exist_ok=True)
            full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
                filename_prefix, base_output_dir, images[0].shape[1], images[0].shape[0]
            )
        else:
            full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
                filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
            )

        base_name = f"{filename}_{counter:05d}"
        black_mask_path = os.path.join(full_output_folder, f"{base_name}_black_mask.mp4")
        first_frame_path = os.path.join(full_output_folder, f"{base_name}_first_frame.jpg")

        mask_frames = [_to_void_black_mask(mask[i], threshold) for i in range(mask.shape[0])]
        _write_lossless_grayscale_video(mask_frames, black_mask_path, fps)

        first_frame = _to_numpy_uint8_image(images[0])
        Image.fromarray(first_frame).save(first_frame_path, quality=95)

        result = {
            "ui": {
                "text": [
                    f"Saved black mask video: {black_mask_path}",
                    f"Saved first frame: {first_frame_path}",
                ]
            },
            "result": (mask, images[:1], black_mask_path, first_frame_path),
        }
        return result


NODE_CLASS_MAPPINGS = {
    "VoidExportBlackMask": VoidExportBlackMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VoidExportBlackMask": "VOID Export Black Mask",
}
