import json
from pathlib import Path

import cv2
import numpy as np
import torch
import folder_paths

from ..vendor_sam3 import _model_cache as model_cache
from ..vendor_sam3 import utils as sam3_utils


class VoidLoadSAM3Model:
    MODEL_DIR = "models/sam3"
    MODEL_FILENAME = "sam3.safetensors"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "precision": (["auto", "bf16", "fp16", "fp32"], {"default": "auto"}),
                "compile": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("SAM3_MODEL_CONFIG",)
    RETURN_NAMES = ("sam3_model_config",)
    FUNCTION = "load"
    CATEGORY = "VOID"

    def load(self, precision="auto", compile=False):
        import comfy.model_management

        load_device = comfy.model_management.get_torch_device()
        checkpoint_path = Path(folder_paths.base_path) / self.MODEL_DIR / self.MODEL_FILENAME

        if not checkpoint_path.exists():
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as e:
                raise ImportError(
                    "huggingface_hub is required to download SAM3 weights automatically."
                ) from e
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            hf_hub_download(
                repo_id="apozz/sam3-safetensors",
                filename=self.MODEL_FILENAME,
                local_dir=str(checkpoint_path.parent),
            )

        bpe_path = str(Path(__file__).resolve().parent.parent / "vendor_sam3" / "sam3" / "bpe_simple_vocab_16e6.txt.gz")

        if precision == "auto":
            if comfy.model_management.should_use_bf16(load_device):
                dtype = torch.bfloat16
            elif comfy.model_management.should_use_fp16(load_device):
                dtype = torch.float16
            else:
                dtype = torch.float32
        else:
            dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        dtype_str = {
            torch.bfloat16: "bf16",
            torch.float16: "fp16",
            torch.float32: "fp32",
        }[dtype]
        config = {
            "checkpoint_path": str(checkpoint_path),
            "bpe_path": bpe_path,
            "precision": precision,
            "dtype": dtype_str,
            "compile": compile,
        }
        return (config,)


def _mask_to_bool(mask):
    if isinstance(mask, torch.Tensor):
        return mask.detach().cpu().float().numpy() > 0.5
    return np.asarray(mask) > 0.5


def _parse_affected_objects_json(raw: str):
    text = str(raw).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("`affected_objects_json` is not valid JSON.") from exc
    if isinstance(parsed, dict):
        return parsed.get("affected_objects", [])
    if isinstance(parsed, list):
        return parsed
    raise ValueError("`affected_objects_json` must be either a JSON list or an object with `affected_objects`.")


def _grid_cells_to_mask(grid_cells, grid_rows, grid_cols, frame_width, frame_height):
    mask = np.zeros((frame_height, frame_width), dtype=bool)
    cell_width = frame_width / grid_cols
    cell_height = frame_height / grid_rows
    for cell in grid_cells:
        row = int(cell.get("row", 0))
        col = int(cell.get("col", 0))
        y1 = int(row * cell_height)
        y2 = int((row + 1) * cell_height)
        x1 = int(col * cell_width)
        x2 = int((col + 1) * cell_width)
        y1 = max(0, min(frame_height, y1))
        y2 = max(0, min(frame_height, y2))
        x1 = max(0, min(frame_width, x1))
        x2 = max(0, min(frame_width, x2))
        if y2 > y1 and x2 > x1:
            mask[y1:y2, x1:x2] = True
    return mask


def _gridify_mask(mask, grid_rows, grid_cols):
    h, w = mask.shape
    gridified = np.zeros((h, w), dtype=bool)
    cell_width = w / grid_cols
    cell_height = h / grid_rows
    for row in range(grid_rows):
        for col in range(grid_cols):
            y1 = int(row * cell_height)
            y2 = int((row + 1) * cell_height)
            x1 = int(col * cell_width)
            x2 = int((col + 1) * cell_width)
            if mask[y1:y2, x1:x2].any():
                gridified[y1:y2, x1:x2] = True
    return gridified


def _filter_by_proximity(mask, primary_mask, dilation=50):
    kernel = np.ones((dilation, dilation), np.uint8)
    proximity = cv2.dilate(primary_mask.astype(np.uint8), kernel, iterations=1) > 0
    return mask & proximity


def _interpolate_int(a, b, t):
    return int(round(a + (b - a) * t))


def _trajectory_to_frame_masks(trajectory_path, object_size_grids, num_frames, grid_rows, grid_cols, frame_width, frame_height):
    if not trajectory_path:
        return [np.zeros((frame_height, frame_width), dtype=bool) for _ in range(num_frames)]

    points = sorted(trajectory_path, key=lambda p: int(p.get("frame", 0)))
    size_rows = max(1, int(object_size_grids.get("rows", 1)))
    size_cols = max(1, int(object_size_grids.get("cols", 1)))
    frame_masks = [np.zeros((frame_height, frame_width), dtype=bool) for _ in range(num_frames)]

    def center_to_cells(grid_row, grid_col):
        top = grid_row - size_rows // 2
        left = grid_col - size_cols // 2
        cells = []
        for r in range(size_rows):
            for c in range(size_cols):
                rr = max(0, min(grid_rows - 1, top + r))
                cc = max(0, min(grid_cols - 1, left + c))
                cells.append({"row": rr, "col": cc})
        return cells

    if len(points) == 1:
        frame_idx = max(0, min(num_frames - 1, int(points[0].get("frame", 0))))
        cells = center_to_cells(int(points[0].get("grid_row", 0)), int(points[0].get("grid_col", 0)))
        mask = _grid_cells_to_mask(cells, grid_rows, grid_cols, frame_width, frame_height)
        for idx in range(frame_idx, num_frames):
            frame_masks[idx] |= mask
        return frame_masks

    for point_idx in range(len(points) - 1):
        start = points[point_idx]
        end = points[point_idx + 1]
        f0 = max(0, min(num_frames - 1, int(start.get("frame", 0))))
        f1 = max(0, min(num_frames - 1, int(end.get("frame", 0))))
        if f1 < f0:
            f0, f1 = f1, f0
            start, end = end, start
        span = max(1, f1 - f0)
        for frame_idx in range(f0, f1 + 1):
            t = (frame_idx - f0) / span
            row = _interpolate_int(int(start.get("grid_row", 0)), int(end.get("grid_row", 0)), t)
            col = _interpolate_int(int(start.get("grid_col", 0)), int(end.get("grid_col", 0)), t)
            cells = center_to_cells(row, col)
            frame_masks[frame_idx] |= _grid_cells_to_mask(cells, grid_rows, grid_cols, frame_width, frame_height)

    first_frame = max(0, min(num_frames - 1, int(points[0].get("frame", 0))))
    last_frame = max(0, min(num_frames - 1, int(points[-1].get("frame", 0))))
    for frame_idx in range(0, first_frame):
        frame_masks[frame_idx] |= frame_masks[first_frame]
    for frame_idx in range(last_frame + 1, num_frames):
        frame_masks[frame_idx] |= frame_masks[last_frame]

    return frame_masks


class VoidBuildGreyMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam3_model_config": ("SAM3_MODEL_CONFIG",),
                "images": ("IMAGE",),
                "black_mask_video": ("IMAGE",),
                "affected_objects_json": ("STRING", {"multiline": True}),
                "grid_rows": ("INT", {"default": 8, "min": 1, "max": 128, "step": 1}),
                "grid_cols": ("INT", {"default": 8, "min": 1, "max": 128, "step": 1}),
                "confidence_threshold": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "proximity_dilation": ("INT", {"default": 50, "min": 1, "max": 512, "step": 1}),
                "max_detections": ("INT", {"default": 3, "min": 1, "max": 50, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("grey_mask_video", "grey_mask_mask", "grey_debug")
    FUNCTION = "build"
    CATEGORY = "VOID"

    def build(
        self,
        sam3_model_config,
        images,
        black_mask_video,
        affected_objects_json,
        grid_rows,
        grid_cols,
        confidence_threshold,
        proximity_dilation,
        max_detections,
    ):
        import comfy.model_management

        get_or_build_model = model_cache.get_or_build_model
        comfy_image_to_pil = sam3_utils.comfy_image_to_pil
        masks_to_comfy_mask = sam3_utils.masks_to_comfy_mask

        if images.ndim != 4 or black_mask_video.ndim != 4:
            raise ValueError("`images` and `black_mask_video` must be IMAGE batches [T,H,W,C].")
        frame_count, frame_height, frame_width, _ = images.shape
        if black_mask_video.shape[:3] != images.shape[:3]:
            raise ValueError(
                f"Shape mismatch: images {tuple(images.shape)} vs black_mask_video {tuple(black_mask_video.shape)}."
            )

        affected_objects = _parse_affected_objects_json(affected_objects_json)

        grey_masks = [np.zeros((frame_height, frame_width), dtype=bool) for _ in range(frame_count)]
        primary_mask = _mask_to_bool(1.0 - black_mask_video[0, ..., 0])
        first_frame_pil = comfy_image_to_pil(images[:1])

        sam3_model = get_or_build_model(sam3_model_config)
        comfy.model_management.load_models_gpu([sam3_model])
        processor = sam3_model.processor

        if hasattr(processor, "sync_device_with_model"):
            processor.sync_device_with_model()
        if hasattr(processor, "set_confidence_threshold"):
            processor.set_confidence_threshold(confidence_threshold)

        debug_layers = []

        for obj in affected_objects:
            category = str(obj.get("category", "physical")).strip().lower()
            noun = str(obj.get("noun", "")).strip()
            will_move = bool(obj.get("will_move", False))

            if category == "visual_artifact" and obj.get("grid_localizations"):
                frame_masks = [np.zeros((frame_height, frame_width), dtype=bool) for _ in range(frame_count)]
                for loc in obj.get("grid_localizations", []):
                    frame_idx = max(0, min(frame_count - 1, int(loc.get("frame", 0))))
                    frame_masks[frame_idx] |= _grid_cells_to_mask(
                        loc.get("grid_regions", []), grid_rows, grid_cols, frame_width, frame_height
                    )
                last_mask = None
                for idx in range(frame_count):
                    if frame_masks[idx].any():
                        last_mask = frame_masks[idx].copy()
                    elif last_mask is not None:
                        frame_masks[idx] |= last_mask
                for idx in range(frame_count - 1, -1, -1):
                    if not frame_masks[idx].any() and idx + 1 < frame_count:
                        frame_masks[idx] |= frame_masks[idx + 1]
                for idx in range(frame_count):
                    grey_masks[idx] |= frame_masks[idx]
                debug_layers.append(frame_masks[0])
                continue

            if will_move and obj.get("trajectory_path") and obj.get("object_size_grids"):
                traj_masks = _trajectory_to_frame_masks(
                    obj.get("trajectory_path", []),
                    obj.get("object_size_grids", {}),
                    frame_count,
                    grid_rows,
                    grid_cols,
                    frame_width,
                    frame_height,
                )
                for idx in range(frame_count):
                    grey_masks[idx] |= traj_masks[idx]
                debug_layers.append(traj_masks[0])
                continue

            if not noun:
                continue

            state = processor.set_image(first_frame_pil)
            state = processor.set_text_prompt(noun, state)
            masks = state.get("masks", None)
            scores = state.get("scores", None)
            if masks is None or len(masks) == 0:
                continue

            if scores is not None and len(scores) > 0:
                sorted_indices = torch.argsort(scores, descending=True)
                masks = masks[sorted_indices]
            if max_detections > 0 and len(masks) > max_detections:
                masks = masks[:max_detections]

            comfy_masks = masks_to_comfy_mask(masks)
            noun_mask = _mask_to_bool(comfy_masks.any(dim=0))
            noun_mask = _filter_by_proximity(noun_mask, primary_mask, dilation=proximity_dilation)
            if not noun_mask.any():
                continue

            noun_mask = _gridify_mask(noun_mask, grid_rows, grid_cols)
            for idx in range(frame_count):
                grey_masks[idx] |= noun_mask
            debug_layers.append(noun_mask)

        grey_mask_mask = torch.stack(
            [torch.from_numpy(np.where(mask, 127.0 / 255.0, 1.0).astype(np.float32)) for mask in grey_masks], dim=0
        )
        grey_mask_video = grey_mask_mask.unsqueeze(-1).repeat(1, 1, 1, 3)

        debug_image = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        debug_image[primary_mask] = [255, 0, 0]
        for layer in debug_layers:
            debug_image[layer] = [0, 255, 0]
        grey_debug = torch.from_numpy(debug_image.astype(np.float32) / 255.0).unsqueeze(0)

        return (grey_mask_video, grey_mask_mask, grey_debug)
