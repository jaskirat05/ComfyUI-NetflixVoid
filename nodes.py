import json
import importlib
import importlib.util
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch


def _to_void_black_mask(mask_tensor: torch.Tensor, threshold: float) -> torch.Tensor:
    binary = (mask_tensor.detach().cpu().float() > threshold).float()
    # VOID Stage 1 convention: object=0, background=255.
    return 1.0 - binary


class VoidExportBlackMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "mask": ("MASK",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("black_mask_video", "black_mask_mask", "first_frame")
    FUNCTION = "export"
    CATEGORY = "VOID"

    def export(self, images, mask, threshold):
        if images.ndim != 4:
            raise ValueError(f"`images` must have shape [T,H,W,C], got {tuple(images.shape)}")
        if mask.ndim != 3:
            raise ValueError(f"`mask` must have shape [T,H,W], got {tuple(mask.shape)}")
        if images.shape[0] != mask.shape[0]:
            raise ValueError(
                f"Frame count mismatch: images has {images.shape[0]} frames, mask has {mask.shape[0]} frames."
            )

        black_mask_mask = torch.stack([_to_void_black_mask(mask[i], threshold) for i in range(mask.shape[0])], dim=0)
        black_mask_video = black_mask_mask.unsqueeze(-1).repeat(1, 1, 1, 3)
        first_frame = images[:1]
        return (black_mask_video, black_mask_mask, first_frame)


def _tensor_image_to_uint8(image_tensor: torch.Tensor):
    return torch.clamp(image_tensor.detach().cpu().float() * 255.0, 0, 255).to(torch.uint8).numpy()


def _uint8_image_to_tensor(image):
    return torch.from_numpy(image.astype("float32") / 255.0)


def _calculate_square_grid(width: int, height: int, min_grid: int = 8):
    aspect_ratio = width / height
    if width >= height:
        grid_rows = min_grid
        grid_cols = max(min_grid, round(min_grid * aspect_ratio))
    else:
        grid_cols = min_grid
        grid_rows = max(min_grid, round(min_grid / aspect_ratio))
    return grid_rows, grid_cols


def _draw_grid_overlay(image, grid_rows: int, grid_cols: int, frame_label=None):
    result = image.copy()
    h, w = result.shape[:2]
    cell_width = w / grid_cols
    cell_height = h / grid_rows

    for col in range(1, grid_cols):
        x = int(col * cell_width)
        cv2.line(result, (x, 0), (x, h), (255, 255, 0), 2)

    for row in range(1, grid_rows):
        y = int(row * cell_height)
        cv2.line(result, (0, y), (w, y), (255, 255, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for col in range(grid_cols):
        x = int((col + 0.5) * cell_width)
        cv2.putText(result, str(col), (x - 8, 20), font, 0.4, (255, 255, 0), 1)

    for row in range(grid_rows):
        y = int((row + 0.5) * cell_height)
        cv2.putText(result, str(row), (10, y + 8), font, 0.4, (255, 255, 0), 1)

    if frame_label:
        cv2.putText(result, frame_label, (10, h - 10), font, 0.5, (255, 255, 0), 2)

    return result


def _overlay_primary_mask(image, black_mask_frame):
    result = image.copy()
    object_mask = black_mask_frame[..., 0] < 0.5 if black_mask_frame.ndim == 3 else black_mask_frame < 0.5
    overlay = result.copy()
    overlay[object_mask] = [255, 0, 0]
    return cv2.addWeighted(result, 0.6, overlay, 0.4, 0)


def _make_vlm_analysis_prompt(instruction: str, grid_rows: int, grid_cols: int, has_multi_frame_grids: bool = True):
    if has_multi_frame_grids:
        grid_context = f"""
1. Multiple grid reference frames sampled through the video.
   - Each frame shows a yellow grid with {grid_rows} rows x {grid_cols} columns.
   - Grid cells are labeled (row, col) starting from (0, 0) at top-left.
2. One masked first frame.
   - The primary object to remove is highlighted in red.
3. Analyze what else is affected if that primary object is removed.
"""
    else:
        grid_context = f"""
1. One masked first frame with yellow grid.
   - Grid size is {grid_rows} rows x {grid_cols} columns.
   - The primary object to remove is highlighted in red.
"""

    return f"""
You are an expert video analyst specializing in physics and object interactions.
Return valid JSON only.

CONTEXT:
{grid_context}
Edit instruction: "{instruction}"

TASK:
1. Identify integral belongings that should be removed together with the primary object.
2. Identify affected objects or visual artifacts caused by the primary object.
3. For each affected object decide whether it will move after removal.
4. For visual artifacts, provide grid_localizations across the reference frames.
5. For moving physical objects, provide object_size_grids and trajectory_path in grid coordinates.
6. Describe the final scene after removal.

OUTPUT JSON FORMAT:
{{
  "edit_instruction": "{instruction}",
  "integral_belongings": [
    {{
      "noun": "bike",
      "why": "person is riding the bike throughout the video"
    }}
  ],
  "affected_objects": [
    {{
      "noun": "shadow",
      "category": "visual_artifact",
      "why": "cast by the primary object",
      "will_move": false,
      "first_appears_frame": 0,
      "movement_description": "Disappears entirely as visual artifact",
      "grid_localizations": [
        {{"frame": 0, "grid_regions": [{{"row": 6, "col": 3}}]}}
      ]
    }},
    {{
      "noun": "guitar",
      "category": "physical",
      "why": "primary object is supporting it against gravity",
      "will_move": true,
      "first_appears_frame": 0,
      "movement_description": "Falls to the ground after removal",
      "object_size_grids": {{"rows": 3, "cols": 2}},
      "trajectory_path": [
        {{"frame": 0, "grid_row": 3, "grid_col": 6}},
        {{"frame": 20, "grid_row": 6, "grid_col": 6}},
        {{"frame": 40, "grid_row": 9, "grid_col": 7}}
      ]
    }}
  ],
  "scene_description": "Describe the scene after removal in 1-2 sentences.",
  "confidence": 0.85
}}
""".strip()


class VoidPrepareVLMAnalysis:
    SAMPLE_POINTS = [0.0, 0.11, 0.22, 0.33, 0.44, 0.56, 0.67, 0.78, 0.89, 1.0]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "black_mask_video": ("IMAGE",),
                "instruction": ("STRING", {"default": "remove the object", "multiline": True}),
                "min_grid": ("INT", {"default": 8, "min": 2, "max": 64, "step": 1}),
                "use_multi_frame_grids": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "STRING", "INT", "INT")
    RETURN_NAMES = (
        "masked_grid_first_frame",
        "grid_reference_frames",
        "qwen_input_frames",
        "prompt",
        "grid_rows",
        "grid_cols",
    )
    FUNCTION = "prepare"
    CATEGORY = "VOID"

    def prepare(self, images, black_mask_video, instruction, min_grid, use_multi_frame_grids):
        if images.ndim != 4 or black_mask_video.ndim != 4:
            raise ValueError("`images` and `black_mask_video` must both be IMAGE batches [T,H,W,C].")
        if images.shape[:3] != black_mask_video.shape[:3]:
            raise ValueError(
                f"Shape mismatch: images {tuple(images.shape)} vs black_mask_video {tuple(black_mask_video.shape)}."
            )

        frame_count, height, width, _ = images.shape
        grid_rows, grid_cols = _calculate_square_grid(width, height, min_grid)

        first_frame = _tensor_image_to_uint8(images[0])
        first_mask = black_mask_video[0].detach().cpu().float().numpy()
        masked_first = _overlay_primary_mask(first_frame, first_mask)
        masked_grid_first = _draw_grid_overlay(masked_first, grid_rows, grid_cols)

        reference_frames = []
        if use_multi_frame_grids:
            used_indices = []
            for sample in self.SAMPLE_POINTS:
                frame_idx = int(sample * max(frame_count - 1, 0))
                frame_idx = max(0, min(frame_idx, frame_count - 1))
                if frame_idx in used_indices:
                    continue
                used_indices.append(frame_idx)
                frame = _tensor_image_to_uint8(images[frame_idx])
                label = f"Frame {frame_idx} ({int(sample * 100)}%)"
                reference_frames.append(_draw_grid_overlay(frame, grid_rows, grid_cols, label))
        else:
            reference_frames.append(masked_grid_first.copy())

        masked_grid_first_tensor = _uint8_image_to_tensor(masked_grid_first).unsqueeze(0)
        reference_frames_tensor = torch.stack([_uint8_image_to_tensor(frame) for frame in reference_frames], dim=0)
        qwen_input_frames = torch.cat([reference_frames_tensor, masked_grid_first_tensor], dim=0)
        prompt = _make_vlm_analysis_prompt(
            instruction=instruction,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            has_multi_frame_grids=use_multi_frame_grids,
        )
        return (
            masked_grid_first_tensor,
            reference_frames_tensor,
            qwen_input_frames,
            prompt,
            grid_rows,
            grid_cols,
        )


def _cleanup_json_response(raw: str):
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Failed to parse VLM response as JSON")
        return json.loads(cleaned[start:end + 1])


class VoidParseVLMAnalysis:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "raw_response": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "FLOAT")
    RETURN_NAMES = (
        "analysis_json",
        "integral_belongings_json",
        "affected_objects_json",
        "scene_description",
        "confidence",
    )
    FUNCTION = "parse"
    CATEGORY = "VOID"

    def parse(self, raw_response):
        parsed = _cleanup_json_response(raw_response)
        result = {
            "edit_instruction": str(parsed.get("edit_instruction", "")).strip(),
            "integral_belongings": [],
            "affected_objects": [],
            "scene_description": str(parsed.get("scene_description", "")).strip(),
            "confidence": float(parsed.get("confidence", 0.0)),
        }

        for item in parsed.get("integral_belongings", [])[:3]:
            noun = str(item.get("noun", "")).strip().lower()
            why = str(item.get("why", "")).strip()[:200]
            if noun:
                result["integral_belongings"].append({"noun": noun, "why": why})

        for item in parsed.get("affected_objects", [])[:5]:
            noun = str(item.get("noun", "")).strip().lower()
            if not noun:
                continue
            obj = {
                "noun": noun,
                "category": str(item.get("category", "physical")).strip().lower(),
                "why": str(item.get("why", "")).strip()[:200],
                "will_move": bool(item.get("will_move", False)),
                "first_appears_frame": int(item.get("first_appears_frame", 0)),
                "movement_description": str(item.get("movement_description", "")).strip()[:300],
            }

            if "currently_moving" in item:
                obj["currently_moving"] = bool(item.get("currently_moving", False))
            if "should_have_stayed" in item:
                obj["should_have_stayed"] = bool(item.get("should_have_stayed", False))
            if "original_position_grid" in item:
                grid = item.get("original_position_grid", {})
                obj["original_position_grid"] = {
                    "row": int(grid.get("row", 0)),
                    "col": int(grid.get("col", 0)),
                }
            if "grid_localizations" in item:
                grid_localizations = []
                for loc in item.get("grid_localizations", []):
                    frame_loc = {"frame": int(loc.get("frame", 0)), "grid_regions": []}
                    for region in loc.get("grid_regions", []):
                        frame_loc["grid_regions"].append(
                            {"row": int(region.get("row", 0)), "col": int(region.get("col", 0))}
                        )
                    if frame_loc["grid_regions"]:
                        grid_localizations.append(frame_loc)
                if grid_localizations:
                    obj["grid_localizations"] = grid_localizations
            if obj["will_move"] and "object_size_grids" in item and "trajectory_path" in item:
                size_grids = item.get("object_size_grids", {})
                obj["object_size_grids"] = {
                    "rows": int(size_grids.get("rows", 2)),
                    "cols": int(size_grids.get("cols", 2)),
                }
                trajectory = []
                for point in item.get("trajectory_path", []):
                    trajectory.append(
                        {
                            "frame": int(point.get("frame", 0)),
                            "grid_row": int(point.get("grid_row", 0)),
                            "grid_col": int(point.get("grid_col", 0)),
                        }
                    )
                if trajectory:
                    obj["trajectory_path"] = trajectory

            result["affected_objects"].append(obj)

        return (
            json.dumps(result, indent=2),
            json.dumps(result["integral_belongings"], indent=2),
            json.dumps(result["affected_objects"], indent=2),
            result["scene_description"],
            result["confidence"],
        )


def _mask_to_bool(mask):
    if isinstance(mask, torch.Tensor):
        return mask.detach().cpu().float().numpy() > 0.5
    return np.asarray(mask) > 0.5


def _load_comfyui_sam3_modules():
    package_name = "comfyui_sam3_ext"
    nodes_package_name = f"{package_name}.nodes"
    root = Path(__file__).resolve().parent.parent / "external" / "ComfyUI-SAM3"
    nodes_root = root / "nodes"

    if not root.exists():
        raise RuntimeError(f"ComfyUI-SAM3 not found at {root}")

    if package_name not in sys.modules:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [str(root)]
        sys.modules[package_name] = pkg

    if nodes_package_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            nodes_package_name,
            nodes_root / "__init__.py",
            submodule_search_locations=[str(nodes_root)],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[nodes_package_name] = module
        spec.loader.exec_module(module)

    model_cache = importlib.import_module(f"{nodes_package_name}._model_cache")
    utils = importlib.import_module(f"{nodes_package_name}.utils")
    return model_cache, utils


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
        try:
            model_cache, sam3_utils = _load_comfyui_sam3_modules()
        except Exception as e:
            raise RuntimeError(f"Unable to import ComfyUI-SAM3 helpers: {e}")
        get_or_build_model = model_cache.get_or_build_model
        comfy_image_to_pil = sam3_utils.comfy_image_to_pil
        masks_to_comfy_mask = sam3_utils.masks_to_comfy_mask

        import comfy.model_management

        if images.ndim != 4 or black_mask_video.ndim != 4:
            raise ValueError("`images` and `black_mask_video` must be IMAGE batches [T,H,W,C].")
        frame_count, frame_height, frame_width, _ = images.shape
        if black_mask_video.shape[:3] != images.shape[:3]:
            raise ValueError(
                f"Shape mismatch: images {tuple(images.shape)} vs black_mask_video {tuple(black_mask_video.shape)}."
            )

        analysis = json.loads(affected_objects_json) if affected_objects_json.strip().startswith("{") else {
            "affected_objects": json.loads(affected_objects_json)
        }
        affected_objects = analysis.get("affected_objects", [])

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

            # Visual artifacts use grid localizations directly.
            if category == "visual_artifact" and obj.get("grid_localizations"):
                frame_masks = [np.zeros((frame_height, frame_width), dtype=bool) for _ in range(frame_count)]
                for loc in obj.get("grid_localizations", []):
                    frame_idx = max(0, min(frame_count - 1, int(loc.get("frame", 0))))
                    frame_masks[frame_idx] |= _grid_cells_to_mask(
                        loc.get("grid_regions", []), grid_rows, grid_cols, frame_width, frame_height
                    )
                # Spread sparse artifact references across neighbouring frames.
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

            # Moving objects use trajectories if available.
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


class VoidCombineQuadmask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "black_mask_video": ("IMAGE",),
                "grey_mask_video": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("quadmask_video", "quadmask_mask")
    FUNCTION = "combine"
    CATEGORY = "VOID"

    def combine(self, black_mask_video, grey_mask_video):
        if black_mask_video.ndim != 4 or grey_mask_video.ndim != 4:
            raise ValueError("`black_mask_video` and `grey_mask_video` must be IMAGE batches [T,H,W,C].")
        if black_mask_video.shape[:3] != grey_mask_video.shape[:3]:
            raise ValueError(
                f"Shape mismatch: black {tuple(black_mask_video.shape)} vs grey {tuple(grey_mask_video.shape)}."
            )

        black = black_mask_video[..., 0].detach().cpu().float()
        grey = grey_mask_video[..., 0].detach().cpu().float()

        black_object = black < 0.5
        grey_affected = grey < 0.75

        quad = torch.ones_like(black)
        quad[black_object & ~grey_affected] = 0.0
        quad[~black_object & grey_affected] = 127.0 / 255.0
        quad[black_object & grey_affected] = 63.0 / 255.0

        # Match original Stage 4 behavior: remove grey-only values on first frame.
        if quad.shape[0] > 0:
            first = quad[0]
            grey_only = torch.isclose(first, torch.tensor(127.0 / 255.0), atol=1e-4)
            first[grey_only] = 1.0
            quad[0] = first

        quadmask_video = quad.unsqueeze(-1).repeat(1, 1, 1, 3)
        return (quadmask_video, quad)


NODE_CLASS_MAPPINGS = {
    "VoidExportBlackMask": VoidExportBlackMask,
    "VoidPrepareVLMAnalysis": VoidPrepareVLMAnalysis,
    "VoidParseVLMAnalysis": VoidParseVLMAnalysis,
    "VoidBuildGreyMask": VoidBuildGreyMask,
    "VoidCombineQuadmask": VoidCombineQuadmask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VoidExportBlackMask": "VOID Export Black Mask",
    "VoidPrepareVLMAnalysis": "VOID Prepare VLM Analysis",
    "VoidParseVLMAnalysis": "VOID Parse VLM Analysis",
    "VoidBuildGreyMask": "VOID Build Grey Mask",
    "VoidCombineQuadmask": "VOID Combine Quadmask",
}
