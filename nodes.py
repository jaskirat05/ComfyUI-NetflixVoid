import json

import cv2
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


NODE_CLASS_MAPPINGS = {
    "VoidExportBlackMask": VoidExportBlackMask,
    "VoidPrepareVLMAnalysis": VoidPrepareVLMAnalysis,
    "VoidParseVLMAnalysis": VoidParseVLMAnalysis,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VoidExportBlackMask": "VOID Export Black Mask",
    "VoidPrepareVLMAnalysis": "VOID Prepare VLM Analysis",
    "VoidParseVLMAnalysis": "VOID Parse VLM Analysis",
}
