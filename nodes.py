import json

import cv2
import torch
from .gemma4_runtime import DEFAULT_MODEL_ID, run_video_inference
from .pq5_model_video_nodes import VoidPQ5LoadModel, VoidPQ5EncodeVideo
from .pq5_prompt_sampler_decode_nodes import (
    VoidPQ5Settings,
    VoidPQ5EncodePrompt,
    VoidPQ5Sampler,
    VoidPQ5DecodeVideo,
    VoidPQ5UnloadCache,
)
from .pq5_quadmask_nodes import VoidPQ5EncodeQuadmask
from .sam3_logic import VoidLoadSAM3Model, VoidBuildGreyMask


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


def _get_vlm_system_prompt():
    return "You are an expert video analyst with deep understanding of physics and object interactions. Always output valid JSON only."


def _make_vlm_analysis_prompt(instruction: str, grid_rows: int, grid_cols: int, has_multi_frame_grids: bool = True):
    if has_multi_frame_grids:
        grid_context = f"""
1. **Multiple Grid Reference Frames**: Sampled frames at 0%, 11%, 22%, 33%, 44%, 56%, 67%, 78%, 89%, 100% of video
   - Each frame shows YELLOW GRID with {grid_rows} rows × {grid_cols} columns
   - Grid cells labeled (row, col) starting from (0, 0) at top-left
   - Frame number shown at bottom
   - Use these to locate objects that appear MID-VIDEO and track object positions across time
2. **First Frame with RED mask**: Shows what will be REMOVED (primary object)
3. **Full Video**: Complete action and interactions"""
    else:
        grid_context = f"""
1. **First Frame with Grid**: PRIMARY OBJECT highlighted in RED + GRID OVERLAY
   - The red overlay shows what will be REMOVED (already masked)
   - Yellow grid with {grid_rows} rows × {grid_cols} columns
   - Grid cells are labeled (row, col) starting from (0, 0) at top-left
2. **Full Video**: Complete scene and action"""

    return f"""
You are an expert video analyst specializing in physics and object interactions.

═══════════════════════════════════════════════════════════════════
CONTEXT
═══════════════════════════════════════════════════════════════════

You will see MULTIPLE inputs:
{grid_context}

Edit instruction: "{instruction}"

IMPORTANT: Some objects may NOT appear in first frame. They may enter later.
Watch the ENTIRE video and note when each object first appears.

═══════════════════════════════════════════════════════════════════
YOUR TASK
═══════════════════════════════════════════════════════════════════

Analyze what would happen if the PRIMARY OBJECT (shown in red) is removed.
Watch the ENTIRE video to see all interactions and movements.

STEP 1: IDENTIFY INTEGRAL BELONGINGS (0-3 items)
─────────────────────────────────────────────────
Items that should be ADDED to the primary removal mask (removed WITH primary object):

✓ INCLUDE:
  • Distinct wearable items: hat, backpack, jacket (if separate/visible)
  • Vehicles/equipment being ridden: bike, skateboard, surfboard, scooter
  • Large carried items that are part of the subject

✗ DO NOT INCLUDE:
  • Generic clothing (shirt, pants, shoes) - already captured with person
  • Held items that could be set down: guitar, cup, phone, tools
  • Objects they're interacting with but not wearing/riding

Examples:
  • Person on bike → integral: "bike"
  • Person with guitar → integral: none (guitar is affected, not integral)
  • Surfer → integral: "surfboard"
  • Boxer → integral: "boxing gloves" (wearable equipment)

STEP 2: IDENTIFY AFFECTED OBJECTS (0-5 objects)
────────────────────────────────────────────────
Objects/effects that are SEPARATE from primary but affected by its removal.

CRITICAL: Do NOT include integral belongings from Step 1.

Two categories:

A) VISUAL ARTIFACTS (disappear when primary removed):
   • shadow, reflection, wake, ripples, splash, footprints
   • These vanish completely - no physics needed

   **CRITICAL FOR VISUAL ARTIFACTS:**
   You MUST provide GRID LOCALIZATIONS across the reference frames.
   Keyword segmentation fails to isolate specific shadows/reflections.

   For each visual artifact:
   - Look at each grid reference frame you were shown
   - Identify which grid cells the artifact occupies in EACH frame
   - List all grid cells (row, col) that contain any part of it
   - Be thorough - include ALL touched cells (over-mask is better than under-mask)

   Format:
   {{
     "noun": "shadow",
     "category": "visual_artifact",
     "grid_localizations": [
       {{"frame": 0, "grid_regions": [{{"row": 6, "col": 3}}, {{"row": 6, "col": 4}}, ...]}},
       {{"frame": 5, "grid_regions": [{{"row": 6, "col": 4}}, ...]}},
       // ... for each reference frame shown
     ]
   }}

B) PHYSICAL OBJECTS (may move, fall, or stay):

   CRITICAL - Understand the difference:

   **SUPPORTING vs ACTING ON:**
   • SUPPORTING = holding UP against gravity → object WILL FALL when removed
     Examples: holding guitar, carrying cup, person sitting on chair
     → will_move: TRUE

   • ACTING ON = touching/manipulating but object rests on stable surface → object STAYS
     Examples: hand crushing can (can on table), hand opening can (can on counter),
              hand pushing object (object on floor)
     → will_move: FALSE

   **Key Questions:**
   1. Is the primary object HOLDING THIS UP against gravity?
      - YES → will_move: true, needs_trajectory: true
      - NO → Check next question

   2. Is this object RESTING ON a stable surface (table, floor, counter)?
      - YES → will_move: false (stays on surface when primary removed)
      - NO → will_move: true

   3. Is the primary object DOING an action TO this object?
      - Opening can, crushing can, pushing button, turning knob
      - When primary removed → action STOPS, object stays in current state
      - will_move: false

   **SPECIAL CASE - Object Currently Moving But Should Have Stayed:**
   If primary object CAUSES another object to move (hitting, kicking, throwing):
   - The object is currently moving in the video
   - But WITHOUT primary, it would have stayed at its original position
   - You MUST provide:
     • "currently_moving": true
     • "should_have_stayed": true
     • "original_position_grid": {{"row": R, "col": C}} - Where it started

   Examples:
   - Golf club hits ball → Ball at tee, then flies (mark original tee position)
   - Person kicks soccer ball → Ball on ground, then rolls (mark original ground position)
   - Hand throws object → Object held, then flies (mark original held position)

   Format:
   {{
     "noun": "golf ball",
     "category": "physical",
     "currently_moving": true,
     "should_have_stayed": true,
     "original_position_grid": {{"row": 6, "col": 7}},
     "why": "ball was stationary until club hit it"
   }}

   For each physical object, determine:
   - **will_move**: true ONLY if object will fall/move when support removed
   - **first_appears_frame**: frame number object first appears (0 if from start)
   - **why**: Brief explanation of relationship to primary object

   IF will_move=TRUE, also provide GRID-BASED TRAJECTORY:
   - **object_size_grids**: {{"rows": R, "cols": C}} - How many grid cells object occupies
     IMPORTANT: Add 1 extra cell padding for safety (better to over-mask than under-mask)
     Example: Object looks 2×1 → report as 3×2

   - **trajectory_path**: List of keyframe positions as grid coordinates
     Format: [{{"frame": N, "grid_row": R, "grid_col": C}}, ...]
     - IMPORTANT: First keyframe should be at first_appears_frame (not frame 0 if object appears later!)
     - Provide 3-5 keyframes spanning from first appearance to end
     - (grid_row, grid_col) is the CENTER position of object at that frame
     - Use the yellow grid reference frames to determine positions
     - For objects appearing mid-video: use the grid samples to locate them
     - Example: Object appears at frame 15, falls to bottom
       [{{"frame": 15, "grid_row": 3, "grid_col": 5}},  ← First appearance
        {{"frame": 25, "grid_row": 6, "grid_col": 5}},  ← Mid-fall
        {{"frame": 35, "grid_row": 9, "grid_col": 5}}]  ← On ground

✓ Objects held/carried at ANY point in video
✓ Objects the primary supports or interacts with
✓ Visual effects visible at any time

✗ Background objects never touched
✗ Other people/animals with no contact
✗ Integral belongings (already in Step 1)

STEP 3: SCENE DESCRIPTION
──────────────────────────
Describe scene WITHOUT the primary object (1-2 sentences).
Focus on what remains and any dynamic changes (falling objects, etc).

═══════════════════════════════════════════════════════════════════
OUTPUT FORMAT (STRICT JSON ONLY)
═══════════════════════════════════════════════════════════════════

EXAMPLES TO LEARN FROM:

Example 1: Person holding guitar
{{
  "affected_objects": [
    {{
      "noun": "guitar",
      "will_move": true,
      "why": "person is SUPPORTING guitar against gravity by holding it",
      "object_size_grids": {{"rows": 3, "cols": 2}},
      "trajectory_path": [
        {{"frame": 0, "grid_row": 4, "grid_col": 5}},
        {{"frame": 15, "grid_row": 6, "grid_col": 5}},
        {{"frame": 30, "grid_row": 8, "grid_col": 6}}
      ]
    }}
  ]
}}

Example 2: Hand crushing can on table
{{
  "affected_objects": [
    {{
      "noun": "can",
      "will_move": false,
      "why": "can RESTS ON TABLE - hand is just acting on it. When hand removed, can stays on table (uncrushed)"
    }}
  ]
}}

Example 3: Hands opening can on counter
{{
  "affected_objects": [
    {{
      "noun": "can",
      "will_move": false,
      "why": "can RESTS ON COUNTER - hands are doing opening action. When hands removed, can stays closed on counter"
    }}
  ]
}}

Example 4: Person sitting on chair
{{
  "affected_objects": [
    {{
      "noun": "chair",
      "will_move": false,
      "why": "chair RESTS ON FLOOR - person sitting on it doesn't make it fall. Chair stays on floor when person removed"
    }}
  ]
}}

Example 5: Person throws ball (ball appears at frame 12)
{{
  "affected_objects": [
    {{
      "noun": "ball",
      "category": "physical",
      "will_move": true,
      "first_appears_frame": 12,
      "why": "ball is SUPPORTED by person's hand, then thrown",
      "object_size_grids": {{"rows": 2, "cols": 2}},
      "trajectory_path": [
        {{"frame": 12, "grid_row": 4, "grid_col": 3}},
        {{"frame": 20, "grid_row": 2, "grid_col": 6}},
        {{"frame": 28, "grid_row": 5, "grid_col": 8}}
      ]
    }}
  ]
}}

Example 6: Person with shadow (shadow needs grid localization)
{{
  "affected_objects": [
    {{
      "noun": "shadow",
      "category": "visual_artifact",
      "why": "cast by person on the floor",
      "will_move": false,
      "first_appears_frame": 0,
      "movement_description": "Disappears entirely as visual artifact",
      "grid_localizations": [
        {{"frame": 0, "grid_regions": [{{"row": 6, "col": 3}}, {{"row": 6, "col": 4}}, {{"row": 7, "col": 3}}, {{"row": 7, "col": 4}}]}},
        {{"frame": 12, "grid_regions": [{{"row": 6, "col": 4}}, {{"row": 6, "col": 5}}, {{"row": 7, "col": 4}}]}},
        {{"frame": 23, "grid_regions": [{{"row": 5, "col": 4}}, {{"row": 6, "col": 4}}, {{"row": 6, "col": 5}}]}},
        {{"frame": 35, "grid_regions": [{{"row": 6, "col": 3}}, {{"row": 6, "col": 4}}, {{"row": 7, "col": 3}}]}},
        {{"frame": 47, "grid_regions": [{{"row": 6, "col": 3}}, {{"row": 7, "col": 3}}, {{"row": 7, "col": 4}}]}}
      ]
    }}
  ]
}}

Example 7: Golf club hits ball (Case 4 - currently moving but should stay)
{{
  "affected_objects": [
    {{
      "noun": "golf ball",
      "category": "physical",
      "currently_moving": true,
      "should_have_stayed": true,
      "original_position_grid": {{"row": 6, "col": 7}},
      "first_appears_frame": 0,
      "why": "ball was stationary on tee until club hit it. Without club, ball would remain at original position."
    }}
  ]
}}

YOUR OUTPUT FORMAT:
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
      "noun": "guitar",
      "category": "physical",
      "why": "person is SUPPORTING guitar against gravity by holding it",
      "will_move": true,
      "first_appears_frame": 0,
      "movement_description": "Will fall from held position to the ground",
      "object_size_grids": {{"rows": 3, "cols": 2}},
      "trajectory_path": [
        {{"frame": 0, "grid_row": 3, "grid_col": 6}},
        {{"frame": 20, "grid_row": 6, "grid_col": 6}},
        {{"frame": 40, "grid_row": 9, "grid_col": 7}}
      ]
    }},
    {{
      "noun": "shadow",
      "category": "visual_artifact",
      "why": "cast by person on floor",
      "will_move": false,
      "first_appears_frame": 0,
      "movement_description": "Disappears entirely as visual artifact"
    }}
  ],
  "scene_description": "An acoustic guitar falling to the ground in an empty room. Natural window lighting.",
  "confidence": 0.85
}}

CRITICAL REMINDERS:
• Watch ENTIRE video before answering
• SUPPORTING vs ACTING ON:
  - Primary HOLDS UP object against gravity → will_move=TRUE (provide grid trajectory)
  - Primary ACTS ON object (crushing, opening) but object on stable surface → will_move=FALSE
  - Object RESTS ON stable surface (table, floor) → will_move=FALSE
• For visual artifacts (shadow, reflection): will_move=false (no trajectory needed)
• For held objects (guitar, cup): will_move=true (MUST provide object_size_grids + trajectory_path)
• For objects on surfaces being acted on (can being crushed, can being opened): will_move=false
• Grid trajectory: Add +1 cell padding to object size (over-mask is better than under-mask)
• Grid trajectory: Use the yellow grid overlay to determine (row, col) positions
• Be conservative - when in doubt, DON'T include
• Output MUST be valid JSON only

GRID INFO: {grid_rows} rows × {grid_cols} columns
EDIT INSTRUCTION: {instruction}
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
        "vlm_prompt",
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
        qwen_input_frames = (
            torch.cat([masked_grid_first_tensor, reference_frames_tensor], dim=0)
            if use_multi_frame_grids
            else reference_frames_tensor
        )
        system_prompt = _get_vlm_system_prompt()
        prompt = _make_vlm_analysis_prompt(
            instruction=instruction,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            has_multi_frame_grids=use_multi_frame_grids,
        )
        vlm_prompt = f"{system_prompt}\n\n{prompt}"
        return (
            masked_grid_first_tensor,
            reference_frames_tensor,
            qwen_input_frames,
            vlm_prompt,
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


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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
        confidence = max(0.0, min(1.0, _safe_float(parsed.get("confidence", 0.0), 0.0)))
        result = {
            "edit_instruction": str(parsed.get("edit_instruction", "")).strip(),
            "integral_belongings": [],
            "affected_objects": [],
            "scene_description": str(parsed.get("scene_description", "")).strip(),
            "confidence": confidence,
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
                "first_appears_frame": _safe_int(item.get("first_appears_frame", 0), 0),
                "movement_description": str(item.get("movement_description", "")).strip()[:300],
            }

            if "currently_moving" in item:
                obj["currently_moving"] = bool(item.get("currently_moving", False))
            if "should_have_stayed" in item:
                obj["should_have_stayed"] = bool(item.get("should_have_stayed", False))
            if "original_position_grid" in item:
                grid = item.get("original_position_grid", {})
                obj["original_position_grid"] = {
                    "row": _safe_int(grid.get("row", 0), 0),
                    "col": _safe_int(grid.get("col", 0), 0),
                }
            if "grid_localizations" in item:
                grid_localizations = []
                for loc in item.get("grid_localizations", []):
                    frame_loc = {"frame": _safe_int(loc.get("frame", 0), 0), "grid_regions": []}
                    for region in loc.get("grid_regions", []):
                        frame_loc["grid_regions"].append(
                            {
                                "row": _safe_int(region.get("row", 0), 0),
                                "col": _safe_int(region.get("col", 0), 0),
                            }
                        )
                    if frame_loc["grid_regions"]:
                        grid_localizations.append(frame_loc)
                if grid_localizations:
                    obj["grid_localizations"] = grid_localizations
            if obj["will_move"] and "object_size_grids" in item and "trajectory_path" in item:
                size_grids = item.get("object_size_grids", {})
                obj["object_size_grids"] = {
                    "rows": _safe_int(size_grids.get("rows", 2), 2),
                    "cols": _safe_int(size_grids.get("cols", 2), 2),
                }
                trajectory = []
                for point in item.get("trajectory_path", []):
                    trajectory.append(
                        {
                            "frame": _safe_int(point.get("frame", 0), 0),
                            "grid_row": _safe_int(point.get("grid_row", 0), 0),
                            "grid_col": _safe_int(point.get("grid_col", 0), 0),
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


class VoidGemma4VideoPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "Describe this video in detail."}),
                "max_frames": ("INT", {"default": 24, "min": 1, "max": 256, "step": 1}),
                "max_new_tokens": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate"
    CATEGORY = "VOID"

    def generate(self, video, prompt, max_frames, max_new_tokens):
        text, sampled_frames = run_video_inference(
            video=video,
            prompt=prompt,
            max_frames=max_frames,
            max_new_tokens=max_new_tokens,
        )
        if not text:
            raise RuntimeError(
                f"Gemma 4 returned an empty response for model `{DEFAULT_MODEL_ID}` using {sampled_frames} sampled frames."
            )
        return (text,)


NODE_CLASS_MAPPINGS = {
    "VoidLoadSAM3Model": VoidLoadSAM3Model,
    "VoidExportBlackMask": VoidExportBlackMask,
    "VoidPrepareVLMAnalysis": VoidPrepareVLMAnalysis,
    "VoidParseVLMAnalysis": VoidParseVLMAnalysis,
    "VoidBuildGreyMask": VoidBuildGreyMask,
    "VoidCombineQuadmask": VoidCombineQuadmask,
    "VoidGemma4VideoPrompt": VoidGemma4VideoPrompt,
    "VoidPQ5LoadModel": VoidPQ5LoadModel,
    "VoidPQ5Settings": VoidPQ5Settings,
    "VoidPQ5EncodePrompt": VoidPQ5EncodePrompt,
    "VoidPQ5EncodeVideo": VoidPQ5EncodeVideo,
    "VoidPQ5EncodeQuadmask": VoidPQ5EncodeQuadmask,
    "VoidPQ5Sampler": VoidPQ5Sampler,
    "VoidPQ5DecodeVideo": VoidPQ5DecodeVideo,
    "VoidPQ5UnloadCache": VoidPQ5UnloadCache,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VoidLoadSAM3Model": "VOID Load SAM3 Model",
    "VoidExportBlackMask": "VOID Export Black Mask",
    "VoidPrepareVLMAnalysis": "VOID Prepare VLM Analysis",
    "VoidParseVLMAnalysis": "VOID Parse VLM Analysis",
    "VoidBuildGreyMask": "VOID Build Grey Mask",
    "VoidCombineQuadmask": "VOID Combine Quadmask",
    "VoidGemma4VideoPrompt": "VOID Gemma 4 E2B Video Prompt",
    "VoidPQ5LoadModel": "VOID PQ5 Load Model",
    "VoidPQ5Settings": "VOID PQ5 Settings",
    "VoidPQ5EncodePrompt": "VOID PQ5 Encode Prompt",
    "VoidPQ5EncodeVideo": "VOID PQ5 Encode Video",
    "VoidPQ5EncodeQuadmask": "VOID PQ5 Encode Quadmask",
    "VoidPQ5Sampler": "VOID PQ5 Sampler",
    "VoidPQ5DecodeVideo": "VOID PQ5 Decode Video",
    "VoidPQ5UnloadCache": "VOID PQ5 Unload Cache",
}
