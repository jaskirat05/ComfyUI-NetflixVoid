# VOID ComfyUI Nodes

This package currently provides three minimal ComfyUI nodes:

- `VOID Export Black Mask`
- `VOID Prepare VLM Analysis`
- `VOID Parse VLM Analysis`

## Intended wiring

1. Use KJ `PointsEditor` to place points.
2. Use Kijai SAM2.1 video segmentation nodes to produce the propagated `MASK`.
3. Feed the original video `IMAGE` batch and the propagated `MASK` into:
   - `VOID Export Black Mask`
4. Feed the original video and `black_mask_video` into:
   - `VOID Prepare VLM Analysis`
5. Feed `qwen_input_frames` and `prompt` into the `1038lab` `QwenVL` node.
6. Feed the raw text output from `QwenVL` into:
   - `VOID Parse VLM Analysis`

## Node outputs

### `VOID Export Black Mask`

Converts the propagated foreground mask into VOID Stage 1 tensors:

- object / foreground = `0`
- background = `255`

It returns:

- `black_mask_video` as an `IMAGE` batch
- `black_mask_mask` as a `MASK` batch
- `first_frame` as a single-frame `IMAGE`

### `VOID Prepare VLM Analysis`

Builds the Stage 2 reference frames:

- a masked + gridded first frame
- multiple gridded sample frames across the clip
- a Stage 2 prompt string for QwenVL

### `VOID Parse VLM Analysis`

Cleans and validates QwenVL's JSON response and returns:

- normalized `analysis_json`
- `integral_belongings_json`
- `affected_objects_json`
- `scene_description`
- `confidence`
