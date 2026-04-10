# VOID ComfyUI Nodes

This package currently provides five ComfyUI nodes:

- `VOID Load SAM3 Model`
- `VOID Export Black Mask`
- `VOID Prepare VLM Analysis`
- `VOID Parse VLM Analysis`
- `VOID Build Grey Mask`
- `VOID Combine Quadmask`

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
7. Feed `affected_objects_json`, `images`, `black_mask_video`, and `VOID Load SAM3 Model.sam3_model_config` into:
   - `VOID Build Grey Mask`
8. Feed `black_mask_video` and `grey_mask_video` into:
   - `VOID Combine Quadmask`

## Node outputs

### `VOID Export Black Mask`

Converts the propagated foreground mask into VOID Stage 1 tensors:

- object / foreground = `0`
- background = `255`

It returns:

- `black_mask_video` as an `IMAGE` batch
- `black_mask_mask` as a `MASK` batch
- `first_frame` as a single-frame `IMAGE`

### `VOID Load SAM3 Model`

Returns a local `sam3_model_config` object for the vendored SAM3 wrapper.
This package vendors the minimal SAM3 wrapper/model code needed for Stage 3,
so it does not require the separate `ComfyUI-SAM3` node pack to be installed.

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

### `VOID Build Grey Mask`

Builds Stage 3 grey masks using:

- direct grid localization for visual artifacts
- trajectory rasterization for moving objects
- direct SAM3 text grounding on the first frame for physical nouns

It returns:

- `grey_mask_video` as an `IMAGE` batch
- `grey_mask_mask` as a `MASK` batch
- `grey_debug` as a debug overlay image

### `VOID Combine Quadmask`

Combines Stage 1 and Stage 3 outputs into the final quadmask tensor values:

- `0` primary object
- `63` overlap
- `127` affected only
- `255` background

returned as normalized tensors.
