# VOID Stage 1 ComfyUI Node

This package provides a minimal ComfyUI node for exporting a propagated
foreground `MASK` tensor into VOID Stage 1 assets:

- `black_mask.mp4`
- `first_frame.jpg`

## Intended wiring

1. Use KJ `PointsEditor` to place points.
2. Use Kijai SAM2.1 video segmentation nodes to produce the propagated `MASK`.
3. Feed the original video `IMAGE` batch and the propagated `MASK` into:
   - `VOID Export Black Mask`

## Output convention

The exported video follows VOID Stage 1 mask format:

- object / foreground = `0`
- background = `255`

These files are meant to feed the later VOID VLM / grey-mask / quadmask stages.
