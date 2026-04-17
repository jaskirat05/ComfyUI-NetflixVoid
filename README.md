# ComfyUI-NetflixVoid

Custom ComfyUI nodes for the VOID workflow:

1. Stage 1: black mask
2. Stage 2: VLM prep + parse
3. Stage 3: grey mask
4. Stage 4: quadmask
5. Stage 5: modular PQ5 inference

## Included nodes

- `VOID Load SAM3 Model`
- `VOID Export Black Mask`
- `VOID Prepare VLM Analysis`
- `VOID Parse VLM Analysis`
- `VOID Build Grey Mask`
- `VOID Combine Quadmask`
- `VOID Gemma 4 E2B Video Prompt`
- `VOID PQ5 Load Model`
- `VOID PQ5 Settings`
- `VOID PQ5 Encode Prompt`
- `VOID PQ5 Encode Video`
- `VOID PQ5 Encode Quadmask`
- `VOID PQ5 Sampler`
- `VOID PQ5 Decode Video`
- `VOID PQ5 Unload Cache`

## PQ5 assets and models

This repo now uses a modular PQ5 layout.

Required locations:

- Transformer checkpoint: `ComfyUI/models/checkpoints` (select in `VOID PQ5 Load Model`)
- VAE checkpoint: `ComfyUI/models/vae` (select in `VOID PQ5 Load Model`)
- Text encoder folder: `ComfyUI/models/text_encoders/void`
- Repo-local PQ5 configs/tokenizer/scheduler: `ComfyUI-NetflixVoid/pq5_assets`

## Recommended modular wiring (PQ5)

1. `VOID PQ5 Load Model` -> `model`
2. `VOID PQ5 Encode Prompt` with `model` + prompt text
3. `VOID PQ5 Encode Video` with `model` + input video
4. `VOID PQ5 Encode Quadmask` with quadmask video
5. `VOID PQ5 Sampler` with `model`, prompt embeds, encoded video, encoded quadmask, and settings
6. `VOID PQ5 Decode Video` with `model`, sampler `latents`, and `original_frame_count`

For pass 2:

- Load a second model using pass-2 checkpoint in `VOID PQ5 Load Model`
- Feed pass-1 output video into `VOID PQ5 Encode Video`
- Reuse/refine quadmask and run sampler+decode again

## Notes

- `VOID PQ5 Load Model` has no fallback for missing checkpoint/VAE/text-encoder paths.
- Use `VOID PQ5 Unload Cache` to clear cached PQ5 pipelines and free memory.
- Keep `grid_rows/grid_cols` from `VOID Prepare VLM Analysis` connected to `VOID Build Grey Mask`.
