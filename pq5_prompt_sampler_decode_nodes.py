import torch
import torch.nn.functional as F

from .pq5.runtime import clear_pipeline_cache


class VoidPQ5Settings:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 300, "step": 1}),
                "guidance_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0x7FFFFFFF, "step": 1}),
            }
        }

    RETURN_TYPES = ("PQ5_SETTINGS",)
    RETURN_NAMES = ("settings",)
    FUNCTION = "build"
    CATEGORY = "VOID/PQ5"

    def build(self, num_inference_steps, guidance_scale, strength, seed):
        return (
            {
                "num_inference_steps": int(num_inference_steps),
                "guidance_scale": float(guidance_scale),
                "strength": float(strength),
                "seed": int(seed),
            },
        )


class VoidPQ5EncodePrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("PQ5_MODEL",),
                "prompt": ("STRING", {"multiline": True, "default": "empty background"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "guidance_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "max_sequence_length": ("INT", {"default": 226, "min": 1, "max": 1024, "step": 1}),
            }
        }

    RETURN_TYPES = ("PQ5_PROMPT_EMBEDS", "PQ5_PROMPT_EMBEDS")
    RETURN_NAMES = ("prompt_embeds", "negative_prompt_embeds")
    FUNCTION = "encode"
    CATEGORY = "VOID/PQ5"

    def encode(self, model, prompt, negative_prompt, guidance_scale, max_sequence_length):
        pipeline = model["bundle"]["pipeline"]
        do_cfg = float(guidance_scale) > 1.0
        prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
            prompt=str(prompt),
            negative_prompt=str(negative_prompt),
            do_classifier_free_guidance=do_cfg,
            num_videos_per_prompt=1,
            max_sequence_length=int(max_sequence_length),
            device=pipeline._execution_device,
        )
        return (prompt_embeds, negative_prompt_embeds)


class VoidPQ5Sampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("PQ5_MODEL",),
                "encoded_video": ("PQ5_VIDEO_TENSOR",),
                "encoded_quadmask": ("PQ5_MASK_TENSOR",),
                "prompt_embeds": ("PQ5_PROMPT_EMBEDS",),
                "negative_prompt_embeds": ("PQ5_PROMPT_EMBEDS",),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 300, "step": 1}),
                "guidance_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0x7FFFFFFF, "step": 1}),
            },
            "optional": {
                "settings": ("PQ5_SETTINGS",),
            },
        }

    RETURN_TYPES = ("PQ5_LATENTS",)
    RETURN_NAMES = ("latents",)
    FUNCTION = "sample"
    CATEGORY = "VOID/PQ5"

    def sample(
        self,
        model,
        encoded_video,
        encoded_quadmask,
        prompt_embeds,
        negative_prompt_embeds,
        num_inference_steps,
        guidance_scale,
        strength,
        seed,
        settings=None,
    ):
        bundle = model["bundle"]
        config = bundle["config"]
        pipeline = bundle["pipeline"]
        generator = bundle["generator"]

        if settings is not None:
            num_inference_steps = int(settings.get("num_inference_steps", num_inference_steps))
            guidance_scale = float(settings.get("guidance_scale", guidance_scale))
            strength = float(settings.get("strength", strength))
            seed = int(settings.get("seed", seed))

        generator.manual_seed(int(seed))

        if encoded_video.ndim != 5 or encoded_quadmask.ndim != 5:
            raise ValueError(
                "Expected `encoded_video` and `encoded_quadmask` to be 5D tensors [B,C,T,H,W]. "
                f"Got {tuple(encoded_video.shape)} and {tuple(encoded_quadmask.shape)}."
            )

        if int(encoded_video.shape[2]) != int(encoded_quadmask.shape[2]):
            raise ValueError(
                "Encoded video and mask must have the same frame count. "
                f"Got video T={int(encoded_video.shape[2])}, mask T={int(encoded_quadmask.shape[2])}. "
                "Use matching max_video_length/temporal_window_size settings."
            )

        sample_size = (int(encoded_quadmask.shape[-2]), int(encoded_quadmask.shape[-1]))
        if sample_size[0] % 8 != 0 or sample_size[1] % 8 != 0:
            raise ValueError(
                f"Mask resolution must be divisible by 8, got HxW={sample_size[0]}x{sample_size[1]}."
            )

        if tuple(int(x) for x in encoded_video.shape[-2:]) != sample_size:
            b, c, t, _, _ = encoded_video.shape
            video_frames = encoded_video.permute(0, 2, 1, 3, 4).reshape(b * t, c, encoded_video.shape[-2], encoded_video.shape[-1])
            video_frames = F.interpolate(video_frames, size=sample_size, mode="area")
            encoded_video = video_frames.reshape(b, t, c, sample_size[0], sample_size[1]).permute(0, 2, 1, 3, 4)

        result = pipeline(
            prompt=None,
            negative_prompt=None,
            height=sample_size[0],
            width=sample_size[1],
            video=encoded_video,
            mask_video=encoded_quadmask,
            num_frames=int(config.video_model.temporal_window_size),
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            output_type="latent",
            return_dict=True,
            strength=float(strength),
            use_trimask=True,
            zero_out_mask_region=bool(config.video_model.zero_out_mask_region),
            skip_unet=bool(config.experiment.skip_unet),
            use_vae_mask=bool(config.video_model.use_vae_mask),
            stack_mask=bool(config.video_model.stack_mask),
        )
        latents = result.videos
        return (latents,)


class VoidPQ5DecodeVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("PQ5_MODEL",),
                "latents": ("PQ5_LATENTS",),
                "original_frame_count": ("INT", {"default": 1, "min": 1, "max": 100000, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("video",)
    FUNCTION = "decode"
    CATEGORY = "VOID/PQ5"

    def decode(self, model, latents, original_frame_count):
        pipeline = model["bundle"]["pipeline"]
        decoded = pipeline.decode_latents(latents)
        if not isinstance(decoded, torch.Tensor):
            decoded = torch.from_numpy(decoded)

        # decoded: [B,C,T,H,W] -> Comfy IMAGE [T,H,W,C]
        video = decoded[0].detach().cpu().float().permute(1, 2, 3, 0).clamp(0.0, 1.0)
        video = video[: int(original_frame_count)]
        return (video,)


class VoidPQ5UnloadCache:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "unload"
    CATEGORY = "VOID/PQ5"

    def unload(self):
        clear_pipeline_cache()
        return ("PQ5 cache cleared",)
