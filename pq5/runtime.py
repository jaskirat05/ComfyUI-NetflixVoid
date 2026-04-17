from pathlib import Path
from typing import Dict

import torch
from diffusers import (
    CogVideoXDDIMScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    PNDMScheduler,
)

from .config import create_runtime_config
from .dist import set_multi_gpus_devices
from .models import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel, T5EncoderModel, T5Tokenizer
from .pipeline import CogVideoXFunInpaintPipeline, CogVideoXFunPipeline
from .utils.fp8_optimization import convert_weight_dtype_wrapper

_PIPELINE_CACHE: Dict[str, Dict[str, object]] = {}


def _patch_transformers_hybridcache():
    import transformers

    if hasattr(transformers, "HybridCache"):
        return

    from transformers.cache_utils import DynamicCache

    class HybridCache(DynamicCache):
        def __init__(
            self,
            config,
            max_batch_size=None,
            max_cache_len=None,
            dtype=None,
            device=None,
            offloading=False,
            offload_only_non_sliding=False,
            **kwargs,
        ):
            super().__init__(
                config=config,
                offloading=offloading,
                offload_only_non_sliding=offload_only_non_sliding,
            )

    transformers.HybridCache = HybridCache
    try:
        import transformers.cache_utils as cache_utils

        cache_utils.HybridCache = HybridCache
    except Exception:
        pass


def _patch_transformers_utils_constants():
    try:
        import transformers.utils as transformers_utils
    except Exception:
        return

    if not hasattr(transformers_utils, "FLAX_WEIGHTS_NAME"):
        transformers_utils.FLAX_WEIGHTS_NAME = "flax_model.msgpack"


def _resolve_transformer_path(void_root: Path, use_pass2: bool) -> Path:
    filename = "void_pass2.safetensors" if use_pass2 else "void_pass1.safetensors"
    transformer_path = void_root / filename
    if not transformer_path.exists():
        raise RuntimeError(f"Expected VOID checkpoint at `{transformer_path}`, but it was not found.")
    return transformer_path


def _validate_base_model(base_model: Path):
    required = ["transformer/config.json", "vae/config.json", "tokenizer", "scheduler/scheduler_config.json"]
    missing = [item for item in required if not (base_model / item).exists()]
    if missing:
        raise RuntimeError(
            "VOID base model directory is missing required files/folders: "
            f"{', '.join(missing)} under `{base_model}`"
        )


def _load_transformer_state_dict(transformer_path: str):
    if transformer_path.endswith("safetensors"):
        from safetensors.torch import load_file

        state_dict = load_file(transformer_path)
    else:
        state_dict = torch.load(transformer_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        return state_dict["state_dict"]
    return state_dict


def _load_state_dict_from_path(path: str):
    if path.endswith("safetensors"):
        from safetensors.torch import load_file

        state_dict = load_file(path)
    else:
        state_dict = torch.load(path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        return state_dict["state_dict"]
    return state_dict


def _build_pipeline(config):
    model_name = config.video_model.model_name
    weight_dtype = config.system.weight_dtype
    device = set_multi_gpus_devices(config.system.ulysses_degree, config.system.ring_degree)

    transformer = CogVideoXTransformer3DModel.from_pretrained(
        model_name,
        subfolder="transformer",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float8_e4m3fn
        if config.system.gpu_memory_mode == "model_cpu_offload_and_qfloat8"
        else weight_dtype,
        use_vae_mask=config.video_model.use_vae_mask,
        stack_mask=config.video_model.stack_mask,
    ).to(weight_dtype)

    if config.video_model.transformer_path:
        state_dict = _load_transformer_state_dict(config.video_model.transformer_path)

        param_name = "patch_embed.proj.weight"
        if (
            (config.video_model.use_vae_mask or config.video_model.stack_mask)
            and param_name in state_dict
            and state_dict[param_name].size(1) != transformer.state_dict()[param_name].size(1)
        ):
            latent_ch = 16
            feat_scale = 8
            feat_dim = int(latent_ch * feat_scale)
            new_weight = transformer.state_dict()[param_name].clone()
            new_weight[:, :feat_dim] = state_dict[param_name][:, :feat_dim]
            new_weight[:, -feat_dim:] = state_dict[param_name][:, -feat_dim:]
            state_dict[param_name] = new_weight

        transformer.load_state_dict(state_dict, strict=False)

    vae_config = AutoencoderKLCogVideoX.load_config(model_name, subfolder="vae")
    vae = AutoencoderKLCogVideoX.from_config(vae_config).to(weight_dtype)
    if not config.video_model.vae_path:
        raise RuntimeError("VAE checkpoint path is required. Select a VAE from ComfyUI/models/vae.")
    vae_sd = _load_state_dict_from_path(config.video_model.vae_path)
    vae.load_state_dict(vae_sd, strict=False)

    tokenizer = T5Tokenizer.from_pretrained(model_name, subfolder="tokenizer")
    if config.video_model.text_encoder_path:
        text_encoder_source = Path(config.video_model.text_encoder_path)
        if text_encoder_source.is_dir():
            text_encoder = T5EncoderModel.from_pretrained(str(text_encoder_source), torch_dtype=weight_dtype)
        else:
            text_encoder = T5EncoderModel.from_pretrained(model_name, subfolder="text_encoder", torch_dtype=weight_dtype)
            te_sd = _load_state_dict_from_path(config.video_model.text_encoder_path)
            text_encoder.load_state_dict(te_sd, strict=False)
    else:
        text_encoder = T5EncoderModel.from_pretrained(model_name, subfolder="text_encoder", torch_dtype=weight_dtype)

    scheduler_cls = {
        "Euler": EulerDiscreteScheduler,
        "Euler A": EulerAncestralDiscreteScheduler,
        "DPM++": DPMSolverMultistepScheduler,
        "PNDM": PNDMScheduler,
        "DDIM_Cog": CogVideoXDDIMScheduler,
        "DDIM_Origin": DDIMScheduler,
    }[config.video_model.sampler_name]
    scheduler = scheduler_cls.from_pretrained(model_name, subfolder="scheduler")

    if transformer.config.in_channels != vae.config.latent_channels:
        pipeline = CogVideoXFunInpaintPipeline(
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
        )
    else:
        pipeline = CogVideoXFunPipeline(
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
        )

    if config.system.ulysses_degree > 1 or config.system.ring_degree > 1:
        transformer.enable_multi_gpus_inference()

    if config.system.gpu_memory_mode == "sequential_cpu_offload":
        pipeline.enable_sequential_cpu_offload(device=device)
    elif config.system.gpu_memory_mode == "model_cpu_offload_and_qfloat8":
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        pipeline.enable_model_cpu_offload(device=device)
    elif config.system.gpu_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    else:
        pipeline.to(device=device)

    generator = torch.Generator(device=device).manual_seed(config.system.seed)
    return pipeline, vae, generator


def load_pipeline_bundle(void_root: Path, base_model: Path, use_pass2: bool):
    cache_key = f"{void_root}:{base_model}:{int(use_pass2)}"
    cached = _PIPELINE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    _patch_transformers_hybridcache()
    _patch_transformers_utils_constants()

    if not void_root.exists():
        raise RuntimeError(f"VOID root path does not exist: `{void_root}`")

    _validate_base_model(base_model)
    transformer_path = _resolve_transformer_path(void_root, use_pass2)

    config = create_runtime_config(str(base_model), str(transformer_path))
    pipeline, vae, generator = _build_pipeline(config)

    cached = {
        "config": config,
        "pipeline": pipeline,
        "vae": vae,
        "generator": generator,
    }
    _PIPELINE_CACHE[cache_key] = cached
    return cached


def load_pipeline_bundle_from_checkpoint(
    base_model: Path,
    transformer_checkpoint: Path,
    vae_checkpoint: Path | None = None,
    text_encoder_checkpoint: Path | None = None,
):
    checkpoint_path = Path(transformer_checkpoint)
    vae_path = Path(vae_checkpoint).resolve() if vae_checkpoint is not None else None
    text_encoder_path = Path(text_encoder_checkpoint).resolve() if text_encoder_checkpoint is not None else None
    cache_key = f"{base_model}:{checkpoint_path.resolve()}:{vae_path}:{text_encoder_path}"
    cached = _PIPELINE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    _patch_transformers_hybridcache()
    _patch_transformers_utils_constants()

    _validate_base_model(base_model)
    if not checkpoint_path.exists():
        raise RuntimeError(f"Selected checkpoint does not exist: `{checkpoint_path}`")
    if vae_path is not None and not vae_path.exists():
        raise RuntimeError(f"Selected VAE checkpoint does not exist: `{vae_path}`")
    if text_encoder_path is not None and not text_encoder_path.exists():
        raise RuntimeError(f"Selected text encoder checkpoint does not exist: `{text_encoder_path}`")

    config = create_runtime_config(
        str(base_model),
        str(checkpoint_path),
        vae_path=str(vae_path) if vae_path is not None else "",
        text_encoder_path=str(text_encoder_path) if text_encoder_path is not None else "",
    )
    pipeline, vae, generator = _build_pipeline(config)

    cached = {
        "config": config,
        "pipeline": pipeline,
        "vae": vae,
        "generator": generator,
    }
    _PIPELINE_CACHE[cache_key] = cached
    return cached


def clear_pipeline_cache():
    for cached in _PIPELINE_CACHE.values():
        pipeline = cached.get("pipeline", None)
        vae = cached.get("vae", None)
        if pipeline is not None:
            try:
                pipeline.to("cpu")
            except Exception:
                pass
        if vae is not None:
            try:
                vae.to("cpu")
            except Exception:
                pass
    _PIPELINE_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
