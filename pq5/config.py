from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch


@dataclass
class SystemConfig:
    weight_dtype: torch.dtype = torch.bfloat16
    seed: int = 42
    gpu_memory_mode: str = "model_cpu_offload_and_qfloat8"
    ulysses_degree: int = 1
    ring_degree: int = 1


@dataclass
class DataConfig:
    sample_size: Tuple[int, int] = (384, 672)
    max_video_length: int = 197
    fps: int = 12


@dataclass
class VideoModelConfig:
    model_name: str = ""
    transformer_path: str = ""
    vae_path: str = ""
    text_encoder_path: str = ""
    sampler_name: str = "DDIM_Origin"
    denoise_strength: float = 1.0
    negative_prompt: str = (
        "The video is not of a high quality, it has a low resolution. "
        "Watermark present in each frame. The background is solid. "
        "Strange body and strange trajectory. Distortion."
    )
    guidance_scale: float = 1.0
    num_inference_steps: int = 50
    temporal_window_size: int = 85
    use_vae_mask: bool = True
    stack_mask: bool = False
    zero_out_mask_region: bool = False


@dataclass
class ExperimentConfig:
    skip_unet: bool = False


@dataclass
class RuntimeConfig:
    system: SystemConfig = field(default_factory=SystemConfig)
    data: DataConfig = field(default_factory=DataConfig)
    video_model: VideoModelConfig = field(default_factory=VideoModelConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)



def create_runtime_config(
    base_model: str,
    transformer_path: str,
    vae_path: str = "",
    text_encoder_path: str = "",
) -> RuntimeConfig:
    config = RuntimeConfig()
    config.video_model.model_name = str(base_model)
    config.video_model.transformer_path = str(transformer_path)
    config.video_model.vae_path = str(vae_path or "")
    config.video_model.text_encoder_path = str(text_encoder_path or "")
    return config
