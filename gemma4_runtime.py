import os
from typing import Dict, List, Tuple

import numpy as np
import torch


DEFAULT_MODEL_ID = os.environ.get("VOID_GEMMA4_MODEL_ID", "google/gemma-4-e2b-it")
DEFAULT_MAX_FRAMES = int(os.environ.get("VOID_GEMMA4_MAX_FRAMES", "24"))
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("VOID_GEMMA4_MAX_NEW_TOKENS", "256"))

_MODEL_CACHE: Dict[str, object] = {}


def _require_dependencies():
    try:
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "Gemma 4 video inference requires `transformers`, `Pillow`, and a compatible PyTorch install."
        ) from exc
    return Image, AutoModelForImageTextToText, AutoProcessor


def _get_execution_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_load_dtype(device: str):
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def _get_cached_model():
    cache_key = DEFAULT_MODEL_ID
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    _, AutoModelForImageTextToText, AutoProcessor = _require_dependencies()
    load_device = _get_execution_device()
    load_dtype = _get_load_dtype(load_device)

    processor = AutoProcessor.from_pretrained(DEFAULT_MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        DEFAULT_MODEL_ID,
        torch_dtype=load_dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to("cpu")

    cached = {
        "processor": processor,
        "model": model,
        "model_id": DEFAULT_MODEL_ID,
    }
    _MODEL_CACHE[cache_key] = cached
    return cached


def _sample_frame_indices(frame_count: int, max_frames: int) -> List[int]:
    if frame_count <= 0:
        raise ValueError("Video batch is empty.")
    if frame_count <= max_frames:
        return list(range(frame_count))

    indices = np.linspace(0, frame_count - 1, num=max_frames)
    deduped = sorted({int(round(index)) for index in indices})
    if deduped[-1] != frame_count - 1:
        deduped[-1] = frame_count - 1
    return deduped


def _tensor_to_pil_frames(video: torch.Tensor, indices: List[int]):
    Image, _, _ = _require_dependencies()
    frames = []
    for index in indices:
        frame = video[index].detach().cpu().float().clamp(0.0, 1.0)
        array = (frame.numpy() * 255.0).round().astype(np.uint8)
        frames.append(Image.fromarray(array))
    return frames


def _move_batch_to_device(batch, device: str):
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _decode_response(processor, generated_ids: torch.Tensor, input_ids: torch.Tensor) -> str:
    prompt_length = input_ids.shape[-1]
    completion_ids = generated_ids[:, prompt_length:]
    decoded = processor.batch_decode(completion_ids, skip_special_tokens=True)
    return decoded[0].strip() if decoded else ""


def offload_model():
    for cached in _MODEL_CACHE.values():
        model = cached["model"]
        model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_video_inference(
    video: torch.Tensor,
    prompt: str,
    max_frames: int = DEFAULT_MAX_FRAMES,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> Tuple[str, int]:
    if video.ndim != 4:
        raise ValueError(f"`video` must be an IMAGE batch [T,H,W,C], got {tuple(video.shape)}.")
    if video.shape[-1] != 3:
        raise ValueError(f"`video` must have 3 channels in the last dimension, got {tuple(video.shape)}.")

    prompt = str(prompt).strip()
    if not prompt:
        raise ValueError("`prompt` must not be empty.")
    if max_frames <= 0:
        raise ValueError("`max_frames` must be greater than 0.")
    if max_new_tokens <= 0:
        raise ValueError("`max_new_tokens` must be greater than 0.")

    cached = _get_cached_model()
    processor = cached["processor"]
    model = cached["model"]
    device = _get_execution_device()
    frame_indices = _sample_frame_indices(video.shape[0], max_frames)
    sampled_frames = _tensor_to_pil_frames(video, frame_indices)

    messages = [
        {
            "role": "user",
            "content": [{"type": "video", "video": sampled_frames}, {"type": "text", "text": prompt}],
        }
    ]

    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    model_inputs = processor(
        text=text,
        videos=[sampled_frames],
        return_tensors="pt",
        num_frames=len(sampled_frames),
    )

    try:
        model.to(device)
        model_inputs = _move_batch_to_device(model_inputs, device)
        with torch.inference_mode():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        response = _decode_response(processor, generated_ids, model_inputs["input_ids"])
        return response, len(frame_indices)
    finally:
        del model_inputs
        if "generated_ids" in locals():
            del generated_ids
        offload_model()
