"""
SAM3 model cache for subprocess worker.

Since isolation env nodes run in a persistent subprocess, we cache the model
at module level to avoid rebuilding on every node execution.
"""
import logging
import torch

log = logging.getLogger("sam3")

_cache = {"config_hash": None, "model": None}


def get_or_build_model(config):
    """Get cached model or build from config dict.

    Args:
        config: dict with keys: precision, compile, checkpoint_path, bpe_path

    Returns:
        SAM3UnifiedModel (ModelPatcher subclass)
    """
    config_hash = hash(frozenset(config.items()))
    if _cache["config_hash"] == config_hash and _cache["model"] is not None:
        log.info("SAM3 model cache hit")
        return _cache["model"]

    log.info("SAM3 model cache miss — building model...")

    from .sam3_model_patcher import SAM3UnifiedModel
    from .sam3.predictor import Sam3VideoPredictor
    from .sam3.utils import Sam3Processor
    from .sam3 import build_sam3_video_model, _load_checkpoint_file, remap_video_checkpoint
    import comfy.model_management
    import comfy.utils

    load_device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()

    checkpoint_path = config["checkpoint_path"]
    bpe_path = config["bpe_path"]
    compile_model = config.get("compile", False)

    # Resolve dtype
    dtype_str = config["dtype"]
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_str]

    log.info(f"Loading model from: {checkpoint_path}")
    if compile_model:
        log.info("torch.compile enabled")

    # Meta-device construction (avoids 2x RAM)
    log.info("Constructing model on meta device (zero memory)...")
    with torch.device("meta"):
        model = build_sam3_video_model(
            checkpoint_path=None,
            load_from_HF=False,
            bpe_path=bpe_path,
            enable_inst_interactivity=True,
            compile=compile_model,
            skip_checkpoint=True,
        )

    # Load checkpoint and remap keys
    log.info("Loading checkpoint into meta model with assign=True...")
    ckpt = _load_checkpoint_file(str(checkpoint_path))
    remapped_ckpt = remap_video_checkpoint(ckpt, enable_inst_interactivity=True)
    del ckpt

    missing_keys, unexpected_keys = model.load_state_dict(
        remapped_ckpt, strict=False, assign=True,
    )
    del remapped_ckpt
    if missing_keys:
        log.debug("SAM3: %d missing keys during load", len(missing_keys))
    if unexpected_keys:
        log.debug("SAM3: %d unexpected keys during load", len(unexpected_keys))

    # Fix leftover meta-device buffers.
    # Non-persistent buffers (like causal attention masks) are NOT saved in the
    # checkpoint, so they stay on meta device after load_state_dict(assign=True).
    # We must rebuild them rather than blindly zeroing, since some (like
    # attn_mask) need specific values (e.g. -inf upper triangle for causality).
    for name, buf in list(model.named_buffers()):
        if buf.device.type == "meta":
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            attr_name = parts[-1]
            # Try to rebuild the buffer via its owner's builder method
            if attr_name == "attn_mask" and hasattr(parent, "build_causal_mask"):
                parent._buffers[attr_name] = parent.build_causal_mask()
            else:
                parent._buffers[attr_name] = torch.zeros_like(buf, device="cpu")

    model.eval()

    # Wrap in predictor
    video_predictor = Sam3VideoPredictor(
        bpe_path=bpe_path,
        enable_inst_interactivity=True,
        compile=compile_model,
        model=model,
    )

    # Set attention dtype
    from .sam3.attention import set_sam3_dtype
    set_sam3_dtype(dtype if dtype != torch.float32 else None)

    # Selective weight casting
    if dtype != torch.float32:
        import os
        detector = video_predictor.model.detector
        for param in detector.backbone.parameters():
            param.data = param.data.to(dtype=dtype)
        if detector.inst_interactive_predictor is not None:
            for param in detector.inst_interactive_predictor.parameters():
                param.data = param.data.to(dtype=dtype)
        if os.environ.get("DEBUG_COMFYUI_SAM3", "").lower() in ("1", "true", "yes"):
            log.warning(
                "SAM3: backbone dtype=%s, inst_interactive_predictor dtype=%s",
                next(detector.backbone.parameters()).dtype,
                next(detector.inst_interactive_predictor.parameters()).dtype
                if detector.inst_interactive_predictor is not None else "N/A",
            )

    # Compilation warmup
    if compile_model:
        log.info("Running compilation warmup (this may take a few minutes on first run)...")
        video_predictor.model.warm_up_compilation()
        log.info("Compilation warmup complete")

    detector = video_predictor.model.detector
    processor = Sam3Processor(
        model=detector,
        resolution=1008,
        device=str(load_device),
        confidence_threshold=0.2
    )

    unified_model = SAM3UnifiedModel(
        video_predictor=video_predictor,
        processor=processor,
        load_device=load_device,
        offload_device=offload_device,
        dtype=dtype,
    )

    log.info(f"Model ready ({unified_model.model_size() / 1024 / 1024:.1f} MB)")

    _cache["config_hash"] = config_hash
    _cache["model"] = unified_model
    return unified_model
