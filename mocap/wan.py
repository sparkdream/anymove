"""Wan 2.2 (Apache-2.0, open weights) on the local GPU: a two-camera clip from the reference frame and a prompt.

python -m mocap.wan REF.png PROMPT.txt OUT.mp4 [seconds] [seed] [steps] [model]
Models (diffusers >= 0.35):
  wan22       TI2V-5B, Wan-AI/Wan2.2-TI2V-5B-Diffusers (34 GB on disk), 1280 x 704 at 24 fps, about 24 GB on the GPU
  wan22-14b   I2V-A14B, its two 14B experts as Q8_0 GGUF files (QuantStack/Wan2.2-I2V-A14B-GGUF, 2 x 15.4 GB) with the
              text encoder and VAE of Wan-AI/Wan2.2-I2V-A14B-Diffusers; 848 x 480 at 16 fps. Only one expert is on the
              GPU at a time: the high-noise one for the first steps, then the low-noise one. Needs the gguf package.
mocap.detect's calibration maps either size back onto the 1280 x 720 layout. One clip per process: the pipeline
leaves about 24 GB on the GPU that only the process exit frees, and a second clip on top spills out of GPU memory.
"""
import gc, pathlib, sys, time

import torch

from mocap.model import device

MODELS = {
    "wan22": {"repo": "Wan-AI/Wan2.2-TI2V-5B-Diffusers", "size": (1280, 704), "fps": 24, "steps": 50, "guidance": 5.0},
    "wan22-14b": {"repo": "Wan-AI/Wan2.2-I2V-A14B-Diffusers", "size": (848, 480), "fps": 16, "steps": 40, "guidance": 3.5,
                  "experts": ("QuantStack/Wan2.2-I2V-A14B-GGUF", "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q8_0.gguf",
                              "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q8_0.gguf")},
}
# Wan's own negative prompt (English version from diffusers)
NEGATIVE = ("Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
            "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly "
            "drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy "
            "background, three legs, many people in the background, walking backwards")


def gguf_state_dict(file, dev):
    """A GGUF file's tensors, moved to the GPU one at a time from the memory-mapped file: diffusers' own loader copies
    the whole file (15 GB per expert) into host memory first."""
    import warnings
    import gguf
    import numpy as np
    from diffusers.quantizers.gguf.utils import GGUFParameter
    plain = (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the mapped file is read-only; the tensors are only read, then copied over
        for t in gguf.GGUFReader(file).tensors:
            w = torch.from_numpy(np.asarray(t.data)).to(dev)
            out[t.name] = w if t.tensor_type in plain else GGUFParameter(w, quant_type=t.tensor_type)
    return out


def expert(spec, path, which, dev):
    """One of the 14B experts from its GGUF file, straight onto the GPU."""
    from diffusers import GGUFQuantizationConfig, WanTransformer3DModel
    from huggingface_hub import hf_hub_download
    repo, high, low = spec["experts"]
    weights = gguf_state_dict(hf_hub_download(repo, high if which == "transformer" else low), dev)
    return WanTransformer3DModel.from_single_file(
        weights, config=path, subfolder=which, quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16, device=dev)  # single-file loading ignores device_map and fills the CPU instead


def free(module):
    """Drop a module's weights now: the pipeline's loop still holds the module itself until its next step."""
    for m in module.modules():
        for k in list(m._parameters):
            m._parameters[k] = None
        for k in list(m._buffers):
            m._buffers[k] = None
    gc.collect()
    torch.accelerator.empty_cache()


def load(spec, dev):
    """Each part straight onto the GPU, so the weights never sit in host memory at once."""
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanTransformer3DModel
    from huggingface_hub import snapshot_download
    from transformers import UMT5EncoderModel
    # a local folder: diffusers asks the Hub about sharded weights even when offline; the 14B's float32 experts are
    # never downloaded, only their configs
    path = snapshot_download(spec["repo"], ignore_patterns=["transformer*/*.safetensors"] if "experts" in spec else None)
    text_encoder = UMT5EncoderModel.from_pretrained(path, subfolder="text_encoder", dtype=torch.bfloat16, device_map=dev)
    if "experts" in spec:
        transformer = expert(spec, path, "transformer", dev)
    else:
        transformer = WanTransformer3DModel.from_pretrained(path, subfolder="transformer", torch_dtype=torch.bfloat16, device_map=dev)
    vae = AutoencoderKLWan.from_pretrained(path, subfolder="vae", torch_dtype=torch.float32, device_map=dev)
    # Decoding whole 1280 x 704 frames peaks near 20 GB on top of the transformer; past the GPU's memory, WSL spills
    # into shared system memory and the decode crawls for hours. Tiles peak at 5 GB.
    vae.enable_tiling()
    extra = {"transformer_2": transformer} if "experts" in spec else {}  # swapped for the low-noise expert later
    return WanImageToVideoPipeline.from_pretrained(path, text_encoder=text_encoder, transformer=transformer, vae=vae,
                                                   torch_dtype=torch.bfloat16, **extra), path


def generate(ref, prompt, out, seconds=5, seed=None, steps=None, model="wan22", negative=NEGATIVE):
    from diffusers.utils import export_to_video, load_image
    spec = MODELS[model]
    dev = device()
    t0 = time.time()
    pipe, path = load(spec, dev)
    print(f"{model} loaded in {time.time() - t0:.0f} s", flush=True)
    with torch.no_grad():
        embeds, negative_embeds = pipe.encode_prompt(prompt, negative, device=dev)
    pipe.text_encoder = None  # 11 GB the denoising doesn't need
    torch.accelerator.empty_cache()

    def swap_experts(pipe, i, t, kwargs):
        """At the boundary, replace the high-noise expert with the low-noise one."""
        ts = pipe.scheduler.timesteps
        boundary = pipe.config.boundary_ratio * pipe.scheduler.config.num_train_timesteps
        if i + 1 < len(ts) and ts[i + 1] < boundary and pipe.transformer_2 is pipe.transformer:
            free(pipe.transformer)
            pipe.transformer_2 = expert(spec, path, "transformer_2", dev)
            print(f"switched to the low-noise expert after step {i + 1}", flush=True)
        return {}

    width, height = spec["size"]
    generator = torch.Generator(dev).manual_seed(seed) if seed is not None else None
    frames = pipe(image=load_image(str(ref)), prompt_embeds=embeds, negative_prompt_embeds=negative_embeds,
                  height=height, width=width, num_frames=seconds * spec["fps"] + 1,
                  num_inference_steps=steps or spec["steps"], guidance_scale=spec["guidance"], generator=generator,
                  callback_on_step_end=swap_experts if "experts" in spec else None).frames[0]
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(out), fps=spec["fps"])
    print(f"{model} ok {time.time() - t0:.0f} s, peak GPU memory {torch.accelerator.max_memory_allocated() / 2**30:.1f} GB", flush=True)


if __name__ == "__main__":
    ref, prompt_file, out = sys.argv[1:4]
    generate(ref, pathlib.Path(prompt_file).read_text().strip(), out,
             seconds=int(sys.argv[4]) if len(sys.argv) > 4 else 5,
             seed=int(sys.argv[5]) if len(sys.argv) > 5 else None,
             steps=int(sys.argv[6]) if len(sys.argv) > 6 else None,
             model=sys.argv[7] if len(sys.argv) > 7 else "wan22")
