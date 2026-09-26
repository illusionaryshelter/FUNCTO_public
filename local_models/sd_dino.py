"""Local SD v1.3 and DINOv2 pair descriptors using the native SD-DINO fusion.

The two images must share their PCA basis. This adapter therefore replaces the
old independent HTTP requests with one pair request; it does not cache features
or alter the caller's interpolation, cosine similarity, masks, or argmax.
Heavy imports and model construction are confined to ``worker_main`` so the
worker can release its RAM and GPU allocations before FUNCTO starts matching.
"""

from pathlib import Path

import numpy as np
from PIL import Image


# BEGIN approved local descriptor profile: these values define the output,
# rather than performance switches which may silently change correspondence.
SD_SIZE = 960
DINO_SIZE = 840
GRID_SIZE = DINO_SIZE // 14
PCA_DIMS = (256, 256, 256)
SEED = 0
FEATURE_SHAPE = (1, sum(PCA_DIMS) + 768, GRID_SIZE, GRID_SIZE)
# END approved local descriptor profile.


def _validate_features(feature1: np.ndarray, feature2: np.ndarray) -> None:
    """Reject incomplete or non-finite worker results before CUDA matching."""
    for index, feature in enumerate((feature1, feature2), start=1):
        if feature.shape != FEATURE_SHAPE or feature.dtype != np.float32:
            raise ValueError(
                f"SD-DINO feature{index} must be FP32 {FEATURE_SHAPE}; "
                f"received {feature.dtype} {feature.shape}"
            )
        if not np.isfinite(feature).all():
            raise ValueError(f"SD-DINO feature{index} contains non-finite values")


def extract_pair(image1: Image.Image, image2: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Return two CPU ``[1,1536,60,60]`` descriptors from one local worker.

    FUNCTO supplies its existing resized crops. The runtime waits for the worker
    to exit before returning, so the unchanged dense matcher has the GPU alone.
    """
    from .runtime import run_worker

    result = run_worker(kind="sd_dino", images=[image1, image2], options={})
    feature1, feature2 = result["feature1"], result["feature2"]
    _validate_features(feature1, feature2)
    return feature1, feature2


def worker_main(input_paths: list[Path], output_path: Path) -> None:
    """Extract one pair offline, releasing SD before constructing DINOv2.

    SD960 and DINO840 both produce the native 60x60 fusion grid. At SD512 the
    first SD layer has only 128 tokens across the pair, insufficient for the
    agreed 256-dimensional PCA; neither dimensions nor the grid are reduced.
    The enclosing runtime disables networking and enforces resource limits.
    """
    if len(input_paths) != 2:
        raise ValueError("SD-DINO requires exactly two images for a shared PCA basis")

    import gc
    import os
    import sys

    # BEGIN offline initialization: missing local resources are errors. The
    # native loaders consume these existing caches; no weights are downloaded.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["XFORMERS_DISABLED"] = "1"
    recovery_root = Path(__file__).resolve().parents[2]
    sd_repo = recovery_root / "sd-dino"
    sd_weights = (
        Path.home() / ".torch/iopath_cache/CompVis/stable-diffusion-v-1-3-original"
        / "resolve/main/sd-v1-3.ckpt"
    )
    odise_weights = (
        Path.home() / ".torch/iopath_cache/NVlabs/ODISE/releases/download/v1.0.0"
        / "odise_label_coco_50e-b67d2efc.pth"
    )
    clip_weights = Path.home() / ".cache/clip/ViT-L-14-336px.pt"

    import torch

    hub_dir = Path(torch.hub.get_dir())
    dino_repo = hub_dir / "facebookresearch_dinov2_main"
    dino_weights = hub_dir / "checkpoints/dinov2_vitb14_pretrain.pth"
    required = (
        sd_repo / "extractor_sd.py", sd_repo / "extractor_dino.py",
        dino_repo / "hubconf.py", sd_weights, odise_weights, clip_weights, dino_weights,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing local SD-DINO resources:\n" + "\n".join(missing))
    os.environ["ODISE_MODEL_ZOO"] = str(sd_weights.parent)
    sys.path.insert(0, str(sd_repo))
    # END offline initialization.

    from extractor_dino import ViTExtractor
    from extractor_sd import load_model, process_features_and_mask
    from utils.utils_correspondence import co_pca, resize

    if not torch.cuda.is_available():
        raise RuntimeError("The local SD-DINO worker requires the configured CUDA GPU")
    torch.set_num_threads(2)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # Retain the tested FP32 arithmetic settings; SAM's BF16 context is isolated
    # in its own worker and must never surround this feature extraction.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False

    images = []
    for path in input_paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))

    with torch.no_grad():
        # BEGIN native raw SD stage. The already integrated SD-DINO optimization
        # selects its GPU dependencies; unused ODISE heads stay registered on CPU.
        model, aug = load_model(
            seed=SEED, diffusion_ver="v1-3", image_size=SD_SIZE,
            num_timesteps=0, block_indices=(2, 5, 8, 11),
            decoder_only=True, encoder_only=False, resblock_only=False,
        )
        model.eval()
        raw_pair = []
        for image in images:
            sd_input = resize(image, SD_SIZE, resize=True, to_pil=True, edge=False)
            raw = process_features_and_mask(
                model, aug, sd_input, input_text=None, mask=False, raw=True,
            )
            raw_pair.append({name: value.detach().cpu() for name, value in raw.items()})
            del raw
        del model, aug, sd_input
        gc.collect()
        torch.cuda.empty_cache()
        # END native raw SD stage.

        # BEGIN native fusion stage. Construct DINO before co-PCA, as in the
        # original extraction code: model initialization consumes RNG state.
        # Each branch is normalized separately before equal-weight concatenation.
        extractor = ViTExtractor(
            "dinov2_vitb14", stride=14, device="cuda",
            model_repo=dino_repo, checkpoint=dino_weights,
        )
        source = {name: value.to("cuda") for name, value in raw_pair[0].items()}
        target = {name: value.to("cuda") for name, value in raw_pair[1].items()}
        sd_pair = co_pca(source, target, dim=list(PCA_DIMS))
        del source, target, raw_pair

        fused_pair = []
        for image, sd_feature in zip(images, sd_pair):
            if tuple(sd_feature.shape) != (1, sum(PCA_DIMS), GRID_SIZE, GRID_SIZE):
                raise ValueError(f"Unexpected SD co-PCA feature shape: {tuple(sd_feature.shape)}")
            dino_input = resize(image, DINO_SIZE, resize=True, to_pil=True, edge=False)
            batch = extractor.preprocess_pil(dino_input).to("cuda")
            dino_desc = extractor.extract_descriptors(batch, layer=11, facet="token")
            if tuple(dino_desc.shape) != (1, 1, GRID_SIZE**2, 768):
                raise ValueError(f"Unexpected DINO descriptor shape: {tuple(dino_desc.shape)}")
            sd_desc = sd_feature.reshape(1, 1, -1, GRID_SIZE**2).permute(0, 1, 3, 2)
            sd_desc = sd_desc / sd_desc.norm(dim=-1, keepdim=True)
            dino_desc = dino_desc / dino_desc.norm(dim=-1, keepdim=True)
            fused = torch.cat((sd_desc, dino_desc), dim=-1)
            fused = fused.permute(0, 1, 3, 2).reshape(FEATURE_SHAPE)
            fused_pair.append(fused.contiguous().cpu().numpy())
            del batch, dino_desc, sd_desc, fused
        del extractor, sd_pair, sd_feature
        gc.collect()
        torch.cuda.empty_cache()
        # END native fusion stage.

    _validate_features(*fused_pair)
    np.savez(output_path, feature1=fused_pair[0], feature2=fused_pair[1])
