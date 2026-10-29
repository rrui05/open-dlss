"""Frozen perceptual teachers with gradients with respect to their image inputs.

The random VGG option is only for offline smoke tests. It does not provide a
meaningful perceptual prior. Requested pretrained teachers never silently fall
back to random weights or to a different model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class _FrozenTeacher(nn.Module):
    """Keep a teacher in evaluation mode without disabling input autograd."""

    def _freeze(self) -> None:
        self.requires_grad_(False)
        self.train(False)

    def train(self, mode: bool = True) -> "_FrozenTeacher":
        super().train(False)
        return self


class _ImageNetTeacher(_FrozenTeacher):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, image: Tensor) -> Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("a perceptual teacher expects BCHW RGB images")
        if not image.is_floating_point():
            raise TypeError("teacher images must be floating point in [0, 1]")
        return (image - self.mean.to(dtype=image.dtype)) / self.std.to(dtype=image.dtype)


def _vgg16_features() -> nn.Sequential:
    """Build torchvision-compatible VGG16 features without importing torchvision."""
    configuration: list[int | str] = [
        64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
        512, 512, 512, "M", 512, 512, 512, "M",
    ]
    layers: list[nn.Module] = []
    channels = 3
    for item in configuration:
        if item == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            out_channels = int(item)
            convolution = nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)
            nn.init.kaiming_normal_(convolution.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(convolution.bias)
            layers.extend([convolution, nn.ReLU(inplace=False)])
            channels = out_channels
    return nn.Sequential(*layers)


def _feature_state(checkpoint: Any) -> dict[str, Tensor]:
    """Accept a feature state dict or the feature portion of a VGG checkpoint."""
    if not isinstance(checkpoint, dict):
        raise ValueError("a local VGG checkpoint must contain a state dictionary")
    for key in ("state_dict", "model", "features"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            checkpoint = checkpoint[key]
            break
    state: dict[str, Tensor] = {}
    for key, value in checkpoint.items():
        if not isinstance(value, Tensor):
            continue
        name = str(key).removeprefix("module.")
        if name.startswith("features."):
            name = name[len("features."):]
        if name.split(".", 1)[0].isdigit():
            state[name] = value
    if not state:
        raise ValueError("the checkpoint contains no VGG16 feature weights")
    return state


class FrozenVGG16(_ImageNetTeacher):
    """Return relu1_2, relu2_2, relu3_3 and relu4_3 feature maps.

    ``weights='imagenet'`` lazily imports torchvision and loads its official
    ImageNet checkpoint. A local path accepts either torchvision's full state
    dictionary or ``vgg.features.state_dict()``. ``weights='none'`` creates random
    features and is suitable only for an offline smoke test, never quality work.
    The returned channel counts are 64, 128, 256 and 512.
    """

    def __init__(self, weights: str | Path = "imagenet") -> None:
        super().__init__()
        self.features = _vgg16_features()
        self.weights_source = str(weights)
        self.quality_teacher = self.weights_source != "none"
        if self.weights_source == "imagenet":
            try:
                from torchvision.models import VGG16_Weights, vgg16
            except (ImportError, RuntimeError) as error:
                raise RuntimeError(
                    "ImageNet VGG16 requires a working torchvision installation; "
                    "use a local VGG16 checkpoint, or weights='none' only for an offline smoke test"
                ) from error
            pretrained = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            self.features.load_state_dict(pretrained.features.state_dict(), strict=True)
            del pretrained
        elif self.weights_source != "none":
            checkpoint = torch.load(Path(weights), map_location="cpu", weights_only=True)
            self.features.load_state_dict(_feature_state(checkpoint), strict=True)
        self._freeze()

    def forward(self, image: Tensor) -> list[Tensor]:
        if min(image.shape[-2:]) < 8:
            raise ValueError("VGG feature extraction needs image dimensions of at least 8")
        features: list[Tensor] = []
        value = self._normalize(image)
        for index, layer in enumerate(self.features):
            value = layer(value)
            if index in (3, 8, 15, 22):
                features.append(value)
            if index == 22:
                break
        return features


class DinoStructure(_ImageNetTeacher):
    """Frozen DINOv2 patch features; the output is BCHW and excludes CLS tokens.

    Images are padded on the right and bottom to patch-size multiples. No image
    processor resizes the scene. ``revision`` can pin an immutable model revision.
    Caller-side ``torch.no_grad()`` is appropriate for the source branch only.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        local_files_only: bool = False,
        revision: str | None = None,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as error:
            raise RuntimeError("DINOv2 requires transformers; no substitute teacher will be used") from error
        options: dict[str, Any] = {"local_files_only": local_files_only, "trust_remote_code": False}
        if revision is not None:
            options["revision"] = revision
        self.model = AutoModel.from_pretrained(model_name, **options)
        patch = getattr(self.model.config, "patch_size", 14)
        if isinstance(patch, (tuple, list)):
            if len(patch) != 2 or patch[0] != patch[1]:
                raise ValueError("DINO structure extraction requires square patches")
            patch = patch[0]
        self.patch_size = int(patch)
        if self.patch_size < 1:
            raise ValueError("invalid DINO patch size")
        self._freeze()

    def forward(self, image: Tensor) -> Tensor:
        value = self._normalize(image)
        height, width = value.shape[-2:]
        pad_height = (-height) % self.patch_size
        pad_width = (-width) % self.patch_size
        if pad_height or pad_width:
            value = F.pad(value, (0, pad_width, 0, pad_height), mode="replicate")
        rows, columns = value.shape[-2] // self.patch_size, value.shape[-1] // self.patch_size
        tokens = self.model(pixel_values=value, return_dict=True).last_hidden_state
        count = rows * columns
        if tokens.ndim != 3 or tokens.shape[1] < count:
            raise ValueError("DINO returned an unexpected patch-token shape")
        # DINO puts its CLS token (and any register tokens) before spatial tokens.
        patches = tokens[:, -count:, :]
        return patches.transpose(1, 2).reshape(image.shape[0], tokens.shape[-1], rows, columns)


class SemanticTeacher(_ImageNetTeacher):
    """Frozen Cityscapes SegFormer returning 19-class, roughly quarter-size logits."""

    def __init__(
        self,
        model_name: str = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
        local_files_only: bool = False,
        revision: str | None = None,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModelForSemanticSegmentation
        except ImportError as error:
            raise RuntimeError("semantic supervision requires transformers; no substitute teacher will be used") from error
        options: dict[str, Any] = {"local_files_only": local_files_only, "trust_remote_code": False}
        if revision is not None:
            options["revision"] = revision
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name, **options)
        if int(self.model.config.num_labels) != 19:
            raise ValueError("the semantic teacher must provide the 19 Cityscapes classes")
        self._freeze()

    def forward(self, image: Tensor) -> Tensor:
        logits = self.model(pixel_values=self._normalize(image), return_dict=True).logits
        if logits.ndim != 4 or logits.shape[1] != 19:
            raise ValueError("the semantic teacher returned invalid class logits")
        return logits
