"""Model utilities for manipulation tasks.

This module provides helper functions for vision encoders:
- ResNet initialization
- BatchNorm to GroupNorm conversion
"""

from typing import Callable

import torch
import torch.nn as nn
import torchvision


def get_resnet(name: str, weights=None, **kwargs) -> nn.Module:
    """Get ResNet model with final layer removed.

    Args:
        name: ResNet variant (resnet18, resnet34, resnet50).
        weights: Pre-trained weights ("IMAGENET1K_V1" or None).
        **kwargs: Additional arguments for ResNet.

    Returns:
        ResNet model with identity final layer.
    """
    # Use standard ResNet implementation from torchvision
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)

    # remove the final fully connected layer
    # for resnet18, the output dim should be 512
    resnet.fc = torch.nn.Identity()
    return resnet


def replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """Replace all submodules matching a predicate.

    Args:
        root_module: Module to process.
        predicate: Function returning True if module should be replaced.
        func: Function returning replacement module.

    Returns:
        Modified root module.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    assert len(bn_list) == 0
    return root_module


def replace_bn_with_gn(
    root_module: nn.Module, features_per_group: int = 16
) -> nn.Module:
    """Replace all BatchNorm layers with GroupNorm.

    Args:
        root_module: Module to process.
        features_per_group: Features per group in GroupNorm.

    Returns:
        Modified module with GroupNorm layers.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features // features_per_group, num_channels=x.num_features
        ),
    )
    return root_module
