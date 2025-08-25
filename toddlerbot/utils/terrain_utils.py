"""
Utility functions for terrain generation and heightmap processing.

Includes:
- Smooth Perlin noise generation for bumpy surfaces.
- Edge fade mask for blending heightfields smoothly.
- Frustum-shaped heightmap with adjustable flat top.
- Center flattening mask for spawn-safe regions.

Used by terrain_types.py.
"""

import numpy as np
from scipy.signal import convolve2d


def interpolant(t):
    """
    Smoothstep interpolant used by Perlin noise.
    Helps create smooth transitions between gradient cells.

    Args:
        t (ndarray): fractional position grid

    Returns:
        ndarray: smoothed position weights
    """
    return t * t * t * (t * (t * 6 - 15) + 10)


def perlin(shape, res, tileable=(False, False)):
    """
    Generate a 2D Perlin noise grid.

    Args:
        shape (tuple): Output array shape (H, W).
        res (tuple): Number of periods along each axis (y, x).
        tileable (tuple): Whether the noise tiles along each axis.

    Returns:
        ndarray: 2D Perlin noise map normalized to [-sqrt(2), sqrt(2)]
    """
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    # Create coordinate grid within each gradient cell
    grid = np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0) % 1

    # Generate random gradient directions
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))

    if tileable[0]:
        gradients[-1, :] = gradients[0, :]
    if tileable[1]:
        gradients[:, -1] = gradients[:, 0]

    # Repeat gradient vectors to fill the output shape
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)

    # Compute dot products between gradient vectors and relative position vectors
    g00 = gradients[: -d[0], : -d[1]]
    g10 = gradients[d[0] :, : -d[1]]
    g01 = gradients[: -d[0], d[1] :]
    g11 = gradients[d[0] :, d[1] :]

    n00 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1])) * g00, axis=2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, axis=2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, axis=2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, axis=2)

    # Interpolate between noise values
    t = interpolant(grid)
    n0 = (1 - t[:, :, 0]) * n00 + t[:, :, 0] * n10
    n1 = (1 - t[:, :, 0]) * n01 + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def edge_slope(size, border_width=5, blur_iterations=20):
    """
    Create a fade-out mask along the edges of a heightmap.

    Args:
        size (int): Size of the heightmap (square).
        border_width (int): Width of the zeroed edge border.
        blur_iterations (int): Number of blurring steps to smooth the mask.

    Returns:
        ndarray: Edge mask that gradually fades to zero.
    """
    img = np.ones((size, size), dtype=np.float32)
    img[:border_width, :] = img[-border_width:, :] = 0
    img[:, :border_width] = img[:, -border_width:] = 0

    # Apply iterative box blur to soften edges
    kernel = np.ones((3, 3)) / 9.0
    for _ in range(blur_iterations):
        img = convolve2d(img, kernel, mode="same", boundary="symm")

    return img


def frustum_with_flat_top(
    size, peak=0.05, flat_ratio=0.3, falloff="quadratic", edge_flat_width=1
):
    """
    Generate a frustum-shaped heightmap with a flat top.

    Args:
        size (int): Size of the output heightmap (square).
        peak (float): Height of the peak.
        flat_ratio (float): Radius ratio of the flat top (0 to 1).
        falloff (str): Type of falloff function ("quadratic", "linear", "cosine").
        edge_flat_width (int): Force a flat border of this width around edges.

    Returns:
        ndarray: Frustum-shaped heightmap.
    """
    cx, cy = size // 2, size // 2
    xv, yv = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    dist = np.sqrt((xv - cx) ** 2 + (yv - cy) ** 2)
    max_dist = np.sqrt(cx**2 + cy**2)

    flat_radius = flat_ratio * size / 2
    mask_flat = dist <= flat_radius
    slope_region = np.clip((dist - flat_radius) / (max_dist - flat_radius), 0, 1)

    # Choose falloff type
    if falloff == "quadratic":
        slope = (1 - slope_region) ** 2
    elif falloff == "linear":
        slope = 1 - slope_region
    elif falloff == "cosine":
        slope = 0.5 * (1 + np.cos(np.pi * slope_region))
    else:
        raise ValueError("Unknown falloff")

    heightmap = peak * slope
    heightmap[mask_flat] = peak

    # Force outer edge to be completely flat at height 0
    if edge_flat_width > 0:
        edge_mask = np.zeros_like(heightmap, dtype=bool)
        edge_mask[:edge_flat_width, :] = True
        edge_mask[-edge_flat_width:, :] = True
        edge_mask[:, :edge_flat_width] = True
        edge_mask[:, -edge_flat_width:] = True
        heightmap[edge_mask] = 0.0

    return heightmap


def center_flat_mask(size, flat_radius_frac=0.05):
    """
    Generate a radial mask that flattens the center of a heightmap.

    Args:
        size (int): Output resolution (square).
        flat_radius_frac (float): Radius of the flat region as a fraction of size.

    Returns:
        ndarray: A radial mask with a smooth transition from center to edges.
    """
    y, x = np.ogrid[-1 : 1 : complex(0, size), -1 : 1 : complex(0, size)]
    r = np.sqrt(x**2 + y**2)
    mask = np.clip((r - flat_radius_frac) / (1 - flat_radius_frac), 0, 1)
    return mask
