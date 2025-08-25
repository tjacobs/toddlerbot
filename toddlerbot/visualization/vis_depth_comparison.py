"""Depth Comparison Script

This script compares depth maps and point clouds between ground truth and prediction results.

Expected folder structure::

    results/
    ├── depth_blake_0529_fs_480x640_object/  # Ground truth folder
    │   ├── 10_depth.npy                     # Depth map files
    │   ├── 10_pcd.ply                       # Point cloud files
    │   └── ...
    └── depth_blake_0529_fs_96x128_object/   # Prediction folder
        ├── 10_depth.npy                     # Depth map files
        ├── 10_pcd.ply                       # Point cloud files
        └── ...

Filename format:
    - Depth maps: {frame_num}_depth.npy
    - Point clouds: {frame_num}_pcd.ply

where frame_num is an integer (e.g., 10, 20, 30, etc.)

Usage examples::

    # Compare frame 10 between two depth result folders
    python test_compare_depth.py --gt-folder results/depth_blake_0529_fs_480x640_object --pred-folder results/depth_blake_0529_fs_96x128_object --frame 10

    # Compare with depth threshold (only evaluate depths <= 2000cm)
    python test_compare_depth.py --gt-folder results/gt_folder --pred-folder results/pred_folder --frame 10 --zmax 2000
"""

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def load_point_cloud(path, voxel_size=None):
    """Load and optionally downsample a point cloud."""
    pcd = o3d.io.read_point_cloud(path)
    if voxel_size:
        pcd = pcd.voxel_down_sample(voxel_size)
    pcd.estimate_normals()
    return pcd


def run_icp(source_pcd, target_pcd, max_dist=0.05):
    result = o3d.pipelines.registration.registration_icp(
        source_pcd,
        target_pcd,
        max_dist,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    source_pcd.transform(result.transformation)
    print("ICP Transformation:\n", result.transformation)
    return source_pcd


def compute_pointcloud_metrics(source, target):
    d1 = np.asarray(source.compute_point_cloud_distance(target))
    d2 = np.asarray(target.compute_point_cloud_distance(source))

    rmse = np.sqrt(np.mean(np.square(d1)))
    mean = np.mean(d1)
    median = np.median(d1)
    max_error = np.max(d1)
    hausdorff = max(np.max(d1), np.max(d2))
    chamfer = np.mean(d1) + np.mean(d2)

    print("PointCloud Metrics:")
    print(f"  Mean     : {mean:.6f}")
    print(f"  Median   : {median:.6f}")
    print(f"  RMSE     : {rmse:.6f}")
    print(f"  Max      : {max_error:.6f}")
    print(f"  Hausdorff: {hausdorff:.6f}")
    print(f"  Chamfer  : {chamfer:.6f}")

    return {
        "mean": mean,
        "median": median,
        "rmse": rmse,
        "max": max_error,
        "hausdorff": hausdorff,
        "chamfer": chamfer,
    }


def compare_depth_maps_npy(depth_path_1, depth_path_2, mask_invalid=True, zmax=None):
    # Generate automatic output name if not provided
    dir1 = os.path.basename(os.path.dirname(depth_path_1))
    dir2 = os.path.basename(os.path.dirname(depth_path_2))
    output_path = (
        f"depth_error_{dir1}_vs_{dir2}_{zmax}.png"
        if zmax is not None
        else f"depth_error_{dir1}_vs_{dir2}.png"
    )
    range_error_path = (
        f"depth_range_error_{dir1}_vs_{dir2}_{zmax}.png"
        if zmax is not None
        else f"depth_range_error_{dir1}_vs_{dir2}.png"
    )

    d1 = np.load(depth_path_1).astype(np.float32) * 100
    d2 = np.load(depth_path_2).astype(np.float32) * 100

    # If shapes are different, resize to the larger resolution
    if d1.shape != d2.shape:
        target_height = max(d1.shape[0], d2.shape[0])
        target_width = max(d1.shape[1], d2.shape[1])

        if d1.shape != (target_height, target_width):
            d1 = cv2.resize(
                d1, (target_width, target_height), interpolation=cv2.INTER_LINEAR
            )
        if d2.shape != (target_height, target_width):
            d2 = cv2.resize(
                d2, (target_width, target_height), interpolation=cv2.INTER_LINEAR
            )

        print(f"Resized depth maps to {target_height}x{target_width}")

    # Create mask for valid pixels (where both depth maps have valid values)
    valid = (d1 > 0) & (d2 > 0) if mask_invalid else np.ones_like(d1, dtype=bool)

    # Add zmax threshold if specified
    if zmax is not None:
        valid = valid & (d1 <= zmax) & (d2 <= zmax)

    # Calculate error only for valid pixels
    error_map = np.zeros_like(d1)
    error_map[valid] = np.abs(d1[valid] - d2[valid])

    # Calculate metrics only on valid pixels
    diff = error_map[valid]
    rmse = np.sqrt(np.mean((d1[valid] - d2[valid]) ** 2))
    mean = np.mean(diff)
    median = np.median(diff)
    max_error = np.max(diff)

    print("Depth Map Metrics:")
    print(f"  Valid Pixels: {np.sum(valid)} / {valid.size}")
    print(f"  Mean        : {mean:.6f}")
    print(f"  Median      : {median:.6f}")
    print(f"  RMSE        : {rmse:.6f}")
    print(f"  Max         : {max_error:.6f}")

    # Visualize error map
    plt.figure(figsize=(10, 8))
    plt.imshow(error_map, cmap="inferno")
    plt.title(
        f"Absolute Depth Error zmax={zmax}\n{dir1} (GT)\nvs.\n{dir2} (Pred)\nRMSE: {rmse:.6f} cm",
        fontsize=10,
    )
    plt.colorbar(label="Error (cm)")
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved depth error map to {output_path}")

    # Create range-based error analysis plot
    plt.figure(figsize=(10, 6))
    plt.scatter(d1[valid], error_map[valid], alpha=0.1, s=1)
    plt.xlabel("Depth Range (cm)")
    plt.ylabel("Absolute Error (cm)")
    plt.title(f"Depth Range vs Error\n{dir1} (GT) vs {dir2} (Pred)")
    plt.grid(True)
    plt.savefig(range_error_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved range-based error analysis to {range_error_path}")

    return {"mean": mean, "median": median, "rmse": rmse, "max": max_error}


def generate_bev_image(points, color, bounds, resolution, overhang_threshold=None):
    x_min, x_max, z_min, z_max = bounds
    width = int((x_max - x_min) / resolution)
    height = int((z_max - z_min) / resolution)
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Filter points based on overhang threshold
    heights = -points[:, 1]
    if overhang_threshold is not None:
        mask = (
            heights <= overhang_threshold
        )  # Keep points below threshold (less negative)
        points = points[mask]
        if len(points) == 0:
            return img

    # Normalize y coordinates to [0, 1] range (negative y is up in MeshLab)
    y_min_height, y_max_height = np.percentile(points[:, 1], [1, 99])
    y_normalized = np.clip(
        (y_max_height - points[:, 1]) / (y_max_height - y_min_height), 0, 1
    )

    # Project onto XZ plane (top view)
    xs = np.clip(((points[:, 0] - x_min) / resolution).astype(int), 0, width - 1)
    zs = np.clip(((points[:, 2] - z_min) / resolution).astype(int), 0, height - 1)

    # Create color array with y-based intensity
    colors = np.array(color) * y_normalized[:, np.newaxis]
    colors = colors.astype(np.uint8)

    # Project points with y-based coloring
    for i in range(len(points)):
        img[height - zs[i] - 1, xs[i]] = colors[i]

    return img


def create_bev_overlap(
    gt_path, pred_path, voxel=0.01, res=0.01, overhang_threshold=None, opacity=0.5
):
    # Generate automatic output name if not provided
    dir1 = os.path.basename(os.path.dirname(gt_path))
    dir2 = os.path.basename(os.path.dirname(pred_path))
    out_path = f"bev_overlap_{dir1}_vs_{dir2}.png"

    gt = load_point_cloud(gt_path, voxel)
    pred = load_point_cloud(pred_path, voxel)

    gt_pts = np.asarray(gt.points)
    pred_pts = np.asarray(pred.points)
    all_pts = np.vstack((gt_pts, pred_pts))

    # Calculate bounds using X and Z coordinates
    x_min, x_max = np.percentile(all_pts[:, 0], [1, 99])
    z_min, z_max = np.percentile(all_pts[:, 2], [1, 99])
    bounds = (x_min, x_max, z_min, z_max)

    img_gt = generate_bev_image(gt_pts, [0, 255, 0], bounds, res, overhang_threshold)
    img_pred = generate_bev_image(
        pred_pts, [255, 0, 0], bounds, res, overhang_threshold
    )

    # Create binary masks for each image
    mask_gt = np.any(img_gt > 0, axis=2)
    mask_pred = np.any(img_pred > 0, axis=2)

    # Create blended image
    bev = np.zeros_like(img_gt, dtype=np.float32)

    # Add ground truth with specified opacity
    bev[mask_gt] = img_gt[mask_gt]

    # Add prediction with specified opacity
    bev[mask_pred] = bev[mask_pred] + img_pred[mask_pred]

    # Convert back to uint8
    bev = bev.astype(np.uint8)

    plt.imsave(out_path, bev)
    print(f"Saved BEV overlap image to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare depth maps and point clouds between ground truth and prediction results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--gt-folder",
        type=str,
        required=True,
        help="Path to ground truth results folder containing depth maps and point clouds",
    )

    parser.add_argument(
        "--pred-folder",
        type=str,
        required=True,
        help="Path to prediction results folder containing depth maps and point clouds",
    )

    parser.add_argument(
        "--frame", type=int, default=10, help="Frame number to compare (default: 10)"
    )

    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.01,
        help="Voxel size for point cloud downsampling (default: 0.01)",
    )

    parser.add_argument(
        "--bev-resolution",
        type=float,
        default=0.01,
        help="Resolution for bird's eye view generation (default: 0.01)",
    )

    parser.add_argument(
        "--overhang-threshold",
        type=float,
        default=1.5,
        help="Overhang threshold for BEV visualization (default: 1.5)",
    )

    parser.add_argument(
        "--zmax",
        type=float,
        default=None,
        help="Maximum depth threshold in cm for evaluation (default: None, no threshold)",
    )

    parser.add_argument(
        "--no-mask-invalid",
        action="store_true",
        help="Don't mask invalid pixels (depth <= 0) during comparison",
    )

    args = parser.parse_args()

    # Validate input folders
    if not os.path.exists(args.gt_folder):
        raise FileNotFoundError(f"Ground truth folder does not exist: {args.gt_folder}")
    if not os.path.exists(args.pred_folder):
        raise FileNotFoundError(f"Prediction folder does not exist: {args.pred_folder}")

    print(f"Comparing frame {args.frame}")
    print(f"Ground truth folder: {args.gt_folder}")
    print(f"Prediction folder: {args.pred_folder}")
    print(f"Voxel size: {args.voxel_size}")
    print(f"BEV resolution: {args.bev_resolution}")
    print(f"Overhang threshold: {args.overhang_threshold}")
    if args.zmax:
        print(f"Depth threshold: {args.zmax} cm")
    print()

    # Point cloud alignment and evaluation
    gt_ply = os.path.join(args.gt_folder, f"{args.frame}_pcd.ply")
    pred_ply = os.path.join(args.pred_folder, f"{args.frame}_pcd.ply")
    assert os.path.exists(gt_ply), f"GT point cloud file does not exist: {gt_ply}"
    assert os.path.exists(pred_ply), f"Pred point cloud file does not exist: {pred_ply}"

    print("Loading and aligning point clouds...")
    gt_pc = load_point_cloud(gt_ply, args.voxel_size)
    pred_pc = load_point_cloud(pred_ply, args.voxel_size)
    pred_aligned = run_icp(pred_pc, gt_pc)
    compute_pointcloud_metrics(pred_aligned, gt_pc)
    create_bev_overlap(
        gt_ply,
        pred_ply,
        voxel=args.voxel_size,
        res=args.bev_resolution,
        overhang_threshold=args.overhang_threshold,
        opacity=0.5,
    )

    # Depth map evaluation
    gt_depth = os.path.join(args.gt_folder, f"{args.frame}_depth.npy")
    pred_depth = os.path.join(args.pred_folder, f"{args.frame}_depth.npy")
    assert os.path.exists(gt_depth), f"GT depth map file does not exist: {gt_depth}"
    assert os.path.exists(pred_depth), (
        f"Pred depth map file does not exist: {pred_depth}"
    )

    print("\nComparing depth maps...")
    compare_depth_maps_npy(
        gt_depth, pred_depth, mask_invalid=not args.no_mask_invalid, zmax=args.zmax
    )
