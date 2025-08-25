"""Test stereo depth estimation using Foundation Stereo model.

This module provides functionality to test stereo depth estimation using the Foundation Stereo
neural network model. It processes stereo camera images to produce depth maps and point clouds.
"""

import argparse
import os
import statistics
import time
import traceback

import cv2
import matplotlib
import numpy as np
import open3d as o3d

from toddlerbot.depth.depth_estimator_foundation_stereo import (
    DepthEstimatorFoundationStereo,
)
from toddlerbot.depth.depth_utils import vis_disparity
from toddlerbot.utils.misc_utils import dump_profiling_data, profile

# Constants
DEBUG_IMAGES_FOLDER = os.path.join("results", "depth_blake_debug")
DEFAULT_OUTDIR = os.path.join(
    "results", f"test_foundation_stereo_{time.strftime('%Y%m%d_%H%M%S')}"
)
CALIB_PARAMS_PATH = os.path.join("toddlerbot", "depth", "params", "calibration.pkl")
DEFAULT_CALIB_HEIGHT = 480
DEFAULT_CALIB_WIDTH = 640
REC_PARAMS_PATH = os.path.join("toddlerbot", "depth", "params", "rectification.npz")
ENGINE_PATH = os.path.join(
    "toddlerbot",
    "depth",
    "models",
    "foundation_stereo_vits_96x128_16.engine",
)

# Number of warm-up iterations to ignore for FPS calculation
NUM_WARM_UP = 10


def parse_args():
    """Parse command line arguments for stereo depth estimation test."""
    parser = argparse.ArgumentParser(
        description="Depth Foundation Stereo Metric Depth Estimation"
    )
    parser.add_argument("--vis", dest="vis", action="store_true", help="visualize")
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--skip-rectify",
        dest="skip_rectify",
        action="store_true",
        help="skip rectification",
    )
    parser.add_argument(
        "--save-output",
        dest="save_output",
        action="store_true",
        help="save the input image, output depth numpy array, and point cloud",
    )
    parser.add_argument(
        "--debug",
        nargs="?",
        const=DEBUG_IMAGES_FOLDER,
        default=None,
        help="Enable debug mode. If no path is provided, uses default debug folder.",
    )
    parser.add_argument(
        "--calib_params",
        type=str,
        default=CALIB_PARAMS_PATH,
        help="path to the calibration parameters file",
    )
    parser.add_argument(
        "--rec_params",
        type=str,
        default=REC_PARAMS_PATH,
        help="path to the rectification parameters file",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default=ENGINE_PATH,
        help="path to the engine file",
    )

    parser.add_argument(
        "--calib_height",
        default=DEFAULT_CALIB_HEIGHT,
        type=int,
        help="This is the height of the images used for calibration.",
    )
    parser.add_argument(
        "--calib_width",
        default=DEFAULT_CALIB_WIDTH,
        type=int,
        help="This is the width of the images used for calibration.",
    )
    parser.add_argument(
        "--zmax",
        default=20,
        type=float,
        help="max depth (meters) to clip in point cloud",
    )
    parser.add_argument(
        "--down_sample",
        type=int,
        default=1,
        help="down-sample the output point cloud by this factor",
    )
    parser.add_argument(
        "--remove_invisible",
        default=1,
        type=int,
        help="remove non-overlapping observations between left and right images from point cloud, so the remaining points are more reliable",
    )
    parser.add_argument(
        "--denoise_cloud",
        type=int,
        default=0,
        help="whether to denoise the point cloud",
    )
    parser.add_argument(
        "--denoise_nb_points",
        type=int,
        default=30,
        help="number of points to consider for radius outlier removal",
    )
    parser.add_argument(
        "--denoise_radius",
        type=float,
        default=0.03,
        help="radius to use for outlier removal",
    )

    return parser.parse_args()


@profile()
def main(args):
    """Run stereo depth estimation test with specified arguments."""
    depth_estimator = DepthEstimatorFoundationStereo(
        calib_params_path=args.calib_params,
        rec_params_path=args.rec_params,
        engine_path=args.engine,
        calib_width=args.calib_width,
        calib_height=args.calib_height,
        skip_rectify=args.skip_rectify,
        debug=args.debug is not None,
    )
    os.makedirs(args.outdir, exist_ok=True)
    cmap = matplotlib.colormaps.get_cmap("Spectral")

    i = 0
    keep_running = True
    loop_times = []
    try:
        while keep_running:
            loop_start = time.time()
            # Press 'q' or ESC in OpenCV window to break
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Original image
            debug_images = None
            if args.debug is not None:
                # Load images from debug folder
                img_left_path = os.path.join(args.debug, f"{i}_original_left.jpg")
                img_right_path = os.path.join(args.debug, f"{i}_original_right.jpg")

                if not os.path.isfile(img_left_path) or not os.path.isfile(
                    img_right_path
                ):
                    print(f"Image not found: {img_left_path} or {img_right_path}")
                    break
                image_left = cv2.imread(img_left_path)
                image_right = cv2.imread(img_right_path)
                debug_images = (image_left, image_right)

            depth_result = depth_estimator.get_depth(
                remove_invisible=args.remove_invisible,
                debug_images=debug_images,
                return_all=True,  # return all images for visualization
            )

            if depth_result is None:
                print(f"No depth at loop {i}")
                continue
            assert depth_result.original_left is not None
            assert depth_result.original_right is not None
            assert depth_result.rectified_left is not None
            assert depth_result.rectified_right is not None
            assert depth_result.depth is not None

            # Show original image and depth map side by side
            depth_vis = vis_disparity(
                depth_result.depth,
                min_val=0,
                max_val=args.zmax,
                invalid_upper_thres=args.zmax,
                invalid_bottom_thres=0.0,
                cmap=cmap,
            )
            disp_vis = vis_disparity(depth_result.disparity)
            combined_frame = np.hstack(
                [
                    depth_result.rectified_left,
                    depth_result.rectified_right,
                    disp_vis,
                    depth_vis,
                ],
                dtype=np.uint8,
            )
            if args.vis:
                # cv2.imshow("Mono Camera Stream", combined_frame)

                # Create a black canvas to add text below the images
                center_h = depth_result.depth.shape[0] // 2
                center_w = depth_result.depth.shape[1] // 2
                center_depth = depth_result.depth[center_h, center_w]
                h, w, _ = combined_frame.shape
                text_area_height = 20
                canvas = np.zeros((h + text_area_height, w, 3), dtype=np.uint8)
                canvas[:h, :] = combined_frame

                # Prepare and add the text to the canvas
                text = f"Depth at center: {center_depth:.4f} m"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                font_color = (255, 255, 255)  # White
                thickness = 1
                text_position = (10, h + text_area_height - 7)

                cv2.putText(
                    canvas, text, text_position, font, font_scale, font_color, thickness
                )
                cv2.imshow("Mono Camera Stream", canvas)

            # Create the point cloud and save it to the output directory
            if args.save_output:
                og_left_path = os.path.join(args.outdir, f"{i}_original_left.jpg")
                cv2.imwrite(og_left_path, depth_result.original_left)

                og_right_path = os.path.join(args.outdir, f"{i}_original_right.jpg")
                cv2.imwrite(og_right_path, depth_result.original_right)

                rectified_left_path = os.path.join(
                    args.outdir, f"{i}_rectified_left.jpg"
                )
                cv2.imwrite(rectified_left_path, depth_result.rectified_left)

                rectified_right_path = os.path.join(
                    args.outdir, f"{i}_rectified_right.jpg"
                )
                cv2.imwrite(rectified_right_path, depth_result.rectified_right)

                npy_path = os.path.join(args.outdir, f"{i}_depth.npy")
                np.save(npy_path, depth_result.depth)

                combined_path = os.path.join(args.outdir, f"{i}_combined.jpg")
                cv2.imwrite(combined_path, combined_frame)

                # Generate point cloud
                pcd = depth_estimator.get_pcl(
                    depth_result.depth,
                    depth_result.rectified_left,
                    is_BGR=True,
                    zmax=args.zmax,
                    denoise_cloud=args.denoise_cloud,
                    denoise_nb_points=args.denoise_nb_points,
                    denoise_radius=args.denoise_radius,
                )
                pcd = pcd.uniform_down_sample(args.down_sample)
                pcd_filename = os.path.join(args.outdir, f"{i}_pcd.ply")
                o3d.io.write_point_cloud(pcd_filename, pcd)

            # Calculate and print the FPS
            current_time = time.time()
            loop_time = (current_time - loop_start) * 1000
            if i > NUM_WARM_UP:
                loop_times.append(loop_time)
            print(f"loop {i} time: {loop_time:.2f}ms")
            i += 1

    except Exception:
        traceback.print_exc()

    finally:
        # Cleanup
        print("Exiting...")
        if loop_times:
            print(f"Average loop time: {statistics.mean(loop_times):.2f}ms")
            print(f"Std loop time: {statistics.stdev(loop_times):.2f}ms")
        profile_path = os.path.join(args.outdir, "profile_output.lprof")
        dump_profiling_data(profile_path)


if __name__ == "__main__":
    """
    python examples/test_foundation_stereo.py --vis --save-output --engine <engine_path> --calib_params <calib_params_path> --rec_params <rec_params_path>
    """
    args = parse_args()
    main(args)
