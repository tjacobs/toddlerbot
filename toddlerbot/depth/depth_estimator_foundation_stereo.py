"""TensorRT-accelerated stereo depth estimation using Foundation Stereo model.

This module provides stereo depth estimation capabilities using a TensorRT engine
for high-performance inference with camera calibration and rectification support.
"""

import atexit
import pickle
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import torch
import torch.nn.functional as F
from PIL import Image

from toddlerbot.depth.depth_utils import (
    depth_to_xyzmap,
    get_rectification_maps,
    to_open3d_Cloud,
)
from toddlerbot.sensing.camera import Camera
from toddlerbot.utils.misc_utils import log, profile

# Initialize CUDA context (gracefully handle missing CUDA)
try:
    cuda.init()
    cuda.Device(0).retain_primary_context().push()
    atexit.register(cuda.Context.pop)
    CUDA_STREAM = torch.cuda.Stream()

    # Normalization constants borrowed from ImageNet. Used for pre-processing before inference.
    MEAN = torch.tensor([0.485, 0.456, 0.406], device="cuda")[:, None, None]
    STD = torch.tensor([0.229, 0.224, 0.225], device="cuda")[:, None, None]

    # Initialize TensorRT logger
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")
    CUDA_AVAILABLE = True
except (RuntimeError, ImportError, AttributeError) as e:
    # Handle cases where CUDA/TensorRT is not available
    CUDA_STREAM = None
    MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    TRT_LOGGER = None
    CUDA_AVAILABLE = False
    print(f"Warning: CUDA/TensorRT not available for depth estimation: {e}")


def gpu_preproc(img: Image.Image) -> torch.Tensor:
    """Preprocess PIL Image for TensorRT inference with GPU normalization."""
    arr = np.asarray(img, np.uint8).copy()
    t = torch.as_tensor(arr, device="cuda").permute(2, 0, 1).unsqueeze_(0).float()
    t.sub_(MEAN).div_(STD)  # normalize
    return t.contiguous()


def pad_to_multiple(t: torch.Tensor, k: int = 32) -> Tuple[torch.Tensor, int, int]:
    """Pad tensor dimensions to nearest multiple of k for model requirements."""
    _, _, h, w = t.shape
    ph = (k - h % k) % k
    pw = (k - w % k) % k
    if ph or pw:  # pad if not multiple of k
        t = F.pad(t, (0, pw, 0, ph))
    return t.contiguous(), ph, pw


@dataclass(frozen=True, slots=True)
class DepthResult:
    """Container for all artifacts produced by depth estimation."""

    depth: np.ndarray
    disparity: Optional[np.ndarray] = None
    rectified_left: Optional[np.ndarray] = None
    rectified_right: Optional[np.ndarray] = None
    original_left: Optional[np.ndarray] = None
    original_right: Optional[np.ndarray] = None


class DepthEstimatorFoundationStereo:
    """Depth estimation using Foundation Stereo model."""

    def __init__(
        self,
        calib_params_path,
        rec_params_path,
        engine_path,
        calib_width,
        calib_height,
        skip_rectify=False,
        debug=False,
    ):
        self.skip_rectify = skip_rectify
        self.debug = debug

        if not self.debug:
            self.camera_left = Camera("left", width=calib_width, height=calib_height)
            self.camera_right = Camera("right", width=calib_width, height=calib_height)

        # initialize tensorrt engine
        self._init_engine(engine_path)

        # initialize calibration and rectification maps
        self._init_calibration(
            calib_params_path, rec_params_path, calib_width, calib_height, skip_rectify
        )

    def _init_engine(self, engine_path) -> None:
        """
        Initialize tensorrt engine and allocate buffers.
        Args:
            engine_path: Path to the tensorrt engine file.
        Returns:
            None
        """
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
            self.ctx = self.engine.create_execution_context()
            self.stream = cuda.Stream()
            self.dev: Dict[str, cuda.DeviceAllocation] = {}
            self.host: Dict[str, np.ndarray] = {}
            expected_shape = None
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                shape = [d if d > 0 else 1 for d in self.engine.get_tensor_shape(name)]
                dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
                nbytes = int(np.prod(shape) * dtype.itemsize)

                # Extract spatial dimensions (height, width)
                spatial_shape = shape[2:]
                if expected_shape is None:
                    expected_shape = spatial_shape

                is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT

                if is_input:
                    self.dev[name] = cuda.mem_alloc(nbytes)
                else:
                    host = cuda.pagelocked_empty(int(np.prod(shape)), dtype=dtype)
                    dev = cuda.mem_alloc(nbytes)
                    self.host[name], self.dev[name] = host, dev

                self.ctx.set_tensor_address(name, int(self.dev[name]))

            assert expected_shape is not None, "Expected shape is not set"
            self.height, self.width = expected_shape
            log(f"Engine shape: {self.height}x{self.width}", header="DepthEstimator")
            log("Initial I/O buffers ready", header="DepthEstimator")

    def _init_calibration(
        self,
        calib_params_path,
        rec_params_path,
        calib_width,
        calib_height,
        skip_rectify,
    ) -> None:
        """
        Initialize calibration and rectification maps.
        Args:
            calib_params_path: Path to the calibration parameters file.
            rec_params_path: Path to the rectification parameters file.
            calib_width: Width used for calibration.
            calib_height: Height used for calibration.
            skip_rectify: Whether to skip rectification.
        Returns:
            None
        """
        with open(calib_params_path, "rb") as f:
            calib_params = pickle.load(f)
            self.K1 = calib_params["K1"]
            self.K2 = calib_params["K2"]
            scale_factor = self.width / calib_width
            assert np.isclose(scale_factor, self.height / calib_height), (
                f"Scale factor must be the same for both width and height: {scale_factor} != {self.height / calib_height}"
            )
            # self.scaled_K = self.K1.copy()
            # self.scaled_K[:2] *= scale_factor
            self.T = (calib_params["T"],)
            self.baseline = np.linalg.norm(self.T)
            if self.T[0][0] < 0:
                self.scaled_K = self.K1.copy()
                self.scaled_K[:2] *= scale_factor
            elif self.T[0][0] > 0:
                self.scaled_K = self.K2.copy()
                self.scaled_K[:2] *= scale_factor
            else:
                raise ValueError("Baseline is zero")

        with open(rec_params_path, "rb") as f:
            rec_params = pickle.load(f)

            # P1 and P2 are the same except P2[0,3]
            self.P1 = rec_params["P1"]
            scale_factor = self.width / calib_width
            scaled_rectified_fx = (
                self.P1[0, 0] * scale_factor
            )  # Focal length from P1 matrix
            scaled_rectified_fy = self.P1[1, 1] * scale_factor
            scaled_rectified_cx = self.P1[0, 2] * scale_factor
            scaled_rectified_cy = self.P1[1, 2] * scale_factor
            self.scaled_rectified_K = np.array(
                [
                    [scaled_rectified_fx, 0, scaled_rectified_cx],
                    [0, scaled_rectified_fy, scaled_rectified_cy],
                    [0, 0, 1],
                ]
            )

            # Pre-compute the full numerator for maximum speed in the inference loop
            self.fx_times_baseline = scaled_rectified_fx * self.baseline

        self.map1_left, self.map2_left, self.map1_right, self.map2_right = (
            get_rectification_maps(
                calib_params, rec_params, (calib_width, calib_height)
            )
        )

        # Pre-compute combined rectification and resize maps
        if not skip_rectify:
            self.combined_map1_left = cv2.resize(
                self.map1_left,
                (self.width, self.height),
                interpolation=cv2.INTER_LINEAR,
            )
            self.combined_map2_left = cv2.resize(
                self.map2_left,
                (self.width, self.height),
                interpolation=cv2.INTER_LINEAR,
            )
            self.combined_map1_right = cv2.resize(
                self.map1_right,
                (self.width, self.height),
                interpolation=cv2.INTER_LINEAR,
            )
            self.combined_map2_right = cv2.resize(
                self.map2_right,
                (self.width, self.height),
                interpolation=cv2.INTER_LINEAR,
            )

    def _infer(self, left: Image.Image, right: Image.Image) -> Dict[str, np.ndarray]:
        """
        Run inference on the tensorrt engine to obtain the disparity map.

        Args:
            left: Left image.
            right: Right image.
        Returns:
            Dictionary of output tensors.
        """
        # preprocessing
        with torch.cuda.stream(CUDA_STREAM):
            left_tensor = gpu_preproc(left)
            right_tensor = gpu_preproc(right)
        CUDA_STREAM.synchronize()

        inputs = [
            n
            for n in self.dev
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
        ]

        if len(inputs) == 1:  # 6-channel path
            inp_tensors = [torch.cat([left_tensor, right_tensor], dim=1)]
        elif len(inputs) == 2:  # 3-ch + 3-ch path
            inp_tensors = [left_tensor, right_tensor]
        else:
            raise RuntimeError("Engine has unexpected number of inputs")

        # upload inputs
        ph = pw = 0
        for name, tensor in zip(inputs, inp_tensors):
            tensor, ph, pw = pad_to_multiple(tensor, 32)
            needed = tensor.element_size() * tensor.numel()
            cuda.memcpy_dtod_async(
                self.dev[name], int(tensor.data_ptr()), needed, self.stream
            )

            if any(d < 0 for d in self.engine.get_tensor_shape(name)):
                self.ctx.set_input_shape(name, tuple(tensor.shape))

        # run inference
        self.ctx.execute_async_v3(stream_handle=self.stream.handle)

        # download outputs
        for name in self.host:
            out_shape = tuple(self.ctx.get_tensor_shape(name))
            needed = np.prod(out_shape) * 4  # float32
            cuda.memcpy_dtoh_async(self.host[name], self.dev[name], self.stream)
        self.stream.synchronize()

        # unpack
        outs: Dict[str, np.ndarray] = {}
        for name, buf in self.host.items():
            out_shape = tuple(self.ctx.get_tensor_shape(name))
            arr = np.frombuffer(buf, dtype=np.float32, count=int(np.prod(out_shape)))
            arr = arr.reshape(out_shape)
            if ph or pw:
                arr = arr[:, :, : -ph or None, : -pw or None]
            outs[name] = arr.squeeze()
        return outs

    def __del__(self) -> None:
        """
        Destructor to release camera resources.
        """
        # if camera is initialized, release it
        if hasattr(self, "camera_left") and self.camera_left is not None:
            self.camera_left.close()
        if hasattr(self, "camera_right") and self.camera_right is not None:
            self.camera_right.close()

    def _process_images(self, img0, img1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process images with combined rectification and resize.
        Args:
            img0: Left image.
            img1: Right image.
        Returns:
            Tuple of processed images.
        """
        if not self.skip_rectify:
            img0 = cv2.remap(
                img0,
                self.combined_map1_left,
                self.combined_map2_left,
                interpolation=cv2.INTER_LINEAR,
            )
            img1 = cv2.remap(
                img1,
                self.combined_map1_right,
                self.combined_map2_right,
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            # Just resize if no rectification
            img0 = cv2.resize(
                img0, (self.width, self.height), interpolation=cv2.INTER_AREA
            )
            img1 = cv2.resize(
                img1, (self.width, self.height), interpolation=cv2.INTER_AREA
            )

        return img0, img1

    @profile()
    def get_depth(
        self,
        *,
        remove_invisible: bool = False,
        debug_images: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        return_all: bool = False,
    ) -> DepthResult:
        """Estimate per-pixel depth from the stereo pair.

        Args:
            remove_invisible: If ``True``, mask points that project outside the
                right-camera image.
            debug_images: Optional pre-captured (left, right) images. When
                ``None``, frames are captured from the cameras.
            return_all: If ``True``, populate every field in :class:`DepthResult`;
                otherwise only ``depth`` is guaranteed to be non-``None``.

        Returns:
            An immutable :class:`DepthResult` whose fields are either populated or
            ``None`` depending on ``return_all``.
        """
        # Acquire images
        if debug_images is not None:
            img_left, img_right = debug_images
        else:
            img_left = self.camera_left.get_frame()
            img_right = self.camera_right.get_frame()

        original_left = img_left.copy() if return_all else None
        original_right = img_right.copy() if return_all else None

        # Pre-process
        img_left, img_right = self._process_images(img_left, img_right)

        rectified_left = img_left.copy() if return_all else None
        rectified_right = img_right.copy() if return_all else None

        # Depth inference
        # Switch the order when the sign of the baseline is positive which means right images are rectified to the left side on the baseline
        # disparity: np.ndarray = next(iter(self._infer(img_left, img_right).values()))
        if self.T[0][0] < 0:
            disparity: np.ndarray = next(
                iter(self._infer(img_left, img_right).values())
            )
        elif self.T[0][0] > 0:
            disparity: np.ndarray = next(
                iter(self._infer(img_right, img_left).values())
            )
        else:
            raise ValueError("Baseline is zero")

        if remove_invisible:
            xx = np.arange(disparity.shape[1])[None, :]
            invalid = (xx - disparity) < 0
            disparity = disparity.copy()
            disparity[invalid] = np.inf

        # depth: np.ndarray = (
        #     self.scaled_K[0, 0] * self.baseline / disparity
        # )  # z = f*B / disparity
        depth: np.ndarray = self.fx_times_baseline / disparity

        return DepthResult(
            depth=depth,
            disparity=disparity if return_all else None,
            rectified_left=rectified_left,
            rectified_right=rectified_right,
            original_left=original_left,
            original_right=original_right,
        )

    def get_pcl(
        self,
        depth,
        resized_image,
        is_BGR=True,
        zmim=0.0,
        zmax=np.inf,
        denoise_cloud=False,
        denoise_nb_points=30,
        denoise_radius=0.03,
    ):
        """
        Convert depth map to point cloud.
        Args:
            depth: Depth map.
            resized_image: Resized image.
            is_BGR: Whether the image is in BGR format.
            zmin: Minimum depth (meters).
            zmax: Maximum depth (meters).
            denoise_cloud: Whether to denoise the point cloud.
            denoise_nb_points: Number of points to use for radius outlier removal.
            denoise_radius: Radius for radius outlier removal.
        Returns:
            Point cloud (open3d.geometry.PointCloud)
        """
        # xyz_map = depth_to_xyzmap(depth, self.scaled_K, zmin=zmim)
        # TODO: Use cv2.reprojectImageTo3D using disparity and Q matrix instead?
        xyz_map = depth_to_xyzmap(depth, self.scaled_rectified_K, zmin=zmim)

        if is_BGR:
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

        pcd = to_open3d_Cloud(xyz_map.reshape(-1, 3), resized_image.reshape(-1, 3))
        points = np.asarray(pcd.points)
        keep_ids = np.where((points[:, 2] > 0) & (points[:, 2] <= zmax))[0]
        pcd = pcd.select_by_index(keep_ids)

        if denoise_cloud:
            _, ind = pcd.remove_radius_outlier(
                nb_points=denoise_nb_points, radius=denoise_radius
            )
            pcd = pcd.select_by_index(ind)

        return pcd
