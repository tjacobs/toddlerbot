"""Benchmark PyTorch vs ONNX model inference performance.

This module compares the inference speed and output accuracy between PyTorch
and ONNX model formats for neural network evaluation.
"""

import argparse
import time

import numpy as np
import onnxruntime as ort
import torch


def load_onnx_inference(onnx_path, device="cpu"):
    """Load ONNX model and return inference function."""
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device == "cuda"
        else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    def inference_fn(obs):
        obs_numpy = obs.cpu().numpy().astype(np.float32)
        outputs = session.run([output_name], {input_name: obs_numpy})
        return torch.from_numpy(outputs[0])

    return inference_fn


def load_pytorch_inference(pt_path, obs_size, device="cpu"):
    """Load PyTorch model and return inference function."""
    # Load PyTorch model (adjust this based on your model structure)
    model = torch.load(pt_path, map_location=device)
    if hasattr(model, "actor"):
        actor = model.actor
    else:
        actor = model  # Assume it's already the actor

    actor.eval()

    def inference_fn(obs):
        with torch.no_grad():
            return actor(obs).cpu() if device == "cuda" else actor(obs)

    return inference_fn


def benchmark_inference(inference_fn, test_obs, n_trials=10000):
    """Benchmark inference function speed over multiple trials."""
    # Warmup
    for _ in range(10):
        _ = inference_fn(test_obs)

    # Benchmark
    start_time = time.time()
    for _ in range(n_trials):
        _ = inference_fn(test_obs)
    return time.time() - start_time


def main():
    """Run ONNX vs PyTorch inference benchmark comparison."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-path", required=True, help="Path to ONNX model")
    parser.add_argument("--pt-path", required=True, help="Path to PyTorch model")
    parser.add_argument("--obs-size", type=int, default=1515, help="Observation size")
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for testing"
    )
    parser.add_argument(
        "--n-trials", type=int, default=10000, help="Number of inference trials"
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    # Create test input
    test_obs = torch.randn(args.batch_size, args.obs_size, dtype=torch.float32)
    if args.device == "cuda":
        test_obs = test_obs.cuda()

    print(f"Testing with obs shape: {test_obs.shape}")
    print(f"Device: {args.device}")
    print(f"Trials: {args.n_trials}")
    print("-" * 50)

    # Load inference functions
    try:
        onnx_fn = load_onnx_inference(args.onnx_path, args.device)
        pytorch_fn = load_pytorch_inference(args.pt_path, args.obs_size, args.device)
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # Verify outputs match
    onnx_out = onnx_fn(test_obs)
    pytorch_out = pytorch_fn(test_obs)
    max_diff = torch.max(torch.abs(onnx_out - pytorch_out)).item()
    print(f"Max output difference: {max_diff:.2e}")

    if max_diff > 1e-4:
        print("WARNING: Large difference between ONNX and PyTorch outputs!")

    # Benchmark
    onnx_time = benchmark_inference(onnx_fn, test_obs, args.n_trials)
    pytorch_time = benchmark_inference(pytorch_fn, test_obs, args.n_trials)

    print(
        f"PyTorch time: {pytorch_time:.4f}s ({pytorch_time / args.n_trials * 1000:.2f}ms per inference)"
    )
    print(
        f"ONNX time:    {onnx_time:.4f}s ({onnx_time / args.n_trials * 1000:.2f}ms per inference)"
    )
    print(f"Speedup:      {pytorch_time / onnx_time:.2f}x")


if __name__ == "__main__":
    main()
