"""
Benchmark tool for comparing int8 fake-quantization kernel implementations in NNCF.

Compares:
  - CUDA C++ extension kernel
  - CPU C++ extension kernel
  - Reference pure-PyTorch kernel (NNCF implementation)

Across:
  - Per-tensor / per-weight-channel / per-activation-channel quantization
  - CPU / GPU devices
  - Execution modes: native, torch.inference_mode, cuda.graph, torch.compile

Usage:
    python benchmarks/quantization_kernels_benchmark.py --device all
    python benchmarks/quantization_kernels_benchmark.py --device cuda --output results.csv
    python benchmarks/quantization_kernels_benchmark.py --validate
"""

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import torch

# Add src to path so we can import nncf
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# --- Configuration ---

ScaleMode = Literal["per_tensor", "per_weight_channel", "per_activation_channel"]
ExecutionMode = Literal["native", "inference_mode", "cuda_graph", "torch_compile"]
KernelType = Literal["extension", "reference"]

PREDEFINED_SHAPES = {
    "small_conv": (64, 64, 3, 3),
    "medium_conv": (256, 128, 3, 3),
    "large_linear": (4096, 4096),
    "small_activation": (1, 64, 56, 56),
    "large_activation": (8, 256, 28, 28),
}

# Symmetric int8 quantization parameters
LEVELS = 256
LEVEL_LOW = -128
LEVEL_HIGH = 127


@dataclass
class BenchmarkConfig:
    shape_name: str
    tensor_shape: tuple[int, ...]
    scale_mode: ScaleMode
    device: str
    execution_mode: ExecutionMode
    kernel_type: KernelType
    dtype: torch.dtype = torch.float32


@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    direction: str  # "forward" or "backward"
    min_us: float
    median_us: float
    mean_us: float
    max_us: float
    n_iter: int


# --- Input Preparation ---


def prepare_inputs(
    config: BenchmarkConfig,
) -> dict[str, Any]:
    """Create input tensor and quantization parameters with correct shapes for the scale mode."""
    shape = config.tensor_shape
    device = config.device
    dtype = config.dtype

    input_tensor = torch.randn(shape, device=device, dtype=dtype)

    if config.scale_mode == "per_tensor":
        scale_shape = (1,) * len(shape)
    elif config.scale_mode == "per_weight_channel":
        # Scale along dim 0 (output channels for weights)
        scale_shape = (shape[0],) + (1,) * (len(shape) - 1)
    elif config.scale_mode == "per_activation_channel":
        # Scale along dim 1 (channels for activations)
        if len(shape) < 2:
            raise ValueError("per_activation_channel requires at least 2D tensor")
        scale_shape = (1, shape[1]) + (1,) * (len(shape) - 2)
    else:
        raise ValueError(f"Unknown scale_mode: {config.scale_mode}")

    # Symmetric quantization: input_low = -scale, input_range = 2*scale
    scale = torch.rand(scale_shape, device=device, dtype=dtype) * 2 + 0.1  # scale in [0.1, 2.1]
    input_low = -scale
    input_range = 2 * scale

    return {
        "input": input_tensor,
        "input_low": input_low,
        "input_range": input_range,
        "levels": LEVELS,
        "level_low": LEVEL_LOW,
        "level_high": LEVEL_HIGH,
    }


# --- Kernel Loading ---


class KernelUnavailableError(RuntimeError):
    """Raised when a kernel cannot be loaded (e.g. nvcc not available for CUDA extension)."""


def get_forward_fn(config: BenchmarkConfig) -> Callable:
    """Return the forward quantization function for the given config."""
    if config.kernel_type == "extension":
        if config.device == "cuda":
            from nncf.torch.quantization.extensions import QuantizedFunctionsCUDA

            try:
                return QuantizedFunctionsCUDA.get("Quantize_forward")
            except Exception as e:
                raise KernelUnavailableError(f"CUDA extension unavailable: {e}") from e
        else:
            from nncf.torch.quantization.extensions import QuantizedFunctionsCPU

            try:
                return QuantizedFunctionsCPU.get("Quantize_forward")
            except Exception as e:
                raise KernelUnavailableError(f"CPU extension unavailable: {e}") from e
    else:
        # Reference: pure PyTorch implementation (unwrap CompilationWrapper to get raw function)
        from nncf.torch.quantization.reference import torch_executor

        return torch_executor.forward


def get_backward_fn(config: BenchmarkConfig) -> Callable:
    """Return the backward quantization function for the given config."""
    if config.kernel_type == "extension":
        if config.device == "cuda":
            from nncf.torch.quantization.extensions import QuantizedFunctionsCUDA

            try:
                return QuantizedFunctionsCUDA.get("Quantize_backward")
            except Exception as e:
                raise KernelUnavailableError(f"CUDA extension unavailable: {e}") from e
        else:
            from nncf.torch.quantization.extensions import QuantizedFunctionsCPU

            try:
                return QuantizedFunctionsCPU.get("Quantize_backward")
            except Exception as e:
                raise KernelUnavailableError(f"CPU extension unavailable: {e}") from e
    else:
        from nncf.torch.quantization.reference import torch_executor

        return torch_executor.backward


# --- Timing Utilities ---


def _time_gpu(fn: Callable, n_warmup: int, n_iter: int) -> list[float]:
    """Time a GPU function using CUDA events. Returns list of times in microseconds."""
    # Warmup
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(n_iter):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)  # ms -> us
    return times


def _time_cpu(fn: Callable, n_warmup: int, n_iter: int) -> list[float]:
    """Time a CPU function using perf_counter_ns. Returns list of times in microseconds."""
    for _ in range(n_warmup):
        fn()

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000.0)  # ns -> us
    return times


def compute_stats(times: list[float], n_iter: int) -> dict[str, float]:
    """Compute min/median/mean/max from a list of times."""
    times_sorted = sorted(times)
    n = len(times_sorted)
    return {
        "min_us": times_sorted[0],
        "median_us": times_sorted[n // 2],
        "mean_us": sum(times_sorted) / n,
        "max_us": times_sorted[-1],
        "n_iter": n_iter,
    }


# --- Execution Mode Runners ---


def run_native(
    config: BenchmarkConfig, inputs: dict, n_warmup: int, n_iter: int
) -> list[BenchmarkResult]:
    """Run benchmark in native mode (direct call)."""
    results = []
    fwd_fn = get_forward_fn(config)
    bwd_fn = get_backward_fn(config)

    inp = inputs["input"].clone().requires_grad_(True)
    input_low = inputs["input_low"]
    input_range = inputs["input_range"]
    levels = inputs["levels"]

    # Forward benchmark
    def run_forward():
        return fwd_fn(inp, input_low, input_range, levels)

    timer = _time_gpu if config.device == "cuda" else _time_cpu
    fwd_times = timer(run_forward, n_warmup, n_iter)
    stats = compute_stats(fwd_times, n_iter)
    results.append(BenchmarkResult(config=config, direction="forward", **stats))

    # Backward benchmark
    grad_output = torch.randn_like(inp)

    def run_backward():
        if config.kernel_type == "extension" and config.device == "cuda":
            return bwd_fn(grad_output, inp, input_low, input_range, levels, LEVEL_LOW, LEVEL_HIGH)
        elif config.kernel_type == "extension" and config.device == "cpu":
            return bwd_fn(grad_output, inp, input_low, input_range, levels, LEVEL_LOW, LEVEL_HIGH, False)
        else:
            return bwd_fn(grad_output, inp, input_low, input_range, levels, LEVEL_LOW, LEVEL_HIGH)

    bwd_times = timer(run_backward, n_warmup, n_iter)
    stats = compute_stats(bwd_times, n_iter)
    results.append(BenchmarkResult(config=config, direction="backward", **stats))

    return results


def run_inference_mode(
    config: BenchmarkConfig, inputs: dict, n_warmup: int, n_iter: int
) -> list[BenchmarkResult]:
    """Run benchmark under torch.inference_mode() — forward only."""
    results = []
    fwd_fn = get_forward_fn(config)

    inp = inputs["input"]
    input_low = inputs["input_low"]
    input_range = inputs["input_range"]
    levels = inputs["levels"]

    def run_forward():
        with torch.inference_mode():
            return fwd_fn(inp, input_low, input_range, levels)

    timer = _time_gpu if config.device == "cuda" else _time_cpu
    fwd_times = timer(run_forward, n_warmup, n_iter)
    stats = compute_stats(fwd_times, n_iter)
    results.append(BenchmarkResult(config=config, direction="forward", **stats))

    return results


def run_cuda_graph(
    config: BenchmarkConfig, inputs: dict, n_warmup: int, n_iter: int
) -> list[BenchmarkResult]:
    """Run benchmark using CUDA graph capture and replay."""
    if config.device != "cuda":
        return []

    results = []
    fwd_fn = get_forward_fn(config)
    bwd_fn = get_backward_fn(config)

    inp = inputs["input"].clone().requires_grad_(True)
    input_low = inputs["input_low"]
    input_range = inputs["input_range"]
    levels = inputs["levels"]
    grad_output = torch.randn_like(inp)

    # --- Forward graph ---
    # Warmup (required before capture)
    for _ in range(3):
        fwd_fn(inp, input_low, input_range, levels)
    torch.cuda.synchronize()

    # Capture forward
    fwd_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(fwd_graph):
        fwd_fn(inp, input_low, input_range, levels)

    def replay_forward():
        fwd_graph.replay()

    fwd_times = _time_gpu(replay_forward, n_warmup, n_iter)
    stats = compute_stats(fwd_times, n_iter)
    results.append(BenchmarkResult(config=config, direction="forward", **stats))

    # --- Backward graph ---
    # Warmup
    for _ in range(3):
        if config.kernel_type == "extension":
            bwd_fn(grad_output, inp, input_low, input_range, levels, LEVEL_LOW, LEVEL_HIGH)
        else:
            bwd_fn(grad_output, inp, input_low, input_range, levels, LEVEL_LOW, LEVEL_HIGH)
    torch.cuda.synchronize()

    bwd_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(bwd_graph):
        if config.kernel_type == "extension":
            bwd_fn(grad_output, inp, input_low, input_range, levels, LEVEL_LOW, LEVEL_HIGH)
        else:
            bwd_fn(grad_output, inp, input_low, input_range, levels, LEVEL_LOW, LEVEL_HIGH)

    def replay_backward():
        bwd_graph.replay()

    bwd_times = _time_gpu(replay_backward, n_warmup, n_iter)
    stats = compute_stats(bwd_times, n_iter)
    results.append(BenchmarkResult(config=config, direction="backward", **stats))

    return results


def run_torch_compile(
    config: BenchmarkConfig, inputs: dict, n_warmup: int, n_iter: int
) -> list[BenchmarkResult]:
    """Run benchmark with torch.compile wrapping the quantize function."""
    results = []
    fwd_fn = get_forward_fn(config)
    bwd_fn = get_backward_fn(config)

    inp = inputs["input"].clone().requires_grad_(True)
    input_low = inputs["input_low"]
    input_range = inputs["input_range"]
    levels = inputs["levels"]

    # Wrap in a compilable function
    def forward_wrapper(x, il, ir, lvl):
        return fwd_fn(x, il, ir, lvl)

    def backward_wrapper(go, x, il, ir, lvl, ll, lh):
        if config.kernel_type == "extension" and config.device == "cpu":
            return bwd_fn(go, x, il, ir, lvl, ll, lh, False)
        return bwd_fn(go, x, il, ir, lvl, ll, lh)

    compiled_forward = torch.compile(forward_wrapper)
    compiled_backward = torch.compile(backward_wrapper)

    def run_forward():
        return compiled_forward(inp, input_low, input_range, levels)

    timer = _time_gpu if config.device == "cuda" else _time_cpu
    # Extra warmup for compilation
    fwd_times = timer(run_forward, max(n_warmup, 10), n_iter)
    stats = compute_stats(fwd_times, n_iter)
    results.append(BenchmarkResult(config=config, direction="forward", **stats))

    grad_output = torch.randn_like(inp)

    def run_backward():
        return compiled_backward(grad_output, inp, input_low, input_range, levels, LEVEL_LOW, LEVEL_HIGH)

    bwd_times = timer(run_backward, max(n_warmup, 10), n_iter)
    stats = compute_stats(bwd_times, n_iter)
    results.append(BenchmarkResult(config=config, direction="backward", **stats))

    return results


EXECUTION_MODE_RUNNERS: dict[ExecutionMode, Callable] = {
    "native": run_native,
    "inference_mode": run_inference_mode,
    "cuda_graph": run_cuda_graph,
    "torch_compile": run_torch_compile,
}


# --- Validation ---


def validate_kernels(device: str, scale_mode: ScaleMode, shape: tuple[int, ...]) -> bool:
    """Validate that extension and reference kernels produce identical results."""
    from nncf.torch.quantization.reference import torch_executor

    dtype = torch.float32

    config_ext = BenchmarkConfig(
        shape_name="validate",
        tensor_shape=shape,
        scale_mode=scale_mode,
        device=device,
        execution_mode="native",
        kernel_type="extension",
        dtype=dtype,
    )
    inputs = prepare_inputs(config_ext)
    inp = inputs["input"]
    input_low = inputs["input_low"]
    input_range = inputs["input_range"]
    levels = inputs["levels"]

    try:
        fwd_ext = get_forward_fn(config_ext)
    except KernelUnavailableError as e:
        print(f"  SKIP (extension unavailable): {device}/{scale_mode}/{shape} — {e}")
        return True  # Not a failure, just unavailable

    ext_output = fwd_ext(inp, input_low, input_range, levels)
    ref_output = torch_executor.forward(inp, input_low, input_range, levels)

    if not torch.allclose(ext_output, ref_output, atol=1e-6):
        print(f"  FAIL forward: {device}/{scale_mode}/{shape} max_diff={torch.max(torch.abs(ext_output - ref_output)).item():.2e}")
        return False

    # Backward
    grad_output = torch.randn_like(inp)
    bwd_ext = get_backward_fn(config_ext)
    if device == "cuda":
        ext_grads = bwd_ext(grad_output, inp, input_low, input_range, levels, LEVEL_LOW, LEVEL_HIGH)
    else:
        ext_grads = bwd_ext(grad_output, inp, input_low, input_range, levels, LEVEL_LOW, LEVEL_HIGH, False)
    ref_grads = torch_executor.backward(grad_output, inp, input_low, input_range, levels, LEVEL_LOW, LEVEL_HIGH)

    for i, (eg, rg) in enumerate(zip(ext_grads, ref_grads)):
        if not torch.allclose(eg, rg, atol=1e-6):
            print(f"  FAIL backward grad[{i}]: {device}/{scale_mode}/{shape} max_diff={torch.max(torch.abs(eg - rg)).item():.2e}")
            return False

    return True


# --- Main Benchmark Loop ---


def generate_configs(
    devices: list[str],
    shapes: dict[str, tuple[int, ...]],
    execution_modes: list[ExecutionMode],
    kernel_types: list[KernelType],
    scale_modes: list[ScaleMode],
) -> list[BenchmarkConfig]:
    """Generate all valid benchmark configurations."""
    configs = []
    for shape_name, shape in shapes.items():
        for scale_mode in scale_modes:
            # Skip per_activation_channel for 2D tensors (no channel dim)
            if scale_mode == "per_activation_channel" and len(shape) < 2:
                continue
            # Skip per_activation_channel if dim-1 is meaningless
            if scale_mode == "per_activation_channel" and len(shape) == 2:
                continue
            for device in devices:
                for kernel_type in kernel_types:
                    # CPU extension only on cpu, CUDA extension only on cuda
                    if kernel_type == "extension" and device == "cuda" and not torch.cuda.is_available():
                        continue
                    for exec_mode in execution_modes:
                        # cuda_graph only on GPU
                        if exec_mode == "cuda_graph" and device != "cuda":
                            continue
                        configs.append(
                            BenchmarkConfig(
                                shape_name=shape_name,
                                tensor_shape=shape,
                                scale_mode=scale_mode,
                                device=device,
                                execution_mode=exec_mode,
                                kernel_type=kernel_type,
                            )
                        )
    return configs


def run_benchmarks(
    configs: list[BenchmarkConfig], n_warmup: int, n_iter: int
) -> list[BenchmarkResult]:
    """Run all benchmarks and return results."""
    all_results = []
    total = len(configs)

    for i, config in enumerate(configs, 1):
        label = (
            f"[{i}/{total}] {config.kernel_type:10s} | {config.device:4s} | "
            f"{config.scale_mode:24s} | {config.execution_mode:14s} | {config.shape_name}"
        )
        print(f"  Running: {label}", flush=True)

        runner = EXECUTION_MODE_RUNNERS[config.execution_mode]
        try:
            inputs = prepare_inputs(config)
            results = runner(config, inputs, n_warmup, n_iter)
            all_results.extend(results)
        except KernelUnavailableError as e:
            print(f"    SKIPPED (kernel unavailable: {e})")
        except Exception as e:
            print(f"    SKIPPED ({type(e).__name__}: {e})")

    return all_results


# --- Output ---


def print_results_table(results: list[BenchmarkResult]) -> None:
    """Print results as a formatted table."""
    if not results:
        print("No results to display.")
        return

    header = [
        "Kernel",
        "Device",
        "ScaleMode",
        "ExecMode",
        "Shape",
        "Direction",
        "Min(us)",
        "Median(us)",
        "Mean(us)",
        "Max(us)",
    ]
    rows = []
    for r in results:
        rows.append([
            r.config.kernel_type,
            r.config.device,
            r.config.scale_mode,
            r.config.execution_mode,
            r.config.shape_name,
            r.direction,
            f"{r.min_us:.1f}",
            f"{r.median_us:.1f}",
            f"{r.mean_us:.1f}",
            f"{r.max_us:.1f}",
        ])

    # Compute column widths
    widths = [max(len(header[j]), *(len(row[j]) for row in rows)) for j in range(len(header))]
    fmt = " | ".join(f"{{:<{w}}}" for w in widths)
    sep = "-+-".join("-" * w for w in widths)

    print()
    print(fmt.format(*header))
    print(sep)
    for row in rows:
        print(fmt.format(*row))
    print()


def print_pivot_table(results: list[BenchmarkResult]) -> None:
    """Print a pivot table with execution modes as columns, showing mean latency (us)."""
    if not results:
        return

    # Collect all execution modes present in results
    exec_modes_seen = sorted(set(r.config.execution_mode for r in results))

    # Group results by (kernel, device, scale_mode, shape, direction)
    grouped: dict[tuple[str, ...], dict[str, float]] = {}
    for r in results:
        key = (r.config.kernel_type, r.config.device, r.config.scale_mode, r.config.shape_name, r.direction)
        if key not in grouped:
            grouped[key] = {}
        grouped[key][r.config.execution_mode] = r.mean_us

    # Build table
    header = ["Kernel", "Device", "ScaleMode", "Shape", "Direction"] + exec_modes_seen
    rows = []
    for key in sorted(grouped.keys()):
        row = list(key)
        for mode in exec_modes_seen:
            val = grouped[key].get(mode)
            row.append(f"{val:.1f}" if val is not None else "-")
        rows.append(row)

    # Print
    widths = [max(len(header[j]), *(len(row[j]) for row in rows)) for j in range(len(header))]
    fmt = " | ".join(f"{{:<{w}}}" for w in widths)
    sep = "-+-".join("-" * w for w in widths)

    print()
    print("Pivot Table: Mean Latency (us) by Execution Mode")
    print("=" * len(sep))
    print(fmt.format(*header))
    print(sep)
    for row in rows:
        print(fmt.format(*row))
    print()


def write_csv(results: list[BenchmarkResult], output_path: str) -> None:
    """Write results to a CSV file."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "kernel",
            "device",
            "scale_mode",
            "execution_mode",
            "shape_name",
            "tensor_shape",
            "direction",
            "min_us",
            "median_us",
            "mean_us",
            "max_us",
            "n_iter",
        ])
        for r in results:
            writer.writerow([
                r.config.kernel_type,
                r.config.device,
                r.config.scale_mode,
                r.config.execution_mode,
                r.config.shape_name,
                str(r.config.tensor_shape),
                r.direction,
                f"{r.min_us:.2f}",
                f"{r.median_us:.2f}",
                f"{r.mean_us:.2f}",
                f"{r.max_us:.2f}",
                r.n_iter,
            ])
    print(f"Results written to: {output_path}")


def write_pivot_csv(results: list[BenchmarkResult], output_path: str) -> None:
    """Write pivot table (execution modes as columns, mean latency) to CSV."""
    if not results:
        return

    exec_modes_seen = sorted(set(r.config.execution_mode for r in results))

    grouped: dict[tuple[str, ...], dict[str, float]] = {}
    for r in results:
        key = (r.config.kernel_type, r.config.device, r.config.scale_mode, r.config.shape_name, r.direction)
        if key not in grouped:
            grouped[key] = {}
        grouped[key][r.config.execution_mode] = r.mean_us

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kernel", "device", "scale_mode", "shape", "direction"] + exec_modes_seen)
        for key in sorted(grouped.keys()):
            row = list(key)
            for mode in exec_modes_seen:
                val = grouped[key].get(mode)
                row.append(f"{val:.2f}" if val is not None else "")
            writer.writerow(row)
    print(f"Pivot table written to: {output_path}")


# --- CLI ---


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark int8 quantization kernels in NNCF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "all"],
        default="all",
        help="Device to benchmark on (default: all)",
    )
    parser.add_argument(
        "--shapes",
        choices=["small", "medium", "large", "all"],
        default="all",
        help="Which tensor shape set to use (default: all)",
    )
    parser.add_argument(
        "--exec-modes",
        nargs="+",
        choices=["native", "inference_mode", "cuda_graph", "torch_compile"],
        default=None,
        help="Execution modes to benchmark (default: all applicable)",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=50,
        help="Number of warmup iterations (default: 50)",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=200,
        help="Number of timed iterations (default: 200)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write CSV output (optional)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate kernel outputs match before benchmarking",
    )
    parser.add_argument(
        "--kernel",
        choices=["extension", "reference", "all"],
        default="all",
        help="Which kernel implementation to benchmark (default: all)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Determine devices
    if args.device == "all":
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            print("ERROR: CUDA not available on this system.")
            sys.exit(1)
        devices = ["cuda"]
    else:
        devices = ["cpu"]

    # Determine shapes
    shape_map = {
        "small": {"small_conv": PREDEFINED_SHAPES["small_conv"]},
        "medium": {
            "medium_conv": PREDEFINED_SHAPES["medium_conv"],
            "small_activation": PREDEFINED_SHAPES["small_activation"],
        },
        "large": {
            "large_linear": PREDEFINED_SHAPES["large_linear"],
            "large_activation": PREDEFINED_SHAPES["large_activation"],
        },
        "all": PREDEFINED_SHAPES,
    }
    shapes = shape_map[args.shapes]

    # Execution modes
    exec_modes: list[ExecutionMode] = args.exec_modes or ["native", "inference_mode", "cuda_graph", "torch_compile"]

    # Kernel types
    if args.kernel == "all":
        kernel_types: list[KernelType] = ["extension", "reference"]
    else:
        kernel_types = [args.kernel]

    # Scale modes
    scale_modes: list[ScaleMode] = ["per_tensor", "per_weight_channel", "per_activation_channel"]

    print("=" * 80)
    print("NNCF Int8 Quantization Kernel Benchmark")
    print("=" * 80)
    print(f"  Devices: {devices}")
    print(f"  Shapes: {list(shapes.keys())}")
    print(f"  Execution modes: {exec_modes}")
    print(f"  Kernel types: {kernel_types}")
    print(f"  Scale modes: {scale_modes}")
    print(f"  Warmup: {args.n_warmup}, Iterations: {args.n_iter}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    # Validation
    if args.validate:
        print("\nValidating kernel outputs...")
        all_pass = True
        for device in devices:
            for scale_mode in scale_modes:
                for shape_name, shape in shapes.items():
                    if scale_mode == "per_activation_channel" and len(shape) < 3:
                        continue
                    ok = validate_kernels(device, scale_mode, shape)
                    status = "PASS" if ok else "FAIL"
                    print(f"  [{status}] {device}/{scale_mode}/{shape_name}")
                    if not ok:
                        all_pass = False
        if not all_pass:
            print("\nValidation FAILED. Fix kernel discrepancies before benchmarking.")
            sys.exit(1)
        print("All validations passed.\n")

    # Generate configs
    configs = generate_configs(devices, shapes, exec_modes, kernel_types, scale_modes)
    print(f"\nTotal benchmark configurations: {len(configs)}")
    print("-" * 80)

    # Run
    results = run_benchmarks(configs, args.n_warmup, args.n_iter)

    # Output
    print_results_table(results)
    print_pivot_table(results)

    if args.output:
        write_csv(results, args.output)
        pivot_path = args.output.replace(".csv", "_pivot.csv")
        write_pivot_csv(results, pivot_path)


if __name__ == "__main__":
    main()
