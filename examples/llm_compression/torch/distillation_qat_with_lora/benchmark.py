# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Benchmark script for the QAT training loop with LoRA adapters.

Compares training throughput under different optimization strategies:
  1. Baseline: no optimization
  2. torch.compile on quantization kernels only (avoids NNCF wrapping issues)
  3. torch.compile on the full forward pass (may require graph-break-free path)
  4. CUDA graphs for the training step (requires static shapes)
  5. Pure-autograd quantize (replace custom autograd.Function with plain ops for compile)

Usage:
    python benchmark.py [--pretrained MODEL] [--seqlen 512] [--microbatch_size 2] [--warmup 5] [--iters 20]
"""

import argparse
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.jit import TracerWarning

warnings.filterwarnings("ignore", category=TracerWarning)


def get_nncf_imports():
    """Lazy imports to allow measuring import overhead."""
    import nncf
    from nncf.data.dataset import Dataset
    from nncf.parameters import CompressionFormat, CompressWeightsMode
    from nncf.quantization.advanced_parameters import AdvancedAWQParameters, AdvancedCompressionParameters
    from nncf.quantization.quantize_model import compress_weights
    from nncf.torch.function_hook.wrapper import get_hook_storage
    from nncf.torch.quantization.layers import AsymmetricLoraQuantizer, SymmetricLoraQuantizer

    return {
        "nncf": nncf,
        "Dataset": Dataset,
        "CompressionFormat": CompressionFormat,
        "CompressWeightsMode": CompressWeightsMode,
        "AdvancedAWQParameters": AdvancedAWQParameters,
        "AdvancedCompressionParameters": AdvancedCompressionParameters,
        "compress_weights": compress_weights,
        "get_hook_storage": get_hook_storage,
        "AsymmetricLoraQuantizer": AsymmetricLoraQuantizer,
        "SymmetricLoraQuantizer": SymmetricLoraQuantizer,
    }


# ─── Helpers ─────────────────────────────────────────────────────────────────


def get_model_input(input_ids: Tensor) -> dict[str, Tensor]:
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.cumsum(attention_mask, axis=1) - 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}


def kl_div(student_logits: Tensor, teacher_logits: Tensor) -> Tensor:
    num_classes = student_logits.shape[-1]
    return F.kl_div(
        input=F.log_softmax(student_logits.view(-1, num_classes), dim=-1),
        target=F.log_softmax(teacher_logits.view(-1, num_classes), dim=-1),
        log_target=True,
        reduction="batchmean",
    )


@contextmanager
def cuda_timer():
    """Context manager that returns elapsed GPU time in milliseconds."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    yield lambda: (end.record(), torch.cuda.synchronize(), start.elapsed_time(end))[-1]
    end.record()
    torch.cuda.synchronize()


class TimingResult:
    def __init__(self, name: str, times_ms: list[float]):
        self.name = name
        self.times_ms = times_ms

    @property
    def mean_ms(self) -> float:
        return sum(self.times_ms) / len(self.times_ms)

    @property
    def std_ms(self) -> float:
        mean = self.mean_ms
        return (sum((t - mean) ** 2 for t in self.times_ms) / len(self.times_ms)) ** 0.5

    def __repr__(self) -> str:
        return f"{self.name}: {self.mean_ms:.2f} ± {self.std_ms:.2f} ms/iter"


# ─── Optimization strategies ────────────────────────────────────────────────


def compile_quantize_kernels(model: nn.Module, nncf_modules: dict) -> None:
    """
    Compile each quantizer's quantize() method directly with torch.compile.

    This works because asymmetric_quantize_lora/symmetric_quantize_lora no longer have
    handle_torch_function wrappers, so torch.compile can trace the full quantize path.
    """
    get_hook_storage = nncf_modules["get_hook_storage"]
    AsymmetricLoraQuantizer = nncf_modules["AsymmetricLoraQuantizer"]
    SymmetricLoraQuantizer = nncf_modules["SymmetricLoraQuantizer"]

    hook_storage = get_hook_storage(model)
    compiled_count = 0
    for _, module in hook_storage.named_hooks():
        if isinstance(module, (AsymmetricLoraQuantizer, SymmetricLoraQuantizer)):
            module.quantize = torch.compile(module.quantize, mode="reduce-overhead", fullgraph=False)
            compiled_count += 1
    print(f"  Compiled {compiled_count} quantizers (mode='reduce-overhead')")


def compile_quantize_kernels_max_autotune(model: nn.Module, nncf_modules: dict) -> None:
    """
    Same as compile_quantize_kernels but with max-autotune mode.
    """
    get_hook_storage = nncf_modules["get_hook_storage"]
    AsymmetricLoraQuantizer = nncf_modules["AsymmetricLoraQuantizer"]
    SymmetricLoraQuantizer = nncf_modules["SymmetricLoraQuantizer"]

    hook_storage = get_hook_storage(model)
    compiled_count = 0
    for _, module in hook_storage.named_hooks():
        if isinstance(module, (AsymmetricLoraQuantizer, SymmetricLoraQuantizer)):
            module.quantize = torch.compile(module.quantize, mode="max-autotune", fullgraph=False)
            compiled_count += 1
    print(f"  Compiled {compiled_count} quantizers (mode='max-autotune')")


def replace_quantize_with_functional(model: nn.Module, nncf_modules: dict) -> None:
    """
    Replace custom autograd.Function-based quantize with a pure-functional equivalent
    that is fully traceable by torch.compile.

    The custom QuantizeAsymmetricTorch/QuantizeSymmetricTorch use torch.autograd.Function
    which causes graph breaks in torch.compile. This replaces them with a plain STE
    implementation using detach() that is compile-friendly.
    """
    get_hook_storage = nncf_modules["get_hook_storage"]
    AsymmetricLoraQuantizer = nncf_modules["AsymmetricLoraQuantizer"]
    SymmetricLoraQuantizer = nncf_modules["SymmetricLoraQuantizer"]

    def _functional_asym_quantize(self, x, execute_traced_op_as_identity=False):
        """Pure-functional asymmetric quantize with STE, compilable by torch.compile."""
        if execute_traced_op_as_identity:
            return x
        eps = self.eps
        input_range_safe = abs(self.input_range) + eps
        input_low = self.input_low
        input_range = input_range_safe
        levels = self.levels

        # LoRA correction
        x = (x + self.lora_B @ self.lora_A).to(x.dtype)

        # Reshape for group quantization
        input_shape = self._lspec.weight_shape
        original_shape = x.shape
        x = x.reshape(input_shape)

        # Forward quantize (clip + round)
        scale = (levels - 1) / input_range
        output = x.clamp(min=input_low, max=input_low + input_range)
        zero_point = (-input_low * scale).round()
        output = output - input_low
        output = output * scale
        output = output - zero_point
        output_q = output.round()
        output_dq = output_q / scale

        # STE: forward uses rounded values, backward passes through
        output_dq = x + (output_dq - x).detach()

        return output_dq.reshape(original_shape).to(x.dtype)

    def _functional_sym_quantize(self, x, execute_traced_op_as_identity=False):
        """Pure-functional symmetric quantize with STE, compilable by torch.compile."""
        if execute_traced_op_as_identity:
            return x
        eps = self.eps
        scale = self.scale
        scale_safe = torch.where(torch.abs(scale) < eps, eps, scale)
        level_low = self.level_low
        level_high = self.level_high
        levels = self.levels

        # LoRA correction
        x = (x + self.lora_B @ self.lora_A).to(x.dtype)

        # Reshape for group quantization
        input_shape = self._lspec.weight_shape
        original_shape = x.shape
        x = x.reshape(input_shape)

        # Symmetric FQ: range is [-scale, 7/8*scale] mapped to [level_low, level_high]
        input_low = torch.where(scale_safe > 0, -scale_safe, -scale_safe / level_low * level_high)
        input_range = torch.abs((2 + 1 / level_low) * scale_safe)

        scale_q = (levels - 1) / input_range
        output = x.clamp(min=input_low, max=input_low + input_range)
        zero_point = (-input_low * scale_q).round()
        output = output - input_low
        output = output * scale_q
        output = output - zero_point
        output_q = output.round()
        output_dq = output_q / scale_q

        # STE
        output_dq = x + (output_dq - x).detach()

        return output_dq.reshape(original_shape).to(x.dtype)

    hook_storage = get_hook_storage(model)
    replaced = 0
    for _, module in hook_storage.named_hooks():
        if isinstance(module, AsymmetricLoraQuantizer):
            import types

            module.quantize = types.MethodType(_functional_asym_quantize, module)
            replaced += 1
        elif isinstance(module, SymmetricLoraQuantizer):
            import types

            module.quantize = types.MethodType(_functional_sym_quantize, module)
            replaced += 1

    print(f"  Replaced {replaced} quantizers with pure-functional STE (compile-friendly)")


def setup_cuda_graph_training(
    model: nn.Module,
    static_input: dict[str, Tensor],
    static_target: Tensor,
    optimizer: torch.optim.Optimizer,
    grad_accumulation_steps: int,
) -> Callable[[], Tensor]:
    """
    Capture the training step as a CUDA graph for replay.

    Requirements:
      - Static input/output shapes (guaranteed by fixed seqlen + batch_size)
      - No CPU-GPU syncs or dynamic control flow in the captured region

    Returns a callable that replays the captured graph and returns the loss.
    """
    # Warmup: run a few iterations to stabilize memory allocations
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            optimizer.zero_grad()
            outputs = model(**static_input).logits
            loss = kl_div(outputs, static_target)
            (loss / grad_accumulation_steps).backward()
            optimizer.step()
    torch.cuda.current_stream().wait_stream(s)

    # Capture
    graph = torch.cuda.CUDAGraph()
    optimizer.zero_grad()
    with torch.cuda.graph(graph):
        outputs = model(**static_input).logits
        static_loss = kl_div(outputs, static_target)
        (static_loss / grad_accumulation_steps).backward()
        optimizer.step()

    def replay() -> Tensor:
        graph.replay()
        return static_loss.detach()

    print("  Captured CUDA graph for training step")
    return replay


# ─── Benchmark runner ────────────────────────────────────────────────────────


def run_training_iters(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: Tensor,
    teacher_logits: Tensor,
    num_iters: int,
    warmup_iters: int,
    grad_accumulation_steps: int = 1,
) -> list[float]:
    """Run training iterations and return per-iter GPU times in ms."""
    model_input = get_model_input(input_ids)
    times = []

    for i in range(warmup_iters + num_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        optimizer.zero_grad()
        outputs = model(**model_input).logits
        loss = kl_div(outputs, teacher_logits)
        (loss / grad_accumulation_steps).backward()
        optimizer.step()

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        if i >= warmup_iters:
            times.append((t1 - t0) * 1000)

    return times


def run_cuda_graph_iters(
    replay_fn: Callable[[], Tensor],
    num_iters: int,
    warmup_iters: int,
) -> list[float]:
    """Run CUDA graph replay iterations and return per-iter GPU times in ms."""
    times = []
    for i in range(warmup_iters + num_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        replay_fn()

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        if i >= warmup_iters:
            times.append((t1 - t0) * 1000)

    return times


# ─── Model setup ─────────────────────────────────────────────────────────────


def setup_model_and_data(args, nncf_modules: dict) -> tuple[nn.Module, Tensor, Tensor, list[dict]]:
    """Load model, compress, prepare data, return (model, input_ids, teacher_logits, param_groups)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {args.pretrained}")
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained)

    device = "cuda"
    seqlen = args.seqlen
    microbatch_size = args.microbatch_size

    # Create calibration data
    input_ids = torch.randint(0, tokenizer.vocab_size, (microbatch_size, seqlen), device=device)

    # Compress model
    Dataset = nncf_modules["Dataset"]
    compress_weights = nncf_modules["compress_weights"]
    CompressionFormat = nncf_modules["CompressionFormat"]
    CompressWeightsMode = nncf_modules["CompressWeightsMode"]
    AdvancedCompressionParameters = nncf_modules["AdvancedCompressionParameters"]
    AdvancedAWQParameters = nncf_modules["AdvancedAWQParameters"]
    get_hook_storage = nncf_modules["get_hook_storage"]
    AsymmetricLoraQuantizer = nncf_modules["AsymmetricLoraQuantizer"]
    SymmetricLoraQuantizer = nncf_modules["SymmetricLoraQuantizer"]

    ckpt_file = Path(args.output_dir) / "benchmark_ckpt.pth"
    if ckpt_file.exists():
        print(f"Loading existing checkpoint: {ckpt_file}")
        from nncf.torch.function_hook.serialization import load_from_config

        ckpt = torch.load(ckpt_file, weights_only=False, map_location="cpu")
        model = load_from_config(model, ckpt["nncf_config"])
        if "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
        hook_storage = get_hook_storage(model)
        hook_storage.load_state_dict(ckpt["nncf_state_dict"])
    else:
        print("Compressing model (INT4_ASYM, FQ_LORA)...")
        example_input = get_model_input(input_ids[:1, :128])
        dataset = Dataset([example_input])
        model = compress_weights(
            model,
            dataset=dataset,
            mode=CompressWeightsMode.INT4_ASYM,
            group_size=64,
            compression_format=CompressionFormat.FQ_LORA,
            advanced_parameters=AdvancedCompressionParameters(
                awq_params=AdvancedAWQParameters(prefer_data_aware_scaling=False),
                lora_adapter_rank=args.lora_rank,
            ),
        )
        # Save checkpoint for reuse
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        hook_storage = get_hook_storage(model)
        ckpt = {"nncf_state_dict": hook_storage.state_dict(), "nncf_config": nncf_modules["nncf"].torch.get_config(model)}
        torch.save(ckpt, ckpt_file)
        print(f"Saved checkpoint: {ckpt_file}")

    # Set trainable params
    model.requires_grad_(False)
    adapters_to_train = []
    scales_to_train = []
    hook_storage = get_hook_storage(model)
    for _, module in hook_storage.named_hooks():
        if isinstance(module, (AsymmetricLoraQuantizer, SymmetricLoraQuantizer)) and module.num_bits == 4:
            module.enable_gradients()
            params = module.get_trainable_params()
            adapters = module.get_adapters()
            adapters_to_train.extend(adapters.values())
            scales_to_train.extend(param for name, param in params.items() if name not in adapters)
    model.train()

    param_groups = [
        {"params": adapters_to_train, "lr": 1e-4},
        {"params": scales_to_train, "lr": 1e-5},
    ]

    # Pre-compute teacher logits for benchmark input
    model_input = get_model_input(input_ids)
    with torch.no_grad():
        teacher_logits = model.lm_head(
            model.model(**model_input).last_hidden_state
        ).to(dtype=torch.bfloat16, device=device)

    return model, input_ids, teacher_logits, param_groups


# ─── Main ────────────────────────────────────────────────────────────────────


def get_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark QAT training loop optimizations")
    parser.add_argument("--pretrained", type=str, default="HuggingFaceTB/SmolLM-1.7B-Instruct")
    parser.add_argument("--output_dir", type=Path, default="output")
    parser.add_argument("--seqlen", type=int, default=512, help="Sequence length for benchmark")
    parser.add_argument("--microbatch_size", type=int, default=2, help="Microbatch size")
    parser.add_argument("--lora_rank", type=int, default=256, help="LoRA adapter rank")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations (not timed)")
    parser.add_argument("--iters", type=int, default=20, help="Timed iterations")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["baseline", "compile_kernels", "functional_ste", "functional_ste_compiled", "cuda_fused_fq", "triton_fused_fq", "triton_fused_lora_fq"],
        choices=[
            "baseline",
            "compile_kernels",
            "compile_kernels_autotune",
            "functional_ste",
            "functional_ste_compiled",
            "cuda_fused_fq",
            "triton_fused_fq",
            "triton_fused_lora_fq",
            "cuda_graph",
        ],
        help="Optimization strategies to benchmark",
    )
    return parser


def main(argv) -> None:
    parser = get_argument_parser()
    args = parser.parse_args(argv)

    assert torch.cuda.is_available(), "CUDA required"
    torch.manual_seed(42)

    nncf_modules = get_nncf_imports()
    get_hook_storage = nncf_modules["get_hook_storage"]
    AsymmetricLoraQuantizer = nncf_modules["AsymmetricLoraQuantizer"]
    SymmetricLoraQuantizer = nncf_modules["SymmetricLoraQuantizer"]

    results: list[TimingResult] = []

    for strategy in args.strategies:
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy}")
        print(f"{'='*60}")
        if strategy in ["compile_kernels", "compile_kernels_autotune", "functional_ste", "functional_ste_compiled"]:
            print("Skip")
            continue

        # Fresh model for each strategy to avoid cross-contamination
        model, input_ids, teacher_logits, param_groups = setup_model_and_data(args, nncf_modules)
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

        if strategy == "baseline":
            print("  No optimization applied")

        elif strategy == "compile_kernels":
            compile_quantize_kernels(model, nncf_modules)

        elif strategy == "compile_kernels_autotune":
            compile_quantize_kernels_max_autotune(model, nncf_modules)

        elif strategy == "functional_ste":
            replace_quantize_with_functional(model, nncf_modules)

        elif strategy == "functional_ste_compiled":
            replace_quantize_with_functional(model, nncf_modules)
            # Now compile the functional quantizers — these don't call handle_torch_function
            # so they never interact with FunctionHookMode's op_calls guard.
            # Use reduce-overhead (CUDA graphs) since the functional kernels are static.
            hook_storage = get_hook_storage(model)
            compiled_count = 0
            for _, module in hook_storage.named_hooks():
                if isinstance(module, (AsymmetricLoraQuantizer, SymmetricLoraQuantizer)):
                    module.quantize = torch.compile(module.quantize, mode="reduce-overhead", fullgraph=True)
                    compiled_count += 1
            print(f"  Compiled {compiled_count} functional quantizers (reduce-overhead, fullgraph=True)")

        elif strategy == "cuda_fused_fq":
            from cuda_fq import replace_quantizers_with_cuda

            replace_quantizers_with_cuda(model, nncf_modules)

        elif strategy == "triton_fused_fq":
            from triton_fq import replace_quantizers_with_triton

            replace_quantizers_with_triton(model, nncf_modules)

        elif strategy == "triton_fused_lora_fq":
            from triton_lora import replace_quantizers_with_triton_lora

            replace_quantizers_with_triton_lora(model, nncf_modules)

        elif strategy == "cuda_graph":
            model_input = get_model_input(input_ids)
            try:
                replay_fn = setup_cuda_graph_training(
                    model, model_input, teacher_logits, optimizer, grad_accumulation_steps=1
                )
                times = run_cuda_graph_iters(replay_fn, args.iters, args.warmup)
                results.append(TimingResult(strategy, times))
                continue
            except Exception as e:
                print(f"  CUDA graph capture failed: {e}")
                print("  (Expected — NNCF hooks may cause dynamic behavior incompatible with CUDA graphs)")
                results.append(TimingResult(strategy, [float("nan")]))
                continue

        # Run benchmark
        #try:
        times = run_training_iters(
            model, optimizer, input_ids, teacher_logits,
            num_iters=args.iters, warmup_iters=args.warmup,
        )
        results.append(TimingResult(strategy, times))
        #        except Exception as e:
        #            print(f"  FAILED: {type(e).__name__}: {e}")
        #            results.append(TimingResult(strategy, [float("nan")]))
        #
        # Cleanup
        del model, optimizer
        torch.cuda.empty_cache()

    # ─── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"Model: {args.pretrained}")
    print(f"Seq length: {args.seqlen}, Microbatch: {args.microbatch_size}, LoRA rank: {args.lora_rank}")
    print(f"Iterations: {args.iters} (warmup: {args.warmup})")
    print(f"{'-'*60}")

    baseline_ms = next((r.mean_ms for r in results if r.name == "baseline"), None)
    for result in results:
        speedup = ""
        if baseline_ms and result.name != "baseline" and result.mean_ms > 0:
            speedup = f"  ({baseline_ms / result.mean_ms:.2f}x vs baseline)"
        print(f"  {result}{speedup}")


if __name__ == "__main__":
    main(sys.argv[1:])
