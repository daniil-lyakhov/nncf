# CUDA Graphs vs torch.compile on NVIDIA GPUs

## Summary

They are **complementary**, not competing — they optimize different layers of the stack.

## Comparison

| | CUDA Graphs | torch.compile |
|---|---|---|
| **Optimizes** | CPU→GPU dispatch overhead | GPU-side compute (kernel fusion, memory) |
| **Mechanism** | Records & replays fixed GPU op sequence | Traces Python → fuses ops → generates optimized kernels |
| **Best gain** | Many small kernels with high launch overhead | Multiple fusable ops (matmul+bias+relu → 1 kernel) |
| **Constraint** | Static shapes, no control flow | Recompiles on new shapes, graph breaks on unsupported ops |

## Context: Isolated Kernel Benchmark

- **CUDA graphs win** — single kernel, nothing to fuse, just eliminates dispatch overhead
- **torch.compile can't help** — it cannot optimize inside a hand-written CUDA kernel

## Context: Full QAT Training Loop

- **torch.compile wins bigger** — fuses the dozens of ops surrounding quantization (linear+quant+activation+norm → fewer kernels)
- **CUDA graphs add on top** — replays the entire compiled step without CPU dispatch

## Ideal Stack (Maximum Performance)

```
torch.compile (fuses ops, generates efficient kernels)
    ↓
CUDA graphs (replays the compiled graph with zero CPU overhead)
```

PyTorch supports this directly: compile the model, then wrap the training step in `torch.cuda.graph`.

## Implications for NNCF

- The current pybind11 kernel blocks torch.compile from optimizing the *surrounding* model ops (graph break)
- Switching to `TORCH_LIBRARY` would let torch.compile fuse everything else while keeping the fast custom kernel as a node
- Then CUDA graphs on top would eliminate remaining dispatch overhead
- The real-world QAT speedup from enabling torch.compile (via `TORCH_LIBRARY`) would likely be **much larger** than the kernel-level differences seen in the isolated benchmark, because it unlocks fusion of the entire model graph

## CUDA Graphs: Integration Complexity in Practice

CUDA graphs are **not a drop-in optimization**. Even after fixing all blockers in NNCF's quantization layers, enabling CUDA graphs requires deep integration into the training/inference pipeline:

### Blockers we had to fix in NNCF

1. **`.item()` calls in forward path** — Any CPU↔GPU sync breaks graph capture. NNCF's `BaseQuantizer` used `.item()` on `enabled`, `level_low`, `level_high`, `num_bits`, and `signed` tensors. Fix: cache Python values and sync them via setters / `_load_state_dict_post_hook`.
2. **Control flow depending on tensor values** — `if self.enabled` read a GPU tensor to decide branching. Graph capture needs all control flow to be static.

### Pipeline constraints (even after the layer fix)

| Requirement | Impact on user code |
|---|---|
| **Static tensor shapes** | Must use `drop_last=True` on DataLoader, or pad all batches |
| **No data-dependent branching** | Cannot conditionally skip quantization based on runtime values |
| **Separate forward/backward graphs** | Training requires `torch.cuda.make_graphed_callables()` — cannot naively wrap the training step in `torch.cuda.graph()` because autograd uses multiple CUDA streams |
| **One-time graph capture** | `make_graphed_callables` must be called once before the training loop, not per-epoch |
| **Validation graph is separate** | Inference graph must be captured independently (model in eval mode, `torch.no_grad()`) |
| **Last-batch fallback** | Validation needs eager fallback when the final batch is smaller than the captured shape |
| **State dict loading** | After loading a checkpoint, cached scalar values must be re-synced (`_load_state_dict_post_hook`) |
| **DDP broadcast** | After parameter broadcast, cached values must be updated |

### Comparison: effort vs. torch.compile

| | CUDA Graphs | torch.compile |
|---|---|---|
| **Integration effort** | High — requires restructuring data loading, training loop, validation loop, checkpoint logic | Low — `model = torch.compile(model)` (if ops are supported) |
| **Maintenance burden** | Every new dynamic behavior (new quantizer param, conditional logic) can silently break capture | Automatic — compiler handles dynamism via guards + recompilation |
| **Debugging** | Cryptic CUDA errors at capture time (`cudaStreamCaptureImplicit`, version mismatch) | Clear graph-break reports with `TORCH_LOGS="graph_breaks"` |
| **Composability** | Fragile — must manually manage multiple graph objects (train graph, eval graph, different batch sizes) | Composable — works with DDP, FSDP, AMP, gradient checkpointing out of the box |

### Bottom line

CUDA graphs deliver measurable speedup (1.7x training, 1.3x validation in our QAT experiment), but they are an **infrastructure-level optimization** that must be woven throughout the pipeline. They are not suitable as a user-facing "enable this flag" feature without significant guardrails. `torch.compile` is the more ergonomic path forward — once NNCF migrates custom ops to `TORCH_LIBRARY`, users get fusion + dispatch elimination with a single line change.

## Why torch.compile Is Incompatible With NNCF Hooks

NNCF's quantization architecture has **three independent mechanisms** that each break `torch.compile`:

### 1. `TorchFunctionMode` — Python-level op interception

NNCF wraps the quantized model's forward in `FunctionHookMode(TorchFunctionMode)` ([hook_executor_mode.py](src/nncf/torch/function_hook/hook_executor_mode.py)). This intercepts **every** `torch.*` operation at the Python level:

```python
class FunctionHookMode(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        op_name = self.get_next_op_call_name(fn_name)
        args, kwargs = self.execute_pre_hooks(args, kwargs, op_meta)  # ← quantizer runs here
        output = func(*args, **kwargs)
        output = self.execute_post_hooks(output, op_meta)
        return output
```

**Why this breaks torch.compile:**
- Dynamo traces Python bytecode to build an FX graph. When it encounters a `TorchFunctionMode.__torch_function__` override, it cannot predict what the hook will do (it's arbitrary Python code with side effects: module call stacks, op counters, parameter caches).
- Dynamo inserts a **guard** asserting that no `TorchFunctionMode` is active. If the mode is active, the guard fails → graph break → recompile → guard fails again → infinite recompilation or assertion error.
- The `ForwardWithHooks.__call__` wrapper enters/exits `FunctionHookMode` as a context manager on every forward call, which is opaque to the tracer.

### 2. `has_torch_function_unary` / `handle_torch_function` in quantize functions

The quantization entry points (`symmetric_quantize`, `asymmetric_quantize`) check if the input tensor has a `__torch_function__` override and dispatch through it:

```python
def symmetric_quantize(input_, levels, level_low, level_high, scale, eps, skip=False):
    if has_torch_function_unary(input_):
        return handle_torch_function(symmetric_quantize, (input_,), ...)
    ...
    return QuantizeSymmetric.apply(input_, scale_safe, level_low, level_high, levels)
```

**Why this breaks torch.compile:**
- `has_torch_function_unary` is a runtime check on the tensor's type. Dynamo must specialize on whether this branch is taken, creating a type guard on every input tensor.
- `handle_torch_function` dispatches to the active `TorchFunctionMode`, which is the same opaque Python code from point 1.
- This creates a data-dependent branch that Dynamo cannot eliminate statically.

### 3. pybind11 extensions as opaque function calls

The actual quantization kernels are loaded via `torch.utils.cpp_extension.load()` — a pybind11 module:

```python
# QuantizedFunctionsCUDA loaded at runtime:
output = QuantizedFunctionsCUDA.get("Quantize_forward")(input_, input_low, input_range, levels)
```

**Why this breaks torch.compile:**
- The pybind11 function is opaque to TorchDynamo. It's not registered in PyTorch's dispatcher, so Inductor cannot lower it to a Triton kernel or even represent it in the FX graph.
- Each call to `QuantizedFunctionsCUDA.get("Quantize_forward")` is a Python-level attribute lookup + function call that Dynamo cannot trace through → **graph break**.
- Without a `FakeTensor` implementation (meta kernel), the compiler cannot propagate shapes/dtypes through the custom op during tracing.
- The `torch.autograd.Function` wrappers (`QuantizeSymmetric`, `QuantizeAsymmetric`) use `ctx.save_for_backward` with tensors produced by the opaque extension, making the backward pass equally untraceable.

### 4. Dynamic module call stack tracking

`FunctionHookMode` maintains a runtime call stack (`module_call_stack`) and per-op counters (`op_calls`) to generate unique hook names:

```python
def get_next_op_call_name(self, fn_name):
    op_name = generate_normalized_op_name(module_name, fn_name)
    call_id = self.op_calls[op_name]
    self.op_calls[op_name] += 1  # ← mutating Python dict on every op
    return generate_normalized_op_name(module_name, fn_name, call_id)
```

**Why this breaks torch.compile:**
- Dynamo cannot trace through mutations to Python dictionaries that determine control flow (which hooks to execute depends on `op_name` which depends on the counter).
- The `_call_impl` monkey-patching (`module._call_impl = types.MethodType(...)`) on `__enter__` is invisible to the FX tracer — it dynamically replaces method implementations at runtime.

### Summary of incompatibility layers

| NNCF Mechanism | torch.compile Failure Mode |
|---|---|
| `TorchFunctionMode` context manager | Guard assertion failure (mode active during trace) |
| `has_torch_function_unary` check | Data-dependent branch → graph break |
| pybind11 extension calls | Opaque call → graph break (no FX representation) |
| `torch.autograd.Function.apply` with opaque ops | Backward not traceable |
| Runtime op counter / call stack | Python side effects → untraceable |
| `_call_impl` monkey-patching | Dynamic method override → guard failure |

### What would fix this

| Fix | Effort | Benefit |
|---|---|---|
| Register ops via `TORCH_LIBRARY` + provide `FakeTensor` meta | Medium | Eliminates graph breaks from opaque calls; enables shape propagation |
| Replace `TorchFunctionMode` with `torch.library` pre/post hooks | High | Removes the mode-based interception entirely; Dynamo can trace through |
| Express quantization as pure PyTorch ops (torchao style) | High | Full Inductor fusion; no custom extensions needed |
| Use `torch.compiler.allow_in_graph` on autograd Functions | Low | Stops graph breaks but disables fusion through the op (barrier) |

The minimal path: `TORCH_LIBRARY` registration + `allow_in_graph` gives compatibility without rewriting the hook system. Full performance requires migrating away from `TorchFunctionMode` to a compiler-friendly hook mechanism.

## Can Quantization Be Fused With Surrounding Ops?

### With `TORCH_LIBRARY` (custom kernel as opaque node): No

Inductor can't fuse *into* or *out of* a custom op. It's a fusion barrier. The benefit is only that surrounding ops fuse with *each other* (no graph break).

```
[matmul + bias]  →  [custom_fake_quant]  →  [relu + dropout]
   ↑ fused              ↑ standalone             ↑ fused
```

### With pure PyTorch ops (reference/torchao style) under torch.compile: Yes

Quantization expressed as pointwise ops is fully fusable by inductor:

```python
output = clamp(input, low, low + range)
output = (output - low) * scale
output = round(output) / scale + low
```

Inductor can fuse **across op boundaries** (epilogue fusion):

```
[matmul → fake_quant → relu]  →  one fused kernel
```

This avoids writing matmul output to global memory and reading it back just for quantization — a significant memory bandwidth saving on large tensors.

### Tradeoff Summary

| Approach | Kernel quality | Cross-op fusion | Net effect |
|----------|---------------|-----------------|------------|
| Hand-written CUDA (current) | Optimal within quant | Impossible (opaque) | Fast kernel, but memory round-trips between ops |
| Pure PyTorch + torch.compile | Compiler-generated | Yes (matmul→quant→relu fused) | Slightly worse kernel, but saves memory bandwidth |

### When does fusion win?

- **Large tensors** (memory-bandwidth bound): fusion saves a full read+write of the activation tensor between matmul and quant — this is often larger than any kernel-level advantage
- **Small tensors** (launch-overhead bound): hand-written kernel + CUDA graphs wins

For real QAT on modern models (large activations), pure PyTorch quantization under `torch.compile` with epilogue fusion could be **faster end-to-end** than the hand-written kernel — not because the quant kernel is better, but because it eliminates a memory round-trip. This is the bet torchao is making.

## Why NNCF's Hook Dispatch Logic Breaks torch.compile (But Simple Modes Don't)

### POC Proof: Simple TorchFunctionMode + torch.compile Works

A proof-of-concept (`benchmarks/poc_function_mode_compile.py`) demonstrates that `TorchFunctionMode` **can** work with `torch.compile` when the hook dispatch logic is statically traceable:

```python
class HookFunctionMode(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in self.pre_hooks:          # ← static dict lookup by function reference
            args, kwargs = self.pre_hooks[func](args, kwargs)
        result = func(*args, **kwargs)
        if func in self.post_hooks:
            result = self.post_hooks[func](result)
        return result
```

This compiles successfully with `fullgraph=False` because Dynamo can trace the dict lookup — the key is a constant function reference, not a dynamically-constructed string.

### Why NNCF's FunctionHookMode Cannot Be Compiled

NNCF's `FunctionHookMode` uses **positional identity** to dispatch hooks. On every op call, it:

```python
# 1. Build op name from mutable runtime state
module_name = self.get_current_relative_name()  # traverses module_call_stack list
op_name = f"{module_name}/{fn_name}"            # string formatting
call_id = self.op_calls[op_name]                # dict read
self.op_calls[op_name] += 1                     # dict mutation ← GRAPH BREAK

# 2. Generate hook key from dynamic string
hook_key = f"{op_name}__{port_id}"              # dynamic string construction

# 3. Look up hook by dynamic key in ModuleDict
if hook_key in storage_dict:                    # data-dependent branch ← GRAPH BREAK
    hooks_dict = storage_dict[hook_key]
    for hook in hooks_dict.values():            # iteration over dynamic collection
        value = hook(value)
```

### The Critical Difference

| POC (compiles) | NNCF (breaks) |
|---|---|
| Hook lookup by **function reference** (static key) | Hook lookup by **dynamically-built string** (module path + call counter) |
| No mutable state between ops | `op_calls` counter incremented on every op |
| No call stack tracking | `module_call_stack` push/pop on every submodule |
| Direct dict `{func: hook_fn}` | `nn.ModuleDict` keyed by constructed `f"{op_name}__{port_id}"` |
| Hook is a single callable | Iterates over a nested `ModuleDict` of hooks |

### What This Means for the Fix

To make NNCF compile-compatible **while keeping the mode active**, you'd need to replace the dynamic positional dispatch with static dispatch — hooks keyed directly by the function/op reference, not by a string built from runtime execution order. This is a fundamental architecture change to how NNCF identifies insertion points.

### The Viable Architecture: Materialize Then Compile

Since making the mode itself compile-friendly requires a fundamental redesign of NNCF's hook identity system, the practical path is:

1. **Register quantize ops via `torch.library.custom_op`** — makes them compile-visible
2. **Use the mode ONLY during calibration/QAT** (always eager, no compile needed)
3. **Materialize/finalize** — inline `torch.ops.nncf.symmetric_quantize(...)` directly into the model's forward (no mode, no hook_storage, no ModuleDict lookups)
4. **torch.compile the finalized model** — all ops are in TORCH_LIBRARY, Dynamo traces them natively

```
# CALIBRATION (eager, mode active — same as today)
with FunctionHookMode(model, hook_storage):
    model(calibration_data)

# FINALIZE (materialize quantizers inline)
finalized_model = nncf.finalize(model)
# forward now directly contains:
#   x = torch.ops.nncf.symmetric_quantize(x, scale, zp, bits)
#   x = F.conv2d(x, weight, bias)

# COMPILE (plain model, all ops in TORCH_LIBRARY)
compiled = torch.compile(finalized_model, fullgraph=True)  # Works!
```

### torch.library APIs Required (PyTorch 2.4+, July 2024)

| API | Purpose |
|-----|---------|
| `@torch.library.custom_op("nncf::symmetric_quantize", mutates_args=())` | Register compile-visible custom op |
| `@torch.library.register_fake("nncf::symmetric_quantize")` | FakeTensor kernel for shape/dtype propagation during tracing |
| `torch.library.register_autograd("nncf::symmetric_quantize", backward_fn, setup_context=...)` | STE backward for QAT under compile |
| `@torch.library.register_kernel("nncf::symmetric_quantize", "cuda")` | Wrap existing pybind11 CUDA extension |

Documentation:
- [Custom Ops Landing Page](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html)
- [Python Custom Ops Tutorial](https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html)
- [torch.library API Reference](https://docs.pytorch.org/docs/stable/library.html)
