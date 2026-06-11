# Compress → Strip → Evaluate with vLLM

## Step 1: Compress + QAT (distillation)

```bash
python main.py \
    --pretrained Qwen/Qwen3-4B \
    --compression_format FQ_STRETCHED_LORA \
    --fq_lr 1e-3 --lora_lr 1e-3 \
    --cosine_epochs 5 --tune_bits 2 3 4 \
    --output_dir output
```

- `compress_weights()` → INT3_SYM, group_size=64, AWQ + Scale Estimation
- QAT via KL-distillation against cached teacher hiddens
- Saves checkpoints to `output/last/`

For 2-stage tuning (int2 first, then int4): `bash run_2stage_tuning.sh`

## Step 2: Strip (absorb LoRA + remove FQs)

```bash
python save_stripped.py -p Qwen/Qwen3-4B -c output/last/nncf_checkpoint_epoch5.pth -o output/last/stripped
```

`nncf.strip(model, strip_format=StripFormat.IN_PLACE)`:
1. Calls `quantizer.quantize(weight)` which does `weight + B@A` (absorbs LoRA) then fake-quantizes
2. Writes qdq'd float back to weight, deletes hook
3. Result: standard HF checkpoint, no NNCF dependency needed

## Step 3: Evaluate with vLLM

```bash
lm_eval --model vllm \
    --model_args '{"pretrained":"output/last/stripped","dtype":"auto","tensor_parallel_size":2}' \
    --tasks gsm8k --fewshot_as_multiturn --apply_chat_template --batch_size auto
```

Or: `bash run_evaluation.sh --pretrained Qwen/Qwen3-4B --output_dirs output`

## Strip formats

| Format | Effect |
|--------|--------|
| `IN_PLACE` | `weight = quantize(weight + B@A)`, removes hooks. Standard HF model. |
| `DQ` | Stores compressed int4/int8 + decompressor module. Best for deployment. |
| `NATIVE` | Replaces NNCF quantizers with PyTorch FakeQuantize. |

## Key flags for int2/int3

- `--tune_bits 2` / `--tune_bits 2 3 4` — select which bit-widths to train
- `--compression_format FQ_STRETCHED_LORA` — ParetoQ-style stretched quantization
- `--gradient_checkpointing` — reduce memory for larger models

## NNCF modules with trainable parameters during QAT

Quantizers are registered as **pre-function hooks** on Linear weight ops via `HookStorage`
(submodule at `model.__nncf_hooks`). During forward, `FunctionHookMode` intercepts each
`linear` op and runs the quantizer on the weight before the matmul.

### Quantizer classes (in `nncf/torch/quantization/layers.py`)

| Class | Format | Trainable params | Quantize formula |
|-------|--------|-----------------|-----------------|
| `StretchedSymmetricLoraQuantizer` | `FQ_STRETCHED_LORA` | `alpha` (step size), `lora_A`, `lora_B` | ParetoQ grid shifted by 0.5 (avoids wasting a level on zero; best for 2-bit) |
| `AsymmetricLoraQuantizer` | `FQ_LORA` | `input_low`, `input_range`, `lora_A`, `lora_B` | Standard asymmetric FQ with LoRA correction |
| `SymmetricLoraQuantizer` | `FQ_LORA` (sym) | `scale`, `lora_A`, `lora_B` | Standard symmetric FQ with LoRA correction |

All inherit from `LoraMixin` which provides LoRA adapters:
- `lora_A`: shape `(rank, in_features)`, initialized to **ones**
- `lora_B`: shape `(out_features, rank)`, initialized to **zeros**
- At init `B @ A = 0` → no perturbation

### Forward pass during training

```
W (frozen original weight)
  → quantizer.quantize(W):
      W' = W + B @ A              ← LoRA correction (trainable)
      output = FakeQuantize(W', scale_params)  ← scale/alpha/input_range (trainable)
  → F.linear(x, output)          ← standard matmul with fake-quantized weight
```

Gradients flow via STE (straight-through estimator) back to LoRA + scale params.
Original weight `W` is frozen.

### How `set_trainable()` selects params

```python
hook_storage = get_hook_storage(model)
for _, module in hook_storage.named_hooks():
    if isinstance(module, (AsymmetricLoraQuantizer, SymmetricLoraQuantizer,
                           StretchedSymmetricLoraQuantizer)):
        if module.num_bits in tune_bits:  # e.g. [2], [4], or [2, 3, 4]
            module.lora_A.requires_grad = True   # → adapters_to_train (lr=lora_lr)
            module.lora_B.requires_grad = True
            module.scale/alpha/input_range.requires_grad = True  # → scales_to_train (lr=fq_lr)
```

Two optimizer param groups: `{"params": adapters, "lr": lora_lr}` and `{"params": scales, "lr": fq_lr}`.

### `set_use_autograd_quantize` toggle

| `False` (default) | Hand-written `torch.autograd.Function` backward (custom STE) |
|---|---|
| `True` | PyTorch autograd computes gradients automatically |

Quantize functions: `asymmetric_quantize_lora`, `symmetric_quantize_lora`,
`stretched_symmetric_quantize_lora` in `nncf/torch/quantization/quantize_functions.py`.

## Known issue: `torch.compile` on individual quantizers fails

Compiling only the quantizer's `quantize()` method (rather than the full model) fails with:

```
AssertionError: Guard failed on the same frame it was created.
Guard fail reason: len(___get_torch_function_mode_stack_at(0).op_calls) == 31
```

**Root cause**: NNCF's `FunctionHookMode` (a `TorchFunctionMode`) sits on the mode stack
during the entire model forward. Its `__torch_function__` intercepts every tensor operation
and mutates `self.op_calls[op_name] += 1` to track which op is being executed.

When `torch.compile` traces a quantizer's `quantize()` method, it sees the mode on the stack
and creates a guard on `len(op_calls)`. On the next call, `op_calls` has grown (other model
ops ran between quantizer invocations), so the guard immediately fails — even on the same
frame it was created.

**Why removing `handle_torch_function` from lora quantize functions is not sufficient**:
The `handle_torch_function` wrapper was only one layer. Even without it, all tensor
operations *inside* the quantize function (`+`, `@`, `abs()`, `torch.where()`, etc.) still
pass through `FunctionHookMode.__torch_function__` because the mode remains on the stack.
Each such op mutates `op_calls`, making the mode's state non-deterministic from the
compiler's perspective.

**Possible fixes**:
1. Set `self.enabled = False` in `FunctionHookMode` during hook execution so internal
   quantizer ops don't mutate `op_calls` and torch.compile doesn't see changing state.
2. `torch.compile(model)` on the full model instead of per-quantizer — lets dynamo handle
   the mode holistically.
3. `torch._dynamo.allow_in_graph(asymmetric_quantize_lora)` — makes the function opaque to
   the compiler (avoids the guard but loses inner kernel fusion).
4. Pop the mode from the stack before calling the compiled quantize, push it back after.

## Known issue: `device_map="auto"` + Python <3.14

When using `device_map="auto"`, HuggingFace accelerate replaces `model.forward` with a
`functools.partial` and applies `functools.update_wrapper` to it. This sets `__wrapped__`
on the partial, causing `inspect.signature()` to follow the `__wrapped__` chain and return
the raw class method signature (including `self`) instead of stripping pre-filled args.

Result: NNCF's graph building fails with `TypeError: missing a required argument: 'self'`.

**Root cause**: CPython bug — `functools.update_wrapper` breaks signature introspection of
`functools.partial` objects.
- [CPython #90917](https://github.com/python/cpython/issues/90917) — open, "functools.update_wrapper breaks the signature of functools.partial objects"
- [CPython #121027](https://github.com/python/cpython/issues/121027) — fixed in Python 3.14, `partial` becomes a method descriptor

**Fix in NNCF**: `ForwardWithHooks.__signature__` strips `self` if it appears as the first param.
Proper fix: use Python ≥3.14.

## Why NNCF has both CUDA extension and PyTorch reference FQ implementations

NNCF provides two backends for fake quantize forward/backward:

| Backend | Class | Used by | Kernel location |
|---------|-------|---------|-----------------|
| CUDA extension | `QuantizedFunctionsCUDA` (`Quantize_forward`/`Quantize_backward`) | `QuantizeAsymmetric`, `QuantizeSymmetric` (non-LoRA) | `nncf/torch/extensions/src/quantization/cuda/functions_cuda_impl.cu` |
| PyTorch reference | `ReferenceQuantizedFunctions` (`RQ.Quantize_forward`/`RQ.Quantize_backward`) | `QuantizeAsymmetricTorch`, `QuantizeSymmetricTorch` (LoRA + group quant) | `nncf/torch/quantization/reference.py` |

### Root cause: the CUDA kernel doesn't support group quantization broadcasting

The CUDA kernel's `get_scale_type()` only supports three scale layouts:

1. **SINGLE_SCALE** — one scale for the entire tensor
2. **PER_WEIGHT_CHANNEL** — `input_low` is 1D with `size(0) == input.size(0)` (one scale per row)
3. **PER_ACTIVATION_CHANNEL** — `input_low` is 1D with `size(1) == input.size(1)` (one scale per channel)

Group quantization requires a **3D broadcasting** pattern:
- Weight reshaped to `(out_features, num_groups_per_row, group_size)` = e.g. `(2048, 128, 64)`
- `input_low` has shape `(2048, 128, 1)` — one scale per group, broadcasting across group_size

The CUDA kernel cannot handle this. The PyTorch reference uses standard PyTorch ops
with arbitrary broadcasting, so it works for any shape.

### Why forcing 262144 "channels" into the CUDA kernel causes OOM

If you flatten `input_low` to `(262144,)` and reshape weight to `(262144, 64)`, the CUDA
kernel treats it as PER_WEIGHT_CHANNEL with 262144 "output channels" (designed for ~2048).

The backward kernel `q_scale_per_weight_channel_cuda_backward` allocates temporary
reduction buffers per channel:
```cpp
dim3 grid_size = get_2d_grid_size_for_per_channel(scale_count);  // (262144, grid_y)
auto dev_tmp_range = at::zeros({262144, grid_y}, float32);  // reduction workspace
auto dev_tmp_low   = at::zeros({262144, grid_y}, float32);
auto dev_last_block_counter_range = at::zeros({262144, 1}, int32);
auto dev_last_block_counter_low   = at::zeros({262144, 1}, int32);
```

Per backward call this is ~4-8MB. But the real issue is the kernel launches **262144 thread
blocks** (each with 1024 threads) to process just 64 elements per block — extremely wasteful
for the GPU scheduler and memory subsystem. Combined with model weights, optimizer states,
and activations already near the 24GB limit, the overhead pushes it over.

### Why Triton doesn't have this problem

The Triton kernel was designed specifically for group quantization:
- Forward: one program per `BLOCK_SIZE` elements, group index computed on the fly
- Backward: uses `tl.atomic_add` for per-group gradient reduction — **zero temporary buffers**

### Summary

| Approach | Group quant support | Temp buffers in backward | Memory overhead |
|----------|-------------------|-------------------------|----------------|
| CUDA extension (forced flat) | Hacky (262144 "channels") | ~4-8MB per layer call | OOM on 24GB |
| PyTorch reference (baseline) | Native (3D broadcasting) | None (pure Python ops) | Higher compute time |
| Triton kernel | Native (group_idx = offset // group_size) | None (atomic_add) | Optimal |
