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
