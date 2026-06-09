"""
Reproducer: inspect.signature returns 'self' for functools.partial + update_wrapper on Python <3.14.

This mimics what HuggingFace accelerate does in hooks.py:183:
    module.forward = functools.update_wrapper(functools.partial(new_forward, module), old_forward)

The bug triggers when the original method is decorated with @functools.wraps (as HuggingFace does),
causing __wrapped__ to point to the raw unbound function. inspect.signature follows __wrapped__
all the way to the unbound function, which still has 'self' in its signature.

References:
  - https://github.com/python/cpython/issues/90917
  - https://github.com/python/cpython/issues/121027
"""
import functools
import inspect
import sys


def hf_decorator(fn):
    """Simulates HuggingFace @add_start_docstrings_to_model_forward or similar."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


class MyModule:
    @hf_decorator
    def forward(self, input_ids=None, attention_mask=None):
        return input_ids


def new_forward(module, *args, **kwargs):
    """Simulates accelerate's hook wrapper."""
    return module._old_forward(*args, **kwargs)


# Simulate what accelerate does at hooks.py:183
module = MyModule()
module._old_forward = module.forward
old_forward = module.forward
module.forward = functools.update_wrapper(functools.partial(new_forward, module), old_forward)

# This is what NNCF does to bind inputs
sig = inspect.signature(module.forward)
params = list(sig.parameters.keys())

print(f"Python {sys.version}")
print(f"Signature params: {params}")

if params[0] == "self":
    print("\nBUG: 'self' appears in signature — bind(**kwargs) will fail")
    try:
        sig.bind(input_ids=1, attention_mask=1)
    except TypeError as e:
        print(f"TypeError: {e}")
        sys.exit(1)
else:
    print("\nOK: signature correctly omits 'self'")
    sig.bind(input_ids=1, attention_mask=1)
