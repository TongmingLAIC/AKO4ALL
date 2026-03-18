"""
TVM FFI Bindings Template for C++ Kernels.

This file provides Python bindings for your C++ kernel using TVM FFI.
The entry point function name should match the `entry_point` setting in config.toml.

Getting started:
  1. Read docs/reference.py to understand the computation semantics
  2. Read docs/definition.json for input/output shapes and dtypes
  3. Implement the kernel logic in kernel.cpp
  4. Update this file to call your C++ implementation
  5. Run: bash scripts/bench.sh
"""

import ctypes
from tvm.ffi import register_func


@register_func("flashinfer.kernel")
def kernel():
    """
    Python binding for your C++ kernel.

    TODO: Implement the binding according to the operator definition.
    This function should:
    1. Accept the inputs as specified in docs/definition.json
    2. Call your C++ implementation
    3. Return outputs as specified in docs/definition.json

    See docs/reference.py for the PyTorch reference implementation.
    """
    pass
