from __future__ import annotations

import subprocess
import sys


def test_interpol_gradcheck_smoke() -> None:
    code = r"""
import torch
from torch.autograd import gradcheck
from interpol import grid_pull, grid_grad, add_identity_grid_

torch.set_num_threads(1)
dtype = torch.double
shape = (3,)
vol = torch.randn((2, 1) + shape, dtype=dtype)
grid = torch.randn([2, *shape, len(shape)], dtype=dtype)
grid = add_identity_grid_(grid)
vol.requires_grad = True
grid.requires_grad = True

kwargs = dict(rtol=1.0, raise_exception=True)
if 'check_undefined_grad' in gradcheck.__code__.co_varnames:
    kwargs['check_undefined_grad'] = False
if 'nondet_tol' in gradcheck.__code__.co_varnames:
    kwargs['nondet_tol'] = 1e-3

assert gradcheck(grid_grad, (vol, grid, 1, 0, True), **kwargs)
assert gradcheck(grid_pull, (vol, grid, 1, 0, True), **kwargs)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(result.stderr.strip() or result.stdout.strip())
