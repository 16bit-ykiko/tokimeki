"""Typed wrappers around third-party models; the only package that touches them.

Importing it caps the CPU thread pools of the numeric libraries before any of them loads:
the CPU is busy with other work and the heavy lifting belongs on the GPU.
"""

import os

CPU_THREADS = 2

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, str(CPU_THREADS))
