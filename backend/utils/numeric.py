"""Small numeric helpers shared by embedding / reranking code."""

from __future__ import annotations

import math

import numpy as np


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation; 1-D input is treated as a single row. Returns float32."""
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def sigmoid(x: float) -> float:
    """Numerically stable logistic function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)
