from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from config import (
    BOX_MAX_DIM_MM,
    BOX_MIN_DIM_MM,
    DISCRETIZE_DIMS,
    DISCRETIZE_STEP_MM,
    PALLET_LENGTH_MM,
    PALLET_WIDTH_MM,
    RANDOM_SEED,
)
from models import Box


def _random_dim(
    rng: np.random.Generator,
    min_mm: int,
    max_mm: int,
    step_mm: int,
    discretize: bool,
) -> int:
    if discretize:
        min_k = math.ceil(min_mm / step_mm)
        max_k = max_mm // step_mm
        if max_k < min_k:
            return int(min_mm)
        return int(rng.integers(min_k, max_k + 1) * step_mm)
    return int(rng.integers(min_mm, max_mm + 1))


def generate_boxes(count: int, seed: Optional[int] = RANDOM_SEED) -> List[Box]:
    rng = np.random.default_rng(seed)
    max_x = min(BOX_MAX_DIM_MM, PALLET_LENGTH_MM)
    max_y = min(BOX_MAX_DIM_MM, PALLET_WIDTH_MM)
    max_z = BOX_MAX_DIM_MM

    boxes: List[Box] = []
    for i in range(1, count + 1):
        dx = _random_dim(rng, BOX_MIN_DIM_MM, max_x, DISCRETIZE_STEP_MM, DISCRETIZE_DIMS)
        dy = _random_dim(rng, BOX_MIN_DIM_MM, max_y, DISCRETIZE_STEP_MM, DISCRETIZE_DIMS)
        dz = _random_dim(rng, BOX_MIN_DIM_MM, max_z, DISCRETIZE_STEP_MM, DISCRETIZE_DIMS)
        boxes.append(Box(box_id=i, dims=(dx, dy, dz)))
    return boxes
