from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Optional, Tuple, List, Set

import numpy as np

from config import EP_SORT_MODE

# Rotation indices map to axis permutations for (x, y, z).
ROTATION_PERMUTATIONS: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
)


@dataclass(frozen=True)
class Box:
    box_id: int
    dims: Tuple[int, int, int]
    rotation: int = 0
    position: Optional[Tuple[int, int, int]] = None

    def rotated_dims(self) -> Tuple[int, int, int]:
        perm = ROTATION_PERMUTATIONS[self.rotation % 6]
        return (self.dims[perm[0]], self.dims[perm[1]], self.dims[perm[2]])

    def with_pose(self, position: Tuple[int, int, int], rotation: int) -> "Box":
        return replace(self, position=position, rotation=rotation)

    def aabb(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.position is None:
            raise ValueError("Box has no position")
        pos = np.array(self.position, dtype=int)
        dims = np.array(self.rotated_dims(), dtype=int)
        return pos, pos + dims


@dataclass
class Pallet:
    length_mm: int
    width_mm: int
    height_mm: int
    boxes: List[Box] = field(default_factory=list)
    eps: Set[Tuple[int, int, int]] = field(default_factory=lambda: {(0, 0, 0)})

    def add_box(self, box: Box, new_eps: Iterable[Tuple[int, int, int]]) -> None:
        self.boxes.append(box)
        self._prune_eps_inside_box(box)
        self.eps.update(new_eps)

    def sorted_eps(self) -> List[Tuple[int, int, int]]:
        if EP_SORT_MODE == "bottom_up":
            key = lambda p: (p[2], p[1], p[0])
        elif EP_SORT_MODE == "back_to_front":
            key = lambda p: (p[1], p[2], p[0])
        else:
            raise ValueError(f"Unknown EP_SORT_MODE: {EP_SORT_MODE}")
        return sorted(self.eps, key=key)

    def _prune_eps_inside_box(self, box: Box) -> None:
        min_corner, max_corner = box.aabb()
        to_remove: List[Tuple[int, int, int]] = []
        for ep in self.eps:
            ep_arr = np.array(ep, dtype=int)
            if np.all(ep_arr >= min_corner) and np.all(ep_arr < max_corner):
                to_remove.append(ep)
        for ep in to_remove:
            self.eps.remove(ep)
