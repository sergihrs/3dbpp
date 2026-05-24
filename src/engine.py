from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import Point, box as shapely_box
from shapely.ops import unary_union

import config
from models import Box, Pallet


def generate_eps_for_box(box: Box) -> List[Tuple[int, int, int]]:
    if box.position is None:
        raise ValueError("Box has no position")
    x, y, z = box.position
    dx, dy, dz = box.rotated_dims()
    return [
        (x + dx, y, z),
        (x, y + dy, z),
        (x, y, z + dz),
    ]


def aabb_intersect(
    min_a: np.ndarray, max_a: np.ndarray, min_b: np.ndarray, max_b: np.ndarray
) -> bool:
    # Strict overlap on all axes; touching faces is allowed.
    return bool(np.all(max_a > min_b) and np.all(min_a < max_b))


def collides(candidate: Box, placed_boxes: Iterable[Box]) -> bool:
    cmin, cmax = candidate.aabb()
    for placed in placed_boxes:
        pmin, pmax = placed.aabb()
        if aabb_intersect(cmin, cmax, pmin, pmax):
            return True
    return False


def is_top_down_clear(candidate: Box, placed_boxes: Iterable[Box]) -> bool:
    if candidate.position is None:
        return False

    placed_list = [box for box in placed_boxes if box.position is not None]
    if not placed_list:
        return True

    x0, y0, z0 = candidate.position
    dx, dy, _ = candidate.rotated_dims()
    margin = config.GRIPPER_MARGIN_MM if config.USE_GRIPPER_MARGIN else 0
    x0_m = x0 - margin
    x1_m = x0 + dx + margin
    y0_m = y0 - margin
    y1_m = y0 + dy + margin

    positions = np.array([box.position for box in placed_list], dtype=int)
    dims = np.array([box.rotated_dims() for box in placed_list], dtype=int)
    px0 = positions[:, 0]
    py0 = positions[:, 1]
    pz0 = positions[:, 2]
    px1 = px0 + dims[:, 0]
    py1 = py0 + dims[:, 1]
    pz1 = pz0 + dims[:, 2]

    overlap_xy = (px0 < x1_m) & (px1 > x0_m) & (py0 < y1_m) & (py1 > y0_m)
    blockers = overlap_xy & (pz1 > z0)
    return not bool(np.any(blockers))


def _support_area_and_cog_ok(candidate: Box, placed_boxes: Iterable[Box]) -> Tuple[float, bool]:
    if candidate.position is None:
        return 0.0, False
    x, y, z = candidate.position
    dx, dy, _ = candidate.rotated_dims()
    if z == 0:
        return float(dx * dy), True

    footprint = shapely_box(x, y, x + dx, y + dy)
    supports: List = []
    for placed in placed_boxes:
        if placed.position is None:
            continue
        px, py, pz = placed.position
        pdx, pdy, pdz = placed.rotated_dims()
        if pz + pdz != z:
            continue
        support_poly = shapely_box(px, py, px + pdx, py + pdy).intersection(footprint)
        if not support_poly.is_empty and support_poly.area > 0:
            supports.append(support_poly)

    if not supports:
        return 0.0, False

    support_union = unary_union(supports)
    center = Point(x + dx / 2.0, y + dy / 2.0)
    return float(support_union.area), bool(support_union.contains(center) or support_union.touches(center))


def has_sufficient_support(candidate: Box, placed_boxes: Iterable[Box], min_ratio: float = 0.8) -> bool:
    if candidate.position is None:
        return False
    dx, dy, _ = candidate.rotated_dims()
    base_area = float(dx * dy)
    if base_area <= 0:
        return False

    support_area, cog_ok = _support_area_and_cog_ok(candidate, placed_boxes)
    if candidate.position[2] == 0:
        return True
    return support_area >= min_ratio * base_area and cog_ok


def is_within_bounds(candidate: Box, pallet: Pallet) -> bool:
    if candidate.position is None:
        return False
    x, y, z = candidate.position
    dx, dy, dz = candidate.rotated_dims()
    if x < 0 or y < 0 or z < 0:
        return False
    if x + dx > pallet.length_mm:
        return False
    if y + dy > pallet.width_mm:
        return False
    if z + dz > pallet.height_mm:
        return False
    return True


def try_place_box(
    box: Box, pallet: Pallet, rotation_override: Optional[int] = None
) -> Tuple[bool, Box]:
    rotations = range(6) if rotation_override is None else [rotation_override % 6]
    for ep in pallet.sorted_eps():
        for rotation in rotations:
            candidate = box.with_pose(ep, rotation)
            if not is_within_bounds(candidate, pallet):
                continue
            if not is_top_down_clear(candidate, pallet.boxes):
                continue
            if collides(candidate, pallet.boxes):
                continue
            if not has_sufficient_support(
                candidate,
                pallet.boxes,
                min_ratio=config.SUPPORT_MIN_RATIO,
            ):
                continue
            new_eps = generate_eps_for_box(candidate)
            pallet.add_box(candidate, new_eps)
            return True, candidate
    return False, box


def pack_boxes(boxes: Iterable[Box], pallet: Pallet) -> Tuple[List[Box], List[Box]]:
    placed: List[Box] = []
    skipped: List[Box] = []
    for box in boxes:
        ok, placed_box = try_place_box(box, pallet)
        if ok:
            placed.append(placed_box)
        else:
            skipped.append(box)
    return placed, skipped


def pack_boxes_with_rotations(
    boxes: Iterable[Box], pallet: Pallet, rotations: Sequence[int]
) -> Tuple[List[Box], List[Box]]:
    boxes_list = list(boxes)
    if len(boxes_list) != len(rotations):
        raise ValueError("Rotations length must match boxes length")

    placed: List[Box] = []
    skipped: List[Box] = []
    for box, rotation in zip(boxes_list, rotations):
        ok, placed_box = try_place_box(box, pallet, rotation_override=rotation)
        if ok:
            placed.append(placed_box)
        else:
            skipped.append(box)
    return placed, skipped
