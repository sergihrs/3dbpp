import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import config
from engine import collides, has_sufficient_support, is_top_down_clear, is_within_bounds
from models import Box, Pallet


class TestEngine(unittest.TestCase):
    def test_collides_touching_faces(self) -> None:
        box_a = Box(box_id=1, dims=(100, 100, 100)).with_pose((0, 0, 0), 0)
        box_b = Box(box_id=2, dims=(100, 100, 100)).with_pose((100, 0, 0), 0)
        self.assertFalse(collides(box_b, [box_a]))

        box_c = Box(box_id=3, dims=(100, 100, 100)).with_pose((50, 0, 0), 0)
        self.assertTrue(collides(box_c, [box_a]))

    def test_is_within_bounds(self) -> None:
        pallet = Pallet(length_mm=1200, width_mm=800, height_mm=1500)
        box_ok = Box(box_id=1, dims=(1200, 800, 100)).with_pose((0, 0, 0), 0)
        self.assertTrue(is_within_bounds(box_ok, pallet))

        box_over = Box(box_id=2, dims=(1200, 800, 100)).with_pose((1, 0, 0), 0)
        self.assertFalse(is_within_bounds(box_over, pallet))

        box_tall = Box(box_id=3, dims=(100, 100, 1600)).with_pose((0, 0, 0), 0)
        self.assertFalse(is_within_bounds(box_tall, pallet))

    def test_support_rules(self) -> None:
        base = Box(box_id=1, dims=(100, 100, 100)).with_pose((0, 0, 0), 0)

        on_floor = Box(box_id=2, dims=(50, 50, 50)).with_pose((0, 0, 0), 0)
        self.assertTrue(has_sufficient_support(on_floor, [base]))

        full_supported = Box(box_id=3, dims=(100, 100, 100)).with_pose((0, 0, 100), 0)
        self.assertTrue(has_sufficient_support(full_supported, [base]))

        half_supported = Box(box_id=4, dims=(100, 100, 100)).with_pose((50, 0, 100), 0)
        self.assertFalse(has_sufficient_support(half_supported, [base]))

    def test_top_down_blocked_by_tall_box(self) -> None:
        prev_margin = config.USE_GRIPPER_MARGIN
        config.USE_GRIPPER_MARGIN = False
        try:
            tall = Box(box_id=1, dims=(100, 100, 300)).with_pose((0, 0, 0), 0)
            candidate = Box(box_id=2, dims=(100, 100, 50)).with_pose((0, 50, 0), 0)
            self.assertFalse(is_top_down_clear(candidate, [tall]))
        finally:
            config.USE_GRIPPER_MARGIN = prev_margin

    def test_top_down_margin_blocks_gap(self) -> None:
        prev_margin = config.USE_GRIPPER_MARGIN
        prev_gap = config.GRIPPER_MARGIN_MM
        config.GRIPPER_MARGIN_MM = 10
        try:
            tall = Box(box_id=1, dims=(100, 100, 300)).with_pose((0, 0, 0), 0)
            candidate = Box(box_id=2, dims=(100, 100, 50)).with_pose((101, 0, 0), 0)

            config.USE_GRIPPER_MARGIN = False
            self.assertTrue(is_top_down_clear(candidate, [tall]))

            config.USE_GRIPPER_MARGIN = True
            self.assertFalse(is_top_down_clear(candidate, [tall]))
        finally:
            config.USE_GRIPPER_MARGIN = prev_margin
            config.GRIPPER_MARGIN_MM = prev_gap


if __name__ == "__main__":
    unittest.main()
