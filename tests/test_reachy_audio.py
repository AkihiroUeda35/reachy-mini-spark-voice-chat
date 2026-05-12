from __future__ import annotations

import unittest

import numpy as np
from reachy_mini.utils import create_head_pose

from reachy_audio import HeadWobbler


class HeadWobblerTests(unittest.TestCase):
    def test_finish_after_audio_moves_head_and_resets_to_base_pose(self) -> None:
        poses: list[np.ndarray] = []
        base_pose = create_head_pose(0.02, -0.01, 0.015, 0.05, -0.08, 0.12, degrees=False)

        def set_pose(pose: np.ndarray) -> None:
            poses.append(pose.copy())

        wobbler = HeadWobbler(set_pose, lambda: base_pose.copy())
        audio = np.full(12_000, 10_000, dtype=np.int16)
        try:
            wobbler.feed_pcm(audio, 24_000)
            self.assertTrue(wobbler.finish(timeout_s=2.0))
        finally:
            wobbler.stop()

        self.assertGreaterEqual(len(poses), 2)
        self.assertTrue(any(not np.allclose(pose, base_pose) for pose in poses[:-1]))
        self.assertTrue(np.allclose(poses[-1], base_pose))

    def test_finish_without_audio_is_immediately_idle(self) -> None:
        poses: list[np.ndarray] = []

        def set_pose(pose: np.ndarray) -> None:
            poses.append(pose.copy())

        wobbler = HeadWobbler(set_pose)
        try:
            self.assertTrue(wobbler.finish(timeout_s=0.1))
        finally:
            wobbler.stop()

        self.assertTrue(np.allclose(poses[-1], create_head_pose(degrees=False)))


if __name__ == "__main__":
    unittest.main()