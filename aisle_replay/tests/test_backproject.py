import unittest

import numpy as np

from aisle_replay.backproject import range_from_mask
from aisle_replay.camera import default_camera
from aisle_replay.clips import build_clip
from aisle_replay.synth import LABEL_PALLET, LABEL_PERSON, LABEL_TOTE


class BackprojectTest(unittest.TestCase):
    def test_project_roundtrip(self):
        camera = default_camera()
        points = np.array(
            [
                [1.2, -0.2, 0.40],
                [2.5, 0.3, 1.10],
                [0.9, 0.0, 0.20],
            ]
        )
        u, v, z = camera.project(points)
        back = camera.backproject(u, v, z)
        np.testing.assert_allclose(back, points, atol=1e-6)

    def test_known_pixel_depth_returns_that_point(self):
        camera = default_camera()
        point = np.array([[1.0, 0.1, 0.30]])
        u, v, z = camera.project(point)
        ui = int(np.rint(u[0]))
        vi = int(np.rint(v[0]))
        self.assertTrue(0 <= ui < camera.width)
        self.assertTrue(0 <= vi < camera.height)
        depth = np.zeros((camera.height, camera.width), dtype=np.uint16)
        mask = np.zeros_like(depth, dtype=bool)
        # 铺一块小邻域，避免单点被近表面滤波丢掉。
        depth[vi - 2 : vi + 3, ui - 2 : ui + 3] = np.uint16(round(float(z[0]) * 1000))
        mask[vi - 2 : vi + 3, ui - 2 : ui + 3] = True
        range_m, lateral = range_from_mask(depth, mask, camera)
        self.assertIsNotNone(range_m)
        self.assertAlmostEqual(range_m, 1.0, delta=0.02)
        self.assertAlmostEqual(lateral, 0.1, delta=0.05)

    def test_tote_at_one_meter_is_within_two_centimeters(self):
        camera = default_camera()
        clip = build_clip("tote_1m", camera)
        errors = []
        for frame in clip.frames:
            mask = frame.label == LABEL_TOTE
            self.assertGreater(int(mask.sum()), 80, "1 m 周转箱在画面里看不见")
            range_m, lateral = range_from_mask(frame.depth_mm, mask, camera)
            self.assertIsNotNone(range_m)
            errors.append(abs(range_m - 1.0) * 100.0)
            self.assertLess(abs(lateral), 0.15)
        self.assertLess(max(errors), 2.0, errors)

    def test_behavior_clips_are_visible(self):
        expected = {"person": LABEL_PERSON, "tote": LABEL_TOTE, "pallet": LABEL_PALLET}
        for name, label_id in expected.items():
            clip = build_clip(name)
            for frame in clip.frames:
                self.assertGreater(int((frame.label == label_id).sum()), 40, name)


if __name__ == "__main__":
    unittest.main()
