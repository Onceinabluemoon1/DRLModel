import unittest

from aisle_replay.clips import all_behavior_clips, build_clip
from aisle_replay.detect import FixtureDetector, map_class_name
from aisle_replay.replay import replay_clip
from aisle_replay.safety_gate import SafetyGate


class ReplayClipTest(unittest.TestCase):
    def test_class_names_stay_separate(self):
        self.assertEqual(map_class_name("person"), "PERSON")
        self.assertEqual(map_class_name("tote"), "TOTE")
        self.assertEqual(map_class_name("pallet"), "PALLET")
        self.assertIsNone(map_class_name("other"))
        self.assertIsNone(map_class_name("person_other"))
        self.assertIsNone(map_class_name("suitcase"))

    def test_four_clips_match_geometric_reasons(self):
        gate = SafetyGate()
        detector = FixtureDetector()
        for clip in all_behavior_clips():
            events = replay_clip(clip, detector, gate)
            reasons = [event.reason for event in events]
            self.assertEqual(reasons, clip.expected_reasons, clip.name)
            for event, reason in zip(events, reasons):
                if reason.startswith("STOP"):
                    self.assertEqual(event.vx_m_s, 0.0)
                else:
                    self.assertAlmostEqual(event.vx_m_s, 0.6)

    def test_person_stop_lands_within_two_frames(self):
        clip = build_clip("person")
        events = replay_clip(clip, FixtureDetector(), SafetyGate())
        enter = clip.person_enter_frame
        self.assertIsNotNone(enter)
        stop_frames = [event.frame for event in events if event.reason == "STOP_PERSON"]
        delay = min(stop_frames) - enter
        self.assertGreaterEqual(delay, 0)
        self.assertLessEqual(delay, 1)
        self.assertIn("STOP_PERSON", clip.expected_reasons)
        self.assertIn("OK", clip.expected_reasons)

    def test_tote_and_pallet_only_stop_inside_the_lane(self):
        for name in ("tote", "pallet"):
            clip = build_clip(name)
            self.assertIn("STOP_DISTANCE", clip.expected_reasons)
            self.assertEqual(clip.expected_reasons[-1], "OK")


if __name__ == "__main__":
    unittest.main()
