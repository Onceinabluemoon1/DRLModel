import unittest

from aisle_replay.prove_kill import prove


class VelocityKilledTest(unittest.TestCase):
    def test_person_replay_still_stops_after_the_velocity_process_dies(self):
        report = prove()
        self.assertTrue(report["velocity_killed"])
        self.assertEqual(report["policy_vx_while_alive"], 0.8)
        self.assertEqual(report["empty_vx_m_s"], 0.6)
        self.assertIn("STOP_PERSON", report["person_reasons"])
        self.assertLessEqual(report["delay_frames"], 1)
        self.assertGreaterEqual(report["delay_frames"], 0)


if __name__ == "__main__":
    unittest.main()
