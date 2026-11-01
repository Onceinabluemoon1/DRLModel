import inspect
import unittest
from pathlib import Path

from aisle_replay.safety_gate import GateCommand, InstanceRange, SafetyGate, load_rules


class SafetyGateTest(unittest.TestCase):
    def setUp(self):
        self.gate = SafetyGate()

    def _cmd(self, *instances: InstanceRange) -> GateCommand:
        return self.gate.evaluate(list(instances))

    def test_rules_match_the_slice(self):
        rules = load_rules()
        self.assertEqual(rules.person_stop_m, 2.5)
        self.assertEqual(rules.object_stop_m, 1.0)
        self.assertEqual(rules.cruise_vx_m_s, 0.6)
        self.assertEqual(rules.max_person_stop_delay_frames, 1)

    def test_evaluate_does_not_take_a_velocity(self):
        params = list(inspect.signature(SafetyGate.evaluate).parameters)
        self.assertEqual(params, ["self", "instances"])

    def test_source_does_not_read_the_velocity_process(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("safety_gate.py", "replay.py", "backproject.py"):
            text = (root / name).read_text(encoding="utf-8")
            self.assertNotIn("velocity_node", text)
            self.assertNotIn("socket", text)

    def test_person_threshold_uses_meters(self):
        self.assertEqual(self._cmd(InstanceRange("PERSON", 0.9, 2.5, 0.0)).reason, "STOP_PERSON")
        self.assertEqual(self._cmd(InstanceRange("PERSON", 0.9, 2.51, 0.0)).reason, "OK")
        far = self._cmd(InstanceRange("PERSON", 0.99, 4.0, 0.0))
        self.assertEqual(far.reason, "OK")
        self.assertAlmostEqual(far.vx_m_s, 0.6)

    def test_missing_range_does_not_invent_a_distance(self):
        cmd = self._cmd(InstanceRange("PERSON", 0.99, None, None))
        self.assertEqual(cmd.reason, "OK")

    def test_person_outside_the_lane_does_not_stop(self):
        cmd = self._cmd(InstanceRange("PERSON", 0.9, 1.2, 1.3))
        self.assertEqual(cmd.reason, "OK")

    def test_tote_and_pallet_use_one_meter(self):
        self.assertEqual(self._cmd(InstanceRange("TOTE", 0.8, 1.0, 0.0)).reason, "STOP_DISTANCE")
        self.assertEqual(self._cmd(InstanceRange("TOTE", 0.8, 1.01, 0.0)).reason, "OK")
        self.assertEqual(self._cmd(InstanceRange("PALLET", 0.8, 0.8, -0.2)).reason, "STOP_DISTANCE")
        self.assertEqual(self._cmd(InstanceRange("TOTE", 0.8, 0.6, 1.4)).reason, "OK")

    def test_person_wins_over_a_closer_tote(self):
        cmd = self._cmd(
            InstanceRange("TOTE", 0.9, 0.5, 0.0),
            InstanceRange("PERSON", 0.4, 2.0, 0.1),
        )
        self.assertEqual(cmd.reason, "STOP_PERSON")
        self.assertEqual(cmd.vx_m_s, 0.0)

    def test_empty_observation_cruises_at_the_rule_speed(self):
        cmd = self._cmd()
        self.assertEqual(cmd.reason, "OK")
        self.assertAlmostEqual(cmd.vx_m_s, 0.6)
        self.assertNotAlmostEqual(cmd.vx_m_s, 0.8)


if __name__ == "__main__":
    unittest.main()
