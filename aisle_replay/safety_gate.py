"""安全门。只读观测和 rules.yaml，不读速度进程。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_RULES_PATH = Path(__file__).resolve().parent / "safety" / "rules.yaml"
_OBJECT_CLASSES = ("TOTE", "PALLET")


@dataclass(frozen=True)
class Rules:
    person_stop_m: float
    object_stop_m: float
    lane_half_width_m: float
    forward_min_m: float
    cruise_vx_m_s: float
    cruise_wz_rad_s: float
    max_person_stop_delay_frames: int


@dataclass(frozen=True)
class InstanceRange:
    class_name: str
    score: float
    range_m: float | None
    lateral_m: float | None


@dataclass(frozen=True)
class GateCommand:
    vx_m_s: float
    wz_rad_s: float
    reason: str


def load_rules(path: Path | None = None) -> Rules:
    raw = yaml.safe_load((path or _RULES_PATH).read_text(encoding="utf-8"))
    return Rules(
        person_stop_m=float(raw["person_stop_m"]),
        object_stop_m=float(raw["object_stop_m"]),
        lane_half_width_m=float(raw["lane_half_width_m"]),
        forward_min_m=float(raw["forward_min_m"]),
        cruise_vx_m_s=float(raw["cruise_vx_m_s"]),
        cruise_wz_rad_s=float(raw["cruise_wz_rad_s"]),
        max_person_stop_delay_frames=int(raw["max_person_stop_delay_frames"]),
    )


class SafetyGate:
    def __init__(self, rules: Rules | None = None):
        self.rules = rules or load_rules()

    def evaluate(self, instances: list[InstanceRange]) -> GateCommand:
        """速度判断的唯一入口。参数里没有策略输出。"""
        rules = self.rules
        if any(self._person_in_stop_zone(item) for item in instances):
            return GateCommand(0.0, 0.0, "STOP_PERSON")
        if any(self._object_blocking(item) for item in instances):
            return GateCommand(0.0, 0.0, "STOP_DISTANCE")
        return GateCommand(rules.cruise_vx_m_s, rules.cruise_wz_rad_s, "OK")

    def _in_forward_lane(self, item: InstanceRange, stop_m: float) -> bool:
        if item.range_m is None or item.lateral_m is None:
            return False
        rules = self.rules
        return (
            rules.forward_min_m < item.range_m <= stop_m
            and abs(item.lateral_m) <= rules.lane_half_width_m
        )

    def _person_in_stop_zone(self, item: InstanceRange) -> bool:
        return item.class_name == "PERSON" and self._in_forward_lane(
            item, self.rules.person_stop_m
        )

    def _object_blocking(self, item: InstanceRange) -> bool:
        return item.class_name in _OBJECT_CLASSES and self._in_forward_lane(
            item, self.rules.object_stop_m
        )
