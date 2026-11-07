"""逐轮判定：这一轮成功了没、夹住的是哪个色、抬起来之后掉没掉。

仿真有真值，真机没有 —— 所以判定分两套，但对外是同一个 Verdict：

  GroundTruthJudge   仿真用。成功 = env.is_success()，颜色 = 本轮设的那一个。
  ThresholdJudge     真机用。成功 = 模块 A（夹爪角度）+ 模块 B（HSV 颜色）。

**影子模式**：仿真里 ThresholdJudge 也照跑，把「阈值会怎么判」和真值并排打出来。
阈值现在还没标定，所以它只报原始证据（合爪时角度、抬起时角度、接触数），
标定完自动开始出结论。真机接上之前，A/B 就已经在仿真里跑了几百轮。
"""

from __future__ import annotations

from dataclasses import dataclass

from layout import Layout


@dataclass
class Verdict:
    """一轮的结论。"""

    success: bool
    color: str                    # 真值颜色（仿真）或模块 B 的判定（真机）
    source: str                   # "truth" | "threshold"

    # 模块 A 的原始证据（从 trace 里取，能取到就有值）
    grip_closed_deg: float | None = None   # 合爪那一刻的 6 号角度
    grip_lifted_deg: float | None = None   # 抬起到位后的 6 号角度
    contacts_lifted: int | None = None     # 抬起时的方块-夹爪接触点数

    # 模块 A/B 按阈值给的结论（没标定就是 None）
    held: str | None = None
    dropped: bool | None = None

    def line(self) -> str:
        g = f"{self.grip_closed_deg:5.1f}°" if self.grip_closed_deg is not None else "  --  "
        l = f"{self.grip_lifted_deg:5.1f}°" if self.grip_lifted_deg is not None else "  --  "
        c = "-" if self.contacts_lifted is None else str(self.contacts_lifted)
        return (f"{'✅' if self.success else '❌'} {self.color:5s}"
                f"  合爪 {g}  举起 {l}  触点 {c}"
                f"  模块A={self.held or '未标定'} 掉={self.dropped if self.dropped is not None else '未标定'}")


def evidence_from_trace(trace) -> dict:
    """trace 是 expert 每阶段 append 的 (stage, x, y, z, contacts, grip_deg)。
    取 closed / lifted 两行 —— 正好是模块 A 判「夹住没」和「掉没掉」要的两组证据。"""
    by_stage = {row[0]: row for row in (trace or [])}
    out = {}
    for stage in ("closed", "lifted"):
        row = by_stage.get(stage)
        if row is not None:
            out[stage] = {"contacts": int(row[4]), "grip_deg": float(row[5])}
    return out


class GroundTruthJudge:
    """仿真主判据：成功是真值，颜色是本轮设的那个。"""

    source = "truth"

    def evaluate(self, backend, layout: Layout, trace) -> Verdict:
        ev = evidence_from_trace(trace)
        v = Verdict(success=backend.is_success(), color=layout.color, source=self.source)
        # trace 存的是关节原始角，换算成 0-180 舵机角，才和真机模块 A 同口径
        if "closed" in ev:
            v.grip_closed_deg = backend.servo_from_raw_joint_deg(ev["closed"]["grip_deg"])
        if "lifted" in ev:
            v.grip_lifted_deg = backend.servo_from_raw_joint_deg(ev["lifted"]["grip_deg"])
            v.contacts_lifted = ev["lifted"]["contacts"]
        v.held = self._shadow_held(v)
        v.dropped = self._shadow_dropped(v)
        return v

    @staticmethod
    def _shadow_held(v: Verdict):
        """影子模式：拿已在运行的阈值给结论；thresholds.json 没标定就返回 None。"""
        try:
            from grasp_check import classify, load_thresholds
            return classify(v.grip_closed_deg, load_thresholds())
        except Exception:
            return None

    @staticmethod
    def _shadow_dropped(v: Verdict):
        if v.grip_closed_deg is None or v.grip_lifted_deg is None:
            return None
        try:
            from grasp_check import gripper_cal, load_thresholds
            delta = gripper_cal(load_thresholds())["drop_delta_deg"]
        except Exception:
            return None
        return (v.grip_lifted_deg - v.grip_closed_deg) <= -delta
