"""真机抓取（包）。**外面仍然 `import skills.pick_real`** —— 这里把各段的公开名字原样转出去，124 处旧引用一行不用改。

五个运行时可写的开关**不在这里转出**：`WAIT_TICK` `STOP_AFTER` `REPLAY`
`BOX_CENTER_CHAN` `JAW_CHECK_IMG` 一律走 `skills.pick_real.knobs.X`，
写错地方会当场 AttributeError，不会静默失效。
"""
from __future__ import annotations

from .knobs import (
    ALIGN_MAX_MOVES, ALIGN_PROBE_DEG, ALIGN_STALL_CHUNKS, AXIS_TOL_PX,
    CHECKPOINT, CHECKPOINT_VERSION, CLEAR_3D_ENFORCE, CLEAR_3D_TOL_MM,
    CLEAR_AXIS_AMP_MAX_MM, CLEAR_AXIS_BACK, CLEAR_AXIS_FAR, CLEAR_BAND_HI_PX,
    CLEAR_BAND_LO_PX, CLEAR_BAND_ON, CLEAR_CONTACT_GAIN, CLEAR_EXIT_EPS_PX,
    CLEAR_FLAT_STOP, CLEAR_FREE_ON, CLEAR_GAIN, CLEAR_GAP_MIN_HALF,
    CLEAR_MIN_DEG_GAIN_MAX, CLEAR_MIN_JOINT_DEG, CLEAR_MIN_STEP_M, CLEAR_STEP_DZ_M,
    CLEAR_STEP_M, CLEAR_TARGET_PX, CLEAR_Z_KEEP_M, CLOSE_MAX_STEPS,
    CLOSE_START_STEPS, CONFIRM_DESCEND, CONFIRM_DESCEND_IMG, CONFIRM_GO,
    CONFIRM_IMG, CONFIRM_LIFT, CONFIRM_LIFT_IMG, CONFIRM_STEP,
    CONFIRM_WAIT_S, DESCEND_STEP, DESCEND_XY_TOL_MM, DESCEND_Z_TOL_MM,
    FIX_CLEAR_REAL, FREE_AXIS, FREE_DESCENT, FREE_GAP,
    GAIN_MAX, GAIN_MIN_ASK_DEG, GAP_FALLBACK_MIN_COS, GAP_FALLBACK_MIN_MM_DEG,
    GAP_FALLBACK_POOL, GRASP_Z_FINAL_REAL, GRASP_Z_MARGIN_REAL, GRASP_Z_OVER_TOP_REAL,
    JAW_RAW_IMG, JAW_SHAPE_JSON, LOOK_IMG, MANUAL_CMD,
    MANUAL_GOTO_TOL, MANUAL_MAX_MOVES, MANUAL_STATE, MANUAL_WAIT_S,
    NUDGE_STEP_M, PLACE_AFTER_PICK, PLACE_AWAY_M, PLACE_DEG_WIN,
    PLACE_OPEN_STEPS, PLACE_R_MAX_M, PLACE_R_MIN_M, PLACE_SAMPLE_MAX,
    PLACE_Z_TOL_MM, REAL_GOTO_TRIES, RESUME_TOL_DEG, SWEEP_4,
    SWEEP_HEAD_FLEX, SWEEP_WAY, _ax_free, _gap_free,
)
from .pose import (
    _align_deg, _declared_half, _fold90, _h,
    _project_px, _px_per_mm, _vertical, _view_radius,
    _wrist,
)
from .motion import (
    _cmd, _descend, _free_for, _goto,
    _ik_free, _joint_dir_xy, _step_min_deg, _trace,
    _wrist_turn,
)
from .rulers import (
    _axis_seg, _box_center_px, _box_deg_px, _box_half_m, _contour_pts,
    _exit_px, _gap_axis_px, _gap_leaving, _gap_need_px,
    _gap_ok, _gap_px, _gap_want_px, _jaw_axis_deg,
    _jaw_axis_model_deg, _jaw_check, _jaw_check_3d, _jaw_mm,
    _jaw_shape, _mark_frozen, _put_lines, _red_rect_deg,
    _save_look, _solve_delta, _want_from_dp, _yaw_from_px,
)
from .gates import (
    _arm_off_checkpoint, _confirm_descend_on, _confirm_lift, _confirm_on,
    _confirm_pass, _confirm_step, _confirm_step_on, _descend_to_grasp,
    _drop_checkpoint, _load_checkpoint, _manual_on, _peek_cmd,
    _publish, _resume_on, _save_checkpoint, _stop_here,
    _stop_set, _take_cmd, _wait_cmd,
)
from .look import (
    _look, _sweep,
)
from .place import (
    _open_gently, _place_real, _place_xy_real,
)
from .grasp import (
    _close_until_stall, _grip_hold, _gripper_deg,
)
from .align import (
    _align, _align_model, _clearance_loop, _set_clearance,
    _set_clearance_manual,
)
from .skill import (
    PickRealSkill, _close_and_lift,
)
from .knobs import (ALIGN_CHUNK, ALIGN_MAX_STEPS, ALIGN_ROT_SLEW, ALIGN_TOL_DEG,
                    CARRY, GENTLE, GRIP_BIAS, GRIP_K, GRIP_RATE, GRIP_SLACK,
                    LIFT_Z, POS_SCALE, SLOW, SPEED, STUCK_TOL,
                    Path, POS_SLEW, ROT_SLEW, RoundState, Skill, _ROOT,
                    acc_amax, anti_jam, follow, geom, np, register,
                    shapes, speed_vmax, sys, vision)
from . import knobs
from . import skill as _skill          # noqa: F401  注册是 import 副作用
