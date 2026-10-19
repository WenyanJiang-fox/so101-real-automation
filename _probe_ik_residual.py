# -*- coding: utf-8 -*-
"""离线复现 run12 那一步竖直补位的逆解残差：free=(1,2)、ow=0、目标 z 降 4.3mm。"""
import numpy as np
import mujoco

from real.arm import load_model, JOINT_ORDER, MJCF, joint_limits

m, d, site, adr = load_model(MJCF)
lim = joint_limits(m)
q_lo = np.array([lim[n][0] for n in JOINT_ORDER])
q_hi = np.array([lim[n][1] for n in JOINT_ORDER])


class Shim:
    q_lo, q_hi = q_lo, q_hi
    ow = 0.0

    def _fk(self, q):
        for i, n in enumerate(JOINT_ORDER):
            d.qpos[adr[n]] = float(q[i])
        mujoco.mj_kinematics(m, d)
        pos = np.array(d.site_xpos[site], dtype=np.float64)
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, d.site_xmat[site])
        return pos, quat

    def _jac(self, q):
        self._fk(q)
        mujoco.mj_comPos(m, d)
        jp = np.zeros((3, m.nv))
        jr = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jp, jr, site)
        cols = [m.jnt_dofadr[adr[n]] for n in JOINT_ORDER]
        return np.vstack([jp[:, cols], jr[:, cols]])


arm = Shim()
q0 = np.radians(np.array([-19.3, 31.7, -15.3, 82.4, -123.2, 99.3]))
pos, quat = arm._fk(q0)
tgt = pos + np.array([0.0, 0.0, -0.0043])
print("base", np.round(pos * 1000, 2), "target", np.round(tgt * 1000, 2))

from skills.pick_real.motion import _ik_free                       # noqa: E402

for it in (3, 12, 40, 200):
    q = _ik_free(arm, tgt, quat, q0, (1, 2), ow=0.0, iters=it)
    got = arm._fk(q)[0]
    print(f"iters={it:3d}  dq(deg)={np.round(np.degrees(q[:5] - q0[:5]), 2)}  "
          f"got={np.round(got * 1000, 2)}  err={np.linalg.norm(got - tgt) * 1000:.2f}mm")
