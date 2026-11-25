"""Check reacher kinematics or view the arm moving in MuJoCo."""
import argparse
import json
from math import cos, sin, tau
from pathlib import Path

import mujoco
import numpy as np

MODEL = Path(__file__).with_name("reacher.xml")


def experiment():
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    poses = []
    for shoulder, wrist in [(0, 0), (0, 90), (90, 0), (30, -60)]:
        a, b = np.deg2rad([shoulder, wrist])
        data.joint("shoulder").qpos[0] = a
        data.joint("wrist").qpos[0] = b
        mujoco.mj_forward(model, data)
        expected = [.12 * (np.cos(a) + np.cos(a+b)),
                    .12 * (np.sin(a) + np.sin(a+b)), .01]
        actual = data.geom("finger").xpos.copy()
        np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=0)
        poses.append({"degrees": [shoulder, wrist], "finger_m": actual.tolist()})
    assert data.time == 0.0
    controls = {}
    for name in ("shoulder", "wrist"):
        motor = model.actuator(name).id
        joint = model.joint(name).id
        assert model.actuator_trnid[motor, 0] == joint
        controls[name] = [int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint]), motor]
    mujoco.mj_resetData(model, data)
    data.actuator("shoulder").ctrl[0] = 0.2
    mujoco.mj_step(model, data, nstep=5)
    assert np.isfinite(data.qvel).all() and np.linalg.norm(data.qvel) > 0
    np.testing.assert_allclose(data.time, 5 * model.opt.timestep)
    assert (model.nq, model.nv, model.nu) == (2, 2, 2)
    return {"shape": [model.nq, model.nv, model.nu], "addresses_q_v_u": controls,
            "poses": poses, "pulse_seconds": data.time, "pulse_qvel": data.qvel.tolist()}


def show_viewer():
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    model.vis.global_.elevation = -89
    shoulder = model.actuator("shoulder").id
    wrist = model.actuator("wrist").id

    def update_controls(active_model, active_data):
        if active_model is model:
            phase = tau * active_data.time / 4.0
            active_data.ctrl[shoulder] = 0.65 * cos(phase)
            active_data.ctrl[wrist] = 0.55 * sin(phase)

    previous_control = mujoco.get_mjcb_control()
    mujoco.set_mjcb_control(update_controls)
    try:
        mujoco.viewer.launch(model, data)
    finally:
        mujoco.set_mjcb_control(previous_control)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer", action="store_true", help="MuJoCo GUI에서 두 관절의 움직임 보기")
    args = parser.parse_args()
    if args.viewer:
        show_viewer()
    else:
        print(json.dumps(experiment(), indent=2))
