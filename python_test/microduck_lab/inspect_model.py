"""Inspect, render, or view the Microduck MJCF scene."""
import argparse
from math import sin, tau
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parent.parent
SCENE = ROOT / "sources/microduck_rl/src/mjlab_microduck/robot/microduck/scene_walk.xml"
RESULTS = Path(__file__).resolve().with_name("results")
CAMERA_LOOKAT = (0.01, 0, 0.13)
CAMERA_DISTANCE = 0.52
CAMERA_AZIMUTH = 152
CAMERA_ELEVATION = -18


def keyframe_data(model, keyframe):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key(keyframe).id)
    mujoco.mj_forward(model, data)
    return data


def standing_model():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    return model, keyframe_data(model, "STAND")


def print_model(model, data, keyframe):
    print(f"nq={model.nq} nv={model.nv} nu={model.nu} dt={model.opt.timestep}")
    print(f"{keyframe} root xyz + wxyz:", data.qpos[:7])
    print("ctrl  actuator -> joint             qpos qvel   angle(rad) target(rad)")
    for i in range(model.nu):
        joint = model.actuator_trnid[i, 0]
        q = model.jnt_qposadr[joint]
        v = model.jnt_dofadr[joint]
        print(i, model.actuator(i).name, "->", model.joint(joint).name,
              q, v, round(data.qpos[q], 6), round(data.ctrl[i], 6))
    for i in range(model.nsensor):
        print("sensor", model.sensor(i).name, "address", model.sensor_adr[i],
              "width", model.sensor_dim[i])


def render_pose(model, data, keyframe):
    from PIL import Image

    camera = mujoco.MjvCamera()
    camera.lookat[:] = CAMERA_LOOKAT
    camera.distance = CAMERA_DISTANCE
    camera.azimuth = CAMERA_AZIMUTH
    camera.elevation = CAMERA_ELEVATION
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"microduck_{keyframe.lower()}.png"
    with mujoco.Renderer(model, width=640, height=480) as renderer:
        renderer.update_scene(data, camera=camera)
        Image.fromarray(renderer.render()).save(path)
    print(f"Saved {keyframe}: {path}")


def show_viewer(keyframe):
    from mujoco import viewer

    # 관절 시연용 지지대를 메모리의 모델에만 추가해 몸통이 넘어지지 않게 한다.
    spec = mujoco.MjSpec.from_file(str(SCENE))
    spec.add_equality(name="viewer_trunk_support", type=mujoco.mjtEq.mjEQ_WELD,
                      objtype=mujoco.mjtObj.mjOBJ_BODY, name1="trunk_base")
    model = spec.compile()
    model.stat.center[:] = CAMERA_LOOKAT
    model.stat.extent = CAMERA_DISTANCE / 1.5
    model.vis.global_.azimuth = CAMERA_AZIMUTH
    model.vis.global_.elevation = CAMERA_ELEVATION
    data = keyframe_data(model, keyframe)
    head_yaw = model.actuator("head_yaw").id
    initial_yaw = float(data.ctrl[head_yaw])

    def update_controls(active_model, active_data):
        if active_model is model:
            active_data.ctrl[head_yaw] = initial_yaw + 0.25 * sin(tau * active_data.time / 3)

    previous_control = mujoco.get_mjcb_control()
    mujoco.set_mjcb_control(update_controls)
    try:
        viewer.launch(model, data)
    finally:
        mujoco.set_mjcb_control(previous_control)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keyframe", choices=("INIT", "STAND"), default="STAND",
                        help="불러올 MJCF 자세 (기본값: STAND)")
    parser.add_argument("--render", action="store_true", help="선택한 초기 자세를 PNG로 저장")
    parser.add_argument("--viewer", action="store_true", help="MuJoCo GUI에서 머리 관절 시연")
    args = parser.parse_args()
    if args.render or not args.viewer:
        model = mujoco.MjModel.from_xml_path(str(SCENE))
        data = keyframe_data(model, args.keyframe)
        if args.render:
            render_pose(model, data, args.keyframe)
        else:
            print_model(model, data, args.keyframe)
    if args.viewer:
        show_viewer(args.keyframe)
