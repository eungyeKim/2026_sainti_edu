"""Print MuJoCo state, control, sensor, and SDK indices for WF_TRON2A."""

from pathlib import Path
import sys

import mujoco


SIMULATOR = Path(__file__).resolve().parent / "vendor" / "tron2_mujoco_sim"
MODEL = SIMULATOR / "robot-description/tron2a/WF_TRON2A/xml/robot.xml"

sys.path.insert(0, str(SIMULATOR))
from tron2_sim.channels import JointChannel
from tron2_sim.spec import LEG_WF, ChannelSpec


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print(f"WF_TRON2A | {MODEL.name} | indices are zero-based")
    print(f"nq={model.nq}  nv={model.nv}  nu={model.nu}  njnt={model.njnt}  "
          f"nbody={model.nbody}  nsensor={model.nsensor}  nsensordata={model.nsensordata}  "
          f"dt={model.opt.timestep:g} s")

    base_id = model.body("base_Link").jntadr[0]
    qpos = model.jnt_qposadr[base_id]
    qvel = model.jnt_dofadr[base_id]
    print(f"\nbase_Link free joint | joint_id={base_id}")
    print(f"qpos[{qpos}:{qpos + 3}] xyz   = {data.qpos[qpos:qpos + 3]}")
    print(f"qpos[{qpos + 3}:{qpos + 7}] wxyz  = {data.qpos[qpos + 3:qpos + 7]}")
    print(f"qvel[{qvel}:{qvel + 3}] linear  xyz = {data.qvel[qvel:qvel + 3]}")
    print(f"qvel[{qvel + 3}:{qvel + 6}] angular xyz = {data.qvel[qvel + 3:qvel + 6]}")

    spec = ChannelSpec("main", LEG_WF)
    channel = JointChannel(spec).resolve(model)
    print(f"\nSDK {spec.name} channel | qpos in rad, ctrl is motor torque in N·m")
    print(f"{'sdk':>3} {'joint_id':>8} {'qpos':>5} {'qvel':>5} {'ctrl':>5}  "
          f"{'joint':<23} {'actuator':<17} {'qpos_init':>9}  ctrlrange")
    for sdk_id, (name, (qpos, qvel, ctrl)) in enumerate(zip(spec.joints, channel.map)):
        joint_id = model.actuator_trnid[ctrl, 0]
        low, high = model.actuator_ctrlrange[ctrl]
        print(f"{sdk_id:3} {joint_id:8} {qpos:5} {qvel:5} {ctrl:5}  "
              f"{name:<23} {model.actuator(ctrl).name:<17} {data.qpos[qpos]:9g}  "
              f"[{low:g}, {high:g}]")

    print("\nSensors | sensordata slices use Python's exclusive stop index")
    print(f"{'id':>3}  {'sensor':<23} {'sensordata':<19} width")
    for sensor_id in range(model.nsensor):
        start = model.sensor_adr[sensor_id]
        width = model.sensor_dim[sensor_id]
        data_slice = f"sensordata[{start}:{start + width}]"
        print(f"{sensor_id:3}  {model.sensor(sensor_id).name:<23} {data_slice:<19} {width:>5}")


if __name__ == "__main__":
    main()
