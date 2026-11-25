"""Replay the pretrained WF_TRON2A policy in MuJoCo. Space pauses, R resets."""

import argparse
from collections import deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import glfw
import mujoco
from mujoco import viewer
import numpy as np
import onnxruntime as ort
import yaml


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "WF_TRON2A"
XML = ROOT / "vendor/tron2_mujoco_sim/robot-description/tron2a/WF_TRON2A/xml/robot.xml"
STRAIGHT_MODEL = MODEL_DIR / "straight_heading.json"


@dataclass(frozen=True)
class StraightHeading:
    bias: float
    speed_gain: float
    heading_gain: float
    lateral_gain: float
    damping: float

    def command(self, speed, heading, cross_track, yaw_rate, limit):
        feedforward = self.bias + self.speed_gain * (speed - 0.35)
        correction = feedforward - self.heading_gain * heading - self.lateral_gain * cross_track - self.damping * yaw_rate
        return max(-limit, min(limit, correction))


class WheelfootReplay:
    def __init__(self, config, speed, straight=None):
        self.model = mujoco.MjModel.from_xml_path(str(XML))
        self.data = mujoco.MjData(self.model)
        self.base_id = self.model.body("base_Link").id
        base_joint = self.model.body_jntadr[self.base_id]
        self.base_qpos = self.model.jnt_qposadr[base_joint]
        self.yaw_limit = config["commands"]["max"]["ang_vel_yaw"]
        self.speed = speed
        self.straight = straight
        self.stand_duration = config["stand_mode"]["stand_duration"]
        self.control = config["control"]
        self.policy_steps = round(
            self.control["decimation"] / config["loop_frequency"] / self.model.opt.timestep
        )
        names = config["init_state"]["joint_names"]
        self.ctrl = np.array([self.model.actuator(name).id for name in names])
        joints = self.model.actuator_trnid[self.ctrl, 0]
        if [self.model.joint(j).name for j in joints] != [f"{name}_Joint" for name in names]:
            raise ValueError("The WF_TRON2A actuator mapping differs from the policy's joint order")
        self.qpos = self.model.jnt_qposadr[joints]
        self.qvel = self.model.jnt_dofadr[joints]
        self.wheels = np.asarray(self.control["joint_control_modes"]) == "velocity"
        light = np.array([name.startswith("proximal_yaw_") for name in names])
        self.default_angles = np.array([config["init_state"]["default_joint_angle"][n] for n in names])
        self.kp = self.joint_values(light, "stiffness", wheel_value=0.0)
        self.kd = self.joint_values(light, "damping", self.control["wheel_joint_damping"])
        self.torque_limits = self.joint_values(light, "torque_limit", self.control["wheel_joint_torque_limit"])
        scales = config["normalization"]
        self.obs_scales = scales.get("obs_scales", {"ang_vel": 0.25, "dof_pos": 1.0, "dof_vel": 0.05})
        self.obs_clip = scales["clip_scales"]["clip_observations"]
        self.action_clip = scales["clip_scales"]["clip_actions"]
        size = config["size"]
        self.history = np.empty((size["obs_history_length"], size["policy_obs_size"]), dtype=np.float32)
        self.action = np.zeros(size["actions_size"], dtype=np.float32)

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.encoder = ort.InferenceSession(str(MODEL_DIR / "encoder.onnx"), options, providers=["CPUExecutionProvider"])
        self.policy = ort.InferenceSession(str(MODEL_DIR / "policy.onnx"), options, providers=["CPUExecutionProvider"])
        self.reset()

    def joint_values(self, light, field, wheel_value):
        return np.where(self.wheels, wheel_value, np.where(
            light, self.control[f"light_joint_{field}"], self.control[f"leg_joint_{field}"]
        ))

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qpos] = self.default_angles
        mujoco.mj_forward(self.model, self.data)
        # Start with the wheel collision geometry just above the ground plane.
        wheel_body = self.model.body("wheel_L_Link").id
        wheel = np.flatnonzero((self.model.geom_bodyid == wheel_body)
                               & (self.model.geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER))[0]
        ground_z = self.data.geom("ground").xpos[2]
        self.data.qpos[self.base_qpos + 2] += (
            ground_z + self.model.geom_size[wheel, 0] + 0.002 - self.data.geom_xpos[wheel, 2]
        )
        mujoco.mj_forward(self.model, self.data)
        self.start_xy = self.data.qpos[self.base_qpos:self.base_qpos + 2].copy()
        self.start_yaw = self.yaw()
        self.start_wheels = self.data.qpos[self.qpos[self.wheels]]
        self.step_count = 0
        self.action.fill(0)

    def yaw(self):
        w, x, y, z = self.data.qpos[self.base_qpos + 3:self.base_qpos + 7]
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def track(self):
        dx, dy = self.data.qpos[self.base_qpos:self.base_qpos + 2] - self.start_xy
        c, s = math.cos(self.start_yaw), math.sin(self.start_yaw)
        heading = math.remainder(self.yaw() - self.start_yaw, math.tau)
        return c * dx + s * dy, -s * dx + c * dy, heading

    def update_policy(self):
        scales = self.obs_scales
        ang_velocity = self.data.sensor("base_imu_gyro").data
        gravity = -self.data.site("base_imu").xmat.reshape(3, 3)[2]
        joint_pos = (self.data.qpos[self.qpos] - self.default_angles)[~self.wheels]
        obs = np.concatenate((scales["ang_vel"] * ang_velocity, gravity, scales["dof_pos"] * joint_pos,
                              scales["dof_vel"] * self.data.qvel[self.qvel], self.action)).astype(np.float32)
        if self.step_count == 0:
            self.history[:] = obs
        self.history[:-1] = self.history[1:]
        self.history[-1] = obs
        latent = self.encoder.run(None, {self.encoder.get_inputs()[0].name: self.history.ravel()})[0]
        speed = self.speed if self.data.time >= self.stand_duration else 0.0
        yaw = 0.0
        if self.straight is not None and speed > 0:
            _, lateral, heading = self.track()
            yaw = self.straight.command(speed, heading, lateral, ang_velocity[2], self.yaw_limit)
        command = np.array([speed, 0, yaw], dtype=np.float32)
        policy_input = np.concatenate((latent, np.clip(obs, -self.obs_clip, self.obs_clip), command))
        output = self.policy.run(None, {self.policy.get_inputs()[0].name: policy_input})[0]
        self.action[:] = np.clip(output, -self.action_clip, self.action_clip)

    def step(self):
        if self.step_count % self.policy_steps == 0:
            self.update_policy()
        pos_target = self.default_angles + self.control["action_scale_pos"] * self.action
        vel_target = np.where(self.wheels, self.control["action_scale_vel"] * self.action, 0.0)
        torque = self.kp * (pos_target - self.data.qpos[self.qpos]) + self.kd * (vel_target - self.data.qvel[self.qvel])
        self.data.ctrl[self.ctrl] = np.clip(torque, -self.torque_limits, self.torque_limits)
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

    def report(self):
        dx, dy, yaw = self.track()
        left, right = self.data.qpos[self.qpos[self.wheels]] - self.start_wheels
        print(f"t={self.data.time:.2f}s | forward={dx:.3f}m | lateral={dy:+.3f}m | "
              f"yaw={math.degrees(yaw):+.2f}deg | wheels L={left:.2f}rad R={right:.2f}rad", flush=True)


def show(replay, stop_step):
    events = deque()
    paused = False
    steps_per_frame = round(1 / (60 * replay.model.opt.timestep))
    # Keep the floor visible while hiding collision overlays.
    replay.model.geom("ground").group[0] = 1
    guide_z = replay.data.geom("ground").xpos[2] + 0.007
    with viewer.launch_passive(replay.model, replay.data, key_callback=events.append) as window:
        with window.lock():
            window.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            window.cam.trackbodyid = replay.base_id
            window.cam.distance = 2.25
            window.cam.azimuth = 140
            window.cam.elevation = -16
            window.opt.geomgroup[0] = 0
            guide = window.user_scn.geoms[0]
            mujoco.mjv_initGeom(guide, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3),
                                np.eye(3).ravel(), np.array([0.2, 0.9, 1.0, 1.0], dtype=np.float32))
            window.user_scn.ngeom = 1
        print("MuJoCo viewer ready | Space: pause/resume | R: reset", flush=True)
        while window.is_running() and (stop_step is None or replay.step_count < stop_step):
            frame_start = time.monotonic()
            while events:
                key = events.popleft()
                if key == glfw.KEY_SPACE:
                    paused = not paused
                elif key == glfw.KEY_R:
                    replay.reset()
            if not paused:
                steps = steps_per_frame if stop_step is None else min(steps_per_frame, stop_step - replay.step_count)
                for _ in range(steps):
                    replay.step()
            forward, _, _ = replay.track()
            direction = np.array([math.cos(replay.start_yaw), math.sin(replay.start_yaw), 0.0])
            origin = np.array([*replay.start_xy, guide_z])
            with window.lock():
                mujoco.mjv_connector(guide, mujoco.mjtGeom.mjGEOM_LINE, 3.0,
                                     origin + (forward - 4.0) * direction,
                                     origin + (forward + 4.0) * direction)
            window.sync()
            time.sleep(max(0.0, steps_per_frame * replay.model.opt.timestep - (time.monotonic() - frame_start)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speed", type=float, default=0.35, help="Forward speed command in m/s (default: 0.35)")
    parser.add_argument("--headless", action="store_true", help="Run the same policy without opening a window")
    parser.add_argument("--duration", type=float, help="Stop after this many seconds of simulation; headless default is 6")
    parser.add_argument("--baseline", action="store_true", help="Use the original LimX policy with zero yaw command")
    args = parser.parse_args()
    config = yaml.safe_load((MODEL_DIR / "params.yaml").read_text())["PointfootCfg"]
    maximum = config["commands"]["max"]["lin_vel_x"]
    if not -maximum <= args.speed <= maximum:
        parser.error(f"--speed must be between {-maximum:g} and {maximum:g} m/s")
    if args.duration is not None and (not np.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration must be a finite positive number")

    straight = None
    if not args.baseline:
        if not STRAIGHT_MODEL.is_file():
            parser.error(f"Straight-heading model missing: {STRAIGHT_MODEL}. Train it or use --baseline.")
        straight = StraightHeading(**json.loads(STRAIGHT_MODEL.read_text())["gains"])
    replay = WheelfootReplay(config, args.speed, straight)
    print(f"WF_TRON2A | {'LimX baseline' if straight is None else 'learned straight heading'} | "
          f"target speed {args.speed:+.2f} m/s", flush=True)
    dt = replay.model.opt.timestep
    if args.headless:
        for _ in range(round((args.duration or 6.0) / dt)):
            replay.step()
    else:
        show(replay, None if args.duration is None else round(args.duration / dt))
    replay.report()


if __name__ == "__main__":
    main()
