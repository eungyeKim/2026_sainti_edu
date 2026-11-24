"""공 낙하를 실시간에 가깝게 재생한다. 닫기 버튼 또는 --seconds로 종료."""
import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer

parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=15.0)
args = parser.parse_args()
model_path = Path(__file__).resolve().parents[2] / "models" / "ball.xml"
model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    with viewer.lock():
        viewer.cam.lookat[:] = [0, 0, 0.4]
        viewer.cam.distance = 2.7
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
    end = time.perf_counter() + args.seconds
    while viewer.is_running() and time.perf_counter() < end:
        start = time.perf_counter()
        if data.time >= 2.0:
            mujoco.mj_resetData(model, data)
            mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(max(0.0, model.opt.timestep - (time.perf_counter() - start)))
print("Viewer closed normally.")
