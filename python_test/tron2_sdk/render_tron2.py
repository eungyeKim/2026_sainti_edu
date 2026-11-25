"""Render the four bundled TRON2A models to PNG with MuJoCo."""

import os
from pathlib import Path
import sys

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
from PIL import Image


ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "vendor/tron2_mujoco_sim/robot-description/tron2a"
OUTPUT = ROOT / "results/render_tron2"
WIDTH = 1200
HEIGHT = 1200

# Model file and camera framing (center height and distance in meters).
VIEWS = {
    "SF_TRON2A": ("robot.xml", 0.75, 1.95),
    "WF_TRON2A": ("robot.xml", 0.72, 1.95),
    "DA_TRON2A": ("robot.xml", 0.69, 2.25),
    "DACH_TRON2A": ("robot_grasper.xml", 0.82, 2.65),
}


def render_variant(name: str, filename: str, center_height: float, distance: float) -> Path:
    model = mujoco.MjModel.from_xml_path(str(MODELS / name / "xml" / filename))
    model.vis.global_.offwidth = WIDTH
    model.vis.global_.offheight = HEIGHT
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    camera = mujoco.MjvCamera()
    camera.lookat[:] = (0, 0, center_height)
    camera.distance = distance
    camera.azimuth = 160
    camera.elevation = -12

    # Group 0 holds the collision geometry and the ground. Keep only the ground.
    model.geom("ground").group[0] = 1
    options = mujoco.MjvOption()
    options.geomgroup[:] = 0
    options.geomgroup[1] = 1

    destination = OUTPUT / f"{name}.png"
    with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
        renderer.update_scene(data, camera=camera, scene_option=options)
        Image.fromarray(renderer.render()).save(destination)
    return destination


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, (filename, center_height, distance) in VIEWS.items():
        print(render_variant(name, filename, center_height, distance))


if __name__ == "__main__":
    main()
