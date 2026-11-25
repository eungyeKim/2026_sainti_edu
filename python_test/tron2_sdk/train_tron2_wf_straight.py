"""Fit the heading correction around the pretrained WF_TRON2A policy in MuJoCo."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import json
import math

import numpy as np
import yaml

from view_tron2_wf import MODEL_DIR, STRAIGHT_MODEL, StraightHeading, WheelfootReplay


@dataclass(frozen=True)
class Scenario:
    speed: float
    seconds: float
    yaw_disturbance: float = 0.0


TRAINING = (Scenario(0.20, 6, 0.18), Scenario(0.35, 6), Scenario(0.55, 6, -0.18))
VALIDATION = (Scenario(0.20, 10), Scenario(0.35, 10), Scenario(0.50, 10),
              Scenario(0.40, 10, -0.25))


def init_worker():
    global replay
    config = yaml.safe_load((MODEL_DIR / "params.yaml").read_text())["PointfootCfg"]
    replay = WheelfootReplay(config, speed=0.35)


def simulate(gains, scenario):
    replay.straight = gains
    replay.speed = scenario.speed
    replay.reset()
    dt = replay.model.opt.timestep
    yaw_dof = replay.model.jnt_dofadr[replay.model.body_jntadr[replay.base_id]] + 5
    count = round(scenario.seconds / dt)
    disturb_at = round(2.5 / dt)
    sum_lateral = sum_yaw = max_lateral = max_heading = 0.0
    samples = 0
    fallen = False
    for step in range(count):
        if step == disturb_at:
            replay.data.qvel[yaw_dof] += scenario.yaw_disturbance
        replay.step()
        if (step + 1) % replay.policy_steps == 0 or step + 1 == count:
            forward, lateral, yaw = replay.track()
            if not np.isfinite([forward, lateral, yaw]).all() or replay.data.qpos[replay.base_qpos + 2] < 0.45:
                fallen = True
                break
            max_lateral = max(max_lateral, abs(lateral))
            max_heading = max(max_heading, abs(yaw))
            sum_lateral += lateral ** 2
            sum_yaw += yaw ** 2
            samples += 1

    forward, lateral, yaw = replay.track()
    if fallen:
        return None
    target = scenario.speed * (scenario.seconds - replay.stand_duration)
    progress_penalty = 16 * max(0, 0.8 - forward / target) ** 2
    yaw_scale = math.radians(2.0)
    score = ((sum_lateral / samples + 0.25 * lateral ** 2) / 0.02 ** 2
             + (sum_yaw / samples + 0.25 * yaw ** 2) / yaw_scale ** 2 + progress_penalty)
    return {
        "score": score,
        "forward_m": forward,
        "lateral_m": lateral,
        "yaw_deg": math.degrees(yaw),
        "max_lateral_m": max_lateral,
        "max_yaw_deg": math.degrees(max_heading),
        "rms_lateral_m": math.sqrt(sum_lateral / samples),
        "rms_yaw_deg": math.degrees(math.sqrt(sum_yaw / samples)),
    }


def score_candidate(candidate):
    gains = StraightHeading(*candidate)
    scores = [simulate(gains, scenario) for scenario in TRAINING]
    return sum(1e6 if metrics is None else metrics["score"] for metrics in scores) / len(scores)


def evaluate(case):
    gains, scenario = case
    return {"scenario": asdict(scenario), "baseline": simulate(None, scenario), "trained": simulate(gains, scenario)}


def train(pool, generations, population, seed):
    rng = np.random.default_rng(seed)
    # Bias and speed-dependent bias, followed by heading, lateral, and yaw-rate feedback.
    mean = np.array([0.055, 0.15, 2.2, 2.0, 0.12])
    spread = np.array([0.06, 0.3, 1.5, 1.4, 0.16])
    floor = np.array([0.003, 0.015, 0.10, 0.10, 0.015])
    low = np.array([-0.2, -0.8, 0.0, 0.0, 0.0])
    high = np.array([0.3, 1.0, 8.0, 8.0, 0.9])
    best, best_score = mean.copy(), math.inf
    for generation in range(generations):
        batch = np.clip(rng.normal(mean, spread, size=(population, len(mean))), low, high)
        batch[0], batch[1] = best, mean
        scores = list(pool.map(score_candidate, batch))
        ranking = np.argsort(scores)
        if scores[ranking[0]] < best_score:
            best, best_score = batch[ranking[0]].copy(), scores[ranking[0]]
        elite = batch[ranking[:max(2, population // 4)]]
        mean = elite.mean(axis=0)
        spread = np.maximum(floor, 0.45 * spread + 0.55 * elite.std(axis=0))
        print(f"Generation {generation + 1}/{generations} | best loss {best_score:.4f}", flush=True)
    return StraightHeading(*best), best_score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--evaluate", action="store_true", help="Compare the saved model with the LimX baseline")
    args = parser.parse_args()
    if args.generations < 1 or args.population < 4 or args.workers < 1:
        parser.error("require at least one generation, four candidates, and one worker")

    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as pool:
        if args.evaluate:
            gains = StraightHeading(**json.loads(STRAIGHT_MODEL.read_text())["gains"])
        else:
            gains, loss = train(pool, args.generations, args.population, args.seed)
        validation = list(pool.map(evaluate, [(gains, scenario) for scenario in VALIDATION]))

    print(json.dumps(validation, indent=2, allow_nan=False))
    if not args.evaluate:
        if any(row["trained"] is None or row["baseline"] is None
               or row["trained"]["score"] >= row["baseline"]["score"] for row in validation):
            raise SystemExit("The fitted model did not improve every validation scenario")
        result = {"gains": asdict(gains), "method": "cross-entropy optimization in MuJoCo", "seed": args.seed,
                  "generations": args.generations, "population": args.population, "training_loss": loss,
                  "validation": validation}
        STRAIGHT_MODEL.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(f"Saved {STRAIGHT_MODEL}")


if __name__ == "__main__":
    main()
