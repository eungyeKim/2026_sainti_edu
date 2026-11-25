"""선택 실습: 작은 MuJoCo 모델을 Gymnasium 계약으로 감싼다. 학습은 하지 않는다."""
import argparse

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env
from common import load_model, save_json


PD_CANDIDATES = {
    "PD1": (2.0, 0.2),
    "PD2": (2.0, 0.4),
    "PD3": (4.0, 0.2),
    "PD4": (4.0, 0.4),
    "PD5": (6.0, 0.2),
    "PD6": (6.0, 0.4),
    "PD7": (8.0, 0.2),
    "PD8": (8.0, 0.4),
    "PD9": (12.0, 0.2),
    "PD10": (12.0, 0.4),
    "PD11": (11.24, 1.272),
}
CONTROLLER_GAINS = {"zero": (0.0, 0.0), "PD": (4.0, 0.6), **PD_CANDIDATES}


class ArmEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        self.model, self.data = load_model("arm.xml")
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
        self.target = np.pi / 4
        self.steps = 0

    def _obs(self):
        return np.array([self.data.joint("hinge").qpos[0],
                         self.data.joint("hinge").qvel[0], self.target], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.joint("hinge").qpos[0] = self.np_random.uniform(-0.1, 0.1)
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        return self._obs(), {}

    def step(self, action):
        torque = 2.0 * float(np.clip(action[0], -1, 1))
        self.data.actuator("motor").ctrl[0] = torque
        # 0.002초 x 10 = 0.020초마다 새 행동을 계산한다.
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        obs = self._obs()
        error = float(self.target - obs[0])
        reward = -(error**2 + 0.01 * float(obs[1])**2 + 0.001 * torque**2)
        terminated = bool(abs(obs[0]) > np.pi)
        truncated = bool(self.steps >= 200)
        return obs, float(reward), terminated, truncated, {"error_rad": error}


def evaluate(mode, seeds):
    kp, kd = CONTROLLER_GAINS[mode]
    return evaluate_gains(kp, kd, seeds)


def evaluate_gains(kp, kd, seeds):
    results = []
    env = ArmEnv()
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            total = 0.0
            for step in range(200):
                error = float(obs[2]) - float(obs[0])
                torque = kp * error - kd * float(obs[1])
                action = np.array([np.clip(torque / 2.0, -1, 1)], dtype=np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                if step == 0:
                    first_reward = reward
                total += reward
                if terminated or truncated:
                    break
            results.append({"seed": seed, "return": total, "final_error_rad": info["error_rad"],
                            "first_step_reward": first_reward, "final_step_reward": reward,
                            "terminated": terminated, "truncated": truncated})
    finally:
        env.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="PD 제어 결과를 모드별로 확인한다.")
    parser.add_argument("--mode", choices=["all", *CONTROLLER_GAINS], default="all",
                        help="평가할 모드 (기본값: all)")
    args = parser.parse_args()

    env = ArmEnv()
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()

    seeds = [11, 22, 33, 44, 55]
    if args.mode == "all":
        modes = CONTROLLER_GAINS
    else:
        modes = [args.mode]

    results = {}
    for mode in modes:
        kp, kd = CONTROLLER_GAINS[mode]
        episodes = evaluate(mode, seeds)
        mean_return = float(np.mean([episode["return"] for episode in episodes]))
        results[mode] = {
            "kp": kp,
            "kd": kd,
            "mean_return": mean_return,
            "mean_abs_final_error_rad": float(np.mean([
                abs(episode["final_error_rad"]) for episode in episodes
            ])),
            "episodes": episodes,
        }
        print(f"{mode:>4}  Kp={kp:g}  Kd={kd:g}  mean return={mean_return:.6f}")

    # 개별 모드에서는 시드별 결과를 출력하고 전체 순위 계산을 건너뛴다.
    if args.mode != "all":
        result = results[args.mode]
        print(f"Mean absolute final error: {result['mean_abs_final_error_rad']:.6g} rad")
        for episode in result["episodes"]:
            final_error = abs(episode["final_error_rad"])
            print(f"  seed={episode['seed']}  return={episode['return']:.6f}  "
                  f"absolute final error={final_error:.6g} rad  "
                  f"terminated={episode['terminated']}  truncated={episode['truncated']}")
        return

    # 중도 종료한 후보는 보상 누적 시간이 짧아 순위에서 제외한다.
    ranking = sorted(
        (mode for mode in PD_CANDIDATES if all(
            episode["truncated"] and not episode["terminated"]
            for episode in results[mode]["episodes"]
        )),
        key=lambda mode: results[mode]["mean_return"],
        reverse=True,
    )
    best_mode = next(iter(ranking), None)
    path = save_json("07_gym_metrics.json", {
        "seeds": seeds,
        "selection_metric": "mean_return",
        "candidate_ranking": ranking,
        "best_candidate": best_mode,
        "controllers": results,
    })
    if best_mode is None:
        print("No PD candidate completed all episodes.")
    else:
        best = results[best_mode]
        print(f"Best candidate: {best_mode}  Kp={best['kp']:g}  Kd={best['kd']:g}  "
              f"mean return={best['mean_return']:.6f}")
    print(f"Saved: {path}")
    print("PASS: Gymnasium contract checked; controllers evaluated, no RL training.")


if __name__ == "__main__":
    main()
