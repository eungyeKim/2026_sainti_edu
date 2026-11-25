"""Day 1 실습 09. SB3 PPO로 한 관절 팔을 학습하고 ONNX로 추론한다."""
import argparse
from csv import DictWriter
from importlib import import_module
from math import isfinite, sqrt
from pathlib import Path
from shutil import copyfile
from statistics import mean
from tempfile import mkdtemp

import torch
from matplotlib.figure import Figure
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from torch import nn

from common import RESULTS, save_json

ArmEnv = import_module("gym_bridge").ArmEnv
CHECKPOINT = RESULTS / "09_arm_ppo.zip"
BEST_CHECKPOINT = RESULTS / "09_arm_ppo_best.zip"
MODEL_FILE = RESULTS / "09_arm_policy.onnx"
LOSS_CSV = RESULTS / "09_arm_loss.csv"
LOSS_PLOT = RESULTS / "09_arm_loss.png"
EVAL_SEEDS = (11, 22, 33, 44, 55)
ROLLOUT_STEPS = 128
DEFAULT_CHECKPOINT_EVERY = 512
DEFAULT_LEARNING_RATE = 0.0015
DEFAULT_ACTIVATION = "leaky_relu"
ACTIVATIONS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
}
LOSS_LABELS = {
    "loss": "Mean total PPO loss",
    "policy_gradient_loss": "Policy gradient loss",
    "value_loss": "Value loss",
    "entropy_loss": "Entropy loss",
}


class OnnxActor(nn.Module):
    """학습된 actor의 결정적 행동만 내보낸다. 입력과 출력은 각각 [1, 3], [1, 1]."""

    def __init__(self, policy):
        super().__init__()
        # 같은 PyTorch 층을 가져오므로 재로딩한 SB3 가중치가 그대로 쓰인다.
        self.features = policy.features_extractor
        self.hidden = policy.mlp_extractor.policy_net
        self.output = policy.action_net

    def forward(self, observation):
        # 이 Box 정책에서 predict(deterministic=True)는 정규분포 평균을 행동 범위로 자른다.
        mean_action = self.output(self.hidden(self.features(observation)))
        return mean_action.clamp(-1.0, 1.0)


def evaluate(env, act):
    """모든 제어기를 같은 시작 각도에서 실행해 보상과 추종 오차를 비교한다."""
    returns, final_errors = [], []
    error_squared, steps, failures = 0.0, 0, 0
    for seed in EVAL_SEEDS:
        obs, _ = env.reset(seed=seed)
        total = 0.0
        while True:
            obs, reward, terminated, truncated, info = env.step(act(obs))
            total += reward
            error = info["error_rad"]
            error_squared += error ** 2
            steps += 1
            if terminated or truncated:
                returns.append(total)
                final_errors.append(abs(error))
                failures += int(terminated)
                break
    return dict(mean_return=mean(returns), rmse_rad=sqrt(error_squared / steps),
                final_abs_error_rad=mean(final_errors), failures=failures, steps=steps)


class TrainingHistory(BaseCallback):
    def __init__(self, evaluation_env, checkpoint_dir, checkpoint_every, loss_file):
        super().__init__()
        self.evaluation_env = evaluation_env
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = checkpoint_every
        self.loss_file = loss_file
        self.writer = DictWriter(loss_file, fieldnames=[
            "env_steps", *LOSS_LABELS, "last_minibatch_loss",
        ])
        self.writer.writeheader()
        self.losses = []
        self.checkpoints = []

    def _on_step(self):
        return True

    def _on_rollout_start(self):
        # 다음 rollout이 시작될 때 직전 rollout의 최적화가 완료되어 있다.
        self._record_update()

    def _on_training_end(self):
        # 마지막 업데이트 뒤에는 새 rollout이 없으므로 여기서 기록한다.
        self._record_update(final=True)

    def _record_update(self, final=False):
        logs = self.logger.name_to_value
        if "train/loss" not in logs:
            return

        steps = self.model.num_timesteps
        policy_loss = float(logs["train/policy_gradient_loss"])
        value_loss = float(logs["train/value_loss"])
        entropy_loss = float(logs["train/entropy_loss"])
        # SB3의 train/loss는 마지막 minibatch 값이다. 각 항의 평균으로 전체 평균도 계산한다.
        row = dict(
            env_steps=steps,
            loss=policy_loss + self.model.vf_coef * value_loss + self.model.ent_coef * entropy_loss,
            last_minibatch_loss=float(logs["train/loss"]),
            policy_gradient_loss=policy_loss,
            value_loss=value_loss,
            entropy_loss=entropy_loss,
        )
        self.losses.append(row)
        self.writer.writerow(row)
        self.loss_file.flush()

        previous_steps = self.checkpoints[-1]["env_steps"] if self.checkpoints else 0
        if final or steps - previous_steps >= self.checkpoint_every:
            path = self.checkpoint_dir / f"ppo_{steps:08d}.zip"
            self.model.save(path)
            score = evaluate(self.evaluation_env, lambda obs: self.model.predict(obs, deterministic=True)[0])
            self.checkpoints.append(dict(env_steps=steps, path=str(path.relative_to(RESULTS)), **score))
            print(f"체크포인트 {steps:>5}스텝 | 평균 누적 보상 {score['mean_return']:.6f}")


def save_loss_plot(losses):
    figure = Figure(figsize=(10, 6), layout="constrained")
    axes = figure.subplots(2, 2, sharex=True)
    steps = [row["env_steps"] for row in losses]
    for axis, (key, label) in zip(axes.flat, LOSS_LABELS.items()):
        axis.plot(steps, [row[key] for row in losses], marker=".", markersize=4)
        axis.set_title(label)
        axis.grid(alpha=0.25)
    figure.supxlabel("Environment steps")
    figure.savefig(LOSS_PLOT, dpi=150)


def train(env, steps, checkpoint_every=DEFAULT_CHECKPOINT_EVERY, *,
          learning_rate=DEFAULT_LEARNING_RATE, activation=DEFAULT_ACTIVATION):
    # microduck_lab과 같은 CPU PPO 구성. actor(pi)와 critic(vf)은 별도의 은닉층을 학습한다.
    model = PPO("MlpPolicy", env, seed=7, device="cpu", n_steps=ROLLOUT_STEPS, batch_size=64,
                n_epochs=10, learning_rate=learning_rate, clip_range=0.2,
                policy_kwargs=dict(net_arch=dict(pi=[32, 32], vf=[32, 32]),
                                   activation_fn=ACTIVATIONS[activation], optimizer_class=torch.optim.Adam))

    # 평가 환경을 분리해 기준선 측정이 학습 환경의 초기 상태나 난수 흐름에 관여하지 않게 한다.
    checkpoint_root = RESULTS / "09_arm_checkpoints"
    checkpoint_root.mkdir(exist_ok=True)
    checkpoint_dir = Path(mkdtemp(prefix="run-", dir=checkpoint_root))
    with ArmEnv() as test_env, LOSS_CSV.open("w", newline="", encoding="utf-8") as loss_file:
        zero = evaluate(test_env, lambda obs: [0.0])
        before = evaluate(test_env, lambda obs: model.predict(obs, deterministic=True)[0])
        history = TrainingHistory(test_env, checkpoint_dir, checkpoint_every, loss_file)

        # 한 번의 learn 호출로 rollout과 PPO 업데이트를 계속 이어 간다.
        model.learn(total_timesteps=steps, callback=history)

    best = max(history.checkpoints, key=lambda checkpoint: checkpoint["mean_return"])
    copyfile(RESULTS / history.checkpoints[-1]["path"], CHECKPOINT)
    copyfile(RESULTS / best["path"], BEST_CHECKPOINT)
    save_loss_plot(history.losses)
    restored = PPO.load(CHECKPOINT, device="cpu")
    with ArmEnv() as test_env:
        after = evaluate(test_env, lambda obs: restored.predict(obs, deterministic=True)[0])

    # critic과 탐색용 표준편차는 학습에 쓰인다. 배포용 그래프에는 actor 경로만 담는다.
    actor = OnnxActor(restored.policy).eval()
    torch.onnx.export(actor, torch.zeros(1, 3), str(MODEL_FILE), opset_version=17,
                      input_names=["observation"], output_names=["action"])
    metrics = dict(training_seed=7, requested_steps=steps, training_steps=model.num_timesteps,
                   learning_rate=learning_rate, activation=activation,
                   evaluation_seeds=EVAL_SEEDS, zero=zero, before=before, sb3=after,
                   checkpoint_every=checkpoint_every, checkpoint_selection="mean_return",
                   checkpoints=history.checkpoints, best_checkpoint=best)
    print(f"실제 학습 {model.num_timesteps}스텝 | SB3 평균 보상 {after['mean_return']:.6f}")
    print(f"최고 체크포인트 {best['env_steps']}스텝 | 평균 누적 보상 {best['mean_return']:.6f}")
    print(CHECKPOINT, BEST_CHECKPOINT, MODEL_FILE, LOSS_CSV, LOSS_PLOT,
          save_json("09_arm_train.json", metrics), sep="\n")


def infer(env):
    import onnxruntime as ort

    # Day 4와 같은 방식으로 세션을 한 번 열고, 매 제어 주기마다 관측 한 건을 보낸다.
    session = ort.InferenceSession(str(MODEL_FILE), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    def act(obs):
        batch = session.run([output_name], {input_name: obs.reshape(1, 3)})[0]
        return batch[0]  # [1, 1] → [1]. 07번 env.step이 행동 × 2를 토크로 적용한다.

    result = evaluate(env, act)
    print(f"ONNX 평균 보상 {result['mean_return']:.6f} | "
          f"최종 평균 절대 오차 {result['final_abs_error_rad']:.5f} rad")
    print(save_json("09_arm_infer.json", dict(evaluation_seeds=EVAL_SEEDS, onnx=result)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    train_args = modes.add_parser("train", help="PPO 학습, ZIP 저장·복원, ONNX 내보내기")
    train_args.add_argument("--steps", type=int, default=4096, help="학습 환경 스텝 수")
    train_args.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY,
                            help="체크포인트 사이의 최소 환경 스텝 수 (기본값: %(default)s)")
    train_args.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE,
                            help="Adam 학습률 (기본값: %(default)s)")
    train_args.add_argument("--activation", choices=ACTIVATIONS, default=DEFAULT_ACTIVATION,
                            help="actor와 critic의 은닉층 활성화 함수 (기본값: %(default)s)")
    modes.add_parser("infer", help="ONNX Runtime으로 팔 제어 및 평가")
    args = parser.parse_args()
    if args.mode == "train":
        if args.steps < 1:
            parser.error("--steps는 양의 정수여야 합니다.")
        if args.checkpoint_every < 1:
            parser.error("--checkpoint-every는 양의 정수여야 합니다.")
        if not isfinite(args.learning_rate) or args.learning_rate <= 0:
            parser.error("--learning-rate는 유한한 양수여야 합니다.")
    if args.mode == "infer" and not MODEL_FILE.is_file():
        parser.error("ONNX 파일이 없습니다. 먼저 train을 실행하세요.")
    torch.set_num_threads(1)
    with ArmEnv() as env:
        if args.mode == "train":
            train(env, args.steps, args.checkpoint_every,
                  learning_rate=args.learning_rate, activation=args.activation)
        else:
            infer(env)
