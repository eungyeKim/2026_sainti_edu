"""Create an RSL-RL initialization checkpoint from LimX's WF_TRON2A ONNX models.

The checkpoint transfers the actor and history encoder. A new PPO run learns
the critic and policy standard deviation and starts with a fresh optimizer.
"""

import argparse
from pathlib import Path

import onnx
from onnx import numpy_helper
import torch


MODEL_DIR = Path(__file__).resolve().parent / "models" / "WF_TRON2A"


def load_mlp(
    path: Path, widths: tuple[int, ...], prefix: str
) -> dict[str, torch.Tensor]:
    model = onnx.load(path)
    onnx.checker.check_model(model)
    graph = model.graph
    expected_ops = ["MatMul", "Add", "Elu"] * (len(widths) - 2) + ["MatMul", "Add"]
    if [node.op_type for node in graph.node] != expected_ops:
        raise ValueError(f"{path.name}: expected a chain of linear layers with ELU")

    tensors = {tensor.name: numpy_helper.to_array(tensor) for tensor in graph.initializer}
    state = {}
    previous = graph.input[0].name
    for i, (n_in, n_out) in enumerate(zip(widths, widths[1:])):
        multiply, add = graph.node[3 * i : 3 * i + 2]
        matrix = tensors[multiply.input[1]]
        bias_name = next(name for name in add.input if name != multiply.output[0])
        bias = tensors[bias_name]
        if (multiply.input[0] != previous or multiply.output[0] not in add.input
                or matrix.shape != (n_in, n_out) or bias.shape != (n_out,)):
            raise ValueError(f"{path.name}: unexpected linear layer {i}")

        name = f"{prefix}.{2 * i}"
        state[f"{name}.weight"] = torch.from_numpy(matrix.T.copy())
        state[f"{name}.bias"] = torch.from_numpy(bias.copy())
        previous = add.output[0]
        if i < len(widths) - 2:
            elu = graph.node[3 * i + 2]
            alpha = next((a.f for a in elu.attribute if a.name == "alpha"), 1.0)
            if elu.input[0] != previous or alpha != 1.0:
                raise ValueError(f"{path.name}: unexpected ELU at layer {i}")
            previous = elu.output[0]

    if previous != graph.output[0].name:
        raise ValueError(f"{path.name}: unexpected graph output")
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=MODEL_DIR / "ppo_warm_start.pt")
    args = parser.parse_args()

    actor = load_mlp(MODEL_DIR / "policy.onnx", (40, 512, 256, 128, 10), "actor")
    encoder = load_mlp(MODEL_DIR / "encoder.onnx", (340, 256, 128, 3), "encoder")
    checkpoint = {
        "model_state_dict": actor,
        "encoder_state_dict": encoder,
        "iter": 0,
        "infos": {"source": "LimX WF_TRON2A policy.onnx + encoder.onnx", "warm_start": True},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
