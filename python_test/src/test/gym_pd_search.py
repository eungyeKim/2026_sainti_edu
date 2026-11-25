"""PD11 주변의 제어 게인을 탐색하고 별도 시드에서 비교한다."""
from statistics import mean

from common import save_json
from gym_bridge import CONTROLLER_GAINS, evaluate_gains


def summarize(episodes):
    return {
        "mean_return": mean(episode["return"] for episode in episodes),
        "mean_first_step_reward": mean(episode["first_step_reward"] for episode in episodes),
        "mean_final_step_reward": mean(episode["final_step_reward"] for episode in episodes),
        "completed_episodes": sum(
            episode["truncated"] and not episode["terminated"] for episode in episodes
        ),
    }


def main():
    search_seeds = [11, 22, 33, 44, 55]
    validation_seeds = list(range(200, 300))
    baseline_kp, baseline_kd = CONTROLLER_GAINS["PD11"]

    # 앞선 탐색에서 좋은 점수가 나온 영역을 중심으로 Kp와 Kd를 함께 움직인다.
    candidates = []
    for kp_offset in range(-15, 16):
        kp = round(baseline_kp + 0.03 * kp_offset, 4)
        kd_center = baseline_kd + 0.10 * (kp - baseline_kp)
        for kd_offset in range(-12, 13):
            kd = round(kd_center + 0.002 * kd_offset, 4)
            episodes = evaluate_gains(kp, kd, search_seeds)
            candidates.append({"kp": kp, "kd": kd, **summarize(episodes)})

    best = max(
        (candidate for candidate in candidates if candidate["completed_episodes"] == len(search_seeds)),
        key=lambda candidate: candidate["mean_return"],
    )
    baseline = next(candidate for candidate in candidates
                    if (candidate["kp"], candidate["kd"]) == (baseline_kp, baseline_kd))

    baseline_episodes = evaluate_gains(baseline_kp, baseline_kd, validation_seeds)
    best_episodes = evaluate_gains(best["kp"], best["kd"], validation_seeds)
    baseline_validation = summarize(baseline_episodes)
    best_validation = summarize(best_episodes)
    improved_seeds = sum(new["return"] > old["return"]
                         for old, new in zip(baseline_episodes, best_episodes))

    path = save_json("07_gym_pd_refinement.json", {
        "model": "arm.xml",
        "target_deg": 45,
        "episode_seconds": 4,
        "selection_metric": "mean_return",
        "search": {
            "seeds": search_seeds,
            "candidate_count": len(candidates),
            "pd11": baseline,
            "best_candidate": best,
            "candidates": candidates,
        },
        "validation": {
            "seeds": validation_seeds,
            "pd11": {**baseline_validation, "episodes": baseline_episodes},
            "best_candidate": {**best_validation, "episodes": best_episodes},
            "mean_return_gain": best_validation["mean_return"] - baseline_validation["mean_return"],
            "improved_seeds": improved_seeds,
        },
    })
    print(f"Searched {len(candidates)} gain pairs. Best Kp={best['kp']:g}, Kd={best['kd']:g}")
    print(f"Search mean return: {baseline['mean_return']:.9f} -> {best['mean_return']:.9f}")
    print(f"Validation mean return: {baseline_validation['mean_return']:.9f} -> "
          f"{best_validation['mean_return']:.9f} ({improved_seeds}/{len(validation_seeds)} improved)")
    print(f"Validation mean final step reward: {best_validation['mean_final_step_reward']:.3g}")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
