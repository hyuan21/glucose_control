"""
Build a corrected multi-dt result table with discounted return.

R = E[sum_{t=0}^{T-1} gamma^t r_t], gamma = 0.99.

Methods:
  PID
  TD3-BC
  PhiBE-TD3-BC                 (first-order)
  Complete PhiBE-TD3-BC        (full Hessian / complete HJB residual)
"""

import argparse
import csv
import os

import gym
import numpy as np

from eval_multidt_stage2 import (
    build_baseline,
    build_phibe,
    register_envs,
)
from utils import get_params, set_env_sample_time, test_algorithm
from utils.general import calculate_risk


FIRST_ORDER_TAG = "phibe_stage2_first_order_lam1e-4_norm_alpha1p75"
FULL_HESSIAN_TAG = "phibe_stage4_full_second_order_lam1e-2_norm_alpha1p25"


def parse_int_list(text):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_float_list(text):
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def discounted_return(bg_values, gamma):
    total = 0.0
    discount = 1.0
    for bg in bg_values:
        total += discount * (-calculate_risk([float(bg)]))
        discount *= gamma
    return total


def metrics(bg_values, reward, actions, gamma):
    bg = np.asarray(bg_values, dtype=np.float64)
    action_arr = np.asarray(actions, dtype=np.float64)
    return {
        "R": discounted_return(bg, gamma),
        "reward_sum": float(reward),
        "TIR": float(np.mean((bg >= 70.0) & (bg <= 180.0)) * 100.0),
        "TBR": float(np.mean(bg < 70.0) * 100.0),
        "TAR": float(np.mean(bg > 180.0) * 100.0),
        "Mean BG": float(np.mean(bg)),
        "Mean Action": float(np.mean(action_arr)),
    }


def evaluate(agent, patient_params, test_seed, dt, test_days, gamma, include_pid=False):
    env = gym.make(patient_params["env_name"])
    actual_dt = float(set_env_sample_time(env, dt))
    max_timesteps = int(round(test_days * 24 * 60 / actual_dt))
    (
        rl_reward,
        rl_bg,
        rl_action,
        _,
        _,
        pid_reward,
        pid_bg,
        pid_action,
    ) = test_algorithm(
        env=env,
        agent_action=agent.select_action,
        seed=test_seed,
        max_timesteps=max_timesteps,
        sequence_length=agent.sequence_length,
        data_processing=agent.data_processing,
        pid_run=(not include_pid),
        params=agent.params,
        sample_time=actual_dt,
    )
    rl_metrics = metrics(rl_bg, rl_reward, rl_action, gamma)
    pid_metrics = metrics(pid_bg, pid_reward, pid_action, gamma) if include_pid else None
    return rl_metrics, pid_metrics, actual_dt


def summarize(rows):
    order = [
        "TD3-BC",
        "PID",
        "PhiBE-TD3-BC",
        "Complete PhiBE-TD3-BC",
    ]
    metric_keys = ["R", "TIR", "TBR", "TAR", "Mean BG", "Mean Action"]
    summary = []
    for dt in sorted({row["delta-t"] for row in rows}):
        for method in order:
            subset = [row for row in rows if row["delta-t"] == dt and row["Method"] == method]
            if not subset:
                continue
            out = {"Method": method, "delta-t": dt, "N": len(subset)}
            for key in metric_keys:
                out[key] = float(np.mean([row[key] for row in subset]))
            summary.append(out)
    return summary


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["Method", "delta-t", "N", "R", "TIR", "TBR", "TAR", "Mean BG", "Mean Action"]
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_latex_table(rows):
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Corrected multi-$\Delta t$ evaluation on Adult\#1. $R=\mathbb{E}[\sum_{t=0}^{T-1}\gamma^t r_t]$, $\gamma=0.99$.}",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Method & $\Delta t$ & N & $R$ & TIR & TBR & TAR & Mean BG & Mean Action \\",
        r"\midrule",
    ]
    prev_dt = None
    for row in rows:
        if prev_dt is not None and row["delta-t"] != prev_dt:
            lines.append(r"\midrule")
        dt_text = f"{row['delta-t']:g} min"
        lines.append(
            (
                f"{row['Method']} & {dt_text} & {row['N']} & {row['R']:.2f} & "
                f"{row['TIR']:.2f}\\% & {row['TBR']:.2f}\\% & {row['TAR']:.2f}\\% & "
                f"{row['Mean BG']:.1f} & {row['Mean Action']:.4f} \\\\"
            )
        )
        prev_dt = row["delta-t"]
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", default="adult#1")
    parser.add_argument("--train-seeds", default="0,1,2")
    parser.add_argument("--test-seeds", default="0,1,2")
    parser.add_argument("--dts", default="1,3,5")
    parser.add_argument("--test-days", type=float, default=10.0)
    parser.add_argument("--training-timesteps", type=int, default=int(1e5))
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda-phibe", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=1.25)
    parser.add_argument("--output-csv", default="./Results/multidt_table_with_discounted_return.csv")
    parser.add_argument("--output-tex", default="./Results/multidt_table_with_discounted_return.tex")
    args = parser.parse_args()

    register_envs()
    patient_params = get_params()[args.patient]
    train_seeds = parse_int_list(args.train_seeds)
    test_seeds = parse_int_list(args.test_seeds)
    dts = parse_float_list(args.dts)

    rows = []
    pid_cache = {}

    for train_seed in train_seeds:
        print(f"Preparing train seed {train_seed}", flush=True)
        agents = {
            "TD3-BC": build_baseline(patient_params, train_seed, args.training_timesteps),
            "PhiBE-TD3-BC": build_phibe(
                patient_params,
                train_seed,
                args.training_timesteps,
                FIRST_ORDER_TAG,
                "first_order",
                args.lambda_phibe,
                args.alpha,
            ),
            "Complete PhiBE-TD3-BC": build_phibe(
                patient_params,
                train_seed,
                args.training_timesteps,
                FULL_HESSIAN_TAG,
                "full_second_order",
                args.lambda_phibe,
                args.alpha,
            ),
        }

        for dt in dts:
            for test_seed in test_seeds:
                for method, agent in agents.items():
                    include_pid = method == "TD3-BC" and (float(dt), test_seed) not in pid_cache
                    print(
                        f"Evaluating method={method} train_seed={train_seed} "
                        f"test_seed={test_seed} dt={dt:g}",
                        flush=True,
                    )
                    rl_row, pid_row, actual_dt = evaluate(
                        agent,
                        patient_params,
                        test_seed,
                        dt,
                        args.test_days,
                        args.gamma,
                        include_pid=include_pid,
                    )
                    rl_row.update({"Method": method, "delta-t": actual_dt})
                    rows.append(rl_row)
                    if pid_row is not None and (actual_dt, test_seed) not in pid_cache:
                        pid_row.update({"Method": "PID", "delta-t": actual_dt})
                        pid_cache[(actual_dt, test_seed)] = pid_row

    rows.extend(pid_cache.values())
    summary = summarize(rows)
    write_csv(summary, args.output_csv)
    latex = format_latex_table(summary)
    os.makedirs(os.path.dirname(args.output_tex), exist_ok=True)
    with open(args.output_tex, "w") as file:
        file.write(latex)

    print(latex)
    print(f"Saved CSV: {args.output_csv}")
    print(f"Saved LaTeX: {args.output_tex}")


if __name__ == "__main__":
    main()
