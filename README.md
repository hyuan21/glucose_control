# Complete PhiBE-TD3-BC for Glucose Control

This repository contains the current research code for adding a continuous-time
physics-informed Bellman equation (PhiBE) regularizer to the TD3-BC offline RL
baseline for type 1 diabetes glucose control.

The current implementation focuses on the adult#1 virtual patient and compares:

- PID controller
- TD3-BC baseline
- Complete PhiBE-TD3-BC with a full Hessian/diffusion term

The main evaluation tests the same trained policy under sampling intervals
delta-t = 1, 3, and 5 minutes.

## Main Result

The current best stable Complete PhiBE setting is:

- lambda_phibe = 0.01
- alpha = 1.25
- training updates = 70,000
- training replay interval = 3 minutes

Summary results are saved in:

- `Results/final_multidt_complete_phibe_alpha1p25_summary.csv`
- `Results/final_multidt_complete_phibe_alpha1p25_raw.csv`

The Overleaf-ready report is in:

- `overleaf/main.tex`

## Environment

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate offline-glucose
```

If PyTorch installation fails on your machine, install the CPU or Apple Silicon
version from the official PyTorch instructions, then re-run the scripts below.

## Reproduce Baseline Data and TD3-BC

Generate the Emerson-style PID replay and train TD3-BC:

```bash
python run_emerson2023_td3bc_reproduction.py \
  --patients adult#1 \
  --train-seeds 0,1,2 \
  --test-seeds 0,1,2 \
  --replay-length 100000 \
  --training-timesteps 100000 \
  --output ./Results/emerson2023_td3bc_adult1.csv
```

This creates files under `Replays/` and `Models/`. These are ignored by Git
because they are generated artifacts.

## Train Complete PhiBE

Train Complete PhiBE for one seed:

```bash
python run_stage1_phibe_adult1.py \
  --train-seed 0 \
  --training-timesteps 100000 \
  --num-train-steps 70000 \
  --phibe-mode full_second_order \
  --lambda-phibe 0.01 \
  --alpha 1.25 \
  --save-tag phibe_stage4_full_second_order_lam1e-2_norm_alpha1p25
```

Repeat with `--train-seed 1` and `--train-seed 2` for the N=9 evaluation.

## Multi-Delta-t Evaluation

Evaluate TD3-BC, PID, and Complete PhiBE under 1, 3, and 5 minute intervals:

```bash
python eval_multidt_stage2.py \
  --train-seeds 0,1,2 \
  --test-seeds 0,1,2 \
  --dts 1,3,5 \
  --test-days 10 \
  --phibe-tag phibe_stage4_full_second_order_lam1e-2_norm_alpha1p25 \
  --phibe-mode full_second_order \
  --lambda-phibe 0.01 \
  --alpha 1.25 \
  --output ./Results/final_multidt_complete_phibe_alpha1p25_raw.csv \
  --summary-output ./Results/final_multidt_complete_phibe_alpha1p25_summary.csv
```

## Useful Scripts

- `TD3_BC.py`: TD3-BC baseline.
- `TD3_BC_phibe.py`: TD3-BC with first-order, diagonal second-order, and full second-order PhiBE.
- `run_emerson2023_td3bc_reproduction.py`: replay generation and baseline reproduction.
- `run_stage1_phibe_adult1.py`: PhiBE training entry point.
- `eval_multidt_stage2.py`: multi-delta-t closed-loop evaluation.
- `diagnose_complete_phibe_terms.py`: diagnostic script for TD loss, PhiBE loss, drift, and diffusion terms.
- `make_continuous_glucose_trajectory_figure.py`: trajectory figure generation.
- `make_multidt_table_with_discounted_return.py`: table generation with discounted return.

## Notes

The current code intentionally keeps training replay collection fixed at the
original 3-minute interval. Changing the training delta-t would require changing
the replay collection protocol and would no longer be directly comparable to the
TD3-BC baseline.
