#!/usr/bin/env python3
"""
Master Automation CLI: Executes training and evaluation across all 4 AirfRANS Benchmark Tasks.
  1. Full Task (Interpolation)
  2. Scarce Task (Few-Shot Generalization)
  3. Reynolds Extrapolation Task (OOD Reynolds)
  4. Angle of Attack Extrapolation Task (OOD Flow Separation)
"""
import os
import subprocess
import sys
import time

TASKS = ["full", "scarce", "reynolds_extrapolation", "aoa_extrapolation"]


def run_command(cmd: str):
    print(f"\n[EXEC] {cmd}")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print(f"[ERROR] Command failed with exit code {res.returncode}: {cmd}")
        return False
    return True


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print("=" * 70)
    print("  Master Suite: 4-Task AirfRANS Neural Operator Benchmark")
    print("=" * 70)

    start_suite = time.time()
    results = {}

    for task in TASKS:
        print(f"\n>>> Launching Benchmark Task: {task.upper()} <<<")
        train_cmd = f"python {os.path.join(root_dir, 'scripts', 'train.py')} task={task}"
        success = run_command(train_cmd)
        if not success:
            print(f"[WARN] Skipping evaluation for task {task} due to training error.")
            continue

        chk_path = f"./outputs/{task}/latest/checkpoints/best_model.pt"
        if not os.path.exists(chk_path):
            chk_path = f"./outputs/{task}/checkpoints/best_model.pt"

        if os.path.exists(chk_path):
            eval_cmd = f"python {os.path.join(root_dir, 'scripts', 'evaluate.py')} --checkpoint {chk_path} --task {task}"
            run_command(eval_cmd)

    total_time = time.time() - start_suite
    print("\n" + "=" * 70)
    print(f"  Master Benchmark Suite Completed in {total_time/3600.0:.2f} hours")
    print("=" * 70)


if __name__ == "__main__":
    main()
