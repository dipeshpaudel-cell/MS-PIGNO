#!/usr/bin/env python3
"""
CLI Preprocessing Script: Multiprocessed Cache Builder for AirfRANS Graphs.
Precomputes LS-FV weights, multi-level graph coarsening, and boundary metadata for fast training.
"""
import argparse
import glob
import multiprocessing
import os
import sys
import time
from tqdm import tqdm

# Ensure codebase root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocessor import AirfRANSGraphPreprocessor


def _process_single_file(args):
    raw_path, cache_dir = args
    base_name = os.path.basename(raw_path)
    parent_dir = os.path.basename(os.path.dirname(raw_path))
    output_path = os.path.join(cache_dir, parent_dir, base_name)

    if os.path.exists(output_path):
        return output_path, True  # Skipped (already cached)

    preprocessor = AirfRANSGraphPreprocessor()
    try:
        preprocessor.process_simulation_file(raw_path, output_path)
        return output_path, False
    except Exception as e:
        print(f"[ERROR] Failed processing {raw_path}: {e}")
        return None, False


def main():
    parser = argparse.ArgumentParser(description="Preprocess AirfRANS graphs into accelerated multi-level cache.")
    parser.add_argument("--raw_data_dir", type=str, default="./Dataset", help="Directory containing raw AirfRANS .pth files.")
    parser.add_argument("--cache_dir", type=str, default="./cache/preprocessed_airfrans", help="Output cache directory.")
    parser.add_argument("--num_workers", type=int, default=max(1, multiprocessing.cpu_count() - 2), help="Number of parallel workers.")
    parser.add_argument("--limit", "--max_sims", dest="limit", type=int, default=None, help="Optional limit on number of simulations for quick testing.")
    args = parser.parse_args()

    file_list = sorted(glob.glob(os.path.join(args.raw_data_dir, "graph_airfrans_data_batch_*", "*.pth")))
    if len(file_list) == 0:
        file_list = sorted(glob.glob(os.path.join(args.raw_data_dir, "*.pth")))

    if len(file_list) == 0:
        print(f"[ERROR] No .pth simulation files found in {args.raw_data_dir}")
        sys.exit(1)

    if args.limit:
        file_list = file_list[:args.limit]

    print(f"=== AirfRANS Multi-Scale Graph Preprocessor ===")
    print(f"Total raw simulation files: {len(file_list)}")
    print(f"Output cache directory:     {args.cache_dir}")
    print(f"Parallel worker processes:  {args.num_workers}")

    os.makedirs(args.cache_dir, exist_ok=True)
    task_args = [(f, args.cache_dir) for f in file_list]

    start_time = time.time()
    skipped_count = 0
    processed_count = 0

    if args.num_workers > 1:
        with multiprocessing.Pool(processes=args.num_workers) as pool:
            for out_path, was_skipped in tqdm(pool.imap_unordered(_process_single_file, task_args), total=len(task_args), desc="Caching AirfRANS"):
                if out_path:
                    if was_skipped:
                        skipped_count += 1
                    else:
                        processed_count += 1
    else:
        for item in tqdm(task_args, desc="Caching AirfRANS (sequential)"):
            out_path, was_skipped = _process_single_file(item)
            if out_path:
                if was_skipped:
                    skipped_count += 1
                else:
                    processed_count += 1

    elapsed = time.time() - start_time
    print(f"\n[DONE] Preprocessing Complete in {elapsed:.2f}s!")
    print(f"  Processed: {processed_count} files | Skipped (already cached): {skipped_count} files")


if __name__ == "__main__":
    main()
