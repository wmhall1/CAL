#!/usr/bin/env python3
"""
image_process.py

Run plotting functions from src.py on multiple files in parallel, safely on HPC.
"""

import sys
import os
import time
import functools
from multiprocessing import get_context
import tempfile

import src  # Your plotting functions and DataVTK live here

# -------------------------------
# Worker function for multiprocessing
# -------------------------------
def worker(filename, fn_name, plot_args):
    import os, tempfile, time

    # Each worker gets its own MPL cache
    os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix=f"mplcache_{os.getpid()}_")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure()
    plt.close()  # Force font cache

    import src  # Import src here, after MPLCONFIGDIR is set
    fn = getattr(src, fn_name)

    pid = os.getpid()
    print(f"[PID {pid}] Processing {filename}", flush=True)
    start_time = time.time()

    try:
        fn(filename, **plot_args)
        elapsed = time.time() - start_time
        print(f"[PID {pid}] ✅ Completed {filename} in {elapsed:.2f}s", flush=True)
    except Exception as e:
        print(f"[PID {pid}] ❌ Error processing {filename}: {e}", flush=True)


# -------------------------------
# Run function in parallel
# -------------------------------
def run_in_parallel(core_count, fn_name, file_list, plot_args):
    if core_count == 1:
        # Serial execution (debugging)
        for filename in file_list:
            worker(filename, fn_name, plot_args)
    else:
        # Multiprocessing with spawn (safe on HPC)
        ctx = get_context("spawn")
        with ctx.Pool(core_count) as pool:
            # functools.partial cannot pickle local functions, so wrap args in tuple
            args_iter = [(f, fn_name, plot_args) for f in file_list]
            pool.starmap(worker, args_iter)


# -------------------------------
# Main function
# -------------------------------
def main():
    if len(sys.argv) < 3:
        raise Exception("Usage: python image_process.py <num_cores> <function_name> <file1> [file2 ...] [-options]")

    core_count = int(sys.argv[1])
    fn_name = sys.argv[2]

    if not hasattr(src, fn_name):
        raise Exception(f"Function '{fn_name}' not found in src.py")

    # Parse files and optional flags
    args = sys.argv[3:]
    opts = [a for a in args if a.startswith("-")]
    file_list = [a for a in args if not a.startswith("-")]

    plot_args = {}
    if "-centered" in opts:
        plot_args["opt"] = True
    if "-skip" in opts:
        plot_args["check_skip"] = True

    print(f"Starting with {core_count} cores on {len(file_list)} files", flush=True)
    start = time.time()

    run_in_parallel(core_count, fn_name, file_list, plot_args)

    elapsed = time.time() - start
    print(f"All tasks completed in {elapsed:.2f}s", flush=True)


if __name__ == "__main__":
    main()

