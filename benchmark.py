"""
Benchmark the OCR pipeline with different numbers of workers.

For every worker count the full pipeline is run several times. Runs are
interleaved round-robin (1, 2, 4, ... then again 1, 2, 4, ...) so slow drift
on the machine affects every configuration equally. One warm-up run is
discarded first.

Speedup uses the formula from the specification:

    S = T1 / Tn     (T = mean total processing time)

Every run's extracted text is compared with the first 1-worker run. Any
difference would indicate a race condition or a lost result, and aborts
the benchmark.

Outputs:
    benchmark_results.csv   one row per worker count (means, speedup, ...)
    benchmark_runs.csv      every individual timed run
    benchmark_summary.md    the results table plus test environment

Usage:
    python benchmark.py [--workers 1 2 4 8] [--runs 10] [--config config.txt]
"""

import argparse
import csv
import logging
import os
import platform
import statistics
import sys
import tempfile
from pathlib import Path

from config import load_config
from ocr import check_ocr_setup
from main import run_pipeline

DEFAULT_WORKERS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]
DEFAULT_RUNS = 10
OUTPUT_DIR = Path("bench_output")
# Configurations within this fraction of the best speedup count as "near
# optimal"; the smallest such worker count is reported as the knee.
KNEE_TOLERANCE = 0.05


def cpu_name():
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass

    return platform.processor() or platform.machine()


def texts_of(stats):
    return {result.filename: result.text for result in stats["results"]}


def run_benchmark(image_dir, worker_counts, runs):
    OUTPUT_DIR.mkdir(exist_ok=True)

    def run_once(n_workers):
        return run_pipeline(
            image_dir, n_workers, OUTPUT_DIR / f"ocr_results_{n_workers}w.csv"
        )

    print("Warm-up run (discarded)...")
    baseline = texts_of(run_once(1))

    samples = {n: [] for n in worker_counts}

    for run_number in range(1, runs + 1):
        for n_workers in worker_counts:
            stats = run_once(n_workers)

            if texts_of(stats) != baseline:
                raise RuntimeError(
                    f"Output with {n_workers} workers differs from the "
                    "1-worker output: possible race condition."
                )

            samples[n_workers].append(stats)

        print(f"  round {run_number}/{runs} done")

    return samples


def measure_overhead(worker_counts, runs):
    """
    Median time of a run over an EMPTY image folder for each worker count.

    This isolates the fixed cost of a run (starting threads, creating one
    Tesseract engine per worker, sentinels, joins) from the per-image work.
    """
    overhead = {}

    with tempfile.TemporaryDirectory() as empty_dir:
        output = Path(empty_dir) / "empty.csv"

        for n_workers in worker_counts:
            overhead[n_workers] = statistics.median(
                run_pipeline(empty_dir, n_workers, output)["total_ms"]
                for _ in range(runs)
            )

    return overhead


def summarize(samples, overhead):
    rows = []
    t1 = statistics.mean(s["total_ms"] for s in samples[min(samples)])
    work_t1 = t1 - overhead[min(samples)]

    for n_workers, runs in samples.items():
        totals = [s["total_ms"] for s in runs]
        mean_ms = statistics.mean(totals)
        per_worker = [count for s in runs for count in s["per_worker"].values()]
        images = runs[0]["images"]
        speedup = t1 / mean_ms
        work_ms = mean_ms - overhead[n_workers]

        rows.append(
            {
                "n_workers": n_workers,
                "runs": len(runs),
                "mean_ms": mean_ms,
                "median_ms": statistics.median(totals),
                "stdev_ms": statistics.stdev(totals) if len(totals) > 1 else 0.0,
                "min_ms": min(totals),
                "max_ms": max(totals),
                "speedup": speedup,
                "efficiency": speedup / n_workers,
                "startup_overhead_ms": overhead[n_workers],
                "speedup_excl_overhead": work_t1 / work_ms,
                "throughput_img_s": images / (mean_ms / 1000),
                "mean_image_ms": statistics.mean(
                    s["sum_image_ms"] / s["images"] for s in runs
                ),
                "mean_loader_read_ms": statistics.mean(
                    s["loader_read_ms"] for s in runs
                ),
                "min_images_per_worker": min(per_worker),
                "max_images_per_worker": max(per_worker),
                "images": images,
            }
        )

    return rows


def write_results(rows, path):
    fields = list(rows[0])

    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.4f}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )


def write_runs(samples, path):
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["n_workers", "run", "total_ms", "per_worker_images"])

        for n_workers, runs in samples.items():
            for run_number, stats in enumerate(runs, start=1):
                counts = " ".join(
                    str(stats["per_worker"].get(f"OCR-Worker-{i}", 0))
                    for i in range(1, n_workers + 1)
                )
                writer.writerow(
                    [n_workers, run_number, f"{stats['total_ms']:.4f}", counts]
                )


def choose_optimal(rows):
    best = max(rows, key=lambda row: row["speedup"])
    knee = min(
        (
            row
            for row in rows
            if row["speedup"] >= best["speedup"] * (1 - KNEE_TOLERANCE)
        ),
        key=lambda row: row["n_workers"],
    )
    return best, knee


def results_table(rows):
    lines = [
        "| Workers | Mean time (ms) | Std dev (ms) | Speedup S = T1/Tn "
        "| Efficiency S/n | Throughput (img/s) | Mean per-image time (ms) "
        "| Images per worker (min-max) | Startup overhead (ms) "
        "| Speedup excl. overhead |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in rows:
        lines.append(
            f"| {row['n_workers']} | {row['mean_ms']:.1f} | {row['stdev_ms']:.1f} "
            f"| {row['speedup']:.2f} | {row['efficiency']:.2f} "
            f"| {row['throughput_img_s']:.0f} | {row['mean_image_ms']:.2f} "
            f"| {row['min_images_per_worker']}-{row['max_images_per_worker']} "
            f"| {row['startup_overhead_ms']:.1f} "
            f"| {row['speedup_excl_overhead']:.2f} |"
        )

    return "\n".join(lines)


def write_summary(rows, environment, best, knee, path):
    lines = [
        "# Benchmark summary",
        "",
        "## Environment",
        "",
    ]
    lines += [f"- **{key}:** {value}" for key, value in environment.items()]
    lines += [
        "",
        "## Results",
        "",
        "Speedup S = T1 / Tn, where T is the mean total processing time "
        "(wall clock, threads started to all threads joined).",
        "",
        "Startup overhead is the median time of a run over an empty image "
        "folder (thread start, one Tesseract engine per worker, sentinels, "
        "joins). Speedup excl. overhead = (T1 - O1) / (Tn - On).",
        "",
        results_table(rows),
        "",
        "## Optimal configuration",
        "",
        f"- Highest speedup: **{best['n_workers']} workers** "
        f"(S = {best['speedup']:.2f}, {best['mean_ms']:.1f} ms).",
        f"- Smallest worker count within {KNEE_TOLERANCE:.0%} of the best: "
        f"**{knee['n_workers']} workers** (S = {knee['speedup']:.2f}).",
        "",
        "Every run produced exactly the same extracted text as the "
        "1-worker run (checked automatically).",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Benchmark the OCR pipeline")
    parser.add_argument("--config", default="config.txt")
    parser.add_argument("--workers", type=int, nargs="+", default=DEFAULT_WORKERS)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    args = parser.parse_args()

    if args.runs < 1 or any(n < 1 for n in args.workers):
        parser.error("--runs and every --workers value must be at least 1")

    worker_counts = sorted(set(args.workers))

    if worker_counts[0] != 1:
        worker_counts.insert(0, 1)  # T1 is needed for the speedup formula

    logging.basicConfig(level=logging.WARNING)

    config = load_config(args.config)
    environment = {
        "CPU": cpu_name(),
        "Logical CPUs": os.cpu_count(),
        "OS": platform.platform(),
        "Python": platform.python_version(),
        "OCR engine": check_ocr_setup(),
        "Images": str(config["image_dir"]),
        "Timed runs per configuration": args.runs,
    }

    for key, value in environment.items():
        print(f"{key}: {value}")

    samples = run_benchmark(config["image_dir"], worker_counts, args.runs)
    print("Measuring startup overhead (empty image folder)...")
    overhead = measure_overhead(worker_counts, args.runs)
    rows = summarize(samples, overhead)
    best, knee = choose_optimal(rows)

    write_results(rows, "benchmark_results.csv")
    write_runs(samples, "benchmark_runs.csv")
    write_summary(rows, environment, best, knee, "benchmark_summary.md")

    print()
    print(results_table(rows))
    print()
    print(f"Highest speedup: {best['n_workers']} workers (S = {best['speedup']:.2f})")
    print(
        f"Smallest count within {KNEE_TOLERANCE:.0%} of best: "
        f"{knee['n_workers']} workers (S = {knee['speedup']:.2f})"
    )
    print("Wrote benchmark_results.csv, benchmark_runs.csv, benchmark_summary.md")


if __name__ == "__main__":
    main()
