import argparse
import logging
import queue
import threading
import time

from config import load_config
from loader import load_images
from ocr import check_ocr_setup
from results import ResultStore, write_csv
from worker import ocr_worker

logger = logging.getLogger("pipeline")

DEFAULT_OUTPUT = "ocr_results.csv"


def run_pipeline(image_dir, n_workers, output_csv=DEFAULT_OUTPUT):
    """
    Run the loader and ``n_workers`` OCR workers over ``image_dir``.

    Returns a dict of run statistics. ``total_ms`` is the wall-clock time from
    starting the threads until every thread has finished (CSV writing is
    timed separately as ``csv_ms``).
    """
    # Bounded queue: the loader blocks when workers fall behind.
    image_queue = queue.Queue(maxsize=2 * n_workers)
    store = ResultStore()
    loader_stats = {}

    start = time.perf_counter()

    workers = []

    for worker_id in range(1, n_workers + 1):
        worker = threading.Thread(
            target=ocr_worker,
            args=(image_queue, store.add),
            name=f"OCR-Worker-{worker_id}",
        )
        worker.start()
        workers.append(worker)

    loader = threading.Thread(
        target=load_images,
        args=(image_dir, image_queue, n_workers, loader_stats),
        name="Image-Loader",
    )
    loader.start()

    # 1. The loader has queued every image and every sentinel.
    loader.join()

    # 2. Every queued item (images and sentinels) has been marked task_done().
    image_queue.join()

    # 3. Every worker thread has exited, so nothing can touch the store now.
    for worker in workers:
        worker.join()

    total_ms = (time.perf_counter() - start) * 1000

    results = store.snapshot()

    csv_start = time.perf_counter()
    rows = write_csv(results, output_csv)
    csv_ms = (time.perf_counter() - csv_start) * 1000

    return {
        "n_workers": n_workers,
        "images": rows,
        "queued": loader_stats.get("queued", 0),
        "load_failures": loader_stats.get("failed", []),
        "loader_read_ms": loader_stats.get("read_ms", 0.0),
        "total_ms": total_ms,
        "csv_ms": csv_ms,
        "sum_image_ms": sum(result.time_ms for result in results),
        "per_worker": store.per_worker_counts(),
        "results": results,
        "output_csv": str(output_csv),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Parallel OCR pipeline")
    parser.add_argument(
        "--config",
        default="config.txt",
        help="configuration file (default: config.txt)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"output CSV file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="only log warnings and errors",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s [%(threadName)s] %(message)s",
    )

    config = load_config(args.config)
    n_workers = config["n_workers"]

    version = check_ocr_setup()
    logger.info("using %s with %d OCR worker(s).", version, n_workers)

    stats = run_pipeline(config["image_dir"], n_workers, args.output)

    print()
    print(f"Images processed : {stats['images']}")
    print(f"OCR workers      : {n_workers}")
    print(f"Total time       : {stats['total_ms']:.2f} ms")
    print(f"Images per worker: {stats['per_worker']}")

    if stats["load_failures"]:
        print(f"Could not load   : {', '.join(stats['load_failures'])}")

    print(f"Results written to {stats['output_csv']}")


if __name__ == "__main__":
    main()
