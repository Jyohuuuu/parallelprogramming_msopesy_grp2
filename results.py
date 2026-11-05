"""
Thread-safe result aggregation and CSV export.

All OCR workers add their results to one shared ``ResultStore``. The list and
the per-worker counters inside it are shared mutable state, so every access
goes through a single ``threading.Lock``. Without it, appends and the
read-modify-write ``count += 1`` could interleave between threads and lose
updates.
"""

import csv
import threading
from collections import Counter

CSV_HEADER = ["id", "filename", "text", "processing_time_ms"]


class ResultStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._results = []
        self._per_worker = Counter()

    def add(self, result):
        """Called concurrently by OCR workers."""
        with self._lock:
            self._results.append(result)
            self._per_worker[result.worker] += 1

    def snapshot(self):
        """Return a copy of all results taken under the lock."""
        with self._lock:
            return list(self._results)

    def per_worker_counts(self):
        """Return {worker name: images processed}, taken under the lock."""
        with self._lock:
            return dict(self._per_worker)

    def __len__(self):
        with self._lock:
            return len(self._results)


def write_csv(results, path):
    """
    Write results to ``path`` sorted by filename, with sequential IDs 1..N.

    Called by the main thread only after every worker has been joined, so no
    worker can still be adding results while the file is written.
    """
    ordered = sorted(results, key=lambda result: result.filename)

    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(CSV_HEADER)

        for row_id, result in enumerate(ordered, start=1):
            writer.writerow(
                [row_id, result.filename, result.text, f"{result.time_ms:.2f}"]
            )

    return len(ordered)
