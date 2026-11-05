# Parallel OCR Pipeline

CSC611M Machine Project 1. One loader thread reads images into a shared
queue, and `n_workers` OCR threads take images from the queue, preprocess
them, and run Tesseract OCR in parallel.

```text
                     +--> OCR worker 1 (preprocess + OCR) --+
Loader --> [ Queue ] +--> OCR worker 2 (preprocess + OCR) --+--> results
                     +--> OCR worker N (preprocess + OCR) --+
```

## Setup

Requires Python 3.10+.

```bash
python -m pip install -r requirements.txt
```

`tesserocr` ships prebuilt wheels that bundle Tesseract, so no separate
Tesseract install is needed on Windows. The provided language model is in
`tessdata/eng.traineddata` and is used automatically.

## Configuration

`config.txt`:

```text
image_dir=images
n_workers=2
```

- `image_dir`: folder of images to process (relative to where you run the program).
- `n_workers`: number of OCR workers. The loader is a separate thread and is not counted.

## Run

From this directory:

```bash
python main.py
```

Options: `--config <file>` (default `config.txt`), `--output <csv>` (default
`ocr_results.csv`), `--quiet` (log warnings and errors only).

The program exits once every image has been processed and every thread has
finished, then prints a summary: images processed, total wall-clock time,
and images handled by each worker.

## Output

`ocr_results.csv`:

| Column | Meaning |
| --- | --- |
| `id` | Sequential integer, 1..N, in filename order |
| `filename` | Image file name |
| `text` | Extracted text |
| `processing_time_ms` | Time this worker spent preprocessing + recognizing the image, in milliseconds |

## Accuracy

```bash
python evaluate.py
```

This compares `ocr_results.csv` with `labels.txt`. Character accuracy is
`1 - (sum of Levenshtein distances / total label characters)`. On the
provided dataset it is **94.15%** (77/100 exact words), against a
requirement of 50%.

## Modules

| File | Role |
| --- | --- |
| `config.py` | Reads and validates `config.txt`. |
| `loader.py` | Loader thread: decodes every image and puts it on the queue, then one sentinel per worker. |
| `worker.py` | OCR worker thread: takes images from the queue until its sentinel, preprocesses, runs OCR, and times each image. |
| `preprocess.py` | Grayscale, 3x3 median filter, 3x3 morphological opening, Otsu binarization, white border (numpy + Pillow). |
| `ocr.py` | Per-worker Tesseract engine (tesserocr) using `tessdata/eng.traineddata`, single-line mode. |
| `results.py` | Lock-protected `ResultStore` shared by all workers, and the CSV writer. |
| `main.py` | `run_pipeline()`: starts the workers and the loader, waits for them all, then writes the CSV. |
| `evaluate.py` | Character accuracy of the CSV against `labels.txt`. |
| `benchmark.py` | Runs the pipeline with different worker counts and computes speedup. |

## Synchronization

| Shared resource | Accessed by | Protection |
| --- | --- | --- |
| Image queue | Loader (put), workers (get, task_done), main (join) | `queue.Queue`, which uses an internal mutex and `not_empty`/`not_full` condition variables. Bounded to `2 * n_workers`. |
| Result list + per-worker counts (`ResultStore`) | All workers (add), main (snapshot) | `threading.Lock` around every read and write. |
| Loader statistics | Loader (write), main (read) | Read only after `loader.join()`. |
| Tesseract engine | Exactly one worker each | Not shared (thread confinement), so no lock is needed. |
| Log output | All threads | `logging`, whose handlers lock internally. |

Shutdown order in `run_pipeline()`:

1. `loader.join()`: every image and one sentinel per worker are queued.
2. `image_queue.join()`: every queued item has had `task_done()` called.
3. `worker.join()` for each worker: every worker saw its sentinel and exited.
4. The main thread takes a snapshot of the results and writes the CSV.
   No worker is alive by then, so the file is written exactly once with
   complete data.

## Benchmarks

```bash
python benchmark.py                          # 1,2,3,4,5,6,8,10,12,16 workers, 10 runs each
python benchmark.py --workers 1 2 4 8 --runs 5
```

Each worker count is run 10 times, round-robin, after one discarded warm-up
run. Speedup is `S = T1 / Tn` using the mean total time. Every run's extracted
text is checked against the 1-worker run, and the benchmark stops if any
output differs. Outputs:

- `benchmark_results.csv`: one row per worker count.
- `benchmark_runs.csv`: every timed run, with images handled per worker.
- `benchmark_summary.md`: results table and test environment.

Results on an AMD Ryzen 5 5600G (6 cores / 12 threads), Windows 11,
Python 3.14, 100 images:

| Workers | Mean time (ms) | Speedup S | Efficiency S/n | Startup overhead (ms) | Speedup excl. overhead |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 583.4 | 1.00 | 1.00 | 26.8 | 1.00 |
| 2 | 322.3 | 1.81 | 0.91 | 27.4 | 1.89 |
| 3 | 244.0 | 2.39 | 0.80 | 32.6 | 2.63 |
| 4 | 215.4 | 2.71 | 0.68 | 39.5 | 3.16 |
| 5 | 198.6 | 2.94 | 0.59 | 42.8 | 3.57 |
| **6** | **192.4** | **3.03** | 0.51 | 44.6 | 3.77 |
| 8 | 194.2 | 3.00 | 0.38 | 55.0 | 4.00 |
| 10 | 203.7 | 2.86 | 0.29 | 67.6 | 4.09 |
| 12 | 216.7 | 2.69 | 0.22 | 78.9 | 4.04 |
| 16 | 245.4 | 2.38 | 0.15 | 101.0 | 3.85 |

**Optimal configuration: 6 workers** (S = 3.03), which matches the CPU's
physical core count. 8 workers performs the same within noise, and
anything above that is slower.

What limits the speedup:

- **Fixed startup cost grows with workers.** Each worker creates its own
  Tesseract engine (about 27 ms), and engine creation is partly serialized.
  A run over an empty folder takes 27 ms with 1 worker and 101 ms with 16.
  With that overhead removed, speedup reaches about 4.0 at 8–12 workers.
- **Per-image work gets slower under load.** Mean per-image time rises from
  5.5 ms (1 worker) to 8.2 ms (6 workers). Workers share caches and memory
  bandwidth, the CPU lowers its boost clock when every core is busy, and
  past 6 workers two threads share each physical core (SMT).
- **The GIL serializes the Python parts.** OCR (tesserocr) and the numpy
  filters release the GIL, but the Python code around them (queue
  operations, the Otsu loop, building results) runs one thread at a time.
- **Oversubscription.** At more than 12 workers there are more threads than
  hardware threads, so context switching adds cost without adding capacity.
- **The loader is not the limit yet.** At 6 workers it spends about 125 ms
  of a 192 ms run decoding images, so it would become the next bottleneck.

Work is spread evenly by the shared queue: whichever worker is free takes
the next image. `benchmark_runs.csv` records how many images each worker
handled in every run.

Pillow's `MedianFilter`/`MaxFilter`/`MinFilter` hold the GIL. With them,
preprocessing (about 45% of each image's work) ran one image at a time
across all threads and capped the speedup near 1.8. `preprocess.py`
therefore implements the same filters with numpy. The output is
pixel-identical, so accuracy is unchanged, and the filters run about 15x
faster.

## Design notes

- **Shared queue.** `queue.Queue(maxsize=2 * n_workers)` is the only channel
  between the loader and the workers. It is bounded, so the loader blocks when
  the workers fall behind and memory use stays flat.
- **Termination.** The loader puts one `None` sentinel per worker in a
  `finally` block, so workers are always told to stop. Each worker always calls
  `task_done()`. The main thread waits on `loader.join()`, then
  `queue.join()`, then each `worker.join()`.
- **No early worker exit.** A worker that hits an error on one image logs it,
  records empty text, and keeps going until its sentinel arrives. This keeps
  the loader from blocking forever on a full queue.
- **One OCR engine per worker.** Tesseract engines are never shared between
  threads, so inference needs no lock. tesserocr releases the GIL while
  Tesseract runs, so the workers recognize images on separate cores at the
  same time. Each engine loads the model once and is reused for every image
  that worker processes.
- **Why tesserocr instead of pytesseract.** pytesseract starts a new
  `tesseract.exe` process for every image. On our Windows test machine,
  antivirus scanning made about every other launch block for about 1 s, so
  timings measured scan overhead rather than OCR. tesserocr runs Tesseract
  in-process: about 4 ms of OCR per image, with the same accuracy.
