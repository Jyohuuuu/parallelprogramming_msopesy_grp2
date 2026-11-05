# Benchmark summary

## Environment

- **CPU:** AMD Ryzen 5 5600G with Radeon Graphics
- **Logical CPUs:** 12
- **OS:** Windows-11-10.0.26200-SP0
- **Python:** 3.14.3
- **OCR engine:** tesseract 5.5.3
- **Images:** images
- **Timed runs per configuration:** 10

## Results

Speedup S = T1 / Tn, where T is the mean total processing time (wall clock, threads started to all threads joined).

Startup overhead is the median time of a run over an empty image folder (thread start, one Tesseract engine per worker, sentinels, joins). Speedup excl. overhead = (T1 - O1) / (Tn - On).

| Workers | Mean time (ms) | Std dev (ms) | Speedup S = T1/Tn | Efficiency S/n | Throughput (img/s) | Mean per-image time (ms) | Images per worker (min-max) | Startup overhead (ms) | Speedup excl. overhead |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 583.4 | 24.4 | 1.00 | 1.00 | 171 | 5.46 | 100-100 | 26.8 | 1.00 |
| 2 | 322.3 | 33.4 | 1.81 | 0.91 | 310 | 5.75 | 48-52 | 27.4 | 1.89 |
| 3 | 244.0 | 10.7 | 2.39 | 0.80 | 410 | 6.10 | 32-36 | 32.6 | 2.63 |
| 4 | 215.4 | 14.6 | 2.71 | 0.68 | 464 | 6.73 | 23-27 | 39.5 | 3.16 |
| 5 | 198.6 | 8.7 | 2.94 | 0.59 | 504 | 7.42 | 16-23 | 42.8 | 3.57 |
| 6 | 192.4 | 6.4 | 3.03 | 0.51 | 520 | 8.23 | 15-19 | 44.6 | 3.77 |
| 8 | 194.2 | 7.3 | 3.00 | 0.38 | 515 | 9.21 | 11-14 | 55.0 | 4.00 |
| 10 | 203.7 | 7.5 | 2.86 | 0.29 | 491 | 8.64 | 8-13 | 67.6 | 4.09 |
| 12 | 216.7 | 7.4 | 2.69 | 0.22 | 461 | 9.17 | 6-11 | 78.9 | 4.04 |
| 16 | 245.4 | 19.0 | 2.38 | 0.15 | 408 | 9.17 | 3-10 | 101.0 | 3.85 |

## Optimal configuration

- Highest speedup: **6 workers** (S = 3.03, 192.4 ms).
- Smallest worker count within 5% of the best: **5 workers** (S = 2.94).

Every run produced exactly the same extracted text as the 1-worker run (checked automatically).
