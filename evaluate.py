"""
Measure OCR accuracy of ocr_results.csv against labels.txt.

Character accuracy = 1 - (total edit distance / total label characters),
where edit distance is the Levenshtein distance between the label and the
extracted text of each image. The project requires at least 50%.

Usage:
    python evaluate.py [--results ocr_results.csv] [--labels labels.txt]
"""

import argparse
import csv
import sys


def levenshtein(a, b):
    """Minimum number of single-character edits to turn ``a`` into ``b``."""
    previous = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):
        current = [i]

        for j, char_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + (char_a != char_b),  # substitution
                )
            )

        previous = current

    return previous[-1]


def read_column_map(path, key, value):
    with open(path, newline="", encoding="utf-8") as csv_file:
        return {row[key]: row[value] for row in csv.DictReader(csv_file)}


def evaluate(results_path, labels_path):
    labels = read_column_map(labels_path, "filename", "word")
    predictions = read_column_map(results_path, "filename", "text")

    total_chars = sum(len(word) for word in labels.values())
    distance = 0
    distance_ignore_case = 0
    exact = 0
    missing = []
    mismatches = []

    for filename, word in sorted(labels.items()):
        if filename not in predictions:
            missing.append(filename)

        text = predictions.get(filename, "")
        distance += levenshtein(word, text)
        distance_ignore_case += levenshtein(word.lower(), text.lower())

        if text == word:
            exact += 1
        else:
            mismatches.append((filename, word, text))

    return {
        "images": len(labels),
        "total_chars": total_chars,
        "char_accuracy": max(0.0, 1 - distance / total_chars),
        "char_accuracy_ignore_case": max(
            0.0, 1 - distance_ignore_case / total_chars
        ),
        "exact_words": exact,
        "missing": missing,
        "mismatches": mismatches,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--results", default="ocr_results.csv")
    parser.add_argument("--labels", default="labels.txt")
    parser.add_argument(
        "--show-mismatches", action="store_true", help="list every wrong word"
    )
    args = parser.parse_args()

    report = evaluate(args.results, args.labels)

    print(f"Images                        : {report['images']}")
    print(f"Label characters              : {report['total_chars']}")
    print(f"Character accuracy            : {report['char_accuracy']:.2%}")
    print(
        "Character accuracy (any case) : "
        f"{report['char_accuracy_ignore_case']:.2%}"
    )
    print(
        "Exact word matches            : "
        f"{report['exact_words']}/{report['images']}"
    )

    if report["missing"]:
        print(f"Missing from results          : {', '.join(report['missing'])}")

    if args.show_mismatches:
        print()
        for filename, word, text in report["mismatches"]:
            print(f"  {filename}: expected {word!r}, got {text!r}")

    passed = report["char_accuracy"] >= 0.5
    print()
    print("PASS: at least 50% of characters recognized." if passed else
          "FAIL: fewer than 50% of characters recognized.")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
