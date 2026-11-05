"""
Tesseract OCR inference through tesserocr (in-process Tesseract C++ API).

Design notes
------------
* Each OCR worker thread creates its OWN engine with ``create_engine()`` and
  uses it for every image it processes. Engines are never shared between
  threads, so no lock is needed around inference (thread confinement).
* tesserocr releases the GIL while Tesseract recognizes an image, so several
  worker threads run OCR truly in parallel on different CPU cores.
* The engine stays loaded for the worker's lifetime, so the language model is
  read once per worker instead of once per image, and no external process is
  started per image (pytesseract launches tesseract.exe for every call).
* The provided ``tessdata/eng.traineddata`` model is used.
"""

import os

# Each Tesseract engine should use a single core: parallelism comes from our
# worker threads, not from OpenMP threads inside Tesseract. This must be set
# before the Tesseract library is loaded.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

from pathlib import Path  # noqa: E402

import tesserocr  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent
TESSDATA_DIR = PROJECT_DIR / "tessdata"
LANGUAGE = "eng"

# Every image contains a single word, so treat it as one line of text.
PAGE_SEGMENTATION_MODE = tesserocr.PSM.SINGLE_LINE
# The provided eng.traineddata is an LSTM model.
ENGINE_MODE = tesserocr.OEM.LSTM_ONLY


def check_ocr_setup():
    """
    Verify the language model is present; return the Tesseract version.

    Called once by the main thread before any worker starts, so a missing
    model is reported immediately instead of by every worker.
    """
    model_path = TESSDATA_DIR / f"{LANGUAGE}.traineddata"

    if not model_path.is_file():
        raise FileNotFoundError(f"Language model not found: {model_path}")

    return tesserocr.tesseract_version().splitlines()[0]


def create_engine():
    """Create a Tesseract engine. Call ``engine.End()`` when finished."""
    return tesserocr.PyTessBaseAPI(
        path=str(TESSDATA_DIR),
        lang=LANGUAGE,
        psm=PAGE_SEGMENTATION_MODE,
        oem=ENGINE_MODE,
    )


def clean_text(raw_text):
    """
    Normalize Tesseract output into a single line.

    Whitespace and newlines are collapsed so the text fits in one CSV cell,
    and tokens with no letters or digits (e.g. "|", ";", "~" produced from
    leftover noise specks) are dropped.
    """
    tokens = raw_text.split()
    return " ".join(
        token for token in tokens if any(char.isalnum() for char in token)
    )


def recognize(engine, image):
    """Run OCR on a PIL image with this thread's engine; return clean text."""
    engine.SetImage(image)
    return clean_text(engine.GetUTF8Text())
