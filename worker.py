import logging
import threading
import time
from dataclasses import dataclass

from loader import SENTINEL
from ocr import create_engine, recognize
from preprocess import preprocess

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    filename: str
    text: str
    time_ms: float
    worker: str


def process_task(engine, task):
    """Preprocess and OCR one image; return (text, elapsed milliseconds)."""
    start = time.perf_counter()
    text = recognize(engine, preprocess(task.image))
    elapsed_ms = (time.perf_counter() - start) * 1000
    return text, elapsed_ms


def ocr_worker(image_queue, on_result):
    """
    OCR worker: take images from the shared queue until a SENTINEL arrives.

    Each worker owns a private Tesseract engine (never shared, so no lock is
    needed for inference). Each image is preprocessed, recognized, timed, and
    handed to ``on_result``.

    The worker must keep consuming until it receives its SENTINEL; otherwise
    the loader could block forever on the bounded queue. So failures are
    contained: if the engine cannot be created, or one image fails, the error
    is logged and an empty-text result is recorded. ``task_done()`` is always
    called, so ``Queue.join()`` in the main thread cannot hang.
    """
    worker_name = threading.current_thread().name

    try:
        engine = create_engine()
    except Exception as error:
        logger.error("could not start OCR engine: %s", error)
        engine = None

    try:
        while True:
            task = image_queue.get()

            try:
                if task is SENTINEL:
                    logger.info("received sentinel; shutting down.")
                    break

                text, elapsed_ms = "", 0.0

                if engine is not None:
                    try:
                        text, elapsed_ms = process_task(engine, task)
                    except Exception as error:
                        logger.error(
                            "OCR failed for %s: %s", task.filename, error
                        )

                on_result(
                    OCRResult(
                        filename=task.filename,
                        text=text,
                        time_ms=round(elapsed_ms, 2),
                        worker=worker_name,
                    )
                )
                logger.info(
                    "%s -> %r (%.2f ms)", task.filename, text, elapsed_ms
                )

            finally:
                image_queue.task_done()

    finally:
        if engine is not None:
            engine.End()
