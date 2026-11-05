import logging
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

# Termination signal: one is queued per OCR worker after all images.
SENTINEL = None

# Every image extension
SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass
class ImageTask:
    """A loaded image waiting in the shared queue for an OCR worker."""

    filename: str
    image: Image.Image


def discover_images(image_dir):
    """Return the supported image files in ``image_dir``, sorted by name."""
    return sorted(
        path
        for path in Path(image_dir).iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def read_image(image_path):
    """Open and fully decode an image so the file handle is released."""
    with Image.open(image_path) as image:
        image.load()
        return image.copy()


def load_images(image_dir, image_queue, n_workers, stats=None):
    """
    Loader worker: read every image from disk and put it on the shared queue.

    ``image_queue`` is bounded, so ``put`` blocks while the workers are busy,
    which keeps memory use flat. One SENTINEL per worker is always queued in
    ``finally`` so every worker is told to stop, even if loading fails.

    ``stats`` (optional dict) is written only by this thread; the main thread
    reads it after ``join()``-ing the loader.
    """
    queued = 0
    failed = []
    read_seconds = 0.0

    try:
        for image_path in discover_images(image_dir):
            start = time.perf_counter()

            try:
                image = read_image(image_path)
            except Exception as error:
                failed.append(image_path.name)
                logger.warning("could not load %s: %s", image_path.name, error)
                continue
            finally:
                read_seconds += time.perf_counter() - start

            image_queue.put(ImageTask(image_path.name, image))
            queued += 1

        logger.info("queued %d image(s).", queued)

    finally:
        for _ in range(n_workers):
            image_queue.put(SENTINEL)

        logger.info("sent %d termination signal(s).", n_workers)

        if stats is not None:
            stats["queued"] = queued
            stats["failed"] = failed
            stats["read_ms"] = read_seconds * 1000
