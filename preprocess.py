"""
Image preprocessing applied by each OCR worker before inference.

The dataset consists of dark single words on a light background with
salt-and-pepper speckle noise. The steps below remove that noise and give
Tesseract a clean, high-contrast, binarized image:

1. Flatten transparency onto white and convert to grayscale.
2. 3x3 median filter to remove isolated 1-pixel speckles.
3. 3x3 grayscale morphological opening (max then min) to remove the
   remaining small dark specks while keeping letter strokes intact.
4. Binarize with an Otsu threshold computed from the image histogram.
5. Pad with a white border (Tesseract struggles with text touching edges).

These settings were chosen by measuring character accuracy against
labels.txt for several variants. Larger median filters erode thin strokes,
and upscaling made results worse on this dataset.

Why numpy instead of Pillow's MedianFilter/MaxFilter/MinFilter: Pillow's
rank filters hold the GIL, so they ran one at a time no matter how many
worker threads existed (about 45% of the per-image work). The numpy versions
below produce byte-identical results about 15x faster, which shrinks the
serialized part of each worker's job and lets threads scale.

Every function is pure (no shared state), so it is safe to call from many
worker threads at once.
"""

import numpy as np
from PIL import Image, ImageOps

BORDER_PX = 10

# Compare-exchange network that leaves the median of 9 values at index 4
# (Paeth's median-of-9, 19 exchanges).
MEDIAN9_NETWORK = [
    (1, 2), (4, 5), (7, 8), (0, 1), (3, 4), (6, 7), (1, 2), (4, 5), (7, 8),
    (0, 3), (5, 8), (4, 7), (3, 6), (1, 4), (2, 5), (4, 7), (4, 2), (6, 4),
    (4, 2),
]


def to_grayscale(image):
    """Composite any alpha channel onto white, then convert to 8-bit gray."""
    if image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, rgba)

    return image.convert("L")


def neighborhood(pixels):
    """
    Return the 9 shifted views of ``pixels`` that make up each pixel's 3x3
    neighborhood. Edges are padded by repeating the border pixels.
    """
    padded = np.pad(pixels, 1, mode="edge")
    height, width = pixels.shape
    return [
        padded[dy:dy + height, dx:dx + width]
        for dy in range(3)
        for dx in range(3)
    ]


def median3(pixels):
    """3x3 median filter."""
    values = [view.copy() for view in neighborhood(pixels)]

    for i, j in MEDIAN9_NETWORK:
        low = np.minimum(values[i], values[j])
        np.maximum(values[i], values[j], out=values[j])
        values[i] = low

    return values[4]


def max3(pixels):
    """3x3 maximum filter (grayscale dilation of the light background)."""
    views = neighborhood(pixels)
    result = views[0].copy()

    for view in views[1:]:
        np.maximum(result, view, out=result)

    return result


def min3(pixels):
    """3x3 minimum filter (grayscale erosion of the light background)."""
    views = neighborhood(pixels)
    result = views[0].copy()

    for view in views[1:]:
        np.minimum(result, view, out=result)

    return result


def remove_specks(pixels):
    """
    Morphological opening for dark-on-light images.

    The max filter spreads the light background over dark regions smaller
    than 3x3 (erasing them). The min filter then restores the surviving
    strokes to their original thickness.
    """
    return min3(max3(pixels))


def otsu_threshold(pixels):
    """Return the Otsu threshold (0-255) of a uint8 image array."""
    histogram = np.bincount(pixels.ravel(), minlength=256).tolist()
    total = sum(histogram)

    if total == 0:
        return 127

    sum_all = sum(level * count for level, count in enumerate(histogram))

    weight_background = 0
    sum_background = 0
    best_threshold = 0
    best_variance = -1.0

    for level, count in enumerate(histogram):
        weight_background += count

        if weight_background == 0:
            continue

        weight_foreground = total - weight_background

        if weight_foreground == 0:
            break

        sum_background += level * count
        mean_background = sum_background / weight_background
        mean_foreground = (sum_all - sum_background) / weight_foreground

        between_variance = (
            weight_background
            * weight_foreground
            * (mean_background - mean_foreground) ** 2
        )

        if between_variance > best_variance:
            best_variance = between_variance
            best_threshold = level

    return best_threshold


def binarize(pixels):
    """Map pixels above the Otsu threshold to white and the rest to black."""
    threshold = otsu_threshold(pixels)
    return np.where(pixels > threshold, 255, 0).astype(np.uint8)


def preprocess(image):
    """Return a cleaned, binarized copy of ``image`` ready for OCR."""
    pixels = np.asarray(to_grayscale(image))
    cleaned = remove_specks(median3(pixels))
    binary = Image.fromarray(binarize(cleaned))

    return ImageOps.expand(binary, border=BORDER_PX, fill=255)
