"""Tesseract wrappers tuned for the HUD's light-on-dark text."""

import hashlib

import cv2
import numpy as np
import pytesseract

DIGITS = "0123456789"
CLOCK_CHARS = "0123456789:-"


def preprocess(region: np.ndarray, scale: int = 3, threshold: int | None = None) -> np.ndarray:
    """Grayscale, upscale, binarise and invert to black text on white.

    With threshold=None Otsu picks the cutoff, which works for a single line
    of text on a flat background. Pass an explicit threshold when dim (dead)
    text must be kept or a coloured background must be dropped.
    """
    gray = region if region.ndim == 2 else cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    big = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if threshold is None:
        _, bw = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, bw = cv2.threshold(big, threshold, 255, cv2.THRESH_BINARY)
    bw = cv2.bitwise_not(bw)
    return cv2.copyMakeBorder(bw, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)


def read_line(
    region: np.ndarray,
    whitelist: str | None = None,
    threshold: int | None = None,
    timeout: float = 30.0,
) -> str:
    img = preprocess(region, threshold=threshold)
    extra = f" -c tessedit_char_whitelist={whitelist}" if whitelist else ""
    text = pytesseract.image_to_string(img, config="--psm 7 --oem 1" + extra, timeout=timeout).strip()
    if not text and has_text(region):
        # Line mode sometimes returns nothing for a short word ("Naga").
        text = pytesseract.image_to_string(img, config="--psm 8 --oem 1" + extra, timeout=timeout).strip()
    return text


def read_int(region: np.ndarray, threshold: int | None = None) -> int | None:
    text = read_line(region, whitelist=DIGITS, threshold=threshold)
    return int(text) if text.isdigit() else None


def crop_hash(region: np.ndarray) -> str:
    """Hash of a coarsened crop, stable across compression noise."""
    gray = region if region.ndim == 2 else cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (max(1, gray.shape[1] // 4), max(1, gray.shape[0] // 4)))
    return hashlib.blake2b((small > 128).tobytes(), digest_size=8).hexdigest()


def has_text(region: np.ndarray, min_brightness: int = 110, min_pixels: int = 8) -> bool:
    """True when the crop contains enough bright (text) pixels."""
    gray = region if region.ndim == 2 else cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    return int((gray > min_brightness).sum()) >= min_pixels
