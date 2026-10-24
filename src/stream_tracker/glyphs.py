"""Template-matching reader for the HUD's fixed-font numbers.

Tesseract reads the HUD's slashed zero as 8, so digits (timer, scores, PTS)
are matched against glyph templates built by tools/build_glyphs.py instead.
"""

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

GLYPH_W, GLYPH_H = 12, 18
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


MIN_CONTRAST = 40


def to_gray(region: np.ndarray) -> np.ndarray:
    return region if region.ndim == 2 else cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)


def segment(region: np.ndarray, min_pixels: int = 3) -> list[tuple[int, np.ndarray]]:
    """Split a single line of text into glyphs by column projection.

    The cutoff sits halfway between background and the brightest text, so
    bright (alive) and dim (dead) rows give the same stroke shapes. Returns
    (x, glyph) pairs; glyph is the bounding box of one character as
    intensities scaled to 0..1. Characters in this font never touch, so empty
    columns are reliable separators.
    """
    gray = to_gray(region).astype(np.float32)
    bg = float(np.median(gray))
    peak = float(np.percentile(gray, 99.5))
    if peak - bg < MIN_CONTRAST:
        return []
    norm = np.clip((gray - bg) / (peak - bg), 0, 1)
    bw = norm > 0.5
    cols = bw.sum(axis=0)
    out = []
    x = 0
    w = len(cols)
    while x < w:
        if cols[x] == 0:
            x += 1
            continue
        start = x
        while x < w and cols[x] > 0:
            x += 1
        part = bw[:, start:x]
        rows = np.nonzero(part.sum(axis=1))[0]
        if part.sum() >= min_pixels:
            out.append((start, norm[rows[0] : rows[-1] + 1, start:x]))
    return out


def normalize(glyph: np.ndarray, line_height: int) -> np.ndarray:
    """Place a glyph in a fixed box, keeping its height relative to the line.

    Keeping relative height and vertical position apart from shape lets ':'
    and '-' be told from digits.
    """
    h, w = glyph.shape
    scale = GLYPH_H / max(line_height, h)
    nh, nw = max(1, round(h * scale)), max(1, min(GLYPH_W, round(w * scale)))
    small = cv2.resize(glyph.astype(np.float32), (nw, nh), interpolation=cv2.INTER_AREA)
    box = np.zeros((GLYPH_H, GLYPH_W), np.float32)
    y0 = (GLYPH_H - nh) // 2 if h < line_height * 0.6 else GLYPH_H - nh
    x0 = (GLYPH_W - nw) // 2
    box[y0 : y0 + nh, x0 : x0 + nw] = small
    return box


def line_height(glyphs: list[tuple[int, np.ndarray]]) -> int:
    return max((g.shape[0] for _, g in glyphs), default=1)


@lru_cache(maxsize=None)
def load_templates(font: str) -> tuple[list[str], np.ndarray]:
    data = np.load(DATA_DIR / f"glyphs_{font}.npz")
    return list(data["labels"]), data["templates"]


def classify(glyph: np.ndarray, font: str, allowed: str | None = None) -> tuple[str, float]:
    labels, templates = load_templates(font)
    dists = np.abs(templates.reshape(len(labels), -1) - glyph.reshape(-1)).mean(axis=1)
    if allowed is not None:
        dists = np.where([lab in allowed for lab in labels], dists, np.inf)
    i = int(np.argmin(dists))
    return labels[i], float(dists[i])


def read(
    region: np.ndarray, font: str, allowed: str | None = None, max_dist: float = 0.2,
    min_x: int = 0,
) -> str | None:
    """Read a line of digits/':'/'-'. None if any glyph matches poorly.

    Glyphs starting left of min_x are ignored (text overflowing from the
    neighbouring column).
    """
    glyphs = [(x, g) for x, g in segment(region) if x >= min_x]
    if not glyphs:
        return ""
    lh = line_height(glyphs)
    text = []
    for _, g in glyphs:
        label, dist = classify(normalize(g, lh), font, allowed)
        if dist > max_dist:
            return None
        text.append(label)
    return "".join(text)

