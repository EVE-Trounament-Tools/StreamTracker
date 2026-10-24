"""Pixel-only checks: is the match HUD on screen, and which timer phase."""

import re
from enum import Enum
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from . import glyphs, layout as L, ocr

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
ANCHORS_FILE = DATA_DIR / "anchors.npz"
ANCHOR_BOXES = {
    "red_bans_label": L.RED_BANS_LABEL,
    "blue_bans_label": L.BLUE_BANS_LABEL,
    "battlefeed_label": L.BATTLEFEED_LABEL,
}
ANCHOR_MIN_SCORE = 0.7
ANCHORS_REQUIRED = 2
RED_RATIO = 0.4  # green/red of bright timer pixels; red TiDi clock is ~0.05
YELLOW_RATIO = 0.8  # blue/red of bright timer pixels; white is ~1, yellow 0.2-0.6


class Phase(str, Enum):
    NONE = "none"  # HUD not on screen
    PRE = "pre"  # HUD up, timer shows -00:00-
    COUNTDOWN = "countdown"  # yellow 0:30 -> 0:00
    LIVE = "live"  # white 10:00 -> 0:00
    UNKNOWN = "unknown"  # HUD up but timer unreadable


def _gray(region: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(region, cv2.COLOR_BGR2GRAY).astype(np.float32)


def save_anchors(frame: np.ndarray, path: Path = ANCHORS_FILE) -> None:
    """Store anchor crops from a frame known to show the HUD."""
    np.savez(path, **{k: _gray(L.crop(frame, b)) for k, b in ANCHOR_BOXES.items()})
    load_anchors.cache_clear()


@lru_cache(maxsize=1)
def load_anchors(path: Path = ANCHORS_FILE) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {k: data[k] for k in data.files}


def anchor_scores(frame: np.ndarray) -> dict[str, float]:
    """Normalized correlation of each anchor against the same spot in frame.

    The template is searched within a few pixels of its box so a small
    overlay shift doesn't break detection.
    """
    out = {}
    pad = 4
    for k, tmpl in load_anchors().items():
        x1, y1, x2, y2 = ANCHOR_BOXES[k]
        region = _gray(frame[max(0, y1 - pad) : y2 + pad, max(0, x1 - pad) : x2 + pad])
        res = cv2.matchTemplate(region, tmpl, cv2.TM_CCOEFF_NORMED)
        out[k] = float(np.nan_to_num(res).max())
    return out


def hud_present(frame: np.ndarray) -> bool:
    scores = anchor_scores(frame)
    return sum(s >= ANCHOR_MIN_SCORE for s in scores.values()) >= ANCHORS_REQUIRED


def timer_color(frame: np.ndarray) -> str:
    """'white' (match clock), 'yellow' (countdown) or 'red' (TiDi overtime clock)."""
    region = L.crop(frame, L.TIMER).astype(np.float32)
    lum = region.max(axis=2)
    text = lum > np.percentile(lum, 97)
    if text.sum() == 0:
        return "white"
    b, g, r = (region[text][:, c].mean() for c in range(3))
    if r <= 0:
        return "white"
    if g / r < RED_RATIO:
        return "red"
    if b / r < YELLOW_RATIO:
        return "yellow"
    return "white"


def parse_clock(text: str | None) -> int | None:
    """'9:44' -> 584 seconds; None for anything else."""
    if not text or text.count(":") != 1:
        return None
    m, s = text.split(":")
    if not (m.isdigit() and s.isdigit() and len(s) == 2):
        return None
    return int(m) * 60 + int(s)


def read_tidi_clock(redness: np.ndarray) -> int | None:
    """Remaining game time from the red "m:ss TiDi:nnn%" clock.

    This font isn't covered by the glyph templates and only shows during
    overtime, so tesseract reads it. Like elsewhere it reads slashed zeros as
    8; an 8 can't be a minute (overtime is short) or a tens-of-seconds digit,
    so those become 0. A trailing 8 stays ambiguous; the tracker resolves it
    from the previous reading.
    """
    text = ocr.read_line(redness, whitelist="0123456789:TiD%")
    m = re.match(r"(\d)(?::)?(\d)(\d)", text)
    if not m:
        return None
    minutes, tens, units = (int(g) for g in m.groups())
    minutes = 0 if minutes == 8 else minutes
    tens = 0 if tens >= 6 else tens
    return minutes * 60 + tens * 10 + units


def read_timer(frame: np.ndarray) -> tuple[Phase, int | None, bool]:
    """Phase, seconds shown on the clock, and whether it is the TiDi clock.

    Under heavy time dilation the white real-time clock runs out before the
    match does; the HUD then shows the remaining game time in red, followed by
    the TiDi factor ("3:55 TiDi:225%"). Assumes the HUD is present.
    """
    color = timer_color(frame)
    region = L.crop(frame, L.TIMER)
    if color == "red":
        # Red text on a dark red background: isolate it by redness.
        r = region[:, :, 2].astype(np.int16)
        gb = np.maximum(region[:, :, 0], region[:, :, 1]).astype(np.int16)
        redness = np.clip(r - gb, 0, 255).astype(np.uint8)
        clock = read_tidi_clock(redness)
        return (Phase.LIVE, clock, True) if clock is not None else (Phase.UNKNOWN, None, True)
    text = glyphs.read(region, "header")
    if text and text.startswith("-") and text.endswith("-"):
        return Phase.PRE, None, False
    clock = parse_clock(text)
    if clock is None:
        return Phase.UNKNOWN, None, False
    if color == "yellow":
        return Phase.COUNTDOWN, clock, False
    return Phase.LIVE, clock, False
