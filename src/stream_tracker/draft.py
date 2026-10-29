"""The pre-match draft panel: bans in the order they were made.

The HUD's "Ship Bans" lists aren't in pick order. Before the match the
broadcast shows a panel along the bottom of the screen with one card per ban,
left to right in pick order: a magenta (red team) or cyan (blue team) bar with
the time the pick took, the ship's icon, its name and class. Left of the cards
are "RED TEAM <name>" and "BLUE TEAM <name>".

find_panel() is pixel-only and cheap; read_panel() runs tesseract.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from . import layout as L, ocr

BAR_MIN_H, BAR_MAX_H = 28, 48
BAR_MIN_W = 100
BAR_FILL = 0.8  # share of a bar's bounding box the colour covers (the time text is a hole)
BAR_TOP_TOLERANCE = 3
MIN_BARS = 2
LABEL_ROW_MIN = 3  # coloured pixels in a row of the "RED TEAM" / "BLUE TEAM" label
LABEL_MIN_H, LABEL_MAX_H = 6, 22
# The labels and team names are centred as a block beside the cards, so they
# move up when a name wraps onto two lines; they are found by the labels'
# colour, searched this far above / below the bars' top.
LABEL_SEARCH_Y = (-30, 200)
TEAM_MAX_H = 90  # a team name, up to two lines, below its label
NAME_Y = (95, 165)  # ship name, below the icon, from the bars' bottom edge
NAME_MIN = 230  # the name is white; the class line under it is ~200
TEAM_MIN = 150
LINE_MIN_H = 8
NAME_OCR_THRESHOLD = 160  # fixed for ship names: Otsu reads "Rapier" as "Rapler" (team names read better with Otsu)


@dataclass
class Panel:
    bars: list[tuple[int, int, str]]  # (x1, x2, side) left to right
    top: int
    bottom: int
    labels: dict[str, tuple[int, int]]  # side -> rows of its "RED TEAM" / "BLUE TEAM" label


@dataclass
class DraftRead:
    teams: dict[str, str]
    bans: list[tuple[str, str]]  # (side, raw text) in pick order


def _masks(region: np.ndarray) -> dict[str, np.ndarray]:
    b, g, r = (region[:, :, c].astype(np.int16) for c in range(3))
    magenta = (r > 140) & (r - g > 90) & (b > 50) & (b < 170) & (r - b > 40)
    cyan = (b > 190) & (g > 120) & (g < 215) & (r < 120) & (b - r > 100)
    return {"red": magenta, "blue": cyan}


def find_panel(frame: np.ndarray) -> Panel | None:
    x0, y0, _, _ = L.DRAFT_BAR_SEARCH
    bars = []
    for side, mask in _masks(L.crop(frame, L.DRAFT_BAR_SEARCH)).items():
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 9), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        for x, y, w, h, area in stats[1:n]:
            if BAR_MIN_H <= h <= BAR_MAX_H and w >= BAR_MIN_W and area >= BAR_FILL * w * h:
                bars.append((int(x + x0), int(x + x0 + w), int(y + y0), int(y + y0 + h), side))
    if len(bars) < MIN_BARS:
        return None
    top = int(np.median([b[2] for b in bars]))
    bars = [b for b in bars if abs(b[2] - top) <= BAR_TOP_TOLERANCE]
    if len(bars) < MIN_BARS:
        return None
    bottom = int(np.median([b[3] for b in bars]))
    # Both team labels must be there: keeps other graphics with coloured boxes out.
    y0 = max(0, top + LABEL_SEARCH_Y[0])
    region = frame[y0 : top + LABEL_SEARCH_Y[1], L.DRAFT_LABEL_X[0] : L.DRAFT_LABEL_X[1]]
    labels = {}
    for side, mask in _masks(region).items():
        runs = [(a, b) for a, b in _runs(mask.sum(axis=1) >= LABEL_ROW_MIN)
                if LABEL_MIN_H <= b - a <= LABEL_MAX_H]
        if not runs:
            return None
        a, b = runs[0]
        labels[side] = (y0 + a, y0 + b)
    if labels["red"][0] >= labels["blue"][0]:
        return None
    return Panel(sorted((x1, x2, side) for x1, x2, _, _, side in bars), top, bottom, labels)


def _runs(rows) -> list[tuple[int, int]]:
    """(start, end) of each run of True rows."""
    runs, start = [], None
    for i, on in enumerate(list(rows) + [False]):
        if on and start is None:
            start = i
        elif not on and start is not None:
            runs.append((start, i))
            start = None
    return runs


def _read_lines(region: np.ndarray, threshold: int, first_only: bool = False,
                ocr_threshold: int | None = None) -> str:
    """OCR each line of white text (all channels above threshold, so the
    coloured labels drop out), top to bottom."""
    white = region.min(axis=2) > threshold
    runs = [(a, b) for a, b in _runs(white.sum(axis=1) >= 2) if b - a >= LINE_MIN_H]
    if first_only:
        runs = runs[:1]
    return " ".join(ocr.read_line(region[max(0, a - 3) : b + 3], threshold=ocr_threshold)
                    for a, b in runs).strip()


def read_panel(frame: np.ndarray, panel: Panel) -> DraftRead:
    teams = {}
    ends = {"red": panel.labels["blue"][0], "blue": panel.labels["blue"][1] + TEAM_MAX_H}
    for side in ("red", "blue"):
        y1 = panel.labels[side][1]
        region = frame[y1 : min(ends[side], y1 + TEAM_MAX_H), L.DRAFT_TEAM_X[0] : L.DRAFT_TEAM_X[1]]
        teams[side] = _read_lines(region, TEAM_MIN)
    bans = []
    for x1, x2, side in panel.bars:
        region = frame[panel.bottom + NAME_Y[0] : panel.bottom + NAME_Y[1], x1:x2]
        bans.append((side, _read_lines(region, NAME_MIN, first_only=True, ocr_threshold=NAME_OCR_THRESHOLD)))
    return DraftRead(teams, bans)
