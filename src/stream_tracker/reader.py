"""Turn a HUD frame into structured readings.

read_frame() is cheap (pixels + glyph templates, ~1ms) and runs every frame.
read_identity() uses tesseract for team, ban, ship and pilot names; it takes
seconds, so the tracker only calls it until those fields are settled.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from . import glyphs, hud, layout as L, ocr

SIDES = ("red", "blue")
DIGITS = "0123456789"

# Row brightness relative to the column headings (~238 normally, less when the
# HUD is drawn faded): alive text ~1.0, dead ~0.5, empty ~0.13.
ALIVE_MIN = 0.75
DEAD_MIN = 0.3
DEAD_MAX = 0.69
BARS_ALIVE_MIN = 0.55
BAN_PRESENT_MIN = 0.63
HEADINGS_MIN = 60  # below this the HUD is fading out and rows can't be judged


@dataclass
class RowState:
    slot: int
    status: str  # alive | dead | empty | unknown
    pts: int | None


@dataclass
class FrameReading:
    hud: bool
    phase: hud.Phase
    clock: int | None = None
    tidi: bool = False  # clock is the red TiDi game clock
    scores: dict[str, int | None] = field(default_factory=dict)
    rows: dict[str, list[RowState]] = field(default_factory=dict)
    # Sides missing here had the HUD too faded to count bans; 0 is a real count.
    ban_count: dict[str, int] = field(default_factory=dict)


@dataclass
class Identity:
    teams: dict[str, str]
    bans: dict[str, list[str]]
    ships: dict[str, list[str]]  # raw ship text per slot
    pilots: dict[str, list[str]]
    ban_count: dict[str, int | None]  # None: HUD too faded to tell how many bans


def _p99(region: np.ndarray) -> float:
    return float(np.percentile(glyphs.to_gray(region), 99))


def headings_level(frame: np.ndarray, side: str) -> float:
    box = L.RED_HEADINGS if side == "red" else L.BLUE_HEADINGS
    return float(np.percentile(glyphs.to_gray(L.crop(frame, box)), 99.5))


def row_status(frame: np.ndarray, side: str, i: int, level: float) -> str:
    """Alive rows are drawn as bright as the column headings; dead rows grey
    out to about half that and lose their health bars."""
    if level < HEADINGS_MIN:
        return "unknown"
    text = max(_p99(L.crop(frame, L.row_box(side, i, "ship"))),
               _p99(L.crop(frame, L.row_box(side, i, "name")))) / level
    bars = _p99(L.crop(frame, L.row_box(side, i, "bars"))) / level
    if text < DEAD_MIN:
        return "empty"
    if text >= ALIVE_MIN and bars >= BARS_ALIVE_MIN:
        return "alive"
    if text <= DEAD_MAX and bars < BARS_ALIVE_MIN:
        return "dead"
    return "unknown"


def _int(text: str | None) -> int | None:
    return int(text) if text and text.isdigit() else None


def ban_count(frame: np.ndarray, side: str) -> int:
    level = max(headings_level(frame, side), HEADINGS_MIN)
    n = 0
    for i in range(L.MAX_BANS):
        if _p99(L.crop(frame, L.ban_box(side, i))) < BAN_PRESENT_MIN * level:
            break
        n += 1
    return n


def read_frame(frame: np.ndarray) -> FrameReading:
    if not hud.hud_present(frame):
        return FrameReading(hud=False, phase=hud.Phase.NONE)
    phase, clock, tidi = hud.read_timer(frame)
    scores = {
        "red": _int(glyphs.read(L.crop(frame, L.RED_SCORE), "header", DIGITS)),
        "blue": _int(glyphs.read(L.crop(frame, L.BLUE_SCORE), "header", DIGITS)),
    }
    rows = {}
    bans = {}
    for side in SIDES:
        rows[side] = []
        level = headings_level(frame, side)
        if level >= HEADINGS_MIN:
            bans[side] = ban_count(frame, side)
        for i in range(L.MAX_ROWS):
            status = row_status(frame, side, i, level)
            pts = None
            if status != "empty":
                pts = _int(glyphs.read(L.crop(frame, L.row_box(side, i, "pts")), "row", DIGITS,
                                       min_x=L.PTS_MIN_X[side]))
            rows[side].append(RowState(i, status, pts))
    return FrameReading(
        hud=True,
        phase=phase,
        clock=clock,
        tidi=tidi,
        scores=scores,
        rows=rows,
        ban_count=bans,
    )


_pool = ThreadPoolExecutor(max_workers=4)


def bars_only_name(region: np.ndarray) -> str:
    """Tesseract returns nothing for a name made only of 'I's (one team is
    named 25 of them). Team names are upper case, so if every glyph is a
    full-height vertical bar, spell it out."""
    gs = glyphs.segment(region)
    if len(gs) < 2:
        return ""
    lh = glyphs.line_height(gs)
    if all(g.shape[1] <= 4 and g.shape[0] >= 0.9 * lh for _, g in gs):
        return "I" * len(gs)
    return ""


def read_identity(frame: np.ndarray, reading: FrameReading) -> Identity:
    """OCR every name on the HUD. Tesseract calls run in parallel."""
    jobs = {
        ("team", "red", 0): L.RED_TEAM,
        ("team", "blue", 0): L.BLUE_TEAM,
    }
    for side in SIDES:
        for i in range(reading.ban_count.get(side, 0)):
            jobs[("ban", side, i)] = L.ban_box(side, i)
        for row in reading.rows.get(side, []):
            if row.status != "empty":
                jobs[("ship", side, row.slot)] = L.row_box(side, row.slot, "ship")
                jobs[("pilot", side, row.slot)] = L.row_box(side, row.slot, "name")
    futures = {k: _pool.submit(ocr.read_line, L.crop(frame, box)) for k, box in jobs.items()}
    text = {k: f.result() for k, f in futures.items()}
    for side in SIDES:
        if not text[("team", side, 0)]:
            team_box = L.RED_TEAM if side == "red" else L.BLUE_TEAM
            text[("team", side, 0)] = bars_only_name(L.crop(frame, team_box))

    def collect(kind, side):
        keys = sorted(k for k in text if k[0] == kind and k[1] == side)
        return [text[k] for k in keys]

    return Identity(
        teams={s: text[("team", s, 0)] for s in SIDES},
        bans={s: collect("ban", s) for s in SIDES},
        ships={s: collect("ship", s) for s in SIDES},
        pilots={s: collect("pilot", s) for s in SIDES},
        ban_count={s: reading.ban_count.get(s) for s in SIDES},
    )
