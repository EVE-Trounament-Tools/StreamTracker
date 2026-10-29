"""Pixel regions of the AT match HUD.

All coordinates are (x1, y1, x2, y2) on a 1920x1080 frame. Capture always
scales frames to this size, so the rest of the code never deals with
resolution.
"""

from dataclasses import dataclass

import numpy as np

FRAME_W, FRAME_H = 1920, 1080

Box = tuple[int, int, int, int]

TIMER: Box = (880, 752, 1040, 772)  # tight: in-space labels can show just above

# Header: team name (right-aligned on red, left-aligned on blue) and score.
RED_TEAM: Box = (100, 750, 772, 778)  # long names run far left
RED_SCORE: Box = (774, 750, 830, 778)
BLUE_SCORE: Box = (1090, 750, 1150, 778)
BLUE_TEAM: Box = (1152, 750, 1820, 778)

# "Ship Bans" labels, used as HUD-present anchors.
RED_BANS_LABEL: Box = (10, 752, 100, 778)
BLUE_BANS_LABEL: Box = (1824, 752, 1912, 778)
BATTLEFEED_LABEL: Box = (900, 836, 1020, 852)

# Roster column headings ("PTS NAME SPEED ... SHIP"): full-brightness text
# that shows how bright the HUD is drawn right now.
RED_HEADINGS: Box = (350, 836, 830, 852)
BLUE_HEADINGS: Box = (1090, 836, 1570, 852)

# Ban lists: up to 10 rows per side.
MAX_BANS = 10
BAN_TOP, BAN_PITCH = 785, 20
RED_BAN_X: tuple[int, int] = (8, 300)
BLUE_BAN_X: tuple[int, int] = (1620, 1912)

# Pre-match draft panel: one card per ban, left to right in pick order, each
# under a magenta (red) or cyan (blue) bar. Card count and width vary, and the
# whole panel sits lower in some years, so positions are measured from the
# bars rather than fixed.
DRAFT_BAR_SEARCH: Box = (355, 760, 1880, 880)
DRAFT_LABEL_X: tuple[int, int] = (155, 240)  # "RED TEAM" label, below the bars' top
DRAFT_TEAM_X: tuple[int, int] = (155, 345)

# Roster table.
MAX_ROWS = 10
ROW_TOP, ROW_PITCH = 862, 18


@dataclass(frozen=True)
class RosterCols:
    pts: tuple[int, int]
    name: tuple[int, int]
    speed: tuple[int, int]
    bars: tuple[int, int]
    ship: tuple[int, int]


RED_COLS = RosterCols(
    pts=(350, 392), name=(394, 543), speed=(543, 593), bars=(595, 722), ship=(732, 832)
)
BLUE_COLS = RosterCols(
    ship=(1088, 1198), bars=(1198, 1326), speed=(1328, 1379), name=(1379, 1528), pts=(1529, 1566)
)


# PTS digits are left-aligned this far into the PTS box; anything further left
# is a long pilot name spilling over.
PTS_MIN_X = {"red": 5, "blue": 2}


def ban_box(side: str, i: int) -> Box:
    x1, x2 = RED_BAN_X if side == "red" else BLUE_BAN_X
    y = BAN_TOP + BAN_PITCH * i
    return (x1, y, x2, y + BAN_PITCH)


def row_box(side: str, i: int, col: str) -> Box:
    cols = RED_COLS if side == "red" else BLUE_COLS
    x1, x2 = getattr(cols, col)
    y = ROW_TOP + ROW_PITCH * i
    return (x1, y, x2, y + ROW_PITCH)


def crop(frame: np.ndarray, box: Box) -> np.ndarray:
    x1, y1, x2, y2 = box
    return frame[y1:y2, x1:x2]
