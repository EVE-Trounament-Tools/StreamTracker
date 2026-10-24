"""Readings on frames from the reference VOD (v2860628871 from 1:40:00).

Fixture f_NNNN.png is NNNN seconds into that segment.
"""

import numpy as np
import pytest

from stream_tracker import layout as L, reader
from stream_tracker.hud import Phase
from stream_tracker.vocab import default_vocab

RED_SHIPS = ["Sleipnir", "Sleipnir", "Basilisk", "Khizriel", "Scythe Fleet Issue", "Tholos",
             "Svipul", "Svipul", "Caracal", "Raptor"]
BLUE_SHIPS = ["Damnation", "Guardian", "Cyclone Fleet Issue", "Cyclone", "Huginn", "Arbitrator",
              "Arbitrator", "Keres", "Maulus", "Maulus"]
RED_PTS = [36, 36, 32, 31, 16, 13, 11, 11, 8, 6]
BLUE_PTS = [33, 32, 30, 26, 25, 14, 14, 12, 7, 7]


@pytest.mark.parametrize(
    "offset, phase, clock",
    [
        (400, Phase.NONE, None),  # promo
        (540, Phase.NONE, None),  # desk with team-comp lower third
        (590, Phase.PRE, None),
        (612, Phase.PRE, None),
        (628, Phase.COUNTDOWN, 18),
        (655, Phase.LIVE, 591),
        (1000, Phase.LIVE, 246),
        (1260, Phase.LIVE, 0),
        (1266, Phase.NONE, None),  # cut away after the match
        (30, Phase.LIVE, 316),  # previous match
    ],
)
def test_phase_and_clock(frame, offset, phase, clock):
    r = reader.read_frame(frame(offset))
    assert r.phase == phase
    assert r.clock == clock


@pytest.mark.parametrize(
    "offset, scores",
    [(662, (0, 0)), (1000, (88, 97)), (1100, (88, 128)), (1260, (114, 128)), (30, (55, 55))],
)
def test_scores(frame, offset, scores):
    r = reader.read_frame(frame(offset))
    assert (r.scores["red"], r.scores["blue"]) == scores


def test_points(frame):
    for offset in (662, 1000, 1260):
        r = reader.read_frame(frame(offset))
        assert [row.pts for row in r.rows["red"]] == RED_PTS
        assert [row.pts for row in r.rows["blue"]] == BLUE_PTS


@pytest.mark.parametrize(
    "offset, red_dead, blue_dead",
    [
        (662, [], []),
        (1000, [2, 4, 5, 6, 7, 8, 9], [0, 2, 4]),
        (1260, [2, 3, 4, 5, 6, 7, 8, 9], [0, 2, 3, 4]),
    ],
)
def test_alive_dead(frame, offset, red_dead, blue_dead):
    r = reader.read_frame(frame(offset))
    for side, dead in (("red", red_dead), ("blue", blue_dead)):
        statuses = [row.status for row in r.rows[side]]
        assert statuses == ["dead" if i in dead else "alive" for i in range(10)]


def test_identity(frame):
    f = frame(1000)
    ident = reader.read_identity(f, reader.read_frame(f))
    vocab = default_vocab()
    assert ident.teams == {"red": "NEON NIGHTMARES", "blue": "SEDITION."}
    assert [vocab.match(b).ship.name for b in ident.bans["red"]] == ["Zarmazd", "Ashimmu", "Curse"]
    assert [vocab.match(b).ship.name for b in ident.bans["blue"]] == ["Crucifier", "Armageddon", "Sentinel"]
    for side, ships, pts in (("red", RED_SHIPS, RED_PTS), ("blue", BLUE_SHIPS, BLUE_PTS)):
        got = [vocab.match(raw, p).ship.name for raw, p in zip(ident.ships[side], pts)]
        assert got == ships
    assert ident.pilots["red"][0] == "Alisa Callam"


def test_ten_bans(frame):
    """Bans can run to 10 per side; copy the real ban rows down to fill all 10."""
    f = frame(662)
    names = []
    for side in ("red", "blue"):
        for i in range(3, L.MAX_BANS):
            src = L.ban_box(side, i % 3)
            dst = L.ban_box(side, i)
            f[dst[1]:dst[3], dst[0]:dst[2]] = f[src[1]:src[3], src[0]:src[2]]
    r = reader.read_frame(f)
    assert r.ban_count == {"red": 10, "blue": 10}
    ident = reader.read_identity(f, r)
    vocab = default_vocab()
    names = [vocab.match(b).ship.name for b in ident.bans["blue"]]
    assert names == (["Crucifier", "Armageddon", "Sentinel"] * 4)[:10]


def test_empty_rows(frame):
    """A team with fewer than 10 pilots leaves blank rows."""
    f = frame(662)
    for i in (8, 9):
        x1, y1, x2, y2 = L.row_box("blue", i, "ship")
        f[y1:y2, 1088:1566] = np.median(f[y1:y2, 1088:1566], axis=(0, 1))
    r = reader.read_frame(f)
    assert [row.status for row in r.rows["blue"]][-3:] == ["alive", "empty", "empty"]


@pytest.mark.parametrize("offset", [30, 662, 1000, 1100, 1260])
def test_score_is_200_minus_surviving_points(frame, offset):
    """Holds for both matches, including Arctic Light's 199-point roster."""
    r = reader.read_frame(frame(offset))
    for side, other in (("red", "blue"), ("blue", "red")):
        surviving = sum(row.pts for row in r.rows[other] if row.status == "alive")
        assert r.scores[side] == 200 - surviving


def test_pilot_name_overflowing_into_pts(frame):
    """Day 2 1300s: 'Caldari Daddy 69420' spills into the PTS column of 13."""
    import cv2
    import numpy as np
    from pathlib import Path

    strip = cv2.imread(str(Path(__file__).parent / "fixtures" / "d2_1300.png"))
    f = np.vstack([np.zeros((740, 1920, 3), np.uint8), strip])
    r = reader.read_frame(f)
    assert [row.pts for row in r.rows["blue"]] == [43, 40, 33, 18, 15, 13, 12, 10, 10, 6]


def _strip(name):
    import cv2
    from pathlib import Path

    strip = cv2.imread(str(Path(__file__).parent / "fixtures" / name))
    return np.vstack([np.zeros((740, 1920, 3), np.uint8), strip])


@pytest.mark.parametrize(
    "name, clock, tidi",
    [
        ("d3_20126.png", 0, False),  # white real-time clock ran out...
        ("d3_20190.png", 3 * 60 + 55, True),  # ...red "3:55 TiDi:225%" game clock
        ("d3_20300.png", 2 * 60 + 6, True),  # "2:06", tesseract says "2:86"
    ],
)
def test_tidi_clock(name, clock, tidi):
    r = reader.read_frame(_strip(name))
    assert (r.phase, r.clock, r.tidi) == (Phase.LIVE, clock, tidi)
