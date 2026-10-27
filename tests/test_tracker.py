import os
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

import numpy as np

from stream_tracker import reader
from stream_tracker.db import Database
from stream_tracker.reader import Identity
from stream_tracker.tracker import MIN_MATCH_HUD_FRAMES, FrameTime, MatchState, Tracker
from stream_tracker.vocab import ShipVocab, load_ships

T0 = datetime(2026, 10, 18, 15, 0, tzinfo=timezone.utc)


def feed(tracker, frames):
    for offset, f in frames:
        tracker.process(f, FrameTime(T0 + timedelta(seconds=offset), offset))


def rows(path, sql):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return [dict(r) for r in conn.execute(sql)]


def test_match_lifecycle(frame, tmp_path):
    """Desk -> PRE -> countdown -> live -> deaths -> 0:00 -> cut away."""
    db_path = tmp_path / "t.sqlite"
    tracker = Tracker(Database(db_path), channel="test")
    seq = [(540, 540), (541, 540)]  # desk
    seq += [(590 + i, 590) for i in range(25)]  # HUD up, -00:00-
    seq += [(628, 628), (629, 628)]  # countdown 0:18
    seq += [(655, 655), (662, 662), (663, 662)]  # live, nothing dead
    seq += [(1000, 1000), (1001, 1000)]  # 10 deaths seen
    seq += [(1260, 1260), (1261, 1260), (1262, 1260)]  # 0:00
    seq += [(1266 + i, 1266) for i in range(5)]  # promo
    feed(tracker, [(t, frame(src)) for t, src in seq])
    tracker.close()

    (m,) = rows(db_path, "SELECT * FROM matches")
    assert m["status"] == "complete" and m["end_reason"] == "time"
    assert (m["red_team"], m["blue_team"]) == ("NEON NIGHTMARES", "SEDITION.")
    assert m["series_id"] is not None and m["series_game"] == 1
    assert (m["red_score"], m["blue_score"]) == (114, 128)
    assert m["joined_late"] == 0
    assert m["countdown_start_utc"] == "2026-10-18T15:10:16+00:00"
    assert m["live_start_utc"] == "2026-10-18T15:10:46+00:00"

    ships = rows(db_path, "SELECT * FROM ships ORDER BY side DESC, slot")
    assert len(ships) == 20
    dead = {(s["side"], s["slot"]) for s in ships if s["died_utc"]}
    assert dead == {("red", i) for i in (2, 3, 4, 5, 6, 7, 8, 9)} | {("blue", i) for i in (0, 2, 3, 4)}
    first_seen = [s for s in ships if s["side"] == "blue" and s["slot"] == 0][0]
    assert first_seen["died_match_s"] == 600 - 246  # first seen dead at 4:06
    bans = rows(db_path, "SELECT ship_type FROM bans WHERE side='red' ORDER BY ordinal")
    assert [b["ship_type"] for b in bans] == ["Zarmazd", "Ashimmu", "Curse"]
    # Points are the PTS column read off the stream.
    read = reader.read_frame(frame(1000)).rows
    assert all(s["points"] == read[s["side"]][s["slot"]].pts for s in ships)


def test_joined_late(frame, tmp_path):
    db_path = tmp_path / "t.sqlite"
    tracker = Tracker(Database(db_path))
    feed(tracker, [(1000 + i, frame(1000)) for i in range(6)])
    tracker.close()
    (m,) = rows(db_path, "SELECT * FROM matches")
    assert m["joined_late"] == 1 and m["status"] == "live"
    unknown = rows(db_path, "SELECT * FROM ships WHERE death_confidence='unknown_time'")
    assert len(unknown) == 10 and all(s["died_utc"] is None for s in unknown)


@pytest.fixture
def running_clock(monkeypatch):
    """Make a repeated still frame read as a clock that keeps running, as the
    real one would: clock = 584 at offset 662, counting down."""
    from stream_tracker import reader
    from stream_tracker.hud import Phase

    read_frame = reader.read_frame
    state = {}

    def fake(frame):
        r = read_frame(frame)
        if r.phase == Phase.LIVE and r.clock not in (None, 0):
            r.clock = max(0, 584 - int(state["offset"] - 662))
        return r

    monkeypatch.setattr(reader, "read_frame", fake)
    orig = Tracker.process

    def process(self, f, t):
        state["offset"] = t.offset
        return orig(self, f, t)

    monkeypatch.setattr(Tracker, "process", process)


def live_frames(frame, start=640, n=30, src=662):
    return [(start + i, frame(src)) for i in range(n)]


def test_short_hud_is_discarded(frame, tmp_path):
    """An ad showing the HUD for under 25s leaves nothing behind."""
    db_path = tmp_path / "t.sqlite"
    tracker = Tracker(Database(db_path), grace_s=10)
    feed(tracker, [(i, frame(1000)) for i in range(15)] + [(15 + i, frame(400)) for i in range(20)])
    tracker.close()
    assert rows(db_path, "SELECT * FROM matches") == []
    assert rows(db_path, "SELECT * FROM ships") == []


def test_foreign_hud_mid_match_is_ignored(frame, tmp_path, running_clock):
    """A 15s ad with another match's HUD doesn't end the current match."""
    db_path = tmp_path / "t.sqlite"
    tracker = Tracker(Database(db_path))
    seq = live_frames(frame, 640, 30)
    seq += [(670 + i, frame(30)) for i in range(15)]  # Of Essence HUD
    seq += live_frames(frame, 685, 30, src=1000)
    seq += [(1246 + i, frame(1260)) for i in range(3)]  # clock 0:00
    feed(tracker, seq)
    tracker.close()
    (m,) = rows(db_path, "SELECT * FROM matches")
    assert m["end_reason"] == "time" and m["red_team"] == "NEON NIGHTMARES"


def test_hud_lost_ends_after_grace(frame, tmp_path, running_clock):
    db_path = tmp_path / "t.sqlite"
    tracker = Tracker(Database(db_path), grace_s=10)
    feed(tracker, live_frames(frame) + [(670 + i, frame(400)) for i in range(15)])
    (m,) = rows(db_path, "SELECT * FROM matches")
    assert m["end_reason"] == "hud_lost"


VOD = os.environ.get("STREAM_TRACKER_VOD")


@pytest.mark.skipif(not VOD, reason="set STREAM_TRACKER_VOD to the reference segment (v2860628871 1:40:00-2:02:00)")
def test_reference_vod(tmp_path):
    from stream_tracker.capture import FileSource

    db_path = tmp_path / "t.sqlite"
    tracker = Tracker(Database(db_path))
    for f, t in FileSource(VOD, start_utc=T0):
        tracker.process(f, t)
    tracker.close(t)
    m1, m2 = rows(db_path, "SELECT * FROM matches ORDER BY id")
    assert (m1["red_team"], m1["blue_team"], m1["joined_late"]) == ("OF ESSENCE", "ARCTIC LIGHT", 1)
    assert (m2["red_score"], m2["blue_score"], m2["end_reason"]) == (114, 128, "time")
    deaths = rows(db_path, f"SELECT * FROM ships WHERE match_id={m2['id']} AND died_utc IS NOT NULL")
    assert len(deaths) == 12
    assert all(d["death_confidence"] == "high" for d in deaths)


def test_unknown_ban_keeps_the_others(tmp_path):
    """AT XXI bans included Cenotaph, which isn't in the AT XXII ship list.
    The other bans on that side must still resolve."""
    db_path = tmp_path / "t.sqlite"
    db = Database(db_path)
    tracker = Tracker(db)
    m = MatchState(id=db.create_match(None, None, T0.isoformat()), team_crops=np.zeros((1, 1), np.float32))
    good = Identity(
        teams={"red": "THE TUSKERS CO.", "blue": "META RELOADED"},
        bans={"red": ["Zarmazd", "Raven Navy Issue", "Cenotaph"], "blue": ["Eos", "Dominix"]},
        ships={"red": [], "blue": []}, pilots={"red": [], "blue": []},
        ban_count={"red": 3, "blue": 2},
    )
    garbage = Identity(teams=good.teams, bans={"red": ["Zarmazd", "xq#v", "~~"], "blue": good.bans["blue"]},
                       ships=good.ships, pilots=good.pilots, ban_count=good.ban_count)
    for ident in (good, garbage, good, good):
        tracker._vote(m, ident)
    assert m.identity_settled
    bans = rows(db_path, "SELECT ship_raw, ship_type FROM bans WHERE side='red' ORDER BY ordinal")
    assert [(b["ship_raw"], b["ship_type"]) for b in bans] == [
        ("Zarmazd", "Zarmazd"), ("Raven Navy Issue", "Raven Navy Issue"), ("Cenotaph", None)]


def _identity(red="THE TUSKERS CO.", blue="META RELOADED", red_bans=("Zarmazd", "Eos"), blue_bans=(),
              ban_count=None):
    counts = ban_count or {"red": len(red_bans), "blue": len(blue_bans)}
    return Identity(teams={"red": red, "blue": blue}, bans={"red": list(red_bans), "blue": list(blue_bans)},
                    ships={"red": [], "blue": []}, pilots={"red": [], "blue": []}, ban_count=counts)


def _hand_match(tmp_path):
    db_path = tmp_path / "t.sqlite"
    db = Database(db_path)
    tracker = Tracker(db)
    m = MatchState(id=db.create_match(None, None, T0.isoformat()), team_crops=np.zeros((1, 1), np.float32))
    return db_path, tracker, m


def test_zero_bans_is_a_valid_list(tmp_path):
    db_path, tracker, m = _hand_match(tmp_path)
    for _ in range(3):
        tracker._vote(m, _identity())
    assert m.identity_settled
    assert rows(db_path, "SELECT * FROM bans WHERE side='blue'") == []
    bans = rows(db_path, "SELECT ship_type FROM bans WHERE side='red' ORDER BY ordinal")
    assert [b["ship_type"] for b in bans] == ["Zarmazd", "Eos"]


def test_faded_hud_doesnt_wipe_bans(tmp_path):
    """A faded HUD can't count bans; its empty lists mustn't outvote the real ones."""
    db_path, tracker, m = _hand_match(tmp_path)
    faded = _identity(red_bans=(), ban_count={"red": None, "blue": None})
    good = _identity(red_bans=("Curse", "Zarmazd", "Ashimmu"))
    for ident in (faded, faded, faded, good, faded, good, good):
        tracker._vote(m, ident)
    assert m.identity_settled
    bans = rows(db_path, "SELECT ordinal, ship_type FROM bans WHERE side='red' ORDER BY ordinal")
    assert [(b["ordinal"], b["ship_type"]) for b in bans] == [(1, "Curse"), (2, "Zarmazd"), (3, "Ashimmu")]


def test_team_colours_recorded_before_settling(tmp_path):
    db_path, tracker, m = _hand_match(tmp_path)
    tracker._vote(m, _identity(red="", blue=""))
    tracker._vote(m, _identity(red="SEDITION.", blue=" "))
    assert not m.identity_settled
    (row,) = rows(db_path, "SELECT red_team, blue_team FROM matches")
    assert (row["red_team"], row["blue_team"]) == ("SEDITION.", None)
    for _ in range(3):
        tracker._vote(m, _identity(red="SEDITION.", blue="NEON NIGHTMARES"))
    assert m.identity_settled
    (row,) = rows(db_path, "SELECT red_team, blue_team FROM matches")
    assert (row["red_team"], row["blue_team"]) == ("SEDITION.", "NEON NIGHTMARES")


def test_abandoned_match_keeps_team_colours(tmp_path):
    db_path, tracker, m = _hand_match(tmp_path)
    tracker.match = m
    m.hud_frames = MIN_MATCH_HUD_FRAMES
    m.last_hud = FrameTime(T0, 0)
    tracker._vote(m, _identity(red="SEDITION.", blue="NEON NIGHTMARES"))
    later = tracker.grace_s + 1
    tracker._on_no_hud(FrameTime(T0 + timedelta(seconds=later), later))
    (row,) = rows(db_path, "SELECT * FROM matches")
    assert row["status"] == "abandoned"
    assert (row["red_team"], row["blue_team"]) == ("SEDITION.", "NEON NIGHTMARES")


def test_points_not_taken_from_ship_list(frame, tmp_path):
    """Stored points come from the HUD even if the ship list disagrees."""
    from dataclasses import replace

    db_path = tmp_path / "t.sqlite"
    bogus = ShipVocab([replace(s, points=999) for s in load_ships()])
    tracker = Tracker(Database(db_path), vocab=bogus)
    feed(tracker, [(1000 + i, frame(1000)) for i in range(6)])
    tracker.close()
    read = reader.read_frame(frame(1000)).rows
    ships = rows(db_path, "SELECT * FROM ships")
    assert ships and all(s["points"] == read[s["side"]][s["slot"]].pts != 999 for s in ships)
