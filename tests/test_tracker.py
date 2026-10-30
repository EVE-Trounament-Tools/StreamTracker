import os
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

import numpy as np

from stream_tracker import reader
from stream_tracker.db import Database
from stream_tracker.reader import Identity
from stream_tracker.tracker import (MIN_LATE_JOIN_HUD_FRAMES, MIN_MATCH_HUD_FRAMES, FrameTime, MatchState,
                                   Tracker)
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
    n = MIN_LATE_JOIN_HUD_FRAMES + 5  # joined late, so shorter would be taken for a match-clip ad
    feed(tracker, live_frames(frame, n=n) + [(640 + n + i, frame(400)) for i in range(15)])
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
    assert row["status"] == "abandoned" and row["match_no"] is None  # never went live: no number
    assert (row["red_team"], row["blue_team"]) == ("SEDITION.", "NEON NIGHTMARES")


@pytest.mark.parametrize("hud_frames, saw_countdown, kept", [
    (60, False, False),  # a match-clip ad: joined mid-clock, cut away after a minute
    (MIN_LATE_JOIN_HUD_FRAMES, False, True),  # a real match joined late
    (60, True, True),  # counted down, then the broadcast cut away
])
def test_late_joined_clip_is_an_ad(tmp_path, hud_frames, saw_countdown, kept):
    db_path, tracker, m = _hand_match(tmp_path)
    tracker.match = m
    m.hud_frames, m.saw_countdown, m.live_started = hud_frames, saw_countdown, True
    m.last_hud = FrameTime(T0, 0)
    later = tracker.grace_s + 1
    tracker._on_no_hud(FrameTime(T0 + timedelta(seconds=later), later))
    found = rows(db_path, "SELECT end_reason FROM matches")
    assert found == ([{"end_reason": "hud_lost"}] if kept else [])


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


DRAFT_BANS = [("blue", "Crucifier"), ("red", "Curse"), ("red", "Zarmazd"), ("blue", "Armageddon"),
              ("blue", "Sentinel"), ("red", "Ashimmu")]


def _fake_draft(monkeypatch, red="Neon Nightmares", blue="Sedition.", bans=DRAFT_BANS):
    """Any frame shows the draft panel; reading it gives these teams and bans."""
    from stream_tracker import draft

    panel = draft.Panel([(0, 1, side) for side, _ in bans], 0, 1, {})
    monkeypatch.setattr(draft, "find_panel", lambda f: panel)
    monkeypatch.setattr(draft, "read_panel", lambda f, p: draft.DraftRead({"red": red, "blue": blue}, list(bans)))


def _see_draft(tracker, frames=5, start=0):
    blank = np.zeros((1, 1, 3), np.uint8)
    for i in range(frames):
        tracker._check_draft(blank, FrameTime(T0 + timedelta(seconds=start + i), start + i))
        if tracker._draft_job is not None:
            tracker._draft_job.result()
            tracker._collect_draft()


def test_bans_follow_the_draft_panel(monkeypatch, frame, tmp_path):
    """The HUD lists bans out of order; the draft panel shown before the
    match has them in pick order."""
    db_path = tmp_path / "t.sqlite"
    tracker = Tracker(Database(db_path), channel="test")
    _fake_draft(monkeypatch)
    _see_draft(tracker, start=500)
    assert tracker.draft is not None
    seq = [(590 + i, 590) for i in range(25)] + [(628, 628), (629, 628)]
    feed(tracker, [(t, frame(src)) for t, src in seq])
    tracker.close()
    bans = rows(db_path, "SELECT side, ordinal, ship_type, pick FROM bans ORDER BY pick")
    assert [(b["side"], b["ship_type"]) for b in bans] == DRAFT_BANS
    assert [b["pick"] for b in bans] == [1, 2, 3, 4, 5, 6]
    red = [b["ship_type"] for b in sorted(bans, key=lambda b: b["ordinal"]) if b["side"] == "red"]
    assert red == ["Curse", "Zarmazd", "Ashimmu"]
    kinds = [e["kind"] for e in rows(db_path, "SELECT kind FROM events")]
    assert "bans_from_draft" in kinds and "draft_ban_mismatch" not in kinds
    assert tracker.draft is None  # used up


def test_draft_sides_matched_by_team_name(monkeypatch, tmp_path):
    db_path, tracker, m = _hand_match(tmp_path)
    _fake_draft(monkeypatch, red="Sedition.", blue="Neon Nightmares",
                bans=[("red", "Crucifier"), ("blue", "Curse")])
    _see_draft(tracker)
    for _ in range(3):
        tracker._vote(m, _identity(red="NEON NIGHTMARES", blue="SEDITION.", red_bans=("Curse",),
                                   blue_bans=("Crucifier",)))
    bans = rows(db_path, "SELECT side, ship_type, pick FROM bans ORDER BY pick")
    assert [(b["side"], b["ship_type"], b["pick"]) for b in bans] == [("blue", "Crucifier", 1), ("red", "Curse", 2)]


def test_other_teams_draft_falls_back_to_hud(monkeypatch, tmp_path):
    db_path, tracker, m = _hand_match(tmp_path)
    _fake_draft(monkeypatch, red="Odin's Call", blue="The Skeleton Crew")
    _see_draft(tracker)
    for _ in range(3):
        tracker._vote(m, _identity())
    bans = rows(db_path, "SELECT ship_type, pick FROM bans WHERE side='red' ORDER BY ordinal")
    assert [(b["ship_type"], b["pick"]) for b in bans] == [("Zarmazd", None), ("Eos", None)]
    assert tracker.draft is not None


def test_stale_draft_falls_back_to_hud(monkeypatch, tmp_path):
    from stream_tracker.tracker import DRAFT_MAX_AGE_S

    db_path, tracker, m = _hand_match(tmp_path)
    _fake_draft(monkeypatch, red="THE TUSKERS CO.", blue="META RELOADED")
    _see_draft(tracker)
    tracker._t = FrameTime(T0, DRAFT_MAX_AGE_S + 10)
    for _ in range(3):
        tracker._vote(m, _identity())
    assert rows(db_path, "SELECT pick FROM bans WHERE pick IS NOT NULL") == []


def test_draft_still_filling_in_restarts_votes(monkeypatch, tmp_path):
    from stream_tracker import draft

    _, tracker, _ = _hand_match(tmp_path)
    _fake_draft(monkeypatch, bans=DRAFT_BANS[:4])
    _see_draft(tracker, frames=2)
    _fake_draft(monkeypatch)
    _see_draft(tracker, frames=2, start=2)
    assert tracker.draft is None  # only 2 reads of the full panel so far
    _see_draft(tracker, frames=1, start=4)
    assert len(tracker.draft.bans) == 6


def test_reopened_match_keeps_draft_order(monkeypatch, tmp_path):
    db_path, tracker, m = _hand_match(tmp_path)
    tracker.match = m
    _fake_draft(monkeypatch, red="THE TUSKERS CO.", blue="META RELOADED",
                bans=[("red", "Eos"), ("red", "Zarmazd")])
    _see_draft(tracker)
    for _ in range(3):
        tracker._vote(m, _identity())
    tracker._reopen(FrameTime(T0, 10))
    for _ in range(3):
        tracker._vote(m, _identity())
    bans = rows(db_path, "SELECT ship_type, pick FROM bans WHERE side='red' ORDER BY ordinal")
    assert [(b["ship_type"], b["pick"]) for b in bans] == [("Eos", 1), ("Zarmazd", 2)]


def test_draft_taken_despite_a_misread_hud_name(monkeypatch, tmp_path):
    """One name matches and the banned ships agree (2873111426: the HUD read
    Rote Kapelle as "UNC KAPELLE")."""
    db_path, tracker, m = _hand_match(tmp_path)
    _fake_draft(monkeypatch, red="Neon Nightmares", blue="Rote Kapelle",
                bans=[("red", "Zarmazd"), ("blue", "Maulus"), ("blue", "Vigil"), ("red", "Eos")])
    _see_draft(tracker)
    for _ in range(3):
        tracker._vote(m, _identity(red="NEON NIGHTMARES", blue="UNC KAPELLE", red_bans=("Eos", "Zarmazd"),
                                   blue_bans=("Vigil", "Maulus")))
    bans = rows(db_path, "SELECT side, ship_type, pick FROM bans ORDER BY pick")
    assert [b["ship_type"] for b in bans] == ["Zarmazd", "Maulus", "Vigil", "Eos"]


def test_draft_taken_when_hud_names_are_stale(monkeypatch, tmp_path):
    """The HUD kept the previous match's team names; the bans still identify it."""
    db_path, tracker, m = _hand_match(tmp_path)
    _fake_draft(monkeypatch, red="Dracarys.", blue="Psychokids.",
                bans=[("red", "Zarmazd"), ("blue", "Maulus"), ("blue", "Vigil"), ("red", "Eos")])
    _see_draft(tracker)
    for _ in range(3):
        tracker._vote(m, _identity(red="CRITICAL. MASS.", blue="THE SKELETON CREW", red_bans=("Eos", "Zarmazd"),
                                   blue_bans=("Vigil", "Maulus")))
    assert rows(db_path, "SELECT count(*) n FROM bans WHERE pick IS NOT NULL") == [{"n": 4}]


def test_draft_not_taken_on_one_name_alone(monkeypatch, tmp_path):
    db_path, tracker, m = _hand_match(tmp_path)
    _fake_draft(monkeypatch, red="THE TUSKERS CO.", blue="Sedition.")
    _see_draft(tracker)
    for _ in range(3):
        tracker._vote(m, _identity())
    assert rows(db_path, "SELECT count(*) n FROM bans WHERE pick IS NOT NULL") == [{"n": 0}]


def test_clearer_read_wins_a_one_ship_disagreement(monkeypatch, tmp_path):
    """The panel read "Rapler" (-> Reaper) and the HUD "?agu": each side
    keeps the read that matches the ship list best, in pick order."""
    db_path, tracker, m = _hand_match(tmp_path)
    _fake_draft(monkeypatch, red="THE TUSKERS CO.", blue="META RELOADED",
                bans=[("red", "Rapler"), ("blue", "Magus"), ("red", "Huginn"), ("blue", "Eos")])
    _see_draft(tracker)
    for _ in range(3):
        tracker._vote(m, _identity(red_bans=("Huginn", "Rapier"), blue_bans=("?agu", "Eos")))
    bans = rows(db_path, "SELECT ship_type, pick FROM bans ORDER BY pick")
    assert [(b["ship_type"], b["pick"]) for b in bans] == [("Rapier", 1), ("Magus", 2), ("Huginn", 3), ("Eos", 4)]


@pytest.mark.parametrize("phase, stored", [
    (None, "XQTY"),
    ("Qualifiers", "XQTY"),
    ("Main Event", "xqtywiznalamywmodxfhhopawzpqyjdwrpeptuaenabjawdzku"),
])
def test_team_names_best_fit_only_past_the_qualifiers(tmp_path, phase, stored):
    db_path = tmp_path / "t.sqlite"
    db = Database(db_path)
    tracker = Tracker(db, phase=phase)
    m = MatchState(id=db.create_match(None, None, T0.isoformat(), phase), team_crops=np.zeros((1, 1), np.float32))
    for _ in range(3):
        tracker._vote(m, _identity(red="XQTY", blue="SEKER MATAR"))
    assert m.identity_settled
    row = rows(db_path, "SELECT red_team, blue_team FROM matches")[0]
    assert row["red_team"] == stored
    assert row["blue_team"] == ("SEKER MATAR" if phase in (None, "Qualifiers") else "Seker Matar")
