import hashlib
import sqlite3
from datetime import datetime, timedelta, timezone

from stream_tracker.db import Database, pilots_match

T0 = datetime(2026, 10, 18, 15, 0, tzinfo=timezone.utc)
PILOTS = {team: [hashlib.md5(f"{team}{i}".encode()).hexdigest()[:12] for i in range(10)] for team in "ABCD"}


def add_match(db, red, blue, minutes, red_score=None, blue_score=None, pilots=None):
    """A settled match `minutes` after T0; pilots default to each team's usual lineup."""
    start = T0 + timedelta(minutes=minutes)
    mid = db.create_match("test", None, start.isoformat(timespec="seconds"))
    db.update_match(mid, red_team=red, blue_team=blue)
    pilots = pilots or {}
    for side, team in (("red", red), ("blue", blue)):
        for i, p in enumerate(pilots.get(side, PILOTS[team])):
            db.upsert_ship(mid, side, i, pilot=p)
    sid, mismatch = db.assign_series(mid)
    if red_score is not None:
        db.update_match(mid, red_score=red_score, blue_score=blue_score, status="complete",
                        ended_utc=(start + timedelta(minutes=11)).isoformat(timespec="seconds"))
        if sid is not None:
            db.refresh_series(sid)
    return mid, sid, mismatch


def series(db, sid):
    return dict(db.conn.execute("SELECT * FROM series WHERE id = ?", (sid,)).fetchone())


def game(db, mid):
    return db.conn.execute("SELECT series_id, series_game FROM matches WHERE id = ?", (mid,)).fetchone()


def test_series_spans_other_matches_and_side_swaps(tmp_path):
    db = Database(tmp_path / "t.sqlite")
    g1, s1, _ = add_match(db, "A", "B", 0, 200, 40)
    other, s_other, _ = add_match(db, "C", "D", 15, 90, 120)
    g2, s2, _ = add_match(db, "B", "A", 30)  # sides swapped
    assert s2 == s1 != s_other
    assert tuple(game(db, g2)) == (s1, 2) and tuple(game(db, other)) == (s_other, 1)
    s = series(db, s1)
    assert (s["team_a"], s["team_b"], s["games"], s["team_a_wins"], s["team_b_wins"]) == ("A", "B", 2, 1, 0)

    # Game 2 finishes: B (red this time) wins, so 1-1.
    db.update_match(g2, red_score=150, blue_score=100, status="complete")
    db.refresh_series(s1)
    s = series(db, s1)
    assert (s["team_a_wins"], s["team_b_wins"]) == (1, 1)

    g3, s3, _ = add_match(db, "A", "B", 60, 200, 10)
    assert s3 == s1 and game(db, g3)["series_game"] == 3
    assert series(db, s1)["team_a_wins"] == 2


def test_same_names_other_pilots_is_not_a_series(tmp_path):
    """The overlay kept the previous match's team names (AT XXII day 6, m4/m5)."""
    db = Database(tmp_path / "t.sqlite")
    _, s1, _ = add_match(db, "A", "B", 0, 200, 9)
    g2, s2, mismatch = add_match(db, "A", "B", 15, pilots={"red": PILOTS["C"], "blue": PILOTS["D"]})
    assert mismatch and s2 is None
    assert tuple(game(db, g2)) == (None, None)
    assert series(db, s1)["games"] == 1


def test_pilots_read_with_truncation_and_ocr_slips(tmp_path):
    db = Database(tmp_path / "t.sqlite")
    _, s1, _ = add_match(db, "A", "B", 0, 200, 9)
    # Truncated by the HUD, and one character misread.
    trimmed = {"red": [p[:7] + "…" for p in PILOTS["A"][:4]],
               "blue": [p[:-1] + "#" for p in PILOTS["B"][:3]] + PILOTS["C"][:7]}
    _, s2, mismatch = add_match(db, "A", "B", 15, pilots=trimmed)
    assert not mismatch and s2 == s1
    assert pilots_match("strange misunderst...", "Strange Misunderstanding")
    assert not pilots_match("MitchellSw", "Melissa Kinrenka")


def test_much_later_rematch_is_a_new_series(tmp_path):
    db = Database(tmp_path / "t.sqlite")
    _, s1, _ = add_match(db, "A", "B", 0, 200, 9)
    _, s2, mismatch = add_match(db, "A", "B", 7 * 60)
    assert not mismatch and s2 not in (None, s1)


def test_abandoned_or_deleted_games_leave_the_series(tmp_path):
    db = Database(tmp_path / "t.sqlite")
    g1, s1, _ = add_match(db, "A", "B", 0, 200, 9)
    g2, _, _ = add_match(db, "A", "B", 15)
    g3, _, _ = add_match(db, "A", "B", 30)
    db.update_match(g2, status="abandoned")
    db.refresh_series(s1)
    assert tuple(game(db, g2)) == (None, None)
    assert game(db, g3)["series_game"] == 2 and series(db, s1)["games"] == 2
    db.delete_match(g3)
    assert series(db, s1)["games"] == 1
    db.delete_match(g1)
    assert db.conn.execute("SELECT count(*) FROM series").fetchone()[0] == 0


def test_reassigning_is_idempotent(tmp_path):
    db = Database(tmp_path / "t.sqlite")
    g1, s1, _ = add_match(db, "A", "B", 0, 200, 9)
    g2, _, _ = add_match(db, "B", "A", 15, 100, 150)
    for mid in (g1, g2, g1, g2):
        assert db.assign_series(mid) == (s1, False)
    s = series(db, s1)
    assert (s["games"], s["team_a_wins"], s["team_b_wins"]) == (2, 2, 0)
    assert db.conn.execute("SELECT count(*) FROM series").fetchone()[0] == 1


def test_old_database_gets_series_columns(tmp_path):
    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE matches (id INTEGER PRIMARY KEY, channel TEXT, source TEXT,"
                 " created_utc TEXT NOT NULL, ended_utc TEXT, red_team TEXT, blue_team TEXT,"
                 " red_score INTEGER, blue_score INTEGER, status TEXT NOT NULL)")
    conn.execute("INSERT INTO matches (created_utc, red_team, blue_team, status)"
                 " VALUES ('2026-10-18T15:00:00+00:00', 'A', 'B', 'complete')")
    conn.commit()
    conn.close()
    db = Database(path)
    sid, _ = db.assign_series(1)
    assert tuple(game(db, 1)) == (sid, 1)


def test_series_ends_once_either_team_plays_someone_else(tmp_path):
    """AT XXI day 3: an early Tuskers vs Meta game, then each played others,
    then the Bo5 final between them."""
    db = Database(tmp_path / "t.sqlite")
    _, early, _ = add_match(db, "A", "B", 0, 200, 26)
    add_match(db, "C", "D", 15, 100, 50)  # other teams: doesn't end it
    _, same, _ = add_match(db, "B", "A", 30, 10, 200)
    assert same == early
    add_match(db, "B", "C", 45, 186, 86)  # B moves on
    _, final, _ = add_match(db, "A", "B", 120, 113, 0)
    assert final not in (None, early)
    assert series(db, early)["games"] == 2 and series(db, final)["games"] == 1


def test_old_database_gets_ban_picks(tmp_path):
    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE bans (id INTEGER PRIMARY KEY, match_id INTEGER NOT NULL, side TEXT NOT NULL,"
                 " ordinal INTEGER NOT NULL, ship_raw TEXT, ship_type TEXT, UNIQUE (match_id, side, ordinal))")
    conn.commit()
    conn.close()
    db = Database(path)
    mid = db.create_match(None, None, T0.isoformat())
    db.set_bans(mid, "red", [("Curse", "Curse"), ("Eos", "Eos")], picks=[2, 3])
    assert [tuple(r) for r in db.conn.execute("SELECT ordinal, pick FROM bans ORDER BY ordinal")] == [(1, 2), (2, 3)]
