"""SQLite storage. Every write commits immediately so a crash loses nothing."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from rapidfuzz import fuzz

# A repeat of the same pairing within this long of the series' last game is
# the next game of that series (Bo3/Bo5); later, it starts a new series.
SERIES_WINDOW_S = 6 * 3600
# Each team must field this many of the same pilots as an earlier game, or the
# overlay was probably showing stale team names.
SERIES_MIN_SHARED_PILOTS = 3
PILOT_MATCH_RATIO = 85

SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY,
    channel TEXT,
    source TEXT,
    created_utc TEXT NOT NULL,
    countdown_start_utc TEXT,
    live_start_utc TEXT,
    ended_utc TEXT,
    end_reason TEXT,            -- time | wipe | hud_lost | stream_end
    red_team TEXT,
    blue_team TEXT,
    red_score INTEGER,
    blue_score INTEGER,
    joined_late INTEGER NOT NULL DEFAULT 0,
    tidi_overtime_s INTEGER,    -- play continued this long past the white 0:00 (time dilation)
    status TEXT NOT NULL,       -- pending | live | complete | abandoned
    series_id INTEGER REFERENCES series(id),
    series_game INTEGER         -- 1-based game number within the series
);

CREATE TABLE IF NOT EXISTS series (
    id INTEGER PRIMARY KEY,
    team_a TEXT NOT NULL,       -- the pair sorted, so sides don't matter
    team_b TEXT NOT NULL,
    games INTEGER NOT NULL DEFAULT 0,
    team_a_wins INTEGER NOT NULL DEFAULT 0,
    team_b_wins INTEGER NOT NULL DEFAULT 0,
    first_utc TEXT,
    last_utc TEXT
);

CREATE TABLE IF NOT EXISTS bans (
    id INTEGER PRIMARY KEY,
    match_id INTEGER NOT NULL REFERENCES matches(id),
    side TEXT NOT NULL CHECK (side IN ('red', 'blue')),
    ordinal INTEGER NOT NULL,   -- 1..10, top to bottom
    ship_raw TEXT,
    ship_type TEXT,             -- NULL when OCR didn't match the ship list
    UNIQUE (match_id, side, ordinal)
);

CREATE TABLE IF NOT EXISTS ships (
    id INTEGER PRIMARY KEY,
    match_id INTEGER NOT NULL REFERENCES matches(id),
    side TEXT NOT NULL CHECK (side IN ('red', 'blue')),
    slot INTEGER NOT NULL,      -- 0-based row in the HUD roster
    ship_raw TEXT,
    ship_type TEXT,
    match_confidence REAL,      -- fuzzy score of ship_raw -> ship_type
    pilot TEXT,
    points INTEGER,
    died_match_s INTEGER,       -- real seconds since the match went live
    died_clock_s INTEGER,       -- clock the HUD showed (remaining seconds)
    died_in_tidi INTEGER,       -- 1 if that was the red TiDi game clock
    died_utc TEXT,
    died_stream_offset_s REAL,
    death_confidence TEXT,      -- high (team score agrees) | low | unknown_time
    gap_before_s REAL,          -- HUD was off screen this long before the death was seen
    UNIQUE (match_id, side, slot)
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    match_id INTEGER REFERENCES matches(id),
    ts_utc TEXT NOT NULL,
    stream_offset_s REAL,
    kind TEXT NOT NULL,
    payload_json TEXT
);
"""

# Columns added after the first release; ALTERed into older databases.
MIGRATIONS = {
    "matches": [("series_id", "INTEGER REFERENCES series(id)"), ("series_game", "INTEGER")],
}


def pilots_match(a: str, b: str) -> bool:
    """Fuzzy pilot-name equality, allowing for the HUD truncating names with '…'."""
    names = []
    truncated = False
    for n in (a, b):
        n = n.lower().strip()
        for tail in ("…", "..."):
            if n.endswith(tail):
                n, truncated = n[: -len(tail)].rstrip(), True
        names.append(n)
    a, b = names
    if truncated:
        a, b = a[: min(len(a), len(b))], b[: min(len(a), len(b))]
    return bool(a) and fuzz.ratio(a, b) >= PILOT_MATCH_RATIO


def shared_pilots(a: list[str], b: list[str]) -> int:
    left = list(b)
    n = 0
    for p in a:
        hit = next((q for q in left if pilots_match(p, q)), None)
        if hit is not None:
            left.remove(hit)
            n += 1
    return n


class Database:
    def __init__(self, path: str | Path):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._migrate()
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def _migrate(self) -> None:
        for table, cols in MIGRATIONS.items():
            have = {r["name"] for r in self.conn.execute(f"PRAGMA table_info({table})")}
            if not have:
                continue  # new database; SCHEMA creates it
            for name, decl in cols:
                if name not in have:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")

    def close(self) -> None:
        self.conn.close()

    def _commit(self, sql: str, params=()) -> sqlite3.Cursor:
        cur = self.conn.execute(sql, params)
        self.conn.commit()
        return cur

    def create_match(self, channel: str | None, source: str | None, created_utc: str) -> int:
        cur = self._commit(
            "INSERT INTO matches (channel, source, created_utc, status) VALUES (?, ?, ?, 'pending')",
            (channel, source, created_utc),
        )
        return cur.lastrowid

    def delete_match(self, match_id: int) -> None:
        series_id = self._series_of(match_id)
        for table in ("events", "ships", "bans"):
            self.conn.execute(f"DELETE FROM {table} WHERE match_id = ?", (match_id,))
        self.conn.execute("DELETE FROM matches WHERE id = ?", (match_id,))
        self.conn.commit()
        if series_id is not None:
            self.refresh_series(series_id)

    def update_match(self, match_id: int, **fields) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k} = ?" for k in fields)
        self._commit(f"UPDATE matches SET {cols} WHERE id = ?", (*fields.values(), match_id))

    def set_bans(self, match_id: int, side: str, bans: list[tuple[str, str | None]]) -> None:
        self.conn.execute("DELETE FROM bans WHERE match_id = ? AND side = ?", (match_id, side))
        self.conn.executemany(
            "INSERT INTO bans (match_id, side, ordinal, ship_raw, ship_type) VALUES (?, ?, ?, ?, ?)",
            [(match_id, side, i + 1, raw, typ) for i, (raw, typ) in enumerate(bans)],
        )
        self.conn.commit()

    def upsert_ship(self, match_id: int, side: str, slot: int, **fields) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO ships (match_id, side, slot) VALUES (?, ?, ?)",
            (match_id, side, slot),
        )
        if fields:
            cols = ", ".join(f"{k} = ?" for k in fields)
            self.conn.execute(
                f"UPDATE ships SET {cols} WHERE match_id = ? AND side = ? AND slot = ?",
                (*fields.values(), match_id, side, slot),
            )
        self.conn.commit()

    def log_event(self, match_id: int | None, ts_utc: str, offset: float | None, kind: str,
                  payload: dict | None = None) -> None:
        self._commit(
            "INSERT INTO events (match_id, ts_utc, stream_offset_s, kind, payload_json) VALUES (?, ?, ?, ?, ?)",
            (match_id, ts_utc, offset, kind, json.dumps(payload) if payload else None),
        )

    # ------------------------------------------------------------------ series

    def _series_of(self, match_id: int) -> int | None:
        row = self.conn.execute("SELECT series_id FROM matches WHERE id = ?", (match_id,)).fetchone()
        return row["series_id"] if row else None

    def _pilots(self, match_id: int, side: str) -> list[str]:
        return [r["pilot"] for r in self.conn.execute(
            "SELECT pilot FROM ships WHERE match_id = ? AND side = ? AND pilot IS NOT NULL",
            (match_id, side))]

    def _same_lineups(self, m: sqlite3.Row, other: sqlite3.Row) -> bool:
        """Does each team field enough of the same pilots in both matches?
        A side whose pilots weren't read doesn't count against it."""
        for side in ("red", "blue"):
            team = m[f"{side}_team"]
            other_side = "red" if other["red_team"] == team else "blue"
            mine, theirs = self._pilots(m["id"], side), self._pilots(other["id"], other_side)
            if mine and theirs and shared_pilots(mine, theirs) < SERIES_MIN_SHARED_PILOTS:
                return False
        return True

    def assign_series(self, match_id: int) -> tuple[int | None, bool]:
        """Put a match in the series for its team pairing, creating one if needed.

        Returns (series_id, mismatch). mismatch is True when a recent series
        exists for these teams but the lineups differ, so the match is left
        out of it (the overlay likely showed stale team names). Safe to call
        again, e.g. after the names are re-read.
        """
        m = self.conn.execute("SELECT * FROM matches WHERE id = ?", (match_id,)).fetchone()
        old = m["series_id"]
        if not m["red_team"] or not m["blue_team"] or m["red_team"] == m["blue_team"]:
            new, mismatch = None, False
        else:
            a, b = sorted((m["red_team"], m["blue_team"]))
            created = datetime.fromisoformat(m["created_utc"])
            new, mismatch = None, False
            for s in self.conn.execute(
                "SELECT * FROM series WHERE team_a = ? AND team_b = ? ORDER BY last_utc DESC", (a, b)
            ).fetchall():
                others = self.conn.execute(
                    "SELECT * FROM matches WHERE series_id = ? AND id != ? AND created_utc <= ?",
                    (s["id"], match_id, m["created_utc"])).fetchall()
                if not others:
                    continue
                last = max(datetime.fromisoformat(o["ended_utc"] or o["created_utc"]) for o in others)
                if (created - last).total_seconds() > SERIES_WINDOW_S:
                    continue
                if self._played_others(a, b, max(o["created_utc"] for o in others), m["created_utc"]):
                    continue  # that series is over; this is a later meeting
                if any(self._same_lineups(m, o) for o in others):
                    new = s["id"]
                else:
                    mismatch = True
                break
            if new is None and not mismatch:
                new = old if old is not None and self._series_teams(old) == (a, b) else None
                if new is None:
                    new = self._commit("INSERT INTO series (team_a, team_b) VALUES (?, ?)", (a, b)).lastrowid
        self._commit("UPDATE matches SET series_id = ? WHERE id = ?", (new, match_id))
        for sid in {old, new} - {None}:
            self.refresh_series(sid)
        return new, mismatch

    def _played_others(self, a: str, b: str, after_utc: str, before_utc: str) -> bool:
        """Did team a or b play anyone else between these times? Matches
        between other teams don't end a series."""
        row = self.conn.execute(
            "SELECT 1 FROM matches WHERE created_utc > ? AND created_utc < ? AND status != 'abandoned'"
            " AND (red_team IN (?, ?) OR blue_team IN (?, ?))"
            " AND NOT (red_team IN (?, ?) AND blue_team IN (?, ?)) LIMIT 1",
            (after_utc, before_utc, a, b, a, b, a, b, a, b)).fetchone()
        return row is not None

    def _series_teams(self, series_id: int) -> tuple[str, str] | None:
        row = self.conn.execute("SELECT team_a, team_b FROM series WHERE id = ?", (series_id,)).fetchone()
        return (row["team_a"], row["team_b"]) if row else None

    def refresh_series(self, series_id: int) -> None:
        """Recompute game numbers and wins from the series' matches."""
        self.conn.execute(
            "UPDATE matches SET series_id = NULL, series_game = NULL WHERE series_id = ? AND status = 'abandoned'",
            (series_id,))
        teams = self._series_teams(series_id)
        members = self.conn.execute(
            "SELECT * FROM matches WHERE series_id = ? ORDER BY created_utc, id", (series_id,)).fetchall()
        if teams is None or not members:
            self.conn.execute("DELETE FROM series WHERE id = ?", (series_id,))
            self.conn.commit()
            return
        wins = dict.fromkeys(teams, 0)
        for game, m in enumerate(members, 1):
            self.conn.execute("UPDATE matches SET series_game = ? WHERE id = ?", (game, m["id"]))
            red, blue = m["red_score"], m["blue_score"]
            if m["status"] == "complete" and red is not None and blue is not None and red != blue:
                winner = m["red_team"] if red > blue else m["blue_team"]
                if winner in wins:
                    wins[winner] += 1
        self.conn.execute(
            "UPDATE series SET games = ?, team_a_wins = ?, team_b_wins = ?, first_utc = ?, last_utc = ?"
            " WHERE id = ?",
            (len(members), wins[teams[0]], wins[teams[1]], members[0]["created_utc"],
             members[-1]["ended_utc"] or members[-1]["created_utc"], series_id))
        self.conn.commit()

    def series_summary(self, match_id: int) -> dict | None:
        row = self.conn.execute(
            "SELECT s.*, m.series_game FROM matches m JOIN series s ON s.id = m.series_id WHERE m.id = ?",
            (match_id,)).fetchone()
        if row is None:
            return None
        return {"series_id": row["id"], "game": row["series_game"],
                "wins": {row["team_a"]: row["team_a_wins"], row["team_b"]: row["team_b_wins"]}}
