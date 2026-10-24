"""SQLite storage. Every write commits immediately so a crash loses nothing."""

import json
import sqlite3
from pathlib import Path

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
    status TEXT NOT NULL        -- pending | live | complete | abandoned
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


class Database:
    def __init__(self, path: str | Path):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

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
        for table in ("events", "ships", "bans"):
            self.conn.execute(f"DELETE FROM {table} WHERE match_id = ?", (match_id,))
        self.conn.execute("DELETE FROM matches WHERE id = ?", (match_id,))
        self.conn.commit()

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
