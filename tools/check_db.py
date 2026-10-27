"""Consistency checks over tracker databases.

usage: python tools/check_db.py runs/*.sqlite
"""

import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stream_tracker.vocab import default_vocab  # noqa: E402

TEAM_POINTS = 200


def ts(s):
    return datetime.fromisoformat(s) if s else None


def check(path: str, issues: list, teams: set) -> list[dict]:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    vocab = default_vocab()
    matches = [dict(m) for m in conn.execute("SELECT * FROM matches ORDER BY id")]
    db = Path(path).stem

    def flag(m, msg):
        issues.append(f"{db} m{m['id']} {m['red_team']} vs {m['blue_team']}: {msg}")

    seen_pairs = defaultdict(list)
    for m in matches:
        teams.update(t for t in (m["red_team"], m["blue_team"]) if t)
        ships = [dict(s) for s in conn.execute(
            "SELECT * FROM ships WHERE match_id=? ORDER BY side DESC, slot", (m["id"],))]
        bans = [dict(b) for b in conn.execute("SELECT * FROM bans WHERE match_id=?", (m["id"],))]
        seen_pairs[frozenset((m["red_team"], m["blue_team"]))].append(m)

        if m["status"] != "complete":
            flag(m, f"status {m['status']}")
        if m["end_reason"] not in ("time", "wipe"):
            flag(m, f"end_reason {m['end_reason']}")
        if m["joined_late"]:
            flag(m, "joined late")
        if not m["red_team"] or not m["blue_team"]:
            flag(m, "missing team name")

        live, end = ts(m["live_start_utc"]), ts(m["ended_utc"])
        if live and end:
            dur = (end - live).total_seconds() - (m["tidi_overtime_s"] or 0)
            if m["end_reason"] == "time" and abs(dur - 600) > 5:
                flag(m, f"ended on time but lasted {dur:.0f}s (excluding TiDi overtime)")
            if m["end_reason"] == "wipe" and dur > 605:
                flag(m, f"wipe after {dur:.0f}s (excluding TiDi overtime)")
        if m["countdown_start_utc"] and live:
            cd = (live - ts(m["countdown_start_utc"])).total_seconds()
            if abs(cd - 30) > 3:
                flag(m, f"countdown to live {cd:.0f}s, expected 30")

        by_side = {"red": [s for s in ships if s["side"] == "red"],
                   "blue": [s for s in ships if s["side"] == "blue"]}
        for side, rows in by_side.items():
            other = "blue" if side == "red" else "red"
            if not rows:
                flag(m, f"no {side} ships")
                continue
            total = sum(s["points"] or 0 for s in rows)
            if total > TEAM_POINTS:
                flag(m, f"{side} roster totals {total} points (> {TEAM_POINTS})")
            for s in rows:
                where = f"{side} slot {s['slot']} {s['ship_raw']!r}"
                if s["points"] is None:
                    flag(m, f"{where}: no points read")
                if not s["ship_type"]:
                    flag(m, f"{where}: ship not in list")
                elif s["points"] is not None and not vocab.ships[s["ship_type"]].fits_points(s["points"]):
                    flag(m, f"{where}: {s['ship_type']} can't be worth {s['points']}")
                if s["death_confidence"] == "low":
                    flag(m, f"{where}: death score mismatch")
                if s["died_utc"] and s["died_match_s"] is None:
                    flag(m, f"{where}: death without match time")
                limit = 600 + (m["tidi_overtime_s"] or 0)
                if s["died_match_s"] is not None and not 0 <= s["died_match_s"] <= limit + 10:
                    flag(m, f"{where}: died at {s['died_match_s']}s")
                elif s["died_match_s"] is not None and s["died_match_s"] > limit:
                    flag(m, f"note: {where} died {s['died_match_s'] - limit}s after the clock ran out")
            # Final score must be 200 minus the opponent's surviving points.
            surviving = sum(s["points"] or 0 for s in by_side[other] if s["death_confidence"] is None)
            score = m[f"{side}_score"]
            if by_side[other] and score is not None and score != TEAM_POINTS - surviving:
                flag(m, f"{side} score {score} != 200 - {other} surviving {surviving}")
        if m["end_reason"] == "wipe" and not any(
            rows and all(s["death_confidence"] for s in rows) for rows in by_side.values()
        ):
            flag(m, "wipe but neither side fully dead")

        for side in ("red", "blue"):
            side_bans = [b for b in bans if b["side"] == side]
            if not side_bans:
                flag(m, f"note: no {side} bans (check the VOD; teams may ban nothing)")
            for b in side_bans:
                if not b["ship_type"]:
                    flag(m, f"{side} ban {b['ship_raw']!r} not in list")
        banned = {b["ship_type"] for b in bans if b["ship_type"]}
        for s in ships:
            if s["ship_type"] in banned:
                flag(m, f"{s['side']} flew banned {s['ship_type']}")

    for pair, ms in seen_pairs.items():
        if len(ms) > 1 and None not in pair:
            ids = [m["id"] for m in ms]
            names = " vs ".join(sorted(pair))
            series = {m.get("series_id") for m in ms}
            loose = [m["id"] for m in ms if m.get("series_id") is None]
            if loose:
                issues.append(f"{db}: {names} appears in matches {ids}; {loose} in no series"
                              " (stale team names? pilots differ)")
            for sid in sorted(series - {None}):
                games = [m["id"] for m in ms if m.get("series_id") == sid]
                if len(games) > 1:
                    issues.append(f"{db}: note: series {names}, matches {games}")

    has_series = conn.execute("SELECT 1 FROM sqlite_master WHERE name='series'").fetchone()
    for s in conn.execute("SELECT * FROM series") if has_series else ():
        games = [m["series_game"] for m in matches if m.get("series_id") == s["id"]]
        if sorted(games) != list(range(1, len(games) + 1)) or len(games) != s["games"]:
            issues.append(f"{db} series {s['id']}: game numbers {sorted(games)} for {s['games']} games")
        if max(s["team_a_wins"], s["team_b_wins"]) > 3:
            issues.append(f"{db} series {s['id']}: {s['team_a_wins']}-{s['team_b_wins']}, longer than a Bo5")
    return matches


def main():
    issues: list[str] = []
    teams: set[str] = set()
    total = 0
    for path in sys.argv[1:]:
        ms = check(path, issues, teams)
        total += len(ms)
        print(f"{Path(path).stem}: {len(ms)} matches")
    for a, b in combinations(sorted(teams), 2):
        if fuzz.ratio(a.lower(), b.lower()) >= 85:
            issues.append(f"similar team names: {a!r} / {b!r}")
    print(f"\n{total} matches, {len(issues)} issues")
    for i in issues:
        print(" -", i)


if __name__ == "__main__":
    main()
