# Stream Tracker

Watches an EVE Alliance Tournament broadcast on Twitch at 1 frame/sec. It tells match play apart from desk segments, ads and promos, and records each match to SQLite:
- team names
- ship bans (up to 10 per side)
- every ship's type, pilot and points
- when each ship dies

It works on a live stream and on recordings/VODs.

## Setup

Needs `ffmpeg` and `tesseract` on the system.

```bash
python -m venv env
env/bin/pip install -e '.[dev]'
```

## Usage

```bash
# live
env/bin/stream-tracker live eveonline --db at.sqlite

# a recording (or a VOD URL, which needs yt-dlp on PATH)
env/bin/stream-tracker replay match.mp4 --start 1:40:00 --start-utc 2026-10-18T15:00:00+00:00 --db test.sqlite

# check the HUD regions against a 1080p screenshot (e.g. if the overlay changes)
env/bin/stream-tracker calibrate frame.png --out calibrate.png [--save-anchors]

# group matches in older databases into series
env/bin/stream-tracker series runs/*.sqlite

# refresh the ship list/points from the AT points sheet
env/bin/stream-tracker update-ships
```

`--debug-dir DIR` saves a frame at every match event (open, live, death, end) for auditing.

### Sending matches to the observed-import endpoint

`--post` (on `live` and `replay`) also sends each match, as soon as it ends, to the observed-import endpoint: one match object per POST, built from what the tracker just recorded. The local database is still written as usual.

The endpoint URL and import token are read from the environment and must never be committed. Put them in `.env` (gitignored):

```bash
OBSERVED_IMPORT_URL=...
OBSERVED_IMPORT_TOKEN=...
```

```bash
set -a; . ./.env; set +a
env/bin/stream-tracker live eveonline --db at.sqlite --post --run-id at22-day1
env/bin/stream-tracker replay runs/2873111426 --start 8:13:00 --db test.sqlite --post
```

`--run-id` and the local match id make the import key (`run_id:match_id`); posting the same match again replaces it. It defaults to the VOD id in a replay input, or `<channel>-<UTC date>` when live. Posts run in the background and retry server and network errors; a failure is logged and never stops tracking. `--start` and `--duration` also work on a `--save-hud` directory.

## How it works

| What | How |
|---|---|
| Match on screen? | Normalized correlation of the "Ship Bans" ×2 and "BATTLEFEED" labels against stored anchors (`data/anchors.npz`) |
| Phase | Timer text: `-00:00-` = pre-match, yellow = 30s countdown, white = 10:00 match clock |
| Clock, scores, PTS | Glyph template matching (`data/glyphs_*.npz`). Tesseract misreads the HUD's slashed zero as 8 |
| Alive / dead | Row brightness: dead rows grey out and lose their health bars |
| Names (teams, bans, ships, pilots) | Tesseract, run in a background thread until 3 reads agree, then locked |
| Ship types | Fuzzy match against `data/ship_points.csv`; `N. X` / `F. X` / cut-off names are handled, and PTS breaks ties |

A match ends when:
- the clock has run out (`time`),
- one side is fully dead (`wipe`), or
- the HUD is gone longer than `--grace` seconds (`hud_lost`). In AT XXII this only happened when the broadcast cut away mid-match and never came back.

If a `hud_lost` match comes back with the same teams and the clock still counting down, it is reopened.

**Time dilation:** the white clock counts real time. Under heavy TiDi the game runs slower, so when the white clock hits 0:00 the HUD switches to a red `m:ss TiDi:nnn%` clock with the game time still left, and play continues. The tracker follows that red clock (read by tesseract, since its font differs) and records the extra time as `tidi_overtime_s`. With mild TiDi, ships can still die a few seconds after 0:00 without a red clock appearing, so a `time` ending waits 10 frames after 0:00 before closing (`ended_utc` is still the first 0:00).

Death times:
- `died_match_s`: real seconds since the match went live (can exceed 600 with TiDi).
- `died_clock_s`: the clock the HUD showed.
- `died_in_tidi`: 1 if that was the red TiDi clock.
- `died_utc` and `died_stream_offset_s`: absolute and stream time.

**Things the tracker works around:**
- **Look-alike ads:** overlay shown for under 25s is dropped, and a mid-match overlay with other teams or an inconsistent clock is ignored for up to 25s.
- **Caster graphics over the overlay:** names only count when 3 reads agree.
- **Faded or dimmed overlay:** alive/dead is judged relative to the column headings' brightness.
- **Stale overlay showing both teams wiped:** ignored.
- **Team names changed mid-match:** the match is recognised by its rosters' points.
- **Pilot names spilling into the PTS column.**
- **In-game labels showing just above the timer.**

**Final scores** are taken as 200 minus the opponent's surviving points whenever every roster PTS was read, since the score digits can stop reading cleanly on a soft encode (AT XXI day 3 on YouTube). A disagreement with the last score read is logged as a `score_mismatch` event.

**Bans** not in the ship list (e.g. Cenotaph in AT XXI) are kept as raw text with `ship_type = NULL`; the rest of that side's bans still resolve.

`death_confidence` is one of:
- `high`: the opponent's score equals 200 minus this side's surviving points (how AT scoring works). A side that brings under 200 points hands its opponent a head start, e.g. a 199-point roster means the opponent starts on 1.
- `low`: the score doesn't match yet. It is upgraded once it does.
- `unknown_time`: the ship was already dead when tracking started, e.g. when joining mid-match.

**Series (Bo3/Bo5):** a match whose team pairing was already played counts as the next game of that series, even with other matches played in between and with the teams on swapped sides. It is linked as soon as the names are read, before the countdown ends, and the series score (`series.team_a_wins`/`team_b_wins`) updates when each game ends. A repeat only joins the series if:
- the series' last game was within 6 hours, and neither team has played anyone else since; otherwise it starts a new series (e.g. a group-stage game and a later final between the same teams);
- each team fields at least 3 of the same pilots as an earlier game. If not, the overlay was probably still showing the previous match's team names (seen once in AT XXII), so the match stays out of the series and a `series_mismatch` event is logged.

`matches.series_id` and `series_game` give each match's place in its series. `stream-tracker series DB...` adds series to databases recorded before this existed.

Team names are snapped to `data/teams.txt` (the AT XXII teams) and to names already in the database when they are ≥88% similar, which fixes OCR slips and overlay typos.

## Tests

```bash
env/bin/python -m pytest
# full replay of the reference VOD segment (v2860628871, 1:40:00-2:02:00):
STREAM_TRACKER_VOD=seg.mp4 env/bin/python -m pytest -k reference
```

`tools/build_glyphs.py` rebuilds the digit templates from that segment.
