# Stream Tracker — live OCR of EVE Alliance Tournament matches → SQLite

## Context
Want a tool that watches a live Twitch stream (EVE AT broadcast), samples 1 frame/sec, detects when a match is on (vs desk segments, ads, interstitials), and records per match: team names, ship bans, ship types per side, and when each ship dies. Output to SQLite. Must keep up in real time on a live stream; also needs a replay mode on recordings for dev/testing.

Project dir `/home/ian/dev/eve/streamTracker` is empty. Sibling `../scrim-trimmer` already does 1fps OpenCV frame sampling + pytesseract OCR (`src/frame_extractor.py`, `src/ocr_processor.py`) — reuse its preprocessing patterns. `../tourny-prediction-stats` has a SQLite layer (`src/db`) — reuse its style. Python 3.13, ffmpeg + tesseract installed, RTX 5080 available, yt-dlp at `~/Downloads/tmp-yt/yt-dlp_linux`.

## Findings from the VOD (v2860628871, 1h40m–2h02m, 1080p60)
Downloaded segment: `/tmp/claude-1000/-home-ian-dev-eve-streamTracker/eea304be-02d8-4c89-8ee6-e4fd0c2303de/scratchpad/vod/seg.mp4` (match = Neon Nightmares vs Sedition.).

Sequence observed: in-match (prev match) → raw client cam → ad/promo bumpers → desk cam with schedule ticker → desk with team-comp lower-third ("RED TEAM", ship icons, 00:00 boxes) → **match HUD appears with timer `-00:00-`** (pre-countdown, can last a while) → **yellow countdown `0:30`→`0:00`** → **white `10:00`→`0:00`** → HUD held briefly at 0:00 → hard cut to promos/other content.

Match HUD (fixed overlay, bottom 1/3 of 1920×1080, y≈745–1080):
- Header bar y≈750–778: red team name + score (x≈560–810) | **timer centered x≈900–1020** | blue score + team name (x≈1095–1300). Score = points killed.
- **Ship Bans**: left list x≈10–200, right list x≈1700–1910, starting y≈785, ~20px pitch. This match had 3 per side, but there can be **up to 10 per side**, so the list can run down to y≈985. The ROI must stop just above the "MATCH COMMENTARY BY" text on the left (y≈1000) and the EVE logo on the right (y≈1010). Count bans by checking which rows have text in them.
- Roster table: header row y≈844 (`PTS NAME SPEED … SHIP` / `SHIP … SPEED NAME PTS`), then rows starting y≈870, pitch ≈18px, up to ~10 rows/side, sorted by PTS desc.
  - Red cols: PTS x≈355–385, NAME 395–540, SPEED 545–590, bars 595–720, SHIP 735–830.
  - Blue cols: SHIP 1090–1195, bars 1200–1325, SPEED 1330–1375, NAME 1380–1525, PTS 1530–1560.
- **Dead ship = row greys out**: text drops from ~white to dim grey, SPEED goes blank, health bars vanish. Pixel-detectable without OCR.
- Ship names sometimes use HUD abbreviations: `F. Cyclone` (= Cyclone Fleet Issue), `F. Scythe`, and `N. Magnate` (= Magnate Navy Issue). It isn't consistent, so the same ship may also show its full name. Pilot names are truncated with `…`.
- **HUD PTS = sheet base points + inflation for duplicate hulls**. Examples: 2× Sleipnir shows 36 = 33 + 3, Svipul shows 11, 2× Arbitrator shows 14. A single Maulus shows 7 = base. This lets us cross-check the ship name we read, and pick between look-alikes such as Maulus (7) vs Maulus Navy Issue.
- Static anchors for HUD-present detection: "Ship Bans" labels, "BATTLEFEED", red/blue gradient header bars, "DEFENSE/ATTACK/CONTROL".

## Ship vocabulary (source of truth)
Google Sheet `1AVYlWlvuMKnA3yuqqDCcAkia8pvhpb9OBcM29WFw5rM`, tab **"New Static Values (Do Not Edit)"** (gid `284772315`), columns F–J: `Ship Name, Ship Class, Points, Hull Type, Inflation Value`. It lists 279 ships, 44 of them Navy/Fleet variants. The tab can be fetched without auth as CSV from `…/export?format=csv&gid=284772315`. A `stream-tracker update-ships` command downloads it to `data/ship_points.csv` and a copy is committed, so the tool works offline. The ship list is not hard-coded anywhere else.

## Architecture (Python package `stream_tracker`)
```
streamTracker/
  pyproject.toml            deps: numpy, opencv-python, pytesseract, rapidfuzz, streamlink
  config.example.toml       channel, db path, quality, debug dir, thresholds
  data/ship_points.csv      ship name, class, points, hull, inflation (from the AT points sheet)
  data/ship_aliases.toml    manual overrides only; abbreviations are generated (see vocab.py)
  src/stream_tracker/
    capture.py   frame sources
    layout.py    ROI table (normalized 0–1 coords, so 720p/1080p both work)
    hud.py       HUD-present + timer-phase detection (pixel/template, no OCR)
    ocr.py       tesseract wrappers + preprocessing + crop-hash cache
    vocab.py     fuzzy ship/team matching (rapidfuzz) + alias expansion
    reader.py    frame -> FrameReading dataclass
    tracker.py   match state machine + temporal voting + death debounce
    db.py        schema + writes
    cli.py       `live`, `replay`, `calibrate`
  tests/fixtures/  ~30 PNG frames cut from the VOD + expected JSON
```

### 1. Capture (`capture.py`)
- **LiveSource**: `streamlink --twitch-disable-ads --stdout twitch.tv/<channel> 1080p60,best` piped into `ffmpeg -i - -vf fps=1 -f rawvideo -pix_fmt bgr24 -` ; read fixed-size frames from stdout in a reader thread, push to a `queue.Queue(maxsize=2)` dropping oldest so processing never lags behind live. Each frame stamped with UTC wall time + monotonic stream offset.
- Auto-reconnect with backoff when stream drops/ends; log gap.
- **FileSource**: same ffmpeg pipeline over a file (`-re` optional) for replay; timestamps = file offset. Also accept a VOD URL via yt-dlp.
- Ads: `--twitch-disable-ads` skips Twitch-inserted ad segments; broadcaster promos/desk are handled by HUD detection.

### 2. Match-presence + phase (`hud.py`) — cheap, every frame
- `hud_present(frame)`: compare a few anchor ROIs ("Ship Bans" ×2, "BATTLEFEED", header gradient color means) against reference templates captured with `calibrate` (normalized cross-correlation, threshold ~0.8). Require ≥3 of 4 anchors.
- `timer_phase(frame)`: on timer ROI, classify text color: white `-00:00-` (dashes present) → PRE; **yellow** text → COUNTDOWN; white digits → LIVE. Then OCR digits (whitelist `0123456789:-`, PSM 7) → seconds.

### 3. Frame reading (`reader.py`, `ocr.py`)
Produces `FrameReading{ts, hud, phase, clock_s, teams{red,blue}, scores, bans{red[],blue[]}, rows{red[],blue[]}}` where row = `{slot, ship_raw, ship, pilot, pts, alive}`.
- Preprocess crops: grayscale, 3× upscale (INTER_CUBIC), Otsu/threshold, invert to black-on-white; tesseract PSM 7.
- **Alive detection is pixel-based**: per row, max/95th-percentile brightness of SHIP text crop + presence of health-bar pixels in bar ROI. Alive ≈ text >200 & bars present; dead ≈ text <170 & no bars. Thresholds in config, set by `calibrate`.
- OCR cost control: team names, bans, ship types, pilots only change at match start → OCR them only while `crop_hash` changes or until N=5 consistent votes, then lock. Per steady-state frame OCR = timer + 2 scores only (~3 tesseract calls, well under 1s). Use a small ThreadPool if needed.
- `vocab.py`:
  - Build the set of names to match against from `ship_points.csv`. For every `X Navy Issue` add `N. X`; for every `X Fleet Issue` add `F. X`. Also add a generic short form for faction-prefixed names, e.g. `Republic Fleet Firetail` → `R.F. Firetail`. Every form maps back to the full ship name.
  - Clean up the OCR text first: normalise `N.`/`F.` spacing and drop the `…`. Then fuzzy-match against the set with `rapidfuzz`.
  - When two candidates score within about 5 points of each other (e.g. `Maulus` vs `Maulus Navy Issue` when the `N.` was misread), pick the one whose points fit the row's PTS. A fit means `PTS == base + k·inflation` for k = 0..9.
  - If the best score is below 80, store the raw text with `ship_type = NULL` and flag it.
  - Bans use the same matching, without the points check.
  - Optionally, a list of known teams can be used to snap OCR'd team names.

### 4. State machine (`tracker.py`)
States: `IDLE → PRE → COUNTDOWN → LIVE → ENDED → IDLE`.
- IDLE→PRE: HUD present for 3 consecutive frames. Create match row (status=`pending`), start voting on teams/bans/rosters.
- PRE→COUNTDOWN: yellow timer seen 2 frames. Record `countdown_start_utc`.
- →LIVE: white timer ≤10:00 after countdown (or directly if we joined mid-match — then mark `joined_late=1`). Record `live_start_utc` (backdate from clock: `utc - (600 - clock_s)`).
- Death: row transitions alive→dead and stays dead for 2 consecutive frames → write death with `match_clock_s` (= 600 − clock) and `died_at_utc` of first dead frame. Cross-check: team score should increase by that ship's PTS; mismatch → `confidence='low'`.
- LIVE→ENDED when any of: clock reads 0:00 for 2 frames; all rows on one side dead (early wipe); HUD absent > `grace_s` (default 90s) while LIVE (`end_reason='hud_lost'`). Short HUD gaps (replays/cam cuts) under grace keep the match open; deaths discovered after a gap get `died_at` = first frame seen dead plus `gap_before_s` so uncertainty is visible.
- Match identity guard: if HUD reappears with different team names → close old, open new.
- Final: store final scores, `status='complete'`.

### 5. SQLite (`db.py`), WAL mode, write per event (crash-safe)
```sql
matches(id PK, channel, source, created_utc, countdown_start_utc, live_start_utc,
        ended_utc, end_reason, red_team, blue_team, red_score, blue_score,
        joined_late INT, status)
bans(id PK, match_id FK, side CHECK(side IN('red','blue')), ordinal 1..10, ship_raw, ship_type)
ships(id PK, match_id FK, side, slot, ship_raw, ship_type, match_confidence, pilot, points,
      died_match_clock_s NULL, died_utc NULL, died_stream_offset_s NULL,
      death_confidence, gap_before_s)
events(id PK, match_id, ts_utc, stream_offset_s, kind, payload_json)  -- audit log
```
Optional `--debug-dir`: save PNG of frame at each state change/death for manual audit.

### 6. CLI
- `stream-tracker live <channel> --db at.sqlite [--debug-dir]`
- `stream-tracker replay <file|vod-url> [--start 1:40:00] --db test.sqlite [--speed max]`
- `stream-tracker calibrate <frame.png>`: draws ROIs over a frame, dumps anchor templates + threshold suggestions; used if the overlay changes between tournaments.

## Implementation order
1. layout.py + calibrate on the fixture frames (verify ROIs visually).
2. hud.py presence/phase, test on fixtures (desk, ad, PRE, countdown, live, 0:00).
3. ocr.py/reader.py/vocab.py with fixture expected JSON.
4. tracker.py + db.py; replay mode end-to-end.
5. LiveSource + reconnect; live soak test.

## Verification
- **Unit tests** (`pytest`): fixture frames cut from `seg.mp4` at known offsets (e.g. 540 desk, 590 PRE, 612/628 countdown, 655/662 live clean, 1000 live with deaths, 1260 end 0:00, 1266 post-match promo). Assert hud/phase/clock, team names `NEON NIGHTMARES`/`SEDITION.`, bans `[Zarmazd, Ashimmu, Curse]` / `[Crucifier, Armageddon, Sentinel]`, rosters (`Sleipnir, Sleipnir, Basilisk, Khizriel, Scythe Fleet Issue, Tholos, Svipul, Svipul, Caracal, Raptor` / `Damnation, Guardian, Cyclone Fleet Issue, Cyclone, Huginn, Arbitrator, Arbitrator, Keres, Maulus, Maulus`), and alive flags at 1000s.
- **vocab tests**:
  - `N. Magnate`, `Magnate Navy Issue` and `N Magnate` (OCR lost the dot) all resolve to `Magnate Navy Issue`.
  - `F. Cyclone` → `Cyclone Fleet Issue`.
  - `Maulus` with PTS 7 → `Maulus`.
  - Every one of the 279 names in the sheet resolves to itself.
- **Bans >3**: build a synthetic HUD frame with 10 bans per side to exercise the ban ROI. Also find a VOD match with more than 3 bans if there is one.
- **Replay e2e**: `replay seg.mp4` → DB has exactly 1 complete match ended by time, final score 114–128, 8 red deaths + 4 blue deaths (per frame at 1260; sums check: blue dead PTS 33+30+26+25=114, red dead PTS =128), each with monotonic death clocks; previous partial match at file start recorded as `joined_late`.
- **Early-wipe case**: find another VOD match ending by annihilation and confirm `end_reason='wipe'`.
- **Real-time check**: replay with `-re` and log per-frame processing time; p99 must be < 1000ms. Then run `live` on the channel during a broadcast and compare DB against the VOD afterwards.

## Implementation notes (deviations from the plan above)
- **Digits use glyph templates, not tesseract.** Tesseract reads the HUD's slashed zero as 8. Templates (`src/stream_tracker/glyphs.py`, built by `tools/build_glyphs.py`) cut frame time from ~3.5s to ~5ms. Tesseract is used only for names, once per match, in a background thread.
- **Score = 200 − opponent's surviving points.** A roster under 200 points gives the opponent a head start: Arctic Light brought 199, so Of Essence started on 1 and finished on 86 with 85 in kills. Death confidence checks this exactly.
- **Countdown's last frame can render as a white 0:00.** Live start waits for 2 running-clock readings.
- `config.example.toml` was not added; everything is a CLI flag.
- Fixtures store only the HUD strip (y≥740) to keep the repo small.
- **Lessons from running all six AT XXII VODs** (202 matches). Each one was fixed and has a regression test where a frame was available:
  - The countdown yellow varies: blue/red ratio 0.2–0.69 (cutoff now 0.8).
  - The first countdown frame can read 0:00.
  - TiDi overtime shows as a red game clock after the white clock ends.
  - Ships can die a few seconds after 0:00.
  - The overlay can fade in or out and be drawn dimmed.
  - The overlay can get stuck showing both teams wiped (200–200) for a minute.
  - Team names can change mid-match, or be 25 capital Is.
  - Pilot names overflow into the PTS column.
  - Caster graphics get drawn over the overlay.
  - Short ship names ("Naga") return nothing from tesseract in line mode.
  - No team made more than 3 bans in AT XXII.
- **Series (Bo3/Bo5).** Matches repeating a team pairing are linked into a `series` row (`matches.series_id`, `series_game`). The link is made when identity settles, and wins are recomputed at each match end, reopen or abandon. A repeat has to come within 6h of the series' last game, and each team must share ≥3 pilots with an earlier game. That pilot check exists because on day 6 (2873111426, m4/m5) the overlay kept CRITICAL. MASS. vs THE SKELETON CREW for a match between two other teams.
- **Ban order from the draft panel.** The HUD ban lists aren't in pick order (2873111426 m75: HUD red Zarmazd, Armageddon, Armageddon Navy; picked Armageddon, Zarmazd, Armageddon Navy). The pre-match draft panel has them in order (`draft.py`). Its layout differs by year: 6 wide cards with bars at y≈790 in AT XXII, 8 narrower ones at y≈826 in AT XXI. So the cards are located from their magenta/cyan bars. `--save-hud` also keeps panel frames.
