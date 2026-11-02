"""Match state machine: feeds 1 fps frames in, writes matches/ships/deaths out.

    IDLE -> PRE (-00:00-) -> COUNTDOWN (yellow) -> LIVE (white 10:00)
         [-> TiDi overtime (red game clock)] -> ENDED -> IDLE

Any of those can be skipped when the stream joins late. A match ends when
every ship on one side is dead, when the clock has run out (white 0:00, or the
red TiDi clock at 0:00), or when the HUD has been off screen for longer than
grace_s.

The white clock counts real time. Under time dilation the game runs slower,
so when it reaches 0:00 the HUD switches to a red clock with the game time
still remaining, and play continues.
"""

import json
import logging
import re
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from rapidfuzz import fuzz

from . import draft, layout as L, reader
from .db import Database
from .hud import Phase
from .reader import SIDES, FrameReading, Identity
from .vocab import KNOWN_TEAMS_FILE, TEAM_SNAP_SCORE, KnownTeams, ShipVocab, TeamNames, default_vocab, looks_like_name

log = logging.getLogger(__name__)

MATCH_LENGTH_S = 600
TEAM_POINTS = 200  # a team's score is 200 minus the opponent's surviving points
HUD_ON_FRAMES = 3  # consecutive HUD frames before a match is opened
HUD_OFF_FRAMES = 3  # consecutive non-HUD frames before an ended match is let go
DEATH_FRAMES = 2  # consecutive dead frames before a death is recorded
# A row greys out this long after the ship died, by the match clock. Measured
# against the AT XXII finals' official death times (day 1, 347 deaths: the
# grey-out trailed by 3-5s in every match, median 4s). Taken off died_match_s
# only; died_utc and died_stream_offset_s stay the frame the row greyed.
DEATH_DISPLAY_LAG_S = 4
FROZEN_MIN_S = 3  # the clock seen showing one value for this long is a frozen picture
# After the clock reaches 0:00, wait this long for a TiDi clock to appear
# before ending the match (the end time is still the first 0:00).
ZERO_HOLD_FRAMES = 10
LIVE_STUCK_MAX_S = 30  # a white clock that hasn't moved in this long starts the match anyway
IDENTITY_VOTES = 3  # agreeing reads before a name is settled
IDENTITY_MAX_READS = 20
UNMATCHED = "?"  # prefixes raw ban text that isn't in the ship list, in votes
# Replays run far faster than 1 fps; wait for a pending name read once this
# many frames have gone by so names still get several reads per match.
IDENTITY_MAX_LAG_FRAMES = 5
SAME_TEAMS_MIN_CORR = 0.9
# Some ads use the same HUD; they last 20s or less. HUD that doesn't fit the
# current match (other teams, or a clock that disagrees with elapsed time) is
# ignored until it has persisted this long, and a match whose HUD was on
# screen for less than this in total is discarded.
FOREIGN_MAX_FRAMES = 25
MIN_MATCH_HUD_FRAMES = 25
# Longer ads replay a match clip: they start mid-clock with no countdown and
# cut away after a minute or so. A match that was never seen counting down and
# loses its HUD within this many HUD frames is one of those. Real matches cut
# off by the broadcast had their countdown and ran for 5+ minutes (AT XXI/XXII).
MIN_LATE_JOIN_HUD_FRAMES = 120
CLOCK_TOLERANCE_S = 3
# The pre-match draft panel gives the bans in pick order. It is used for the
# next match between its teams if that starts within this long of the panel
# last being on screen; a panel gone this long is a new one when it returns.
DRAFT_MAX_AGE_S = 1800
DRAFT_GAP_S = 60
DRAFT_MIN_BANS_TO_AGREE = 3  # fewer bans than this say too little to recognise a match by
BAN_MISREAD_MIN = 60  # name similarity of a draft ship and a HUD ship taken for one misread
OTHER = {"red": "blue", "blue": "red"}


@dataclass
class FrameTime:
    utc: datetime
    offset: float  # seconds since the source started


@dataclass
class Draft:
    """A settled read of the draft panel."""
    teams: dict[str, str]
    bans: list[tuple[str, str, str | None]]  # (side, raw, ship_type) in pick order
    last_seen: FrameTime


PRELIM_PHASE = "Qualifiers"


def same_team(a: str | None, b: str | None) -> bool:
    """Team names from the draft panel and the HUD: the same, allowing OCR
    slips and a name cut short on either."""
    if not (a and b):
        return False
    a, b = a.upper().rstrip(". "), b.upper().rstrip(". ")
    if fuzz.ratio(a, b) >= TEAM_SNAP_SCORE:
        return True
    return min(len(a), len(b)) >= 4 and fuzz.partial_ratio(a, b) >= 95


def _letters(text: str) -> str:
    """Lower-case letters and spaces of an OCR read, without edge specks."""
    return re.sub(r"[^a-z ]", "", text.lower()).strip()


def iso(t: datetime) -> str:
    return t.isoformat(timespec="seconds")


def channel_url(channel: str | None) -> str:
    """The Twitch channel page, the placeholder vod_url sent for live matches."""
    if channel and "://" in channel:
        return channel
    return f"https://www.twitch.tv/{channel or ''}"


def vod_link(base: str, offset: float) -> str:
    """A Twitch VOD URL that opens `offset` seconds in, e.g. ...?t=1h2m5s."""
    s = max(0, int(offset))
    sep = "&" if "?" in base else "?"
    return f"{base}{sep}t={s // 3600}h{s % 3600 // 60}m{s % 60}s"


@dataclass
class Slot:
    side: str
    slot: int
    pts: Counter = field(default_factory=Counter)  # PTS as read off the HUD; never derived from ship_points.csv
    seen_alive: bool = False
    dead_streak: int = 0
    first_dead: FrameTime | None = None
    first_dead_clock: int | None = None
    first_dead_tidi: bool = False
    first_dead_gap: float = 0.0
    dead: bool = False
    death_confidence: str | None = None

    @property
    def points(self) -> int | None:
        return self.pts.most_common(1)[0][0] if self.pts else None


@dataclass
class MatchState:
    id: int
    team_crops: np.ndarray  # team-name strip, to recognise this match again cheaply
    phase: Phase = Phase.PRE
    slots: dict[tuple[str, int], Slot] = field(default_factory=dict)
    votes: dict[tuple, Counter] = field(default_factory=dict)
    raw: dict[tuple, tuple] = field(default_factory=dict)  # OCR text behind each voted value
    identity_reads: int = 0
    identity_settled: bool = False
    saw_countdown: bool = False
    countdown_t: FrameTime | None = None
    countdown_dated: bool = False
    live_started: bool = False
    live_start_offset: float | None = None
    time_up: FrameTime | None = None  # first frame the clock showed a final 0:00
    zero_frames: int = 0
    tidi_start: FrameTime | None = None  # white clock ran out; red TiDi clock shown
    last_tidi_clock: int | None = None
    live_candidates: list = field(default_factory=list)  # (FrameTime, clock) before live_started
    last_clock: int | None = None
    last_clock_t: FrameTime | None = None
    # The white clock's value and the first and last frames seen showing it.
    # Unlike last_clock it isn't moved by re-anchoring, so a frozen clock
    # shows how long it froze.
    clock_shown: tuple[int, FrameTime, FrameTime] | None = None
    hud_frames: int = 0
    foreign_streak: int = 0
    last_hud: FrameTime | None = None
    last_scores: dict[str, int | None] = field(default_factory=dict)
    ended: bool = False
    end_reason: str | None = None
    series_id: int | None = None
    match_no: int | None = None  # number within the tournament phase, from when it went live
    written_teams: dict[str, str | None] = field(default_factory=dict)  # provisional names in the DB
    # What has been written for this match, kept in memory so the finished
    # match can be sent on (observed import) without reading the DB back.
    row: dict = field(default_factory=dict)
    bans: dict[str, list[tuple[str, str | None, int | None]]] = field(default_factory=dict)  # (raw, type, pick)
    # Bans from the draft panel as (side, raw, type, pick), once taken for this match.
    draft_bans: list[tuple[str, str, str | None, int]] | None = None
    ships: dict[tuple[str, int], dict] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)


def team_strip(frame: np.ndarray) -> np.ndarray:
    strips = [L.crop(frame, L.RED_TEAM), L.crop(frame, L.BLUE_TEAM)]
    return np.hstack([cv2.cvtColor(s, cv2.COLOR_BGR2GRAY) for s in strips]).astype(np.float32)


def same_teams(a: np.ndarray, b: np.ndarray) -> bool:
    res = cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)
    return float(np.nan_to_num(res).max()) >= SAME_TEAMS_MIN_CORR


class Tracker:
    def __init__(
        self,
        db: Database,
        channel: str | None = None,
        source: str | None = None,
        vocab: ShipVocab | None = None,
        grace_s: float = 90.0,
        debug_dir: str | Path | None = None,
        run_id: str | None = None,
        phase: str | None = None,
        first_match: int = 1,
        vod: str | None = None,
        known_teams: Path = KNOWN_TEAMS_FILE,
        on_match_end: Callable[[dict], None] | None = None,
        on_match_closed: Callable[[int], None] | None = None,
    ):
        self.db = db
        self.channel = channel
        self.source = source
        self.run_id = run_id
        self.phase = phase  # tournament phase sent with each import object; None lets the Feed card decide
        self.first_match = first_match  # lowest number for this phase's next match (continues a phase across DBs)
        self.vod = vod  # Twitch VOD URL the offsets are into; None when live
        self.on_match_end = on_match_end  # called with each finished match's import object
        self.on_match_closed = on_match_closed  # called with each finished match's id (zkill linking)
        self.vocab = vocab or default_vocab()
        self.teams = TeamNames.from_file()
        # Past the Qualifiers every team is known, so names are best-fit to the tournament's spelling.
        self.known_teams = (KnownTeams.from_file(Path(known_teams))
                            if phase not in (None, PRELIM_PHASE) else None)
        for row in db.conn.execute("SELECT red_team, blue_team FROM matches"):
            for name in row:
                self.teams.add(name)
        self.grace_s = grace_s
        self.debug_dir = Path(debug_dir) if debug_dir else None
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.match: MatchState | None = None
        self.hud_streak = 0
        self.no_hud_streak = 0
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._identity_job: Future | None = None
        self._identity_lag = 0
        self._frame: np.ndarray | None = None
        self._last_hud: tuple[np.ndarray, FrameReading] | None = None
        self._t: FrameTime | None = None
        # Team strip of the last match that finished, so a post-match replay
        # of the HUD isn't mistaken for a new match.
        self._finished_teams: np.ndarray | None = None
        self.draft: Draft | None = None
        self._draft_job: Future | None = None
        self._draft_lag = 0
        self._draft_votes: Counter = Counter()
        self._draft_bars = 0
        self._draft_settled = False
        self._draft_seen: FrameTime | None = None

    # ------------------------------------------------------------------ frames

    def process(self, frame: np.ndarray, t: FrameTime) -> FrameReading:
        self._frame, self._t = frame, t
        r = reader.read_frame(frame)
        if self._identity_job is not None:
            self._identity_lag += 1
            if self._identity_lag >= IDENTITY_MAX_LAG_FRAMES:
                wait([self._identity_job])
        self._collect_identity()
        if self._draft_job is not None:
            self._draft_lag += 1
            if self._draft_lag >= IDENTITY_MAX_LAG_FRAMES:
                wait([self._draft_job])
        self._collect_draft()
        if r.hud:
            self.hud_streak += 1
            self.no_hud_streak = 0
            self._on_hud(frame, r, t)
        else:
            self.hud_streak = 0
            self.no_hud_streak += 1
            self._check_draft(frame, t)
            self._on_no_hud(t)
        return r

    def close(self, t: FrameTime | None = None) -> None:
        """Flush at end of input. An unfinished match is left as-is."""
        if self._identity_job:
            wait([self._identity_job])
            self._collect_identity()
        if self.match and not self.match.ended:
            if self.match.time_up is not None:
                self._end(self.match.time_up, "time")
            elif self.match.live_started and t is not None:
                self._end(t, "hud_lost" if self.no_hud_streak else "stream_end")
            else:
                self._finish_identity(self.match)
        self._pool.shutdown(wait=True)

    def _on_hud(self, frame: np.ndarray, r: FrameReading, t: FrameTime) -> None:
        m = self.match
        if m is not None and not m.ended and not self._fits_match(m, frame, r, t):
            m.foreign_streak += 1
            if m.foreign_streak < FOREIGN_MAX_FRAMES:
                self._on_no_hud(t)
                return
            # Persisted too long to be an ad: the broadcast moved on.
            self._end(m.last_hud, "hud_lost")
            m = self.match
            if m is not None and m.last_clock is not None and r.clock is not None:
                m.last_clock, m.last_clock_t = r.clock, t  # e.g. the clock was paused
            if m is not None and not same_teams(m.team_crops, team_strip(frame)):
                self.match = m = None
        elif m is not None:
            m.foreign_streak = 0
        if m is not None and m.ended and (
            not same_teams(m.team_crops, team_strip(frame))
            or r.phase in (Phase.PRE, Phase.COUNTDOWN)  # rematch between the same teams
        ):
            self.match = m = None
        if m is not None and m.ended:
            # HUD still up after the match finished (e.g. held on 0:00).
            # A hud_lost ending is reopened if the same match comes back.
            if r.phase != Phase.LIVE:
                return
            if m.end_reason == "hud_lost" and (m.last_clock is None or (r.clock or 0) <= m.last_clock):
                self._reopen(t)
            elif m.end_reason == "time" and r.tidi and r.clock and m.tidi_start is None:
                self._reopen(t)  # the white clock ran out but play goes on under TiDi
            else:
                return
        if m is None:
            if self.hud_streak < HUD_ON_FRAMES:
                return
            if (r.phase not in (Phase.PRE, Phase.COUNTDOWN) and self._finished_teams is not None
                    and same_teams(self._finished_teams, team_strip(frame))):
                return
            m = self._open(frame, t)

        gap = (t.offset - m.last_hud.offset) if m.last_hud else 0.0
        gap = gap if gap > 2 else 0.0
        frozen = self._frozen_for(r, t)
        m.last_hud = t
        m.hud_frames += 1
        self._last_hud = (frame, r)
        self._maybe_read_identity(frame, r)
        self._update_phase(r, t)
        self._update_slots(r, t, max(gap, frozen))
        self._check_scores(r, t)
        self._check_end(r, t)

    def _frozen_for(self, r: FrameReading, t: FrameTime) -> float:
        """How long the picture was frozen before this frame, judged by the
        white clock: it ticks every second, so one that showed the same value
        for longer and has now moved on was a frozen picture (AT XXII finals
        day 1: ~20s). Deaths seen as it unfreezes happened during the freeze,
        so this counts as time the HUD was off screen. 0 otherwise."""
        m = self.match
        if (not m.live_started or r.phase != Phase.LIVE or r.tidi or r.clock is None
                or m.clock_shown is None or r.clock >= m.clock_shown[0]):
            return 0.0
        _, first, last = m.clock_shown
        if last.offset - first.offset < FROZEN_MIN_S:
            return 0.0  # ticking normally (or unreadable for a while, which isn't a freeze)
        return t.offset - first.offset - 1

    def _same_roster(self, m: MatchState, r: FrameReading) -> bool:
        """Do this frame's PTS columns match the rosters seen so far?"""
        known = agree = 0
        for (side, i), s in m.slots.items():
            if s.points is None:
                continue
            known += 1
            row = r.rows[side][i] if i < len(r.rows.get(side, [])) else None
            agree += row is not None and row.pts == s.points
        return known >= 4 and agree >= 0.8 * known

    def _fits_match(self, m: MatchState, frame: np.ndarray, r: FrameReading, t: FrameTime) -> bool:
        """False for HUD frames that can't belong to the match in progress."""
        strip = team_strip(frame)
        if not same_teams(m.team_crops, strip):
            if not self._same_roster(m, r):
                return False
            # Same rosters under a new team name: the caster fixed the name.
            m.team_crops = strip
        if m.live_started:
            if r.phase in (Phase.PRE, Phase.COUNTDOWN):
                return False
            if (r.phase == Phase.LIVE and not r.tidi and r.clock is not None
                    and m.last_clock_t is not None and m.tidi_start is None):
                expected = m.last_clock - (t.offset - m.last_clock_t.offset)
                if abs(r.clock - max(expected, 0)) > CLOCK_TOLERANCE_S:
                    # With the same rosters, a clock that hasn't gone back up is
                    # this match after a freeze or pause (clock behind: AT XXII
                    # finals, the picture froze at 6:57 for ~30s) or a skip in
                    # the stream (clock ahead: the VOD drops a few seconds), not
                    # a clip of an earlier moment. Re-anchor on it.
                    if r.clock <= m.last_clock + 1 and self._same_roster(m, r):
                        m.last_clock, m.last_clock_t = r.clock, t
                        return True
                    return False
        return True

    def _on_no_hud(self, t: FrameTime) -> None:
        m = self.match
        if m is None:
            return
        if m.ended:
            if self.no_hud_streak >= HUD_OFF_FRAMES and m.end_reason != "hud_lost":
                self.match = None
            return
        if m.time_up is not None:
            self._end(m.time_up, "time")
            return
        if m.last_hud and t.offset - m.last_hud.offset > self.grace_s:
            if m.live_started:
                self._end(m.last_hud, "hud_lost")
            elif not self._discard_if_brief(m):
                self._finish_identity(m)  # record the teams (and their colours) anyway
                self._update(m, status="abandoned")
                if m.series_id is not None:
                    self.db.refresh_series(m.series_id)
                self._event(t, "abandoned")
                self.match = None

    # ---------------------------------------------------------------- lifecycle

    def _open(self, frame: np.ndarray, t: FrameTime) -> MatchState:
        mid = self.db.create_match(self.channel, self.source, iso(t.utc), self.phase)
        self.match = MatchState(id=mid, team_crops=team_strip(frame), last_hud=t)
        self.match.row = dict(channel=self.channel, source=self.source, created_utc=iso(t.utc),
                              joined_late=0, status="pending")
        self._event(t, "match_opened")
        log.info("match %d opened at %.0fs", mid, t.offset)
        return self.match

    def _discard_if_brief(self, m: MatchState, reason: str | None = None) -> bool:
        """Delete a match whose HUD was barely on screen (an ad, not a match):
        under MIN_MATCH_HUD_FRAMES, or a late-joined clip that lost the HUD
        within MIN_LATE_JOIN_HUD_FRAMES."""
        clip = reason == "hud_lost" and not m.saw_countdown and m.hud_frames < MIN_LATE_JOIN_HUD_FRAMES
        if m.hud_frames >= MIN_MATCH_HUD_FRAMES and not clip:
            return False
        if self._identity_job is not None:
            wait([self._identity_job])
            self._identity_job = None
        self.db.delete_match(m.id)
        log.info("match %d discarded: HUD on screen for only %ds%s", m.id, m.hud_frames,
                 " after joining mid-clock (a match-clip ad)" if clip else "")
        if self.match is m:
            self.match = None
        return True

    def _reopen(self, t: FrameTime) -> None:
        m = self.match
        m.ended = False
        m.end_reason = None
        m.time_up = None
        m.zero_frames = 0
        m.foreign_streak = 0
        if m.identity_reads < IDENTITY_MAX_READS:
            m.identity_settled = False  # keep voting; names settled early on few reads
        self._update(m, status="live", ended_utc=None, end_reason=None)
        if m.series_id is not None:
            self.db.refresh_series(m.series_id)  # its result no longer counts
        self._event(t, "match_reopened")

    def _end(self, t: FrameTime, reason: str) -> None:
        m = self.match
        if self._discard_if_brief(m, reason):
            return
        self._finish_identity(m)
        m.ended = True
        m.end_reason = reason
        self._finished_teams = m.team_crops
        scores = dict(m.last_scores)
        for side in SIDES:
            # A score is 200 minus the opponent's surviving points. Prefer that
            # to the last score read: on a soft encode the score digits can
            # stop reading cleanly well before the end (AT XXI day 3 on YouTube).
            other = [s for s in m.slots.values() if s.side != side]
            if other and all(s.points is not None for s in other):
                derived = TEAM_POINTS - sum(s.points for s in other if not s.dead)
                if scores.get(side) != derived:
                    self._event(t, "score_mismatch", {"side": side, "read": scores.get(side), "derived": derived})
                scores[side] = derived
        overtime = None
        if m.tidi_start is not None:
            overtime = round(t.offset - m.tidi_start.offset)
        self._update(
            m,
            tidi_overtime_s=overtime,
            ended_utc=iso(t.utc),
            end_reason=reason,
            red_score=scores.get("red"),
            blue_score=scores.get("blue"),
            status="complete",
        )
        self._event(t, "match_ended", {"reason": reason, "scores": scores})
        log.info("match %d ended (%s) %s", m.id, reason, scores)
        if m.series_id is not None:
            self.db.refresh_series(m.series_id)
            series = self.db.series_summary(m.id)
            if series:
                # Only the game number: local series ids are per database, not global.
                m.row["series_game"] = series["game"]
                self._event(t, "series_updated", series)
                log.info("match %d: series %d game %d, %s", m.id, series["series_id"],
                         series["game"], series["wins"])
        if self.on_match_end is not None:
            try:
                self.on_match_end(self.export(m))
            except Exception:
                log.exception("match %d: sending to observed import failed", m.id)
        if self.on_match_closed is not None:
            try:
                self.on_match_closed(m.id)
            except Exception:
                log.exception("match %d: zkill linking failed", m.id)

    # ------------------------------------------------------------------- phase

    def _update_phase(self, r: FrameReading, t: FrameTime) -> None:
        m = self.match
        if r.phase == Phase.COUNTDOWN and not m.live_started:
            if not m.saw_countdown:
                m.saw_countdown = True
                m.countdown_t = t
            # The first yellow frame can still read 0:00, so date the start
            # from a running countdown.
            if not m.countdown_dated and r.clock is not None and 0 < r.clock <= 30:
                m.countdown_dated = True
                start = t.utc - timedelta(seconds=30 - r.clock)
                self._update(m, countdown_start_utc=iso(start))
                self._event(t, "countdown", {"clock": r.clock})
        if r.tidi:
            self._update_tidi(r, t)
            return
        if r.phase == Phase.LIVE and r.clock is not None and m.live_started:
            # The clock only counts down; a jump up is a misread.
            if m.last_clock is not None and r.clock > m.last_clock + 1:
                r.clock = None
        if (not m.live_started and r.phase == Phase.LIVE and r.clock is not None and r.clock <= 30
                and m.countdown_t is not None and t.offset - m.countdown_t.offset < 60):
            # A countdown frame whose yellow wasn't recognised.
            r.phase = Phase.COUNTDOWN
        if not m.live_started and r.phase == Phase.LIVE and r.clock:
            # The countdown's last 0:00 can render white, and a stray read
            # shouldn't start the match, so wait for a running clock.
            m.live_candidates = [(ct, c) for ct, c in m.live_candidates
                                 if c >= r.clock and t.offset - ct.offset <= LIVE_STUCK_MAX_S]
            m.live_candidates.append((t, r.clock))
            # Date the start from the first frame whose clock had just moved.
            # A clock still showing its first value may be a frozen picture:
            # AT XXII finals day 1, a joined-late match sat on 9:53 for 7s and
            # then showed 9:31, so dating from the 9:53 put the start 9s late.
            moved = next((i for i in range(1, len(m.live_candidates))
                          if m.live_candidates[i][1] < m.live_candidates[i - 1][1]), None)
            if moved is None and t.offset - m.live_candidates[0][0].offset >= LIVE_STUCK_MAX_S:
                moved = 0  # a clock that never moves (paused): date it from when it was first seen
            if moved is not None:
                first_t, first_clock = m.live_candidates[moved]
                m.live_started = True
                m.last_clock, m.last_clock_t = r.clock, t
                m.clock_shown = (r.clock, t, t)
                m.live_start_offset = first_t.offset - (MATCH_LENGTH_S - first_clock)
                start = first_t.utc - timedelta(seconds=MATCH_LENGTH_S - first_clock)
                late = not m.saw_countdown
                # Numbered on going live, so abandoned and discarded matches use no number.
                m.match_no = self.db.next_match_no(self.phase, self.first_match)
                vod = {"vod_url": vod_link(self.vod, m.live_start_offset)} if self.vod else {}
                self._update(m, live_start_utc=iso(start), joined_late=int(late),
                             status="live", match_no=m.match_no, **vod)
                log.info("match %d is %s match %d", m.id, self.phase or "phase-less", m.match_no)
                self._event(t, "live", {"clock": first_clock, "joined_late": late})
        if r.phase in (Phase.PRE, Phase.COUNTDOWN, Phase.LIVE):
            m.phase = r.phase
        if m.live_started and r.phase == Phase.LIVE and r.clock is not None:
            if m.clock_shown is None or r.clock != m.clock_shown[0]:
                m.clock_shown = (r.clock, t, t)
            else:
                m.clock_shown = (r.clock, m.clock_shown[1], t)
            m.last_clock, m.last_clock_t = r.clock, t

    def _update_tidi(self, r: FrameReading, t: FrameTime) -> None:
        """Red TiDi game clock after the white clock ran out."""
        m = self.match
        if not m.live_started:
            return
        if m.tidi_start is None:
            m.tidi_start = t
            m.time_up = None  # the white 0:00 wasn't the end
            m.zero_frames = 0
            self._event(t, "tidi_overtime", {"clock": r.clock})
        if r.clock is None:
            return
        last = m.last_tidi_clock
        if last is not None:
            if r.clock % 10 == 8 and abs(r.clock - 8 - last) < abs(r.clock - last):
                r.clock -= 8  # a slashed zero read as 8
            if r.clock > last + 1:
                r.clock = None  # counts down only; a jump up is a misread
                return
        m.last_tidi_clock = r.clock

    # ------------------------------------------------------------------ deaths

    def _update_slots(self, r: FrameReading, t: FrameTime, gap: float) -> None:
        m = self.match
        if all(row.status in ("dead", "empty") for side in SIDES for row in r.rows[side]):
            # Both sides wiped is impossible; the HUD shows this when its
            # state is stale (seen for over a minute before it reset).
            return
        for side in SIDES:
            for row in r.rows[side]:
                if row.status == "empty":
                    continue
                key = (side, row.slot)
                s = m.slots.get(key)
                if s is None:
                    s = m.slots[key] = Slot(side, row.slot)
                    self._ship(m, side, row.slot)
                if row.pts is not None:
                    before = s.points
                    s.pts[row.pts] += 1
                    if s.points != before:
                        self._ship(m, side, row.slot, points=s.points)
                if s.dead:
                    continue
                if row.status == "alive":
                    s.seen_alive = True
                    s.dead_streak = 0
                    s.first_dead = None
                elif row.status == "dead":
                    if s.dead_streak == 0:
                        s.first_dead = t
                        s.first_dead_clock = r.clock if r.phase == Phase.LIVE else None
                        s.first_dead_tidi = r.tidi
                        s.first_dead_gap = gap
                    s.dead_streak += 1
                    if s.dead_streak >= DEATH_FRAMES:
                        self._record_death(s, r)

    def _record_death(self, s: Slot, r: FrameReading) -> None:
        m = self.match
        s.dead = True
        t = s.first_dead
        if not s.seen_alive:
            # Already dead when we first saw the HUD: time unknown.
            s.death_confidence = "unknown_time"
            fields = dict(death_confidence="unknown_time")
        else:
            died_match_s = None
            if m.live_start_offset is not None:
                died_match_s = max(0, round(t.offset - m.live_start_offset - DEATH_DISPLAY_LAG_S))
            s.death_confidence = "high" if self._score_agrees(s.side, r) else "low"
            fields = dict(
                died_match_s=died_match_s,
                died_clock_s=s.first_dead_clock,
                died_in_tidi=int(s.first_dead_tidi),
                died_utc=iso(t.utc),
                died_stream_offset_s=t.offset,
                death_confidence=s.death_confidence,
                gap_before_s=s.first_dead_gap or None,
            )
        self._ship(m, s.side, s.slot, **fields)
        self._event(t, "death", {"side": s.side, "slot": s.slot, "points": s.points, **fields})
        log.info("match %d: %s slot %d died (%s)", m.id, s.side, s.slot, s.death_confidence)

    def _score_agrees(self, side: str, r: FrameReading) -> bool:
        """The other team's score must equal 200 minus this side's surviving
        points. Ships showing dead but not yet confirmed count as gone, since
        the score updates on the frame the row greys out."""
        score = r.scores.get("blue" if side == "red" else "red")
        slots = [s for s in self.match.slots.values() if s.side == side]
        surviving = [s.points for s in slots if not s.dead and s.dead_streak == 0]
        if score is None or not slots or None in surviving:
            return False
        return score == TEAM_POINTS - sum(surviving)

    def _check_scores(self, r: FrameReading, t: FrameTime) -> None:
        """Upgrade low-confidence deaths once the score catches up."""
        m = self.match
        if r.scores.get("red") is not None or r.scores.get("blue") is not None:
            m.last_scores = {k: v for k, v in r.scores.items() if v is not None} | {
                k: v for k, v in m.last_scores.items() if r.scores.get(k) is None
            }
        for side in SIDES:
            low = [s for s in m.slots.values() if s.side == side and s.death_confidence == "low"]
            if low and self._score_agrees(side, r):
                for s in low:
                    s.death_confidence = "high"
                    self._ship(m, side, s.slot, death_confidence="high")

    def _check_end(self, r: FrameReading, t: FrameTime) -> None:
        m = self.match
        if not m.live_started or m.ended:
            return
        out_of_time = r.phase == Phase.LIVE and r.clock == 0 and (r.tidi or m.tidi_start is None)
        if out_of_time:
            if m.time_up is None:
                m.time_up = t
            m.zero_frames += 1
            if m.zero_frames >= ZERO_HOLD_FRAMES:
                self._end(m.time_up, "time")
                return
        elif r.phase == Phase.LIVE and r.clock:
            m.time_up = None
            m.zero_frames = 0
        for side in SIDES:
            slots = [s for s in m.slots.values() if s.side == side]
            if slots and all(s.dead for s in slots):
                self._end(t, "wipe")
                return

    # ---------------------------------------------------------------- identity

    def _maybe_read_identity(self, frame: np.ndarray, r: FrameReading) -> None:
        m = self.match
        if m.identity_settled or self._identity_job is not None:
            return
        self._identity_job = self._pool.submit(reader.read_identity, frame.copy(), r)
        self._identity_lag = 0
        self._identity_match = m

    def _collect_identity(self) -> None:
        job = self._identity_job
        if job is None or not job.done():
            return
        self._identity_job = None
        m = self._identity_match
        try:
            ident = job.result()
        except Exception:
            log.exception("identity read failed")
            return
        self._vote(m, ident)

    def _finish_identity(self, m: MatchState) -> None:
        """Settle names now, with whatever reads exist (the match is ending)."""
        if m.identity_settled:
            return
        if self._identity_job is not None:
            wait([self._identity_job])
            self._collect_identity()
            if m.identity_settled:
                return
        if m.identity_reads == 0 and self._last_hud is not None:
            frame, r = self._last_hud
            try:
                self._vote(m, reader.read_identity(frame, r))
            except Exception:
                log.exception("identity read failed")
        if not m.identity_settled and m.identity_reads:
            self._settle_identity(m)

    def _vote(self, m: MatchState, ident: Identity) -> None:
        """Count one read. Ship reads only vote when they resolve to the ship
        list, so reads through caster graphics drawn over the HUD drop out. A
        ban list votes with any name it can't resolve kept as raw text (a ship
        missing from the list, e.g. one from another year), which garbage reads
        won't agree on."""
        m.identity_reads += 1

        def vote(key, value):
            m.votes.setdefault(key, Counter())[value] += 1

        for side in SIDES:
            if ident.teams[side].strip():
                vote(("team", side), ident.teams[side])
            # On a faded HUD the ban count is unknown, and an empty list means nothing.
            if ident.ban_count.get(side) is not None:
                self._vote_bans(m, side, ident.bans[side], vote)
            for i, (raw, pilot) in enumerate(zip(ident.ships[side], ident.pilots[side])):
                slot = m.slots.get((side, i))
                hit = self.vocab.match(raw, slot.points if slot else None)
                if hit.ship:
                    vote(("ship", side, i), hit.ship.name)
                    m.raw[("ship", side, i, hit.ship.name)] = (raw, hit.score)
                vote(("ship_raw", side, i), raw)
                vote(("pilot", side, i), pilot)
        self._write_teams(m)
        required = [("team", s) for s in SIDES] + [("bans", s) for s in SIDES]
        required += [("ship", side, i) for side, i in m.slots]
        settled = all(key in m.votes and m.votes[key].most_common(1)[0][1] >= IDENTITY_VOTES
                      for key in required)
        if settled or m.identity_reads >= IDENTITY_MAX_READS:
            self._settle_identity(m)

    def _vote_bans(self, m: MatchState, side: str, bans: list[str], vote) -> None:
        """Vote the ban list top-down as one tuple, so the stored ordinals
        follow the order shown on stream. No bans at all is a valid list.
        Unmatched reads that aren't name-like are dropped: the ban count can
        overshoot, and the extra rows are OCR of the game scene behind them."""
        hits = [self.vocab.match(raw).ship for raw in bans]
        kept = [(raw, hit) for raw, hit in zip(bans, hits) if hit or looks_like_name(raw)]
        raws = tuple(raw for raw, _ in kept)
        hits = [hit for _, hit in kept]
        if any(hits) or not raws:
            names = tuple(h.name if h else UNMATCHED + raw for h, raw in zip(hits, raws))
            vote(("bans", side), names)
            m.raw[("bans", side, names)] = raws
        vote(("bans_raw", side), raws)

    def _write_teams(self, m: MatchState) -> None:
        """Store the leading team name for each colour as soon as there is
        one, so a match that never settles still records who was red/blue."""
        teams = {}
        for side in SIDES:
            c = m.votes.get(("team", side))
            teams[side] = self.teams.snap(c.most_common(1)[0][0]) if c else None
        teams = self._fit_teams(teams)
        if teams != m.written_teams:
            m.written_teams = teams
            self._update(m, red_team=teams["red"], blue_team=teams["blue"])

    def _fit_teams(self, teams: dict[str, str | None]) -> dict[str, str | None]:
        """Team names as stored and sent: best-fit to the known teams outside
        the Qualifiers, else as read."""
        if self.known_teams is None:
            return teams
        red, blue = self.known_teams.fit_pair(teams["red"], teams["blue"])
        return {"red": red, "blue": blue}

    def _settle_identity(self, m: MatchState) -> None:
        m.identity_settled = True

        def best(key):
            c = m.votes.get(key)
            return c.most_common(1)[0][0] if c else None

        teams = {side: self.teams.snap(best(("team", side))) for side in SIDES}
        for name in teams.values():
            self.teams.add(name)
        fitted = self._fit_teams(teams)  # the HUD spellings stay in teams, for matching the draft panel
        self._update(m, red_team=fitted["red"], blue_team=fitted["blue"])
        hud = {}
        for side in SIDES:
            names = best(("bans", side))
            if names is not None:
                hud[side] = [(raw, None if name.startswith(UNMATCHED) else name)
                             for raw, name in zip(m.raw[("bans", side, names)], names)]
            else:
                hud[side] = [(raw, None) for raw in best(("bans_raw", side)) or ()]
        drafted = m.draft_bans or self._take_draft(m, teams, hud)
        for side in SIDES:
            bans = hud[side]
            picks = None
            if drafted is not None:
                hud_bans = bans
                merged = self._reconcile_bans([(raw, typ, pick) for s, raw, typ, pick in drafted if s == side],
                                              hud_bans)
                bans = [(raw, typ) for raw, typ, _ in merged]
                picks = [pick for _, _, pick in merged]
                if sorted(typ or raw for raw, typ in bans) != sorted(typ or raw for raw, typ in hud_bans):
                    self._log(m, *self._now(), "draft_ban_mismatch",
                              {"side": side, "draft": [list(b) for b in bans],
                               "hud": [list(b) for b in hud_bans]})
            self.db.set_bans(m.id, side, bans, picks)
            m.bans[side] = [(raw, typ, pick) for (raw, typ), pick in zip(bans, picks or [None] * len(bans))]
            slots = sorted({k[2] for k in m.votes if k[0] == "ship_raw" and k[1] == side})
            for i in slots:
                name = best(("ship", side, i))
                raw, score = m.raw[("ship", side, i, name)] if name else (best(("ship_raw", side, i)), None)
                self._ship(
                    m, side, i,
                    ship_raw=raw,
                    ship_type=name,
                    match_confidence=score,
                    pilot=best(("pilot", side, i)),
                )
        log.info("match %d identity: %s vs %s", m.id, teams["red"], teams["blue"])
        ts, offset = self._now()
        self._log(m, ts, offset, "identity_settled", {"reads": m.identity_reads})
        self._teams_from_pilots(m, fitted)
        # Repeat pairings are games of one series (Bo3/Bo5), even with other
        # matches played in between.
        m.series_id, mismatch = self.db.assign_series(m.id)
        if mismatch:
            log.warning("match %d: %s vs %s played recently with other pilots; not joined to that series",
                        m.id, teams["red"], teams["blue"])
            self._log(m, ts, offset, "series_mismatch", None)
        elif m.series_id is not None:
            series = self.db.series_summary(m.id)
            log.info("match %d: series %d game %d", m.id, m.series_id, series["game"])
            self._log(m, ts, offset, "series", series)

    def _teams_from_pilots(self, m: MatchState, fitted: dict[str, str | None]) -> None:
        """Correct a side's team name from its pilots, outside the Qualifiers.

        AT XXII finals day 1: the overlay put PSYCHOKIDS. over Rote Kapelle's
        roster and vice versa (match 24), and read the IIII… team as
        "SUMUUOUIOUNNNNNNOTIME" (match 19). Both teams had played earlier in
        the day with the same pilots. Only a swap of the two names, or a name
        that fits no known team, is corrected; a known name is otherwise kept,
        as a roster can be shared between a team and its alt.
        """
        if self.known_teams is None:
            return
        roster = {side: self.db.team_for_pilots(self.db.pilots(m.id, side), m.id) for side in SIDES}
        new = dict(fitted)
        if roster["red"] and roster["red"] == fitted["blue"] and roster["blue"] in (fitted["red"], None):
            new = {"red": fitted["blue"], "blue": fitted["red"]}  # labels swapped
        elif roster["blue"] and roster["blue"] == fitted["red"] and roster["red"] is None:
            new = {"red": fitted["blue"], "blue": fitted["red"]}
        else:
            for side in SIDES:
                team = roster[side]
                if (team and team != fitted[side] and fitted[side] not in self.known_teams.names
                        and team != new[OTHER[side]]):
                    new[side] = team
        if new == fitted:
            return
        log.warning("match %d: teams from pilots: %s vs %s (overlay: %s vs %s)",
                    m.id, new["red"], new["blue"], fitted["red"], fitted["blue"])
        self._log(m, *self._now(), "team_from_pilots", {"overlay": fitted, "pilots": new})
        self._update(m, red_team=new["red"], blue_team=new["blue"])

    # ------------------------------------------------------------------- draft

    def _check_draft(self, frame: np.ndarray, t: FrameTime) -> None:
        """Read the draft panel while it is on screen, until 3 reads agree."""
        panel = draft.find_panel(frame)
        if panel is None:
            return
        if self._draft_seen is not None and t.offset - self._draft_seen.offset > DRAFT_GAP_S:
            self._draft_settled = False  # a new panel
            self._draft_votes.clear()
        self._draft_seen = t
        if len(panel.bars) != self._draft_bars:
            # Still being filled in, pick by pick: earlier reads are stale.
            self._draft_bars = len(panel.bars)
            self._draft_settled = False
            self._draft_votes.clear()
        if self._draft_settled:
            if self.draft is not None:
                self.draft.last_seen = t
            return
        if self._draft_job is None:
            self._draft_job = self._pool.submit(draft.read_panel, frame.copy(), panel)
            self._draft_lag = 0

    def _collect_draft(self) -> None:
        job = self._draft_job
        if job is None or not job.done():
            return
        self._draft_job = None
        try:
            read = job.result()
        except Exception:
            log.exception("draft read failed")
            return
        if len(read.bans) != self._draft_bars or self._draft_settled:
            return
        # Ship names vote as resolved names, so small OCR differences in
        # the raw text still agree; team names vote on their own.
        bans = []
        for side, raw in read.bans:
            hit = self.vocab.match(raw)
            # The panel's font reads i as l: "Rapler" ties Reaper and Rapier.
            alt = self.vocab.match(raw.replace("l", "i"))
            if alt.score > hit.score:
                hit = alt
            bans.append((side, raw, hit.ship.name if hit.ship else None))
        key = ("bans", tuple((side, typ or UNMATCHED + raw) for side, raw, typ in bans))
        self._draft_votes[key] += 1
        for side in SIDES:
            if read.teams[side]:
                self._draft_votes[("team", side, self.teams.snap(read.teams[side]))] += 1
        if self._draft_votes[key] < IDENTITY_VOTES:
            return
        teams = []
        for side in SIDES:
            votes = [(n, k[2]) for k, n in self._draft_votes.items() if k[:2] == ("team", side)]
            teams.append(max(votes)[1] if votes else None)
        if not all(teams):
            return
        self._draft_settled = True
        self.draft = Draft(dict(zip(SIDES, teams)), bans, self._draft_seen)
        log.info("draft: %s vs %s, bans %s", teams[0], teams[1],
                 ", ".join(f"{typ or raw} ({side})" for side, raw, typ in bans))
        self._log(None, *self._now(), "draft", {"teams": list(teams), "bans": [list(b) for b in bans]})

    def _take_draft(self, m: MatchState, teams: dict[str, str | None],
                    hud: dict[str, list[tuple[str, str | None]]]) -> list | None:
        """The last draft, if it was recent and for this match.

        Sides are matched by team name, not panel colour. Both names should
        match, but the HUD's can be misread ("UNC KAPELLE") or stale (the
        previous match's teams), so a draft whose banned ships agree with the
        HUD's is also taken when one name matches, or when the HUD names are
        no help at all.
        """
        d = self.draft
        if d is None:
            return None
        if self._t is not None and self._t.offset - d.last_seen.offset > DRAFT_MAX_AGE_S:
            return None
        sides = {}
        for dside, name in d.teams.items():
            hits = [side for side in SIDES if self._same_known_team(name, teams[side])]
            if len(hits) == 1:
                sides[dside] = hits[0]
        if len(set(sides.values())) < len(sides):
            return None
        if len(sides) < 2:
            if len(sides) == 1:
                (dside, side), = sides.items()
                flip = dside != side
            else:
                flip = False
            sides = {dside: OTHER[dside] if flip else dside for dside in SIDES}
            if not self._draft_agrees(d, sides, hud):
                return None
        self.draft = None
        m.draft_bans = [(sides[side], raw, typ, pick) for pick, (side, raw, typ) in enumerate(d.bans, 1)]
        self._log(m, *self._now(), "bans_from_draft", {"bans": [list(b) for b in m.draft_bans]})
        return m.draft_bans

    def _reconcile_bans(self, drafted: list[tuple[str, str | None, int | None]],
                        hud: list[tuple[str, str | None]]) -> list[tuple[str, str | None, int | None]]:
        """One side's bans as (raw, type, pick): the draft's in pick order,
        then the HUD's that the draft lacks.

        The broadcast can cut away from the panel before every pick is in
        (AT XXII finals: 3 of 12 cards shown), and series "conquest" bans
        are on the HUD only, so HUD ships missing from the draft are kept,
        after the drafted ones, with no pick number. HUD reads that matched
        no ship are noise more often than not and aren't added.

        A draft ship the HUD lacks that looks like one the HUD has instead
        is one ship that one of them misread: "Rapler" (-> Reaper) on the
        panel and "; Dapier" (-> Rapier) on the HUD, "Fos" or a cut-off
        "nageddon N:" on the panel. Of the two ships, the one closer to both
        raw reads together is kept, in the draft's place; on a tie the
        draft's.
        """
        name = lambda b: b[1] or b[0]
        merged = list(drafted)
        only_hud = list((Counter(map(name, hud)) - Counter(map(name, drafted))).elements())
        only_draft = Counter(map(name, drafted)) - Counter(map(name, hud))
        raw_of = {name(b): b for b in drafted}

        def similar(d: str, h: str) -> float:
            sim = fuzz.ratio(d.lower(), h.lower())
            if raw_of[d][1] is None and len(_letters(d)) >= 5:
                sim = max(sim, fuzz.partial_ratio(_letters(d), h.lower()))  # a cut-off read
            return sim

        pairs = sorted(((similar(d, h), d, h) for d in only_draft for h in set(only_hud)), reverse=True)
        for sim, d, h in pairs:
            if sim < BAN_MISREAD_MIN or only_draft[d] == 0 or h not in only_hud:
                continue
            i = next(i for i, b in enumerate(merged) if name(b) == d)
            hb = next(b for b in hud if name(b) == h)
            reads = (_letters(merged[i][0]), _letters(hb[0]))
            fit = lambda ship: sum(fuzz.ratio(r, ship.lower()) for r in reads) if ship else -1.0
            if fit(hb[1]) > fit(merged[i][1]):
                merged[i] = (hb[0], hb[1], merged[i][2])
            only_draft[d] -= 1
            only_hud.remove(h)
        for b in hud:
            if b[1] and name(b) in only_hud:
                only_hud.remove(name(b))
                merged.append((b[0], b[1], None))
        return merged

    def _same_known_team(self, a: str | None, b: str | None) -> bool:
        """same_team, or (outside the Qualifiers) both names best-fit to the
        same known team: the draft panel's misreads ("Rote Kapalla",
        "xgtywiznalamyu") differ from the HUD's too much for same_team."""
        if same_team(a, b):
            return True
        if self.known_teams is None or not (a and b):
            return False
        fa, fb = self.known_teams.fit(a), self.known_teams.fit(b)
        return fa == fb and fa in self.known_teams.names

    @staticmethod
    def _draft_agrees(d: Draft, sides: dict[str, str], hud: dict[str, list]) -> bool:
        """Do the draft's bans name the same ships as the HUD's lists? One
        ship read wrong on either is allowed. HUD reads that matched no ship
        don't count against it: they are as likely noise as a ship."""
        if len(d.bans) < DRAFT_MIN_BANS_TO_AGREE:
            return False
        off = 0
        for dside, side in sides.items():
            want = Counter(typ or raw for s, raw, typ in d.bans if s == dside)
            got = Counter(typ for raw, typ in hud.get(side, []) if typ)
            off += sum(((want - got) + (got - want)).values())
        return off <= 2

    # ------------------------------------------------------------------- utils

    def _now(self) -> tuple[str, float | None]:
        t = self._t
        return (iso(t.utc), t.offset) if t else (iso(datetime.now().astimezone()), None)

    def export(self, m: MatchState) -> dict:
        """The match as one observed-import object (see README), from memory."""
        bans = [dict(side=side, ordinal=i, pick=pick, ship_type=typ, ship_raw=raw)
                for side in SIDES for i, (raw, typ, pick) in enumerate(m.bans.get(side, []), 1)]
        ship_keys = ("ship_type", "ship_raw", "pilot", "points", "died_match_s", "died_utc",
                     "match_confidence", "death_confidence")
        ships = [dict(side=side, slot=slot, **{k: fields.get(k) for k in ship_keys})
                 for (side, slot), fields in sorted(m.ships.items(), key=lambda kv: (kv[0][0] != "red", kv[0][1]))]
        phase = {"phase": self.phase} if self.phase else {}
        row = {k: v for k, v in m.row.items() if k != "match_no"}
        if self.source == "live" and not row.get("vod_url"):
            # The endpoint requires a vod_url, but live offsets can't point into a VOD
            # that doesn't exist yet. Send the channel as a placeholder (the DB keeps
            # NULL); replaying the VOD with --post later replaces it with the real link.
            row["vod_url"] = channel_url(self.channel)
        return dict(run_id=self.run_id, match_id=m.match_no, **row, **phase, bans=bans, ships=ships,
                    events=list(m.events))

    def _update(self, m: MatchState, **fields) -> None:
        self.db.update_match(m.id, **fields)
        m.row.update(fields)

    def _ship(self, m: MatchState, side: str, slot: int, **fields) -> None:
        self.db.upsert_ship(m.id, side, slot, **fields)
        m.ships.setdefault((side, slot), {}).update(fields)

    def _log(self, m: MatchState | None, ts: str, offset: float | None, kind: str,
             payload: dict | None = None) -> None:
        self.db.log_event(m.id if m else None, ts, offset, kind, payload)
        if m is not None:
            m.events.append(dict(kind=kind, ts_utc=ts, stream_offset_s=offset,
                                 payload_json=json.dumps(payload) if payload else None))

    def _event(self, t: FrameTime, kind: str, payload: dict | None = None) -> None:
        """Log an event; with debug_dir, also save the current frame for auditing."""
        mid = self.match.id if self.match else None
        self._log(self.match, iso(t.utc), t.offset, kind, payload)
        if self.debug_dir is not None and self._frame is not None:
            name = f"m{mid}_{int(self._t.offset):06d}_{kind}.jpg"
            cv2.imwrite(str(self.debug_dir / name), self._frame)
