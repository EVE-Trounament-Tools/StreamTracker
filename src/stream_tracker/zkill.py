"""Link recorded ship deaths to their zKillboard killmails.

AT kills land on zKillboard in the tournament region (10000004, UUA-F4), but
not always, and a pilot can die there several times in one day. Each death
is paired with the killmail for the same pilot and ship whose time is closest
to it, and left unlinked if even the closest is too far away.

The tracker's died_utc is the wall time of the frame where the row greyed out,
so it trails the game by the stream delay (live), or by the VOD's start-time
error (replay). That offset is near constant over a stream and is estimated
from the deaths themselves before pairing.
"""

import gzip
import heapq
import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from rapidfuzz import fuzz

from .db import Database

log = logging.getLogger(__name__)

REGION_ID = 10000004  # UUA-F4, the Alliance Tournament region
ZKILL_PAGE_URL = "https://zkillboard.com/api/kills/regionID/{region}/page/{page}/"
ESI_NAMES_URL = "https://esi.evetech.net/latest/universe/names/?datasource=tranquility"
USER_AGENT = "stream-tracker/0.1 (EVE Alliance Tournament stream tracker)"
TIMEOUT_S = 30
RETRIES = 3
PAGE_DELAY_S = 1.0  # between zKill pages; it rate-limits
MAX_PAGES = 100

TOLERANCE_S = 60  # a killmail this far from the (offset-corrected) death is not that death
OFFSET_SEARCH_S = 900  # stream offsets considered when estimating it
OFFSET_WINDOW_S = 20  # deaths read off the stream agree with the killmails to within this
MIN_OFFSET_PAIRS = 3  # fewer pairs than this can't estimate an offset; 0 is used
NAME_MIN = 60  # pilot-name similarity needed when the ship type also matches (OCR'd names are noisy)
NAME_MIN_NO_SHIP = 85  # ... and when the ship type wasn't read
SETTLE_S = 3600  # zKill has every kill by this long after it happened
RETRY_DELAYS_S = (180, 600, 1800)  # background re-checks after a live match ends


def parse_utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def utc_iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------- fetching

def _request(url: str, data: bytes | None = None) -> object:
    headers = {"user-agent": USER_AGENT, "accept-encoding": "gzip", "accept": "application/json"}
    if data is not None:
        headers["content-type"] = "application/json"
    for attempt in range(RETRIES + 1):
        if attempt:
            time.sleep(5.0 * attempt)
        req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                body = resp.read()
                if resp.headers.get("content-encoding") == "gzip":
                    body = gzip.decompress(body)
                return json.loads(body or b"null")
        except urllib.error.HTTPError as e:
            if e.code != 429 and e.code < 500:
                raise
            log.warning("%s failed (%d, try %d)", url, e.code, attempt + 1)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            log.warning("%s failed (try %d): %s", url, attempt + 1, e)
    raise RuntimeError(f"{url}: gave up after {RETRIES + 1} tries")


def fetch_region_kills(start: datetime, region: int = REGION_ID, skip: set[int] = frozenset(),
                       page_delay_s: float = PAGE_DELAY_S) -> list[dict]:
    """Every player killmail in the region from `start` until now. zKill lists
    them newest first, so pages are read until one reaches back past `start`."""
    kills = []
    for page in range(1, MAX_PAGES + 1):
        if page > 1:
            time.sleep(page_delay_s)
        rows = _request(ZKILL_PAGE_URL.format(region=region, page=page)) or []
        for k in rows:
            when = parse_utc(k["killmail_time"])
            victim = k.get("victim") or {}
            if when < start or not victim.get("character_id") or k["killmail_id"] in skip:
                continue
            kills.append(dict(killmail_id=k["killmail_id"], hash=(k.get("zkb") or {}).get("hash"),
                              killmail_utc=utc_iso(when), solar_system_id=k.get("solar_system_id"),
                              character_id=victim["character_id"], ship_type_id=victim.get("ship_type_id")))
        if not rows or parse_utc(rows[-1]["killmail_time"]) < start:
            break
    else:
        log.warning("zkill: stopped after %d pages without reaching %s", MAX_PAGES, utc_iso(start))
    return kills


def resolve_names(ids: set[int]) -> dict[int, str]:
    """Character and type names from ESI. A batch with an id ESI doesn't know
    is refused whole, so such a batch is split until the bad id is alone."""
    names: dict[int, str] = {}

    def lookup(batch: list[int]) -> None:
        if not batch:
            return
        try:
            for row in _request(ESI_NAMES_URL, json.dumps(batch).encode()):
                names[row["id"]] = row["name"]
        except urllib.error.HTTPError as e:
            if e.code != 404 or len(batch) == 1:
                log.warning("esi names: %s for %d ids", e.code, len(batch))
                return
            mid = len(batch) // 2
            lookup(batch[:mid])
            lookup(batch[mid:])

    ids = sorted(i for i in ids if i)
    for i in range(0, len(ids), 1000):
        lookup(ids[i:i + 1000])
    return names


def update_killmails(db: Database, start: datetime, end: datetime, region: int = REGION_ID,
                     now: datetime | None = None) -> int:
    """Make sure the killmail cache holds every kill from start to end. Skips
    zKill when a fetch made after end + SETTLE_S already reached back to start."""
    now = now or datetime.now(timezone.utc)
    if db.zkill_fetched(region, utc_iso(start), utc_iso(end + timedelta(seconds=SETTLE_S))):
        return 0
    kills = fetch_region_kills(start, region, skip=db.known_killmail_ids())
    names = resolve_names({k["character_id"] for k in kills} | {k["ship_type_id"] for k in kills})
    for k in kills:
        k["victim_name"] = names.get(k["character_id"])
        k["ship_type"] = names.get(k["ship_type_id"])
    db.store_killmails(kills, region, utc_iso(start), utc_iso(now))
    log.info("zkill: %d new killmails in region %d since %s", len(kills), region, utc_iso(start))
    return len(kills)


# ------------------------------------------------------------------- matching

@dataclass
class Death:
    key: tuple[int, str, int]  # (match_id, side, slot)
    pilot: str | None
    ship_type: str | None
    died: datetime | None  # None for unknown_time deaths
    gap_s: float = 0.0  # HUD was off screen this long before the death was seen
    window: tuple[datetime, datetime] | None = None  # the match, for unknown_time deaths


@dataclass
class Kill:
    killmail_id: int
    hash: str | None
    time: datetime
    victim: str | None
    ship_type: str | None


@dataclass
class Link:
    kill: Kill
    delta_s: float | None  # kill time - (died + offset)


def _clean_name(name: str) -> tuple[str, bool]:
    n = name.lower().strip()
    truncated = False
    for tail in ("…", "..."):
        if n.endswith(tail):
            n, truncated = n[: -len(tail)], True
    n = re.sub(r"[^\w' -]", "", n).strip()
    return n, truncated


def name_score(hud: str | None, victim: str | None) -> float:
    """Similarity of a pilot name read off the HUD to a killmail victim's name,
    allowing for the HUD cutting long names short with '…'."""
    if not hud or not victim:
        return 0.0
    a, truncated = _clean_name(hud)
    b, _ = _clean_name(victim)
    if truncated:
        b = b[: len(a)]
    return fuzz.ratio(a, b) if a and b else 0.0


def is_candidate(d: Death, k: Kill) -> bool:
    if d.ship_type:
        return (k.ship_type or "").casefold() == d.ship_type.casefold() and name_score(d.pilot, k.victim) >= NAME_MIN
    return name_score(d.pilot, k.victim) >= NAME_MIN_NO_SHIP


def candidate_pairs(deaths: list[Death], kills: list[Kill]) -> dict[int, list[Kill]]:
    return {i: [k for k in kills if is_candidate(d, k)] for i, d in enumerate(deaths)}


def estimate_offset(deaths: list[Death], cands: dict[int, list[Kill]]) -> tuple[float, int]:
    """(offset_s, support): the kill-minus-death time most deaths agree on, the
    median of the densest OFFSET_WINDOW_S of deltas. (0, n) without enough support."""
    deltas = sorted(
        (k.time - d.died).total_seconds()
        for i, d in enumerate(deaths) if d.died is not None
        for k in cands[i] if abs((k.time - d.died).total_seconds()) <= OFFSET_SEARCH_S)
    best: list[float] = []
    lo = 0
    for hi in range(len(deltas)):
        while deltas[hi] - deltas[lo] > OFFSET_WINDOW_S:
            lo += 1
        window = deltas[lo:hi + 1]
        if len(window) > len(best) or (len(window) == len(best) and abs(window[len(window) // 2])
                                        < abs(best[len(best) // 2])):
            best = window
    if len(best) < MIN_OFFSET_PAIRS:
        return 0.0, len(best)
    return best[len(best) // 2], len(best)


def pair_deaths(deaths: list[Death], kills: list[Kill], tolerance_s: float = TOLERANCE_S,
                offset_s: float | None = None) -> tuple[dict[tuple, Link], float, int]:
    """Pair deaths with killmails, one to one. Returns (links by death key, offset, support).

    A death takes the candidate kill nearest its offset-corrected time, within
    tolerance (more on the early side when the HUD was off screen before the
    death was seen). Closest pairs are settled first, so a pilot who died twice
    gets each kill matched to the nearer death. An unknown_time death links
    only when exactly one unused candidate falls inside its match.
    """
    cands = candidate_pairs(deaths, kills)
    support = 0
    if offset_s is None:
        offset_s, support = estimate_offset(deaths, cands)
    pairs = []
    for i, d in enumerate(deaths):
        if d.died is None:
            continue
        for k in cands[i]:
            err = (k.time - d.died).total_seconds() - offset_s
            if -tolerance_s - (d.gap_s or 0) <= err <= tolerance_s:
                pairs.append((abs(err), -name_score(d.pilot, k.victim), i, k.killmail_id, err, k))
    pairs.sort(key=lambda p: p[:4])
    links: dict[tuple, Link] = {}
    used: set[int] = set()
    for _, _, i, kid, err, k in pairs:
        if deaths[i].key in links or kid in used:
            continue
        links[deaths[i].key] = Link(k, round(err, 1))
        used.add(kid)
    shift = timedelta(seconds=offset_s)
    for i, d in enumerate(deaths):
        if d.died is not None or d.window is None:
            continue
        start, end = d.window
        inside = [k for k in cands[i] if k.killmail_id not in used
                  and start + shift - timedelta(seconds=tolerance_s) <= k.time <= end + shift + timedelta(seconds=tolerance_s)]
        if len(inside) == 1:
            links[d.key] = Link(inside[0], None)
            used.add(inside[0].killmail_id)
    return links, offset_s, support


# ------------------------------------------------------------------- linking

def _death(row) -> Death:
    window = None
    if row["live_start_utc"]:
        start = parse_utc(row["live_start_utc"])
        end = parse_utc(row["ended_utc"]) if row["ended_utc"] else start + timedelta(minutes=15)
        window = (start, end)
    return Death(key=(row["match_id"], row["side"], row["slot"]), pilot=row["pilot"], ship_type=row["ship_type"],
                 died=parse_utc(row["died_utc"]) if row["died_utc"] else None,
                 gap_s=row["gap_before_s"] or 0.0, window=window)


def link_deaths(db: Database, match_ids: list[int] | None = None, tolerance_s: float = TOLERANCE_S,
                offset_s: float | None = None, region: int = REGION_ID, fetch: bool = True) -> dict:
    """Fetch killmails covering these matches' deaths (all matches by default)
    and link each death to its killmail. Re-running redoes the links."""
    deaths = [d for d in map(_death, db.deaths(match_ids)) if d.pilot]
    summary = {"deaths": len(deaths), "linked": 0, "offsets": {}}
    times = [t for d in deaths for t in ((d.died,) if d.died else d.window or ())]
    if not times:
        return summary
    margin = timedelta(seconds=OFFSET_SEARCH_S + tolerance_s)
    start, end = min(times) - margin, max(times) + margin
    if fetch:
        update_killmails(db, start, end, region)
    kills = [Kill(r["killmail_id"], r["hash"], parse_utc(r["killmail_utc"]), r["victim_name"], r["ship_type"])
             for r in db.killmails_between(utc_iso(start), utc_iso(end), region)]

    # Offsets are per match when it has enough deaths to say, since one
    # database can hold several streams; others use the pooled estimate.
    pooled, pooled_support = (offset_s, 0) if offset_s is not None else estimate_offset(
        deaths, candidate_pairs(deaths, kills))
    by_match: dict[int, list[Death]] = {}
    for d in deaths:
        by_match.setdefault(d.key[0], []).append(d)
    links: dict[tuple, Link] = {}
    for mid, ds in by_match.items():
        offset = offset_s
        if offset is None:
            offset, support = estimate_offset(ds, candidate_pairs(ds, kills))
            if support < MIN_OFFSET_PAIRS:
                offset, support = pooled, pooled_support
        got, offset, _ = pair_deaths(ds, kills, tolerance_s, offset)
        summary["offsets"][mid] = offset
        links.update(got)

    # A killmail can only be one death, also across matches.
    seen: dict[int, tuple] = {}
    for key, link in sorted(links.items(), key=lambda kv: abs(kv[1].delta_s) if kv[1].delta_s is not None else 1e9):
        if link.kill.killmail_id in seen:
            del links[key]
        else:
            seen[link.kill.killmail_id] = key

    for mid, ds in by_match.items():
        linked = 0
        for d in ds:
            link = links.get(d.key)
            fields = dict(killmail_id=None, killmail_hash=None, killmail_utc=None, killmail_delta_s=None,
                          zkill_victim=None)
            if link:
                linked += 1
                fields = dict(killmail_id=link.kill.killmail_id, killmail_hash=link.kill.hash,
                              killmail_utc=utc_iso(link.kill.time), killmail_delta_s=link.delta_s,
                              zkill_victim=link.kill.victim)
            db.upsert_ship(*d.key, **fields)
        summary["linked"] += linked
        db.log_event(mid, utc_iso(datetime.now(timezone.utc)), None, "zkill",
                     {"linked": linked, "deaths": len(ds), "offset_s": summary["offsets"][mid]})
    return summary


class ZkillLinker:
    """Links each finished match's deaths in the background.

    zKill gets kills a few minutes after they happen, so live matches are
    checked again RETRY_DELAYS_S after they end, until every death is linked.
    Replays of finished streams are checked once, straight away. Uses its own
    database connection; the tracker keeps writing on its own.
    """

    def __init__(self, db_path: str | Path, delays_s: tuple[float, ...] = RETRY_DELAYS_S,
                 tolerance_s: float = TOLERANCE_S, region: int = REGION_ID,
                 clock: Callable[[], float] = time.monotonic):
        self.db_path = db_path
        self.delays_s = delays_s
        self.tolerance_s = tolerance_s
        self.region = region
        self.clock = clock
        self._jobs: list[tuple[float, int, int]] = []  # (due, match_id, attempt)
        self._cv = threading.Condition()
        self._closing = False
        self._thread = threading.Thread(target=self._run, name="zkill", daemon=True)
        self._thread.start()

    def submit(self, match_id: int) -> None:
        with self._cv:
            self._jobs = [j for j in self._jobs if j[1] != match_id]  # a reopened match starts over
            heapq.heappush(self._jobs, (self.clock() + self.delays_s[0], match_id, 0))
            self._cv.notify()

    def close(self) -> None:
        """Finish jobs that are due now; later re-checks are dropped
        (stream-tracker zkill fills those in)."""
        with self._cv:
            self._closing = True
            dropped = sorted({mid for due, mid, _ in self._jobs if due > self.clock()})
            if dropped:
                log.info("zkill: matches %s not checked yet; run stream-tracker zkill later", dropped)
            self._cv.notify()
        self._thread.join()

    def _run(self) -> None:
        db = None
        try:
            while True:
                with self._cv:
                    while True:
                        now = self.clock()
                        if self._jobs and self._jobs[0][0] <= now:
                            _, mid, attempt = heapq.heappop(self._jobs)
                            break
                        if self._closing:
                            return
                        self._cv.wait(self._jobs[0][0] - now if self._jobs else None)
                db = db or Database(self.db_path)
                try:
                    s = link_deaths(db, [mid], self.tolerance_s, region=self.region)
                    log.info("zkill: match %d, %d of %d deaths linked", mid, s["linked"], s["deaths"])
                    done = s["linked"] >= s["deaths"]
                except Exception:
                    log.exception("zkill: match %d failed", mid)
                    done = False
                if not done and attempt + 1 < len(self.delays_s):
                    with self._cv:
                        heapq.heappush(self._jobs, (self.clock() + self.delays_s[attempt + 1] - self.delays_s[attempt],
                                                    mid, attempt + 1))
        finally:
            if db is not None:
                db.close()
