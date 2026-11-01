import io
import json
import time
from datetime import datetime, timedelta, timezone

from stream_tracker import zkill
from stream_tracker.db import Database
from stream_tracker.zkill import Death, Kill, ZkillLinker, link_deaths, name_score, pair_deaths, utc_iso

T0 = datetime(2026, 10, 31, 14, 0, tzinfo=timezone.utc)


def at(s: float) -> datetime:
    return T0 + timedelta(seconds=s)


def death(n, pilot, ship, s, gap=0.0, window=None):
    return Death(key=(1, "red", n), pilot=pilot, ship_type=ship, died=at(s) if s is not None else None,
                 gap_s=gap, window=window)


def kill(kid, victim, ship, s):
    return Kill(kid, f"h{kid}", at(s), victim, ship)


def test_pilot_dying_twice_gets_the_nearest_kill_each_time():
    deaths = [death(0, "Foo Bar", "Raven", 0), death(1, "Foo Bar", "Raven", 2400)]
    kills = [kill(1, "Foo Bar", "Raven", 2398), kill(2, "Foo Bar", "Raven", -3)]
    links, _, _ = pair_deaths(deaths, kills, offset_s=0)
    assert links[(1, "red", 0)].kill.killmail_id == 2
    assert links[(1, "red", 1)].kill.killmail_id == 1
    assert links[(1, "red", 0)].delta_s == -3


def test_closest_kill_too_far_is_not_linked():
    links, _, _ = pair_deaths([death(0, "Foo Bar", "Raven", 0)], [kill(1, "Foo Bar", "Raven", 300)], offset_s=0)
    assert links == {}


def test_stream_offset_is_estimated():
    # Kills happened 45s before the stream showed them (past the tolerance without the offset).
    deaths = [death(i, f"Pilot {c}", "Raven", i * 30) for i, c in enumerate("ABCD")]
    kills = [kill(i, f"Pilot {c}", "Raven", i * 30 - 75 + (i % 2)) for i, c in enumerate("ABCD")]
    kills.append(kill(9, "Pilot A", "Raven", 3600))  # the same pilot's death in a later match
    links, offset, support = pair_deaths(deaths, kills, tolerance_s=10)
    assert support == 4 and -75 <= offset <= -74
    assert {k: v.kill.killmail_id for k, v in links.items()} == {(1, "red", i): i for i in range(4)}


def test_offset_falls_back_to_zero_without_support():
    _, offset, support = pair_deaths([death(0, "Foo Bar", "Raven", 0)], [kill(1, "Foo Bar", "Raven", -20)])
    assert (offset, support) == (0.0, 1)


def test_truncated_and_noisy_names():
    assert name_score("Beacon Malaf Morg...", "Beacon Malaf Morgenstern") == 100
    assert name_score("aria Yatolila", "Daria Yatolila") >= zkill.NAME_MIN
    links, _, _ = pair_deaths([death(0, "Foo Ba…", "Raven", 0)], [kill(1, "Foo Bartholomew", "Raven", 2)], offset_s=0)
    assert links[(1, "red", 0)].kill.killmail_id == 1


def test_wrong_ship_or_pilot_not_linked():
    d = [death(0, "Foo Bar", "Raven", 0)]
    assert pair_deaths(d, [kill(1, "Foo Bar", "Raven Navy Issue", 1)], offset_s=0)[0] == {}
    assert pair_deaths(d, [kill(1, "Someone Else", "Raven", 1)], offset_s=0)[0] == {}


def test_kill_links_only_once():
    deaths = [death(0, "Foo Bar", "Raven", 0), death(1, "Foo Baz", "Raven", 5)]
    links, _, _ = pair_deaths(deaths, [kill(1, "Foo Bar", "Raven", 1)], offset_s=0)
    assert list(links) == [(1, "red", 0)]


def test_hud_gap_allows_an_earlier_kill():
    d = [death(0, "Foo Bar", "Raven", 0, gap=150)]
    assert pair_deaths(d, [kill(1, "Foo Bar", "Raven", -140)], offset_s=0)[0]
    assert not pair_deaths(d, [kill(1, "Foo Bar", "Raven", 140)], offset_s=0)[0]


def test_unknown_time_links_only_when_unique():
    window = (at(0), at(600))
    d = [death(0, "Foo Bar", "Raven", None, window=window)]
    links, _, _ = pair_deaths(d, [kill(1, "Foo Bar", "Raven", 100), kill(2, "Foo Bar", "Raven", 3000)], offset_s=0)
    assert links[(1, "red", 0)].kill.killmail_id == 1 and links[(1, "red", 0)].delta_s is None
    links, _, _ = pair_deaths(d, [kill(1, "Foo Bar", "Raven", 100), kill(2, "Foo Bar", "Raven", 400)], offset_s=0)
    assert links == {}


# ------------------------------------------------------------------- end to end, HTTP mocked

def zk(kid, char, ship, s):
    return {"killmail_id": kid, "killmail_time": at(s).strftime("%Y-%m-%dT%H:%M:%SZ"), "solar_system_id": 1,
            "victim": {"character_id": char, "ship_type_id": ship}, "zkb": {"hash": f"h{kid}"}}


class FakeHttp:
    """zKill pages (newest first) and ESI names."""

    def __init__(self, pages, names):
        self.pages, self.names, self.urls = pages, names, []

    def __call__(self, req, timeout=None):
        url = req.full_url
        self.urls.append(url)
        if "universe/names" in url:
            body = [{"id": i, "name": self.names[i]} for i in json.loads(req.data)]
        else:
            page = int(url.rstrip("/").split("/")[-1])
            body = self.pages[page - 1] if page <= len(self.pages) else []
        return _Resp(io.BytesIO(json.dumps(body).encode()))


class _Resp:
    def __init__(self, f):
        self.f, self.headers = f, {}

    def read(self):
        return self.f.read()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def make_db(tmp_path):
    db = Database(tmp_path / "t.sqlite")
    mid = db.create_match("test", None, utc_iso(at(-60)))
    db.update_match(mid, live_start_utc=utc_iso(at(0)), ended_utc=utc_iso(at(600)), status="complete")
    db.upsert_ship(mid, "red", 0, pilot="Foo Bar", ship_type="Raven", died_utc=utc_iso(at(100)),
                   death_confidence="high")
    db.upsert_ship(mid, "blue", 0, pilot="Baz Qux", ship_type="Stork", died_utc=utc_iso(at(200)),
                   death_confidence="high")
    db.upsert_ship(mid, "blue", 1, pilot="Alive One", ship_type="Kirin")
    return db, mid


def test_link_deaths_end_to_end(tmp_path, monkeypatch):
    names = {11: "Foo Bar", 12: "Baz Qux", 638: "Raven", 33: "Stork"}
    pages = [[zk(3, 11, 638, 5000), zk(2, 12, 33, 198), zk(1, 11, 638, 99)],
             [zk(0, 11, 638, -86400)]]  # a day earlier: paging stops here
    http = FakeHttp(pages, names)
    monkeypatch.setattr(zkill.urllib.request, "urlopen", http)
    monkeypatch.setattr(zkill, "PAGE_DELAY_S", 0)
    db, mid = make_db(tmp_path)

    s = link_deaths(db)
    assert (s["linked"], s["deaths"]) == (2, 2)
    rows = {r["side"] + str(r["slot"]): dict(r) for r in db.conn.execute("SELECT * FROM ships")}
    assert rows["red0"]["killmail_id"] == 1 and rows["red0"]["killmail_hash"] == "h1"
    assert rows["red0"]["zkill_victim"] == "Foo Bar" and rows["red0"]["killmail_delta_s"] == -1
    assert rows["blue0"]["killmail_id"] == 2
    assert rows["blue1"]["killmail_id"] is None
    assert sum("page/" in u for u in http.urls) == 2
    # Everything back to the window's start is cached, newer kills included.
    assert db.conn.execute("SELECT count(*) FROM killmails").fetchone()[0] == 3
    event = db.conn.execute("SELECT payload_json FROM events WHERE kind = 'zkill'").fetchone()[0]
    assert json.loads(event)["linked"] == 2

    # Re-running redoes the same links.
    assert link_deaths(db)["linked"] == 2
    assert db.conn.execute("SELECT count(*) FROM ships WHERE killmail_id IS NOT NULL").fetchone()[0] == 2


def test_linker_retries_until_linked(tmp_path, monkeypatch):
    db, mid = make_db(tmp_path)
    db.close()
    calls = []

    def fake_link(db, match_ids, tolerance_s, region):
        calls.append(match_ids)
        return {"linked": len(calls), "deaths": 2, "offsets": {}}

    monkeypatch.setattr(zkill, "link_deaths", fake_link)
    linker = ZkillLinker(tmp_path / "t.sqlite", delays_s=(0, 0, 0))
    linker.submit(mid)
    for _ in range(200):
        if len(calls) >= 2:
            break
        time.sleep(0.01)
    linker.close()
    assert calls == [[mid], [mid]]  # linked on the second try, so no third
