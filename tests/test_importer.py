import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from stream_tracker.cli import main
from stream_tracker.db import Database
from stream_tracker.importer import ImportClient
from stream_tracker.tracker import FrameTime, Tracker

from test_tracker import T0, feed


@pytest.fixture
def server():
    """A stub endpoint that answers with the queued statuses, then 200."""
    state = {"statuses": [], "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            state["requests"].append((self.headers, body))
            status = state["statuses"].pop(0) if state["statuses"] else 200
            out = {"action": "create", "game_no": 1, "reasons": [], "warnings": []} if status == 200 else {"error": "x"}
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(out).encode())

        def log_message(self, *a):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{httpd.server_port}/import"
    yield state
    httpd.shutdown()


def test_posts_one_object_with_token(server):
    client = ImportClient(server["url"], "test-token", retry_delay_s=0)
    result = client._post({"run_id": "r", "match_id": 1})
    assert result["action"] == "create"
    headers, body = server["requests"][0]
    assert headers["x-import-token"] == "test-token"
    assert body == {"run_id": "r", "match_id": 1}


def test_retries_server_errors_only(server):
    client = ImportClient(server["url"], "t", retry_delay_s=0)
    server["statuses"] = [500, 500]
    assert client._post({"match_id": 1})["action"] == "create"
    assert len(server["requests"]) == 3
    server["statuses"] = [422]
    assert client._post({"match_id": 2}) is None
    assert len(server["requests"]) == 4


def test_unreachable_never_raises():
    client = ImportClient("http://127.0.0.1:9/none", "t", retry_delay_s=0)
    assert client._post({"match_id": 1}) is None


def test_post_needs_env(monkeypatch, tmp_path):
    monkeypatch.delenv("OBSERVED_IMPORT_URL", raising=False)
    monkeypatch.delenv("OBSERVED_IMPORT_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="OBSERVED_IMPORT_URL"):
        main(["replay", str(tmp_path / "123456789.mp4"), "--post", "--db", str(tmp_path / "x.sqlite")])


def test_finished_match_is_exported_from_memory(frame, tmp_path):
    """A recorded match is handed on as one import object, and the DB is still written."""
    sent = []
    db = Database(tmp_path / "t.sqlite")
    tracker = Tracker(db, channel="test", source="runs/42", run_id="42", on_match_end=sent.append)
    seq = [(590 + i, 590) for i in range(25)]
    seq += [(628, 628), (629, 628), (655, 655), (662, 662), (663, 662)]
    seq += [(1000, 1000), (1001, 1000), (1260, 1260), (1261, 1260)]
    seq += [(1262 + i, 1260) for i in range(10)]
    feed(tracker, [(t, frame(src)) for t, src in seq])
    tracker.close(FrameTime(T0, 1300))

    assert len(sent) == 1
    m = sent[0]
    assert m["run_id"] == "42" and m["match_id"] == 1
    assert m["red_team"] and m["blue_team"] and m["status"] == "complete"
    row = dict(db.conn.execute("SELECT * FROM matches WHERE id = 1").fetchone())
    for k in ("created_utc", "live_start_utc", "ended_utc", "end_reason", "red_team", "blue_team",
              "red_score", "blue_score", "joined_late", "status"):
        assert m[k] == row[k], k
    ships = {(s["side"], s["slot"]): s for s in m["ships"]}
    for r in db.conn.execute("SELECT * FROM ships WHERE match_id = 1"):
        s = ships[(r["side"], r["slot"])]
        assert (s["pilot"], s["ship_type"], s["points"], s["died_utc"]) == (
            r["pilot"], r["ship_type"], r["points"], r["died_utc"])
    bans = [(b["side"], b["ordinal"], b["ship_type"]) for b in m["bans"]]
    assert bans == [(r["side"], r["ordinal"], r["ship_type"]) for r in db.conn.execute(
        "SELECT * FROM bans WHERE match_id = 1 ORDER BY side DESC, ordinal")]
    assert [e["kind"] for e in m["events"]] == [r["kind"] for r in db.conn.execute(
        "SELECT kind FROM events WHERE match_id = 1 ORDER BY id")]
    json.dumps(m)
