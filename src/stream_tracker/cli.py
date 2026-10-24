"""stream-tracker command line."""

import argparse
import logging
import time
from datetime import datetime

import cv2
import numpy as np

from . import hud, layout as L, reader
from pathlib import Path

from .capture import FileSource, HudRecorder, LiveSource, StripCache, resolve_vod
from .db import Database
from .tracker import Tracker
from .vocab import update_ship_csv

log = logging.getLogger("stream_tracker")


def parse_time(text: str) -> float:
    """'1:44:54', '104:54' or '6294' -> seconds."""
    total = 0.0
    for part in text.split(":"):
        total = total * 60 + float(part)
    return total


def run(source, tracker: Tracker, report_every: int = 600, recorder: HudRecorder | None = None) -> None:
    times = []
    t = None
    try:
        for frame, t in source:
            start = time.perf_counter()
            r = tracker.process(frame, t)
            times.append(time.perf_counter() - start)
            if recorder is not None:
                recorder.add(frame, t)
            if len(times) % report_every == 0:
                ms = np.array(times[-report_every:]) * 1000
                log.info("t=%.0fs phase=%s clock=%s scores=%s | %.1fms mean, %.1fms max",
                         t.offset, r.phase.value, r.clock, r.scores, ms.mean(), ms.max())
    except KeyboardInterrupt:
        log.info("interrupted")
    finally:
        tracker.close(t)
        if recorder is not None:
            recorder.close()
        if times:
            ms = np.array(times) * 1000
            log.info("%d frames: p50 %.1fms, p99 %.1fms, max %.1fms",
                     len(ms), np.percentile(ms, 50), np.percentile(ms, 99), ms.max())


def cmd_live(args) -> None:
    db = Database(args.db)
    tracker = Tracker(db, channel=args.channel, source="live", grace_s=args.grace,
                      debug_dir=args.debug_dir)
    run(LiveSource(args.channel, quality=args.quality), tracker)


def cmd_replay(args) -> None:
    if (Path(args.input) / "index.csv").exists():
        source = StripCache(args.input)
    else:
        path = args.input
        if "://" in path and ".m3u8" not in path:
            path = resolve_vod(path)
        start_utc = datetime.fromisoformat(args.start_utc) if args.start_utc else None
        source = FileSource(
            path,
            start=parse_time(args.start) if args.start else 0.0,
            duration=parse_time(args.duration) if args.duration else None,
            realtime=args.realtime,
            start_utc=start_utc,
        )
    db = Database(args.db)
    tracker = Tracker(db, channel=args.channel, source=args.input, grace_s=args.grace,
                      debug_dir=args.debug_dir)
    recorder = HudRecorder(args.save_hud) if args.save_hud else None
    run(source, tracker, recorder=recorder)


def cmd_calibrate(args) -> None:
    """Draw every ROI over a frame; optionally save it as the HUD anchor reference."""
    frame = cv2.imread(args.frame)
    if frame.shape[:2] != (L.FRAME_H, L.FRAME_W):
        frame = cv2.resize(frame, (L.FRAME_W, L.FRAME_H))
    out = frame.copy()
    boxes = [L.TIMER, L.RED_TEAM, L.RED_SCORE, L.BLUE_SCORE, L.BLUE_TEAM, *hud.ANCHOR_BOXES.values()]
    for side in reader.SIDES:
        boxes += [L.ban_box(side, i) for i in range(L.MAX_BANS)]
        for i in range(L.MAX_ROWS):
            boxes += [L.row_box(side, i, c) for c in ("pts", "name", "speed", "bars", "ship")]
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 1)
    cv2.imwrite(args.out, out)
    print(f"wrote {args.out}")
    if args.save_anchors:
        hud.save_anchors(frame)
        print(f"saved anchors to {hud.ANCHORS_FILE}")
    if hud.ANCHORS_FILE.exists():
        print("anchor scores:", hud.anchor_scores(frame))
        r = reader.read_frame(frame)
        print("reading:", r)


def cmd_update_ships(args) -> None:
    print(f"wrote {update_ship_csv()} ships")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="stream-tracker")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--db", default="matches.sqlite")
        p.add_argument("--grace", type=float, default=90.0,
                       help="seconds the HUD may be off screen before a live match is ended")
        p.add_argument("--debug-dir", help="save a frame at every match event")

    p = sub.add_parser("live", help="track a live Twitch channel")
    p.add_argument("channel", help="channel name or stream URL")
    p.add_argument("--quality", default="1080p60,1080p,best")
    common(p)
    p.set_defaults(func=cmd_live)

    p = sub.add_parser("replay", help="track a recording, VOD URL or --save-hud cache")
    p.add_argument("input", help="video file, VOD/m3u8 URL, or a --save-hud directory")
    p.add_argument("--save-hud", help="cache HUD strips here for fast offline re-runs")
    p.add_argument("--start", help="offset into the input, e.g. 1:40:00")
    p.add_argument("--duration")
    p.add_argument("--realtime", action="store_true", help="read at playback speed")
    p.add_argument("--start-utc", help="ISO time of the input's start, for absolute timestamps")
    p.add_argument("--channel")
    common(p)
    p.set_defaults(func=cmd_replay)

    p = sub.add_parser("calibrate", help="overlay ROIs on a frame and check readings")
    p.add_argument("frame")
    p.add_argument("--out", default="calibrate.png")
    p.add_argument("--save-anchors", action="store_true",
                   help="use this frame as the HUD-present reference")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("update-ships", help="refresh data/ship_points.csv from the AT points sheet")
    p.set_defaults(func=cmd_update_ships)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
