"""Frame sources: 1 fps, 1920x1080 BGR frames from a live stream or a file.

ffmpeg does the sampling and scaling, so frames arrive already at the HUD's
reference resolution.
"""

import csv
import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

from . import draft, hud
from .layout import FRAME_H, FRAME_W
from .tracker import FrameTime

log = logging.getLogger(__name__)

FRAME_BYTES = FRAME_W * FRAME_H * 3
FFMPEG_OUT = [
    "-an", "-sn",
    "-vf", f"fps=1,scale={FRAME_W}:{FRAME_H}",
    "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
]


def _read_frames(proc: subprocess.Popen) -> Iterator[np.ndarray]:
    while True:
        buf = proc.stdout.read(FRAME_BYTES)
        if len(buf) < FRAME_BYTES:
            return
        yield np.frombuffer(buf, np.uint8).reshape(FRAME_H, FRAME_W, 3)


class FileSource:
    """A local video file or anything ffmpeg opens directly.

    Timestamps are file offsets; utc is start_utc + offset so replays produce
    plausible absolute times.
    """

    def __init__(self, path: str, start: float = 0.0, duration: float | None = None,
                 realtime: bool = False, start_utc: datetime | None = None):
        self.path = path
        self.start = start
        self.duration = duration
        self.realtime = realtime
        self.start_utc = start_utc or datetime.now(timezone.utc)

    def __iter__(self) -> Iterator[tuple[np.ndarray, FrameTime]]:
        cmd = ["ffmpeg", "-loglevel", "error"]
        if self.realtime:
            cmd.append("-re")
        if self.start:
            cmd += ["-ss", str(self.start)]
        # Only 1 frame/s is kept, so skip decoding frames nothing references:
        # about twice as fast, and the frames kept are still exact.
        cmd += ["-skip_frame", "noref"]
        if "://" in self.path:
            # Long network reads (VODs) must survive dropped connections.
            cmd += ["-reconnect", "1", "-reconnect_on_network_error", "1",
                    "-reconnect_delay_max", "30"]
        cmd += ["-i", self.path]
        if self.duration:
            cmd += ["-t", str(self.duration)]
        proc = subprocess.Popen(cmd + FFMPEG_OUT, stdout=subprocess.PIPE)
        try:
            for i, frame in enumerate(_read_frames(proc)):
                offset = self.start + i
                yield frame, FrameTime(self.start_utc + timedelta(seconds=offset), offset)
        finally:
            proc.kill()
            proc.wait()


def resolve_vod(url: str) -> str:
    """Direct media URL for a VOD via yt-dlp, for replaying without downloading."""
    exe = os.environ.get("YT_DLP") or shutil.which("yt-dlp")
    if exe is None:
        raise RuntimeError("yt-dlp not found; put it on PATH or set YT_DLP")
    out = subprocess.run([exe, "-f", "1080p60/1080p/best", "-g", url],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip().splitlines()[0]


class LiveSource:
    """A live Twitch channel via streamlink -> ffmpeg.

    A reader thread keeps only the newest frames, so if processing ever falls
    behind it skips frames instead of drifting behind live. Reconnects with
    backoff when the stream drops or is offline.
    """

    def __init__(self, channel: str, quality: str = "1080p60,1080p,best",
                 max_backoff: float = 60.0):
        self.channel = channel
        self.quality = quality
        self.max_backoff = max_backoff
        self._queue: queue.Queue = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._started = time.monotonic()

    def _put(self, item) -> None:
        while True:
            try:
                self._queue.put_nowait(item)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    log.warning("processing behind live; dropped a frame")
                except queue.Empty:
                    pass

    def _run(self) -> None:
        try:
            self._capture()
        except BaseException as e:  # surface in the consumer instead of hanging it
            self._put(e)

    def _capture(self) -> None:
        backoff = 2.0
        url = self.channel if "://" in self.channel else f"https://twitch.tv/{self.channel}"
        while not self._stop.is_set():
            streamlink = subprocess.Popen(
                [sys.executable, "-m", "streamlink", "--twitch-disable-ads", "--stdout", "--loglevel", "error",
                 url, self.quality],
                stdout=subprocess.PIPE,
            )
            ffmpeg = subprocess.Popen(
                ["ffmpeg", "-loglevel", "error", "-i", "-"] + FFMPEG_OUT,
                stdin=streamlink.stdout, stdout=subprocess.PIPE,
            )
            streamlink.stdout.close()
            got = False
            try:
                for frame in _read_frames(ffmpeg):
                    if self._stop.is_set():
                        break
                    got = True
                    backoff = 2.0
                    now = datetime.now(timezone.utc)
                    self._put((frame, FrameTime(now, time.monotonic() - self._started)))
            finally:
                for p in (ffmpeg, streamlink):
                    p.kill()
                    p.wait()
            if self._stop.is_set():
                break
            log.warning("stream ended or unavailable; reconnecting in %.0fs", backoff)
            self._stop.wait(backoff)
            if not got:
                backoff = min(backoff * 2, self.max_backoff)

    def __iter__(self) -> Iterator[tuple[np.ndarray, FrameTime]]:
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()
        try:
            while True:
                item = self._queue.get()
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            self._stop.set()

    def stop(self) -> None:
        self._stop.set()


# ------------------------------------------------------------------ HUD cache
#
# Streaming a 9-hour VOD takes a long time, so a replay can also save the HUD
# strip of every frame that looks anything like the HUD or shows the draft
# panel, plus an index of all frames. StripCache replays that offline in seconds.

STRIP_TOP = 740  # every HUD region lies below this row
CACHE_MIN_ANCHOR = 0.4  # looser than detection, so near-misses are kept too


class HudRecorder:
    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._index = open(self.dir / "index.csv", "a", newline="")
        self._csv = csv.writer(self._index)

    def add(self, frame: np.ndarray, t: FrameTime) -> None:
        score = max(hud.anchor_scores(frame).values())
        # The draft panel (bans in pick order) is in the strip too.
        saved = score >= CACHE_MIN_ANCHOR or draft.find_panel(frame) is not None
        if saved:
            cv2.imwrite(str(self.dir / f"{int(t.offset):06d}.png"), frame[STRIP_TOP:],
                        [cv2.IMWRITE_PNG_COMPRESSION, 1])
        self._csv.writerow([f"{t.offset:.0f}", t.utc.isoformat(), f"{score:.3f}", int(saved)])
        self._index.flush()

    def close(self) -> None:
        self._index.close()


class StripCache:
    """Replays a HudRecorder directory. Frames without a saved strip come back
    black, which reads as no HUD."""

    def __init__(self, directory: str | Path, start: float = 0.0, duration: float | None = None):
        self.dir = Path(directory)
        self.start = start
        self.end = start + duration if duration is not None else None

    def __iter__(self) -> Iterator[tuple[np.ndarray, FrameTime]]:
        blank = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
        with open(self.dir / "index.csv", newline="") as f:
            for offset, utc, _score, saved in csv.reader(f):
                if float(offset) < self.start:
                    continue
                if self.end is not None and float(offset) >= self.end:
                    break
                t = FrameTime(datetime.fromisoformat(utc), float(offset))
                if saved == "1":
                    strip = cv2.imread(str(self.dir / f"{int(t.offset):06d}.png"))
                    frame = blank.copy()
                    frame[STRIP_TOP:] = strip
                    yield frame, t
                else:
                    yield blank, t
