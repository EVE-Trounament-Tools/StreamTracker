"""Build digit templates (data/glyphs_header.npz, data/glyphs_row.npz).

Ground truth comes from the reference VOD segment (AT XXII, Neon Nightmares vs
Sedition., v2860628871 from 1:40:00): the match clock reads 9:44 at 662s into
the segment and counts down one per second, and the rosters' PTS are fixed.
Speed digits are labelled by tesseract and only kept when it reads a digit
it never confuses ('9' is otherwise missing from PTS).

usage: python tools/build_glyphs.py <segment.mp4> [--sheet out.png]
"""

import argparse
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stream_tracker import glyphs, layout as L, ocr  # noqa: E402

CLOCK_AT_662 = 9 * 60 + 44
LIVE_FROM, LIVE_TO = 640, 1240
RED_PTS = [36, 36, 32, 31, 16, 13, 11, 11, 8, 6]
BLUE_PTS = [33, 32, 30, 26, 25, 14, 14, 12, 7, 7]


def frame_at(video: str, t: float) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-ss", str(t), "-i", video, "-frames:v", "1",
         "-vf", "scale=1920:1080", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, np.uint8).reshape(L.FRAME_H, L.FRAME_W, 3)


def add(samples, region, text):
    gs = glyphs.segment(region)
    if len(gs) != len(text):
        return False
    lh = glyphs.line_height(gs)
    for (_, g), ch in zip(gs, text):
        samples[ch].append(glyphs.normalize(g, lh))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--sheet")
    args = ap.parse_args()

    header = defaultdict(list)
    row = defaultdict(list)
    for t in range(LIVE_FROM, LIVE_TO, 3):
        f = frame_at(args.video, t)
        clock = CLOCK_AT_662 - (t - 662)
        add(header, L.crop(f, L.TIMER), f"{clock // 60}:{clock % 60:02d}")
        if (t - LIVE_FROM) % 30 == 0:
            for side, pts in (("red", RED_PTS), ("blue", BLUE_PTS)):
                for i, p in enumerate(pts):
                    add(row, L.crop(f, L.row_box(side, i, "pts")), str(p))
                    sp = L.crop(f, L.row_box(side, i, "speed"))
                    text = ocr.read_line(sp, whitelist=ocr.DIGITS)
                    gs = glyphs.segment(sp)
                    if len(gs) == len(text):
                        lh = glyphs.line_height(gs)
                        for (_, g), ch in zip(gs, text):
                            if ch == "9":
                                row[ch].append(glyphs.normalize(g, lh))
    # Pre-match "-00:00-" supplies '-'.
    for t in (575, 590, 600):
        add(header, L.crop(frame_at(args.video, t), L.TIMER), "-00:00-")
    # Scores are in the same font as the clock.
    for t, (r, b) in ((1000, ("88", "97")), (1260, ("114", "128"))):
        f = frame_at(args.video, t)
        add(header, L.crop(f, L.RED_SCORE), r)
        add(header, L.crop(f, L.BLUE_SCORE), b)

    out = Path(__file__).resolve().parents[1] / "data"
    sheet_rows = []
    for font, samples in (("header", header), ("row", row)):
        labels = sorted(samples)
        templates = np.stack([np.median(np.stack(samples[k]), axis=0) for k in labels])
        np.savez(out / f"glyphs_{font}.npz", labels=np.array(labels), templates=templates)
        print(font, {k: len(samples[k]) for k in labels})
        sheet_rows.append(np.hstack([np.pad(t, 2) for t in templates]))
    if args.sheet:
        w = max(r.shape[1] for r in sheet_rows)
        img = np.vstack([np.pad(r, ((0, 0), (0, w - r.shape[1]))) for r in sheet_rows])
        cv2.imwrite(args.sheet, cv2.resize((img * 255).astype(np.uint8), None, fx=8, fy=8,
                                           interpolation=cv2.INTER_NEAREST))


if __name__ == "__main__":
    main()
