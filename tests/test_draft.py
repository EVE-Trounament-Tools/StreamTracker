import cv2
import numpy as np
import pytest

from stream_tracker import draft
from stream_tracker.vocab import default_vocab

from conftest import FIXTURES, STRIP_TOP


def draft_frame(name):
    strip = cv2.imread(str(FIXTURES / f"draft_{name}.png"))
    return np.vstack([np.zeros((STRIP_TOP, strip.shape[1], 3), np.uint8), strip])


AT22 = ("Odin's Call", "The Skeleton Crew",
        [("red", "Armageddon"), ("blue", "Sleipnir"), ("blue", "Tholos"), ("red", "Zarmazd"),
         ("red", "Armageddon Navy Issue"), ("blue", "Leshak")])
AT21 = ("Meta Reloaded", "Evasive Maneuvering.",
        [("red", "Cenotaph"), ("blue", "Rapier"), ("blue", "Huginn"), ("red", "Raven Navy Issue"),
         ("red", "Apocalypse Navy Issue"), ("blue", "Armageddon Navy Issue"), ("red", "Basilisk"),
         ("blue", "Vexor")])


# A red name on two lines moves the team labels up (2873111426 at 42:23).
AT22_WRAP = ("That Escalated Quickly.", "Kitchen Sinkhole",
             [("red", "Tholos"), ("blue", "Keres"), ("blue", "Zarmazd"), ("red", "Absolution"),
              ("red", "Astarte"), ("blue", "Basilisk")])


@pytest.mark.parametrize("name, expected", [("at22", AT22), ("at21", AT21), ("at22_wrap", AT22_WRAP)])
def test_draft_panel_in_pick_order(name, expected):
    """VOD 2873111426 8:10:49 (AT XXII, 6 bans) and 2601816926 6:14:58 (AT XXI,
    8 bans, a narrower layout further down the screen)."""
    red, blue, bans = expected
    frame = draft_frame(name)
    panel = draft.find_panel(frame)
    assert panel is not None
    assert [side for _, _, side in panel.bars] == [side for side, _ in bans]
    read = draft.read_panel(frame, panel)
    assert read.teams == {"red": red, "blue": blue}
    vocab = default_vocab()
    names = [(side, vocab.match(raw).ship.name if vocab.match(raw).ship else raw) for side, raw in read.bans]
    assert names == bans  # Cenotaph isn't in the AT XXII ship list and stays raw


@pytest.mark.parametrize("offset", [30, 540, 590, 628, 655, 1000, 1260, 1266])
def test_no_draft_panel_on_other_frames(frame, offset):
    assert draft.find_panel(frame(offset)) is None
