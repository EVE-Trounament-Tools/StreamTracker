"""Snap OCR'd ship names to the AT ship list (data/ship_points.csv).

The HUD shows navy/fleet hulls in several ways: full name ("Magnate Navy
Issue"), abbreviated ("N. Magnate", "F. Cyclone"), or cut off by the column
width ("Dragoon Navy"). Every form is generated from the sheet and maps back
to the full name.
"""

import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from rapidfuzz import fuzz, process

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SHIP_CSV = DATA_DIR / "ship_points.csv"
SHEET_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/1AVYlWlvuMKnA3yuqqDCcAkia8pvhpb9OBcM29WFw5rM"
    "/export?format=csv&gid=284772315"
)
MIN_SCORE = 80
TIE_MARGIN = 12
MAX_DUPLICATES = 10


@dataclass(frozen=True)
class Ship:
    name: str
    ship_class: str
    points: int
    hull: str
    inflation: int

    def fits_points(self, pts: int) -> bool:
        """True when pts is this hull's base cost plus k duplicate inflations."""
        if self.inflation == 0:
            return pts == self.points
        k, rem = divmod(pts - self.points, self.inflation)
        return rem == 0 and 0 <= k < MAX_DUPLICATES


@dataclass(frozen=True)
class Match:
    raw: str
    ship: Ship | None
    score: float


def load_ships(path: Path = SHIP_CSV) -> list[Ship]:
    with open(path, newline="") as f:
        return [
            Ship(r["name"].strip(), r["ship_class"].strip(), int(r["points"]), r["hull"].strip(),
                 int(r["inflation"] or 0))
            for r in csv.DictReader(f)
        ]


def aliases(name: str) -> set[str]:
    """Display forms of a ship name, including the full name."""
    forms = {name}
    m = re.fullmatch(r"(.+) (Navy|Fleet) Issue", name)
    if m:
        base, kind = m.groups()
        forms.add(f"{kind[0]}. {base}")
        forms.add(f"{base} {kind}")
    # Faction prefixes: "Republic Fleet Firetail" -> "R.F. Firetail".
    m = re.fullmatch(r"(\w+) (Navy|Fleet) (\w+)", name)
    if m and m.group(3) != "Issue":
        forms.add(f"{m.group(1)[0]}.{m.group(2)[0]}. {m.group(3)}")
        forms.add(f"{m.group(2)[0]}. {m.group(3)}")
    return forms


def clean(raw: str) -> str:
    text = raw.replace("…", "").replace("...", "").strip()
    # "N.Magnate" / "N Magnate" -> "N. Magnate"
    text = re.sub(r"^([A-Z](?:\.[A-Z])?)\.?\s*(?=[A-Z][a-z])", r"\1. ", text)
    return re.sub(r"\s+", " ", text)


def _score(text: str, form: str, **kwargs) -> float:
    """fuzz.ratio, but a long text that is a prefix of a form also scores well,
    since the HUD cuts names off at the column edge."""
    score = fuzz.ratio(text, form)
    if len(text) >= 8 and len(form) > len(text):
        score = max(score, fuzz.ratio(text, form[: len(text)]) - 3)
    return score


class ShipVocab:
    def __init__(self, ships: list[Ship]):
        self.ships = {s.name: s for s in ships}
        self.forms: dict[str, str] = {}
        for s in ships:
            for form in aliases(s.name):
                self.forms.setdefault(form.lower(), s.name)
        self._choices = list(self.forms)

    def match(self, raw: str, pts: int | None = None) -> Match:
        text = clean(raw)
        if not text:
            return Match(raw, None, 0.0)
        hits = process.extract(text.lower(), self._choices, scorer=_score, limit=8)
        best: dict[str, float] = {}
        for form, score, _ in hits:
            name = self.forms[form]
            best[name] = max(best.get(name, 0.0), score)
        ranked = sorted(best.items(), key=lambda kv: -kv[1])
        name, score = ranked[0]
        if pts is not None:
            close = [n for n, sc in ranked if sc >= score - TIE_MARGIN]
            fitting = [n for n in close if self.ships[n].fits_points(pts)]
            if fitting:
                name = fitting[0]
                score = best[name]
        if score < MIN_SCORE:
            return Match(raw, None, score)
        return Match(raw, self.ships[name], score)


TEAMS_FILE = DATA_DIR / "teams.txt"
TEAM_SNAP_SCORE = 88


class TeamNames:
    """Snaps an OCR'd team name to a known spelling when they're nearly the
    same (an OCR slip like "SGOLYARIS", or an overlay typo like "ONLYFEET.")."""

    def __init__(self, names: list[str] | None = None):
        self.names: list[str] = []
        for n in names or []:
            self.add(n)

    @classmethod
    def from_file(cls, path: Path = TEAMS_FILE) -> "TeamNames":
        names = path.read_text().splitlines() if path.exists() else []
        return cls([n.strip() for n in names if n.strip()])

    def add(self, name: str) -> None:
        if name and name not in self.names:
            self.names.append(name)

    def snap(self, name: str | None) -> str | None:
        if not name or not self.names:
            return name
        hit = process.extractOne(name, self.names, scorer=fuzz.ratio,
                                 processor=str.upper, score_cutoff=TEAM_SNAP_SCORE)
        return hit[0] if hit else name


KNOWN_TEAMS_FILE = DATA_DIR / "known_teams.txt"
BEST_FIT_MIN = 60
PREFIX_MIN = 4  # a cut-off name must keep this many characters to count as a prefix


def _team_words(name: str) -> list[str]:
    """Upper-case words, slashed zero read as O, punctuation dropped."""
    return re.sub(r"[^A-Z0-9 ]", " ", name.upper().replace("0", "O")).split()


class KnownTeams:
    """Best-fits a team name to the tournament's own spelling of it.

    Unlike TeamNames.snap, which only fixes near-identical OCR slips, this
    always picks the closest known team, so it only suits a phase whose
    teams are all known in advance (not the Qualifiers). It handles names
    the overlay cuts off ("XQTY", "34-YRS-OLD MEN"), shortens ("CATLAS TOR
    USECODEWIZARD", "THE BLUE.") or letter-spaces ("L A Z E R H A W K S"),
    plus aliases for overlay names that look nothing like the team's
    ("TEETH" for "IIIIIIII…").

    The file has one team per line, its spelling as the tournament has it,
    optionally followed by " | alias | alias…"; lines starting with # are
    comments. A tournament JSON (a "teams" list of {"name": …}) also works.
    """

    def __init__(self, names: list[str], aliases: dict[str, str] | None = None):
        self.names = names
        self.aliases = {"".join(_team_words(a)): n for a, n in (aliases or {}).items()}
        self._keys = {n: ("".join(_team_words(n)), " ".join(_team_words(n))) for n in names}

    @classmethod
    def from_file(cls, path: Path = KNOWN_TEAMS_FILE) -> "KnownTeams":
        if path.suffix == ".json":
            import json
            return cls([t["name"] for t in json.loads(path.read_text())["teams"]])
        names, aliases = [], {}
        for line in path.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            name, *alts = (part.strip() for part in line.split("|"))
            names.append(name)
            aliases.update({a: name for a in alts if a})
        return cls(names, aliases)

    def score(self, name: str, known: str) -> float:
        squashed, words = "".join(_team_words(name)), " ".join(_team_words(name))
        k_squashed, k_words = self._keys[known]
        best = fuzz.ratio(squashed, k_squashed)
        if len(squashed) >= PREFIX_MIN and k_squashed.startswith(squashed):
            best = max(best, 95)
        # every word of the name in the team's ("THE BLUE." in "OUT OF THE BLUE.")
        return max(best, 0.95 * fuzz.token_set_ratio(words, k_words))

    def fit(self, name: str | None, exclude: tuple[str | None, ...] = ()) -> str | None:
        """The known team closest to name, else name unchanged when nothing
        scores BEST_FIT_MIN. Teams in exclude (the other side's) are skipped."""
        if not name or not self.names:
            return name
        alias = self.aliases.get("".join(_team_words(name)))
        if alias and alias not in exclude:
            return alias
        scored = [(self.score(name, k), k) for k in self.names if k not in exclude]
        if not scored:
            return name
        score, best = max(scored)
        return best if score >= BEST_FIT_MIN else name

    def fit_pair(self, red: str | None, blue: str | None) -> tuple[str | None, str | None]:
        """Fit both sides, never to the same team: the clearer read picks first."""
        if not (red and blue):
            return self.fit(red), self.fit(blue)
        r, b = self.fit(red), self.fit(blue)
        if r != b:
            return r, b
        if self.score(red, r) >= self.score(blue, b):
            return r, self.fit(blue, exclude=(r,))
        return self.fit(red, exclude=(b,)), b


@lru_cache(maxsize=1)
def default_vocab() -> ShipVocab:
    return ShipVocab(load_ships())


def update_ship_csv(path: Path = SHIP_CSV, url: str = SHEET_CSV_URL) -> int:
    """Download the sheet's "New Static Values" tab into ship_points.csv."""
    import io
    import urllib.request

    with urllib.request.urlopen(url, timeout=30) as resp:
        rows = list(csv.reader(io.StringIO(resp.read().decode("utf-8"))))
    n = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "ship_class", "points", "hull", "inflation"])
        for r in rows[1:]:
            # Columns F-J: Ship Name, Ship Class, Points, Hull Type, Inflation Value.
            if len(r) > 9 and r[5].strip() and r[7].strip().isdigit():
                w.writerow([c.strip() for c in r[5:10]])
                n += 1
    default_vocab.cache_clear()
    return n
