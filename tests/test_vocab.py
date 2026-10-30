import pytest

from stream_tracker.vocab import clean, default_vocab, load_ships


@pytest.fixture(scope="module")
def vocab():
    return default_vocab()


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("N. Magnate", "Magnate Navy Issue"),
        ("N.Magnate", "Magnate Navy Issue"),
        ("N Magnate", "Magnate Navy Issue"),
        ("Magnate Navy Issue", "Magnate Navy Issue"),
        ("F. Cyclone", "Cyclone Fleet Issue"),
        ("F. Scythe", "Scythe Fleet Issue"),
        ("N. Prophecy", "Prophecy Navy Issue"),
        ("N. Augoror", "Augoror Navy Issue"),
        ("Dragoon Navy", "Dragoon Navy Issue"),
        ("Hurricane Fleet Is", "Hurricane Fleet Issue"),
        ("Sleipnir", "Sleipnir"),
        ("Svipul", "Svipul"),
        ("Zarmmazd", "Zarmazd"),
        ("Khizriel", "Khizriel"),
    ],
)
def test_match(vocab, raw, expected):
    assert vocab.match(raw).ship.name == expected


def test_points_break_ties(vocab):
    # A misread prefix leaves "Maulus" and "N. Maulus" close; PTS decides.
    navy = vocab.ships["Maulus Navy Issue"]
    assert vocab.match("H. Maulus", pts=7).ship.name == "Maulus"
    assert vocab.match("H. Maulus", pts=navy.points).ship.name == "Maulus Navy Issue"
    assert vocab.match("Maulus", pts=7).ship.name == "Maulus"


def test_fits_points(vocab):
    sleipnir = vocab.ships["Sleipnir"]
    assert sleipnir.fits_points(33) and sleipnir.fits_points(36)
    assert not sleipnir.fits_points(34)


def test_every_ship_matches_itself(vocab):
    for ship in load_ships():
        assert vocab.match(ship.name).ship.name == ship.name


def test_garbage_is_rejected(vocab):
    assert vocab.match("xq#@").ship is None
    assert vocab.match("").ship is None


def test_clean():
    assert clean("Dr Clutch McSwagD…") == "Dr Clutch McSwagD"
    assert clean("N.Magnate") == "N. Magnate"


def test_team_names_snap():
    from stream_tracker.vocab import TeamNames

    teams = TeamNames(["SOLYARIS CHTONIUM", "ONLYFLEETS.", "THE BLUE.", "OUT OF THE BLUE."])
    assert teams.snap("SGOLYARIS CHTONIUM") == "SOLYARIS CHTONIUM"
    assert teams.snap("ONLYFEET.") == "ONLYFLEETS."
    assert teams.snap("THE BLUE.") == "THE BLUE."
    assert teams.snap("NEW TEAM") == "NEW TEAM"


def test_known_teams_best_fit():
    from stream_tracker.vocab import KnownTeams

    teams = KnownTeams.from_file()
    # cut off, shortened, letter-spaced, slashed zero
    assert teams.fit("XQTY") == "xqtywiznalamywmodxfhhopawzpqyjdwrpeptuaenabjawdzku"
    assert teams.fit("XOTYWIZNALAMYWMODXFHHOPAWZPQYJOWRPEPTUAENABJAWDZKU") == \
        "xqtywiznalamywmodxfhhopawzpqyjdwrpeptuaenabjawdzku"
    assert teams.fit("34-YRS-OLD MEN") == "34-yrs-old men get nagged by wives for playing EVE"
    assert teams.fit("CATLAS TOR") == "Catlas Tor Love Appreciation Society UseCodeWIZARD"
    assert teams.fit("CATLAS TOR USECODEWIZARD") == "Catlas Tor Love Appreciation Society UseCodeWIZARD"
    assert teams.fit("LAZERHAWKS") == "L A Z E R H A W K S"
    assert teams.fit("WE FORM VOLTA") == "WE FORM V0LTA"
    assert teams.fit("THE BLUE.") == "Out of the Blue."
    assert teams.fit("UNC KAPELLE") == "Rote Kapelle"
    assert teams.fit("SGOLYARIS CHTONIUM") == "Solyaris Chtonium"
    # aliases for overlay names that look nothing like the team
    assert teams.fit("TEETH") == "IIIIIIIIIIIIIIIIIIIIIIIIIII"
    # nothing close: left as read
    assert teams.fit("QQQQ") == "QQQQ"
    assert teams.fit(None) is None


def test_known_teams_pair_never_fits_both_sides_to_one_team():
    from stream_tracker.vocab import KnownTeams

    teams = KnownTeams(["ROTE KAPELLE", "KAPELLE ROTE ZWEI"])
    assert teams.fit_pair("ROTE KAPELLE", "KAPELLE") == ("ROTE KAPELLE", "KAPELLE ROTE ZWEI")


def test_known_teams_from_tournament_json(tmp_path):
    from stream_tracker.vocab import KnownTeams

    path = tmp_path / "t.json"
    path.write_text('{"teams": [{"name": "WE FORM V0LTA"}, {"name": "Seker Matar"}]}')
    assert KnownTeams.from_file(path).fit("SEKER MATAR") == "Seker Matar"
