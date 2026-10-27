"""Build static HTML pages for browsing tracker databases offline (one DB per day).

usage: python tools/build_site.py runs/final/*.sqlite --out runs/final/site

All pages go in one flat folder with relative links, so it opens straight from disk.
"""

import argparse
import json
import sqlite3
from collections import Counter
from datetime import datetime
from html import escape
from pathlib import Path

MATCH_S = 600
VOD_LEAD_S = 5  # start VOD links this many seconds before the moment

CSS = """
:root { --bg:#0d1117; --panel:#131a24; --line:#232c3a; --text:#d6dde8; --dim:#6b778a;
  --red:#ff6b6b; --blue:#8b9cff; --win:#4ade80; --lose:#f87171; --accent:#58a6ff; --warn:#f0a35a; }
* { box-sizing:border-box; }
body { margin:0; padding:16px; background:var(--bg); color:var(--text);
  font:13px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; }
a { color:var(--accent); text-decoration:none; } a:hover { text-decoration:underline; }
h1 { font-size:18px; margin:4px 0 8px; } h2 { font-size:14px; margin:0 0 8px; }
.nav { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:10px; }
.nav a { border:1px solid var(--line); border-radius:4px; padding:2px 8px; background:var(--panel); }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:10px 12px; margin-bottom:12px; }
.row { display:flex; gap:12px; flex-wrap:wrap; } .row > .panel { flex:1 1 420px; min-width:0; }
.panel.red { border-top:2px solid var(--red); } .panel.blue { border-top:2px solid var(--blue); }
table { border-collapse:collapse; width:100%; }
th { text-align:left; color:var(--dim); font-size:11px; text-transform:uppercase; font-weight:600;
  padding:4px 6px; border-bottom:1px solid var(--line); }
td { padding:4px 6px; border-bottom:1px solid var(--line); vertical-align:middle; }
tr:last-child td { border-bottom:none; }
td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
.dim { color:var(--dim); }
tr.dead td { color:var(--dim); } tr.dead td.strike { text-decoration:line-through; }
.badge { display:inline-block; border:1px solid var(--line); border-radius:4px; padding:0 6px;
  font-size:11px; color:var(--dim); white-space:nowrap; }
.badge.wipe { color:var(--lose); border-color:#5a2a2a; } .badge.time { color:var(--accent); border-color:#1f3a5a; }
.badge.hud_lost, .badge.warn { color:var(--warn); border-color:#5a3d1f; }
.chip { display:inline-block; border:1px solid var(--line); border-radius:4px; padding:1px 8px; margin:2px 4px 2px 0; }
.chip .n { color:var(--dim); margin-right:4px; }
.team { display:inline-block; border:1px solid var(--line); border-radius:4px; padding:1px 8px; }
.team.win { color:var(--win); border-color:#1f5a35; } .team.lose { color:var(--lose); border-color:#5a2a2a; }
.team.draw { color:var(--text); }
.red-t { color:var(--red); } .blue-t { color:var(--blue); }
.meta { display:flex; gap:14px; flex-wrap:wrap; color:var(--dim); margin-bottom:12px; align-items:center; }
.meta b { color:var(--text); font-weight:500; }
.score { font-size:16px; font-weight:600; font-variant-numeric:tabular-nums; }
.phead { display:flex; justify-content:space-between; align-items:baseline; gap:8px; margin-bottom:6px; }
.timeline { position:relative; height:44px; margin:6px 0 18px; border-bottom:1px solid var(--line); }
.timeline .tick { position:absolute; width:3px; margin-left:-1px; border-radius:1px; }
.timeline .tick.red { background:var(--red); top:2px; height:18px; }
.timeline .tick.blue { background:var(--blue); top:22px; height:18px; }
.timeline .lbl { position:absolute; top:46px; font-size:10px; color:var(--dim); transform:translateX(-50%); }
.timeline .end { position:absolute; top:0; bottom:0; border-left:1px dashed var(--warn); }
.wrap { overflow-x:auto; }
tr.click:hover td { background:#18212e; }
"""


def ts(s):
    return datetime.fromisoformat(s) if s else None


def mmss(s):
    if s is None:
        return "—"
    s = int(s)
    return f"{s // 60}:{s % 60:02d}"


def hms(s):
    s = max(0, int(s))
    return f"{s // 3600}h{s % 3600 // 60:02d}m{s % 60:02d}s"


def page(title, body):
    return (f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{escape(title)}</title><style>{CSS}</style></head>\n<body>\n{body}\n</body></html>\n")


def ship_name(row):
    return row["ship_type"] or row["ship_raw"] or "?"


def winner(m):
    rs, bs = m["red_score"], m["blue_score"]
    if rs is None or bs is None or rs == bs:
        return None
    return "red" if rs > bs else "blue"


def team_chip(m, side):
    w = winner(m)
    cls = "draw" if w is None else ("win" if w == side else "lose")
    mark = " ✓" if w == side else ""
    return f'<span class="team {cls}">{escape(m[side + "_team"] or "?")}{mark}</span>'


def end_badge(m):
    reason = m["end_reason"] or "?"
    out = f'<span class="badge {escape(reason)}">{escape(reason)}</span>'
    if reason not in ("time", "wipe"):
        out += ' <span class="badge warn">incomplete</span>'
    if m["tidi_overtime_s"]:
        out += f' <span class="badge warn">TiDi +{mmss(m["tidi_overtime_s"])}</span>'
    return out


def load_day(path):
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    matches = []
    for m in conn.execute("SELECT * FROM matches ORDER BY live_start_utc, id"):
        m = dict(m)
        m["ships"] = [dict(s) for s in conn.execute(
            "SELECT * FROM ships WHERE match_id=? ORDER BY side DESC, slot", (m["id"],))]
        m["bans"] = [dict(b) for b in conn.execute(
            "SELECT * FROM bans WHERE match_id=? ORDER BY side DESC, ordinal", (m["id"],))]
        live = conn.execute(
            "SELECT stream_offset_s, payload_json FROM events WHERE match_id=? AND kind='live' ORDER BY id LIMIT 1",
            (m["id"],)).fetchone()
        m["live_offset_s"] = None
        if live and live["stream_offset_s"] is not None:
            clock = json.loads(live["payload_json"] or "{}").get("clock")
            # the live event fires when the clock is first read; back up to 10:00
            m["live_offset_s"] = live["stream_offset_s"] - (MATCH_S - clock if clock else 0)
        matches.append(m)
    conn.close()
    source = matches[0]["source"] if matches else ""
    return {"vod": Path(source or path).name, "matches": matches,
            "date": (matches[0]["live_start_utc"] or "")[:10] if matches else "?"}


def vod_link(vod, offset, text):
    if offset is None or not vod.isdigit():
        return text
    return (f'<a href="https://www.twitch.tv/videos/{vod}?t={hms(offset - VOD_LEAD_S)}" '
            f'target="_blank" rel="noopener" title="Open VOD at this moment">{text}</a>')


def match_file(day_n, match_n):
    return f"day-{day_n}-match-{match_n:02d}.html"


def render_index(days):
    rows = []
    flown, banned = Counter(), Counter()
    for n, d in enumerate(days, 1):
        reasons = Counter(m["end_reason"] for m in d["matches"])
        for m in d["matches"]:
            flown.update(ship_name(s) for s in m["ships"])
            banned.update(ship_name(b) for b in m["bans"])
        vod = escape(d["vod"])
        rows.append(
            f'<tr class="click"><td><a href="day-{n}.html">Day {n}</a></td><td>{escape(d["date"])}</td>'
            f'<td class="num">{len(d["matches"])}</td><td class="num">{reasons["wipe"]}</td>'
            f'<td class="num">{reasons["time"]}</td><td class="num">{sum(v for k, v in reasons.items() if k not in ("wipe", "time"))}</td>'
            f'<td><a href="https://www.twitch.tv/videos/{vod}" target="_blank" rel="noopener">{vod}</a></td></tr>')
    total = sum(len(d["matches"]) for d in days)

    def top(counter):
        return "".join(f'<tr><td>{escape(k)}</td><td class="num">{v}</td></tr>' for k, v in counter.most_common(15))

    body = f"""<h1>Alliance Tournament XXII Prelims</h1>
<div class="meta"><span><b>{len(days)}</b> days</span><span><b>{total}</b> matches</span></div>
<div class="panel"><div class="wrap"><table>
<tr><th>Day</th><th>Date (UTC)</th><th class="num">Matches</th><th class="num">Wipe</th><th class="num">Time</th>
<th class="num">Incomplete</th><th>VOD</th></tr>
{''.join(rows)}
</table></div></div>
<div class="row">
<div class="panel"><h2>Most flown ships</h2><table><tr><th>Ship</th><th class="num">Fielded</th></tr>{top(flown)}</table></div>
<div class="panel"><h2>Most banned ships</h2><table><tr><th>Ship</th><th class="num">Bans</th></tr>{top(banned)}</table></div>
</div>"""
    return page("AT XXII Prelims", body)


def render_day(n, d, n_days):
    rows = []
    for i, m in enumerate(d["matches"], 1):
        live = ts(m["live_start_utc"])
        rows.append(
            f'<tr class="click"><td><a href="{match_file(n, i)}">Match {i}</a></td>'
            f'<td>{live.strftime("%H:%M") if live else "—"}</td>'
            f'<td>{team_chip(m, "red")}</td>'
            f'<td class="num score"><span class="red-t">{m["red_score"] if m["red_score"] is not None else "—"}</span>'
            f' – <span class="blue-t">{m["blue_score"] if m["blue_score"] is not None else "—"}</span></td>'
            f'<td>{team_chip(m, "blue")}</td><td>{end_badge(m)}</td></tr>')
    nav = ['<a href="index.html">Index</a>']
    if n > 1:
        nav.append(f'<a href="day-{n - 1}.html">‹ Day {n - 1}</a>')
    if n < n_days:
        nav.append(f'<a href="day-{n + 1}.html">Day {n + 1} ›</a>')
    body = f"""<div class="nav">{''.join(nav)}</div>
<h1>Day {n} <span class="dim">· {escape(d["date"])}</span></h1>
<div class="meta"><span><b>{len(d["matches"])}</b> matches</span>
<span>VOD <a href="https://www.twitch.tv/videos/{escape(d["vod"])}" target="_blank" rel="noopener">{escape(d["vod"])}</a></span></div>
<div class="panel"><div class="wrap"><table>
<tr><th>#</th><th>Live (UTC)</th><th>Red</th><th class="num">Score</th><th>Blue</th><th>Result</th></tr>
{''.join(rows)}
</table></div></div>"""
    return page(f"AT XXII Prelims · Day {n}", body)


def render_match(day_n, d, i, m):
    vod = d["vod"]
    live, end = ts(m["live_start_utc"]), ts(m["ended_utc"])
    length = int((end - live).total_seconds()) if live and end else None
    if length is not None and m["tidi_overtime_s"]:
        length = max(length, MATCH_S + m["tidi_overtime_s"])

    def bans_box(side):
        chips = "".join(f'<span class="chip"><span class="n">{b["ordinal"]}</span>{escape(ship_name(b))}</span>'
                        for b in m["bans"] if b["side"] == side)
        return (f'<div class="panel"><h2 class="{side}-t">{escape(m[side + "_team"] or side)} bans</h2>'
                f'{chips or "<span class=dim>none recorded</span>"}</div>')

    def roster(side):
        ships = [s for s in m["ships"] if s["side"] == side]
        rows = []
        for s in ships:
            dead = s["death_confidence"] is not None
            died = mmss(s["died_match_s"]) if s["died_match_s"] is not None else ("?" if dead else "—")
            rows.append(
                f'<tr class="{"dead" if dead else ""}"><td class="num dim">{s["slot"] + 1}</td>'
                f'<td class="strike">{escape(ship_name(s))}</td><td class="strike">{escape(s["pilot"] or "")}</td>'
                f'<td class="num">{s["points"] if s["points"] is not None else "?"}</td>'
                f'<td class="num">{vod_link(vod, s["died_stream_offset_s"], died) if dead else died}</td></tr>')
        pts = sum(s["points"] or 0 for s in ships)
        score = m[side + "_score"]
        return f"""<div class="panel {side}"><div class="phead">
<h2><span class="dim">{"Red" if side == "red" else "Blue"}</span> {team_chip(m, side)}</h2>
<span><span class="dim">score</span> <span class="score">{score if score is not None else "—"}</span>
<span class="dim">· {pts} / 200 pts</span></span></div>
<table><tr><th class="num">#</th><th>Ship</th><th>Pilot</th><th class="num">Pts</th><th class="num">Died</th></tr>
{''.join(rows)}</table></div>"""

    deaths = [s for s in m["ships"] if s["death_confidence"] is not None]
    deaths.sort(key=lambda s: (s["died_match_s"] is None, s["died_match_s"] or 0, s["side"] != "red", s["slot"]))
    kill_rows = []
    for k, s in enumerate(deaths, 1):
        tags = []
        if s["died_in_tidi"]:
            tags.append('<span class="badge warn">TiDi</span>')
        if s["death_confidence"] == "low":
            tags.append('<span class="badge warn">low confidence</span>')
        if s["died_match_s"] is None:
            when = '<span class="badge warn">time unknown</span>'
        else:
            when = vod_link(vod, s["died_stream_offset_s"], mmss(s["died_match_s"]))
        clock = mmss(s["died_clock_s"]) if s["died_clock_s"] is not None else "—"
        kill_rows.append(
            f'<tr><td class="num dim">#{k}</td><td class="num">{when}</td><td class="num dim">{clock}</td>'
            f'<td class="{s["side"]}-t">{escape(ship_name(s))}</td><td>{escape(s["pilot"] or "")}</td>'
            f'<td class="num">{s["points"] if s["points"] is not None else "?"}</td>'
            f'<td class="dim">{escape(m[s["side"] + "_team"] or "")}</td><td>{" ".join(tags)}</td></tr>')

    timed = [s for s in deaths if s["died_match_s"] is not None]
    span = max([MATCH_S, length or 0] + [s["died_match_s"] for s in timed])
    ticks = "".join(
        f'<div class="tick {s["side"]}" style="left:{100 * s["died_match_s"] / span:.2f}%" '
        f'title="{mmss(s["died_match_s"])} · {escape(ship_name(s))} ({escape(s["pilot"] or "")})"></div>'
        for s in timed)
    labels = "".join(f'<span class="lbl" style="left:{100 * t / span:.2f}%">{mmss(t)}</span>'
                     for t in range(0, int(span) + 1, 120))
    if span > MATCH_S:
        ticks += f'<div class="end" style="left:{100 * MATCH_S / span:.2f}%" title="10:00, TiDi overtime after this"></div>'
    if length is not None and length < span:
        ticks += f'<div class="end" style="left:{100 * length / span:.2f}%" title="match ended {mmss(length)}"></div>'

    n_matches = len(d["matches"])
    nav = ['<a href="index.html">Index</a>', f'<a href="day-{day_n}.html">← Day {day_n}</a>']
    if i > 1:
        nav.append(f'<a href="{match_file(day_n, i - 1)}">‹ Match {i - 1}</a>')
    if i < n_matches:
        nav.append(f'<a href="{match_file(day_n, i + 1)}">Match {i + 1} ›</a>')

    body = f"""<div class="nav">{''.join(nav)}</div>
<h1>Day {day_n}, Match {i}</h1>
<div class="meta">{team_chip(m, "red")} <span>vs</span> {team_chip(m, "blue")} {end_badge(m)}
<span>Live <b>{live.strftime("%Y-%m-%d %H:%M:%S") if live else "—"}</b> UTC</span>
<span>Length <b>{mmss(length)}</b></span>
<span>{vod_link(vod, m["live_offset_s"], "Watch on VOD ↗")}</span></div>
<div class="row">{bans_box("red")}{bans_box("blue")}</div>
<div class="row">{roster("red")}{roster("blue")}</div>
<div class="panel"><h2>Kill order <span class="dim">· {len(deaths)} kills</span></h2>
<div class="timeline">{ticks}{labels}</div>
<div class="wrap"><table>
<tr><th class="num">#</th><th class="num">Match time</th><th class="num">HUD clock</th><th>Ship</th><th>Pilot</th>
<th class="num">Pts</th><th>Team</th><th></th></tr>
{''.join(kill_rows) or '<tr><td colspan="8" class="dim">No kills recorded.</td></tr>'}
</table></div></div>"""
    return page(f"Day {day_n} Match {i} · {m['red_team']} vs {m['blue_team']}", body)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dbs", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    days = [load_day(p) for p in args.dbs]
    days = [d for d in days if d["matches"]]
    days.sort(key=lambda d: d["matches"][0]["live_start_utc"] or "")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    (out / "index.html").write_text(render_index(days))
    count = 1
    for n, d in enumerate(days, 1):
        (out / f"day-{n}.html").write_text(render_day(n, d, len(days)))
        count += 1
        for i, m in enumerate(d["matches"], 1):
            (out / match_file(n, i)).write_text(render_match(n, d, i, m))
            count += 1
    print(f"wrote {count} pages to {out}/ (open {out / 'index.html'})")


if __name__ == "__main__":
    main()
