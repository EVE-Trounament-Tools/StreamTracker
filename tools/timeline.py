"""Print a match's event timeline. usage: python tools/timeline.py db.sqlite match_id"""
import sqlite3
import sys

conn = sqlite3.connect(sys.argv[1])
mid = int(sys.argv[2])
print(conn.execute("SELECT * FROM matches WHERE id=?", (mid,)).fetchone())
for row in conn.execute("SELECT stream_offset_s, kind, payload_json FROM events WHERE match_id=? ORDER BY id", (mid,)):
    print(*row)
