#!/usr/bin/env bash
# Stream Twitch VODs through the tracker in parallel, one DB + HUD cache each.
# usage: YT_DLP=/path/to/yt-dlp tools/run_vods.sh <outdir> <vod_id>...
set -u
out=$1; shift
ytdlp=${YT_DLP:-yt-dlp}
root=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$out"
for id in "$@"; do
  (
    meta=$("$ytdlp" --no-update -f "1080p60/1080p/best" --print "%(timestamp)s" -g "https://www.twitch.tv/videos/$id" 2>/dev/null)
    ts=$(echo "$meta" | head -1)
    url=$(echo "$meta" | tail -1)
    start=$(date -u -d "@$ts" +%Y-%m-%dT%H:%M:%S+00:00)
    "$root/env/bin/stream-tracker" replay "$url" --start-utc "$start" --channel fenriscreations \
      --db "$out/$id.sqlite" --save-hud "$out/$id" > "$out/$id.log" 2>&1
    echo "$id done: $?"
  ) &
done
wait
