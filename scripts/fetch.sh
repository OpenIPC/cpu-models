#!/bin/bash
# fetch.sh <list> <cache-dir>: every "file sha256 url" line of <list>, into
# <cache-dir>, hash-checked. A cached file whose hash does not match is
# fetched again, and a download lands in the cache only once its hash
# matches, so an interrupted one can never be mistaken for the file.
set -euo pipefail
mkdir -p "$2"
grep -v '^#' "$1" | while read -r file sha url; do
	[ -n "$file" ] || continue
	dst="$2/$file"
	if [ -f "$dst" ] && echo "$sha  $dst" | sha256sum -c --status 2>/dev/null; then
		continue
	fi
	tmp="$dst.part"
	if ! curl -sfL --retry 3 -o "$tmp" "$url" ||
	   ! echo "$sha  $tmp" | sha256sum -c --status; then
		rm -f "$tmp"
		echo "fetch.sh: $file from $url: download failed or sha256 mismatch" >&2
		exit 1
	fi
	mv "$tmp" "$dst"
done
