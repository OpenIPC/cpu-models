# fetch_open.py <datasets.txt> <out-dir>
#
# Fetches every dataset of datasets.txt into <out-dir>/<workspace>__<project>/
# and writes SOURCES.md there: what came from where, at which version, under
# which licence and with what attribution, for the release to carry. A
# dataset already fetched (its directory holds data.yaml) is left alone.
#
# The licence a listing states is checked again against what the source
# reports now; a dataset whose licence has changed is refused.
import io
import json
import os
import sys
import urllib.request
import zipfile

ALLOWED = {"CC BY 4.0": "CC-BY-4.0", "CC0 1.0": "CC0-1.0", "Public Domain": "CC0-1.0", "MIT": "MIT"}


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "cpu-models"}),
                                timeout=600) as r:
        return r.read()


def roboflow(key, proj, ver, fmt, want, dst):
    meta = json.loads(get(f"https://api.roboflow.com/{proj}?api_key={key}"))
    p = meta["project"]
    lic = ALLOWED.get(p.get("license"))
    if lic != want:
        raise SystemExit(f"{proj}: licence is now {p.get('license')!r}, listed as {want}")
    if not os.path.exists(os.path.join(dst, "data.yaml")):
        exp = json.loads(get(f"https://api.roboflow.com/{proj}/{ver}/{fmt}?api_key={key}"))
        link = exp["export"]["link"]
        os.makedirs(dst, exist_ok=True)
        zipfile.ZipFile(io.BytesIO(get(link))).extractall(dst)
    ws, name = proj.split("/")
    return (f"- **{p.get('name', name)}** — https://universe.roboflow.com/{proj}, version {ver}, "
            f"{p.get('license')}, by {ws} on Roboflow Universe.")


def main(listing, out):
    key = os.environ.get("ROBOFLOW_API_KEY")
    lines = []
    for row in open(listing):
        if not row.strip() or row.startswith("#"):
            continue
        src, proj, ver, fmt, lic = row.split()[:5]
        dst = os.path.join(out, proj.replace("/", "__"))
        if src == "roboflow":
            if not key:
                raise SystemExit("ROBOFLOW_API_KEY is not set")
            lines.append(roboflow(key, proj, ver, fmt, lic, dst))
        else:
            raise SystemExit(f"unknown source {src}")
        n = sum(len(fs) for _, _, fs in os.walk(dst))
        print(f"{proj} v{ver}: {n} files", flush=True)
    with open(os.path.join(out, "SOURCES.md"), "w") as f:
        f.write("# Open datasets used\n\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
