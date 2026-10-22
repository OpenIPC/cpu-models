# fetch_open.py <datasets.txt> <out-dir> [--verify]
#
# Fetches every dataset of datasets.txt into <out-dir>/<workspace>__<project>/
# and writes SOURCES.md there: what came from where, at which version, under
# which licence and with what attribution, for the release to carry.
#
# Every dataset is checked against the contents digest datasets.txt pins,
# whether just downloaded or already there: a re-generated export, or a
# directory changed since, would otherwise change what is trained and
# measured on without a word. A mismatch is refused, never repaired: remove
# the directory and fetch again, or re-pin on purpose. --verify checks the
# directories alone, with no network and no key.
#
# The licence a listing states is checked again against what the source
# reports now; a dataset whose licence has changed is refused.
import hashlib
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


def tree_sha256(d):
    """One digest of a directory's contents: each file's path relative to it
    and its sha256, in path order."""
    h = hashlib.sha256()
    for root, dirs, files in os.walk(d):
        dirs.sort()
        for f in sorted(files):
            p = os.path.join(root, f)
            with open(p, "rb") as fh:
                h.update(os.path.relpath(p, d).encode() + b"\0" +
                         hashlib.sha256(fh.read()).hexdigest().encode() + b"\n")
    return h.hexdigest()


def verify(proj, dst, sha):
    got = tree_sha256(dst)
    if got != sha:
        raise SystemExit(f"{proj}: {dst} has contents {got}, datasets.txt pins {sha}; "
                         "remove it and fetch again, or re-pin on purpose")


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


def main(listing, out, *flags):
    only_verify = "--verify" in flags
    key = os.environ.get("ROBOFLOW_API_KEY")
    lines = []
    for row in open(listing):
        if not row.strip() or row.startswith("#"):
            continue
        src, proj, ver, fmt, lic, sha = row.split()[:6]
        dst = os.path.join(out, proj.replace("/", "__"))
        if only_verify:
            verify(proj, dst, sha)
            print(f"{proj} v{ver}: contents as pinned", flush=True)
            continue
        if src == "roboflow":
            if not key:
                raise SystemExit("ROBOFLOW_API_KEY is not set")
            lines.append(roboflow(key, proj, ver, fmt, lic, dst))
        else:
            raise SystemExit(f"unknown source {src}")
        verify(proj, dst, sha)
        n = sum(len(fs) for _, _, fs in os.walk(dst))
        print(f"{proj} v{ver}: {n} files", flush=True)
    if only_verify:
        return
    with open(os.path.join(out, "SOURCES.md"), "w") as f:
        f.write("# Open datasets used\n\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
