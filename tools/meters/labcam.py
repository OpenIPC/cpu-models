# labcam.py <name> <camera-url> <out-dir> [jump-host]
#
# Keeps one frame from a majestic camera that is watching a meter: the luma
# plane of /image.yuv420, which is what a reader on the camera will be given,
# not a JPEG of it. <camera-url> carries the credentials
# (http://user:pass@host); [jump-host] is an ssh destination to fetch through
# when this machine cannot route to the camera itself, otherwise the
# environment's proxy, if any, is used.
#
# A <camera-url> ending in .jpg is fetched as it is and its luma kept
# instead, for a camera whose raw frame is not to be asked for.
#
# Under <out-dir>/<name>/: <unix-time>.png (the Y plane, lossless) and a line
# in index.jsonl with the frame's size, mean luma and the plane's range: not
# every camera keeps luma to 16..235.
import http.client
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.parse
import urllib.request

import cv2
import numpy as np

SIZES = {1382400: (1280, 720), 3110400: (1920, 1080), 2764800: (2560, 720),
         4147200: (2560, 1080), 5529600: (2560, 1440), 7372800: (2560, 1920)}


def fetch(url, jump):
    if jump:
        cmd = "wget -q -T 20 -O - " + shlex.quote(url)
        # The URL carries the camera's credentials, so the jump host must be
        # the one known_hosts says it is: a first connection is made by hand.
        return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "LogLevel=ERROR",
                               "-o", "StrictHostKeyChecking=yes", jump, cmd],
                              check=True, capture_output=True, timeout=60).stdout
    pw = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    p = urllib.parse.urlsplit(url)
    if p.username:
        pw.add_password(None, f"{p.scheme}://{p.hostname}", p.username, p.password)
        url = p._replace(netloc=p.hostname + (f":{p.port}" if p.port else "")).geturl()
    op = urllib.request.build_opener(urllib.request.ProxyHandler(),
                                     urllib.request.HTTPBasicAuthHandler(pw))
    return op.open(url, timeout=30).read()


def fetch_retry(url, jump, tries=3):
    # A chunked answer through a proxy is sometimes cut short.
    for i in range(tries):
        try:
            return fetch(url, jump)
        except (OSError, subprocess.SubprocessError, http.client.HTTPException):
            if i == tries - 1:
                raise
            time.sleep(3)


def main(name, base, out, jump=None):
    at = int(time.time())
    if base.endswith(".jpg"):
        raw = fetch_retry(base, jump)
        bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        y = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[..., 0])
        h, w = y.shape
    else:
        raw = fetch_retry(base.rstrip("/") + "/image.yuv420", jump)
        if len(raw) not in SIZES:
            raise SystemExit(f"{name}: {len(raw)} bytes is no 4:2:0 frame size known here")
        w, h = SIZES[len(raw)]
        y = np.frombuffer(raw, np.uint8)[:w * h].reshape(h, w)
    dst = os.path.join(out, name)
    os.makedirs(dst, exist_ok=True)
    cv2.imwrite(os.path.join(dst, f"{at}.png"), y)
    rec = {"at": at, "width": w, "height": h, "luma": round(float(y.mean()), 2),
           "source": "jpeg" if base.endswith(".jpg") else "yuv420",
           "y_min": int(y.min()), "y_max": int(y.max())}
    with open(os.path.join(dst, "index.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(name, rec)


if __name__ == "__main__":
    main(*sys.argv[1:])
