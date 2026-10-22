# harvest.py <snapshot-id> <out-dir> [base-url]
#
# Keeps a camera's openipc.org wall archive before it ages out: the wall holds
# about a day of one camera's hourly frames, and a meter's drums barely move in
# a day, so every day not kept is glyphs that never come back.
#
# Any snapshot id of the camera will do; its archive lists the rest. Frames
# arrive over the wall's socket as the page receives them: the first 4096
# bytes XORed with the connection id, an HEVC keyframe of length-prefixed NAL
# units, its parameter sets in an hvcC record. Each is kept twice under
# <out-dir>/<camera>/: the bitstream as Annex-B (.hevc, what the camera
# encoded, nothing lost) and decoded to PNG by ffmpeg, with one line per frame
# in index.jsonl: time, id, mean luma (night frames are kept, and told apart
# later), and the camera's own fields. Frames already kept are skipped, so it
# is safe to run as often as wanted.
import asyncio
import base64
import json
import os
import struct
import subprocess
import sys

import aiohttp
import numpy as np

UA = {"User-Agent": "Mozilla/5.0 (cpu-models harvest)"}


def annexb_from_hvcc(h):
    # ISO/IEC 14496-15: 22 bytes of profile and sizes, then the arrays of
    # parameter sets (VPS, SPS, PPS), each a type byte and counted NAL units.
    out, p = b"", 23
    for _ in range(h[22]):
        p += 1
        cnt = struct.unpack(">H", h[p:p + 2])[0]
        p += 2
        for _ in range(cnt):
            ln = struct.unpack(">H", h[p:p + 2])[0]
            p += 2
            out += b"\0\0\0\1" + h[p:p + ln]
            p += ln
    return out


def annexb_from_lp(d):
    out, p = b"", 0
    while p + 4 <= len(d):
        ln = struct.unpack(">I", d[p:p + 4])[0]
        p += 4
        out += b"\0\0\0\1" + d[p:p + ln]
        p += ln
    if p != len(d):
        raise ValueError("NAL lengths do not add up")
    return out


def decode(hevc, png):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "hevc", "-i", hevc,
                    "-frames:v", "1", png], check=True)
    # The mean of the luma plane, from the PNG just written: a gate for later,
    # not a judgement now.
    y = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", png, "-f", "rawvideo",
                        "-pix_fmt", "gray", "-"], check=True, capture_output=True).stdout
    return round(float(np.frombuffer(y, np.uint8).mean()), 2)


async def get_json(s, url):
    async with s.get(url) as r:
        r.raise_for_status()
        return await r.json()


async def harvest(sid, out, base):
    wss = "ws" + base[4:] + "/api/v1/wall/socket"
    async with aiohttp.ClientSession(headers=UA | {"Origin": base}) as s:
        snap = (await get_json(s, f"{base}/api/v1/wall/snapshot/{sid}.json"))["snapshot"]
        cam = snap["camera"]
        frames = (await get_json(s, f"{base}/api/v1/wall/snapshot/{sid}/archive.json"))["frames"]
        dst = os.path.join(out, cam)
        os.makedirs(dst, exist_ok=True)
        index = os.path.join(dst, "index.jsonl")
        have = set()
        if os.path.exists(index):
            have = {json.loads(l)["id"] for l in open(index)}
        todo = [f for f in frames if f["id"] not in have]
        print(f"{cam}: {len(frames)} in the archive, {len(todo)} new")
        if not todo:
            return
        # Each frame's own page carries the grant for its full-size variant.
        meta, grants = {}, {}
        for f in todo:
            j = await get_json(s, f"{base}/api/v1/wall/snapshot/{f['id']}.json")
            meta[f["id"]], grants[f["id"]] = j["snapshot"], j["grant"]
        got = 0
        async with s.ws_connect(wss, heartbeat=20) as ws:
            async for m in ws:
                c = json.loads(m.data)
                if c["type"] == "hello":
                    key = c["connection_id"].encode()
                    for f in todo:
                        await ws.send_str(json.dumps({"type": "grant", "grant": grants[f["id"]]}))
                        await ws.send_str(json.dumps(
                            {"type": "request", "variant": "fullhd", "ids": [f["id"]]}))
                elif c["type"] == "frame":
                    fid = c["id"]
                    d = bytearray(base64.b64decode(c["frame"]))
                    for i in range(min(4096, len(d))):
                        d[i] ^= key[i % len(key)]
                    if not c.get("codec", "").startswith("hvc1"):
                        raise SystemExit(f"{fid}: codec {c.get('codec')}, only HEVC is handled")
                    sm = meta[fid]
                    stem = os.path.join(dst, f"{sm['at']}_{fid}")
                    with open(stem + ".hevc", "wb") as w:
                        w.write(annexb_from_hvcc(base64.b64decode(c["description"])))
                        w.write(annexb_from_lp(bytes(d)))
                    luma = decode(stem + ".hevc", stem + ".png")
                    rec = {"id": fid, "at": sm["at"], "luma": luma,
                           "width": c.get("width"), "height": c.get("height")}
                    rec |= {k: sm.get(k) for k in ("soc", "sensor", "firmware", "caption")}
                    with open(index, "a") as w:
                        w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    got += 1
                    if got == len(todo):
                        break
                elif c["type"] == "error":
                    raise SystemExit(f"wall refused: {c}")
        print(f"{cam}: kept {got}")


def main(sid, out, base="https://openipc.org"):
    asyncio.run(asyncio.wait_for(harvest(sid, out, base.rstrip("/")), 300))


if __name__ == "__main__":
    main(*sys.argv[1:])
