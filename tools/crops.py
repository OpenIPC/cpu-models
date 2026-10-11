# crops.py <coco-dir> <out-dir> [n_person] [n_background]
#
# The labelled set the person model's presets are measured on, in the
# geometry the camera hands the model: 320x192 crops around a person at the
# scale the detector was trained for (the person ~47% of the crop's height, as
# majestic places a crop), off-centre the way a motion region is, and crops of
# person-free COCO val2017 images for the false alarms. Each crop is NV21,
# BT.709 limited range, as the VPSS delivers it; labels.json lists the people
# inside each in crop pixels, marking crowds and people less than half inside
# the crop to be ignored. Seeded: the same inputs give the same crops.
import json
import os
import random
import sys

import cv2
import numpy as np

W, H = 320, 192
FILL = 0.47


def nv21(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 16 + (0.2126 * r + 0.7152 * g + 0.0722 * b) * 219 / 255
    cb = 128 + (-0.1146 * r - 0.3854 * g + 0.5 * b) * 224 / 255
    cr = 128 + (0.5 * r - 0.4542 * g - 0.0458 * b) * 224 / 255
    sub = lambda c: c.reshape(H // 2, 2, W // 2, 2).mean((1, 3))
    vu = np.stack([sub(cr), sub(cb)], -1).reshape(H // 2, W)
    return (np.clip(y, 0, 255).round().astype(np.uint8),
            np.clip(vu, 0, 255).round().astype(np.uint8))


def main(coco, out, n_person=2000, n_bg=1000):
    n_person, n_bg = int(n_person), int(n_bg)
    rnd = random.Random(20261011)
    ann = json.load(open(f"{coco}/annotations/instances_val2017.json"))
    imgs = {i["id"]: i for i in ann["images"]}
    people = {}
    for a in ann["annotations"]:
        if a["category_id"] == 1:
            people.setdefault(a["image_id"], []).append(a)
    # A directory holding crops from another run would mix them in.
    if os.path.isdir(f"{out}/crops") and os.listdir(f"{out}/crops"):
        sys.exit(f"crops.py: {out}/crops is not empty")
    os.makedirs(f"{out}/crops", exist_ok=True)
    labels = {}

    def emit(name, img, x0, y0, cw, ch, anns):
        x0, y0, cw, ch = (int(round(v)) for v in (x0, y0, cw, ch))
        crop = img[y0:y0 + ch, x0:x0 + cw]
        rgb = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (W, H),
                         interpolation=cv2.INTER_AREA).astype(np.float32)
        y, vu = nv21(rgb)
        with open(f"{out}/crops/{name}.nv21", "wb") as f:
            f.write(y.tobytes() + vu.tobytes())
        sx, sy = W / cw, H / ch
        boxes = []
        for a in anns:
            bx, by, bw, bh = a["bbox"]
            ix0, iy0 = max(bx, x0), max(by, y0)
            ix1, iy1 = min(bx + bw, x0 + cw), min(by + bh, y0 + ch)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            visible = (ix1 - ix0) * (iy1 - iy0) / max(bw * bh, 1e-9)
            boxes.append({"box": [(ix0 - x0) * sx, (iy0 - y0) * sy,
                                  (ix1 - ix0) * sx, (iy1 - iy0) * sy],
                          "ignore": bool(a["iscrowd"]) or visible < 0.5})
        labels[name] = boxes

    cands = [(iid, a) for iid, anns in sorted(people.items()) for a in anns
             if not a["iscrowd"] and a["bbox"][3] >= 24]
    rnd.shuffle(cands)
    made = 0
    for iid, a in cands:
        if made >= n_person:
            break
        im = imgs[iid]
        bx, by, bw, bh = a["bbox"]
        ch = bh / FILL
        cw = ch * W / H
        if cw > im["width"] or ch > im["height"]:
            continue
        cx = bx + bw / 2 + rnd.uniform(-cw / 3, cw / 3)
        cy = by + bh / 2 + rnd.uniform(-ch / 6, ch / 6)
        x0 = min(max(cx - cw / 2, 0), im["width"] - cw)
        y0 = min(max(cy - ch / 2, 0), im["height"] - ch)
        img = cv2.imread(f"{coco}/val2017/{im['file_name']}")
        emit(f"p{made:05d}_{iid}", img, x0, y0, cw, ch, people[iid])
        made += 1
    empty = sorted(i for i in imgs if i not in people)
    rnd.shuffle(empty)
    for k, iid in enumerate(empty[:n_bg]):
        im = imgs[iid]
        ch = rnd.uniform(0.35, 1.0) * min(im["height"], im["width"] * H / W)
        cw = ch * W / H
        x0 = rnd.uniform(0, im["width"] - cw)
        y0 = rnd.uniform(0, im["height"] - ch)
        img = cv2.imread(f"{coco}/val2017/{im['file_name']}")
        emit(f"b{k:05d}_{iid}", img, x0, y0, cw, ch, [])
    with open(f"{out}/labels.json", "w") as f:
        json.dump(labels, f, sort_keys=True)
    print(f"crops.py: {made} person crops, {min(n_bg, len(empty))} person-free")


if __name__ == "__main__":
    main(*sys.argv[1:])
