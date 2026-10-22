# cpu-models

Detection models that OpenIPC cameras run on their CPU, with majestic's own
int8 engine, and the release that publishes them. The first is the person
detector majestic's object detection uses on cameras without an NPU.

## The person detector

`models/person/model.onnx` is a statically quantized (QDQ) int8 network:
MobileNetV1-0.25 with an FPN-lite head, 66.7k parameters, a 320x192 YUV
input, and boxes at strides 8, 16 and 32. It was trained for majestic on COCO
2017 person boxes and fine-tuned on hard negatives mined from a lab camera's
own crops; while those mined crops were sorted, an Ultralytics YOLO model was
used to find which ones contained a person.

The camera hands it a crop of the motion region at the scale it was trained
for, in NV21; it reports people with a score from 0 to 100.

## Releases

Published only by the `release` workflow (Actions -> release -> Run workflow,
with a new tag), never by hand. A release carries:

- `person.qnet`: the model as majestic's engine loads it, exported from the
  ONNX by `tools/export_qnet.py`. Releases up to and including the first one
  with a classifier engine shipped `person.pdnet` instead, the older blob
  format (`MAJPDNT1`); the engine reads only `MAJQNET1` now, so majestic
  moves to `.qnet` in the same change that takes this engine;
- `person.json`: the score thresholds behind majestic's three presets, for this
  model, installed beside it:
  - `balanced`: the F1-optimal threshold;
  - `fewerFalseAlarms`: the F0.5 optimum;
  - `catchEverything`: the highest threshold keeping recall at 0.90, else the
    F2 optimum;
- `points.md`: what each preset achieves; and `SHA256SUMS`.

## Building

```sh
pip install -r requirements.txt               # Python 3.11
sudo apt-get install gcc-arm-linux-gnueabihf qemu-user-static
bash scripts/build.sh [out]
```

`scripts/build.sh` is the whole build, the same here and in the workflow. It:

1. fetches COCO val2017 and its annotations, hash-checked (`coco.txt`);
2. exports the blob;
3. builds the engine (`engine/qnet.c`) for the host and for ARMv7 with NEON,
   the camera's path;
4. checks both against `tools/qref.py`, the model's arithmetic written out in
   numpy. On 40 crops, every value of all six head maps must be identical,
   and the ARM build, run under qemu-arm, must find exactly the host's boxes
   on all 3000 crops;
5. measures the presets: `tools/crops.py` cuts 3000 labelled 320x192 crops
   from COCO val2017 in the camera's geometry, 2000 around a person and 1000
   with nobody in them, and `tools/points.py` scores the engine's detections
   on them.

A build of one commit is byte-identical anywhere.

`engine/` is the engine majestic carries; a change to it lands here and there
together.

## License

MIT.
