# What the drum model learnt from

- Synthetic drums rendered by `tools/meters/synth.py` with the fonts of
  `tools/meters/fonts.txt` (SIL OFL 1.1), on meter faces drawn procedurally
  or taken from daylight frames of a camera on the OpenIPC wall.
- The train splits of these open datasets, under CC BY 4.0:
  - **Water Meter Dataset**, version 3, by water-meter-yfzw4 on Roboflow Universe:
    https://universe.roboflow.com/water-meter-yfzw4/water-meter-dataset
  - **Odometer**, version 24, by meter-reader-ai on Roboflow Universe:
    https://universe.roboflow.com/meter-reader-ai/odometer-ch73o-dhlvv

`model.onnx` is the network after BatchNorm folding and static int8 (QDQ)
quantization by `tools/meters/export_drum.py`; `tools/export_pdnet.py` turns
it into the blob the engine loads.
