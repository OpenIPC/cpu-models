#!/bin/bash
# build.sh [out-dir]
#
# The whole build, the same here and in .github/workflows/release.yml:
#   1. fetch COCO val2017 and its annotations, hash-checked (coco.txt)
#   2. export the int8 model to the blob majestic's engine loads (person.qnet)
#   3. build the engine's tools for this host, and for ARMv7 with NEON (the
#      camera's path), run the ARM ones under qemu-arm
#   4. check both against the reference arithmetic on 40 crops, bit for bit:
#      nothing below is measured on an engine that does not compute the model
#   5. measure the presets on the labelled crops (points.py) and write
#      person.json and points.md
#   6. the drum model the same way: export drum.qnet, run it on the host and
#      on ARMv7 NEON under qemu over crops cut from the same COCO images
#      (drum_crops.py), check both against the reference bit for bit, input
#      and logits, and require the two to agree byte for byte
#   7. SHA256SUMS of everything published
#
# Needs Python 3.11 with requirements.txt, gcc, gcc-arm-linux-gnueabihf and
# qemu-user-static.
set -euo pipefail
src=$(cd "$(dirname "$0")/.." && pwd)
out=$(readlink -m "${1:-$src/out}")
cache=$src/.cache
mkdir -p "$out" "$cache"

bash "$src/scripts/fetch.sh" "$src/coco.txt" "$cache"
# Extracted fresh every build, from the archives just hash-checked: files left
# in a reused output directory are not what was pinned.
coco=$out/coco
rm -rf "$coco" "$out/eval"
mkdir -p "$coco"
unzip -qo -d "$coco" "$cache/val2017.zip"
unzip -qo -d "$coco" "$cache/annotations_trainval2017.zip" annotations/instances_val2017.json

python3 "$src/tools/export_qnet.py" "$src/models/person/model.onnx" "$out/person.qnet"

cflags=(-O2 -std=c11 -Wall -Wextra -I"$src/engine")
gcc "${cflags[@]}" -o "$out/dets" "$src/tools/dets.c" "$src/engine/qnet.c" -lm
gcc "${cflags[@]}" -o "$out/dump_heads" "$src/tools/dump_heads.c" "$src/engine/qnet.c" -lm
# _FILE_OFFSET_BITS: a 32-bit readdir cannot return the 64-bit directory
# offsets some host filesystems hand out, and finds nothing.
arm=(arm-linux-gnueabihf-gcc "${cflags[@]}" -D_FILE_OFFSET_BITS=64 -static -march=armv7-a -mfpu=neon -mfloat-abi=hard)
"${arm[@]}" -o "$out/dump_heads.arm" "$src/tools/dump_heads.c" "$src/engine/qnet.c" -lm
"${arm[@]}" -o "$out/dets.arm" "$src/tools/dets.c" "$src/engine/qnet.c" -lm

python3 "$src/tools/crops.py" "$coco" "$out/eval"
mapfile -t sample < <(ls "$out"/eval/crops/*.nv21 | sort | awk 'NR % 75 == 1' | head -40)
for run in "$out/dump_heads" "qemu-arm-static $out/dump_heads.arm"; do
	rm -f "$out"/eval/crops/*.heads
	$run "$out/person.qnet" "${sample[@]}"
	echo "== exactness: ${run%% *}"
	python3 "$src/tools/check_exact.py" "$src/models/person/model.onnx" "${sample[@]}"
done
rm -f "$out"/eval/crops/*.heads

"$out/dets" "$out/person.qnet" "$out/eval/crops" > "$out/dets.txt"
# The camera's path must find the same boxes, not just the same heads.
qemu-arm-static "$out/dets.arm" "$out/person.qnet" "$out/eval/crops" > "$out/dets.arm.txt"
cmp "$out/dets.txt" "$out/dets.arm.txt" || { echo "ARM and host detections differ"; exit 1; }

python3 "$src/tools/points.py" "$out/eval/labels.json" "$out/dets.txt" \
	"$out/person.json" "$out/points.md"

# The drum model: a classifier, so its whole output is checked, every crop.
python3 "$src/tools/export_qnet.py" "$src/models/drum/model.onnx" "$out/drum.qnet"
gcc "${cflags[@]}" -o "$out/classify" "$src/tools/meters/classify.c" "$src/engine/qnet.c" -lm
"${arm[@]}" -o "$out/classify.arm" "$src/tools/meters/classify.c" "$src/engine/qnet.c" -lm
python3 "$src/tools/meters/drum_crops.py" "$coco" "$out/drum_crops.raw" 400
for target in host arm; do
	run=("$out/classify")
	[ "$target" = arm ] && run=(qemu-arm-static "$out/classify.arm")
	rm -f "$out/drum_crops.raw.out"
	"${run[@]}" "$out/drum.qnet" "$out/drum_crops.raw"
	echo "== drum exactness: $target"
	python3 "$src/tools/meters/check_drum.py" "$src/models/drum/model.onnx" "$out/drum_crops.raw" 0
	mv "$out/drum_crops.raw.out" "$out/drum_crops.$target.out"
done
cmp "$out/drum_crops.host.out" "$out/drum_crops.arm.out" ||
	{ echo "ARM and host drum outputs differ"; exit 1; }

cd "$out" && sha256sum person.qnet person.json drum.qnet > SHA256SUMS && cat SHA256SUMS
