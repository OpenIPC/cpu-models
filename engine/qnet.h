#pragma once

/* qnet: an int8 network for the camera's CPU, run directly.
 *
 * Loads the blob tools/export_qnet.py writes from a quantized model: int8
 * activations, per-channel int8 weights, int32 accumulators, one rounding per
 * requantization, and the heads taken to float straight from their
 * accumulators. A blob is one of two kinds, and each entry point refuses the
 * other: a detector (the person detector, an NV21 frame in, qnet_detect) or
 * a classifier (a luma crop in, its outputs out, qnet_classify). */

#include <stddef.h>
#include <stdint.h>

typedef struct QNet QNet;

typedef struct {
    int x, y, w, h;
    int score; /* 0..100 */
} QBox;

QNet *qnet_open(const char *path);
void qnet_close(QNet *n);
int qnet_width(const QNet *n);
int qnet_height(const QNet *n);

/* Detect people on one NV21 frame of exactly qnet_width() x qnet_height():
 * boxes in input pixels, scores 0..100, at most `max`, best first, after
 * suppression. Returns the count, or -1. */
int qnet_detect(
    QNet *n, const uint8_t *y, int ystride, const uint8_t *vu, int vustride,
    QBox *out, int max, int score_min);

/* The raw head outputs of the last qnet_detect(), for checking against the
 * reference: level 0..2, cls (1 channel) or reg (4), HWC float. */
const float *qnet_head(const QNet *n, int level, int reg, int *h, int *w);

/* Classify one luma crop of exactly qnet_width() x qnet_height(): the
 * network's outputs (a classifier's logits), at most `max` of them, in
 * order. Returns how many, or -1. */
int qnet_classify(QNet *n, const uint8_t *y, int ystride, float *out, int max);

/* The int8 input the last run was given, HWC: for checking against the
 * reference. */
const int8_t *qnet_input(const QNet *n);
