#pragma once

/* pdnet: the CPU person detector's int8 network, run directly.
 *
 * Loads the blob export_pdnet.py writes from the quantized model and runs it
 * on an NV21 frame of the network's input size: int8 activations, per-channel
 * int8 weights, int32 accumulators, one rounding per requantization, and the
 * detection heads taken to float straight from their accumulators. */

#include <stddef.h>
#include <stdint.h>

typedef struct PdNet PdNet;

typedef struct {
    int x, y, w, h;
    int score; /* 0..100 */
} PdBox;

PdNet *pdnet_open(const char *path);
void pdnet_close(PdNet *n);
int pdnet_width(const PdNet *n);
int pdnet_height(const PdNet *n);

/* Detect people on one NV21 frame of exactly pdnet_width() x pdnet_height():
 * boxes in input pixels, scores 0..100, at most `max`, best first, after
 * suppression. Returns the count, or -1. */
int pdnet_detect(
    PdNet *n, const uint8_t *y, int ystride, const uint8_t *vu, int vustride,
    PdBox *out, int max, int score_min);

/* The raw head outputs of the last pdnet_detect(), for checking against the
 * reference: level 0..2, cls (1 channel) or reg (4), HWC float. */
const float *pdnet_head(const PdNet *n, int level, int reg, int *h, int *w);
