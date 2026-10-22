/* classify <blob> <crops.raw>: every crop in <crops.raw> (pdnet_width() x
 * pdnet_height() luma bytes each, back to back) through pdnet_classify(),
 * written to <crops.raw>.out as, per crop, the int8 input the engine made of
 * it and its float outputs: for comparison with the reference
 * (check_drum.py) and for scoring. */
#include <stdio.h>
#include <stdlib.h>

#include "pdnet.h"

#define MAX_OUT 64

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: classify <blob> <crops.raw>\n");
        return 2;
    }
    PdNet *n = pdnet_open(argv[1]);
    if (!n) {
        fprintf(stderr, "cannot open %s\n", argv[1]);
        return 1;
    }
    const int w = pdnet_width(n), h = pdnet_height(n);
    unsigned char *crop = malloc((size_t)w * h);
    FILE *in = fopen(argv[2], "rb");
    char name[4096];
    snprintf(name, sizeof name, "%s.out", argv[2]);
    FILE *out = fopen(name, "wb");
    if (!crop || !in || !out) {
        fprintf(stderr, "cannot read %s or write %s\n", argv[2], name);
        return 1;
    }
    float logits[MAX_OUT];
    long count = 0;
    while (fread(crop, 1, (size_t)w * h, in) == (size_t)w * h) {
        const int k = pdnet_classify(n, crop, w, logits, MAX_OUT);
        if (k <= 0) {
            fprintf(stderr, "classify failed on crop %ld\n", count);
            return 1;
        }
        fwrite(pdnet_input(n), 1, (size_t)w * h, out);
        fwrite(logits, sizeof(float), (size_t)k, out);
        count++;
    }
    fclose(in);
    fclose(out);
    fprintf(stderr, "classify: %ld crops\n", count);
    pdnet_close(n);
    free(crop);
    return 0;
}
