#ifndef GIF_H
#define GIF_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <math.h>
#include <float.h>

// Default color palette
static const uint8_t DEFAULT_PALETTE[16 * 3] = {
    0x00, 0x00, 0x00,    // Black
    0xFF, 0x00, 0x00,    // Red
    0x00, 0xFF, 0x00,    // Green
    0x00, 0x00, 0xFF,    // Blue
    0xFF, 0xFF, 0x00,    // Yellow
    0xFF, 0x00, 0xFF,    // Magenta
    0x00, 0xFF, 0xFF,    // Cyan
    0xFF, 0xFF, 0xFF,    // White
    0x80, 0x80, 0x80,    // Gray
    0x80, 0x00, 0x00,    // Dark Red
    0x00, 0x80, 0x00,    // Dark Green
    0x00, 0x00, 0x80,    // Dark Blue
    0x80, 0x80, 0x00,    // Dark Yellow
    0x80, 0x00, 0x80,    // Dark Magenta
    0x00, 0x80, 0x80,    // Dark Cyan
    0xC0, 0xC0, 0xC0     // Light Gray
};

typedef struct {
    uint16_t w, h;
    int depth;
    int bgindex;
    int fd;
    int offset;
    int nframes;
    uint8_t *frame, *back;
    uint32_t partial;
    uint8_t buffer[0xFF];
    uint8_t palette[16 * 3];
} ge_GIF;

typedef struct Node {
    uint16_t key;
    struct Node *children[];
} Node;

// Error handling write functions
static void safe_write(int fd, const void *buf, size_t count) {
    if (write(fd, buf, count) != (ssize_t)count) {
        perror("Write error");
        exit(EXIT_FAILURE);
    }
}

static void write_num(int fd, uint16_t n) {
    uint8_t bytes[2] = {n & 0xFF, n >> 8};
    safe_write(fd, bytes, 2);
}

// LZW compression helpers
static Node *new_node(uint16_t key, int degree) {
    Node *node = calloc(1, sizeof(*node) + degree * sizeof(Node *));
    if (node) node->key = key;
    return node;
}

static Node *new_trie(int degree, int *nkeys) {
    Node *root = new_node(0, degree);
    *nkeys = 0;
    for (; *nkeys < degree; (*nkeys)++)
        root->children[*nkeys] = new_node(*nkeys, degree);
    *nkeys += 2;
    return root;
}

static void del_trie(Node *root, int degree) {
    if (!root) return;
    for (int i = 0; i < degree; i++)
        del_trie(root->children[i], degree);
    free(root);
}

// GIF creation and frame handling
ge_GIF *ge_new_gif(const char *fname, uint16_t width, uint16_t height, int depth, int bgindex, int loop) {
    ge_GIF *gif = calloc(1, sizeof(*gif) + (bgindex < 0 ? 2 : 1) * width * height);
    if (!gif) return NULL;

    gif->w = width;
    gif->h = height;
    gif->bgindex = bgindex;
    gif->frame = (uint8_t *)&gif[1];
    gif->back = &gif->frame[width * height];
    gif->depth = abs(depth) > 1 ? abs(depth) : 2;

    gif->fd = creat(fname, 0666);
    if (gif->fd == -1) {
        free(gif);
        return NULL;
    }

    memcpy(gif->palette, DEFAULT_PALETTE, sizeof(gif->palette));

    // Write GIF header and global color table
    safe_write(gif->fd, "GIF89a", 6);
    write_num(gif->fd, width);
    write_num(gif->fd, height);
    safe_write(gif->fd, (uint8_t[]){0xF0 | (gif->depth - 1), (uint8_t)bgindex, 0x00}, 3);
    safe_write(gif->fd, gif->palette, 3 << gif->depth);

    // Write loop information if needed
    if (loop >= 0 && loop <= 0xFFFF) {
        safe_write(gif->fd, "!\xFF\x0BNETSCAPE2.0\x03\x01", 16);
        write_num(gif->fd, loop);
        safe_write(gif->fd, "\0", 1);
    }

    return gif;
}

// Frame encoding helpers
static void put_key(ge_GIF *gif, uint16_t key, int key_size) {
    int byte_offset = gif->offset / 8;
    int bit_offset = gif->offset % 8;
    gif->partial |= ((uint32_t)key) << bit_offset;
    int bits_to_write = bit_offset + key_size;

    while (bits_to_write >= 8) {
        gif->buffer[byte_offset++] = gif->partial & 0xFF;
        if (byte_offset == 0xFF) {
            safe_write(gif->fd, "\xFF", 1);
            safe_write(gif->fd, gif->buffer, 0xFF);
            byte_offset = 0;
        }
        gif->partial >>= 8;
        bits_to_write -= 8;
    }
    gif->offset = (gif->offset + key_size) % (0xFF * 8);
}

static void end_key(ge_GIF *gif) {
    int byte_offset = gif->offset / 8;
    if (gif->offset % 8)
        gif->buffer[byte_offset++] = gif->partial & 0xFF;
    if (byte_offset) {
        safe_write(gif->fd, (uint8_t[]){byte_offset}, 1);
        safe_write(gif->fd, gif->buffer, byte_offset);
    }
    safe_write(gif->fd, "\0", 1);
    gif->offset = gif->partial = 0;
}

static void put_image(ge_GIF *gif, uint16_t w, uint16_t h, uint16_t x, uint16_t y) {
    int nkeys, key_size;
    Node *node, *root;
    int degree = 1 << gif->depth;

    safe_write(gif->fd, ",", 1);
    write_num(gif->fd, x);
    write_num(gif->fd, y);
    write_num(gif->fd, w);
    write_num(gif->fd, h);
    safe_write(gif->fd, (uint8_t[]){0x00, gif->depth}, 2);

    root = node = new_trie(degree, &nkeys);
    key_size = gif->depth + 1;
    put_key(gif, degree, key_size);

    // LZW compression
    for (int i = y; i < y + h; i++) {
        for (int j = x; j < x + w; j++) {
            uint8_t pixel = gif->frame[i * gif->w + j] & (degree - 1);
            Node *child = node->children[pixel];

            if (child) {
                node = child;
            } else {
                put_key(gif, node->key, key_size);
                if (nkeys < 0x1000) {
                    if (nkeys == (1 << key_size)) key_size++;
                    node->children[pixel] = new_node(nkeys++, degree);
                } else {
                    put_key(gif, degree, key_size);
                    del_trie(root, degree);
                    root = node = new_trie(degree, &nkeys);
                    key_size = gif->depth + 1;
                }
                node = root->children[pixel];
            }
        }
    }

    put_key(gif, node->key, key_size);
    put_key(gif, degree + 1, key_size);
    end_key(gif);
    del_trie(root, degree);
}

static int get_bbox(ge_GIF *gif, uint16_t *w, uint16_t *h, uint16_t *x, uint16_t *y) {
    int left = gif->w, right = 0, top = gif->h, bottom = 0;

    for (int i = 0; i < gif->h; i++) {
        for (int j = 0; j < gif->w; j++) {
            int k = i * gif->w + j;
            uint8_t back = gif->bgindex >= 0 ? gif->bgindex : gif->back[k];
            if (gif->frame[k] != back) {
                if (j < left)   left   = j;
                if (j > right)  right  = j;
                if (i < top)    top    = i;
                if (i > bottom) bottom = i;
            }
        }
    }

    if (left != gif->w && top != gif->h) {
        *x = left; *y = top;
        *w = right - left + 1;
        *h = bottom - top + 1;
        return 1;
    }
    return 0;
}

static void floyd_steinberg_dithering(uint8_t *input, ge_GIF *gif) {
    double (*error_buffer)[3] = calloc(gif->w * gif->h, sizeof(*error_buffer));
    
    // Initialize error buffer
    for (int i = 0; i < gif->h * gif->w; i++)
        for (int c = 0; c < 3; c++)
            error_buffer[i][c] = input[i * 3 + c];

    // Apply dithering
    for (int y = 0; y < gif->h; y++) {
        for (int x = 0; x < gif->w; x++) {
            int idx = y * gif->w + x;
            uint8_t pixel[3];
            for (int c = 0; c < 3; c++)
                pixel[c] = (uint8_t)fmax(0, fmin(255, round(error_buffer[idx][c])));

            // Find nearest color
            uint8_t best_color = 0;
            double min_dist = DBL_MAX;
            for (int i = 0; i < 8; i++) {
                double dist = 0;
                for (int c = 0; c < 3; c++) {
                    double diff = pixel[c] - gif->palette[i * 3 + c];
                    dist += diff * diff;
                }
                if (dist < min_dist) {
                    min_dist = dist;
                    best_color = i;
                }
            }

            gif->frame[idx] = best_color;

            // Distribute error
            double error[3];
            for (int c = 0; c < 3; c++)
                error[c] = error_buffer[idx][c] - gif->palette[best_color * 3 + c];

            const struct { int x, y; float w; } pattern[] = {
                {1, 0, 7.0/16}, {-1, 1, 3.0/16}, {0, 1, 5.0/16}, {1, 1, 1.0/16}
            };

            for (int i = 0; i < 4; i++) {
                int nx = x + pattern[i].x, ny = y + pattern[i].y;
                if (nx >= 0 && nx < gif->w && ny < gif->h) {
                    int nidx = ny * gif->w + nx;
                    for (int c = 0; c < 3; c++)
                        error_buffer[nidx][c] += error[c] * pattern[i].w;
                }
            }
        }
    }

    free(error_buffer);
}

void ge_add_frame(ge_GIF *gif, uint8_t *input, uint16_t delay) {
    uint16_t w, h, x, y;

    floyd_steinberg_dithering(input, gif);

    if (delay || (gif->bgindex >= 0)) {
        safe_write(gif->fd, (uint8_t[]){
            '!', 0xF9, 0x04, ((gif->bgindex >= 0 ? 2 : 1) << 2) + 1
        }, 4);
        write_num(gif->fd, delay);
        safe_write(gif->fd, (uint8_t[]){(uint8_t)gif->bgindex, 0x00}, 2);
    }

    if (gif->nframes == 0) {
        w = gif->w; h = gif->h; x = y = 0;
    } else if (!get_bbox(gif, &w, &h, &x, &y)) {
        w = h = 1; x = y = 0;
    }

    put_image(gif, w, h, x, y);
    gif->nframes++;

    if (gif->bgindex < 0) {
        uint8_t *tmp = gif->back;
        gif->back = gif->frame;
        gif->frame = tmp;
    }
}

void ge_close_gif(ge_GIF* gif) {
    safe_write(gif->fd, ";", 1);
    close(gif->fd);
    free(gif);
}

#endif /* GIF_H */