#ifndef GIFENC_H
#define GIFENC_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

typedef struct ge_GIF {
    uint16_t w, h;
    int depth;
    int bgindex;
    int fd;
    int offset;
    int nframes;
    uint8_t *frame, *back;
    uint32_t partial;
    uint8_t buffer[0xFF];
    uint8_t palette[8 * 3];
} ge_GIF;

#define write_num(fd, n) do { \
    ssize_t _write_result = write((fd), (uint8_t []) {(n) & 0xFF, (n) >> 8}, 2); \
    if (_write_result != 2) { \
        perror("Error writing number"); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

void safe_write(int fd, const void *buf, size_t count) {
    ssize_t result = write(fd, buf, count);
    if (result != (ssize_t) count) {
        perror("Error writing data");
        exit(EXIT_FAILURE);
    }
}

static uint8_t vga[0x30] = {
    0x00, 0x00, 0x00, 0xAA, 0x00, 0x00, 0x00, 0xAA, 0x00, 0xAA, 0x55, 0x00,
    0x00, 0x00, 0xAA, 0xAA, 0x00, 0xAA, 0x00, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA,
    0x55, 0x55, 0x55, 0xFF, 0x55, 0x55, 0xFF, 0xFF, 0x55, 0xFF, 0x55, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF,
};

typedef struct Node {
    uint16_t key;
    struct Node *children[];
} Node;

static Node *new_node(uint16_t key, int degree) {
    Node *node = calloc(1, sizeof(*node) + degree * sizeof(Node *));
    if (node) node->key = key;
    return node;
}

static Node *new_trie(int degree, int *nkeys) {
    Node *root = new_node(0, degree);
    for (*nkeys = 0; *nkeys < degree; (*nkeys)++)
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

static void write_and_store(int s, uint8_t *dst, int fd, const uint8_t *src, size_t len) {
    safe_write(fd, src, len);
    if (s) {
        memcpy(dst, src, len);
        dst += len;
    }
}

static void put_loop(ge_GIF *gif, uint16_t loop) {
    safe_write(gif->fd, (uint8_t[]) {'!', 0xFF, 0x0B}, 3);
    safe_write(gif->fd, "NETSCAPE2.0", 11);
    safe_write(gif->fd, (uint8_t[]) {0x03, 0x01}, 2);
    write_num(gif->fd, loop);
    safe_write(gif->fd, "\0", 1);
}

ge_GIF *ge_new_gif(const char *fname, uint16_t width, uint16_t height,
                   uint8_t *palette, int depth, int bgindex, int loop) {
    ge_GIF *gif = calloc(1, sizeof(*gif) + (bgindex < 0 ? 2 : 1) * width * height);
    if (!gif) return NULL;

    gif->w = width; gif->h = height;
    gif->bgindex = bgindex;
    gif->frame = (uint8_t *) &gif[1];
    gif->back = &gif->frame[width * height];
    gif->fd = creat(fname, 0666);
    if (gif->fd == -1) { free(gif); return NULL; }
    if (palette) memcpy(gif->palette, palette, 8 * 3);

    safe_write(gif->fd, "GIF89a", 6);
    write_num(gif->fd, width);
    write_num(gif->fd, height);

    int store_gct = palette && depth < 0;
    int custom_gct = palette && depth > 0;
    gif->depth = abs(depth) > 1 ? abs(depth) : 2;

    safe_write(gif->fd, (uint8_t[]) {0xF0 | (gif->depth - 1), (uint8_t) bgindex, 0x00}, 3);

    if (custom_gct) {
        safe_write(gif->fd, palette, 3 << gif->depth);
    } else if (gif->depth <= 4) {
        write_and_store(store_gct, palette, gif->fd, vga, 3 << gif->depth);
    } else {
        write_and_store(store_gct, palette, gif->fd, vga, sizeof(vga));
        for (int r = 0, i = 0x10; r < 6 && i < (1 << gif->depth); r++) {
            for (int g = 0; g < 6 && i < (1 << gif->depth); g++) {
                for (int b = 0; b < 6 && i < (1 << gif->depth); b++, i++) {
                    uint8_t rgb[3] = {r * 51, g * 51, b * 51};
                    write_and_store(store_gct, palette, gif->fd, rgb, 3);
                }
            }
        }
    }

    if (loop >= 0 && loop <= 0xFFFF)
        put_loop(gif, (uint16_t) loop);
    return gif;
}

static void put_key(ge_GIF *gif, uint16_t key, int key_size) {
    int byte_offset = gif->offset / 8;
    int bit_offset = gif->offset % 8;
    gif->partial |= ((uint32_t) key) << bit_offset;
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
        safe_write(gif->fd, (uint8_t[]) {byte_offset}, 1);
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
    safe_write(gif->fd, (uint8_t[]) {0x00, gif->depth}, 2);

    root = node = new_trie(degree, &nkeys);
    key_size = gif->depth + 1;
    put_key(gif, degree, key_size);

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

    for (int i = 0, k = 0; i < gif->h; i++) {
        for (int j = 0; j < gif->w; j++, k++) {
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

void floyd_steinberg_dithering(uint8_t *input, ge_GIF *gif) {
    int width = gif->w;
    int height = gif->h;

    // Create a temporary buffer to store the error diffusion
    double (*error_buffer)[3] = calloc(width * height, sizeof(*error_buffer));
    
    // Copy input to error buffer
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < 3; c++) {
                error_buffer[y * width + x][c] = input[(y * width + x) * 3 + c];
            }
        }
    }

    // Apply Floyd-Steinberg dithering
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            uint8_t pixel[3];
            for (int c = 0; c < 3; c++) {
                pixel[c] = (uint8_t)fmax(0, fmin(255, round(error_buffer[y * width + x][c])));
            }

            // Find nearest color in palette
            uint8_t nearest_color = 0;
            double min_distance = DBL_MAX;
            for (int i = 0; i < 8; i++) {
                double distance = sqrt(
                    pow((double)(pixel[0] - gif->palette[i * 3]), 2.0) +
                    pow((double)(pixel[1] - gif->palette[i * 3 + 1]), 2.0) +
                    pow((double)(pixel[2] - gif->palette[i * 3 + 2]), 2.0)
                );
                if (distance < min_distance) {
                    min_distance = distance;
                    nearest_color = i;
                }
            }

            gif->frame[y * width + x] = nearest_color;

            double error[3];
            for (int c = 0; c < 3; c++) {
                error[c] = error_buffer[y * width + x][c] - gif->palette[nearest_color * 3 + c];
            }

            if (x + 1 < width) {
                for (int c = 0; c < 3; c++)
                    error_buffer[y * width + (x + 1)][c] += error[c] * 7.0 / 16.0;
            }
            if (y + 1 < height) {
                if (x > 0) {
                    for (int c = 0; c < 3; c++)
                        error_buffer[(y + 1) * width + (x - 1)][c] += error[c] * 3.0 / 16.0;
                }
                for (int c = 0; c < 3; c++)
                    error_buffer[(y + 1) * width + x][c] += error[c] * 5.0 / 16.0;
                if (x + 1 < width) {
                    for (int c = 0; c < 3; c++)
                        error_buffer[(y + 1) * width + (x + 1)][c] += error[c] * 1.0 / 16.0;
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
        safe_write(gif->fd, (uint8_t[]) {'!', 0xF9, 0x04,
                                         ((gif->bgindex >= 0 ? 2 : 1) << 2) + 1}, 4);
        write_num(gif->fd, delay);
        safe_write(gif->fd, (uint8_t[]) {(uint8_t) gif->bgindex, 0x00}, 2);
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

#endif /* GIFENC_H */