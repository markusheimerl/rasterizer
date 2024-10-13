// rasterizer.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

// Constants
#define WIDTH 640
#define HEIGHT 480
#define MAX_VERTICES 100000
#define MAX_TEXCOORDS 100000
#define MAX_FACES 200000
#define NUM_FRAMES 60

// Data structures
typedef struct {
    float x, y, z;
} Vec3;

typedef struct {
    float u, v;
} Vec2;

typedef struct {
    int v_idx[3];  // Vertex indices
    int vt_idx[3]; // Texture coordinate indices
} Face;

typedef struct {
    uint8_t* data;
    int width;
    int height;
    int channels;
} Image;

typedef struct {
    uint8_t r, g, b;
} Color;

// Global variables
Vec3 vertices[MAX_VERTICES];
Vec2 texcoords[MAX_TEXCOORDS];
Face faces[MAX_FACES];
int num_vertices = 0;
int num_texcoords = 0;
int num_faces = 0;

Image texture;

Color framebuffer[WIDTH * HEIGHT];
float zbuffer[WIDTH * HEIGHT];

// Function prototypes
void load_obj(const char* filename);
void load_tga(const char* filename, Image* image);
void matrix_identity(float mat[4][4]);
void matrix_multiply(float result[4][4], float a[4][4], float b[4][4]);
Vec3 matrix_transform(Vec3 v, float mat[4][4]);
void matrix_scale(float mat[4][4], float sx, float sy, float sz);
void matrix_rotate_y(float mat[4][4], float angle);
void matrix_translate(float mat[4][4], float tx, float ty, float tz);
void matrix_perspective(float mat[4][4], float fov, float aspect, float near, float far);
void clear_buffers();
void put_pixel(int x, int y, float z, Color color);
void draw_triangle(Vec3 v0, Vec3 v1, Vec3 v2, Vec2 uv0, Vec2 uv1, Vec2 uv2);
void save_framebuffer(const char* filename);

int main() {
    load_obj("african_head.obj");
    load_tga("african_head_diffuse.tga", &texture);

    // Transformation matrices
    float model[4][4], view[4][4], projection[4][4], mvp[4][4];
    float temp[4][4];

    // Set up projection matrix
    float aspect = (float)WIDTH / (float)HEIGHT;
    matrix_perspective(projection, M_PI / 3.0f, aspect, 0.1f, 100.0f);

    // Set up view matrix (camera)
    matrix_identity(view);
    // Simple camera at (0, 0, 3)
    view[3][2] = -3.0f;

    // For each frame
    for (int frame = 0; frame < NUM_FRAMES; frame++) {
        clear_buffers();

        // Model transformations
        float angle = frame * (2 * M_PI / NUM_FRAMES);
        float rotation[4][4], translation[4][4], scale[4][4];

        matrix_rotate_y(rotation, angle);
        matrix_translate(translation, 0.0f, 0.0f, 0.0f);
        matrix_scale(scale, 1.0f, 1.0f, 1.0f);

        // Combine transformations: model = translation * rotation * scale
        matrix_multiply(temp, rotation, scale);
        matrix_multiply(model, translation, temp);

        // Compute MVP matrix: mvp = projection * view * model
        matrix_multiply(temp, view, model);
        matrix_multiply(mvp, projection, temp);

        // Render each face
        for (int i = 0; i < num_faces; i++) {
            Face* face = &faces[i];

            Vec3 v0 = vertices[face->v_idx[0]];
            Vec3 v1 = vertices[face->v_idx[1]];
            Vec3 v2 = vertices[face->v_idx[2]];

            Vec2 uv0 = texcoords[face->vt_idx[0]];
            Vec2 uv1 = texcoords[face->vt_idx[1]];
            Vec2 uv2 = texcoords[face->vt_idx[2]];

            // Transform vertices
            v0 = matrix_transform(v0, mvp);
            v1 = matrix_transform(v1, mvp);
            v2 = matrix_transform(v2, mvp);

            // Perspective division and viewport transformation
            v0.x = (v0.x + 1.0f) * 0.5f * WIDTH;
            v0.y = (1.0f - v0.y) * 0.5f * HEIGHT;
            v1.x = (v1.x + 1.0f) * 0.5f * WIDTH;
            v1.y = (1.0f - v1.y) * 0.5f * HEIGHT;
            v2.x = (v2.x + 1.0f) * 0.5f * WIDTH;
            v2.y = (1.0f - v2.y) * 0.5f * HEIGHT;

            // Draw the triangle
            draw_triangle(v0, v1, v2, uv0, uv1, uv2);
        }

        // Save the framebuffer as an image
        char filename[64];
        sprintf(filename, "frame_%03d.ppm", frame);
        save_framebuffer(filename);
        printf("Saved %s\n", filename);
    }

    // Free resources
    free(texture.data);

    return 0;
}

// Function implementations

void load_obj(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Failed to open OBJ file: %s\n", filename);
        exit(1);
    }
    char line[256];
    while (fgets(line, sizeof(line), file)) {
        if (strncmp(line, "v ", 2) == 0) {
            Vec3 v;
            sscanf(line + 2, "%f %f %f", &v.x, &v.y, &v.z);
            vertices[num_vertices++] = v;
        } else if (strncmp(line, "vt ", 3) == 0) {
            Vec2 vt;
            sscanf(line + 3, "%f %f", &vt.u, &vt.v);
            texcoords[num_texcoords++] = vt;
        } else if (strncmp(line, "f ", 2) == 0) {
            Face f;
            int matches = sscanf(line + 2, "%d/%d %d/%d %d/%d",
                                 &f.v_idx[0], &f.vt_idx[0],
                                 &f.v_idx[1], &f.vt_idx[1],
                                 &f.v_idx[2], &f.vt_idx[2]);
            if (matches != 6) {
                fprintf(stderr, "Failed to parse face: %s", line);
                continue;
            }
            // OBJ indices start at 1
            for (int i = 0; i < 3; i++) {
                f.v_idx[i] -= 1;
                f.vt_idx[i] -= 1;
            }
            faces[num_faces++] = f;
        }
    }
    fclose(file);
}

void load_tga(const char* filename, Image* image) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Failed to open TGA file: %s\n", filename);
        exit(1);
    }

    uint8_t header[18];
    fread(header, 1, 18, file);

    image->width = header[12] + (header[13] << 8);
    image->height = header[14] + (header[15] << 8);
    uint8_t depth = header[16];

    if (depth != 24 && depth != 32) {
        fprintf(stderr, "Unsupported TGA depth: %d\n", depth);
        exit(1);
    }

    image->channels = depth / 8;
    int size = image->width * image->height * image->channels;
    image->data = (uint8_t*)malloc(size);

    fread(image->data, 1, size, file);
    fclose(file);

    // TGA images are stored bottom-left to top-right, so we need to flip them vertically
    int row_size = image->width * image->channels;
    uint8_t* temp_row = (uint8_t*)malloc(row_size);
    for (int y = 0; y < image->height / 2; y++) {
        uint8_t* row_top = image->data + y * row_size;
        uint8_t* row_bottom = image->data + (image->height - 1 - y) * row_size;
        memcpy(temp_row, row_top, row_size);
        memcpy(row_top, row_bottom, row_size);
        memcpy(row_bottom, temp_row, row_size);
    }
    free(temp_row);
}

void matrix_identity(float mat[4][4]) {
    memset(mat, 0, 16 * sizeof(float));
    for (int i = 0; i < 4; i++) {
        mat[i][i] = 1.0f;
    }
}

void matrix_multiply(float result[4][4], float a[4][4], float b[4][4]) {
    float temp[4][4];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            temp[i][j] = 0;
            for (int k = 0; k < 4; k++) {
                temp[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    memcpy(result, temp, 16 * sizeof(float));
}

Vec3 matrix_transform(Vec3 v, float mat[4][4]) {
    float x = v.x * mat[0][0] + v.y * mat[1][0] + v.z * mat[2][0] + mat[3][0];
    float y = v.x * mat[0][1] + v.y * mat[1][1] + v.z * mat[2][1] + mat[3][1];
    float z = v.x * mat[0][2] + v.y * mat[1][2] + v.z * mat[2][2] + mat[3][2];
    float w = v.x * mat[0][3] + v.y * mat[1][3] + v.z * mat[2][3] + mat[3][3];

    if (w != 0.0f) {
        x /= w;
        y /= w;
        z /= w;
    }
    Vec3 result = { x, y, z };
    return result;
}

void matrix_scale(float mat[4][4], float sx, float sy, float sz) {
    matrix_identity(mat);
    mat[0][0] = sx;
    mat[1][1] = sy;
    mat[2][2] = sz;
}

void matrix_rotate_y(float mat[4][4], float angle) {
    matrix_identity(mat);
    float c = cosf(angle);
    float s = sinf(angle);
    mat[0][0] = c;
    mat[0][2] = s;
    mat[2][0] = -s;
    mat[2][2] = c;
}

void matrix_translate(float mat[4][4], float tx, float ty, float tz) {
    matrix_identity(mat);
    mat[3][0] = tx;
    mat[3][1] = ty;
    mat[3][2] = tz;
}

void matrix_perspective(float mat[4][4], float fov, float aspect, float near, float far) {
    float f = 1.0f / tanf(fov / 2.0f);
    memset(mat, 0, 16 * sizeof(float));
    mat[0][0] = f / aspect;
    mat[1][1] = f;
    mat[2][2] = (far + near) / (near - far);
    mat[2][3] = -1.0f;
    mat[3][2] = (2 * far * near) / (near - far);
}

void clear_buffers() {
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        framebuffer[i].r = 0;
        framebuffer[i].g = 0;
        framebuffer[i].b = 0;
        zbuffer[i] = -INFINITY;
    }
}

void put_pixel(int x, int y, float z, Color color) {
    if (x < 0 || x >= WIDTH || y < 0 || y >= HEIGHT) return;
    int idx = y * WIDTH + x;
    if (z > zbuffer[idx]) {
        zbuffer[idx] = z;
        framebuffer[idx] = color;
    }
}

void draw_triangle(Vec3 v0, Vec3 v1, Vec3 v2, Vec2 uv0, Vec2 uv1, Vec2 uv2) {
    // Bounding box
    int minX = fmax(0, floorf(fminf(fminf(v0.x, v1.x), v2.x)));
    int maxX = fmin(WIDTH - 1, ceilf(fmaxf(fmaxf(v0.x, v1.x), v2.x)));
    int minY = fmax(0, floorf(fminf(fminf(v0.y, v1.y), v2.y)));
    int maxY = fmin(HEIGHT - 1, ceilf(fmaxf(fmaxf(v0.y, v1.y), v2.y)));

    float area = (v1.x - v0.x) * (v2.y - v0.y) - (v2.x - v0.x) * (v1.y - v0.y);
    if (fabs(area) < 1e-6) return; // Degenerate triangle

    // For each pixel in bounding box
    for (int y = minY; y <= maxY; y++) {
        for (int x = minX; x <= maxX; x++) {
            // Barycentric coordinates
            float w0 = ((v1.x - v0.x) * (y - v0.y) - (v1.y - v0.y) * (x - v0.x)) / area;
            float w1 = ((v2.x - v1.x) * (y - v1.y) - (v2.y - v1.y) * (x - v1.x)) / area;
            float w2 = ((v0.x - v2.x) * (y - v2.y) - (v0.y - v2.y) * (x - v2.x)) / area;
            if (w0 >= 0 && w1 >= 0 && w2 >= 0) {
                // Perspective correct interpolation
                float z = w0 * v0.z + w1 * v1.z + w2 * v2.z;
                float u = w0 * uv0.u / v0.z + w1 * uv1.u / v1.z + w2 * uv2.u / v2.z;
                float v = w0 * uv0.v / v0.z + w1 * uv1.v / v1.z + w2 * uv2.v / v2.z;
                float inv_z = w0 / v0.z + w1 / v1.z + w2 / v2.z;
                u /= inv_z;
                v /= inv_z;

                // Sample texture
                int tex_x = fmin(texture.width - 1, fmax(0, (int)(u * texture.width)));
                int tex_y = fmin(texture.height - 1, fmax(0, (int)((v) * texture.height)));
                int tex_idx = (tex_y * texture.width + tex_x) * texture.channels;
                Color color = {
                    texture.data[tex_idx],
                    texture.data[tex_idx + 1],
                    texture.data[tex_idx + 2]
                };
                put_pixel(x, y, z, color);
            }
        }
    }
}

void save_framebuffer(const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Failed to save framebuffer to %s\n", filename);
        exit(1);
    }
    fprintf(f, "P6\n%d %d\n255\n", WIDTH, HEIGHT);
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        fputc(framebuffer[i].r, f);
        fputc(framebuffer[i].g, f);
        fputc(framebuffer[i].b, f);
    }
    fclose(f);
}

