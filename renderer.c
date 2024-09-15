// Include necessary headers
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"          // For image loading
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"    // For image saving

// Define constants
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAX_VERTICES 100000
#define MAX_TEXCOORDS 100000
#define MAX_FACES 100000

// Define vector and matrix structures
typedef struct { float x, y, z; } Vec3f;
typedef struct { float x, y; } Vec2f;
typedef struct { float x, y, z, w; } Vec4f; // Added Vec4f definition
typedef struct { float m[4][4]; } Mat4f;

// Global variables for vertices, texture coordinates, faces, and textures
Vec3f vertices[MAX_VERTICES];
int num_vertices = 0;

Vec2f texture_coords[MAX_TEXCOORDS];
int num_texture_coords = 0;

typedef struct {
    int vertex_indices[3];
    int texture_indices[3];
} Face;

Face faces[MAX_FACES];
int num_faces = 0;

unsigned char *texture_data;
int texture_width, texture_height, texture_channels;

unsigned char *image_data;
float *depth_buffer;

// Vector and matrix utility functions
Vec3f vec3f_sub(Vec3f a, Vec3f b) {
    return (Vec3f){ a.x - b.x, a.y - b.y, a.z - b.z };
}

Vec3f vec3f_normalize(Vec3f v) {
    float length = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
    return (Vec3f){ v.x / length, v.y / length, v.z / length };
}

Vec3f vec3f_cross(Vec3f a, Vec3f b) {
    return (Vec3f){
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x
    };
}

float vec3f_dot(Vec3f a, Vec3f b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

Vec4f mat4f_mul_vec4f(Mat4f mat, Vec4f vec) {
    Vec4f result = {0};
    result.x = mat.m[0][0]*vec.x + mat.m[0][1]*vec.y + mat.m[0][2]*vec.z + mat.m[0][3]*vec.w;
    result.y = mat.m[1][0]*vec.x + mat.m[1][1]*vec.y + mat.m[1][2]*vec.z + mat.m[1][3]*vec.w;
    result.z = mat.m[2][0]*vec.x + mat.m[2][1]*vec.y + mat.m[2][2]*vec.z + mat.m[2][3]*vec.w;
    result.w = mat.m[3][0]*vec.x + mat.m[3][1]*vec.y + mat.m[3][2]*vec.z + mat.m[3][3]*vec.w;
    return result;
}

// Matrix transformation functions
Mat4f identMat4f() {
    Mat4f mat = {0};
    mat.m[0][0] = mat.m[1][1] = mat.m[2][2] = mat.m[3][3] = 1.0f;
    return mat;
}

Mat4f multMat4f(Mat4f a, Mat4f b) {
    Mat4f result = {0};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 4; ++k)
                result.m[i][j] += a.m[i][k]*b.m[k][j];
    return result;
}

Mat4f translMat4f(float tx, float ty, float tz) {
    Mat4f mat = identMat4f();
    mat.m[0][3] = tx; mat.m[1][3] = ty; mat.m[2][3] = tz;
    return mat;
}

Mat4f xRotMat4f(float angle) {
    float c = cosf(angle), s = sinf(angle);
    Mat4f mat = identMat4f();
    mat.m[1][1] = c; mat.m[1][2] = -s;
    mat.m[2][1] = s; mat.m[2][2] = c;
    return mat;
}

Mat4f yRotMat4f(float angle) {
    float c = cosf(angle), s = sinf(angle);
    Mat4f mat = identMat4f();
    mat.m[0][0] = c; mat.m[0][2] = s;
    mat.m[2][0] = -s; mat.m[2][2] = c;
    return mat;
}

Mat4f zRotMat4f(float angle) {
    float c = cosf(angle), s = sinf(angle);
    Mat4f mat = identMat4f();
    mat.m[0][0] = c; mat.m[0][1] = -s;
    mat.m[1][0] = s; mat.m[1][1] = c;
    return mat;
}

Mat4f scaleMat4f(float sx, float sy, float sz) {
    Mat4f mat = identMat4f();
    mat.m[0][0] = sx; mat.m[1][1] = sy; mat.m[2][2] = sz;
    return mat;
}

Mat4f modelMat4f(float tx, float ty, float tz, float rx, float ry, float rz, float sx, float sy, float sz) {
    Mat4f translation = translMat4f(tx, ty, tz);
    Mat4f rotation = multMat4f(multMat4f(xRotMat4f(rx), yRotMat4f(ry)), zRotMat4f(rz));
    Mat4f scale = scaleMat4f(sx, sy, sz);
    return multMat4f(translation, multMat4f(rotation, scale));
}

Mat4f create_projection_matrix(float fov, float aspect_ratio, float near, float far) {
    float f = 1.0f / tanf(fov / 2.0f);
    Mat4f mat = {0};
    mat.m[0][0] = f / aspect_ratio; mat.m[1][1] = f;
    mat.m[2][2] = (far + near) / (near - far); mat.m[2][3] = (2 * far * near) / (near - far);
    mat.m[3][2] = -1.0f;
    return mat;
}

Mat4f create_view_matrix(Vec3f eye, Vec3f center, Vec3f up) {
    Vec3f f = vec3f_normalize(vec3f_sub(center, eye));
    Vec3f s = vec3f_normalize(vec3f_cross(f, up));
    Vec3f u = vec3f_cross(s, f);
    Mat4f mat = identMat4f();

    mat.m[0][0] = s.x; mat.m[0][1] = s.y; mat.m[0][2] = s.z; mat.m[0][3] = -vec3f_dot(s, eye);
    mat.m[1][0] = u.x; mat.m[1][1] = u.y; mat.m[1][2] = u.z; mat.m[1][3] = -vec3f_dot(u, eye);
    mat.m[2][0] = -f.x; mat.m[2][1] = -f.y; mat.m[2][2] = -f.z; mat.m[2][3] = vec3f_dot(f, eye);
    return mat;
}

// OBJ file parsing function
void parse_obj_file(const char *file_path) {
    FILE *file = fopen(file_path, "r");
    if (!file) { printf("Failed to open OBJ file %s\n", file_path); exit(1); }

    char line[256];
    while (fgets(line, sizeof(line), file)) {
        if (strncmp(line, "v ", 2) == 0) {
            Vec3f vertex;
            sscanf(line + 2, "%f %f %f", &vertex.x, &vertex.y, &vertex.z);
            vertices[num_vertices++] = vertex;
        } else if (strncmp(line, "vt ", 3) == 0) {
            Vec2f texcoord;
            sscanf(line + 3, "%f %f", &texcoord.x, &texcoord.y);
            texture_coords[num_texture_coords++] = texcoord;
        } else if (strncmp(line, "f ", 2) == 0) {
            Face face;
            char *token = strtok(line + 2, " \n");
            int idx = 0;
            while (token && idx < 3) {
                int vi = 0, ti = 0;
                sscanf(token, "%d/%d", &vi, &ti);
                face.vertex_indices[idx] = vi - 1;
                face.texture_indices[idx] = ti - 1;
                idx++;
                token = strtok(NULL, " \n");
            }
            faces[num_faces++] = face;
        }
    }
    fclose(file);
}

// Edge function for rasterization
float edge_function(Vec2f v0, Vec2f v1, Vec2f p) {
    return (p.x - v0.x)*(v1.y - v0.y) - (p.y - v0.y)*(v1.x - v0.x);
}

// Triangle rendering function
void render_triangles(int width, int height, Mat4f mvp_matrix) {
    image_data = (unsigned char *)calloc(width * height * 3, sizeof(unsigned char));
    depth_buffer = (float *)malloc(width * height * sizeof(float));
    for (int i = 0; i < width * height; i++) depth_buffer[i] = INFINITY;

    for (int face_idx = 0; face_idx < num_faces; face_idx++) {
        Face face = faces[face_idx];

        Vec4f v_homogeneous[3];
        Vec2f vt[3];

        for (int i = 0; i < 3; i++) {
            Vec3f v = vertices[face.vertex_indices[i]];
            vt[i] = texture_coords[face.texture_indices[i]];
            v_homogeneous[i] = (Vec4f){ v.x, v.y, v.z, 1.0f };
            v_homogeneous[i] = mat4f_mul_vec4f(mvp_matrix, v_homogeneous[i]);
            v_homogeneous[i].x /= v_homogeneous[i].w;
            v_homogeneous[i].y /= v_homogeneous[i].w;
            v_homogeneous[i].z /= v_homogeneous[i].w;
            v_homogeneous[i].x = (v_homogeneous[i].x + 1.0f) * 0.5f * width;
            v_homogeneous[i].y = (1.0f - v_homogeneous[i].y) * 0.5f * height;
        }

        // Bounding box
        float min_x = fminf(fminf(v_homogeneous[0].x, v_homogeneous[1].x), v_homogeneous[2].x);
        float max_x = fmaxf(fmaxf(v_homogeneous[0].x, v_homogeneous[1].x), v_homogeneous[2].x);
        float min_y = fminf(fminf(v_homogeneous[0].y, v_homogeneous[1].y), v_homogeneous[2].y);
        float max_y = fmaxf(fmaxf(v_homogeneous[0].y, v_homogeneous[1].y), v_homogeneous[2].y);

        int x0 = fmaxf(0, floorf(min_x));
        int x1 = fminf(width - 1, ceilf(max_x));
        int y0 = fmaxf(0, floorf(min_y));
        int y1 = fminf(height - 1, ceilf(max_y));

        Vec2f v_screen[3] = {
            { v_homogeneous[0].x, v_homogeneous[0].y },
            { v_homogeneous[1].x, v_homogeneous[1].y },
            { v_homogeneous[2].x, v_homogeneous[2].y }
        };

        float area = edge_function(v_screen[0], v_screen[1], v_screen[2]);

        for (int y = y0; y <= y1; y++) {
            for (int x = x0; x <= x1; x++) {
                Vec2f p = { x + 0.5f, y + 0.5f };
                float w0 = edge_function(v_screen[1], v_screen[2], p);
                float w1 = edge_function(v_screen[2], v_screen[0], p);
                float w2 = edge_function(v_screen[0], v_screen[1], p);

                if (w0 >= 0 && w1 >= 0 && w2 >= 0) {
                    w0 /= area; w1 /= area; w2 /= area;

                    float depth = w0*v_homogeneous[0].z + w1*v_homogeneous[1].z + w2*v_homogeneous[2].z;

                    int idx = y * width + x;
                    if (depth < depth_buffer[idx]) {
                        depth_buffer[idx] = depth;

                        // Texture coordinates interpolation
                        float u = w0*vt[0].x + w1*vt[1].x + w2*vt[2].x;
                        float v = 1.0f - (w0*vt[0].y + w1*vt[1].y + w2*vt[2].y);

                        int tex_x = (int)(u * texture_width) % texture_width;
                        int tex_y = (int)(v * texture_height) % texture_height;
                        int tex_idx = (tex_y * texture_width + tex_x) * texture_channels;

                        image_data[idx * 3 + 0] = texture_data[tex_idx];
                        image_data[idx * 3 + 1] = texture_data[tex_idx + 1];
                        image_data[idx * 3 + 2] = texture_data[tex_idx + 2];
                    }
                }
            }
        }
    }
    free(depth_buffer);
}

// Main function
int main() {
    const char *input_file = "african_head.obj";
    const char *texture_file = "african_head_diffuse.tga";
    const char *output_file = "output.png";
    int width = 800, height = 600;

    parse_obj_file(input_file);

    // Load texture image
    texture_data = stbi_load(texture_file, &texture_width, &texture_height, &texture_channels, 3);
    if (!texture_data) {
        printf("Failed to load texture image %s\n", texture_file);
        return 1;
    }

    // Create matrices
    Mat4f model_matrix = modelMat4f(0, 0, 0, 0, M_PI / 4.0f, 0, 1, 1, 1);
    Mat4f view_matrix = create_view_matrix(
        (Vec3f){0, 0, 3},
        (Vec3f){0, 0, 0},
        (Vec3f){0, 1, 0}
    );
    Mat4f projection_matrix = create_projection_matrix(
        M_PI / 4.0f, // 45 degrees in radians
        (float)width / (float)height,
        0.1f,
        100.0f
    );

    Mat4f mvp_matrix = multMat4f(projection_matrix, multMat4f(view_matrix, model_matrix));

    // Render triangles
    render_triangles(width, height, mvp_matrix);

    // Save image
    if (!stbi_write_png(output_file, width, height, 3, image_data, width * 3)) {
        printf("Failed to save output image %s\n", output_file);
    }

    // Free resources
    free(image_data);
    stbi_image_free(texture_data);

    return 0;
}
