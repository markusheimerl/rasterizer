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
#define MAX_FACE_VERTICES 10

// Define vector and matrix structures
typedef struct {
    float x, y, z;
} Vec3f;

typedef struct {
    float x, y;
} Vec2f;

typedef struct {
    float x, y, z, w;
} Vec4f;

typedef struct {
    float m[4][4];
} Mat4f;

// Function prototypes
Mat4f identMat4f();
Mat4f multMat4f(Mat4f a, Mat4f b);
Mat4f translMat4f(float tx, float ty, float tz);
Mat4f xRotMat4f(float angle);
Mat4f yRotMat4f(float angle);
Mat4f zRotMat4f(float angle);
Mat4f scaleMat4f(float sx, float sy, float sz);
Mat4f modelMat4f(float tx, float ty, float tz, float rx, float ry, float rz, float sx, float sy, float sz);
Mat4f create_projection_matrix(float fov, float aspect_ratio, float near, float far);
Mat4f create_view_matrix(Vec3f eye, Vec3f center, Vec3f up);
void parse_obj_file(const char *file_path);
float edge_function(Vec2f v0, Vec2f v1, Vec2f p);
void render_triangles(int width, int height, Mat4f model_matrix, Mat4f view_matrix, Mat4f projection_matrix);

// Global variables for vertices, texture coordinates, faces, and textures
Vec3f vertices[MAX_VERTICES];
int num_vertices = 0;

Vec2f texture_coords[MAX_TEXCOORDS];
int num_texture_coords = 0;

typedef struct {
    int vertex_indices[MAX_FACE_VERTICES];
    int texture_indices[MAX_FACE_VERTICES];
    int num_vertices;
} Face;

Face faces[MAX_FACES];
int num_faces = 0;

unsigned char *texture_data;
int texture_width, texture_height, texture_channels;

unsigned char *image_data;
float *depth_buffer;

// Vector and matrix utility functions
Vec3f vec3f_sub(Vec3f a, Vec3f b) {
    Vec3f result = {a.x - b.x, a.y - b.y, a.z - b.z};
    return result;
}

Vec3f vec3f_normalize(Vec3f v) {
    float length = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    Vec3f result = {v.x / length, v.y / length, v.z / length};
    return result;
}

Vec3f vec3f_cross(Vec3f a, Vec3f b) {
    Vec3f result = {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
    return result;
}

float vec3f_dot(Vec3f a, Vec3f b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec4f mat4f_mul_vec4f(Mat4f mat, Vec4f vec) {
    Vec4f result;
    result.x = mat.m[0][0]*vec.x + mat.m[0][1]*vec.y + mat.m[0][2]*vec.z + mat.m[0][3]*vec.w;
    result.y = mat.m[1][0]*vec.x + mat.m[1][1]*vec.y + mat.m[1][2]*vec.z + mat.m[1][3]*vec.w;
    result.z = mat.m[2][0]*vec.x + mat.m[2][1]*vec.y + mat.m[2][2]*vec.z + mat.m[2][3]*vec.w;
    result.w = mat.m[3][0]*vec.x + mat.m[3][1]*vec.y + mat.m[3][2]*vec.z + mat.m[3][3]*vec.w;
    return result;
}

// Matrix transformation functions
Mat4f identMat4f() {
    Mat4f mat = {0};
    mat.m[0][0] = 1.0f;
    mat.m[1][1] = 1.0f;
    mat.m[2][2] = 1.0f;
    mat.m[3][3] = 1.0f;
    return mat;
}

Mat4f multMat4f(Mat4f a, Mat4f b) {
    Mat4f result = {0};
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
            result.m[i][j] = 0;
            for(int k = 0; k < 4; k++) {
                result.m[i][j] += a.m[i][k]*b.m[k][j];
            }
        }
    }
    return result;
}

Mat4f translMat4f(float tx, float ty, float tz) {
    Mat4f mat = identMat4f();
    mat.m[0][3] = tx;
    mat.m[1][3] = ty;
    mat.m[2][3] = tz;
    return mat;
}

Mat4f xRotMat4f(float angle) {
    float c = cosf(angle);
    float s = sinf(angle);
    Mat4f mat = identMat4f();
    mat.m[1][1] = c;
    mat.m[1][2] = -s;
    mat.m[2][1] = s;
    mat.m[2][2] = c;
    return mat;
}

Mat4f yRotMat4f(float angle) {
    float c = cosf(angle);
    float s = sinf(angle);
    Mat4f mat = identMat4f();
    mat.m[0][0] = c;
    mat.m[0][2] = s;
    mat.m[2][0] = -s;
    mat.m[2][2] = c;
    return mat;
}

Mat4f zRotMat4f(float angle) {
    float c = cosf(angle);
    float s = sinf(angle);
    Mat4f mat = identMat4f();
    mat.m[0][0] = c;
    mat.m[0][1] = -s;
    mat.m[1][0] = s;
    mat.m[1][1] = c;
    return mat;
}

Mat4f scaleMat4f(float sx, float sy, float sz) {
    Mat4f mat = identMat4f();
    mat.m[0][0] = sx;
    mat.m[1][1] = sy;
    mat.m[2][2] = sz;
    return mat;
}

Mat4f modelMat4f(float tx, float ty, float tz, float rx, float ry, float rz, float sx, float sy, float sz) {
    Mat4f modelmatrix = identMat4f();
    modelmatrix = multMat4f(translMat4f(tx, ty, tz), modelmatrix);
    Mat4f rotation = multMat4f(multMat4f(xRotMat4f(rx), yRotMat4f(ry)), zRotMat4f(rz));
    modelmatrix = multMat4f(rotation, modelmatrix);
    modelmatrix = multMat4f(scaleMat4f(sx, sy, sz), modelmatrix);
    return modelmatrix;
}

Mat4f create_projection_matrix(float fov, float aspect_ratio, float near, float far) {
    float f = 1.0f / tanf(fov / 2.0f);
    Mat4f mat = {0};
    mat.m[0][0] = f / aspect_ratio;
    mat.m[1][1] = f;
    mat.m[2][2] = (far + near) / (near - far);
    mat.m[2][3] = (2 * far * near) / (near - far);
    mat.m[3][2] = -1.0f;
    return mat;
}

Mat4f create_view_matrix(Vec3f eye, Vec3f center, Vec3f up) {
    Vec3f f = vec3f_normalize(vec3f_sub(center, eye));
    Vec3f s = vec3f_normalize(vec3f_cross(f, up));
    Vec3f u = vec3f_cross(s, f);

    Mat4f mat = identMat4f();

    mat.m[0][0] = s.x;
    mat.m[0][1] = s.y;
    mat.m[0][2] = s.z;
    mat.m[0][3] = -vec3f_dot(s, eye);

    mat.m[1][0] = u.x;
    mat.m[1][1] = u.y;
    mat.m[1][2] = u.z;
    mat.m[1][3] = -vec3f_dot(u, eye);

    mat.m[2][0] = -f.x;
    mat.m[2][1] = -f.y;
    mat.m[2][2] = -f.z;
    mat.m[2][3] = vec3f_dot(f, eye);

    mat.m[3][0] = 0.0f;
    mat.m[3][1] = 0.0f;
    mat.m[3][2] = 0.0f;
    mat.m[3][3] = 1.0f;

    return mat;
}

// OBJ file parsing function
void parse_obj_file(const char *file_path) {
    FILE *file = fopen(file_path, "r");
    if (!file) {
        printf("Failed to open OBJ file %s\n", file_path);
        exit(1);
    }

    char line[256];
    while (fgets(line, sizeof(line), file)) {
        if (strncmp(line, "v ", 2) == 0) {
            // Vertex position
            Vec3f vertex;
            sscanf(line + 2, "%f %f %f", &vertex.x, &vertex.y, &vertex.z);
            vertices[num_vertices++] = vertex;
        } else if (strncmp(line, "vt ", 3) == 0) {
            // Texture coordinate
            Vec2f texcoord;
            sscanf(line + 3, "%f %f", &texcoord.x, &texcoord.y);
            texture_coords[num_texture_coords++] = texcoord;
        } else if (strncmp(line, "f ", 2) == 0) {
            // Face
            Face face;
            face.num_vertices = 0;
            char *token = strtok(line + 2, " \n");
            while (token) {
                int vi = 0, ti = 0;
                if (strstr(token, "/")) {
                    // Format: vertex_index/texture_index
                    sscanf(token, "%d/%d", &vi, &ti);
                } else {
                    // Only vertex indices
                    sscanf(token, "%d", &vi);
                }
                if (vi < 0) vi += num_vertices + 1;
                if (ti < 0) ti += num_texture_coords + 1;
                face.vertex_indices[face.num_vertices] = vi - 1;
                face.texture_indices[face.num_vertices] = ti - 1;
                face.num_vertices++;
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
void render_triangles(int width, int height, Mat4f model_matrix, Mat4f view_matrix, Mat4f projection_matrix) {
    image_data = (unsigned char *)malloc(width * height * 3);
    memset(image_data, 0, width * height * 3);
    depth_buffer = (float *)malloc(width * height * sizeof(float));
    for (int i = 0; i < width * height; i++) {
        depth_buffer[i] = INFINITY;
    }

    Mat4f view_model_matrix = multMat4f(view_matrix, model_matrix);

    for (int face_idx = 0; face_idx < num_faces; face_idx++) {
        Face face = faces[face_idx];

        // Only process triangles
        if (face.num_vertices != 3) {
            continue;
        }

        // Get vertices and texture coordinates
        Vec3f v0 = vertices[face.vertex_indices[0]];
        Vec3f v1 = vertices[face.vertex_indices[1]];
        Vec3f v2 = vertices[face.vertex_indices[2]];

        Vec2f vt0 = texture_coords[face.texture_indices[0]];
        Vec2f vt1 = texture_coords[face.texture_indices[1]];
        Vec2f vt2 = texture_coords[face.texture_indices[2]];

        // Apply view-model matrix
        Vec4f v0_h = {v0.x, v0.y, v0.z, 1.0f};
        Vec4f v1_h = {v1.x, v1.y, v1.z, 1.0f};
        Vec4f v2_h = {v2.x, v2.y, v2.z, 1.0f};

        v0_h = mat4f_mul_vec4f(view_model_matrix, v0_h);
        v1_h = mat4f_mul_vec4f(view_model_matrix, v1_h);
        v2_h = mat4f_mul_vec4f(view_model_matrix, v2_h);

        // Apply projection matrix
        v0_h = mat4f_mul_vec4f(projection_matrix, v0_h);
        v1_h = mat4f_mul_vec4f(projection_matrix, v1_h);
        v2_h = mat4f_mul_vec4f(projection_matrix, v2_h);

        // Perspective division
        v0_h.x /= v0_h.w;
        v0_h.y /= v0_h.w;
        v0_h.z /= v0_h.w;

        v1_h.x /= v1_h.w;
        v1_h.y /= v1_h.w;
        v1_h.z /= v1_h.w;

        v2_h.x /= v2_h.w;
        v2_h.y /= v2_h.w;
        v2_h.z /= v2_h.w;

        // Convert to screen space
        v0_h.x = (v0_h.x + 1.0f) * width / 2.0f;
        v0_h.y = (1.0f - v0_h.y) * height / 2.0f;

        v1_h.x = (v1_h.x + 1.0f) * width / 2.0f;
        v1_h.y = (1.0f - v1_h.y) * height / 2.0f;

        v2_h.x = (v2_h.x + 1.0f) * width / 2.0f;
        v2_h.y = (1.0f - v2_h.y) * height / 2.0f;

        // Prepare for rasterization
        Vec2f v0_screen = {v0_h.x, v0_h.y};
        Vec2f v1_screen = {v1_h.x, v1_h.y};
        Vec2f v2_screen = {v2_h.x, v2_h.y};

        // Calculate bounding box
        int min_x = fmaxf(0, floorf(fminf(fminf(v0_screen.x, v1_screen.x), v2_screen.x)));
        int max_x = fminf(width - 1, ceilf(fmaxf(fmaxf(v0_screen.x, v1_screen.x), v2_screen.x)));
        int min_y = fmaxf(0, floorf(fminf(fminf(v0_screen.y, v1_screen.y), v2_screen.y)));
        int max_y = fminf(height - 1, ceilf(fmaxf(fmaxf(v0_screen.y, v1_screen.y), v2_screen.y)));

        // Rasterize the triangle
        for (int y = min_y; y <= max_y; y++) {
            for (int x = min_x; x <= max_x; x++) {
                Vec2f p = {(float)x + 0.5f, (float)y + 0.5f};

                float w0 = edge_function(v1_screen, v2_screen, p);
                float w1 = edge_function(v2_screen, v0_screen, p);
                float w2 = edge_function(v0_screen, v1_screen, p);

                if (w0 >= 0 && w1 >= 0 && w2 >= 0) {
                    // Calculate area
                    float area = edge_function(v0_screen, v1_screen, v2_screen);
                    // Barycentric coordinates
                    w0 /= area;
                    w1 /= area;
                    w2 /= area;

                    // Interpolate depth
                    float depth = w0 * v0_h.z + w1 * v1_h.z + w2 * v2_h.z;

                    // Depth test
                    int index = y * width + x;
                    if (depth < depth_buffer[index]) {
                        depth_buffer[index] = depth;

                        // Interpolate texture coordinates
                        float tx = w0 * vt0.x + w1 * vt1.x + w2 * vt2.x;
                        float ty = 1.0f - (w0 * vt0.y + w1 * vt1.y + w2 * vt2.y); // Invert v coordinate

                        // Sample texture color
                        int tex_x = (int)(tx * texture_width) % texture_width;
                        int tex_y = (int)(ty * texture_height) % texture_height;
                        int tex_index = (tex_y * texture_width + tex_x) * texture_channels;

                        unsigned char r = texture_data[tex_index];
                        unsigned char g = texture_data[tex_index + 1];
                        unsigned char b = texture_data[tex_index + 2];

                        // Set pixel color
                        int img_index = (y * width + x) * 3;
                        image_data[img_index] = r;
                        image_data[img_index + 1] = g;
                        image_data[img_index + 2] = b;
                    }
                }
            }
        }
    }

    // Free depth buffer
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
        (float)(M_PI / 4.0f), // 45 degrees in radians
        (float)width / (float)height,
        0.1f,
        100.0f
    );

    // Render triangles
    render_triangles(width, height, model_matrix, view_matrix, projection_matrix);

    // Save image
    if (!stbi_write_png(output_file, width, height, 3, image_data, width * 3)) {
        printf("Failed to save output image %s\n", output_file);
    }

    // Free resources
    free(image_data);
    stbi_image_free(texture_data);

    return 0;
}
