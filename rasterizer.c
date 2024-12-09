#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "gif.h"
#include "bmp.h"
#include "obj.h"

// gcc -O3 rasterizer.c -lm && ./a.out

// Constants
#define WIDTH 640
#define HEIGHT 480
#define FRAMES 60
#define ASPECT_RATIO ((double)WIDTH / (double)HEIGHT)
#define FOV_Y 60.0
#define NEAR_PLANE 0.1
#define FAR_PLANE 100.0

// Vector macros
#define VEC_SUB(a,b,r) { (r)[0]=(a)[0]-(b)[0]; (r)[1]=(a)[1]-(b)[1]; (r)[2]=(a)[2]-(b)[2]; }
#define VEC_DOT(a,b) ((a)[0]*(b)[0]+(a)[1]*(b)[1]+(a)[2]*(b)[2])
#define VEC_CROSS(a,b,r) { (r)[0]=(a)[1]*(b)[2]-(a)[2]*(b)[1]; (r)[1]=(a)[2]*(b)[0]-(a)[0]*(b)[2]; (r)[2]=(a)[0]*(b)[1]-(a)[1]*(b)[0]; }
#define VEC_NORM(v) { double l=sqrt(VEC_DOT(v,v)); if(l>0){ (v)[0]/=l; (v)[1]/=l; (v)[2]/=l; } }
#define VEC_SCALE(v,s,r) { (r)[0]=(v)[0]*(s); (r)[1]=(v)[1]*(s); (r)[2]=(v)[2]*(s); }
#define VEC_ADD(a,b,r) { (r)[0]=(a)[0]+(b)[0]; (r)[1]=(a)[1]+(b)[1]; (r)[2]=(a)[2]+(b)[2]; }

// Rotate point around Y-axis
void rotate_y(double angle, double point[3]) {
    double s = sin(angle), c = cos(angle), x = point[0], z = point[2];
    point[0] = c * x + s * z; point[2] = -s * x + c * z;
}

// Sample texture
void sample_texture(double u, double v, double color[3], unsigned char *texture_data, int texture_width, int texture_height) {
    // Clamp and wrap UV coordinates
    u = fmod(u, 1.0); if (u < 0) u += 1.0;
    v = fmod(v, 1.0); if (v < 0) v += 1.0;

    u = fmin(fmax(u, 0.0), 1.0);
    v = 1.0 - fmin(fmax(v, 0.0), 1.0);

    int x = (int)(u * (texture_width - 1));
    int y = (int)(v * (texture_height - 1));

    int idx = (y * texture_width + x) * 3;

    if (idx < 0 || idx >= texture_width * texture_height * 3) {
        fprintf(stderr, "Texture index out of bounds: %d\n", idx);
        return;
    }

    color[0] = texture_data[idx] / 255.0;
    color[1] = texture_data[idx + 1] / 255.0;
    color[2] = texture_data[idx + 2] / 255.0;
}

void draw_triangle(double *image, const double pts[3][4], const double uv[3][2], unsigned char *texture_data, int texture_width, int texture_height) {
    double bbox_min_x = fmin(fmin(pts[0][0], pts[1][0]), pts[2][0]);
    double bbox_min_y = fmin(fmin(pts[0][1], pts[1][1]), pts[2][1]);
    double bbox_max_x = fmax(fmax(pts[0][0], pts[1][0]), pts[2][0]);
    double bbox_max_y = fmax(fmax(pts[0][1], pts[1][1]), pts[2][1]);

    for (int x = (int)fmax(bbox_min_x, 0); x <= (int)fmin(bbox_max_x, WIDTH - 1); x++) {
        for (int y = (int)fmax(bbox_min_y, 0); y <= (int)fmin(bbox_max_y, HEIGHT - 1); y++) {
            double lambda[3];
            double denominator = ((pts[1][1] - pts[2][1]) * (pts[0][0] - pts[2][0]) + 
                                (pts[2][0] - pts[1][0]) * (pts[0][1] - pts[2][1]));
            lambda[0] = ((pts[1][1] - pts[2][1]) * (x - pts[2][0]) + 
                        (pts[2][0] - pts[1][0]) * (y - pts[2][1])) / denominator;
            lambda[1] = ((pts[2][1] - pts[0][1]) * (x - pts[2][0]) + 
                        (pts[0][0] - pts[2][0]) * (y - pts[2][1])) / denominator;
            lambda[2] = 1.0 - lambda[0] - lambda[1];

            if (lambda[0] >= 0 && lambda[0] <= 1 && lambda[1] >= 0 && lambda[1] <= 1 && lambda[2] >= 0 && lambda[2] <= 1) {
                double z = lambda[0] * pts[0][2] + lambda[1] * pts[1][2] + lambda[2] * pts[2][2];
                int idx = y * WIDTH + x;
                if (z < image[idx * 4 + 3]) {
                    image[idx * 4 + 3] = z;
                    double u = lambda[0] * uv[0][0] + lambda[1] * uv[1][0] + lambda[2] * uv[2][0];
                    double v = lambda[0] * uv[0][1] + lambda[1] * uv[1][1] + lambda[2] * uv[2][1];
                    double color[3];
                    sample_texture(u, v, color, texture_data, texture_width, texture_height);
                    image[idx * 4] = color[0];
                    image[idx * 4 + 1] = color[1];
                    image[idx * 4 + 2] = color[2];
                }
            }
        }
    }
}

void render_frame(uint8_t *image, double (*vertices)[3], double (*texcoords)[2], int (*triangles)[3], int (*texcoord_indices)[3], int num_triangles, unsigned char *texture_data, int texture_width, int texture_height) {
    // Allocate depth buffer
    double *depth_buffer = malloc(WIDTH * HEIGHT * sizeof(double));
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        depth_buffer[i] = DBL_MAX;
    }
    
    // Clear image buffer
    memset(image, 0, WIDTH * HEIGHT * 3);

    // Render triangles
    for (int i = 0; i < num_triangles; i++) {
        double verts[3][4], uv_coords[3][2];
        double f = 1.0 / tan((FOV_Y * M_PI / 180.0) / 2.0);
        double aspect = (double)WIDTH / HEIGHT;
        
        // Project vertices
        for (int j = 0; j < 3; j++) {
            double *vertex = vertices[triangles[i][j]];
            double z = fmax(vertex[2], NEAR_PLANE);
            verts[j][0] = (-(f / aspect) * vertex[0] / z + 1.0) * WIDTH / 2.0;
            verts[j][1] = (f * vertex[1] / z + 1.0) * HEIGHT / 2.0;
            verts[j][2] = z;
            verts[j][3] = 1.0 / z;  // Store 1/z for perspective correction
            uv_coords[j][0] = texcoords[texcoord_indices[i][j]][0];
            uv_coords[j][1] = texcoords[texcoord_indices[i][j]][1];
        }
        
        // Calculate bounding box
        double bbox_min_x = fmin(fmin(verts[0][0], verts[1][0]), verts[2][0]);
        double bbox_min_y = fmin(fmin(verts[0][1], verts[1][1]), verts[2][1]);
        double bbox_max_x = fmax(fmax(verts[0][0], verts[1][0]), verts[2][0]);
        double bbox_max_y = fmax(fmax(verts[0][1], verts[1][1]), verts[2][1]);

        // Rasterize
        for (int x = (int)fmax(bbox_min_x, 0); x <= (int)fmin(bbox_max_x, WIDTH - 1); x++) {
            for (int y = (int)fmax(bbox_min_y, 0); y <= (int)fmin(bbox_max_y, HEIGHT - 1); y++) {
                // Calculate barycentric coordinates
                double lambda[3];
                double denominator = ((verts[1][1] - verts[2][1]) * (verts[0][0] - verts[2][0]) + 
                                    (verts[2][0] - verts[1][0]) * (verts[0][1] - verts[2][1]));
                
                lambda[0] = ((verts[1][1] - verts[2][1]) * (x - verts[2][0]) + 
                           (verts[2][0] - verts[1][0]) * (y - verts[2][1])) / denominator;
                lambda[1] = ((verts[2][1] - verts[0][1]) * (x - verts[2][0]) + 
                           (verts[0][0] - verts[2][0]) * (y - verts[2][1])) / denominator;
                lambda[2] = 1.0 - lambda[0] - lambda[1];

                if (lambda[0] >= 0 && lambda[0] <= 1 && 
                    lambda[1] >= 0 && lambda[1] <= 1 && 
                    lambda[2] >= 0 && lambda[2] <= 1) {
                    
                    // Perspective-correct interpolation
                    double w0 = verts[0][3];  // 1/z values
                    double w1 = verts[1][3];
                    double w2 = verts[2][3];
                    
                    // Calculate interpolated depth
                    double z_interpolated = 1.0 / (lambda[0] * w0 + lambda[1] * w1 + lambda[2] * w2);
                    
                    int idx = y * WIDTH + x;
                    if (z_interpolated < depth_buffer[idx]) {
                        depth_buffer[idx] = z_interpolated;
                        
                        // Perspective-correct texture coordinate interpolation
                        double u = (lambda[0] * uv_coords[0][0] * w0 + 
                                  lambda[1] * uv_coords[1][0] * w1 + 
                                  lambda[2] * uv_coords[2][0] * w2) * z_interpolated;
                        double v = (lambda[0] * uv_coords[0][1] * w0 + 
                                  lambda[1] * uv_coords[1][1] * w1 + 
                                  lambda[2] * uv_coords[2][1] * w2) * z_interpolated;
                        
                        double color[3];
                        sample_texture(u, v, color, texture_data, texture_width, texture_height);
                        image[idx * 3] = (uint8_t)(color[0] * 255.0);
                        image[idx * 3 + 1] = (uint8_t)(color[1] * 255.0);
                        image[idx * 3 + 2] = (uint8_t)(color[2] * 255.0);
                    }
                }
            }
        }
    }
    
    free(depth_buffer);
}

int main() {
    // Declare arrays with fixed sizes
    double (*vertices)[3] = malloc(100000 * sizeof(*vertices));
    double (*initial_vertices)[3] = malloc(100000 * sizeof(*initial_vertices));
    double (*transformed_vertices)[3] = malloc(100000 * sizeof(*transformed_vertices));
    double (*texcoords)[2] = malloc(100000 * sizeof(*texcoords));
    int (*triangles)[3] = malloc(200000 * sizeof(*triangles));
    int (*texcoord_indices)[3] = malloc(200000 * sizeof(*texcoord_indices));
    
    if (!vertices || !initial_vertices || !transformed_vertices || !texcoords || !triangles || !texcoord_indices) {
        fprintf(stderr, "Failed to allocate memory for mesh data\n");
        exit(1);
    }

    int num_vertices = 0, num_texcoords = 0, num_triangles = 0;
    unsigned char *texture_data = NULL;
    int texture_width = 0, texture_height = 0, texture_channels = 0;

    // Parse OBJ file
    parse_obj_file("drone.obj", vertices, initial_vertices, texcoords, triangles, texcoord_indices, &num_vertices, &num_texcoords, &num_triangles);

    // Load texture
    texture_data = load_bmp("drone.bmp", &texture_width, &texture_height, &texture_channels);

    // Allocate image buffer
    uint8_t *image = malloc(WIDTH * HEIGHT * 3);

    uint8_t palette[8 * 3] = {0x00, 0x00, 0x00, 0xFF, 0x00, 0x00, 0x00, 0xFF, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    ge_GIF *gif = ge_new_gif("output_rasterizer.gif", WIDTH, HEIGHT, palette, 3, -1, 0);

    double scale_factor = 1.0;
    double translation[3] = {0, 1, 3};
    double angle_per_frame = (2.0 * M_PI) / FRAMES;

    // Render frames
    for (int frame_num = 0; frame_num < FRAMES; frame_num++) {
        printf("Rendering frame %d/%d\n", frame_num + 1, FRAMES);
        
        // Transform vertices for this frame
        for (int i = 0; i < num_vertices; i++) {
            // Scale
            transformed_vertices[i][0] = initial_vertices[i][0] * scale_factor;
            transformed_vertices[i][1] = -initial_vertices[i][1] * scale_factor;
            transformed_vertices[i][2] = initial_vertices[i][2] * scale_factor;
            
            // Rotate
            rotate_y(frame_num * angle_per_frame, transformed_vertices[i]);
            
            // Translate
            transformed_vertices[i][0] += translation[0];
            transformed_vertices[i][1] += translation[1];
            transformed_vertices[i][2] += translation[2];
        }
        
        // Render with transformed vertices
        render_frame(image, transformed_vertices, texcoords, triangles, texcoord_indices, num_triangles, texture_data, texture_width, texture_height);
        ge_add_frame(gif, image, 6);
    }

    // Cleanup
    ge_close_gif(gif);
    free(image);
    free(texture_data);
    free(vertices);
    free(initial_vertices);
    free(transformed_vertices);
    free(texcoords);
    free(triangles);
    free(texcoord_indices);

    return 0;
}