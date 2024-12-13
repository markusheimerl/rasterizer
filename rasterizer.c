#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "gif.h"
#include "bmp.h"
#include "obj.h"
#include "mat.h"

// gcc -O3 rasterizer.c -lm && ./a.out

#define WIDTH 640
#define HEIGHT 480
#define FRAMES 60
#define ASPECT_RATIO ((double)WIDTH / (double)HEIGHT)
#define FOV_Y 60.0
#define NEAR_PLANE 0.1
#define FAR_PLANE 100.0

typedef struct {
    double (*vertices)[3];
    double (*initial_vertices)[3];
    double (*transformed_vertices)[3];
    double (*texcoords)[2];
    int (*triangles)[3];
    int (*texcoord_indices)[3];
    int num_vertices;
    int num_texcoords;
    int num_triangles;
    unsigned char *texture_data;
    int texture_width;
    int texture_height;
    int texture_channels;
    double model_matrix[4][4];
} Object3D;

void transform_vertex(double m[4][4], double in[3], double out[3]) {
    double temp[4] = {in[0], in[1], in[2], 1.0};
    double result[4] = {0};
    
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
            result[i] += m[i][j] * temp[j];
        }
    }
    
    out[0] = result[0] / result[3];
    out[1] = -result[1] / result[3];  // Flip Y coordinate
    out[2] = result[2] / result[3];
}

void sample_texture(double u, double v, double color[3], unsigned char *texture_data, int texture_width, int texture_height) {
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

void render_frame(uint8_t *image, Object3D **objects, int num_objects) {
    double *depth_buffer = malloc(WIDTH * HEIGHT * sizeof(double));
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        depth_buffer[i] = DBL_MAX;
    }

    for (int obj_idx = 0; obj_idx < num_objects; obj_idx++) {
        Object3D *obj = objects[obj_idx];
        
        for (int i = 0; i < obj->num_triangles; i++) {
            double verts[3][4], uv_coords[3][2];
            double f = 1.0 / tan((FOV_Y * M_PI / 180.0) / 2.0);
            double aspect = (double)WIDTH / HEIGHT;
            
            for (int j = 0; j < 3; j++) {
                double *vertex = obj->transformed_vertices[obj->triangles[i][j]];
                double z = fmax(vertex[2], NEAR_PLANE);
                verts[j][0] = (-(f / aspect) * vertex[0] / z + 1.0) * WIDTH / 2.0;
                verts[j][1] = (f * vertex[1] / z + 1.0) * HEIGHT / 2.0;
                verts[j][2] = z;
                verts[j][3] = 1.0 / z;
                uv_coords[j][0] = obj->texcoords[obj->texcoord_indices[i][j]][0];
                uv_coords[j][1] = obj->texcoords[obj->texcoord_indices[i][j]][1];
            }

            double bbox_min_x = fmin(fmin(verts[0][0], verts[1][0]), verts[2][0]);
            double bbox_min_y = fmin(fmin(verts[0][1], verts[1][1]), verts[2][1]);
            double bbox_max_x = fmax(fmax(verts[0][0], verts[1][0]), verts[2][0]);
            double bbox_max_y = fmax(fmax(verts[0][1], verts[1][1]), verts[2][1]);

            for (int x = (int)fmax(bbox_min_x, 0); x <= (int)fmin(bbox_max_x, WIDTH - 1); x++) {
                for (int y = (int)fmax(bbox_min_y, 0); y <= (int)fmin(bbox_max_y, HEIGHT - 1); y++) {
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
                        
                        double w0 = verts[0][3];
                        double w1 = verts[1][3];
                        double w2 = verts[2][3];
                        double z_interpolated = 1.0 / (lambda[0] * w0 + lambda[1] * w1 + lambda[2] * w2);
                        
                        int idx = y * WIDTH + x;
                        if (z_interpolated < depth_buffer[idx]) {
                            depth_buffer[idx] = z_interpolated;
                            
                            double u = (lambda[0] * uv_coords[0][0] * w0 + 
                                      lambda[1] * uv_coords[1][0] * w1 + 
                                      lambda[2] * uv_coords[2][0] * w2) * z_interpolated;
                            double v = (lambda[0] * uv_coords[0][1] * w0 + 
                                      lambda[1] * uv_coords[1][1] * w1 + 
                                      lambda[2] * uv_coords[2][1] * w2) * z_interpolated;
                            
                            double color[3];
                            sample_texture(u, v, color, obj->texture_data, obj->texture_width, obj->texture_height);
                            image[idx * 3] = (uint8_t)(color[0] * 255.0);
                            image[idx * 3 + 1] = (uint8_t)(color[1] * 255.0);
                            image[idx * 3 + 2] = (uint8_t)(color[2] * 255.0);
                        }
                    }
                }
            }
        }
    }
    free(depth_buffer);
}

Object3D* create_object(const char* obj_file, const char* texture_file) {
    Object3D* obj = malloc(sizeof(Object3D));
    
    obj->vertices = malloc(100000 * sizeof(*obj->vertices));
    obj->initial_vertices = malloc(100000 * sizeof(*obj->initial_vertices));
    obj->transformed_vertices = malloc(100000 * sizeof(*obj->transformed_vertices));
    obj->texcoords = malloc(100000 * sizeof(*obj->texcoords));
    obj->triangles = malloc(200000 * sizeof(*obj->triangles));
    obj->texcoord_indices = malloc(200000 * sizeof(*obj->texcoord_indices));
    obj->num_vertices = 0;
    obj->num_texcoords = 0;
    obj->num_triangles = 0;
    
    parse_obj_file(obj_file, obj->vertices, obj->initial_vertices, obj->texcoords, 
                  obj->triangles, obj->texcoord_indices, &obj->num_vertices, 
                  &obj->num_texcoords, &obj->num_triangles);

    obj->texture_data = load_bmp(texture_file, &obj->texture_width, 
                                &obj->texture_height, &obj->texture_channels);
    
    matrix_identity(obj->model_matrix);
    
    return obj;
}

void update_object_vertices(Object3D* obj) {
    for (int i = 0; i < obj->num_vertices; i++) {
        transform_vertex(obj->model_matrix, obj->initial_vertices[i], obj->transformed_vertices[i]);
    }
}

void free_object(Object3D* obj) {
    free(obj->vertices);
    free(obj->initial_vertices);
    free(obj->transformed_vertices);
    free(obj->texcoords);
    free(obj->triangles);
    free(obj->texcoord_indices);
    free(obj->texture_data);
    free(obj);
}

int main() {
    Object3D* drone = create_object("drone.obj", "drone.bmp");
    Object3D* ground = create_object("ground.obj", "ground.bmp");

    // Set up initial transformations
    matrix_translate(ground->model_matrix, 0.0, 1.5, 3.0);
    matrix_scale(ground->model_matrix, 30.0);

    Object3D* objects[] = {drone, ground};
    int num_objects = sizeof(objects) / sizeof(objects[0]);

    uint8_t *image = malloc(WIDTH * HEIGHT * 3);

    uint8_t palette[8 * 3] = {
        0x00, 0x00, 0x00, 
        0xFF, 0x00, 0x00, 
        0x00, 0xFF, 0x00, 
        0x00, 0x00, 0xFF, 
        0xFF, 0xFF, 0x00, 
        0xFF, 0x00, 0xFF, 
        0x00, 0xFF, 0xFF, 
        0xFF, 0xFF, 0xFF
    };
    
    ge_GIF *gif = ge_new_gif("output_rasterizer.gif", WIDTH, HEIGHT, palette, 3, -1, 0);
    double angle_per_frame = (2.0 * M_PI) / FRAMES;

    for (int frame_num = 0; frame_num < FRAMES; frame_num++) {
        printf("Rendering frame %d/%d\n", frame_num + 1, FRAMES);
        
        memset(image, 0, WIDTH * HEIGHT * 3);
        
        // Update drone's transformation
        matrix_identity(drone->model_matrix);
        matrix_translate(drone->model_matrix, 0.0, 1.0, 3.0);
        matrix_scale(drone->model_matrix, 1.0);
        matrix_rotate_y(drone->model_matrix, frame_num * angle_per_frame);
        
        // Update vertices for all objects
        for (int i = 0; i < num_objects; i++) {
            update_object_vertices(objects[i]);
        }
        
        render_frame(image, objects, num_objects);
        ge_add_frame(gif, image, 6);
    }

    ge_close_gif(gif);
    free(image);
    for (int i = 0; i < num_objects; i++) {
        free_object(objects[i]);
    }

    return 0;
}