#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "gif.h"
#include "bmp.h"
#include "obj.h"
#include "mat.h"
#include "object3d.h"

// gcc -O3 rasterizer.c -lm && ./a.out

#define WIDTH 640
#define HEIGHT 480
#define FRAMES 60
#define ASPECT_RATIO ((double)WIDTH / (double)HEIGHT)
#define FOV_Y 60.0
#define NEAR_PLANE 0.1
#define FAR_PLANE 100.0

void sample_texture(double u, double v, double color[3], unsigned char *texture_data, int texture_width, int texture_height) {
    u = fmod(u, 1.0); if (u < 0) u += 1.0;
    v = fmod(v, 1.0); if (v < 0) v += 1.0;
    u = fmin(fmax(u, 0.0), 1.0);
    v = 1.0 - fmin(fmax(v, 0.0), 1.0);
    int x = (int)(u * (texture_width - 1));
    int y = (int)(v * (texture_height - 1));
    int idx = (y * texture_width + x) * 3;
    color[0] = texture_data[idx] / 255.0;
    color[1] = texture_data[idx + 1] / 255.0;
    color[2] = texture_data[idx + 2] / 255.0;
}

void render_frame(uint8_t *image, Object3D **objects, int num_objects) {
    double *depth_buffer = calloc(WIDTH * HEIGHT, sizeof(double));
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        depth_buffer[i] = DBL_MAX;
    }
    for (int obj_idx = 0; obj_idx < num_objects; obj_idx++) {
        Object3D *obj = objects[obj_idx];

        for (int tri_idx = 0; tri_idx < obj->num_triangles; tri_idx++) {
            double triangle[3][4];
            double uv[3][2];
            
            for (int v = 0; v < 3; v++) {
                double *vertex = obj->transformed_vertices[obj->triangles[tri_idx][v]];
                triangle[v][0] = vertex[0];
                triangle[v][1] = vertex[1];
                triangle[v][2] = vertex[2];
                triangle[v][3] = 1.0 / vertex[2];
                
                uv[v][0] = obj->texcoords[obj->texcoord_indices[tri_idx][v]][0];
                uv[v][1] = obj->texcoords[obj->texcoord_indices[tri_idx][v]][1];
            }

            int min_x = fmax(0, floor(fmin(fmin(triangle[0][0], triangle[1][0]), triangle[2][0])));
            int min_y = fmax(0, floor(fmin(fmin(triangle[0][1], triangle[1][1]), triangle[2][1])));
            int max_x = fmin(WIDTH - 1, ceil(fmax(fmax(triangle[0][0], triangle[1][0]), triangle[2][0])));
            int max_y = fmin(HEIGHT - 1, ceil(fmax(fmax(triangle[0][1], triangle[1][1]), triangle[2][1])));

            double bary_denom = ((triangle[1][1] - triangle[2][1]) * (triangle[0][0] - triangle[2][0]) + 
                               (triangle[2][0] - triangle[1][0]) * (triangle[0][1] - triangle[2][1]));
            
            if (fabs(bary_denom) < 1e-6) continue;

            for (int y = min_y; y <= max_y; y++) {
                for (int x = min_x; x <= max_x; x++) {
                    double lambda0 = ((triangle[1][1] - triangle[2][1]) * (x - triangle[2][0]) + 
                                    (triangle[2][0] - triangle[1][0]) * (y - triangle[2][1])) / bary_denom;
                    double lambda1 = ((triangle[2][1] - triangle[0][1]) * (x - triangle[2][0]) + 
                                    (triangle[0][0] - triangle[2][0]) * (y - triangle[2][1])) / bary_denom;
                    double lambda2 = 1.0 - lambda0 - lambda1;

                    if (lambda0 >= 0 && lambda0 <= 1 && 
                        lambda1 >= 0 && lambda1 <= 1 && 
                        lambda2 >= 0 && lambda2 <= 1) {
                        
                        double z = 1.0 / (lambda0 * triangle[0][3] + 
                                        lambda1 * triangle[1][3] + 
                                        lambda2 * triangle[2][3]);
                        
                        int idx = y * WIDTH + x;
                        if (z < depth_buffer[idx]) {
                            depth_buffer[idx] = z;
                            
                            double u = (lambda0 * uv[0][0] * triangle[0][3] + 
                                      lambda1 * uv[1][0] * triangle[1][3] + 
                                      lambda2 * uv[2][0] * triangle[2][3]) * z;
                            double v = (lambda0 * uv[0][1] * triangle[0][3] + 
                                      lambda1 * uv[1][1] * triangle[1][3] + 
                                      lambda2 * uv[2][1] * triangle[2][3]) * z;
                            
                            double color[3];
                            sample_texture(u, v, color, obj->texture_data, 
                                         obj->texture_width, obj->texture_height);
                            
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

void update_object_vertices(Object3D* obj) {
    // Convert FOV to a scaling factor
    double f = 1.0 / tan((FOV_Y * M_PI / 180.0) / 2.0);
    
    for (int i = 0; i < obj->num_vertices; i++) {
        double x = obj->initial_vertices[i][0];
        double y = obj->initial_vertices[i][1];
        double z = obj->initial_vertices[i][2];
        
        // Apply model transformation (stored in model_matrix)
        double tx = obj->model_matrix[0][0] * x + obj->model_matrix[0][1] * y + 
                   obj->model_matrix[0][2] * z + obj->model_matrix[0][3];
        double ty = obj->model_matrix[1][0] * x + obj->model_matrix[1][1] * y + 
                   obj->model_matrix[1][2] * z + obj->model_matrix[1][3];
        double tz = obj->model_matrix[2][0] * x + obj->model_matrix[2][1] * y + 
                   obj->model_matrix[2][2] * z + obj->model_matrix[2][3];
        
        // Ensure minimum z distance
        tz = fmax(tz, NEAR_PLANE);
        
        // Simple perspective projection
        double px = -f * tx / tz;
        double py = -f * ty / tz;
        
        // Convert to screen coordinates
        obj->transformed_vertices[i][0] = (px / ASPECT_RATIO + 1.0) * WIDTH / 2.0;
        obj->transformed_vertices[i][1] = (py + 1.0) * HEIGHT / 2.0;
        obj->transformed_vertices[i][2] = tz;
    }
}

int main() {
    Object3D* drone = create_object("drone.obj", "drone.bmp");
    Object3D* ground = create_object("ground.obj", "ground.bmp");

    // Set up initial transformations
    matrix_translate(ground->model_matrix, 0.0, -1.0, 3.0);
    matrix_scale(ground->model_matrix, 30.0);

    Object3D* objects[] = {drone, ground};
    int num_objects = sizeof(objects) / sizeof(objects[0]);

    uint8_t *image = malloc(WIDTH * HEIGHT * 3);

    ge_GIF *gif = ge_new_gif("output_rasterizer.gif", WIDTH, HEIGHT, 3, -1, 0);
    double angle_per_frame = (2.0 * M_PI) / FRAMES;

    for (int frame_num = 0; frame_num < FRAMES; frame_num++) {
        printf("Rendering frame %d/%d\n", frame_num + 1, FRAMES);
        
        memset(image, 0, WIDTH * HEIGHT * 3);
        
        // Update drone's transformation
        matrix_identity(drone->model_matrix);
        matrix_translate(drone->model_matrix, 0.0, 0.5, 3.0);
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