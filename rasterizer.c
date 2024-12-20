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

void sample_texture(double u, double v, double color[3], unsigned char *texture_data, 
                   int texture_width, int texture_height) {
    // Normalize UV coordinates
    u = u - floor(u);
    v = 1.0 - (v - floor(v));
    
    // Calculate texture coordinates
    int x = (int)(u * texture_width);
    int y = (int)(v * texture_height);
    x = fmin(fmax(x, 0), texture_width - 1);
    y = fmin(fmax(y, 0), texture_height - 1);
    
    // Sample color
    int idx = (y * texture_width + x) * 3;
    static const double inv255 = 1.0 / 255.0;
    for (int i = 0; i < 3; i++) {
        color[i] = texture_data[idx + i] * inv255;
    }
}

void get_triangle_bounds(double triangle[3][4], int *min_x, int *min_y, int *max_x, int *max_y) {
    *min_x = fmax(0, floor(fmin(fmin(triangle[0][0], triangle[1][0]), triangle[2][0])));
    *min_y = fmax(0, floor(fmin(fmin(triangle[0][1], triangle[1][1]), triangle[2][1])));
    *max_x = fmin(WIDTH - 1, ceil(fmax(fmax(triangle[0][0], triangle[1][0]), triangle[2][0])));
    *max_y = fmin(HEIGHT - 1, ceil(fmax(fmax(triangle[0][1], triangle[1][1]), triangle[2][1])));
}

void calculate_barycentric(double triangle[3][4], int x, int y, double *lambda0, double *lambda1) {
    double bary_denom = ((triangle[1][1] - triangle[2][1]) * (triangle[0][0] - triangle[2][0]) + 
                        (triangle[2][0] - triangle[1][0]) * (triangle[0][1] - triangle[2][1]));
    
    if (fabs(bary_denom) < 1e-6) {
        *lambda0 = *lambda1 = -1;  // Invalid coordinates
        return;
    }
    
    *lambda0 = ((triangle[1][1] - triangle[2][1]) * (x - triangle[2][0]) + 
                (triangle[2][0] - triangle[1][0]) * (y - triangle[2][1])) / bary_denom;
    *lambda1 = ((triangle[2][1] - triangle[0][1]) * (x - triangle[2][0]) + 
                (triangle[0][0] - triangle[2][0]) * (y - triangle[2][1])) / bary_denom;
}

void render_frame(uint8_t *image, Object3D **objects, int num_objects) {
    double *depth_buffer = calloc(WIDTH * HEIGHT, sizeof(double));
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        depth_buffer[i] = -DBL_MAX;
    }

    for (int obj_idx = 0; obj_idx < num_objects; obj_idx++) {
        Object3D *obj = objects[obj_idx];
        
        for (int tri_idx = 0; tri_idx < obj->num_triangles; tri_idx++) {
            double triangle[3][4];
            double uv[3][2];
            
            // Setup triangle vertices and UV coordinates
            for (int v = 0; v < 3; v++) {
                double *vertex = obj->transformed_vertices[obj->triangles[tri_idx][v]];
                memcpy(triangle[v], vertex, 3 * sizeof(double));
                triangle[v][3] = 1.0 / vertex[2];
                
                memcpy(uv[v], obj->texcoords[obj->texcoord_indices[tri_idx][v]], 2 * sizeof(double));
            }

            // Get triangle bounds
            int min_x, min_y, max_x, max_y;
            get_triangle_bounds(triangle, &min_x, &min_y, &max_x, &max_y);

            // Rasterize triangle
            for (int y = min_y; y <= max_y; y++) {
                for (int x = min_x; x <= max_x; x++) {
                    double lambda0, lambda1;
                    calculate_barycentric(triangle, x, y, &lambda0, &lambda1);
                    double lambda2 = 1.0 - lambda0 - lambda1;

                    if (lambda0 >= 0 && lambda0 <= 1 && 
                        lambda1 >= 0 && lambda1 <= 1 && 
                        lambda2 >= 0 && lambda2 <= 1) {
                        
                        double z = lambda0 * triangle[0][2] + lambda1 * triangle[1][2] + lambda2 * triangle[2][2];
                        int idx = y * WIDTH + x;
                        
                        if (z > depth_buffer[idx]) {
                            depth_buffer[idx] = z;
                            
                            // Calculate texture coordinates
                            double u = (lambda0 * uv[0][0] * triangle[0][3] + 
                                      lambda1 * uv[1][0] * triangle[1][3] + 
                                      lambda2 * uv[2][0] * triangle[2][3]) * z;
                            double v = (lambda0 * uv[0][1] * triangle[0][3] + 
                                      lambda1 * uv[1][1] * triangle[1][3] + 
                                      lambda2 * uv[2][1] * triangle[2][3]) * z;
                            
                            // Sample and write color
                            double color[3];
                            sample_texture(u, v, color, obj->texture.data, 
                                         obj->texture.width, obj->texture.height);
                            
                            for (int i = 0; i < 3; i++) {
                                image[idx * 3 + i] = (uint8_t)(color[i] * 255.0);
                            }
                        }
                    }
                }
            }
        }
    }
    free(depth_buffer);
}

void create_projection_matrix(double projection[4][4]) {
    const double f = 1.0 / tan(FOV_Y * M_PI / 360.0);
    memset(projection, 0, 16 * sizeof(double));
    projection[0][0] = f / ASPECT_RATIO;
    projection[1][1] = f;
    projection[2][2] = (FAR_PLANE + NEAR_PLANE) / (NEAR_PLANE - FAR_PLANE);
    projection[2][3] = (2 * FAR_PLANE * NEAR_PLANE) / (NEAR_PLANE - FAR_PLANE);
    projection[3][2] = -1;
}

void update_object_vertices(Object3D* obj, double view_matrix[4][4]) {
    double projection[4][4], viewport[4][4], modelview[4][4], mvp[4][4], final_transform[4][4];
    
    // Create matrices
    create_projection_matrix(projection);
    
    memset(viewport, 0, 16 * sizeof(double));
    viewport[0][0] = WIDTH / 2.0;
    viewport[1][1] = HEIGHT / 2.0;
    viewport[0][3] = WIDTH / 2.0;
    viewport[1][3] = HEIGHT / 2.0;
    viewport[2][2] = viewport[3][3] = 1.0;
    
    // Calculate transformation chain
    matrix_multiply(view_matrix, obj->model_matrix, modelview);
    matrix_multiply(projection, modelview, mvp);
    matrix_multiply(viewport, mvp, final_transform);
    
    // Transform vertices
    for (int i = 0; i < obj->num_vertices; i++) {
        const double* vertex = obj->initial_vertices[i];
        double* transformed = obj->transformed_vertices[i];
        double result[4] = {0};
        
        // Matrix multiplication
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 3; k++) {
                result[j] += final_transform[j][k] * vertex[k];
            }
            result[j] += final_transform[j][3];
        }
        
        // Perspective division
        if (fabs(result[3]) > 1e-6) {
            double inv_w = 1.0 / result[3];
            for (int j = 0; j < 3; j++) {
                transformed[j] = result[j] * inv_w;
            }
        } else {
            memset(transformed, 0, 3 * sizeof(double));
        }
    }
}

int main() {
    // Scene setup
    Object3D* objects[] = {
        create_object("drone.obj", "drone.bmp"),
        create_object("ground.obj", "ground.bmp")
    };
    const int num_objects = sizeof(objects) / sizeof(objects[0]);
    
    // Initialize ground
    matrix_translate(objects[1]->model_matrix, 0.0, -0.5, 0.0);
    matrix_scale(objects[1]->model_matrix, 1.0);
    
    // Initialize rendering
    uint8_t *frame_buffer = calloc(WIDTH * HEIGHT * 3, sizeof(uint8_t));
    ge_GIF *gif = ge_new_gif("output_rasterizer.gif", WIDTH, HEIGHT, 3, -1, 0);
    
    // Camera setup
    double camera_pos[3] = {-2.0, 1.0, -2.0};
    double camera_target[3] = {0.0, 0.0, 0.0};
    double camera_up[3] = {0.0, 1.0, 0.0};
    
    // Animation loop
    const double angle_per_frame = (2.0 * M_PI) / FRAMES;
    for (int frame = 0; frame < FRAMES; frame++) {
        printf("Rendering frame %d/%d\n", frame + 1, FRAMES);
        
        memset(frame_buffer, 0, WIDTH * HEIGHT * 3);
        
        // Update drone position
        Object3D *drone = objects[0];
        matrix_identity(drone->model_matrix);
        matrix_translate(drone->model_matrix, 0.0, 0.5, 0.0);
        matrix_scale(drone->model_matrix, 0.5);
        matrix_rotate_y(drone->model_matrix, frame * angle_per_frame);
        
        // Render frame
        double view_matrix[4][4];
        calculate_view_matrix(camera_pos, camera_target, camera_up, view_matrix);
        camera_pos[0] += 0.05;
        camera_pos[2] += 0.05;
        camera_target[0] += 0.05;
        camera_target[2] += 0.05;
        
        for (int i = 0; i < num_objects; i++) {
            update_object_vertices(objects[i], view_matrix);
        }
        render_frame(frame_buffer, objects, num_objects);
        
        ge_add_frame(gif, frame_buffer, 6);
    }
    
    // Cleanup
    ge_close_gif(gif);
    free(frame_buffer);
    for (int i = 0; i < num_objects; i++) {
        free_object(objects[i]);
    }
    
    return 0;
}