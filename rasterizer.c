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
    u = u - floor(u);
    v = 1.0 - (v - floor(v));
    int x = (int)(u * texture_width);
    int y = (int)(v * texture_height);
    x = (x < 0) ? 0 : (x >= texture_width ? texture_width - 1 : x);
    y = (y < 0) ? 0 : (y >= texture_height ? texture_height - 1 : y);
    int idx = (y * texture_width + x) * 3;
    static const double inv255 = 1.0 / 255.0;
    color[0] = texture_data[idx] * inv255;
    color[1] = texture_data[idx + 1] * inv255;
    color[2] = texture_data[idx + 2] * inv255;
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
                        
                        double z = lambda0 * triangle[0][2] + lambda1 * triangle[1][2] + lambda2 * triangle[2][2];
                        
                        int idx = y * WIDTH + x;
                        if (z > depth_buffer[idx]) {
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

void update_object_vertices(Object3D* obj, double view_matrix[4][4]) {
    double f = 1.0 / tan(FOV_Y * M_PI / 360.0);
    double projection[4][4] = {
        {f / ASPECT_RATIO, 0, 0, 0},
        {0, f, 0, 0},
        {0, 0, (FAR_PLANE + NEAR_PLANE) / (NEAR_PLANE - FAR_PLANE), 
            (2 * FAR_PLANE * NEAR_PLANE) / (NEAR_PLANE - FAR_PLANE)},
        {0, 0, -1, 0}
    };

    double viewport[4][4] = {
        {WIDTH / 2.0, 0, 0, WIDTH / 2.0},
        {0, HEIGHT / 2.0, 0, HEIGHT / 2.0},
        {0, 0, 1.0, 0},
        {0, 0, 0, 1.0}
    };

    // Compute model-view matrix
    double modelview[4][4];
    matrix_multiply(view_matrix, obj->model_matrix, modelview);

    // Compute final transformation matrix (MVP * Viewport)
    double mvp[4][4], final[4][4];
    matrix_multiply(projection, modelview, mvp);
    matrix_multiply(viewport, mvp, final);

    // Transform all vertices
    for (int i = 0; i < obj->num_vertices; i++) {
        const double* vertex = obj->initial_vertices[i];
        double* transformed = obj->transformed_vertices[i];
        
        // Apply transformation
        double x = final[0][0] * vertex[0] + final[0][1] * vertex[1] + 
                  final[0][2] * vertex[2] + final[0][3];
        double y = final[1][0] * vertex[0] + final[1][1] * vertex[1] + 
                  final[1][2] * vertex[2] + final[1][3];
        double z = final[2][0] * vertex[0] + final[2][1] * vertex[1] + 
                  final[2][2] * vertex[2] + final[2][3];
        double w = final[3][0] * vertex[0] + final[3][1] * vertex[1] + 
                  final[3][2] * vertex[2] + final[3][3];
        
        if (fabs(w) > 1e-6) {
            double inv_w = 1.0 / w;
            transformed[0] = x * inv_w;
            transformed[1] = y * inv_w;
            transformed[2] = z * inv_w;
        } else {
            transformed[0] = transformed[1] = transformed[2] = 0;
        }
    }
}

int main() {
    // Initialize objects
    Object3D* objects[] = {
        create_object("drone.obj", "drone.bmp"),
        create_object("ground.obj", "ground.bmp")
    };
    const int num_objects = sizeof(objects) / sizeof(objects[0]);
    
    // Set up initial ground transformation
    matrix_translate(objects[1]->model_matrix, 0.0, -0.5, 2.0);
    matrix_scale(objects[1]->model_matrix, 1.0);
    
    // Initialize rendering resources
    uint8_t *image = calloc(WIDTH * HEIGHT * 3, sizeof(uint8_t));
    ge_GIF *gif = ge_new_gif("output_rasterizer.gif", WIDTH, HEIGHT, 3, -1, 0);
    
    // Animation parameters
    const double angle_per_frame = (2.0 * M_PI) / FRAMES;
    
    // Camera parameters
    double camera_pos[3] = {0.0, 0.0, 0.0};  // Position the camera above and behind
    double camera_target[3] = {0.0, 0.0, 1.0}; // Look at the center of the scene
    double camera_up[3] = {0.0, 1.0, 0.0};    // Up vector
    
    // Render animation frames
    for (int frame = 0; frame < FRAMES; frame++) {
        printf("Rendering frame %d/%d\n", frame + 1, FRAMES);
        
        // Clear frame buffer
        memset(image, 0, WIDTH * HEIGHT * 3);
        
        // Update drone transformation
        Object3D *drone = objects[0];
        matrix_identity(drone->model_matrix);
        matrix_translate(drone->model_matrix, 0.0, 0.5, 2.0);
        matrix_scale(drone->model_matrix, 1.0);
        matrix_rotate_y(drone->model_matrix, frame * angle_per_frame);
        
        // Calculate view matrix
        double view_matrix[4][4];
   
        // Create look-at matrix
        double z_axis[3] = {
            camera_target[0] - camera_pos[0],
            camera_target[1] - camera_pos[1],
            camera_target[2] - camera_pos[2]
        };
        normalize_vector(z_axis);
        
        double x_axis[3];
        cross_product(camera_up, z_axis, x_axis);
        normalize_vector(x_axis);
        
        double y_axis[3];
        cross_product(z_axis, x_axis, y_axis);
        
        // Build view matrix
        view_matrix[0][0] = x_axis[0];
        view_matrix[0][1] = x_axis[1];
        view_matrix[0][2] = x_axis[2];
        view_matrix[0][3] = -dot_product(x_axis, camera_pos);
        
        view_matrix[1][0] = y_axis[0];
        view_matrix[1][1] = y_axis[1];
        view_matrix[1][2] = y_axis[2];
        view_matrix[1][3] = -dot_product(y_axis, camera_pos);
        
        view_matrix[2][0] = z_axis[0];
        view_matrix[2][1] = z_axis[1];
        view_matrix[2][2] = z_axis[2];
        view_matrix[2][3] = -dot_product(z_axis, camera_pos);
        
        view_matrix[3][0] = 0.0;
        view_matrix[3][1] = 0.0;
        view_matrix[3][2] = 0.0;
        view_matrix[3][3] = 1.0;
        
        // Update and render all objects
        for (int i = 0; i < num_objects; i++) {
            update_object_vertices(objects[i], view_matrix);
        }
        render_frame(image, objects, num_objects);
        
        // Add frame to GIF
        ge_add_frame(gif, image, 6);
    }
    
    // Cleanup
    ge_close_gif(gif);
    free(image);
    for (int i = 0; i < num_objects; i++) {
        free_object(objects[i]);
    }
    
    return 0;
}