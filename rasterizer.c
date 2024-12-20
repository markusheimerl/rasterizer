#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "gif.h"
#include "bmp.h"
#include "obj.h"
#include "mat.h"
#include "object3d.h"
#include <stdbool.h>

// gcc -O3 rasterizer.c -lm && ./a.out

#define WIDTH 640
#define HEIGHT 480
#define FRAMES 60
#define ASPECT_RATIO ((double)WIDTH / (double)HEIGHT)
#define FOV_Y 60.0
#define NEAR_PLANE 0.01
#define FAR_PLANE 100.0

// Structure to represent a vertex with all its attributes
typedef struct {
    double pos[4];     // Homogeneous coordinates (x, y, z, w)
    double uv[2];      // Texture coordinates
} Vertex;

// Helper function to interpolate between two vertices
Vertex interpolate_vertex(Vertex* v1, Vertex* v2, double t) {
    Vertex result;
    
    // Interpolate position
    for (int i = 0; i < 4; i++) {
        result.pos[i] = v1->pos[i] + t * (v2->pos[i] - v1->pos[i]);
    }
    
    // Interpolate texture coordinates
    for (int i = 0; i < 2; i++) {
        result.uv[i] = v1->uv[i] + t * (v2->uv[i] - v1->uv[i]);
    }
    
    return result;
}

// Function to clip a triangle against a plane
// Function to clip a triangle against screen space bounds
int clip_triangle_against_plane(Vertex* input, int input_count, Vertex* output,
                              int plane_index, int min_val, int max_val) {
    int output_count = 0;
    
    for (int i = 0; i < input_count; i++) {
        int curr = i;
        int next = (i + 1) % input_count;
        
        Vertex* v1 = &input[curr];
        Vertex* v2 = &input[next];
        
        // Calculate distances to the plane
        double d1 = v1->pos[plane_index];
        double d2 = v2->pos[plane_index];
        
        // Check if point is inside screen bounds
        bool v1_inside = (d1 >= min_val && d1 <= max_val);
        bool v2_inside = (d2 >= min_val && d2 <= max_val);
        
        // If first vertex is inside
        if (v1_inside) {
            output[output_count++] = *v1;
        }
        
        // If vertices are on different sides
        if (v1_inside != v2_inside) {
            double t;
            if (d1 < min_val) {
                t = (min_val - d1) / (d2 - d1);
            } else if (d1 > max_val) {
                t = (max_val - d1) / (d2 - d1);
            }
            output[output_count++] = interpolate_vertex(v1, v2, t);
        }
    }
    
    return output_count;
}

// Function to perform full clipping of a triangle
int clip_triangle(Vertex* input, Vertex* output) {
    Vertex temp[10];
    int vertex_count;
    
    // Copy input vertices to temp buffer
    for (int i = 0; i < 3; i++) {
        temp[i] = input[i];
    }
    vertex_count = 3;
    
    // Near plane clipping (Z > 0)
    for (int i = 0; i < vertex_count; i++) {
        if (temp[i].pos[2] <= 0) return 0;
    }
    
    // Clip against screen bounds
    vertex_count = clip_triangle_against_plane(temp, vertex_count, output, 0, 0, WIDTH-1);  // X
    if (vertex_count < 3) return 0;
    
    vertex_count = clip_triangle_against_plane(output, vertex_count, temp, 1, 0, HEIGHT-1); // Y
    if (vertex_count < 3) return 0;
    
    // Copy final result back to output
    for (int i = 0; i < vertex_count; i++) {
        output[i] = temp[i];
    }
    
    return vertex_count;
}

void sample_texture(double u, double v, double color[3], unsigned char *texture_data, 
                   int texture_width, int texture_height) {
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
    printf("Starting frame render\n");
    double *depth_buffer = calloc(WIDTH * HEIGHT, sizeof(double));
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        depth_buffer[i] = -DBL_MAX;
    }

    for (int obj_idx = 0; obj_idx < num_objects; obj_idx++) {
        Object3D *obj = objects[obj_idx];

for (int tri_idx = 0; tri_idx < obj->num_triangles; tri_idx++) {
    // Set up input vertices for clipping
    Vertex input_vertices[3];
for (int v = 0; v < 3; v++) {
    double* vertex = obj->transformed_vertices[obj->triangles[tri_idx][v]];
    // Store screen space coordinates directly
    input_vertices[v].pos[0] = vertex[0];
    input_vertices[v].pos[1] = vertex[1];
    input_vertices[v].pos[2] = vertex[2];
    input_vertices[v].pos[3] = 1.0;
    input_vertices[v].uv[0] = obj->texcoords[obj->texcoord_indices[tri_idx][v]][0];
    input_vertices[v].uv[1] = obj->texcoords[obj->texcoord_indices[tri_idx][v]][1];
    
    // Debug print
    printf("Vertex %d: %.2f, %.2f, %.2f\n", v, vertex[0], vertex[1], vertex[2]);
}

    // Clip the triangle
    Vertex clipped_vertices[10];
    int num_vertices = clip_triangle(input_vertices, clipped_vertices);
    printf("Triangle %d clipped into %d vertices\n", tri_idx, num_vertices);
printf("First vertex: %.2f, %.2f, %.2f\n", 
       clipped_vertices[0].pos[0], 
       clipped_vertices[0].pos[1], 
       clipped_vertices[0].pos[2]);

    // Process each triangle in the clipped polygon
    for (int i = 1; i < num_vertices - 1; i++) {
        double triangle[3][4];
        double uv[3][2];

        // Set up triangle vertices with perspective division
        for (int j = 0; j < 3; j++) {
            int idx = (j == 0) ? 0 : (i + j - 1);
            double w = clipped_vertices[idx].pos[3];
            double inv_w = 1.0 / w;
            
            // Perform perspective division
            triangle[j][0] = clipped_vertices[idx].pos[0] * inv_w;
            triangle[j][1] = clipped_vertices[idx].pos[1] * inv_w;
            triangle[j][2] = clipped_vertices[idx].pos[2] * inv_w;
            triangle[j][3] = inv_w;  // Store 1/w for perspective-correct texturing
            
            uv[j][0] = clipped_vertices[idx].uv[0];
            uv[j][1] = clipped_vertices[idx].uv[1];
        }
        

                // Calculate bounding box
                int min_x = fmax(0, floor(fmin(fmin(triangle[0][0], triangle[1][0]), triangle[2][0])));
                int min_y = fmax(0, floor(fmin(fmin(triangle[0][1], triangle[1][1]), triangle[2][1])));
                int max_x = fmin(WIDTH - 1, ceil(fmax(fmax(triangle[0][0], triangle[1][0]), triangle[2][0])));
                int max_y = fmin(HEIGHT - 1, ceil(fmax(fmax(triangle[0][1], triangle[1][1]), triangle[2][1])));

                // Calculate barycentric denominator
                double bary_denom = ((triangle[1][1] - triangle[2][1]) * (triangle[0][0] - triangle[2][0]) + 
                                   (triangle[2][0] - triangle[1][0]) * (triangle[0][1] - triangle[2][1]));

                if (fabs(bary_denom) < 1e-6) continue;

                // Rasterize triangle
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
    }
    free(depth_buffer);
}

void update_object_vertices(Object3D* obj, double view_matrix[4][4]) {
    // Create projection matrix
    const double f = 1.0 / tan(FOV_Y * M_PI / 360.0);
    double projection[4][4] = {
        {f / ASPECT_RATIO, 0, 0, 0},
        {0, f, 0, 0},
        {0, 0, (FAR_PLANE + NEAR_PLANE) / (NEAR_PLANE - FAR_PLANE), 
         (2 * FAR_PLANE * NEAR_PLANE) / (NEAR_PLANE - FAR_PLANE)},
        {0, 0, -1, 0}
    };

    // Create viewport matrix
    double viewport[4][4] = {
        {WIDTH / 2.0,  0,            0,   WIDTH / 2.0},
        {0,            HEIGHT / 2.0,  0,   HEIGHT / 2.0},
        {0,            0,            1.0,  0},
        {0,            0,            0,    1.0}
    };

    // Calculate transformation matrices
    double modelview[4][4];
    double mvp[4][4];
    double final_transform[4][4];
    
    matrix_multiply(view_matrix, obj->model_matrix, modelview);    // Model-View
    matrix_multiply(projection, modelview, mvp);                   // Model-View-Projection
    matrix_multiply(viewport, mvp, final_transform);              // Final transform including viewport

    // Transform all vertices
    for (int i = 0; i < obj->num_vertices; i++) {
        const double* vertex = obj->initial_vertices[i];
        double* transformed = obj->transformed_vertices[i];
        
        // Calculate homogeneous coordinates
        const double x = final_transform[0][0] * vertex[0] + final_transform[0][1] * vertex[1] + 
                        final_transform[0][2] * vertex[2] + final_transform[0][3];
        const double y = final_transform[1][0] * vertex[0] + final_transform[1][1] * vertex[1] + 
                        final_transform[1][2] * vertex[2] + final_transform[1][3];
        const double z = final_transform[2][0] * vertex[0] + final_transform[2][1] * vertex[1] + 
                        final_transform[2][2] * vertex[2] + final_transform[2][3];
        const double w = final_transform[3][0] * vertex[0] + final_transform[3][1] * vertex[1] + 
                        final_transform[3][2] * vertex[2] + final_transform[3][3];

        // Perform perspective division
        if (fabs(w) > 1e-6) {
            const double inv_w = 1.0 / w;
            transformed[0] = x * inv_w;
            transformed[1] = y * inv_w;
            transformed[2] = z * inv_w;
        } else {
            transformed[0] = transformed[1] = transformed[2] = 0;
        }
    }
}

int main() {
    // Initialize scene objects
    Object3D* objects[] = {
        create_object("drone.obj", "drone.bmp"),
        create_object("ground.obj", "ground.bmp")
    };
    const int num_objects = sizeof(objects) / sizeof(objects[0]);
    
    // Set up static ground object
    matrix_translate(objects[1]->model_matrix, 0.0, -0.5, 0.0);
    matrix_scale(objects[1]->model_matrix, 1.0);
    
    // Initialize rendering context
    uint8_t *frame_buffer = calloc(WIDTH * HEIGHT * 3, sizeof(uint8_t));
    ge_GIF *gif = ge_new_gif("output_rasterizer.gif", WIDTH, HEIGHT, 3, -1, 0);
    
    // Define camera setup
    double camera_pos[3] = {-2.0, 2.0, -2.0};
    double camera_target[3] = {0.0, 0.0, 0.0};
    double camera_up[3] = {0.0, 1.0, 0.0};
    
    // Animation parameters
    const double angle_per_frame = (2.0 * M_PI) / FRAMES;
    
    // Render animation frames
    for (int frame = 0; frame < FRAMES; frame++) {
        camera_pos[0] += 0.05;
        camera_pos[2] += 0.05;
        camera_target[0] += 0.05;
        camera_target[2] += 0.05;
        printf("Rendering frame %d/%d\n", frame + 1, FRAMES);
        
        // Clear frame buffer
        memset(frame_buffer, 0, WIDTH * HEIGHT * 3);
        
        // Update drone transformation for this frame
        Object3D *drone = objects[0];
        matrix_identity(drone->model_matrix);
        matrix_translate(drone->model_matrix, 0.0, 0.5, 0.0);
        matrix_scale(drone->model_matrix, 0.5);
        matrix_rotate_y(drone->model_matrix, frame * angle_per_frame);
        
        // Calculate view matrix
        double view_matrix[4][4];
        calculate_view_matrix(camera_pos, camera_target, camera_up, view_matrix);
        
        // Update and render scene
        for (int i = 0; i < num_objects; i++) {
            update_object_vertices(objects[i], view_matrix);
        }
        render_frame(frame_buffer, objects, num_objects);
        
        // Add frame to GIF
        ge_add_frame(gif, frame_buffer, 6);
    }
    
    // Cleanup resources
    ge_close_gif(gif);
    free(frame_buffer);
    for (int i = 0; i < num_objects; i++) {
        free_object(objects[i]);
    }
    
    return 0;
}