#include <stdio.h>
#include <stdlib.h>
#include "gif.h"
#include "rasterizer.h"

#define FRAMES 120

int main() {
    // Initialize meshes
    Mesh* meshes[] = {
        create_mesh("drone.obj", "drone.bmp"),
        create_mesh("ground.obj", "ground.bmp")
    };

    // Initialize visualization buffers
    uint8_t *frame_buffer = calloc(WIDTH * HEIGHT * 3, sizeof(uint8_t));
    ge_GIF *gif = ge_new_gif("output_rasterizer.gif", WIDTH, HEIGHT, 4, -1, 0);

    // Initialize camera
    double camera_pos[3] = {-2.0, 2.0, -2.0};
    double camera_target[3] = {0.0, 0.0, 0.0};

    // Transform ground mesh
    transform_mesh(meshes[1], (double[3]){0.0, -0.5, 0.0}, 1.0, (double[9]){1,0,0, 0,1,0, 0,0,1});

    // Main simulation loop
    for(int frame = 0; frame < FRAMES; frame++) {
        transform_mesh(meshes[0], (double[3]){0.0, 1.0, 0.0}, 0.5, (double[9]){cos(frame * 0.1), 0, sin(frame * 0.1), 0, 1, 0, -sin(frame * 0.1), 0, cos(frame * 0.1)});

        // Render frame and add to GIF
        memset(frame_buffer, 0, WIDTH * HEIGHT * 3);
        vertex_shader(meshes, 2, camera_pos, camera_target);
        rasterize(frame_buffer, meshes, 2);
        ge_add_frame(gif, frame_buffer, 6);

        // Print progress
        printf("Rendered frame %d/%d\n", frame + 1, FRAMES);
    }

    // Cleanup
    free(frame_buffer);
    free_meshes(meshes, 2);
    ge_close_gif(gif);
    
    return 0;
}