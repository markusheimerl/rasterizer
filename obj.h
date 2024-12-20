#ifndef OBJ_H
#define OBJ_H

void parse_obj_file(const char *filename, 
                   double (*vertices)[3], 
                   double (*initial_vertices)[3],
                   double (*texcoords)[2],
                   int (*triangles)[3],
                   int (*texcoord_indices)[3],
                   int *num_vertices,
                   int *num_texcoords,
                   int *num_triangles) {
    FILE *file = fopen(filename, "r");
    if (!file) { 
        fprintf(stderr, "Failed to open OBJ file %s\n", filename); 
        exit(1); 
    }

    char line[256];
    while (fgets(line, sizeof(line), file)) {
        if (line[0] == 'v') {
            if (line[1] == ' ') {  // Vertex
                double x, y, z;
                sscanf(line + 2, "%lf %lf %lf", &x, &y, &z);
                vertices[*num_vertices][0] = initial_vertices[*num_vertices][0] = x;
                vertices[*num_vertices][1] = initial_vertices[*num_vertices][1] = y;
                vertices[*num_vertices][2] = initial_vertices[*num_vertices][2] = z;
                (*num_vertices)++;
            }
            else if (line[1] == 't') {  // Texture coordinate
                sscanf(line + 3, "%lf %lf", 
                       &texcoords[*num_texcoords][0], 
                       &texcoords[*num_texcoords][1]);
                (*num_texcoords)++;
            }
        }
        else if (line[0] == 'f') {  // Face
            int vi[3], ti[3] = {-1, -1, -1};
            char *format;
            int matches;

            // Try different face formats
            if (sscanf(line + 2, "%d/%d/%*d %d/%d/%*d %d/%d/%*d", 
                      &vi[0], &ti[0], &vi[1], &ti[1], &vi[2], &ti[2]) == 6 ||
                sscanf(line + 2, "%d/%d %d/%d %d/%d", 
                      &vi[0], &ti[0], &vi[1], &ti[1], &vi[2], &ti[2]) == 6) {
                // Format with texture coords handled
            }
            else if (sscanf(line + 2, "%d %d %d", &vi[0], &vi[1], &vi[2]) != 3) {
                fprintf(stderr, "Failed to parse face: %s", line);
                continue;
            }

            // Store face data
            for (int i = 0; i < 3; i++) {
                triangles[*num_triangles][i] = vi[i] - 1;
                texcoord_indices[*num_triangles][i] = (ti[i] != -1) ? ti[i] - 1 : -1;
            }
            (*num_triangles)++;
        }
    }
    fclose(file);
}

#endif /* OBJ_H */