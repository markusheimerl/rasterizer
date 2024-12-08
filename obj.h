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
    if (!file) { fprintf(stderr, "Failed to open OBJ file %s\n", filename); exit(1); }

    char line[256];
    while (fgets(line, sizeof(line), file)) {
        if (strncmp(line, "v ", 2) == 0) {
            double x, y, z;
            sscanf(line + 2, "%lf %lf %lf", &x, &y, &z);
            vertices[*num_vertices][0] = x; vertices[*num_vertices][1] = y; vertices[*num_vertices][2] = z;
            initial_vertices[*num_vertices][0] = x; initial_vertices[*num_vertices][1] = y; initial_vertices[*num_vertices][2] = z;
            (*num_vertices)++;
        } else if (strncmp(line, "vt ", 3) == 0) {
            double u, v;
            sscanf(line + 3, "%lf %lf", &u, &v);
            texcoords[*num_texcoords][0] = u; texcoords[*num_texcoords][1] = v;
            (*num_texcoords)++;
        } else if (strncmp(line, "f ", 2) == 0) {
            int vi[3], ti[3];
            int matches = sscanf(line + 2, "%d/%d %d/%d %d/%d", &vi[0], &ti[0], &vi[1], &ti[1], &vi[2], &ti[2]);
            if (matches != 6) {
                matches = sscanf(line + 2, "%d/%d/%*d %d/%d/%*d %d/%d/%*d", &vi[0], &ti[0], &vi[1], &ti[1], &vi[2], &ti[2]);
                if (matches != 6) {
                    matches = sscanf(line + 2, "%d %d %d", &vi[0], &vi[1], &vi[2]);
                    if (matches != 3) { fprintf(stderr, "Failed to parse face: %s", line); continue; }
                    else { ti[0] = ti[1] = ti[2] = -1; }
                }
            }
            triangles[*num_triangles][0] = vi[0] - 1;
            triangles[*num_triangles][1] = vi[1] - 1;
            triangles[*num_triangles][2] = vi[2] - 1;
            texcoord_indices[*num_triangles][0] = (ti[0] != -1) ? ti[0] - 1 : -1;
            texcoord_indices[*num_triangles][1] = (ti[1] != -1) ? ti[1] - 1 : -1;
            texcoord_indices[*num_triangles][2] = (ti[2] != -1) ? ti[2] - 1 : -1;
            (*num_triangles)++;
        }
    }
    fclose(file);
}

#endif /* OBJ_H */