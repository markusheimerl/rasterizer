#ifndef BMP_H
#define BMP_H

unsigned char* load_bmp(const char* filename, int* width, int* height, int* channels) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Failed to open BMP file: %s\n", filename);
        return NULL;
    }

    unsigned char header[54];
    if (fread(header, 1, 54, file) != 54 || header[0] != 'B' || header[1] != 'M') {
        fprintf(stderr, "Invalid BMP file format\n");
        fclose(file);
        return NULL;
    }

    // Extract image information
    *width = *(int*)&header[18];
    *height = abs(*(int*)&header[22]);
    int bits_per_pixel = *(short*)&header[28];
    int src_channels = bits_per_pixel / 8;
    *channels = 3;  // Output is always RGB

    if (bits_per_pixel != 24 && bits_per_pixel != 32) {
        fprintf(stderr, "Only 24-bit and 32-bit BMP files are supported (got %d-bit)\n", bits_per_pixel);
        fclose(file);
        return NULL;
    }

    // Calculate row size and seek to pixel data
    int row_size = ((*width * bits_per_pixel + 31) / 32) * 4;
    int pixel_offset = *(int*)&header[10];
    fseek(file, pixel_offset, SEEK_SET);

    unsigned char* data = malloc(*width * *height * 3);
    if (!data) {
        fprintf(stderr, "Failed to allocate memory\n");
        fclose(file);
        return NULL;
    }

    // Read pixel data
    int is_top_down = (*(int*)&header[22]) < 0;
    unsigned char pixel[4];
    int padding = row_size - (*width * src_channels);

    for (int y = 0; y < *height; y++) {
        int row_idx = is_top_down ? y : (*height - 1 - y);
        unsigned char* row = &data[row_idx * *width * 3];

        for (int x = 0; x < *width; x++) {
            if (fread(pixel, 1, src_channels, file) != src_channels) {
                fprintf(stderr, "Failed to read pixel data\n");
                free(data);
                fclose(file);
                return NULL;
            }
            
            row[x * 3 + 0] = pixel[2];  // Red
            row[x * 3 + 1] = pixel[1];  // Green
            row[x * 3 + 2] = pixel[0];  // Blue
        }
        
        if (padding > 0) {
            fseek(file, padding, SEEK_CUR);
        }
    }

    fclose(file);
    printf("Successfully loaded BMP: %dx%d pixels, %d-bit source, %d channels output\n", 
           *width, *height, bits_per_pixel, *channels);
    
    return data;
}

#endif /* BMP_H */