#ifndef BMP_H
#define BMP_H

unsigned char* load_bmp(const char* filename, int* width, int* height, int* channels) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Failed to open BMP file: %s\n", filename);
        return NULL;
    }

    // Read BMP header
    unsigned char file_header[14];
    unsigned char info_header[40];
    
    if (fread(file_header, 1, 14, file) != 14) {
        fprintf(stderr, "Failed to read BMP file header\n");
        fclose(file);
        return NULL;
    }
    
    if (file_header[0] != 'B' || file_header[1] != 'M') {
        fprintf(stderr, "Not a valid BMP file\n");
        fclose(file);
        return NULL;
    }

    if (fread(info_header, 1, 40, file) != 40) {
        fprintf(stderr, "Failed to read BMP info header\n");
        fclose(file);
        return NULL;
    }

    // Get image information
    *width = *(int*)&info_header[4];
    *height = abs(*(int*)&info_header[8]); // Handle both top-down and bottom-up BMPs
    int bits_per_pixel = *(short*)&info_header[14];
    int compression = *(int*)&info_header[16];
    
    if (bits_per_pixel != 24 && bits_per_pixel != 32) {
        fprintf(stderr, "Only 24-bit and 32-bit BMP files are supported (got %d-bit)\n", bits_per_pixel);
        fclose(file);
        return NULL;
    }

    int src_channels = bits_per_pixel / 8;
    *channels = 3; // We'll always output 3 channels (RGB)

    // Calculate row size with padding
    int row_size = ((*width * bits_per_pixel + 31) / 32) * 4;

    // Seek to pixel data
    int pixel_data_offset = *(int*)&file_header[10];
    fseek(file, pixel_data_offset, SEEK_SET);

    // Allocate memory for image data
    unsigned char* data = malloc(*width * *height * 3);
    if (!data) {
        fprintf(stderr, "Failed to allocate memory\n");
        fclose(file);
        return NULL;
    }

    // Read pixel data
    int is_top_down = (*(int*)&info_header[8]) < 0;
    for (int y = 0; y < *height; y++) {
        int row_idx = is_top_down ? y : (*height - 1 - y);
        unsigned char* row = &data[row_idx * *width * 3];
        
        for (int x = 0; x < *width; x++) {
            unsigned char pixel[4];
            if (fread(pixel, 1, src_channels, file) != src_channels) {
                fprintf(stderr, "Failed to read pixel data\n");
                free(data);
                fclose(file);
                return NULL;
            }
            
            // Store as RGB
            row[x * 3 + 0] = pixel[2]; // Red
            row[x * 3 + 1] = pixel[1]; // Green
            row[x * 3 + 2] = pixel[0]; // Blue
        }
        
        // Skip padding bytes
        int padding = row_size - (*width * src_channels);
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