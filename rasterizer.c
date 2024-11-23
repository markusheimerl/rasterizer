#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>

// gcc -O3 -fopenmp rasterizer.c -lm -lavformat -lavcodec -lavutil -lswscale -lswresample -lpthread

// Include stb_image for image loading
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Include FFmpeg headers
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>

// Constants
#define WIDTH 640
#define HEIGHT 480
#define N_FRAMES 60

// Maximum counts
#define MAX_VERTICES 100000
#define MAX_TEXCOORDS 100000
#define MAX_TRIANGLES 200000

// Vector macros
#define VEC_SUB(a,b,r) { (r)[0]=(a)[0]-(b)[0]; (r)[1]=(a)[1]-(b)[1]; (r)[2]=(a)[2]-(b)[2]; }
#define VEC_DOT(a,b) ((a)[0]*(b)[0]+(a)[1]*(b)[1]+(a)[2]*(b)[2])
#define VEC_CROSS(a,b,r) { \
    (r)[0]=(a)[1]*(b)[2]-(a)[2]*(b)[1]; \
    (r)[1]=(a)[2]*(b)[0]-(a)[0]*(b)[2]; \
    (r)[2]=(a)[0]*(b)[1]-(a)[1]*(b)[0]; \
}
#define VEC_NORM(v) { double l=sqrt(VEC_DOT(v,v)); if(l>0){ (v)[0]/=l; (v)[1]/=l; (v)[2]/=l; } }
#define VEC_SCALE(v,s,r) { (r)[0]=(v)[0]*(s); (r)[1]=(v)[1]*(s); (r)[2]=(v)[2]*(s); }
#define VEC_ADD(a,b,r) { (r)[0]=(a)[0]+(b)[0]; (r)[1]=(a)[1]+(b)[1]; (r)[2]=(a)[2]+(b)[2]; }

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Global variables
double vertices[MAX_VERTICES][3];
double initial_vertices[MAX_VERTICES][3];
double texcoords[MAX_TEXCOORDS][2];
int triangles[MAX_TRIANGLES][3];
int texcoord_indices[MAX_TRIANGLES][3];
int num_vertices = 0, num_texcoords = 0, num_triangles = 0;

// Z-buffer
double *zbuffer;

// Texture data
unsigned char *texture_data = NULL;
int texture_width, texture_height, texture_channels;

// Function to parse OBJ file
void parse_obj_file(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) { fprintf(stderr, "Failed to open OBJ file %s\n", filename); exit(1); }

    char line[256];
    while (fgets(line, sizeof(line), file)) {
        if (strncmp(line, "v ", 2) == 0) {
            double x, y, z;
            sscanf(line + 2, "%lf %lf %lf", &x, &y, &z);
            vertices[num_vertices][0] = x;
            vertices[num_vertices][1] = y;
            vertices[num_vertices][2] = z;
            initial_vertices[num_vertices][0] = x;
            initial_vertices[num_vertices][1] = y;
            initial_vertices[num_vertices][2] = z;
            num_vertices++;
        } else if (strncmp(line, "vt ", 3) == 0) {
            double u, v;
            sscanf(line + 3, "%lf %lf", &u, &v);
            texcoords[num_texcoords][0] = u;
            texcoords[num_texcoords][1] = v;
            num_texcoords++;
        } else if (strncmp(line, "f ", 2) == 0) {
            int vi[3], ti[3];
            sscanf(line + 2, "%d/%d %d/%d %d/%d", 
                   &vi[0], &ti[0], &vi[1], &ti[1], &vi[2], &ti[2]);
            triangles[num_triangles][0] = vi[0] - 1;
            triangles[num_triangles][1] = vi[1] - 1;
            triangles[num_triangles][2] = vi[2] - 1;
            texcoord_indices[num_triangles][0] = ti[0] - 1;
            texcoord_indices[num_triangles][1] = ti[1] - 1;
            texcoord_indices[num_triangles][2] = ti[2] - 1;
            num_triangles++;
        }
    }
    fclose(file);
    printf("Loaded: %d vertices, %d texcoords, %d triangles\n", 
           num_vertices, num_texcoords, num_triangles);
}

// Matrix operations
void matrix_multiply(double m[4][4], double v[4], double result[4]) {
    for (int i = 0; i < 4; i++) {
        result[i] = 0;
        for (int j = 0; j < 4; j++) {
            result[i] += m[i][j] * v[j];
        }
    }
}

// Perspective projection matrix
void perspective_matrix(double fov, double aspect, double near, double far, double matrix[4][4]) {
    double f = 1.0 / tan(fov * M_PI / 360.0);
    memset(matrix, 0, 16 * sizeof(double));
    matrix[0][0] = f / aspect;
    matrix[1][1] = f;
    matrix[2][2] = (far + near) / (near - far);
    matrix[2][3] = (2 * far * near) / (near - far);
    matrix[3][2] = -1;
}

// Viewport transformation
void viewport_transform(double x, double y, double w, double h, double depth, double matrix[4][4]) {
    memset(matrix, 0, 16 * sizeof(double));
    matrix[0][0] = w/2.0;
    matrix[0][3] = x + w/2.0;
    matrix[1][1] = h/2.0;
    matrix[1][3] = y + h/2.0;
    matrix[2][2] = depth/2.0;
    matrix[2][3] = depth/2.0;
    matrix[3][3] = 1;
}

// Barycentric coordinates
void barycentric(int x, int y, double *v0, double *v1, double *v2, double *coords) {
    double v0x = v0[0], v0y = v0[1];
    double v1x = v1[0], v1y = v1[1];
    double v2x = v2[0], v2y = v2[1];
    
    double denom = ((v1y - v2y) * (v0x - v2x) + (v2x - v1x) * (v0y - v2y));
    if (fabs(denom) < 1e-10) {
        coords[0] = coords[1] = coords[2] = -1;
        return;
    }
    
    coords[0] = ((v1y - v2y) * (x - v2x) + (v2x - v1x) * (y - v2y)) / denom;
    coords[1] = ((v2y - v0y) * (x - v2x) + (v0x - v2x) * (y - v2y)) / denom;
    coords[2] = 1 - coords[0] - coords[1];
}

// Rotate point around Y-axis
void rotate_y(double angle, double point[3]) {
    double s = sin(angle), c = cos(angle);
    double x = point[0], z = point[2];
    point[0] = c * x + s * z;
    point[2] = -s * x + c * z;
}

// Draw triangle with lighting
void draw_triangle(double *v0, double *v1, double *v2, 
                  double *t0, double *t1, double *t2,
                  unsigned char *image) {
    // Compute triangle normal for lighting
    double edge1[3], edge2[3], normal[3];
    VEC_SUB(v1, v0, edge1);
    VEC_SUB(v2, v0, edge2);
    VEC_CROSS(edge1, edge2, normal);
    VEC_NORM(normal);
    
    // Simple directional light from front
    double light_dir[3] = {0, 0, 1};
    VEC_NORM(light_dir);
    double light_intensity = fmax(0.4, VEC_DOT(normal, light_dir)) * 2.0; // Increased brightness

    // Compute bounding box
    int minX = (int)fmax(0, fmin(fmin(v0[0], v1[0]), v2[0]));
    int maxX = (int)fmin(WIDTH-1, fmax(fmax(v0[0], v1[0]), v2[0]));
    int minY = (int)fmax(0, fmin(fmin(v0[1], v1[1]), v2[1]));
    int maxY = (int)fmin(HEIGHT-1, fmax(fmax(v0[1], v1[1]), v2[1]));

    // Rasterize
    for (int y = minY; y <= maxY; y++) {
        for (int x = minX; x <= maxX; x++) {
            double coords[3];
            barycentric(x, y, v0, v1, v2, coords);
            
            if (coords[0] >= 0 && coords[1] >= 0 && coords[2] >= 0) {
                // Interpolate Z
                double z = coords[0]*v0[2] + coords[1]*v1[2] + coords[2]*v2[2];
                int idx = y*WIDTH + x;
                
                if (z < zbuffer[idx]) {
                    zbuffer[idx] = z;
                    
                    // Interpolate texture coordinates
                    double u = coords[0]*t0[0] + coords[1]*t1[0] + coords[2]*t2[0];
                    double v = coords[0]*t0[1] + coords[1]*t1[1] + coords[2]*t2[1];
                    
                    // Sample texture
                    int tx = (int)(u * (texture_width - 1));
                    int ty = (int)((1-v) * (texture_height - 1));
                    tx = (tx < 0) ? 0 : (tx >= texture_width ? texture_width-1 : tx);
                    ty = (ty < 0) ? 0 : (ty >= texture_height ? texture_height-1 : ty);
                    
                    int tidx = (ty * texture_width + tx) * texture_channels;
                    int iidx = (y * WIDTH + x) * 3;
                    
                    // Apply lighting with increased brightness
                    image[iidx] = (unsigned char)fmin(255, texture_data[tidx] * light_intensity);
                    image[iidx+1] = (unsigned char)fmin(255, texture_data[tidx+1] * light_intensity);
                    image[iidx+2] = (unsigned char)fmin(255, texture_data[tidx+2] * light_intensity);
                }
            }
        }
    }
}

int main() {
    // Read the OBJ file
    parse_obj_file("african_head.obj");

    // Load the texture
    texture_data = stbi_load("african_head_diffuse.tga", &texture_width, &texture_height, &texture_channels, 3);
    if (!texture_data) {
        fprintf(stderr, "Failed to load texture image\n");
        return 1;
    }
    printf("Loaded texture: %dx%d with %d channels\n", texture_width, texture_height, texture_channels);

    // Allocate buffers
    zbuffer = malloc(WIDTH * HEIGHT * sizeof(double));
    unsigned char *output_image = malloc(WIDTH * HEIGHT * 3);

    // Initialize FFmpeg
    avformat_network_init();

    // Set up FFmpeg encoding
    AVFormatContext *oc = NULL;
    AVOutputFormat *fmt = NULL;
    AVStream *video_st = NULL;
    AVCodecContext *c = NULL;
    AVCodec *codec = NULL;

    // Allocate the output media context
    avformat_alloc_output_context2(&oc, NULL, NULL, "output.mp4");
    if (!oc) {
        fprintf(stderr, "Could not deduce output format from file extension.\n");
        exit(1);
    }
    fmt = oc->oformat;

    // Add the video stream using the default format codecs and initialize the codec
    codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    if (!codec) {
        fprintf(stderr, "Codec not found\n");
        exit(1);
    }

    video_st = avformat_new_stream(oc, NULL);
    if (!video_st) {
        fprintf(stderr, "Could not allocate stream\n");
        exit(1);
    }

    c = avcodec_alloc_context3(codec);
    if (!c) {
        fprintf(stderr, "Could not allocate video codec context\n");
        exit(1);
    }

    // Set codec parameters
    c->codec_id = AV_CODEC_ID_H264;
    c->bit_rate = 400000;
    c->width = WIDTH;
    c->height = HEIGHT;
    video_st->time_base = (AVRational){1, 30};
    c->time_base = video_st->time_base;
    c->gop_size = 10;
    c->max_b_frames = 1;
    c->pix_fmt = AV_PIX_FMT_YUV420P;

    if (oc->oformat->flags & AVFMT_GLOBALHEADER)
        c->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    // Open the codec
    if (avcodec_open2(c, codec, NULL) < 0) {
        fprintf(stderr, "Could not open codec\n");
        exit(1);
    }

    // Copy the stream parameters to the muxer
    if (avcodec_parameters_from_context(video_st->codecpar, c) < 0) {
        fprintf(stderr, "Could not copy the stream parameters\n");
        exit(1);
    }

    // Open the output file
    if (!(fmt->flags & AVFMT_NOFILE)) {
        if (avio_open(&oc->pb, "output.mp4", AVIO_FLAG_WRITE) < 0) {
            fprintf(stderr, "Could not open 'output.mp4'\n");
            exit(1);
        }
    }

    // Write the stream header
    if (avformat_write_header(oc, NULL) < 0) {
        fprintf(stderr, "Error occurred when opening output file\n");
        exit(1);
    }

    // For image conversion
    struct SwsContext *sws_ctx = sws_getContext(
        WIDTH, HEIGHT, AV_PIX_FMT_RGB24,
        WIDTH, HEIGHT, AV_PIX_FMT_YUV420P,
        SWS_BILINEAR, NULL, NULL, NULL
    );

    AVFrame *frame = av_frame_alloc();
    frame->format = c->pix_fmt;
    frame->width = c->width;
    frame->height = c->height;

    // Allocate the buffers for the frame data
    if (av_image_alloc(frame->data, frame->linesize, c->width, c->height, c->pix_fmt, 32) < 0) {
        fprintf(stderr, "Could not allocate raw picture buffer\n");
        exit(1);
    }

    AVPacket pkt;
    int frame_count = 0;

    double angle_per_frame = (2.0 * M_PI) / N_FRAMES;
    
    // Projection matrices
    double projection[4][4];
    perspective_matrix(60.0, (double)WIDTH/HEIGHT, 0.1, 50.0, projection);
    
    double viewport[4][4];
    viewport_transform(0, 0, WIDTH, HEIGHT, 255, viewport);

    for (int frame_num = 0; frame_num < N_FRAMES; frame_num++) {
        printf("Rendering frame %d/%d\n", frame_num + 1, N_FRAMES);

        // Clear buffers
        for (int i = 0; i < WIDTH * HEIGHT; i++) {
            zbuffer[i] = INFINITY;
            output_image[i*3] = 128;     // Light gray background
            output_image[i*3+1] = 128;
            output_image[i*3+2] = 128;
        }

        // Transform vertices
        double rotation_y_angle = frame_num * angle_per_frame;
        
        for (int i = 0; i < num_triangles; i++) {
            double v0[4], v1[4], v2[4];
            double p0[4], p1[4], p2[4];
            
            // Get vertices
            for (int j = 0; j < 3; j++) {
                v0[j] = initial_vertices[triangles[i][0]][j];
                v1[j] = initial_vertices[triangles[i][1]][j];
                v2[j] = initial_vertices[triangles[i][2]][j];
            }
            v0[3] = v1[3] = v2[3] = 1.0;

            // Scale the model
            for (int j = 0; j < 3; j++) {
                v0[j] *= 1.5;
                v1[j] *= 1.5;
                v2[j] *= 1.5;
            }

            // Apply transformations
            rotate_y(rotation_y_angle, v0);
            rotate_y(rotation_y_angle, v1);
            rotate_y(rotation_y_angle, v2);

            // Translate away from camera
            v0[2] += 3.0; v1[2] += 3.0; v2[2] += 3.0;

            // Apply projection
            matrix_multiply(projection, v0, p0);
            matrix_multiply(projection, v1, p1);
            matrix_multiply(projection, v2, p2);

            // Perspective divide
            for (int j = 0; j < 3; j++) {
                p0[j] /= p0[3];
                p1[j] /= p1[3];
                p2[j] /= p2[3];
            }

            // Viewport transform
            matrix_multiply(viewport, p0, v0);
            matrix_multiply(viewport, p1, v1);
            matrix_multiply(viewport, p2, v2);

            // Get texture coordinates
            double *t0 = texcoords[texcoord_indices[i][0]];
            double *t1 = texcoords[texcoord_indices[i][1]];
            double *t2 = texcoords[texcoord_indices[i][2]];

            // Draw triangle
            draw_triangle(v0, v1, v2, t0, t1, t2, output_image);
        }

        // Convert RGB to YUV and encode
        const uint8_t *inData[1] = { output_image };
        int inLinesize[1] = { 3 * WIDTH };
        sws_scale(sws_ctx, inData, inLinesize, 0, HEIGHT, frame->data, frame->linesize);

        frame->pts = frame_count++;

        // Encode the frame
        av_init_packet(&pkt);
        pkt.data = NULL;
        pkt.size = 0;

        int ret = avcodec_send_frame(c, frame);
        if (ret < 0) {
            fprintf(stderr, "Error sending a frame for encoding\n");
            exit(1);
        }

        while (ret >= 0) {
            ret = avcodec_receive_packet(c, &pkt);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                break;
            else if (ret < 0) {
                fprintf(stderr, "Error during encoding\n");
                exit(1);
            }

            av_packet_rescale_ts(&pkt, c->time_base, video_st->time_base);
            pkt.stream_index = video_st->index;

            ret = av_interleaved_write_frame(oc, &pkt);
            if (ret < 0) {
                fprintf(stderr, "Error while writing output packet\n");
                exit(1);
            }
            av_packet_unref(&pkt);
        }
    }

    // Flush the encoder
    avcodec_send_frame(c, NULL);
    while (avcodec_receive_packet(c, &pkt) == 0) {
        av_packet_rescale_ts(&pkt, c->time_base, video_st->time_base);
        pkt.stream_index = video_st->index;
        av_interleaved_write_frame(oc, &pkt);
        av_packet_unref(&pkt);
    }

    // Write the trailer
    av_write_trailer(oc);

    // Clean up
    avcodec_close(c);
    avcodec_free_context(&c);
    av_frame_free(&frame);
    sws_freeContext(sws_ctx);
    if (!(fmt->flags & AVFMT_NOFILE))
        avio_closep(&oc->pb);
    avformat_free_context(oc);

    // Free resources
    free(zbuffer);
    free(output_image);
    stbi_image_free(texture_data);

    printf("Video saved to 'output.mp4'\n");
    return 0;
}