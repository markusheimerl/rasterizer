#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <limits.h>
// gcc -O3 -fopenmp rasterizer.c -lm -lavformat -lavcodec -lavutil -lswscale -lswresample -lpthread
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
#define FRAMES 60

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

// Global variables
double vertices[100000][3];
double initial_vertices[100000][3];
double texcoords[100000][2];
int triangles[200000][3];
int texcoord_indices[200000][3];
int num_vertices = 0, num_texcoords = 0, num_triangles = 0;

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
            vertices[num_vertices][0] = x; vertices[num_vertices][1] = y; vertices[num_vertices][2] = z;
            initial_vertices[num_vertices][0] = x; initial_vertices[num_vertices][1] = y; initial_vertices[num_vertices][2] = z;
            num_vertices++;
        } else if (strncmp(line, "vt ", 3) == 0) {
            double u, v;
            sscanf(line + 3, "%lf %lf", &u, &v);
            texcoords[num_texcoords][0] = u; texcoords[num_texcoords][1] = v;
            num_texcoords++;
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
            triangles[num_triangles][0] = vi[0] - 1; triangles[num_triangles][1] = vi[1] - 1; triangles[num_triangles][2] = vi[2] - 1;
            texcoord_indices[num_triangles][0] = (ti[0] != -1) ? ti[0] - 1 : -1;
            texcoord_indices[num_triangles][1] = (ti[1] != -1) ? ti[1] - 1 : -1;
            texcoord_indices[num_triangles][2] = (ti[2] != -1) ? ti[2] - 1 : -1;
            num_triangles++;
        }
    }
    fclose(file);
}

// Rotate point around Y-axis
void rotate_y(double angle, double point[3]) {
    double s = sin(angle), c = cos(angle), x = point[0], z = point[2];
    point[0] = c * x + s * z; point[2] = -s * x + c * z;
}

// Sample texture
void sample_texture(double u, double v, double color[3]) {
    u = fmin(fmax(u, 0.0), 1.0); v = fmin(fmax(v, 0.0), 1.0);
    int x = (int)(u * (texture_width - 1)), y = (int)((1.0 - v) * (texture_height - 1));
    int idx = (y * texture_width + x) * texture_channels;
    color[0] = texture_data[idx] / 255.0;
    color[1] = texture_data[idx + 1] / 255.0;
    color[2] = texture_data[idx + 2] / 255.0;
}

// Function to draw a filled triangle using the barycentric coordinate method
void draw_triangle(double *image, const double pts[3][4], const double uv[3][2]) {
    double bbox_min_x = fmin(fmin(pts[0][0], pts[1][0]), pts[2][0]);
    double bbox_min_y = fmin(fmin(pts[0][1], pts[1][1]), pts[2][1]);
    double bbox_max_x = fmax(fmax(pts[0][0], pts[1][0]), pts[2][0]);
    double bbox_max_y = fmax(fmax(pts[0][1], pts[1][1]), pts[2][1]);

    for (int x = (int)fmax(bbox_min_x, 0); x <= (int)fmin(bbox_max_x, WIDTH - 1); x++) {
        for (int y = (int)fmax(bbox_min_y, 0); y <= (int)fmin(bbox_max_y, HEIGHT - 1); y++) {
            double lambda[3];
            double denominator = ((pts[1][1] - pts[2][1]) * (pts[0][0] - pts[2][0]) + (pts[2][0] - pts[1][0]) * (pts[0][1] - pts[2][1]));
            lambda[0] = ((pts[1][1] - pts[2][1]) * (x - pts[2][0]) + (pts[2][0] - pts[1][0]) * (y - pts[2][1])) / denominator;
            lambda[1] = ((pts[2][1] - pts[0][1]) * (x - pts[2][0]) + (pts[0][0] - pts[2][0]) * (y - pts[2][1])) / denominator;
            lambda[2] = 1.0 - lambda[0] - lambda[1];

            if (lambda[0] >= 0 && lambda[0] <= 1 && lambda[1] >= 0 && lambda[1] <= 1 && lambda[2] >= 0 && lambda[2] <= 1) {
                double z = lambda[0] * pts[0][2] + lambda[1] * pts[1][2] + lambda[2] * pts[2][2];
                int idx = y * WIDTH + x;
                if (z < image[idx * 4 + 3]) {
                    image[idx * 4 + 3] = z; // update depth
                    double u = lambda[0] * uv[0][0] + lambda[1] * uv[1][0] + lambda[2] * uv[2][0];
                    double v = lambda[0] * uv[0][1] + lambda[1] * uv[1][1] + lambda[2] * uv[2][1];
                    double color[3];
                    sample_texture(u, v, color);
                    image[idx * 4] = color[0];
                    image[idx * 4 + 1] = color[1];
                    image[idx * 4 + 2] = color[2];
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
    if (!texture_data) { fprintf(stderr, "Failed to load texture image\n"); return 1; }

    // Allocate image buffers
    double *image = malloc(WIDTH * HEIGHT * 4 * sizeof(double));
    unsigned char *output_image = malloc(WIDTH * HEIGHT * 3);

    // Initialize FFmpeg
    avformat_network_init();

    AVOutputFormat *output_format = NULL;
    AVFormatContext *av_format_ctx = NULL;
    AVStream *video_stream = NULL;
    AVCodecContext *codec_ctx = NULL;
    AVCodec *codec = NULL;

    // Allocate the output media context
    avformat_alloc_output_context2(&av_format_ctx, NULL, NULL, "output.mp4");
    if (!av_format_ctx) {
        fprintf(stderr, "Could not deduce output format from file extension.\n");
        return 1;
    }
    output_format = av_format_ctx->oformat;

    codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    if (!codec) {
        fprintf(stderr, "Codec not found\n");
        return 1;
    }

    video_stream = avformat_new_stream(av_format_ctx, NULL);
    if (!video_stream) {
        fprintf(stderr, "Could not allocate stream\n");
        return 1;
    }

    codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) {
        fprintf(stderr, "Could not allocate video codec context\n");
        return 1;
    }

    codec_ctx->codec_id = AV_CODEC_ID_H264;
    codec_ctx->bit_rate = 400000;
    codec_ctx->width = WIDTH;
    codec_ctx->height = HEIGHT;
    video_stream->time_base = (AVRational){1, 30};
    codec_ctx->time_base = video_stream->time_base;
    codec_ctx->gop_size = 10;
    codec_ctx->max_b_frames = 1;
    codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;

    if (av_format_ctx->oformat->flags & AVFMT_GLOBALHEADER)
        codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    if (avcodec_open2(codec_ctx, codec, NULL) < 0) {
        fprintf(stderr, "Could not open codec\n");
        return 1;
    }

    if (avcodec_parameters_from_context(video_stream->codecpar, codec_ctx) < 0) {
        fprintf(stderr, "Could not copy the stream parameters\n");
        return 1;
    }

    if (!(output_format->flags & AVFMT_NOFILE)) {
        if (avio_open(&av_format_ctx->pb, "output.mp4", AVIO_FLAG_WRITE) < 0) {
            fprintf(stderr, "Could not open output file.\n");
            return 1;
        }
    }

    if (avformat_write_header(av_format_ctx, NULL) < 0) {
        fprintf(stderr, "Error occurred when opening output file\n");
        return 1;
    }

    // For image conversion
    struct SwsContext *sws_ctx = sws_getContext(
        WIDTH, HEIGHT, AV_PIX_FMT_RGB24,
        WIDTH, HEIGHT, AV_PIX_FMT_YUV420P,
        SWS_BILINEAR, NULL, NULL, NULL
    );

    AVFrame *frame = av_frame_alloc();
    frame->format = codec_ctx->pix_fmt;
    frame->width = codec_ctx->width;
    frame->height = codec_ctx->height;

    // Allocate the buffers for the frame data
    if (av_image_alloc(frame->data, frame->linesize, codec_ctx->width, codec_ctx->height, codec_ctx->pix_fmt, 32) < 0) {
        fprintf(stderr, "Could not allocate raw picture buffer\n");
        return 1;
    }

    AVPacket pkt;
    int frame_count = 0;

    // Define constants for transformations
    double scale_factor = 0.5;
    double translation[3] = {0, 0, -3};
    double initial_rotation = 0.0;
    double angle_per_frame = (2.0 * M_PI) / FRAMES;

    for (int frame_num = 0; frame_num < FRAMES; frame_num++) {
        printf("Rendering frame %d/%d\n", frame_num + 1, FRAMES);

        memset(image, 255, WIDTH * HEIGHT * 4 * sizeof(double));
        for (int i = 0; i < WIDTH * HEIGHT; i++) {
            image[i * 4 + 3] = DBL_MAX; // Set depth buffer to max
        }

        // Transformation: rotate and translate
        double rotation_y_angle = initial_rotation + frame_num * angle_per_frame;
        for (int i = 0; i < num_vertices; i++) {
            vertices[i][0] = initial_vertices[i][0] * scale_factor;
            vertices[i][1] = initial_vertices[i][1] * scale_factor;
            vertices[i][2] = initial_vertices[i][2] * scale_factor;

            rotate_y(rotation_y_angle, vertices[i]);

            vertices[i][0] += translation[0];
            vertices[i][1] += translation[1];
            vertices[i][2] += translation[2];
        }

        // Perspective transformation
        for (int i = 0; i < num_triangles; i++) {
            double verts[3][4];
            double uv_coords[3][2];
            for (int j = 0; j < 3; j++) {
                double *vertex = vertices[triangles[i][j]];
                verts[j][0] = (vertex[0] / -vertex[2]) * WIDTH / 2 + WIDTH / 2.0;
                verts[j][1] = (vertex[1] / -vertex[2]) * HEIGHT / 2 + HEIGHT / 2.0;
                verts[j][2] = vertex[2];
                uv_coords[j][0] = texcoords[texcoord_indices[i][j]][0];
                uv_coords[j][1] = texcoords[texcoord_indices[i][j]][1];
            }
            draw_triangle(image, verts, uv_coords);
        }

        // Prepare frame data
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                int idx = (y * WIDTH + x) * 4;
                int out_idx = (y * WIDTH + x) * 3;
                output_image[out_idx] = (unsigned char)(fmax(0.0, fmin(1.0, image[idx])) * 255);
                output_image[out_idx + 1] = (unsigned char)(fmax(0.0, fmin(1.0, image[idx + 1])) * 255);
                output_image[out_idx + 2] = (unsigned char)(fmax(0.0, fmin(1.0, image[idx + 2])) * 255);
            }
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

        int ret = avcodec_send_frame(codec_ctx, frame);
        if (ret < 0) {
            fprintf(stderr, "Error sending a frame for encoding\n");
            exit(1);
        }

        while (ret >= 0) {
            ret = avcodec_receive_packet(codec_ctx, &pkt);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                break;
            else if (ret < 0) {
                fprintf(stderr, "Error during encoding\n");
                exit(1);
            }

            // Rescale output packet timestamp values from codec to stream timebase
            av_packet_rescale_ts(&pkt, codec_ctx->time_base, video_stream->time_base);
            pkt.stream_index = video_stream->index;

            // Write the compressed frame to the media file
            ret = av_interleaved_write_frame(av_format_ctx, &pkt);
            if (ret < 0) {
                fprintf(stderr, "Error while writing output packet\n");
                exit(1);
            }
            av_packet_unref(&pkt);
        }
    }

    // Flush the encoder
    avcodec_send_frame(codec_ctx, NULL);
    while (avcodec_receive_packet(codec_ctx, &pkt) == 0) {
        av_packet_rescale_ts(&pkt, codec_ctx->time_base, video_stream->time_base);
        pkt.stream_index = video_stream->index;
        av_interleaved_write_frame(av_format_ctx, &pkt);
        av_packet_unref(&pkt);
    }

    // Write the trailer
    av_write_trailer(av_format_ctx);

    // Clean up
    avcodec_close(codec_ctx);
    avcodec_free_context(&codec_ctx);
    av_frame_free(&frame);
    sws_freeContext(sws_ctx);
    if (!(output_format->flags & AVFMT_NOFILE))
        avio_closep(&av_format_ctx->pb);
    avformat_free_context(av_format_ctx);

    // Free resources
    free(image);
    free(output_image);
    stbi_image_free(texture_data);

    printf("Video saved to 'output.mp4'\n");

    return 0;
}