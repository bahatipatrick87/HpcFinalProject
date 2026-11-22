/*
 * pipeline_kernels.cpp — C++ core of the pipeline, loaded by each MPI rank
 * from Python with ctypes (src/main_mpi.py).
 *   OpenCL (GPU): Stage 2 fft_magnitude, Stage 3 highpass_filter.
 *   OpenMP (CPU): Stage 4 extract_features_cpu, plus CPU versions of
 *                 Stages 2-3 used only when no OpenCL device is found.
 */
#include "pipeline_kernels.h"   // C API declared for Python (extern "C")
#include <cmath>
#include <cstring>
#include <ctime>                // clock_gettime
#include <iostream>
#include <utility>              // std::swap
#include <vector>
#include <omp.h>                // OpenMP

#define CL_SILENCE_DEPRECATION  // hide Apple's "OpenCL deprecated" warnings
#ifdef __APPLE__
#include <OpenCL/opencl.h>      // macOS framework header
#else
#include <CL/cl.h>              // Linux / other platforms
#endif

/* ---- Host wall-clock timer (seconds) ---- */
static double wall_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);        // monotonic: never jumps back
    return ts.tv_sec + ts.tv_nsec * 1e-9;       // seconds + nanoseconds
}

/* ---- Global OpenCL state: created once per process (= per MPI rank) ---- */
static cl_context        g_context      = nullptr;  // device workspace
static cl_command_queue  g_queue        = nullptr;  // host -> device job queue
static cl_program        g_program      = nullptr;  // compiled KERNEL_SRC
static cl_kernel         g_fft_kernel   = nullptr;  // Stage 2: FFT magnitude
static cl_kernel         g_filt_kernel  = nullptr;  // Stage 3: high-pass filter
static bool              g_initialized  = false;    // init_opencl() done
static bool              g_gpu_avail    = false;    // false -> OpenMP fallback
static int               g_num_gpus     = 0;        // GPUs found
static std::vector<cl_device_id> g_all_devices;     // their device ids

/* Return -1 with a message if an OpenCL call failed. */
#define CHECK_CL(err, msg)                                                 \
    if ((err) != CL_SUCCESS) {                                             \
        std::cerr << "[CPP] OpenCL error: " << (msg)                      \
                  << " (code=" << (err) << ")\n";                          \
        return -1;                                                         \
    }

/* 
 * OpenCL device code. Kept as a string and compiled at run time by the
 * driver (clBuildProgram) for whatever device is present.
 *  */
static const char *KERNEL_SRC = R"CL(

/* In-place radix-2 Cooley-Tukey FFT of one line of length n (power of two). */
#define MAX_FFT_LEN 256
void fft_1d(__private float* re, __private float* im, int n)
{
    /* 1) Bit-reversal permutation (e.g. n = 8: index 1 = 001b <-> 4 = 100b) */
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;    // j = bit-reversed i
        j ^= bit;
        if (i < j) {                            // swap each pair once
            float t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    /* 2) log2(n) butterfly stages: len = 2, 4, ..., n */
    for (int len = 2; len <= n; len <<= 1) {
        float step = -6.28318530718f / (float)len;   // -2*pi/len (numpy sign)
        for (int i = 0; i < n; i += len) {           // each block of size len
            for (int k = 0; k < len / 2; ++k) {
                float wr = cos(step * (float)k), wi = sin(step * (float)k); // twiddle w
                int u = i + k, v = i + k + len / 2;  // butterfly pair
                float tr = re[v] * wr - im[v] * wi;  // t = x[v] * w
                float ti = re[v] * wi + im[v] * wr;
                re[v] = re[u] - tr;  im[v] = im[u] - ti;   // x[v] = x[u] - t
                re[u] += tr;         im[u] += ti;          // x[u] = x[u] + t
            }
        }
    }
}

/* Stage 2: 2D FFT magnitude. Layout HWC: [image][y][x][channel].
 * pass 0: one work-item per (image, channel, row)    -> complex scratch
 * pass 1: one work-item per (image, channel, column) -> |spectrum| in out */
__kernel void fft_magnitude(__global const float* in,
                            __global float* scratch,
                            __global float* out,
                            int width, int height, int channels, int pass)
{
    float re[MAX_FFT_LEN], im[MAX_FFT_LEN];  // private copy of one line
    int gid  = get_global_id(0);             // this work-item's index
    int len  = (pass == 0) ? height : width; // lines per image-channel
    int line = gid % len;                    // row (pass 0) or column (pass 1)
    int c    = (gid / len) % channels;       // channel 0..2
    int img  = gid / (len * channels);      // image in the batch
    size_t base = (size_t)img * height * width * channels;  // image offset

    if (pass == 0) {
        for (int x = 0; x < width; ++x) {    // load row: real input, im = 0
            re[x] = in[base + (size_t)(line * width + x) * channels + c];
            im[x] = 0.0f;
        }
        fft_1d(re, im, width);               // 1D FFT along the row
        for (int x = 0; x < width; ++x) {    // store (re, im) interleaved
            size_t e = base + (size_t)(line * width + x) * channels + c;
            scratch[2 * e] = re[x];  scratch[2 * e + 1] = im[x];
        }
    } else {
        for (int y = 0; y < height; ++y) {   // load column from scratch
            size_t e = base + (size_t)(y * width + line) * channels + c;
            re[y] = scratch[2 * e];  im[y] = scratch[2 * e + 1];
        }
        fft_1d(re, im, height);              // 1D FFT along the column
        for (int y = 0; y < height; ++y) {   // write magnitude |re + i*im|
            size_t e = base + (size_t)(y * width + line) * channels + c;
            out[e] = sqrt(re[y] * re[y] + im[y] * im[y]);
        }
    }
}

/* Stage 3: high-pass filter H = 1 - exp(-d^2/50), one work-item per element. */
__kernel void highpass_filter(__global const float* in,
                              __global float* out,
                              int width,
                              int height)
{
    int gid = get_global_id(0);                          // element index
    int channels        = 3;
    int pixels_per_img  = width * height;
    int spatial         = (gid / channels) % pixels_per_img; // drop channel, image
    int x  = spatial % width;                            // column in image
    int y  = spatial / width;                            // row in image

    int dx = x < width /2 ? x : width  - x;   // wrap-around distance from DC
    int dy = y < height/2 ? y : height - y;   // (DC is at (0,0), unshifted FFT)
    float d  = sqrt((float)(dx*dx + dy*dy));
    float hp = 1.0f - exp(-d * d / 50.0f);   // 0 at DC, close to 1 far away

    out[gid] = in[gid] * hp;                  // apply mask
}

)CL";

/* 
 * init_opencl: OpenCL set-up, once per rank.
 * platform -> device -> context -> queue -> build program -> kernels.
 * Returns 0 on success, -1 if no device (OpenMP fallback is then used).
 *  */
int init_opencl(int mpi_rank) {
    if (g_initialized) return 0;                         // already set up

    // (1) Platform
    cl_int err;
    cl_uint nplat = 0;
    err = clGetPlatformIDs(0, nullptr, &nplat);          // count platforms
    if (err != CL_SUCCESS || nplat == 0) {
        std::cerr << "[CPP] No OpenCL platforms — CPU fallback active.\n";
        g_gpu_avail = false; return -1;
    }
    std::vector<cl_platform_id> plats(nplat);
    clGetPlatformIDs(nplat, plats.data(), nullptr);      // get their ids
    cl_platform_id plat = plats[0];                      // use the first one

    // (2) Device
    cl_uint num_devices = 0;
    clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);  // count GPUs

    cl_device_id selected;
    if (num_devices > 0) {
        g_all_devices.resize(num_devices);
        clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, num_devices, g_all_devices.data(), nullptr);
        g_num_gpus = (int)num_devices;
        // Round-robin: device_idx = mpi_rank % num_devices. With one GPU all ranks
        // share it (contention, report §5.5); with several GPUs ranks spread
        // across them with no code change.
        int device_idx = mpi_rank % (int)num_devices;
        selected  = g_all_devices[device_idx];
        char name[256] = {};
        clGetDeviceInfo(selected, CL_DEVICE_NAME, sizeof(name), name, nullptr);
        std::cout << "[CPP] Rank " << mpi_rank << " -> GPU[" << device_idx << "] "
                  << name << " (" << num_devices << " total)\n";   // e.g. "Apple M4"
    } else {
        // No GPU: try an OpenCL CPU device so the same kernels still run
        cl_uint ncpu = 0;
        err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_CPU, 1, &selected, &ncpu);
        if (err != CL_SUCCESS || ncpu == 0) {
            std::cerr << "[CPP] No GPU or CPU OpenCL device — CPU fallback.\n";
            g_gpu_avail = false; return -1;
        }
        g_num_gpus = 0;
        char name[256] = {};
        clGetDeviceInfo(selected, CL_DEVICE_NAME, sizeof(name), name, nullptr);
        std::cout << "[CPP] Rank " << mpi_rank
                  << " -> CPU OpenCL fallback: " << name << "\n";
    }

    // (3) Context and (4) in-order command queue on the selected device
    g_context = clCreateContext(nullptr, 1, &selected, nullptr, nullptr, &err);
    CHECK_CL(err, "clCreateContext");
    g_queue = clCreateCommandQueue(g_context, selected, 0, &err);
    CHECK_CL(err, "clCreateCommandQueue");

    // (5) Program: compile KERNEL_SRC for this device at run time
    g_program = clCreateProgramWithSource(g_context, 1, &KERNEL_SRC, nullptr, &err);
    CHECK_CL(err, "clCreateProgramWithSource");

    err = clBuildProgram(g_program, 1, &selected, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {                             // print compiler log
        size_t log_sz;
        clGetProgramBuildInfo(g_program, selected, CL_PROGRAM_BUILD_LOG,
                              0, nullptr, &log_sz);
        std::vector<char> log(log_sz);
        clGetProgramBuildInfo(g_program, selected, CL_PROGRAM_BUILD_LOG,
                              log_sz, log.data(), nullptr);
        std::cerr << "[CPP] Build log:\n" << log.data() << "\n";
        return -1;
    }

    // (6) Kernel handles, looked up by function name
    g_fft_kernel  = clCreateKernel(g_program, "fft_magnitude",   &err);
    CHECK_CL(err, "clCreateKernel(fft_magnitude)");
    g_filt_kernel = clCreateKernel(g_program, "highpass_filter",  &err);
    CHECK_CL(err, "clCreateKernel(highpass_filter)");

    g_initialized = true;
    g_gpu_avail   = true;
    std::cout << "[CPP] OpenCL initialized: 2 kernels (fft_magnitude, highpass_filter).\n";
    return 0;
}

/* ---- Release OpenCL objects in reverse order of creation ---- */
void cleanup_opencl() {
    if (!g_initialized) return;
    clReleaseKernel(g_fft_kernel);
    clReleaseKernel(g_filt_kernel);
    clReleaseProgram(g_program);
    clReleaseCommandQueue(g_queue);
    clReleaseContext(g_context);
    g_initialized = false;
}

/* ==========================================================================
 * run_kernel_timed: one single-pass kernel (the filter) on the GPU.
 * Steps: (7) copy in -> (8) set args -> (9) launch -> (10) copy out -> (11) free.
 * clFinish after each step so the host clock times finished work (Table 5.5).
 * ========================================================================== */
static int run_kernel_timed(cl_kernel kern,
                             const float *h_in, float *h_out,
                             size_t data_sz, size_t gws_override,
                             int width, int height,
                             GpuStageTiming *timing) {
    size_t bytes = data_sz * sizeof(float);
    cl_int err;
    double t0, t1, t2, t3;

    // (7) Host -> device: input copied from the NumPy array, output empty
    t0 = wall_sec();
    cl_mem d_in  = clCreateBuffer(g_context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  bytes, (void *)h_in, &err);
    if (err != CL_SUCCESS) return -1;
    cl_mem d_out = clCreateBuffer(g_context, CL_MEM_WRITE_ONLY,
                                  bytes, nullptr, &err);
    if (err != CL_SUCCESS) { clReleaseMemObject(d_in); return -1; }
    clFinish(g_queue);                                   // wait: copy done
    t1 = wall_sec();

    // (8) Kernel arguments, in the order of the kernel signature
    clSetKernelArg(kern, 0, sizeof(cl_mem), &d_in);
    clSetKernelArg(kern, 1, sizeof(cl_mem), &d_out);
    clSetKernelArg(kern, 2, sizeof(int), &width);
    clSetKernelArg(kern, 3, sizeof(int), &height);

    // (9) Launch gws work-items (1-D); work-group size chosen by the driver
    size_t gws = gws_override;
    err = clEnqueueNDRangeKernel(g_queue, kern, 1, nullptr,
                                  &gws, nullptr, 0, nullptr, nullptr);
    clFinish(g_queue);                                   // wait: kernel done
    t2 = wall_sec();
    if (err != CL_SUCCESS) { clReleaseMemObject(d_in); clReleaseMemObject(d_out); return -1; }

    // (10) Device -> host: blocking read into the NumPy output array
    err = clEnqueueReadBuffer(g_queue, d_out, CL_TRUE, 0, bytes, h_out, 0, nullptr, nullptr);
    clFinish(g_queue);
    t3 = wall_sec();

    // (11) Free device memory
    clReleaseMemObject(d_in);
    clReleaseMemObject(d_out);

    if (timing) {                                        // phase breakdown
        timing->t_host_to_device = t1 - t0;
        timing->t_kernel         = t2 - t1;
        timing->t_device_to_host = t3 - t2;
        timing->t_total_gpu      = t3 - t0;
    }
    return (err == CL_SUCCESS) ? 0 : -1;
}

/* ==========================================================================
 * run_fft_timed: two-pass FFT kernel on the GPU.
 * Works in chunks of <= 1024 images to bound GPU memory when sequential
 * mode passes a whole block (P=1 would need 5.4 GB of scratch). A pipeline
 * batch (256 images) is always a single chunk.
 * ========================================================================== */
static const int FFT_CHUNK_IMAGES = 1024;

// Power of two in [2, 256]: n & (n-1) == 0 only for powers of two
static bool is_fft_size(int n) { return n >= 2 && n <= 256 && (n & (n - 1)) == 0; }

static int run_fft_timed(const float *h_in, float *h_out, int batch,
                         int width, int height, int ch, GpuStageTiming *timing) {
    if (!is_fft_size(width) || !is_fft_size(height)) {
        std::cerr << "[CPP] FFT size must be a power of two up to 256, got "
                  << width << "x" << height << "\n";
        return -1;
    }
    double t_h2d = 0, t_kern = 0, t_d2h = 0;
    size_t ppi = (size_t)width * height * ch;            // floats per image
    for (int start = 0; start < batch; start += FFT_CHUNK_IMAGES) {
        int n = (batch - start < FFT_CHUNK_IMAGES) ? batch - start : FFT_CHUNK_IMAGES;
        size_t bytes = (size_t)n * ppi * sizeof(float);  // bytes in this chunk
        cl_int err;

        // (7) Buffers: input (copied in), scratch (2x: re + im), output
        double t0 = wall_sec();
        cl_mem d_in = clCreateBuffer(g_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     bytes, (void *)(h_in + (size_t)start * ppi), &err);
        if (err != CL_SUCCESS) return -1;
        cl_mem d_scr = clCreateBuffer(g_context, CL_MEM_READ_WRITE, 2 * bytes, nullptr, &err);
        if (err != CL_SUCCESS) { clReleaseMemObject(d_in); return -1; }
        cl_mem d_out = clCreateBuffer(g_context, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);
        if (err != CL_SUCCESS) { clReleaseMemObject(d_in); clReleaseMemObject(d_scr); return -1; }
        clFinish(g_queue);
        double t1 = wall_sec();

        // (8) Arguments 0-5, shared by both passes
        clSetKernelArg(g_fft_kernel, 0, sizeof(cl_mem), &d_in);
        clSetKernelArg(g_fft_kernel, 1, sizeof(cl_mem), &d_scr);
        clSetKernelArg(g_fft_kernel, 2, sizeof(cl_mem), &d_out);
        clSetKernelArg(g_fft_kernel, 3, sizeof(int), &width);
        clSetKernelArg(g_fft_kernel, 4, sizeof(int), &height);
        clSetKernelArg(g_fft_kernel, 5, sizeof(int), &ch);

        // (9) Pass 0 = rows, pass 1 = columns; one work-item per
        //     (image, channel, line): 49,152 per pass at batch 256 (256*3*64)
        for (int pass = 0; pass < 2 && err == CL_SUCCESS; ++pass) {
            int lines = (pass == 0) ? height : width;
            size_t gws = (size_t)n * ch * lines;
            clSetKernelArg(g_fft_kernel, 6, sizeof(int), &pass);   // arg 6 = pass
            err = clEnqueueNDRangeKernel(g_queue, g_fft_kernel, 1, nullptr,
                                         &gws, nullptr, 0, nullptr, nullptr);
            clFinish(g_queue);   // sync point: pass 1 reads what pass 0 wrote
        }
        double t2 = wall_sec();

        // (10) Copy result into this chunk's slice of the NumPy output
        if (err == CL_SUCCESS)
            err = clEnqueueReadBuffer(g_queue, d_out, CL_TRUE, 0, bytes,
                                      h_out + (size_t)start * ppi, 0, nullptr, nullptr);
        clFinish(g_queue);
        double t3 = wall_sec();

        // (11) Free this chunk's buffers
        clReleaseMemObject(d_in);
        clReleaseMemObject(d_scr);
        clReleaseMemObject(d_out);
        if (err != CL_SUCCESS) return -1;
        t_h2d += t1 - t0;  t_kern += t2 - t1;  t_d2h += t3 - t2;   // accumulate
    }
    if (timing) {
        timing->t_host_to_device = t_h2d;
        timing->t_kernel         = t_kern;
        timing->t_device_to_host = t_d2h;
        timing->t_total_gpu      = t_h2d + t_kern + t_d2h;
    }
    return 0;
}

/* ==========================================================================
 * OpenMP CPU fallback for Stages 2-3: same math as the two kernels.
 * Used only if no OpenCL device exists (and by tests / OpenMP experiment).
 * ========================================================================== */

/* CPU copy of fft_1d (same bit reversal + butterflies). */
static void cpu_fft_1d(std::vector<float> &re, std::vector<float> &im, int n) {
    for (int i = 1, j = 0; i < n; ++i) {                 // bit reversal
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { std::swap(re[i], re[j]); std::swap(im[i], im[j]); }
    }
    for (int len = 2; len <= n; len <<= 1) {             // butterfly stages
        float step = -6.28318530718f / (float)len;
        for (int i = 0; i < n; i += len) {
            for (int k = 0; k < len / 2; ++k) {
                float wr = cosf(step * k), wi = sinf(step * k);
                int u = i + k, v = i + k + len / 2;
                float tr = re[v] * wr - im[v] * wi;
                float ti = re[v] * wi + im[v] * wr;
                re[v] = re[u] - tr;  im[v] = im[u] - ti;
                re[u] += tr;         im[u] += ti;
            }
        }
    }
}

/* OpenMP 2D FFT magnitude: images split across threads. */
static void cpu_fft_magnitude(const float *in, float *out,
                              int batch, int width, int height, int ch) {
    size_t ppi = (size_t)width * height * ch;
#pragma omp parallel for schedule(static)    // equal-cost images -> static split
    for (int i = 0; i < batch; ++i) {        // one image per iteration
        const float *src = in  + (size_t)i * ppi;
        float       *dst = out + (size_t)i * ppi;
        // Declared inside the loop -> private to each thread (no data race)
        std::vector<float> sre((size_t)width * height), sim((size_t)width * height);
        std::vector<float> re(width > height ? width : height), im(re.size());
        for (int c = 0; c < ch; ++c) {
            for (int y = 0; y < height; ++y) {          // rows
                for (int x = 0; x < width; ++x) {
                    re[x] = src[(size_t)(y * width + x) * ch + c];
                    im[x] = 0.0f;
                }
                cpu_fft_1d(re, im, width);
                for (int x = 0; x < width; ++x) {
                    sre[y * width + x] = re[x];
                    sim[y * width + x] = im[x];
                }
            }
            for (int x = 0; x < width; ++x) {           // columns, magnitude
                for (int y = 0; y < height; ++y) {
                    re[y] = sre[y * width + x];
                    im[y] = sim[y * width + x];
                }
                cpu_fft_1d(re, im, height);
                for (int y = 0; y < height; ++y)
                    dst[(size_t)(y * width + x) * ch + c] =
                        sqrtf(re[y] * re[y] + im[y] * im[y]);
            }
        }
    }
}

/* OpenMP high-pass filter: same mask, applied to all channels of a pixel. */
static void cpu_highpass(const float *in, float *out,
                         int batch, int width, int height, int ch) {
    size_t ppi = (size_t)width * height * ch;
#pragma omp parallel for schedule(static)    // one image per iteration
    for (int i = 0; i < batch; ++i) {
        const float *src = in  + (size_t)i * ppi;
        float       *dst = out + (size_t)i * ppi;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int dx = x < width /2 ? x : width  - x;     // wrap-around distance
                int dy = y < height/2 ? y : height - y;
                float d  = sqrtf((float)(dx*dx + dy*dy));
                float hp = 1.0f - expf(-d * d / 50.0f);     // mask value
                int base = (y * width + x) * ch;
                for (int c = 0; c < ch; ++c)
                    dst[base + c] = src[base + c] * hp;
            }
        }
    }
}

/* ==========================================================================
 * Public functions called from Python: GPU path if available, else OpenMP.
 * ========================================================================== */

/* CPU FFT called directly (tests, run_openmp_fft_scaling.py). */
int process_images_fft_cpu(const float *image_data, int batch_size,
                           int width, int height, int channels,
                           float *output_data) {
    if (!is_fft_size(width) || !is_fft_size(height)) return -1;
    cpu_fft_magnitude(image_data, output_data, batch_size, width, height, channels);
    return 0;
}

/* Stage 2 — called by src/stages/fft_stage.py once per batch. */
int process_images_fft(const float *image_data, int batch_size,
                       int width, int height, int channels,
                       float *output_data) {
    if (!g_initialized) init_opencl(0);                  // lazy set-up
    if (!g_gpu_avail) {                                  // no device: OpenMP
        cpu_fft_magnitude(image_data, output_data, batch_size, width, height, channels);
        return 0;
    }
    return run_fft_timed(image_data, output_data, batch_size, width, height, channels, nullptr);
}

/* Stage 3 — called by src/stages/filter_stage.py once per batch. */
int apply_frequency_filter_gpu(const float *spectrum_data, int batch_size,
                                int width, int height, int channels,
                                float *filtered_data) {
    if (!g_initialized) init_opencl(0);
    if (!g_gpu_avail) {
        cpu_highpass(spectrum_data, filtered_data, batch_size, width, height, channels);
        return 0;
    }
    // One work-item per element: GWS = batch * height * width * channels
    size_t data_sz = (size_t)batch_size * width * height * channels;
    return run_kernel_timed(g_filt_kernel, spectrum_data, filtered_data,
                             data_sz, data_sz, width, height, nullptr);
}

/* Old combined Stage 2+3 call, kept for compatibility; not used by the pipeline. */
int process_images_frequency_domain(const float *image_data, int batch_size,
                                    int width, int height, int channels,
                                    float *output_data) {
    if (!g_initialized) init_opencl(0);
    size_t data_sz = (size_t)batch_size * width * height * channels;
    std::vector<float> tmp(data_sz);                     // FFT result
    int rc = process_images_fft(image_data, batch_size, width, height, channels,
                                 tmp.data());
    if (rc != 0) return rc;
    return apply_frequency_filter_gpu(tmp.data(), batch_size, width, height, channels,
                                       output_data);
}

/* Timed Stage 2: also returns H2D / kernel / D2H times (Table 5.5). */
int process_images_fft_timed(const float *image_data, int batch_size,
                              int width, int height, int channels,
                              float *output_data, GpuStageTiming *timing) {
    if (!g_initialized) init_opencl(0);
    if (!g_gpu_avail) {                                  // CPU: all time = "kernel"
        double t0 = wall_sec();
        cpu_fft_magnitude(image_data, output_data, batch_size, width, height, channels);
        double t1 = wall_sec();
        if (timing) { timing->t_host_to_device=0; timing->t_kernel=t1-t0;
                      timing->t_device_to_host=0; timing->t_total_gpu=t1-t0; }
        return 0;
    }
    return run_fft_timed(image_data, output_data, batch_size, width, height, channels, timing);
}

/* Timed Stage 3: also returns H2D / kernel / D2H times (Table 5.5). */
int apply_frequency_filter_gpu_timed(const float *spectrum_data, int batch_size,
                                      int width, int height, int channels,
                                      float *filtered_data, GpuStageTiming *timing) {
    if (!g_initialized) init_opencl(0);
    if (!g_gpu_avail) {
        double t0 = wall_sec();
        cpu_highpass(spectrum_data, filtered_data, batch_size, width, height, channels);
        double t1 = wall_sec();
        if (timing) { timing->t_host_to_device=0; timing->t_kernel=t1-t0;
                      timing->t_device_to_host=0; timing->t_total_gpu=t1-t0; }
        return 0;
    }
    size_t data_sz = (size_t)batch_size * width * height * channels;
    return run_kernel_timed(g_filt_kernel, spectrum_data, filtered_data,
                             data_sz, data_sz, width, height, timing);
}

/* ==========================================================================
 * Stage 4 (OpenMP) — called by src/stages/feature_stage.py once per batch.
 * Mean intensity of each image: one float per image, later gathered by MPI.
 * ========================================================================== */
int extract_features_cpu(const float *processed_data, int batch_size, int width,
                         int height, int channels, float *features_out) {
    size_t ppi = (size_t)width * height * channels;      // floats per image (12,288 for 64x64 RGB)
#pragma omp parallel for schedule(static)    // images split evenly over threads
    for (int i = 0; i < batch_size; ++i) {   // independent iterations
        const float *row = processed_data + (size_t)i * ppi;   // image i
        double sum = 0.0;                    // private per iteration; double for accuracy
        for (size_t j = 0; j < ppi; ++j) sum += row[j];
        features_out[i] = (float)(sum / (double)ppi);   // only thread i writes [i]: no race
    }                                        // implicit barrier: all threads finish here
    return 0;
}

/* ==========================================================================
 * GPU device queries — prove which device exists (e.g. "Apple M4"),
 * used by experiments/verify_gpu_profiling.py and the tests.
 * ========================================================================== */

/* Number of GPUs over all OpenCL platforms. */
int query_gpu_device_count(void) {
    if (g_initialized) return g_num_gpus;
    cl_uint nplat = 0;
    if (clGetPlatformIDs(0, nullptr, &nplat) != CL_SUCCESS || nplat == 0) return 0;
    std::vector<cl_platform_id> plats(nplat);
    clGetPlatformIDs(nplat, plats.data(), nullptr);
    cl_uint total = 0;
    for (cl_uint p = 0; p < nplat; ++p) {                // sum GPUs per platform
        cl_uint n = 0;
        if (clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, 0, nullptr, &n) == CL_SUCCESS)
            total += n;
    }
    return (int)total;
}

/* Copy the name of GPU number idx into buf (always NUL-terminated). */
int query_gpu_device_name(int idx, char *buf, int buf_len) {
    if (!buf || buf_len <= 0) return -1;
    cl_uint nplat = 0;
    if (clGetPlatformIDs(0, nullptr, &nplat) != CL_SUCCESS || nplat == 0) return -1;
    std::vector<cl_platform_id> plats(nplat);
    clGetPlatformIDs(nplat, plats.data(), nullptr);
    int seen = 0;                                        // GPUs counted so far
    for (cl_uint p = 0; p < nplat; ++p) {
        cl_uint ngpu = 0;
        if (clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, 0, nullptr, &ngpu) != CL_SUCCESS)
            continue;
        std::vector<cl_device_id> devs(ngpu);
        clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, ngpu, devs.data(), nullptr);
        for (cl_uint d = 0; d < ngpu; ++d) {
            if (seen++ == idx) {                         // found GPU number idx
                clGetDeviceInfo(devs[d], CL_DEVICE_NAME,
                                (size_t)buf_len, buf, nullptr);
                buf[buf_len-1] = '\0';
                return 0;
            }
        }
    }
    strncpy(buf, "unknown", (size_t)buf_len);            // idx not found
    buf[buf_len-1] = '\0';
    return -1;
}
