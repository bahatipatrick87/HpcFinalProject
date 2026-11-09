#include "pipeline_kernels.h"
#include <cmath>
#include <cstring>
#include <ctime>
#include <iostream>
#include <utility>
#include <vector>
#include <omp.h>

#define CL_SILENCE_DEPRECATION
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

/* ---- portable wall-clock timer ---- */
static double wall_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---- Global OpenCL state ---- */
static cl_context        g_context      = nullptr;
static cl_command_queue  g_queue        = nullptr;
static cl_program        g_program      = nullptr;
static cl_kernel         g_fft_kernel   = nullptr;  /* Stage 2: FFT magnitude */
static cl_kernel         g_filt_kernel  = nullptr;  /* Stage 3: high-pass filter */
static bool              g_initialized  = false;
static bool              g_gpu_avail    = false;
static int               g_num_gpus     = 0;
static std::vector<cl_device_id> g_all_devices;

#define CHECK_CL(err, msg)                                                 \
    if ((err) != CL_SUCCESS) {                                             \
        std::cerr << "[CPP] OpenCL error: " << (msg)                      \
                  << " (code=" << (err) << ")\n";                          \
        return -1;                                                         \
    }

/* -----------------------------------------------------------------------
 * OpenCL kernel source: TWO separate kernels so Stage 2 and Stage 3 each
 * have their own GPU kernel, matching the pipeline architecture in the report.
 *
 * Kernel 1 — fft_magnitude:
 *   True 2D FFT of each image channel (radix-2 Cooley-Tukey, rows then
 *   columns) returning the complex-magnitude spectrum |re + i·im|, with the
 *   DC term at (0,0) as in numpy.fft.fft2. Run as two dispatches: pass 0
 *   transforms the rows into a complex scratch buffer, pass 1 the columns.
 *
 * Kernel 2 — highpass_filter:
 *   Applies element-wise high-pass mask H(u,v) = 1 - exp(-d²/50) where d is
 *   the wrap-around distance from the DC component at (0,0), suppressing low-freq.
 * ----------------------------------------------------------------------- */
static const char *KERNEL_SRC = R"CL(

/* In-place radix-2 FFT of one line; n must be a power of two. */
#define MAX_FFT_LEN 256
void fft_1d(__private float* re, __private float* im, int n)
{
    for (int i = 1, j = 0; i < n; ++i) {        /* bit-reversal permutation */
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {    /* butterflies */
        float step = -6.28318530718f / (float)len;
        for (int i = 0; i < n; i += len) {
            for (int k = 0; k < len / 2; ++k) {
                float wr = cos(step * (float)k), wi = sin(step * (float)k);
                int u = i + k, v = i + k + len / 2;
                float tr = re[v] * wr - im[v] * wi;
                float ti = re[v] * wi + im[v] * wr;
                re[v] = re[u] - tr;  im[v] = im[u] - ti;
                re[u] += tr;         im[u] += ti;
            }
        }
    }
}

/* Stage 2: 2D FFT magnitude spectrum.
 *
 * Data layout: [batch][height][width][channels] (HWC, channels interleaved).
 * pass 0: one work-item per (image, channel, row); real input -> scratch.
 * pass 1: one work-item per (image, channel, column); scratch -> |spectrum|.
 * scratch holds interleaved (re, im) pairs for every element.
 */
__kernel void fft_magnitude(__global const float* in,
                            __global float* scratch,
                            __global float* out,
                            int width, int height, int channels, int pass)
{
    float re[MAX_FFT_LEN], im[MAX_FFT_LEN];
    int gid  = get_global_id(0);
    int len  = (pass == 0) ? height : width;     /* lines per image-channel */
    int line = gid % len;
    int c    = (gid / len) % channels;
    int img  = gid / (len * channels);
    size_t base = (size_t)img * height * width * channels;

    if (pass == 0) {
        for (int x = 0; x < width; ++x) {
            re[x] = in[base + (size_t)(line * width + x) * channels + c];
            im[x] = 0.0f;
        }
        fft_1d(re, im, width);
        for (int x = 0; x < width; ++x) {
            size_t e = base + (size_t)(line * width + x) * channels + c;
            scratch[2 * e] = re[x];  scratch[2 * e + 1] = im[x];
        }
    } else {
        for (int y = 0; y < height; ++y) {
            size_t e = base + (size_t)(y * width + line) * channels + c;
            re[y] = scratch[2 * e];  im[y] = scratch[2 * e + 1];
        }
        fft_1d(re, im, height);
        for (int y = 0; y < height; ++y) {
            size_t e = base + (size_t)(y * width + line) * channels + c;
            out[e] = sqrt(re[y] * re[y] + im[y] * im[y]);   /* magnitude */
        }
    }
}

/* Stage 3: high-pass filter H(u,v) = 1 - exp(-d^2 / 50)
 *
 * Data layout: [batch][height][width][channels] — HWC, channels interleaved.
 * GWS = total_elements = batch * height * width * channels.
 * Each work-item handles one scalar element; its spatial (y,x) is derived by
 * dividing out the channel stride, so the same mask applies to all channels.
 */
__kernel void highpass_filter(__global const float* in,
                              __global float* out,
                              int width,
                              int height)
{
    int gid = get_global_id(0);

    /* HWC layout: element gid = image_idx*(H*W*3) + y*W*3 + x*3 + c
     * Spatial pixel index within one image (ignoring channel & batch):
     *   spatial_in_image = (gid / 3) % (width * height)          */
    int channels        = 3;
    int pixels_per_img  = width * height;
    int spatial         = (gid / channels) % pixels_per_img;
    int x  = spatial % width;
    int y  = spatial / width;

    int dx = x < width /2 ? x : width  - x;
    int dy = y < height/2 ? y : height - y;
    float d  = sqrt((float)(dx*dx + dy*dy));
    float hp = 1.0f - exp(-d * d / 50.0f);

    out[gid] = in[gid] * hp;
}

)CL";

/* ---- init_opencl ---- */
int init_opencl(int mpi_rank) {
    if (g_initialized) return 0;

    cl_int err;
    cl_uint nplat = 0;
    err = clGetPlatformIDs(0, nullptr, &nplat);
    if (err != CL_SUCCESS || nplat == 0) {
        std::cerr << "[CPP] No OpenCL platforms — CPU fallback active.\n";
        g_gpu_avail = false; return -1;
    }
    std::vector<cl_platform_id> plats(nplat);
    clGetPlatformIDs(nplat, plats.data(), nullptr);
    cl_platform_id plat = plats[0];

    cl_uint ngpu = 0;
    clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, nullptr, &ngpu);

    cl_device_id selected;
    if (ngpu > 0) {
        g_all_devices.resize(ngpu);
        clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, ngpu, g_all_devices.data(), nullptr);
        g_num_gpus = (int)ngpu;
        /* Round-robin device assignment: rank 0 gets GPU 0, rank 1 gets GPU 1,
         * wrapping with modulo. On this single-GPU testbed every rank maps to
         * the same device (all ranks contend for it, see report §5.5); on a
         * multi-GPU node this same line would spread ranks across devices
         * with no other code change required. */
        int idx = mpi_rank % (int)ngpu;
        selected  = g_all_devices[idx];
        char name[256] = {};
        clGetDeviceInfo(selected, CL_DEVICE_NAME, sizeof(name), name, nullptr);
        std::cout << "[CPP] Rank " << mpi_rank << " -> GPU[" << idx << "] "
                  << name << " (" << ngpu << " total)\n";
    } else {
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

    g_context = clCreateContext(nullptr, 1, &selected, nullptr, nullptr, &err);
    CHECK_CL(err, "clCreateContext");
    g_queue = clCreateCommandQueue(g_context, selected, 0, &err);
    CHECK_CL(err, "clCreateCommandQueue");
    g_program = clCreateProgramWithSource(g_context, 1, &KERNEL_SRC, nullptr, &err);
    CHECK_CL(err, "clCreateProgramWithSource");

    err = clBuildProgram(g_program, 1, &selected, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_sz;
        clGetProgramBuildInfo(g_program, selected, CL_PROGRAM_BUILD_LOG,
                              0, nullptr, &log_sz);
        std::vector<char> log(log_sz);
        clGetProgramBuildInfo(g_program, selected, CL_PROGRAM_BUILD_LOG,
                              log_sz, log.data(), nullptr);
        std::cerr << "[CPP] Build log:\n" << log.data() << "\n";
        return -1;
    }

    g_fft_kernel  = clCreateKernel(g_program, "fft_magnitude",   &err);
    CHECK_CL(err, "clCreateKernel(fft_magnitude)");
    g_filt_kernel = clCreateKernel(g_program, "highpass_filter",  &err);
    CHECK_CL(err, "clCreateKernel(highpass_filter)");

    g_initialized = true;
    g_gpu_avail   = true;
    std::cout << "[CPP] OpenCL initialized: 2 kernels (fft_magnitude, highpass_filter).\n";
    return 0;
}

/* ---- cleanup_opencl ---- */
void cleanup_opencl() {
    if (!g_initialized) return;
    clReleaseKernel(g_fft_kernel);
    clReleaseKernel(g_filt_kernel);
    clReleaseProgram(g_program);
    clReleaseCommandQueue(g_queue);
    clReleaseContext(g_context);
    g_initialized = false;
}

/* ================================================================
 * Helper: run ONE kernel (either fft_magnitude or highpass_filter)
 * with timing. Returns 0 on success.
 *
 * Always measures all three phases (H2D transfer, kernel execution, D2H
 * transfer) even when the caller doesn't need the timing breakdown, by
 * calling clFinish() after each step — this is what makes the host-clock
 * phase timings in Table 5.5 of the report possible; driver timestamps on
 * the testbed are unreliable for timing (report §5.5.3).
 * ================================================================ */
static int run_kernel_timed(cl_kernel kern,
                             const float *h_in, float *h_out,
                             size_t data_sz, size_t gws_override,
                             int extra_arg0, int extra_arg1,
                             bool use_extra_args,
                             GpuStageTiming *timing) {
    size_t bytes = data_sz * sizeof(float);
    cl_int err;
    double t0, t1, t2, t3;

    t0 = wall_sec();
    cl_mem d_in  = clCreateBuffer(g_context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  bytes, (void *)h_in, &err);
    if (err != CL_SUCCESS) return -1;
    cl_mem d_out = clCreateBuffer(g_context, CL_MEM_WRITE_ONLY,
                                  bytes, nullptr, &err);
    if (err != CL_SUCCESS) { clReleaseMemObject(d_in); return -1; }
    clFinish(g_queue);
    t1 = wall_sec();

    clSetKernelArg(kern, 0, sizeof(cl_mem), &d_in);
    clSetKernelArg(kern, 1, sizeof(cl_mem), &d_out);
    if (use_extra_args) {
        clSetKernelArg(kern, 2, sizeof(int), &extra_arg0);
        clSetKernelArg(kern, 3, sizeof(int), &extra_arg1);
    } else {
        clSetKernelArg(kern, 2, sizeof(int), &extra_arg0); /* n_elem */
    }
    size_t gws = gws_override;
    err = clEnqueueNDRangeKernel(g_queue, kern, 1, nullptr,
                                  &gws, nullptr, 0, nullptr, nullptr);
    clFinish(g_queue);
    t2 = wall_sec();
    if (err != CL_SUCCESS) { clReleaseMemObject(d_in); clReleaseMemObject(d_out); return -1; }

    err = clEnqueueReadBuffer(g_queue, d_out, CL_TRUE, 0, bytes, h_out, 0, nullptr, nullptr);
    clFinish(g_queue);
    t3 = wall_sec();

    clReleaseMemObject(d_in);
    clReleaseMemObject(d_out);

    if (timing) {
        timing->t_host_to_device = t1 - t0;
        timing->t_kernel         = t2 - t1;
        timing->t_device_to_host = t3 - t2;
        timing->t_total_gpu      = t3 - t0;
    }
    return (err == CL_SUCCESS) ? 0 : -1;
}

/* ================================================================
 * Helper: run the two-pass fft_magnitude kernel with timing.
 *
 * Images are dispatched in chunks of at most FFT_CHUNK_IMAGES so the complex
 * scratch buffer stays bounded even when sequential mode passes a rank's whole
 * subset at once (at P=1 it would otherwise need 5.4 GB).
 * ================================================================ */
static const int FFT_CHUNK_IMAGES = 1024;

static bool is_fft_size(int n) { return n >= 2 && n <= 256 && (n & (n - 1)) == 0; }

static int run_fft_timed(const float *h_in, float *h_out, int batch,
                         int width, int height, int ch, GpuStageTiming *timing) {
    if (!is_fft_size(width) || !is_fft_size(height)) {
        std::cerr << "[CPP] FFT size must be a power of two up to 256, got "
                  << width << "x" << height << "\n";
        return -1;
    }
    double t_h2d = 0, t_kern = 0, t_d2h = 0;
    size_t ppi = (size_t)width * height * ch;
    for (int start = 0; start < batch; start += FFT_CHUNK_IMAGES) {
        int n = (batch - start < FFT_CHUNK_IMAGES) ? batch - start : FFT_CHUNK_IMAGES;
        size_t bytes = (size_t)n * ppi * sizeof(float);
        cl_int err;
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

        clSetKernelArg(g_fft_kernel, 0, sizeof(cl_mem), &d_in);
        clSetKernelArg(g_fft_kernel, 1, sizeof(cl_mem), &d_scr);
        clSetKernelArg(g_fft_kernel, 2, sizeof(cl_mem), &d_out);
        clSetKernelArg(g_fft_kernel, 3, sizeof(int), &width);
        clSetKernelArg(g_fft_kernel, 4, sizeof(int), &height);
        clSetKernelArg(g_fft_kernel, 5, sizeof(int), &ch);
        for (int pass = 0; pass < 2 && err == CL_SUCCESS; ++pass) {
            int lines = (pass == 0) ? height : width;
            size_t gws = (size_t)n * ch * lines;
            clSetKernelArg(g_fft_kernel, 6, sizeof(int), &pass);
            err = clEnqueueNDRangeKernel(g_queue, g_fft_kernel, 1, nullptr,
                                         &gws, nullptr, 0, nullptr, nullptr);
            clFinish(g_queue);   /* pass 1 reads what pass 0 wrote */
        }
        double t2 = wall_sec();
        if (err == CL_SUCCESS)
            err = clEnqueueReadBuffer(g_queue, d_out, CL_TRUE, 0, bytes,
                                      h_out + (size_t)start * ppi, 0, nullptr, nullptr);
        clFinish(g_queue);
        double t3 = wall_sec();

        clReleaseMemObject(d_in);
        clReleaseMemObject(d_scr);
        clReleaseMemObject(d_out);
        if (err != CL_SUCCESS) return -1;
        t_h2d += t1 - t0;  t_kern += t2 - t1;  t_d2h += t3 - t2;
    }
    if (timing) {
        timing->t_host_to_device = t_h2d;
        timing->t_kernel         = t_kern;
        timing->t_device_to_host = t_d2h;
        timing->t_total_gpu      = t_h2d + t_kern + t_d2h;
    }
    return 0;
}

/* ================================================================
 * CPU fallback paths (OpenMP)
 *
 * These mirror the two OpenCL kernels' math exactly (same radix-2 FFT, same
 * high-pass mask) so results are consistent whether the
 * GPU path or this path runs. Only taken if init_opencl() found no GPU
 * device at all; exercised in practice on a machine without OpenCL support,
 * not during the normal Apple M4 GPU runs this report evaluates.
 * ================================================================ */
static void cpu_fft_1d(std::vector<float> &re, std::vector<float> &im, int n) {
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { std::swap(re[i], re[j]); std::swap(im[i], im[j]); }
    }
    for (int len = 2; len <= n; len <<= 1) {
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

static void cpu_fft_magnitude(const float *in, float *out,
                              int batch, int width, int height, int ch) {
    size_t ppi = (size_t)width * height * ch;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < batch; ++i) {
        const float *src = in  + (size_t)i * ppi;
        float       *dst = out + (size_t)i * ppi;
        std::vector<float> sre((size_t)width * height), sim((size_t)width * height);
        std::vector<float> re(width > height ? width : height), im(re.size());
        for (int c = 0; c < ch; ++c) {
            for (int y = 0; y < height; ++y) {          /* rows */
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
            for (int x = 0; x < width; ++x) {           /* columns, magnitude */
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

static void cpu_highpass(const float *in, float *out,
                         int batch, int width, int height, int ch) {
    /* HWC layout: element = [batch][y][x][c]. Apply same H(u,v) to all channels. */
    size_t ppi = (size_t)width * height * ch;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < batch; ++i) {
        const float *src = in  + (size_t)i * ppi;
        float       *dst = out + (size_t)i * ppi;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int dx = x < width /2 ? x : width  - x;
                int dy = y < height/2 ? y : height - y;
                float d  = sqrtf((float)(dx*dx + dy*dy));
                float hp = 1.0f - expf(-d * d / 50.0f);
                int base = (y * width + x) * ch;
                for (int c = 0; c < ch; ++c)
                    dst[base + c] = src[base + c] * hp;
            }
        }
    }
}

/* ================================================================
 * process_images_fft_cpu  —  the CPU fallback path, callable directly so the
 * tests can check it against numpy even on a machine with a GPU.
 * ================================================================ */
int process_images_fft_cpu(const float *image_data, int batch_size,
                           int width, int height, int channels,
                           float *output_data) {
    if (!is_fft_size(width) || !is_fft_size(height)) return -1;
    cpu_fft_magnitude(image_data, output_data, batch_size, width, height, channels);
    return 0;
}

/* ================================================================
 * process_images_fft  —  Stage 2: FFT magnitude (GPU)
 * ================================================================ */
int process_images_fft(const float *image_data, int batch_size,
                       int width, int height, int channels,
                       float *output_data) {
    if (!g_initialized) init_opencl(0);
    if (!g_gpu_avail) {
        cpu_fft_magnitude(image_data, output_data, batch_size, width, height, channels);
        return 0;
    }
    return run_fft_timed(image_data, output_data, batch_size, width, height, channels, nullptr);
}

/* ================================================================
 * apply_frequency_filter_gpu  —  Stage 3: high-pass filter (GPU)
 * ================================================================ */
int apply_frequency_filter_gpu(const float *spectrum_data, int batch_size,
                                int width, int height, int channels,
                                float *filtered_data) {
    if (!g_initialized) init_opencl(0);
    if (!g_gpu_avail) {
        cpu_highpass(spectrum_data, filtered_data, batch_size, width, height, channels);
        return 0;
    }
    /* The filter kernel now iterates over ALL elements (HWC layout).
     * GWS = data_sz = batch * height * width * channels. */
    size_t data_sz = (size_t)batch_size * width * height * channels;
    return run_kernel_timed(g_filt_kernel, spectrum_data, filtered_data,
                             data_sz, data_sz, width, height, true, nullptr);
}

/* ================================================================
 * process_images_frequency_domain  —  combined Stage 2+3 (backward compat.)
 * Calls process_images_fft then apply_frequency_filter_gpu sequentially.
 * ================================================================ */
int process_images_frequency_domain(const float *image_data, int batch_size,
                                    int width, int height, int channels,
                                    float *output_data) {
    if (!g_initialized) init_opencl(0);
    size_t data_sz = (size_t)batch_size * width * height * channels;
    std::vector<float> tmp(data_sz);
    int rc = process_images_fft(image_data, batch_size, width, height, channels,
                                 tmp.data());
    if (rc != 0) return rc;
    return apply_frequency_filter_gpu(tmp.data(), batch_size, width, height, channels,
                                       output_data);
}

/* ================================================================
 * Timed variants
 * ================================================================ */
int process_images_fft_timed(const float *image_data, int batch_size,
                              int width, int height, int channels,
                              float *output_data, GpuStageTiming *timing) {
    if (!g_initialized) init_opencl(0);
    if (!g_gpu_avail) {
        double t0 = wall_sec();
        cpu_fft_magnitude(image_data, output_data, batch_size, width, height, channels);
        double t1 = wall_sec();
        if (timing) { timing->t_host_to_device=0; timing->t_kernel=t1-t0;
                      timing->t_device_to_host=0; timing->t_total_gpu=t1-t0; }
        return 0;
    }
    return run_fft_timed(image_data, output_data, batch_size, width, height, channels, timing);
}

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
                             data_sz, data_sz, width, height, true, timing);
}

/* ================================================================
 * extract_features_cpu  —  Stage 4: mean intensity (OpenMP)
 * ================================================================ */
int extract_features_cpu(const float *processed_data, int batch_size, int width,
                         int height, int channels, float *features_out) {
    size_t ppi = (size_t)width * height * channels;
    /* This is the one OpenMP-parallel stage in the whole pipeline (Stage 4).
     * Each loop iteration (one image) is independent, so a static schedule
     * splits the batch evenly across threads with no synchronisation needed
     * beyond the implicit barrier at the end of the parallel region. */
#pragma omp parallel for schedule(static)
    for (int i = 0; i < batch_size; ++i) {
        const float *row = processed_data + (size_t)i * ppi;
        double sum = 0.0;
        for (size_t j = 0; j < ppi; ++j) sum += row[j];
        features_out[i] = (float)(sum / (double)ppi);
    }
    return 0;
}

int extract_features_cpu_timed(const float *processed_data, int batch_size,
                               int width, int height, int channels,
                               float *features_out, double *t_feature_s) {
    double t0 = wall_sec();
    int rc = extract_features_cpu(processed_data, batch_size, width, height,
                                  channels, features_out);
    if (t_feature_s) *t_feature_s = wall_sec() - t0;
    return rc;
}

/* ================================================================
 * GPU device queries
 *
 * Used by experiments/verify_gpu_profiling.py and the test suite to confirm,
 * independently of the pipeline itself, which physical device is actually
 * present — e.g. printing "Apple M4" rather than trusting that a GPU path
 * was taken.
 * ================================================================ */
int query_gpu_device_count(void) {
    if (g_initialized) return g_num_gpus;
    cl_uint nplat = 0;
    if (clGetPlatformIDs(0, nullptr, &nplat) != CL_SUCCESS || nplat == 0) return 0;
    std::vector<cl_platform_id> plats(nplat);
    clGetPlatformIDs(nplat, plats.data(), nullptr);
    cl_uint total = 0;
    for (cl_uint p = 0; p < nplat; ++p) {
        cl_uint n = 0;
        if (clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, 0, nullptr, &n) == CL_SUCCESS)
            total += n;
    }
    return (int)total;
}

int query_gpu_device_name(int idx, char *buf, int buf_len) {
    if (!buf || buf_len <= 0) return -1;
    cl_uint nplat = 0;
    if (clGetPlatformIDs(0, nullptr, &nplat) != CL_SUCCESS || nplat == 0) return -1;
    std::vector<cl_platform_id> plats(nplat);
    clGetPlatformIDs(nplat, plats.data(), nullptr);
    int seen = 0;
    for (cl_uint p = 0; p < nplat; ++p) {
        cl_uint ngpu = 0;
        if (clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, 0, nullptr, &ngpu) != CL_SUCCESS)
            continue;
        std::vector<cl_device_id> devs(ngpu);
        clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, ngpu, devs.data(), nullptr);
        for (cl_uint d = 0; d < ngpu; ++d) {
            if (seen++ == idx) {
                clGetDeviceInfo(devs[d], CL_DEVICE_NAME,
                                (size_t)buf_len, buf, nullptr);
                buf[buf_len-1] = '\0';
                return 0;
            }
        }
    }
    strncpy(buf, "unknown", (size_t)buf_len);
    buf[buf_len-1] = '\0';
    return -1;
}
