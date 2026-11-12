#ifndef PIPELINE_KERNELS_H
#define PIPELINE_KERNELS_H

/* Public C API for the compiled hybrid-parallelism core (pipeline_kernels.cpp).
 * Python (main_mpi.py and the src/stages modules) loads the compiled
 * .dylib/.so via ctypes and calls these functions directly on raw NumPy
 * array memory — no serialization, no copying beyond what OpenCL itself
 * requires. extern "C" disables C++ name mangling so ctypes can resolve the
 * symbols by their plain function names. */
#ifdef __cplusplus
extern "C" {
#endif

/* Initialize OpenCL context. mpi_rank selects GPU via round-robin if multiple
 * GPUs exist. Returns 0 on success, -1 if no GPU found (CPU fallback active). */
int init_opencl(int mpi_rank);

/* Release all OpenCL resources. */
void cleanup_opencl();

/* GPU Stage 2 — FFT Magnitude:
 *   Applies a true 2D FFT (radix-2, per channel) to image_data and writes the
 *   complex-magnitude spectrum to output_data, DC term at (0,0). width and
 *   height must be powers of two up to 256.
 *   Dimensions: batch_size × width × height × channels floats.
 *   Uses OpenCL kernel on GPU if available; OpenMP CPU fallback otherwise.
 *   Returns 0 on success, non-zero on error. */
int process_images_fft(const float *image_data, int batch_size,
                       int width, int height, int channels,
                       float *output_data);

/* CPU fallback path of process_images_fft, callable directly for testing. */
int process_images_fft_cpu(const float *image_data, int batch_size,
                           int width, int height, int channels,
                           float *output_data);

/* GPU Stage 3 — High-Pass Filter:
 *   Applies element-wise high-pass filter H(u,v) = 1 - exp(-d^2/50) to
 *   spectrum_data (result of process_images_fft) and writes the result to
 *   the separate array filtered_data. Same dimensions as above.
 *   Uses OpenCL kernel on GPU if available; OpenMP CPU fallback otherwise.
 *   Returns 0 on success, non-zero on error. */
int apply_frequency_filter_gpu(const float *spectrum_data, int batch_size,
                                int width, int height, int channels,
                                float *filtered_data);

/* Backward-compatible combined call (Stage 2 + Stage 3 in one GPU round-trip).
 * Used only as a fallback by fft_stage.py with older library builds; the
 * pipeline calls the two functions above, one per Python stage. */
int process_images_frequency_domain(const float *image_data, int batch_size,
                                    int width, int height, int channels,
                                    float *output_data);

/* CPU Stage 4 — Feature Extraction:
 *   Computes the mean intensity of each image. An OpenMP parallel loop
 *   gives each thread whole images, so no reduction across threads is needed.
 *   features_out receives one float32 per image.
 *   Returns 0 on success. */
int extract_features_cpu(const float *processed_data, int batch_size, int width,
                         int height, int channels, float *features_out);

/* Stage timing struct — filled by the _timed variants below. All values are
 * wall-clock seconds measured inside the C library. */
typedef struct {
    double t_host_to_device; /* CPU → GPU transfer */
    double t_kernel;         /* OpenCL kernel execution */
    double t_device_to_host; /* GPU → CPU transfer */
    double t_total_gpu;      /* end-to-end GPU stage time */
} GpuStageTiming;

/* Timed variant of process_images_fft. */
int process_images_fft_timed(const float *image_data, int batch_size,
                              int width, int height, int channels,
                              float *output_data, GpuStageTiming *timing);

/* Timed variant of apply_frequency_filter_gpu. */
int apply_frequency_filter_gpu_timed(const float *spectrum_data, int batch_size,
                                      int width, int height, int channels,
                                      float *filtered_data, GpuStageTiming *timing);

/* Timed variant of extract_features_cpu. t_feature_s set to elapsed seconds. */
int extract_features_cpu_timed(const float *processed_data, int batch_size,
                               int width, int height, int channels,
                               float *features_out, double *t_feature_s);

/* Query total number of OpenCL GPU devices visible on this platform. */
int query_gpu_device_count(void);

/* Copy GPU device name at index idx into buf (buf_len chars, NUL-terminated).
 * Returns 0 on success. */
int query_gpu_device_name(int idx, char *buf, int buf_len);

#ifdef __cplusplus
}
#endif

#endif /* PIPELINE_KERNELS_H */
