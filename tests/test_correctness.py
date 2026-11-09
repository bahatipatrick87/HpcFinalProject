"""
Correctness tests for the C++ pipeline kernels.

Tests each stage individually using the new split API:
  - process_images_fft          (Stage 2: FFT magnitude, GPU)
  - apply_frequency_filter_gpu  (Stage 3: high-pass filter, GPU)
  - extract_features_cpu        (Stage 4: mean feature, OpenMP)
  - process_images_frequency_domain (combined S2+S3, backward compat)

Run from project root:
    python -m pytest tests/test_correctness.py -v
or:
    python tests/test_correctness.py
"""
import unittest
import numpy as np
import ctypes
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from main_mpi import load_library, configure_c_api


def _ptr(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


class TestPipelineKernels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.lib = load_library()
        configure_c_api(cls.lib)
        cls.lib.cleanup_opencl.argtypes = []
        cls.lib.cleanup_opencl.restype  = None
        cls.lib.init_opencl(0)
        cls.W, cls.H, cls.C = 64, 64, 3

    @classmethod
    def tearDownClass(cls):
        cls.lib.cleanup_opencl()

    # ------------------------------------------------------------------
    # Stage 2: FFT magnitude
    # ------------------------------------------------------------------

    def test_fft_black_image_is_zero(self):
        """FFT of a zero image must produce zero magnitude everywhere."""
        img = np.zeros((1, self.H, self.W, self.C), dtype=np.float32)
        out = np.empty_like(img)
        err = self.lib.process_images_fft(
            _ptr(img), 1, self.W, self.H, self.C, _ptr(out))
        self.assertEqual(err, 0)
        np.testing.assert_allclose(out, 0.0, atol=1e-6,
            err_msg="FFT of black image must be zero")

    def test_fft_produces_nonnegative_magnitude(self):
        """FFT output must be non-negative (it is a magnitude)."""
        rng = np.random.default_rng(42)
        img = rng.random((2, self.H, self.W, self.C), dtype=np.float32)
        out = np.empty_like(img)
        err = self.lib.process_images_fft(
            _ptr(img), 2, self.W, self.H, self.C, _ptr(out))
        self.assertEqual(err, 0)
        self.assertTrue(np.all(out >= 0.0),
            "FFT magnitude must be non-negative")

    def test_fft_matches_numpy(self):
        """GPU kernel must equal |numpy.fft.fft2| per channel, for every image."""
        rng = np.random.default_rng(7)
        img = rng.random((3, self.H, self.W, self.C), dtype=np.float32)
        out = np.empty_like(img)
        err = self.lib.process_images_fft(
            _ptr(img), 3, self.W, self.H, self.C, _ptr(out))
        self.assertEqual(err, 0)
        expected = np.abs(np.fft.fft2(img, axes=(1, 2)))
        np.testing.assert_allclose(out, expected, atol=2e-3, rtol=2e-3,
            err_msg="GPU FFT must match numpy.fft.fft2")

    def test_fft_dc_is_sum_of_pixels(self):
        """The (0,0) term of an FFT is the sum of the channel's pixels."""
        rng = np.random.default_rng(8)
        img = rng.random((1, self.H, self.W, self.C), dtype=np.float32)
        out = np.empty_like(img)
        self.lib.process_images_fft(_ptr(img), 1, self.W, self.H, self.C, _ptr(out))
        np.testing.assert_allclose(out[0, 0, 0], img[0].sum(axis=(0, 1)), rtol=1e-3)

    def test_fft_sinusoid_has_single_peak(self):
        """A pure cosine of frequency 5 must put its energy in bins +5 and -5."""
        x = np.arange(self.W, dtype=np.float32)
        row = np.cos(2 * np.pi * 5 * x / self.W).astype(np.float32)
        img = np.zeros((1, self.H, self.W, self.C), dtype=np.float32)
        img[0, :, :, :] = row[None, :, None]
        out = np.empty_like(img)
        self.lib.process_images_fft(_ptr(img), 1, self.W, self.H, self.C, _ptr(out))
        spec = out[0, :, :, 0]
        self.assertAlmostEqual(float(spec[0, 5]), self.H * self.W / 2, delta=1.0)
        self.assertAlmostEqual(float(spec[0, self.W - 5]), self.H * self.W / 2, delta=1.0)
        spec[0, 5] = spec[0, self.W - 5] = 0.0
        self.assertLess(float(spec.max()), 1e-1)

    def test_fft_large_batch_crosses_chunks(self):
        """Batches larger than one dispatch chunk must still be correct."""
        rng = np.random.default_rng(9)
        img = rng.random((1030, 8, 8, 3), dtype=np.float32)
        out = np.empty_like(img)
        err = self.lib.process_images_fft(_ptr(img), 1030, 8, 8, 3, _ptr(out))
        self.assertEqual(err, 0)
        expected = np.abs(np.fft.fft2(img, axes=(1, 2)))
        np.testing.assert_allclose(out, expected, atol=1e-3, rtol=1e-3)

    def test_fft_rejects_non_power_of_two(self):
        """A radix-2 FFT must refuse sizes that are not powers of two."""
        img = np.zeros((1, 30, 30, 3), dtype=np.float32)
        out = np.empty_like(img)
        self.assertNotEqual(
            self.lib.process_images_fft(_ptr(img), 1, 30, 30, 3, _ptr(out)), 0)

    def test_cpu_fallback_matches_numpy(self):
        """The OpenMP CPU fallback must equal |numpy.fft.fft2| too."""
        rng = np.random.default_rng(10)
        img = rng.random((2, self.H, self.W, self.C), dtype=np.float32)
        out = np.empty_like(img)
        fn = self.lib.process_images_fft_cpu
        fn.restype = ctypes.c_int
        err = fn(_ptr(img), 2, self.W, self.H, self.C, _ptr(out))
        self.assertEqual(err, 0)
        expected = np.abs(np.fft.fft2(img, axes=(1, 2)))
        np.testing.assert_allclose(out, expected, atol=2e-3, rtol=2e-3)

    def test_fft_batch_size_two(self):
        """FFT runs correctly on a batch of 2 images."""
        img = np.ones((2, self.H, self.W, self.C), dtype=np.float32) * 0.5
        out = np.empty_like(img)
        err = self.lib.process_images_fft(
            _ptr(img), 2, self.W, self.H, self.C, _ptr(out))
        self.assertEqual(err, 0)
        self.assertEqual(out.shape, img.shape)

    # ------------------------------------------------------------------
    # Stage 3: high-pass filter
    # ------------------------------------------------------------------

    def test_filter_black_spectrum_stays_zero(self):
        """High-pass filter of a zero spectrum must stay zero."""
        spec = np.zeros((1, self.H, self.W, self.C), dtype=np.float32)
        out  = np.empty_like(spec)
        err  = self.lib.apply_frequency_filter_gpu(
            _ptr(spec), 1, self.W, self.H, self.C, _ptr(out))
        self.assertEqual(err, 0)
        np.testing.assert_allclose(out, 0.0, atol=1e-6,
            err_msg="Filter of zero spectrum must be zero")

    def test_filter_attenuates_dc(self):
        """High-pass filter must attenuate DC component (centre pixel)."""
        # Constant unit spectrum
        spec = np.ones((1, self.H, self.W, self.C), dtype=np.float32)
        out  = np.empty_like(spec)
        err  = self.lib.apply_frequency_filter_gpu(
            _ptr(spec), 1, self.W, self.H, self.C, _ptr(out))
        self.assertEqual(err, 0)
        # The DC pixel (0,0) should have very low gain (H≈0 near origin)
        dc_value = float(out[0, 0, 0, 0])
        self.assertLess(dc_value, 0.1,
            f"DC component should be attenuated, got {dc_value:.4f}")

    def test_filter_preserves_high_freq(self):
        """High-pass filter must pass high-frequency components (H close to 1)."""
        spec = np.ones((1, self.H, self.W, self.C), dtype=np.float32)
        out  = np.empty_like(spec)
        self.lib.apply_frequency_filter_gpu(
            _ptr(spec), 1, self.W, self.H, self.C, _ptr(out))
        # Nyquist corner (H/2, W/2) is far from DC → filter close to 1
        corner = float(out[0, self.H // 2, self.W // 2, 0])
        self.assertGreater(corner, 0.8,
            f"High-freq corner should pass through, got {corner:.4f}")

    # ------------------------------------------------------------------
    # Stage 4: feature extraction
    # ------------------------------------------------------------------

    def test_feature_zero_is_zero(self):
        """Mean of an all-zero image must be 0."""
        data = np.zeros((1, self.H, self.W, self.C), dtype=np.float32)
        feat = np.empty(1, dtype=np.float32)
        err  = self.lib.extract_features_cpu(
            _ptr(data), 1, self.W, self.H, self.C, _ptr(feat))
        self.assertEqual(err, 0)
        self.assertAlmostEqual(float(feat[0]), 0.0, places=6)

    def test_feature_ones_is_one(self):
        """Mean of an all-ones image must be 1."""
        data = np.ones((1, self.H, self.W, self.C), dtype=np.float32)
        feat = np.empty(1, dtype=np.float32)
        err  = self.lib.extract_features_cpu(
            _ptr(data), 1, self.W, self.H, self.C, _ptr(feat))
        self.assertEqual(err, 0)
        self.assertAlmostEqual(float(feat[0]), 1.0, places=4)

    def test_feature_batch_correct(self):
        """Each image in a batch gets its own correct mean."""
        batch = 4
        data  = np.zeros((batch, self.H, self.W, self.C), dtype=np.float32)
        for i in range(batch):
            data[i] = float(i) * 0.25      # means: 0.0, 0.25, 0.5, 0.75
        feat = np.empty(batch, dtype=np.float32)
        err  = self.lib.extract_features_cpu(
            _ptr(data), batch, self.W, self.H, self.C, _ptr(feat))
        self.assertEqual(err, 0)
        for i in range(batch):
            self.assertAlmostEqual(float(feat[i]), i * 0.25, places=4,
                msg=f"Batch item {i}: expected {i*0.25:.2f}, got {feat[i]:.4f}")

    # ------------------------------------------------------------------
    # End-to-end: combined Stage 2+3 (backward-compat function)
    # ------------------------------------------------------------------

    def test_combined_fft_filter_black(self):
        """Combined FFT+filter on a black image should produce 0 features."""
        img  = np.zeros((1, self.H, self.W, self.C), dtype=np.float32)
        proc = np.empty_like(img)
        feat = np.empty(1, dtype=np.float32)
        err  = self.lib.process_images_frequency_domain(
            _ptr(img), 1, self.W, self.H, self.C, _ptr(proc))
        self.assertEqual(err, 0)
        err  = self.lib.extract_features_cpu(
            _ptr(proc), 1, self.W, self.H, self.C, _ptr(feat))
        self.assertEqual(err, 0)
        self.assertAlmostEqual(float(feat[0]), 0.0, places=5)

    def test_combined_fft_filter_nonzero_input(self):
        """Combined FFT+filter on a non-zero image should give a non-negative feature."""
        img  = np.ones((1, self.H, self.W, self.C), dtype=np.float32) * 0.5
        proc = np.empty_like(img)
        feat = np.empty(1, dtype=np.float32)
        self.lib.process_images_frequency_domain(
            _ptr(img), 1, self.W, self.H, self.C, _ptr(proc))
        self.lib.extract_features_cpu(
            _ptr(proc), 1, self.W, self.H, self.C, _ptr(feat))
        self.assertGreaterEqual(float(feat[0]), 0.0)

    # ------------------------------------------------------------------
    # GPU query functions
    # ------------------------------------------------------------------

    def test_gpu_device_count_returns_int(self):
        """query_gpu_device_count must return a non-negative integer."""
        n = self.lib.query_gpu_device_count()
        self.assertGreaterEqual(n, 0)
        print(f"\n[INFO] Detected {n} OpenCL GPU device(s)")

    def test_gpu_device_name_returns_string(self):
        """query_gpu_device_name must populate a buffer for device 0."""
        buf = ctypes.create_string_buffer(256)
        rc  = self.lib.query_gpu_device_name(0, buf, 256)
        name = buf.value.decode(errors="replace")
        print(f"\n[INFO] GPU[0] name: '{name}' (rc={rc})")
        # rc=0 means found; rc=-1 means no GPU (CPU fallback) — both are OK
        self.assertIn(rc, (0, -1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
