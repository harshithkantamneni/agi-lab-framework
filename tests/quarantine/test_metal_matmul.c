/* test_metal_matmul.c -- Tests for Metal GPU-accelerated FP32 matmul.
 *
 * Verifies that the Metal GPU matmul produces results matching the CPU
 * Accelerate cblas_sgemm within floating-point tolerance (1e-4).
 * Also benchmarks GPU vs CPU for various matrix sizes.
 *
 * Build: see Makefile target test_metal_matmul
 * Requires: Metal device, default.metallib in build/
 */

/* NOTE: relative paths adjusted during Cycle 30 quarantine move.
 * Build rule passes -Isrc/swift -Isrc/core -Isrc/tests so header names
 * resolve without leading "../src/swift/" / "../src/tests/" prefix. */
#include "metal_bridge.h"
#include "unity.h"

#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* ---- Helpers ---- */

/* Fill buffer with deterministic pseudo-random floats in [-1, 1]. */
static void fill_random(float *buf, int n, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < n; i++) {
        buf[i] = 2.0f * ((float)rand() / (float)RAND_MAX) - 1.0f;
    }
}

/* CPU reference matmul using Accelerate cblas_sgemm. */
static void cpu_matmul(const float *A, const float *B, float *C, int M, int N,
                       int K) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K,
                B, N, 0.0f, C, N);
}

/* Compute max absolute difference between two buffers. */
static float max_abs_diff(const float *a, const float *b, int n) {
    float maxd = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > maxd)
            maxd = d;
    }
    return maxd;
}

/* Get time in seconds. */
static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ---- Tests ---- */

static void test_metal_init(void) {
    int ret = metal_init();
    ASSERT_EQUAL_INT(0, ret);
    ASSERT_EQUAL_INT(1, metal_is_ready());
}

static void test_metal_init_idempotent(void) {
    /* Calling metal_init() again should be a no-op success. */
    int ret = metal_init();
    ASSERT_EQUAL_INT(0, ret);
    ASSERT_EQUAL_INT(1, metal_is_ready());
}

static void test_matmul_small_2x2(void) {
    /* 2x2 @ 2x2 -- verifiable by hand. */
    float A[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float B[] = {5.0f, 6.0f, 7.0f, 8.0f};
    float C_gpu[4] = {0};
    float C_ref[4] = {0};

    metal_matmul(A, B, C_gpu, 2, 2, 2);
    cpu_matmul(A, B, C_ref, 2, 2, 2);

    /* Expected: C = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
     *             = [[19, 22], [43, 50]] */
    ASSERT_EQUAL_FLOAT(19.0f, C_gpu[0], 1e-4f);
    ASSERT_EQUAL_FLOAT(22.0f, C_gpu[1], 1e-4f);
    ASSERT_EQUAL_FLOAT(43.0f, C_gpu[2], 1e-4f);
    ASSERT_EQUAL_FLOAT(50.0f, C_gpu[3], 1e-4f);

    float diff = max_abs_diff(C_gpu, C_ref, 4);
    ASSERT_TRUE(diff < 1e-4f);
}

static void test_matmul_identity(void) {
    /* A @ eye = A for 4x4. */
    int N = 4;
    float A[16];
    float eye[16] = {0};
    float C_gpu[16] = {0};

    fill_random(A, 16, 42);
    for (int i = 0; i < N; i++)
        eye[i * N + i] = 1.0f;

    metal_matmul(A, eye, C_gpu, N, N, N);

    for (int i = 0; i < 16; i++) {
        ASSERT_EQUAL_FLOAT(A[i], C_gpu[i], 1e-4f);
    }
}

static void test_matmul_nonsquare(void) {
    /* Non-square: A [3,5] @ B [5,7] = C [3,7]. */
    int M = 3, K = 5, N = 7;
    float A[15], B[35];
    float C_gpu[21] = {0};
    float C_ref[21] = {0};

    fill_random(A, M * K, 123);
    fill_random(B, K * N, 456);

    metal_matmul(A, B, C_gpu, M, N, K);
    cpu_matmul(A, B, C_ref, M, N, K);

    float diff = max_abs_diff(C_gpu, C_ref, M * N);
    ASSERT_TRUE(diff < 1e-4f);
}

static void test_matmul_128x128(void) {
    int M = 128, K = 128, N = 128;
    float *A = (float *)malloc((size_t)(M * K) * sizeof(float));
    float *B = (float *)malloc((size_t)(K * N) * sizeof(float));
    float *C_gpu = (float *)malloc((size_t)(M * N) * sizeof(float));
    float *C_ref = (float *)malloc((size_t)(M * N) * sizeof(float));

    fill_random(A, M * K, 100);
    fill_random(B, K * N, 200);

    metal_matmul(A, B, C_gpu, M, N, K);
    cpu_matmul(A, B, C_ref, M, N, K);

    float diff = max_abs_diff(C_gpu, C_ref, M * N);
    ASSERT_TRUE(diff < 1e-3f);

    free(A);
    free(B);
    free(C_gpu);
    free(C_ref);
}

static void test_matmul_256x256(void) {
    int M = 256, K = 256, N = 256;
    float *A = (float *)malloc((size_t)(M * K) * sizeof(float));
    float *B = (float *)malloc((size_t)(K * N) * sizeof(float));
    float *C_gpu = (float *)malloc((size_t)(M * N) * sizeof(float));
    float *C_ref = (float *)malloc((size_t)(M * N) * sizeof(float));

    fill_random(A, M * K, 300);
    fill_random(B, K * N, 400);

    metal_matmul(A, B, C_gpu, M, N, K);
    cpu_matmul(A, B, C_ref, M, N, K);

    float diff = max_abs_diff(C_gpu, C_ref, M * N);
    ASSERT_TRUE(diff < 1e-2f);

    free(A);
    free(B);
    free(C_gpu);
    free(C_ref);
}

static void test_matmul_odd_sizes(void) {
    /* Test non-power-of-2 sizes to verify boundary handling. */
    int M = 33, K = 65, N = 17;
    float *A = (float *)malloc((size_t)(M * K) * sizeof(float));
    float *B = (float *)malloc((size_t)(K * N) * sizeof(float));
    float *C_gpu = (float *)malloc((size_t)(M * N) * sizeof(float));
    float *C_ref = (float *)malloc((size_t)(M * N) * sizeof(float));

    fill_random(A, M * K, 777);
    fill_random(B, K * N, 888);

    metal_matmul(A, B, C_gpu, M, N, K);
    cpu_matmul(A, B, C_ref, M, N, K);

    float diff = max_abs_diff(C_gpu, C_ref, M * N);
    ASSERT_TRUE(diff < 1e-3f);

    free(A);
    free(B);
    free(C_gpu);
    free(C_ref);
}

static void test_matmul_single_row(void) {
    /* Vector-matrix multiply: [1, K] @ [K, N] = [1, N]. */
    int M = 1, K = 64, N = 128;
    float *A = (float *)malloc((size_t)(M * K) * sizeof(float));
    float *B = (float *)malloc((size_t)(K * N) * sizeof(float));
    float *C_gpu = (float *)malloc((size_t)(M * N) * sizeof(float));
    float *C_ref = (float *)malloc((size_t)(M * N) * sizeof(float));

    fill_random(A, M * K, 555);
    fill_random(B, K * N, 666);

    metal_matmul(A, B, C_gpu, M, N, K);
    cpu_matmul(A, B, C_ref, M, N, K);

    float diff = max_abs_diff(C_gpu, C_ref, M * N);
    ASSERT_TRUE(diff < 1e-4f);

    free(A);
    free(B);
    free(C_gpu);
    free(C_ref);
}

static void test_matmul_zeros(void) {
    /* Zero matrix should give zero output. */
    int M = 16, K = 16, N = 16;
    float A[256] = {0};
    float B[256];
    float C_gpu[256] = {0};

    fill_random(B, 256, 111);
    metal_matmul(A, B, C_gpu, M, N, K);

    for (int i = 0; i < M * N; i++) {
        ASSERT_EQUAL_FLOAT(0.0f, C_gpu[i], 1e-6f);
    }
}

static void test_metal_cleanup(void) {
    metal_cleanup();
    ASSERT_EQUAL_INT(0, metal_is_ready());

    /* Re-init for benchmark */
    int ret = metal_init();
    ASSERT_EQUAL_INT(0, ret);
}

/* ---- Benchmark ---- */

static void benchmark_matmul(int M, int N, int K) {
    float *A = (float *)malloc((size_t)(M * K) * sizeof(float));
    float *B = (float *)malloc((size_t)(K * N) * sizeof(float));
    float *C = (float *)malloc((size_t)(M * N) * sizeof(float));

    fill_random(A, M * K, 42);
    fill_random(B, K * N, 43);

    int warmup = 3;
    int iters = 10;

    /* Warmup GPU */
    for (int i = 0; i < warmup; i++) {
        metal_matmul(A, B, C, M, N, K);
    }

    /* Benchmark GPU */
    double gpu_start = get_time_sec();
    for (int i = 0; i < iters; i++) {
        metal_matmul(A, B, C, M, N, K);
    }
    double gpu_time = (get_time_sec() - gpu_start) / iters;

    /* Warmup CPU */
    for (int i = 0; i < warmup; i++) {
        cpu_matmul(A, B, C, M, N, K);
    }

    /* Benchmark CPU */
    double cpu_start = get_time_sec();
    for (int i = 0; i < iters; i++) {
        cpu_matmul(A, B, C, M, N, K);
    }
    double cpu_time = (get_time_sec() - cpu_start) / iters;

    double gflops = 2.0 * M * N * K / 1e9;
    printf("    %4dx%4dx%4d  GPU: %.3f ms (%.1f GFLOPS)  CPU: %.3f ms (%.1f "
           "GFLOPS)  Speedup: %.2fx\n",
           M, N, K, gpu_time * 1000.0, gflops / gpu_time,
           cpu_time * 1000.0, gflops / cpu_time, cpu_time / gpu_time);

    free(A);
    free(B);
    free(C);
}

/* ---- Main ---- */

int main(void) {
    printf("=== Metal GPU Matmul Tests ===\n\n");

    RUN_TEST(test_metal_init);
    RUN_TEST(test_metal_init_idempotent);
    RUN_TEST(test_matmul_small_2x2);
    RUN_TEST(test_matmul_identity);
    RUN_TEST(test_matmul_nonsquare);
    RUN_TEST(test_matmul_128x128);
    RUN_TEST(test_matmul_256x256);
    RUN_TEST(test_matmul_odd_sizes);
    RUN_TEST(test_matmul_single_row);
    RUN_TEST(test_matmul_zeros);
    RUN_TEST(test_metal_cleanup);

    printf("\n=== GPU vs CPU Benchmark ===\n");
    benchmark_matmul(128, 128, 128);
    benchmark_matmul(256, 256, 256);
    benchmark_matmul(512, 512, 512);
    benchmark_matmul(1024, 1024, 1024);

    metal_cleanup();
    TEST_REPORT();
}
