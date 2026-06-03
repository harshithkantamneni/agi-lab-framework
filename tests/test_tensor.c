/* test_tensor.c -- Comprehensive tests for core tensor library.
 *
 * Tests are organized in groups:
 *   1. Memory Pool: create, alloc, free, reset, peak tracking, alignment
 *   2. Tensor: create, view, get/set, contiguous, nbytes per dtype
 *   3. Operations: add, mul, scale, relu, swiglu, softmax, matmul, rms_norm
 *   4. INT4 quantization: pack/unpack, quantize/dequantize, quantized matmul
 *
 * Built with: clang -std=c17 -g -O0 -mcpu=apple-m3 -fsanitize=address,undefined
 *             -I src/tests -I src/core
 *             tests/test_tensor.c src/core/memory_pool.c src/core/tensor.c src/core/ops.c
 *             -framework Accelerate -o build/tests/test_tensor
 */

#include "../src/tests/unity.h"
#include "memory_pool.h"
#include "ops.h"
#include "tensor.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ========================================================================
 * MEMORY POOL TESTS
 * ======================================================================== */

static void test_pool_create_weights(void) {
    MemoryPool *pool = pool_create(1024 * 1024, POOL_WEIGHTS);
    ASSERT_NOT_NULL(pool);
    ASSERT_EQUAL_INT(POOL_WEIGHTS, pool->type);
    ASSERT_EQUAL_INT(1024 * 1024, (long long)pool->capacity);
    ASSERT_EQUAL_INT(0, (long long)pool_used(pool));
    ASSERT_EQUAL_INT(0, (long long)pool_peak(pool));
    pool_destroy(pool);
}

static void test_pool_create_activations(void) {
    MemoryPool *pool = pool_create(512 * 1024, POOL_ACTIVATIONS);
    ASSERT_NOT_NULL(pool);
    ASSERT_EQUAL_INT(POOL_ACTIVATIONS, pool->type);
    pool_destroy(pool);
}

static void test_pool_create_scratch(void) {
    MemoryPool *pool = pool_create(256 * 1024, POOL_SCRATCH);
    ASSERT_NOT_NULL(pool);
    ASSERT_EQUAL_INT(POOL_SCRATCH, pool->type);
    pool_destroy(pool);
}

static void test_pool_alloc_basic(void) {
    MemoryPool *pool = pool_create(4096, POOL_SCRATCH);
    ASSERT_NOT_NULL(pool);

    void *p1 = pool_alloc(pool, 128, 16);
    ASSERT_NOT_NULL(p1);
    ASSERT_TRUE(pool_used(pool) >= 128);

    void *p2 = pool_alloc(pool, 256, 16);
    ASSERT_NOT_NULL(p2);
    ASSERT_TRUE(pool_used(pool) >= 128 + 256);

    /* Pointers must not overlap */
    ASSERT_TRUE(p1 != p2);

    pool_destroy(pool);
}

static void test_pool_alloc_alignment_16(void) {
    MemoryPool *pool = pool_create(4096, POOL_SCRATCH);
    ASSERT_NOT_NULL(pool);

    void *p = pool_alloc(pool, 100, 16);
    ASSERT_NOT_NULL(p);
    ASSERT_EQUAL_INT(0, (long long)((uintptr_t)p % 16));

    void *p2 = pool_alloc(pool, 33, 16);
    ASSERT_NOT_NULL(p2);
    ASSERT_EQUAL_INT(0, (long long)((uintptr_t)p2 % 16));

    pool_destroy(pool);
}

static void test_pool_alloc_alignment_4096(void) {
    /* Need enough space for 4096-byte alignment + data */
    MemoryPool *pool = pool_create(64 * 1024, POOL_SCRATCH);
    ASSERT_NOT_NULL(pool);

    void *p = pool_alloc(pool, 256, 4096);
    ASSERT_NOT_NULL(p);
    ASSERT_EQUAL_INT(0, (long long)((uintptr_t)p % 4096));

    pool_destroy(pool);
}

static void test_pool_alloc_exhaustion(void) {
    MemoryPool *pool = pool_create(256, POOL_SCRATCH);
    ASSERT_NOT_NULL(pool);

    /* Allocating more than capacity should fail */
    void *p = pool_alloc(pool, 512, 16);
    ASSERT_NULL(p);

    pool_destroy(pool);
}

static void test_pool_peak_tracking(void) {
    MemoryPool *pool = pool_create(4096, POOL_SCRATCH);
    ASSERT_NOT_NULL(pool);

    pool_alloc(pool, 100, 16);
    size_t after_first = pool_used(pool);
    ASSERT_TRUE(after_first >= 100);

    pool_alloc(pool, 200, 16);
    size_t after_second = pool_used(pool);
    ASSERT_TRUE(after_second > after_first);

    size_t peak_before_reset = pool_peak(pool);
    ASSERT_TRUE(peak_before_reset >= after_second);

    pool_reset(pool);
    ASSERT_EQUAL_INT(0, (long long)pool_used(pool));

    /* Peak should NOT decrease after reset */
    ASSERT_TRUE(pool_peak(pool) >= peak_before_reset);

    pool_destroy(pool);
}

static void test_pool_reset_scratch(void) {
    MemoryPool *pool = pool_create(4096, POOL_SCRATCH);
    ASSERT_NOT_NULL(pool);

    pool_alloc(pool, 1024, 16);
    ASSERT_TRUE(pool_used(pool) >= 1024);

    pool_reset(pool);
    ASSERT_EQUAL_INT(0, (long long)pool_used(pool));

    /* After reset, should be able to allocate again from the start */
    void *p = pool_alloc(pool, 1024, 16);
    ASSERT_NOT_NULL(p);

    pool_destroy(pool);
}

static void test_pool_reset_activations(void) {
    MemoryPool *pool = pool_create(4096, POOL_ACTIVATIONS);
    ASSERT_NOT_NULL(pool);

    pool_alloc(pool, 512, 16);
    pool_alloc(pool, 512, 16);
    ASSERT_TRUE(pool_used(pool) >= 1024);

    pool_reset(pool);
    ASSERT_EQUAL_INT(0, (long long)pool_used(pool));

    pool_destroy(pool);
}

static void test_pool_free_weights(void) {
    MemoryPool *pool = pool_create(4096, POOL_WEIGHTS);
    ASSERT_NOT_NULL(pool);

    void *p1 = pool_alloc(pool, 128, 16);
    ASSERT_NOT_NULL(p1);
    size_t used_after_alloc = pool_used(pool);

    pool_free(pool, p1);
    ASSERT_TRUE(pool_used(pool) < used_after_alloc);

    /* Re-allocate should succeed (block returned to free list) */
    void *p2 = pool_alloc(pool, 128, 16);
    ASSERT_NOT_NULL(p2);

    pool_destroy(pool);
}

/* ========================================================================
 * TENSOR TESTS
 * ======================================================================== */

static void test_tensor_create_fp32(void) {
    MemoryPool *pool = pool_create(1024 * 1024, POOL_SCRATCH);
    int32_t shape[] = {4, 8};
    Tensor *t = tensor_create(pool, shape, 2, DTYPE_FP32);

    ASSERT_NOT_NULL(t);
    ASSERT_EQUAL_INT(2, t->ndim);
    ASSERT_EQUAL_INT(4, t->shape[0]);
    ASSERT_EQUAL_INT(8, t->shape[1]);
    ASSERT_EQUAL_INT(DTYPE_FP32, t->dtype);
    ASSERT_FALSE(t->is_view);
    ASSERT_NOT_NULL(t->data);

    /* nbytes = 4 * 8 * 4 = 128 */
    ASSERT_EQUAL_INT(128, (long long)tensor_nbytes(t));
    ASSERT_EQUAL_INT(32, (long long)tensor_numel(t));

    tensor_destroy(t);
    pool_destroy(pool);
}

static void test_tensor_create_fp16(void) {
    MemoryPool *pool = pool_create(1024 * 1024, POOL_SCRATCH);
    int32_t shape[] = {16, 32};
    Tensor *t = tensor_create(pool, shape, 2, DTYPE_FP16);

    ASSERT_NOT_NULL(t);
    /* nbytes = 16 * 32 * 2 = 1024 */
    ASSERT_EQUAL_INT(1024, (long long)tensor_nbytes(t));

    tensor_destroy(t);
    pool_destroy(pool);
}

static void test_tensor_create_int8(void) {
    MemoryPool *pool = pool_create(1024 * 1024, POOL_SCRATCH);
    int32_t shape[] = {100};
    Tensor *t = tensor_create(pool, shape, 1, DTYPE_INT8);

    ASSERT_NOT_NULL(t);
    ASSERT_EQUAL_INT(100, (long long)tensor_nbytes(t));

    tensor_destroy(t);
    pool_destroy(pool);
}

static void test_tensor_create_int4(void) {
    MemoryPool *pool = pool_create(1024 * 1024, POOL_SCRATCH);
    /* INT4: 2 values per byte, so 100 elements = 50 bytes */
    int32_t shape[] = {100};
    Tensor *t = tensor_create(pool, shape, 1, DTYPE_INT4);

    ASSERT_NOT_NULL(t);
    ASSERT_EQUAL_INT(50, (long long)tensor_nbytes(t));
    ASSERT_EQUAL_INT(100, (long long)tensor_numel(t));

    tensor_destroy(t);
    pool_destroy(pool);
}

static void test_tensor_create_4d(void) {
    MemoryPool *pool = pool_create(1024 * 1024, POOL_SCRATCH);
    int32_t shape[] = {2, 32, 8, 128}; /* batch, seq, heads, dim */
    Tensor *t = tensor_create(pool, shape, 4, DTYPE_FP32);

    ASSERT_NOT_NULL(t);
    ASSERT_EQUAL_INT(4, t->ndim);
    ASSERT_EQUAL_INT(2, t->shape[0]);
    ASSERT_EQUAL_INT(32, t->shape[1]);
    ASSERT_EQUAL_INT(8, t->shape[2]);
    ASSERT_EQUAL_INT(128, t->shape[3]);

    /* numel = 2*32*8*128 = 65536, nbytes = 65536*4 = 262144 */
    ASSERT_EQUAL_INT(65536, (long long)tensor_numel(t));
    ASSERT_EQUAL_INT(262144, (long long)tensor_nbytes(t));

    tensor_destroy(t);
    pool_destroy(pool);
}

static void test_tensor_fill_and_get_fp32(void) {
    MemoryPool *pool = pool_create(1024 * 1024, POOL_SCRATCH);
    int32_t shape[] = {3, 4};
    Tensor *t = tensor_create(pool, shape, 2, DTYPE_FP32);

    tensor_fill(t, 3.14f);

    int32_t idx0[] = {0, 0};
    int32_t idx1[] = {1, 2};
    int32_t idx2[] = {2, 3};
    ASSERT_EQUAL_FLOAT(3.14f, tensor_get(t, idx0), 1e-6f);
    ASSERT_EQUAL_FLOAT(3.14f, tensor_get(t, idx1), 1e-6f);
    ASSERT_EQUAL_FLOAT(3.14f, tensor_get(t, idx2), 1e-6f);

    tensor_destroy(t);
    pool_destroy(pool);
}

static void test_tensor_set_and_get_fp32(void) {
    MemoryPool *pool = pool_create(1024 * 1024, POOL_SCRATCH);
    int32_t shape[] = {2, 3};
    Tensor *t = tensor_create(pool, shape, 2, DTYPE_FP32);
    tensor_fill(t, 0.0f);

    int32_t idx[] = {1, 2};
    tensor_set(t, idx, 42.0f);
    ASSERT_EQUAL_FLOAT(42.0f, tensor_get(t, idx), 1e-6f);

    /* Other elements should still be 0 */
    int32_t idx0[] = {0, 0};
    ASSERT_EQUAL_FLOAT(0.0f, tensor_get(t, idx0), 1e-6f);

    tensor_destroy(t);
    pool_destroy(pool);
}

static void test_tensor_set_and_get_int8(void) {
    MemoryPool *pool = pool_create(1024 * 1024, POOL_SCRATCH);
    int32_t shape[] = {10};
    Tensor *t = tensor_create(pool, shape, 1, DTYPE_INT8);
    tensor_fill(t, 0.0f);

    int32_t idx[] = {5};
    tensor_set(t, idx, 42.0f);
    ASSERT_EQUAL_FLOAT(42.0f, tensor_get(t, idx), 0.1f);

    /* Negative values */
    int32_t idx2[] = {3};
    tensor_set(t, idx2, -7.0f);
    ASSERT_EQUAL_FLOAT(-7.0f, tensor_get(t, idx2), 0.1f);

    tensor_destroy(t);
    pool_destroy(pool);
}

static void test_tensor_contiguous(void) {
    MemoryPool *pool = pool_create(1024 * 1024, POOL_SCRATCH);
    int32_t shape[] = {4, 8};
    Tensor *t = tensor_create(pool, shape, 2, DTYPE_FP32);

    ASSERT_TRUE(tensor_is_contiguous(t));

    tensor_destroy(t);
    pool_destroy(pool);
}

static void test_tensor_view_basic(void) {
    MemoryPool *pool = pool_create(1024 * 1024, POOL_SCRATCH);
    int32_t shape[] = {4, 8};
    Tensor *t = tensor_create(pool, shape, 2, DTYPE_FP32);
    tensor_fill(t, 1.0f);

    /* View rows [1, 3) -- shape should be (2, 8) */
    Tensor *v = tensor_view(t, 0, 1, 3);
    ASSERT_NOT_NULL(v);
    ASSERT_TRUE(v->is_view);
    ASSERT_EQUAL_INT(2, v->shape[0]);
    ASSERT_EQUAL_INT(8, v->shape[1]);
    ASSERT_EQUAL_INT(2, v->ndim);

    /* Reading through the view should work */
    int32_t idx[] = {0, 0};
    ASSERT_EQUAL_FLOAT(1.0f, tensor_get(v, idx), 1e-6f);

    /* Writing through the view should affect the original */
    tensor_set(v, idx, 99.0f);
    int32_t orig_idx[] = {1, 0};
    ASSERT_EQUAL_FLOAT(99.0f, tensor_get(t, orig_idx), 1e-6f);

    tensor_destroy(v);
    tensor_destroy(t);
    pool_destroy(pool);
}

static void test_tensor_view_dim1(void) {
    MemoryPool *pool = pool_create(1024 * 1024, POOL_SCRATCH);
    int32_t shape[] = {3, 10};
    Tensor *t = tensor_create(pool, shape, 2, DTYPE_FP32);

    /* Set value at (1, 5) */
    int32_t set_idx[] = {1, 5};
    tensor_set(t, set_idx, 7.7f);

    /* View columns [4, 7) -- shape should be (3, 3) */
    Tensor *v = tensor_view(t, 1, 4, 7);
    ASSERT_NOT_NULL(v);
    ASSERT_EQUAL_INT(3, v->shape[0]);
    ASSERT_EQUAL_INT(3, v->shape[1]);

    /* Element (1,5) in original is (1,1) in view (col 5 - offset 4 = 1) */
    int32_t view_idx[] = {1, 1};
    ASSERT_EQUAL_FLOAT(7.7f, tensor_get(v, view_idx), 1e-5f);

    tensor_destroy(v);
    tensor_destroy(t);
    pool_destroy(pool);
}

static void test_tensor_strides(void) {
    MemoryPool *pool = pool_create(1024 * 1024, POOL_SCRATCH);
    int32_t shape[] = {2, 3, 4};
    Tensor *t = tensor_create(pool, shape, 3, DTYPE_FP32);

    /* Row-major strides: [3*4, 4, 1] = [12, 4, 1] */
    ASSERT_EQUAL_INT(12, t->stride[0]);
    ASSERT_EQUAL_INT(4, t->stride[1]);
    ASSERT_EQUAL_INT(1, t->stride[2]);

    tensor_destroy(t);
    pool_destroy(pool);
}

/* ========================================================================
 * OPERATIONS TESTS
 * ======================================================================== */

static void test_op_add(void) {
    MemoryPool *pool = pool_create(1024 * 1024, POOL_SCRATCH);
    int32_t shape[] = {4};
    Tensor *a = tensor_create(pool, shape, 1, DTYPE_FP32);
    Tensor *b = tensor_create(pool, shape, 1, DTYPE_FP32);
    Tensor *out = tensor_create(pool, shape, 1, DTYPE_FP32);

    float a_vals[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_vals[] = {10.0f, 20.0f, 30.0f, 40.0f};
    for (int i = 0; i < 4; i++) {
        int32_t idx[] = {i};
        tensor_set(a, idx, a_vals[i]);
        tensor_set(b, idx, b_vals[i]);
    }

    op_add(out, a, b);

    float expected[] = {11.0f, 22.0f, 33.0f, 44.0f};
    for (int i = 0; i < 4; i++) {
        int32_t idx[] = {i};
        ASSERT_EQUAL_FLOAT(expected[i], tensor_get(out, idx), 1e-5f);
    }

    tensor_destroy(out);
    tensor_destroy(b);
    tensor_destroy(a);
    pool_destroy(pool);
}

static void test_op_mul(void) {
    MemoryPool *pool = pool_create(1024 * 1024, POOL_SCRATCH);
    int32_t shape[] = {4};
    Tensor *a = tensor_create(pool, shape, 1, DTYPE_FP32);
    Tensor *b = tensor_create(pool, shape, 1, DTYPE_FP32);
    Tensor *out = tensor_create(pool, shape, 1, DTYPE_FP32);

    float a_vals[] = {2.0f, 3.0f, 4.0f, 5.0f};
    float b_vals[] = {0.5f, 1.0f, 1.5f, 2.0f};
    for (int i = 0; i < 4; i++) {
        int32_t idx[] = {i};
        tensor_set(a, idx, a_vals[i]);
        tensor_set(b, idx, b_vals[i]);
    }

    op_mul(out, a, b);

    float expected[] = {1.0f, 3.0f, 6.0f, 10.0f};
    for (int i = 0; i < 4; i++) {
        int32_t idx[] = {i};
        ASSERT_EQUAL_FLOAT(expected[i], tensor_get(out, idx), 1e-5f);
    }

    tensor_destroy(out);
    tensor_destroy(b);
    tensor_destroy(a);
    pool_destroy(pool);
}

static void test_op_scale(void) {
    MemoryPool *pool = pool_create(1024 * 1024, POOL_SCRATCH);
    int32_t shape[] = {3};
    Tensor *a = tensor_create(pool, shape, 1, DTYPE_FP32);
    Tensor *out = tensor_create(pool, shape, 1, DTYPE_FP32);

    float vals[] = {1.0f, 2.0f, 3.0f};
    for (int i = 0; i < 3; i++) {
        int32_t idx[] = {i};
        tensor_set(a, idx, vals[i]);
    }

    op_scale(out, a, 2.5f);

    float expected[] = {2.5f, 5.0f, 7.5f};
    for (int i = 0; i < 3; i++) {
        int32_t idx[] = {i};
        ASSERT_EQUAL_FLOAT(expected[i], tensor_get(out, idx), 1e-5f);
    }

    tensor_destroy(out);
    tensor_destroy(a);
    pool_destroy(pool);
}

static void test_op_relu(void) {
    MemoryPool *pool = pool_create(1024 * 1024, POOL_SCRATCH);
    int32_t shape[] = {5};
    Tensor *a = tensor_create(pool, shape, 1, DTYPE_FP32);
    Tensor *out = tensor_create(pool, shape, 1, DTYPE_FP32);

    float vals[] = {-2.0f, -0.5f, 0.0f, 0.5f, 2.0f};
    for (int i = 0; i < 5; i++) {
        int32_t idx[] = {i};
        tensor_set(a, idx, vals[i]);
    }

    op_relu(out, a);

    float expected[] = {0.0f, 0.0f, 0.0f, 0.5f, 2.0f};
    for (int i = 0; i < 5; i++) {
        int32_t idx[] = {i};
        ASSERT_EQUAL_FLOAT(expected[i], tensor_get(out, idx), 1e-5f);
    }

    tensor_destroy(out);
    tensor_destroy(a);
    pool_destroy(pool);
}

static void test_op_swiglu(void) {
    MemoryPool *pool = pool_create(1024 * 1024, POOL_SCRATCH);
    int32_t shape[] = {3};
    Tensor *gate = tensor_create(pool, shape, 1, DTYPE_FP32);
    Tensor *up = tensor_create(pool, shape, 1, DTYPE_FP32);
    Tensor *out = tensor_create(pool, shape, 1, DTYPE_FP32);

    /* SwiGLU: out = silu(gate) * up, where silu(x) = x * sigmoid(x) */
    float gate_vals[] = {0.0f, 1.0f, -1.0f};
    float up_vals[] = {1.0f, 2.0f, 3.0f};
    for (int i = 0; i < 3; i++) {
        int32_t idx[] = {i};
        tensor_set(gate, idx, gate_vals[i]);
        tensor_set(up, idx, up_vals[i]);
    }

    op_swiglu(out, gate, up);

    /* silu(0) = 0*sigmoid(0) = 0*0.5 = 0.0 -> 0.0 * 1.0 = 0.0
     * silu(1) = 1*sigmoid(1) = 1*0.7311 = 0.7311 -> 0.7311 * 2.0 = 1.4621
     * silu(-1) = -1*sigmoid(-1) = -1*0.2689 = -0.2689 -> -0.2689 * 3.0 = -0.8068 */
    int32_t idx0[] = {0};
    int32_t idx1[] = {1};
    int32_t idx2[] = {2};
    ASSERT_EQUAL_FLOAT(0.0f, tensor_get(out, idx0), 1e-3f);
    ASSERT_EQUAL_FLOAT(1.4621f, tensor_get(out, idx1), 1e-3f);
    ASSERT_EQUAL_FLOAT(-0.8068f, tensor_get(out, idx2), 1e-3f);

    tensor_destroy(out);
    tensor_destroy(up);
    tensor_destroy(gate);
    pool_destroy(pool);
}

static void test_op_softmax(void) {
    MemoryPool *pool = pool_create(1024 * 1024, POOL_SCRATCH);
    int32_t shape[] = {4};
    Tensor *a = tensor_create(pool, shape, 1, DTYPE_FP32);
    Tensor *out = tensor_create(pool, shape, 1, DTYPE_FP32);

    float vals[] = {1.0f, 2.0f, 3.0f, 4.0f};
    for (int i = 0; i < 4; i++) {
        int32_t idx[] = {i};
        tensor_set(a, idx, vals[i]);
    }

    op_softmax(out, a, 0);

    /* Sum of softmax output must be 1.0 */
    float sum = 0.0f;
    for (int i = 0; i < 4; i++) {
        int32_t idx[] = {i};
        float v = tensor_get(out, idx);
        ASSERT_TRUE(v > 0.0f);
        ASSERT_TRUE(v < 1.0f);
        sum += v;
    }
    ASSERT_EQUAL_FLOAT(1.0f, sum, 1e-5f);

    /* Values should be monotonically increasing (since input is sorted) */
    int32_t i0[] = {0}, i1[] = {1}, i2[] = {2}, i3[] = {3};
    ASSERT_TRUE(tensor_get(out, i0) < tensor_get(out, i1));
    ASSERT_TRUE(tensor_get(out, i1) < tensor_get(out, i2));
    ASSERT_TRUE(tensor_get(out, i2) < tensor_get(out, i3));

    tensor_destroy(out);
    tensor_destroy(a);
    pool_destroy(pool);
}

static void test_op_softmax_2d(void) {
    MemoryPool *pool = pool_create(1024 * 1024, POOL_SCRATCH);
    /* 2 rows, 3 cols -- softmax along dim 1 (each row sums to 1) */
    int32_t shape[] = {2, 3};
    Tensor *a = tensor_create(pool, shape, 2, DTYPE_FP32);
    Tensor *out = tensor_create(pool, shape, 2, DTYPE_FP32);

    /* Row 0: [1, 2, 3], Row 1: [4, 5, 6] */
    float vals[2][3] = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
    for (int r = 0; r < 2; r++) {
        for (int c = 0; c < 3; c++) {
            int32_t idx[] = {r, c};
            tensor_set(a, idx, vals[r][c]);
        }
    }

    op_softmax(out, a, 1);

    /* Each row should sum to 1.0 */
    for (int r = 0; r < 2; r++) {
        float row_sum = 0.0f;
        for (int c = 0; c < 3; c++) {
            int32_t idx[] = {r, c};
            row_sum += tensor_get(out, idx);
        }
        ASSERT_EQUAL_FLOAT(1.0f, row_sum, 1e-5f);
    }

    tensor_destroy(out);
    tensor_destroy(a);
    pool_destroy(pool);
}

static void test_op_sum(void) {
    MemoryPool *pool = pool_create(1024 * 1024, POOL_SCRATCH);
    int32_t shape[] = {4};
    Tensor *a = tensor_create(pool, shape, 1, DTYPE_FP32);

    float vals[] = {1.0f, 2.0f, 3.0f, 4.0f};
    for (int i = 0; i < 4; i++) {
        int32_t idx[] = {i};
        tensor_set(a, idx, vals[i]);
    }

    float result = op_sum(a);
    ASSERT_EQUAL_FLOAT(10.0f, result, 1e-5f);

    tensor_destroy(a);
    pool_destroy(pool);
}

static void test_op_max(void) {
    MemoryPool *pool = pool_create(1024 * 1024, POOL_SCRATCH);
    int32_t shape[] = {5};
    Tensor *a = tensor_create(pool, shape, 1, DTYPE_FP32);

    float vals[] = {3.0f, 1.0f, 4.0f, 1.0f, 5.0f};
    for (int i = 0; i < 5; i++) {
        int32_t idx[] = {i};
        tensor_set(a, idx, vals[i]);
    }

    float result = op_max(a);
    ASSERT_EQUAL_FLOAT(5.0f, result, 1e-5f);

    tensor_destroy(a);
    pool_destroy(pool);
}

static void test_op_matmul_2x3_3x2(void) {
    MemoryPool *pool = pool_create(1024 * 1024, POOL_SCRATCH);

    int32_t sa[] = {2, 3};
    int32_t sb[] = {3, 2};
    int32_t so[] = {2, 2};

    Tensor *a = tensor_create(pool, sa, 2, DTYPE_FP32);
    Tensor *b = tensor_create(pool, sb, 2, DTYPE_FP32);
    Tensor *out = tensor_create(pool, so, 2, DTYPE_FP32);

    /* a = [[1, 2, 3], [4, 5, 6]] */
    float a_vals[2][3] = {{1, 2, 3}, {4, 5, 6}};
    for (int r = 0; r < 2; r++)
        for (int c = 0; c < 3; c++) {
            int32_t idx[] = {r, c};
            tensor_set(a, idx, a_vals[r][c]);
        }

    /* b = [[7, 8], [9, 10], [11, 12]] */
    float b_vals[3][2] = {{7, 8}, {9, 10}, {11, 12}};
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 2; c++) {
            int32_t idx[] = {r, c};
            tensor_set(b, idx, b_vals[r][c]);
        }

    op_matmul(out, a, b);

    /* Expected: [[58, 64], [139, 154]] */
    int32_t i00[] = {0, 0}, i01[] = {0, 1}, i10[] = {1, 0}, i11[] = {1, 1};
    ASSERT_EQUAL_FLOAT(58.0f, tensor_get(out, i00), 1e-3f);
    ASSERT_EQUAL_FLOAT(64.0f, tensor_get(out, i01), 1e-3f);
    ASSERT_EQUAL_FLOAT(139.0f, tensor_get(out, i10), 1e-3f);
    ASSERT_EQUAL_FLOAT(154.0f, tensor_get(out, i11), 1e-3f);

    tensor_destroy(out);
    tensor_destroy(b);
    tensor_destroy(a);
    pool_destroy(pool);
}

static void test_op_matmul_identity(void) {
    MemoryPool *pool = pool_create(1024 * 1024, POOL_SCRATCH);

    int32_t shape[] = {3, 3};
    Tensor *a = tensor_create(pool, shape, 2, DTYPE_FP32);
    Tensor *eye = tensor_create(pool, shape, 2, DTYPE_FP32);
    Tensor *out = tensor_create(pool, shape, 2, DTYPE_FP32);

    /* Fill a with arbitrary values */
    float a_vals[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    tensor_fill(eye, 0.0f);
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            int32_t idx[] = {r, c};
            tensor_set(a, idx, a_vals[r][c]);
        }
        int32_t diag[] = {r, r};
        tensor_set(eye, diag, 1.0f);
    }

    op_matmul(out, a, eye);

    /* a @ I = a */
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) {
            int32_t idx[] = {r, c};
            ASSERT_EQUAL_FLOAT(a_vals[r][c], tensor_get(out, idx), 1e-5f);
        }

    tensor_destroy(out);
    tensor_destroy(eye);
    tensor_destroy(a);
    pool_destroy(pool);
}

static void test_op_rms_norm(void) {
    MemoryPool *pool = pool_create(1024 * 1024, POOL_SCRATCH);
    int32_t shape[] = {4};
    Tensor *a = tensor_create(pool, shape, 1, DTYPE_FP32);
    Tensor *weight = tensor_create(pool, shape, 1, DTYPE_FP32);
    Tensor *out = tensor_create(pool, shape, 1, DTYPE_FP32);

    float vals[] = {1.0f, 2.0f, 3.0f, 4.0f};
    for (int i = 0; i < 4; i++) {
        int32_t idx[] = {i};
        tensor_set(a, idx, vals[i]);
        tensor_set(weight, idx, 1.0f); /* unit weight */
    }

    op_rms_norm(out, a, weight, 1e-6f);

    /* rms = sqrt(mean(1+4+9+16)) = sqrt(30/4) = sqrt(7.5) = 2.7386... */
    /* Normalized: [1/2.7386, 2/2.7386, 3/2.7386, 4/2.7386] */
    float rms = sqrtf((1.0f + 4.0f + 9.0f + 16.0f) / 4.0f + 1e-6f);
    for (int i = 0; i < 4; i++) {
        int32_t idx[] = {i};
        float expected = vals[i] / rms;
        ASSERT_EQUAL_FLOAT(expected, tensor_get(out, idx), 1e-4f);
    }

    tensor_destroy(out);
    tensor_destroy(weight);
    tensor_destroy(a);
    pool_destroy(pool);
}

static void test_op_rms_norm_with_weight(void) {
    MemoryPool *pool = pool_create(1024 * 1024, POOL_SCRATCH);
    int32_t shape[] = {3};
    Tensor *a = tensor_create(pool, shape, 1, DTYPE_FP32);
    Tensor *weight = tensor_create(pool, shape, 1, DTYPE_FP32);
    Tensor *out = tensor_create(pool, shape, 1, DTYPE_FP32);

    float a_vals[] = {2.0f, 4.0f, 6.0f};
    float w_vals[] = {0.5f, 1.0f, 2.0f};
    for (int i = 0; i < 3; i++) {
        int32_t idx[] = {i};
        tensor_set(a, idx, a_vals[i]);
        tensor_set(weight, idx, w_vals[i]);
    }

    op_rms_norm(out, a, weight, 1e-6f);

    float rms = sqrtf((4.0f + 16.0f + 36.0f) / 3.0f + 1e-6f);
    for (int i = 0; i < 3; i++) {
        int32_t idx[] = {i};
        float expected = (a_vals[i] / rms) * w_vals[i];
        ASSERT_EQUAL_FLOAT(expected, tensor_get(out, idx), 1e-4f);
    }

    tensor_destroy(out);
    tensor_destroy(weight);
    tensor_destroy(a);
    pool_destroy(pool);
}

/* ========================================================================
 * INT4 QUANTIZATION TESTS
 * ======================================================================== */

static void test_int4_pack_unpack(void) {
    /* Test all 16x16 combinations of 4-bit values */
    for (int lo = -8; lo <= 7; lo++) {
        for (int hi = -8; hi <= 7; hi++) {
            uint8_t packed = int4_pack((int8_t)lo, (int8_t)hi);
            int8_t got_lo = int4_unpack_lo(packed);
            int8_t got_hi = int4_unpack_hi(packed);
            ASSERT_EQUAL_INT(lo, (int)got_lo);
            ASSERT_EQUAL_INT(hi, (int)got_hi);
        }
    }
}

static void test_int4_specific_values(void) {
    /* Zero */
    uint8_t p = int4_pack(0, 0);
    ASSERT_EQUAL_INT(0, (int)int4_unpack_lo(p));
    ASSERT_EQUAL_INT(0, (int)int4_unpack_hi(p));

    /* Max positive */
    p = int4_pack(7, 7);
    ASSERT_EQUAL_INT(7, (int)int4_unpack_lo(p));
    ASSERT_EQUAL_INT(7, (int)int4_unpack_hi(p));

    /* Min negative */
    p = int4_pack(-8, -8);
    ASSERT_EQUAL_INT(-8, (int)int4_unpack_lo(p));
    ASSERT_EQUAL_INT(-8, (int)int4_unpack_hi(p));

    /* Mixed */
    p = int4_pack(-1, 3);
    ASSERT_EQUAL_INT(-1, (int)int4_unpack_lo(p));
    ASSERT_EQUAL_INT(3, (int)int4_unpack_hi(p));
}

static void test_quantize_dequantize_roundtrip(void) {
    MemoryPool *pool = pool_create(1024 * 1024, POOL_SCRATCH);

    int32_t shape[] = {8};
    Tensor *fp32 = tensor_create(pool, shape, 1, DTYPE_FP32);
    Tensor *q4 = tensor_create(pool, shape, 1, DTYPE_INT4);
    Tensor *deq = tensor_create(pool, shape, 1, DTYPE_FP32);

    /* Values in a small range that INT4 can represent without too much loss */
    float vals[] = {-1.0f, -0.5f, -0.25f, 0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
    for (int i = 0; i < 8; i++) {
        int32_t idx[] = {i};
        tensor_set(fp32, idx, vals[i]);
    }

    op_quantize_int4(q4, fp32);
    op_dequantize(deq, q4);

    /* Roundtrip should be approximate (quantization error expected) */
    for (int i = 0; i < 8; i++) {
        int32_t idx[] = {i};
        float orig = vals[i];
        float recovered = tensor_get(deq, idx);
        /* Allow ~15% error from 4-bit quantization (16 levels over range) */
        ASSERT_EQUAL_FLOAT(orig, recovered, 0.2f);
    }

    tensor_destroy(deq);
    tensor_destroy(q4);
    tensor_destroy(fp32);
    pool_destroy(pool);
}

static void test_op_matmul_quantized(void) {
    /* Compare quantized matmul against FP32 reference.
     * Use small values to limit quantization error. */
    MemoryPool *pool = pool_create(1024 * 1024, POOL_SCRATCH);

    int32_t sa[] = {2, 4};
    int32_t sb[] = {4, 3};
    int32_t so[] = {2, 3};

    /* FP32 reference */
    Tensor *a_fp32 = tensor_create(pool, sa, 2, DTYPE_FP32);
    Tensor *b_fp32 = tensor_create(pool, sb, 2, DTYPE_FP32);
    Tensor *ref_out = tensor_create(pool, so, 2, DTYPE_FP32);

    /* Small values so quantization error is manageable */
    float a_data[2][4] = {{0.1f, 0.2f, 0.3f, 0.4f}, {0.5f, 0.6f, 0.7f, 0.8f}};
    float b_data[4][3] = {
        {0.1f, 0.2f, 0.3f}, {0.4f, 0.5f, 0.6f}, {0.7f, 0.8f, 0.9f}, {1.0f, 1.1f, 1.2f}};

    for (int r = 0; r < 2; r++)
        for (int c = 0; c < 4; c++) {
            int32_t idx[] = {r, c};
            tensor_set(a_fp32, idx, a_data[r][c]);
        }
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 3; c++) {
            int32_t idx[] = {r, c};
            tensor_set(b_fp32, idx, b_data[r][c]);
        }

    /* FP32 reference matmul */
    op_matmul(ref_out, a_fp32, b_fp32);

    /* Quantized path: a as FP16, b as INT4 */
    Tensor *a_fp16 = tensor_create(pool, sa, 2, DTYPE_FP16);
    Tensor *b_int4 = tensor_create(pool, sb, 2, DTYPE_INT4);
    Tensor *q_out = tensor_create(pool, so, 2, DTYPE_FP16);

    /* Copy a to FP16 */
    for (int r = 0; r < 2; r++)
        for (int c = 0; c < 4; c++) {
            int32_t idx[] = {r, c};
            tensor_set(a_fp16, idx, a_data[r][c]);
        }

    /* Quantize b to INT4 */
    op_quantize_int4(b_int4, b_fp32);

    /* Quantized matmul */
    op_matmul_q4(q_out, a_fp16, b_int4);

    /* Compare: allow larger tolerance due to INT4 quantization */
    for (int r = 0; r < 2; r++)
        for (int c = 0; c < 3; c++) {
            int32_t idx[] = {r, c};
            float ref = tensor_get(ref_out, idx);
            float got = tensor_get(q_out, idx);
            /* INT4 has 16 levels, so significant error is expected */
            ASSERT_EQUAL_FLOAT(ref, got, 0.5f);
        }

    tensor_destroy(q_out);
    tensor_destroy(b_int4);
    tensor_destroy(a_fp16);
    tensor_destroy(ref_out);
    tensor_destroy(b_fp32);
    tensor_destroy(a_fp32);
    pool_destroy(pool);
}

/* ========================================================================
 * DTYPE HELPER TESTS
 * ======================================================================== */

static void test_dtype_size(void) {
    ASSERT_EQUAL_INT(4, (long long)dtype_size(DTYPE_FP32));
    ASSERT_EQUAL_INT(2, (long long)dtype_size(DTYPE_FP16));
    ASSERT_EQUAL_INT(1, (long long)dtype_size(DTYPE_INT8));
    ASSERT_EQUAL_INT(0, (long long)dtype_size(DTYPE_INT4)); /* special case */
}

static void test_dtype_name(void) {
    ASSERT_EQUAL_STR("fp32", dtype_name(DTYPE_FP32));
    ASSERT_EQUAL_STR("fp16", dtype_name(DTYPE_FP16));
    ASSERT_EQUAL_STR("int8", dtype_name(DTYPE_INT8));
    ASSERT_EQUAL_STR("int4", dtype_name(DTYPE_INT4));
}

/* ========================================================================
 * MAIN
 * ======================================================================== */

int main(void) {
    printf("\n========================================\n");
    printf("  Core Tensor Library Tests\n");
    printf("========================================\n\n");

    /* Memory Pool */
    printf("--- Memory Pool ---\n");
    RUN_TEST(test_pool_create_weights);
    RUN_TEST(test_pool_create_activations);
    RUN_TEST(test_pool_create_scratch);
    RUN_TEST(test_pool_alloc_basic);
    RUN_TEST(test_pool_alloc_alignment_16);
    RUN_TEST(test_pool_alloc_alignment_4096);
    RUN_TEST(test_pool_alloc_exhaustion);
    RUN_TEST(test_pool_peak_tracking);
    RUN_TEST(test_pool_reset_scratch);
    RUN_TEST(test_pool_reset_activations);
    RUN_TEST(test_pool_free_weights);

    /* Tensor */
    printf("\n--- Tensor ---\n");
    RUN_TEST(test_tensor_create_fp32);
    RUN_TEST(test_tensor_create_fp16);
    RUN_TEST(test_tensor_create_int8);
    RUN_TEST(test_tensor_create_int4);
    RUN_TEST(test_tensor_create_4d);
    RUN_TEST(test_tensor_fill_and_get_fp32);
    RUN_TEST(test_tensor_set_and_get_fp32);
    RUN_TEST(test_tensor_set_and_get_int8);
    RUN_TEST(test_tensor_contiguous);
    RUN_TEST(test_tensor_view_basic);
    RUN_TEST(test_tensor_view_dim1);
    RUN_TEST(test_tensor_strides);

    /* Dtype helpers */
    printf("\n--- DType Helpers ---\n");
    RUN_TEST(test_dtype_size);
    RUN_TEST(test_dtype_name);

    /* Operations */
    printf("\n--- Operations ---\n");
    RUN_TEST(test_op_add);
    RUN_TEST(test_op_mul);
    RUN_TEST(test_op_scale);
    RUN_TEST(test_op_relu);
    RUN_TEST(test_op_swiglu);
    RUN_TEST(test_op_softmax);
    RUN_TEST(test_op_softmax_2d);
    RUN_TEST(test_op_sum);
    RUN_TEST(test_op_max);
    RUN_TEST(test_op_matmul_2x3_3x2);
    RUN_TEST(test_op_matmul_identity);
    RUN_TEST(test_op_rms_norm);
    RUN_TEST(test_op_rms_norm_with_weight);

    /* INT4 Quantization */
    printf("\n--- INT4 Quantization ---\n");
    RUN_TEST(test_int4_pack_unpack);
    RUN_TEST(test_int4_specific_values);
    RUN_TEST(test_quantize_dequantize_roundtrip);
    RUN_TEST(test_op_matmul_quantized);

    TEST_REPORT();
}
