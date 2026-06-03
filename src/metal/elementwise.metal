// elementwise.metal — Element-wise Operations for M3 Pro
//
// Operations:
//   1. SwiGLU: output = silu(gate) * up, where silu(x) = x * sigmoid(x)
//   2. Add: output = a + b
//   3. Multiply: output = a * b
//   4. Scale: output = input * scalar
//   5. GELU (approximate): x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//   6. RoPE (Rotary Position Embedding): applies rotation to query/key vectors
//
// All kernels use 1D grid dispatch with 256-thread threadgroups.
// Each thread processes 4 elements (vectorized) for memory coalescing.
//
// Threadgroup: 256 threads (8 SIMD groups of 32)
// Each thread processes 4 consecutive half values.
// Total throughput: 1024 elements per threadgroup per dispatch.

#include <metal_stdlib>
using namespace metal;

// --- SwiGLU Activation ---
// SwiGLU(gate, up) = silu(gate) * up
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
// Used in expert FFN: output = SwiGLU(W_gate * x, W_up * x)
kernel void swiglu_fp16(
    device const half* gate      [[buffer(0)]],   // [N]
    device const half* up        [[buffer(1)]],   // [N]
    device half* output          [[buffer(2)]],   // [N]
    constant uint& N             [[buffer(3)]],
    uint gid                     [[thread_position_in_grid]]
) {
    uint idx = gid * 4;
    if (idx >= N) return;

    // Process 4 elements for memory coalescing
    uint remaining = min(N - idx, 4u);
    for (uint i = 0; i < remaining; i++) {
        float x = float(gate[idx + i]);
        float silu_x = x / (1.0f + exp(-x));
        output[idx + i] = half(silu_x * float(up[idx + i]));
    }
}

// --- Element-wise Add ---
kernel void add_fp16(
    device const half* a         [[buffer(0)]],
    device const half* b         [[buffer(1)]],
    device half* output          [[buffer(2)]],
    constant uint& N             [[buffer(3)]],
    uint gid                     [[thread_position_in_grid]]
) {
    uint idx = gid * 4;
    if (idx >= N) return;

    uint remaining = min(N - idx, 4u);
    for (uint i = 0; i < remaining; i++) {
        output[idx + i] = a[idx + i] + b[idx + i];
    }
}

// --- Element-wise Multiply ---
kernel void mul_fp16(
    device const half* a         [[buffer(0)]],
    device const half* b         [[buffer(1)]],
    device half* output          [[buffer(2)]],
    constant uint& N             [[buffer(3)]],
    uint gid                     [[thread_position_in_grid]]
) {
    uint idx = gid * 4;
    if (idx >= N) return;

    uint remaining = min(N - idx, 4u);
    for (uint i = 0; i < remaining; i++) {
        output[idx + i] = a[idx + i] * b[idx + i];
    }
}

// --- Scale (multiply by scalar) ---
kernel void scale_fp16(
    device const half* input     [[buffer(0)]],
    device half* output          [[buffer(1)]],
    constant float& scalar       [[buffer(2)]],
    constant uint& N             [[buffer(3)]],
    uint gid                     [[thread_position_in_grid]]
) {
    uint idx = gid * 4;
    if (idx >= N) return;

    half s = half(scalar);
    uint remaining = min(N - idx, 4u);
    for (uint i = 0; i < remaining; i++) {
        output[idx + i] = input[idx + i] * s;
    }
}

// --- GELU Approximate ---
// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
kernel void gelu_fp16(
    device const half* input     [[buffer(0)]],
    device half* output          [[buffer(1)]],
    constant uint& N             [[buffer(2)]],
    uint gid                     [[thread_position_in_grid]]
) {
    uint idx = gid * 4;
    if (idx >= N) return;

    const float SQRT_2_OVER_PI = 0.7978845608f; // sqrt(2/pi)
    const float GELU_COEFF = 0.044715f;

    uint remaining = min(N - idx, 4u);
    for (uint i = 0; i < remaining; i++) {
        float x = float(input[idx + i]);
        float inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x);
        output[idx + i] = half(0.5f * x * (1.0f + tanh(inner)));
    }
}

// --- Rotary Position Embedding (RoPE) ---
// Applies rotation to pairs of elements in query/key vectors:
//   x_rotated[2i]   = x[2i]   * cos(theta_i) - x[2i+1] * sin(theta_i)
//   x_rotated[2i+1] = x[2i]   * sin(theta_i) + x[2i+1] * cos(theta_i)
// where theta_i = pos / (10000 ^ (2i / head_dim))
//
// head_dim = 128 for our architecture (d_model=4096, 32 heads)
kernel void rope_fp16(
    device half* qk              [[buffer(0)]],   // [seq_len, num_heads, head_dim] in-place
    constant uint& seq_len       [[buffer(1)]],
    constant uint& num_heads     [[buffer(2)]],
    constant uint& head_dim      [[buffer(3)]],
    constant uint& position_offset [[buffer(4)]],  // for KV cache continuation
    uint2 gid                    [[thread_position_in_grid]]
    // gid.x = pair index within head (0..head_dim/2-1)
    // gid.y = (seq_pos * num_heads + head_idx)
) {
    uint pair_idx = gid.x;
    uint half_dim = head_dim / 2;
    if (pair_idx >= half_dim) return;

    uint flat_idx = gid.y;
    uint head_idx = flat_idx % num_heads;
    uint seq_pos = flat_idx / num_heads;
    if (seq_pos >= seq_len) return;

    // Compute rotation angle
    float theta = float(seq_pos + position_offset) /
                  pow(10000.0f, 2.0f * float(pair_idx) / float(head_dim));
    float cos_theta = cos(theta);
    float sin_theta = sin(theta);

    // Element indices
    uint base = (seq_pos * num_heads + head_idx) * head_dim;
    uint i0 = base + pair_idx * 2;
    uint i1 = i0 + 1;

    float x0 = float(qk[i0]);
    float x1 = float(qk[i1]);

    qk[i0] = half(x0 * cos_theta - x1 * sin_theta);
    qk[i1] = half(x0 * sin_theta + x1 * cos_theta);
}

// --- Fused Bias + SwiGLU ---
// Combines bias addition with SwiGLU for expert FFN:
//   gate_biased = gate + bias_gate
//   up_biased = up + bias_up
//   output = silu(gate_biased) * up_biased
kernel void fused_bias_swiglu_fp16(
    device const half* gate      [[buffer(0)]],
    device const half* up        [[buffer(1)]],
    device const half* bias_gate [[buffer(2)]],
    device const half* bias_up   [[buffer(3)]],
    device half* output          [[buffer(4)]],
    constant uint& N             [[buffer(5)]],
    uint gid                     [[thread_position_in_grid]]
) {
    uint idx = gid * 4;
    if (idx >= N) return;

    uint remaining = min(N - idx, 4u);
    for (uint i = 0; i < remaining; i++) {
        float g = float(gate[idx + i]) + float(bias_gate[idx + i]);
        float u = float(up[idx + i]) + float(bias_up[idx + i]);
        float silu_g = g / (1.0f + exp(-g));
        output[idx + i] = half(silu_g * u);
    }
}
