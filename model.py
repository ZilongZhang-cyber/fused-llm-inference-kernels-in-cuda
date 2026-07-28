"""
Fused LLM Inference Kernels in CUDA

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - warp_reduce_sum
__device__ float warp_reduce_sum(float val) {
    // TODO: implement warp-level sum reduction using shuffle intrinsics
    val += __shfl_xor_sync(0xffffffff, val,16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

# Step 2 - warp_reduce_max
__device__ float warp_reduce_max(float val) {
    // TODO: implement warp-level max reduction using shuffle intrinsics
    val = max(val, __shfl_xor_sync(0xFFFFFFFF, val, 16));
    val = max(val, __shfl_xor_sync(0xFFFFFFFF, val, 8));
    val = max(val, __shfl_xor_sync(0xFFFFFFFF, val, 4));
    val = max(val, __shfl_xor_sync(0xFFFFFFFF, val, 2));
    val = max(val, __shfl_xor_sync(0xFFFFFFFF, val, 1));
    return val;
}

# Step 3 - block_reduce_sum
__device__ float block_reduce_sum(float val, float* shared) {
    // TODO: block-level sum via warp_reduce_sum + shared memory; result valid on thread 0
    // ===== 第 1 级：每个 warp 内部求和 =====
    val = warp_reduce_sum(val);

    int lane    = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    if (lane == 0) {
        shared[warp_id] = val;
    }

    __syncthreads();

    // ===== 第 2 级：只有 warp 0 来做最终归约 =====
    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) >> 5;
        if (lane < num_warps) {
            val = shared[lane];
        } else {
            val = 0.0f;
        }

        val = warp_reduce_sum(val);
    }

    return val;
}

# Step 4 - block_reduce_max
__device__ float block_reduce_max(float val, float* shared) {
    // TODO: block-wide max via warp_reduce_max + shared memory
        // ===== 第 1 级：每个 warp 内部求最大值 =====
    // 调用已有的 warp_reduce_max，做完后每个 warp 的 lane 0 持有该 warp 的部分最大值
    val = warp_reduce_max(val);

    int lane    = threadIdx.x & 31;   // threadIdx.x % 32
    int warp_id = threadIdx.x >> 5;   // threadIdx.x / 32

    // 只有每个 warp 的 lane 0 把部分最大值写进 shared memory
    if (lane == 0) {
        shared[warp_id] = val;
    }

    // ===== 块级同步：确保所有 warp 都写完 =====
    __syncthreads();

    // ===== 第 2 级：只有 warp 0 来做最终归约 =====
    if (warp_id == 0) {
        // 向上取整：保证最后一个不满的 warp 也被算进去
        int num_warps = (blockDim.x + 31) >> 5;

        // warp 0 的每个 lane 读一个部分最大值
        // 超出 num_warps 的 lane 补 -INFINITY（关键！不能补 0）
        if (lane < num_warps) {
            val = shared[lane];
        } else {
            val = -INFINITY;   // max(x, -∞) = x，不影响结果
        }

        // 再调一次 warp_reduce_max，把所有部分最大值合并成最终结果
        // 结果落在 warp 0 的 lane 0 = thread 0 上
        val = warp_reduce_max(val);
    }

    // 此时只有 thread 0 的 val 是全 block 的最大值
    return val;
}

# Step 5 - add_residual_kernel
__global__ void add_residual_kernel(const float* x, const float* residual,
                                    float* out, int n) {
  // TODO: implement elementwise residual addition out[i] = x[i] + residual[i]
    // 计算当前线程负责处理的全局索引
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查：超出 n 的线程直接返回，不做任何操作
    if (i < n) {
        out[i] = x[i] + residual[i];
    }
}

# Step 6 - gelu_kernel
__global__ void gelu_kernel(const float* x, float* out, int n) {
    // TODO: Apply GELU (tanh approximation) elementwise to x, write into out
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        
        // GELU tanh 近似：
        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float x3 = v * v * v;
        float inner = 0.7978845608028654f * (v + 0.044715f * x3);
        out[i] = 0.5f * v * (1.0f + tanhf(inner));
    }
}

# Step 7 - silu_kernel
__global__ void silu_kernel(const float* x, float* out, int n) {
    // TODO: apply SiLU elementwise: out[i] = x[i] / (1 + exp(-x[i]))
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float v = x[i];
        // SiLU(x) = x / (1 + exp(-x))
        out[i] = v / (1.0f + expf(-v));
    }
}

# Step 8 - swiglu_kernel
__global__ void swiglu_kernel(const float* gate, const float* up, float* out, int n) {
    // TODO: out[i] = silu(gate[i]) * up[i] for all i in [0, n)
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float g = gate[i];
        // SwiGLU = SiLU(gate) * up = (gate / (1 + exp(-gate))) * up
        float silu = g / (1.0f + expf(-g));
        out[i] = silu * up[i];
    }
}

# Step 9 - rmsnorm_kernel
__global__ void rmsnorm_kernel(const float* x, const float* weight, float* out, int n, float eps) {
    // TODO: Apply RMSNorm per row (one block per row)
        // 当前 block 处理第几行
    int row = blockIdx.x;
    // 该行数据的起始地址
    const float* x_row = x + row * n;
    float* out_row = out + row * n;

    // shared memory 用来做 block 级归约
    __shared__ float shared[32];

    // ===== 第 1 步：每个线程算自己负责元素的平方和 =====
    // 用 strided loop 处理 n > blockDim.x 的情况
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = x_row[i];
        local_sum += v * v;
    }

    // ===== 第 2 步：block 内归约求平方和（复用 block_reduce_sum 模式）=====
    // warp 内归约
    local_sum = warp_reduce_sum(local_sum);

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    // 每个 warp 的 lane 0 写部分和到 shared memory
    if (lane == 0) {
        shared[warp_id] = local_sum;
    }

    __syncthreads();

    // 第一个 warp 做最终归约
    int num_warps = (blockDim.x + 31) >> 5;
    if (warp_id == 0) {
        float val = (lane < num_warps) ? shared[lane] : 0.0f;
        val = warp_reduce_sum(val);

        // ===== 第 3 步：thread 0 算 RMS，写到 shared[0] 广播给所有线程 =====
        if (lane == 0) {
            float rms = sqrtf(val / (float)n + eps);
            shared[0] = rms;
        }
    }

    // 同步：确保所有线程都能读到 RMS
    __syncthreads();

    // 所有线程读取同一个 RMS
    float rms = shared[0];

    // ===== 第 4 步：每个线程做归一化 + 缩放 =====
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out_row[i] = (x_row[i] / rms) * weight[i];
    }
}

# Step 10 - layernorm_kernel
__global__ void layernorm_kernel(const float* x, const float* weight, const float* bias, float* out, int n, float eps) {
    // TODO: per-row LayerNorm using block_reduce_sum for mean and variance
    // 当前 block 处理第几行
    int row = blockIdx.x;
    const float* x_row = x + row * n;
    float* out_row = out + row * n;

    // 归约用的 scratch buffer（每个 warp 一个 float，最多 32 个 warp）
    __shared__ float shared[32];
    // 用来广播 mean 和 inv_std 给所有线程
    __shared__ float s_mean;
    __shared__ float s_inv_std;

    // ===== 第 1 步：每个线程累加自己负责元素的 sum 和 sum_sq =====
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = x_row[i];
        local_sum += v;
        local_sum_sq += v * v;
    }

    // ===== 第 2 步：两次 block 级归约（复用 block_reduce_sum）=====
    // 第一次归约：求 Σx（结果在 thread 0）
    float total_sum = block_reduce_sum(local_sum, shared);
    if (threadIdx.x == 0) {
        s_mean = total_sum / (float)n;
    }

    // 同步：确保 shared buffer 可以安全复用于第二次归约
    __syncthreads();

    // 第二次归约：求 Σx²（结果在 thread 0）
    float total_sum_sq = block_reduce_sum(local_sum_sq, shared);

    // ===== 第 3 步：thread 0 算方差和 1/std，广播 =====
    if (threadIdx.x == 0) {
        float mean = s_mean;
        // 方差公式：var = E[x²] - (E[x])²
        float var = total_sum_sq / (float)n - mean * mean;
        s_inv_std = rsqrtf(var + eps);   // 1 / sqrt(var + eps)
    }

    // 同步：确保所有线程都能读到 mean 和 inv_std
    __syncthreads();

    float mean = s_mean;
    float inv_std = s_inv_std;

    // ===== 第 4 步：每个线程做归一化 + 缩放 + 偏置 =====
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out_row[i] = (x_row[i] - mean) * inv_std * weight[i] + bias[i];
    }
}

# Step 11 - fused_add_rmsnorm_kernel (not yet solved)
# TODO: implement

# Step 12 - softmax_row_kernel (not yet solved)
# TODO: implement

# Step 13 - causal_softmax_kernel (not yet solved)
# TODO: implement

# Step 14 - embedding_lookup_kernel (not yet solved)
# TODO: implement

# Step 15 - rope_kernel (not yet solved)
# TODO: implement

# Step 16 - linear_kernel (not yet solved)
# TODO: implement

# Step 17 - fused_linear_bias_gelu_kernel (not yet solved)
# TODO: implement

# Step 18 - mlp_swiglu_forward (not yet solved)
# TODO: implement

# Step 19 - rmsnorm_residual_block (not yet solved)
# TODO: implement

# Step 20 - run_transformer_ffn (not yet solved)
# TODO: implement

