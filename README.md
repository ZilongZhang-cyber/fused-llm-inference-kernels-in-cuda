# Fused LLM Inference Kernels in CUDA

Implement high-performance CUDA kernels for LLM inference, from warp/block reductions and activations through fused RMSNorm, Softmax, RoPE, and SwiGLU MLP blocks. Learn the GPU primitives that make modern transformer inference efficient.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** warp_reduce_sum
- [x] **2.** warp_reduce_max
- [x] **3.** block_reduce_sum
- [x] **4.** block_reduce_max
- [x] **5.** add_residual_kernel
- [x] **6.** gelu_kernel
- [x] **7.** silu_kernel
- [x] **8.** swiglu_kernel
- [x] **9.** rmsnorm_kernel
- [x] **10.** layernorm_kernel
- [x] **11.** fused_add_rmsnorm_kernel
- [x] **12.** softmax_row_kernel
- [x] **13.** causal_softmax_kernel
- [x] **14.** embedding_lookup_kernel
- [x] **15.** rope_kernel
- [x] **16.** linear_kernel
- [x] **17.** fused_linear_bias_gelu_kernel
- [x] **18.** mlp_swiglu_forward
- [x] **19.** rmsnorm_residual_block
- [x] **20.** run_transformer_ffn

---

Built on Deep-ML.

---

# 项目知识点总结与面试问答

## 项目考察的知识点地图（四层金字塔）

```
第 4 层（18-20）：Host 编排层 ── mlp_swiglu_forward / rmsnorm_residual_block / run_transformer_ffn
第 3 层（9-17）： 完整算子层 ── rmsnorm / layernorm / softmax / rope / linear / fused_*
第 2 层（5-8）：  elementwise 层 ── add_residual / gelu / silu / swiglu
第 1 层（1-4）：  归约原语层 ── warp_reduce / block_reduce
```

- **第 1 层**：GPU 硬件层级模型——warp 锁步、shuffle 寄存器通信、两级归约。
- **第 2 层**：CUDA 最基础编程模型——全局索引 + 边界检查 + 单精度规范。
- **第 3 层**：归约与 elementwise 组合成真实 Transformer 算子，含数值稳定性与数据布局。
- **第 4 层**：host/device 分工、显存生命周期管理、kernel 流水线编排。

**贯穿全项目的暗线**：GPU 上大多数 kernel 的瓶颈是显存带宽而非计算，所以要用 shuffle（不出寄存器）、kernel fusion（中间结果不落显存）、buffer 原地复用来减少数据搬运——"搬数据比算贵"。

## 面试问答

### Q1. warp/block/grid 层级模型，shuffle 指令原理

**层级模型**（从小到大）：
- **thread（线程）**：最小执行单位，有自己的寄存器。
- **warp**：32 个线程为一组，硬件上**锁步执行**（同一时刻执行同一条指令），是 GPU 调度的基本单位。
- **block（线程块）**：若干 warp 组成（最多 1024 线程 = 32 个 warp），块内线程可通过 shared memory 通信、用 `__syncthreads()` 同步。
- **grid（网格）**：一次 kernel 启动的所有 block，block 之间基本无法直接通信。

**shuffle 指令原理**：`__shfl_xor_sync(mask, val, offset)` 让 warp 内每个 lane 直接读取 `lane_id ^ offset` 号 lane 寄存器里的值，**数据不经过 shared memory / 显存，纯寄存器交换**，一条指令完成 32 个 lane 的并行交换。蝶形（butterfly）模式下 offset 取 16/8/4/2/1 共 5 步，即可让所有 lane 同时得到归约结果（归约 + 广播一步到位）。

### Q2. 归约怎么写？两级归约为什么这么设计？

**写法**（以 block_reduce_sum 为例）：
1. 每个 warp 先调 `warp_reduce_sum`（shuffle 蝶形，5 步），得到各自的部分和；
2. 每个 warp 的 lane 0 把部分和写入 shared memory（`shared[warp_id]`）；
3. `__syncthreads()` 确保全部写完；
4. 由 warp 0 的 32 个 lane 读回这些部分和（越界 lane 补单位元），再调一次 `warp_reduce_sum`，结果落在 thread 0。

**为什么两级**：
- shuffle 只能在 warp 内交换，跨 warp 必须借 shared memory 中转；
- CUDA 规定 block 最多 1024 线程 = **最多 32 个 warp**，部分和最多 32 个，恰好一个 warp 的 32 个 lane 能一次装下——所以第二级只需一次 warp 归约即可收尾，这是硬件参数刻意设计的巧合。

**易错点**：warp 数要**向上取整** `(blockDim.x + 31) / 32`，否则 blockDim 不是 32 倍数时会丢数据；max 归约的空位补 `-INFINITY` 而不是 0（0 会污染全负数数据的结果），sum 归约补 0。

### Q3. shared memory 和 `__syncthreads()` 的作用与陷阱

**作用**：
- shared memory 是 block 内所有线程共享的片上高速缓存（比全局显存快一个数量级），用作跨 warp 通信的中转站和数据复用的暂存区；
- `__syncthreads()` 是 block 级屏障：所有线程都到达此处后才能继续，用来保证"写完再读"的顺序。

**陷阱**：
- **写后读缺同步**：warp A 还没写完 shared memory，warp B 就去读 → 结果错误且难复现；
- **条件分支里调用 `__syncthreads()`**：如果部分线程进不了该分支，屏障永远凑不齐人 → 死锁；
- shared memory 容量很小（每 block 几十 KB），超配会降低占用率（occupancy）。

### Q4. 什么是 kernel fusion？为什么能加速？

**定义**：把多个连续的算子合并进一个 kernel（或一次逻辑调用）完成，中间结果保留在寄存器/片上，不写回全局显存。

**加速原因**：
1. **省显存往返**：不融合时，前一个 kernel 要把 [M,N] 中间矩阵完整写回显存，后一个 kernel 再完整读回来；融合后这一写一读完全消失。对 GELU 这类计算量极小的 elementwise 算子，单独跑时时间几乎全花在读写显存上，融合等于"白嫖"了这部分计算。
2. **省 kernel 启动开销**：每次 launch 有固定的微秒级开销，融合减少启动次数。
3. **省临时 buffer**：不需要为中间结果分配显存。

**本项目实例**：`fused_add_rmsnorm_kernel`（残差加法 + RMSNorm 一次循环完成，加法结果顺手累加平方和）、`fused_linear_bias_gelu_kernel`（点积 + bias + GELU 全在寄存器里做完，只写一次显存）。

### Q5. softmax 数值稳定性怎么保证？

朴素公式 `exp(x_i) / Σexp(x_j)` 中，x 稍大（如 100）`expf` 就会上溢成 inf，全盘报废。

**解法**：利用 softmax 的平移不变性，每个元素先减去该行最大值 m 再取 exp：

```
softmax(x_i) = exp(x_i - m) / Σ exp(x_j - m)，  m = max_j(x_j)
```

减完后指数最大为 0，`exp` 最大为 1，永不上溢；下溢成 0 无害。实现上需要**两次 block 归约**：先 `block_reduce_max` 求 m，再 `block_reduce_sum` 求分母，各自算完由 thread 0 写入 shared 变量广播给全 block。另需 eps 或保证分母非零防除零（causal softmax 中有效列至少含对角线元素，分母 ≥ exp(0) 的贡献）。

### Q6. RoPE / RMSNorm / SwiGLU 这些现代 LLM 组件的原理和实现

**RMSNorm**：LayerNorm 的简化版——不减均值、无 bias，只用均方根缩放：`out = x / sqrt(mean(x²) + eps) * weight`。少一次均值归约，计算更省，效果相当（LLaMA 系标配）。实现：一行一 block，strided loop 累加平方和 → block_reduce_sum → thread 0 算 `rsqrtf` 后经 shared 变量广播 → 各线程做 elementwise 缩放。

**RoPE（旋转位置编码）**：Attention 的点积天然不感知词序，需注入位置信息。RoPE 把 head_dim 按 (偶,奇) 配对成二维向量，按位置 pos 和维度对 p 查预计算的 cos/sin 表做二维旋转：`even' = even·c − odd·s; odd' = even·s + odd·c`。只旋转 Q、K（位置只需影响打分），V 不动；旋转使 Q·K 点积只依赖**相对位置差**，外推性好。实现是纯 elementwise：一个线程负责一个 (pos, head, pair)，in-place 更新时必须先把旧值读进局部变量再写回。

**SwiGLU**：现代 LLM MLP 的门控激活。三路权重：`out = (SiLU(x@W_gate^T) ⊙ (x@W_up^T)) @ W_down^T`。gate 路经 SiLU 变成 0~1 附近的"开度"，逐元素调制 up 路的信号。比 GELU-MLP 多一套升维权重（W1 拆成 W_gate + W_up），通常把 intermediate 维度缩到 2/3 保持参数量持平。LLaMA/Mistral/Qwen 均采用。

### Q7. host/device 分工、显存管理、stream 顺序语义

**分工**：`__global__` kernel 在 GPU 上做计算；host 函数（如 `mlp_swiglu_forward`、`run_transformer_ffn`）在 CPU 上当"包工头"——分配临时显存、按序发射 kernel、释放显存。kernel 发射是**异步**的：CPU 发完命令立即返回，GPU 在后台排队执行。

**显存管理**：中间结果 buffer 由 launcher 内部 `cudaMalloc` / 用完 `cudaFree`；尺寸计算先转 `size_t` 防 int 溢出；elementwise kernel 输出可**原地写回输入 buffer**（如 swiglu 结果写回 gate_buf）节省显存。

**stream 顺序语义**：同一 stream（含默认流）中 kernel 按发射顺序**串行执行**，后一个自动等前一个完成，因此流水线不需要显式同步；只有 CPU 需要读结果时才 `cudaDeviceSynchronize()` / `cudaMemcpy`（后者隐式同步）。互相独立的 kernel（如 gate 和 up 投影）理论上可放不同 stream 并行。

**启动配置的关键区别**：elementwise kernel 用 `blocks = (total + threads-1) / threads`；行归约类 kernel（rmsnorm/softmax 等）必须 **blocks = 行数**（一行一 block，块内线程要协作），行宽超过线程数由 strided loop 兜住。

### Q8. 朴素 GEMM 的问题在哪？

本项目 `linear_kernel` 是"一线程一输出元素"的朴素实现，问题：

1. **数据零复用**：out[m][n] 和 out[m][n+1] 都要读 x 的第 m 整行，相邻线程从显存重复读同样数据，每个输入元素被读 N 或 M 次，显存带宽被浪费；算术强度（FLOP/字节）极低，完全卡在访存上。
2. **没用上专用硬件**：现代 GPU 的 Tensor Core 一条指令算一小块矩阵乘，吞吐比普通 FMA 高一个数量级，朴素写法用不到。

**优化路线（下一步计划）**：
- **Shared memory tiling**：把 x 和 weight 按 tile（如 32×32）分块搬进 shared memory，块内线程共享复用，显存读取量降为原来的 1/tile_size；
- 寄存器分块、双缓冲流水线进一步提升复用与隐藏延迟；
- 用 Tensor Core（WMMA/MMA 指令）或直接调 cuBLAS/CUTLASS；
- Attention 部分把 Q·Kᵀ → softmax → ·V 融合成一个 kernel，即 FlashAttention 的核心思想。

## 与真实推理引擎的对应

把 20 个函数按数据流排列，即是一个简化版 LLaMA 推理引擎（仅缺 attention 的两个矩阵乘组装）：

```
token_ids → embedding_lookup → [每层循环：
    rmsnorm_residual_block → linear(Q/K/V) → rope → causal_softmax → linear(O)
    rmsnorm_residual_block → run_transformer_ffn(SwiGLU MLP)
] → rmsnorm → linear(lm_head) → 下一个 token
```
