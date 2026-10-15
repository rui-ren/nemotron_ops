// Nemotron-3.5 Bolt decode ops for ONNX Runtime (CUDA EP, custom-op ABI), domain "bolt".
//
//   MambaStep   : Mamba2 mixer core of R <= 16 rows (conv1d+SiLU, dt/decay, SSM state update, D skip, gated grouped RMSNorm).
//                 Replaces the ~35-node ONNX chain between in_proj and out_proj; states are updated in place friendly.
//   MoeRoute    : fp32 router (sigmoid, bias-corrected top-k, normalisation, scaling) emitting the dense top-k selection /
//                 weight matrices consumed by QMoE (router_probs / router_weights).
//   Relu2       : y = bf16(relu(x)^2)
//   Nvfp4Gemv   : y[R,N] = x[R,K] @ W^T with a modelopt NVFP4 weight (u8 [N,K/2], fp8 e4m3 block-16 scales, fp32 global scale).
#ifndef ORT_API_MANUAL_INIT
#define ORT_API_MANUAL_INIT
#endif
#include "onnxruntime_cxx_api.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace {

using bf16 = __nv_bfloat16;
constexpr int kMaxRows = 16;

__device__ __forceinline__ float B2F(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float Silu(float v) { return v / (1.0f + expf(-v)); }

void CheckLaunch() {
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

cudaStream_t Stream(const OrtApi& api, OrtKernelContext* context) {
  void* raw = nullptr;
  Ort::ThrowOnError(api.KernelContext_GetGPUComputeStream(context, &raw));
  return static_cast<cudaStream_t>(raw);
}

template <typename T>
const T* In(const Ort::KernelContext& c, size_t i) {
  return reinterpret_cast<const T*>(c.GetInput(i).GetTensorRawData());
}
template <typename T>
T* Out(Ort::UnownedValue& v) {
  return reinterpret_cast<T*>(v.GetTensorMutableRawData());
}
int64_t Dim(const Ort::ConstValue& v, size_t i) { return v.GetTensorTypeAndShapeInfo().GetShape().at(i); }
int64_t Count(const Ort::ConstValue& v) {
  int64_t n = 1;
  for (auto d : v.GetTensorTypeAndShapeInfo().GetShape()) n *= d;
  return n;
}

// ------------------------------------------------------------------------------------------------------------------
// MambaStep
// ------------------------------------------------------------------------------------------------------------------
// proj row layout: [gate (di) | xBC (conv_dim = di + 2*ng*ds) | dt (heads)]
__global__ void MambaConvKernel(const bf16* __restrict__ proj, int R, int proj_stride, int di, int conv_dim,
                                const bf16* __restrict__ conv_state_in, const bf16* __restrict__ conv_w,
                                const bf16* __restrict__ conv_b, bf16* __restrict__ conv_state_out,
                                bf16* __restrict__ xbc_act) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= conv_dim) return;
  float inp[3 + kMaxRows];
  inp[0] = B2F(conv_state_in[c * 3 + 0]);
  inp[1] = B2F(conv_state_in[c * 3 + 1]);
  inp[2] = B2F(conv_state_in[c * 3 + 2]);
  for (int r = 0; r < R; ++r) inp[3 + r] = B2F(proj[static_cast<int64_t>(r) * proj_stride + di + c]);
  const float w0 = B2F(conv_w[c * 4 + 0]), w1 = B2F(conv_w[c * 4 + 1]), w2 = B2F(conv_w[c * 4 + 2]),
              w3 = B2F(conv_w[c * 4 + 3]);
  const float bias = B2F(conv_b[c]);
  for (int r = 0; r < R; ++r) {
    float acc = bias + w0 * inp[r] + w1 * inp[r + 1] + w2 * inp[r + 2] + w3 * inp[r + 3];
    xbc_act[static_cast<int64_t>(r) * conv_dim + c] = __float2bfloat16(Silu(acc));
  }
  conv_state_out[c * 3 + 0] = __float2bfloat16(inp[R]);
  conv_state_out[c * 3 + 1] = __float2bfloat16(inp[R + 1]);
  conv_state_out[c * 3 + 2] = __float2bfloat16(inp[R + 2]);
}

// One block per head. State layout [heads, ds, hd] (hd contiguous), fp32. Thread (p, half) owns state[n in half][p].
template <int DS, int HD>
__global__ void MambaSsmKernel(const bf16* __restrict__ proj, int R, int proj_stride, int di, int conv_dim, int heads,
                               int heads_per_group, const bf16* __restrict__ xbc_act,
                               const float* __restrict__ ssm_in, const float* __restrict__ dt_bias,
                               const float* __restrict__ a_neg, const float* __restrict__ d_skip, float tmin,
                               float* __restrict__ ssm_out, float* __restrict__ ypre) {
  const int h = blockIdx.x;
  const int g = h / heads_per_group;
  const int tid = threadIdx.x;
  const int p = tid % HD;
  const int half = tid / HD;  // 0 or 1
  constexpr int NH = DS / 2;
  __shared__ float sB[kMaxRows][DS];
  __shared__ float sC[kMaxRows][DS];
  __shared__ float sPart[kMaxRows][2][HD];
  __shared__ float sDt[kMaxRows];
  const int b_off = di + g * DS;
  const int c_off = di + (heads / heads_per_group) * DS + g * DS;
  for (int i = tid; i < R * DS; i += blockDim.x) {
    const int r = i / DS, n = i % DS;
    sB[r][n] = B2F(xbc_act[static_cast<int64_t>(r) * conv_dim + b_off + n]);
    sC[r][n] = B2F(xbc_act[static_cast<int64_t>(r) * conv_dim + c_off + n]);
  }
  if (tid < R) {
    const float raw = B2F(proj[static_cast<int64_t>(tid) * proj_stride + di + conv_dim + h]) + dt_bias[h];
    float dt = raw > 20.0f ? raw : log1pf(expf(raw));
    sDt[tid] = fmaxf(dt, tmin);
  }
  __syncthreads();
  float st[NH];
  const float* sp = ssm_in + (static_cast<int64_t>(h) * DS + half * NH) * HD + p;
#pragma unroll
  for (int n = 0; n < NH; ++n) st[n] = sp[static_cast<int64_t>(n) * HD];
  const float A = a_neg[h];
  const float D = d_skip[h];
  for (int r = 0; r < R; ++r) {
    const float dt = sDt[r];
    const float dA = expf(A * dt);
    const float x = B2F(xbc_act[static_cast<int64_t>(r) * conv_dim + h * HD + p]);
    const float dtx = dt * x;
    float acc = 0.f;
#pragma unroll
    for (int n = 0; n < NH; ++n) {
      const int nn = half * NH + n;
      st[n] = st[n] * dA + dtx * sB[r][nn];
      acc += sC[r][nn] * st[n];
    }
    sPart[r][half][p] = acc;
  }
  float* op = ssm_out + (static_cast<int64_t>(h) * DS + half * NH) * HD + p;
#pragma unroll
  for (int n = 0; n < NH; ++n) op[static_cast<int64_t>(n) * HD] = st[n];
  __syncthreads();
  for (int i = tid; i < R * HD; i += blockDim.x) {
    const int r = i / HD, pp = i % HD;
    const float x = B2F(xbc_act[static_cast<int64_t>(r) * conv_dim + h * HD + pp]);
    ypre[static_cast<int64_t>(r) * di + h * HD + pp] = sPart[r][0][pp] + sPart[r][1][pp] + D * x;
  }
}

// grid (R, groups), block 256; gate is proj[r][0..di)
__global__ void MambaNormKernel(const bf16* __restrict__ proj, int proj_stride, int di, int group_size, float eps,
                                const float* __restrict__ ypre, const float* __restrict__ norm_w, bf16* __restrict__ y) {
  const int r = blockIdx.x, g = blockIdx.y;
  const int64_t base = static_cast<int64_t>(r) * di + static_cast<int64_t>(g) * group_size;
  const bf16* gate = proj + static_cast<int64_t>(r) * proj_stride + static_cast<int64_t>(g) * group_size;
  float ss = 0.f;
  for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
    const float gt = B2F(gate[i]);
    const float v = ypre[base + i] * (gt / (1.0f + expf(-gt)));
    ss += v * v;
  }
  __shared__ float red[32];
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffff, ss, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    if (threadIdx.x == 0) red[0] = v;
  }
  __syncthreads();
  const float rs = rsqrtf(red[0] / group_size + eps);
  for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
    const float gt = B2F(gate[i]);
    const float v = ypre[base + i] * (gt / (1.0f + expf(-gt)));
    y[base + i] = __float2bfloat16(v * rs * norm_w[g * group_size + i]);
  }
}

struct MambaStepState {
  const OrtApi& api;
  int heads, hd, ds, ng;
  float eps, tmin;
  MambaStepState(const OrtApi& a, const OrtKernelInfo* info) : api(a) {
    Ort::ConstKernelInfo k(info);
    heads = static_cast<int>(k.GetAttribute<int64_t>("heads"));
    hd = static_cast<int>(k.GetAttribute<int64_t>("head_dim"));
    ds = static_cast<int>(k.GetAttribute<int64_t>("d_state"));
    ng = static_cast<int>(k.GetAttribute<int64_t>("n_groups"));
    eps = k.GetAttribute<float>("epsilon");
    tmin = k.GetAttribute<float>("time_step_min");
    if (hd != 64 || ds != 128) throw std::invalid_argument("MambaStep: only head_dim=64, d_state=128 are implemented");
  }
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      auto proj = c.GetInput(0);
      const int R = static_cast<int>(Dim(proj, 0));
      if (R < 1 || R > kMaxRows) throw std::invalid_argument("MambaStep: rows must be in [1,16]");
      const int di = heads * hd;
      const int conv_dim = di + 2 * ng * ds;
      const int stride = di + conv_dim + heads;
      if (Dim(proj, 1) != stride) throw std::invalid_argument("MambaStep: bad proj width");
      auto y = c.GetOutput(0, std::vector<int64_t>{R, di});
      auto conv_out = c.GetOutput(1, c.GetInput(1).GetTensorTypeAndShapeInfo().GetShape());
      auto ssm_out = c.GetOutput(2, c.GetInput(2).GetTensorTypeAndShapeInfo().GetShape());
      auto xbc_act = c.GetOutput(3, std::vector<int64_t>{R, conv_dim});
      auto ypre = c.GetOutput(4, std::vector<int64_t>{R, di});
      cudaStream_t s = Stream(api, raw);
      MambaConvKernel<<<(conv_dim + 127) / 128, 128, 0, s>>>(In<bf16>(c, 0), R, stride, di, conv_dim, In<bf16>(c, 1),
                                                           In<bf16>(c, 3), In<bf16>(c, 4), Out<bf16>(conv_out),
                                                           Out<bf16>(xbc_act));
      MambaSsmKernel<128, 64><<<heads, 128, 0, s>>>(In<bf16>(c, 0), R, stride, di, conv_dim, heads, heads / ng,
                                                     Out<bf16>(xbc_act), In<float>(c, 2), In<float>(c, 5),
                                                     In<float>(c, 6), In<float>(c, 7), tmin, Out<float>(ssm_out),
                                                     Out<float>(ypre));
      MambaNormKernel<<<dim3(R, ng), 256, 0, s>>>(In<bf16>(c, 0), stride, di, di / ng, eps, Out<float>(ypre),
                                                  In<float>(c, 8), Out<bf16>(y));
      CheckLaunch();
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_FAIL, e.what());
    }
  }
};

// ------------------------------------------------------------------------------------------------------------------
// MoeRoute
// ------------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ float WarpSum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  return v;
}

// grid (R, ceil(E/8)), block 256: one warp per expert logit.
__global__ void RouterLogitsKernel(const bf16* __restrict__ x, const bf16* __restrict__ w, int K, int E,
                                   float* __restrict__ scores) {
  const int r = blockIdx.x;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int e = blockIdx.y * 8 + warp;
  if (e >= E) return;
  const bf16* xr = x + static_cast<int64_t>(r) * K;
  const bf16* wr = w + static_cast<int64_t>(e) * K;
  float acc = 0.f;
  for (int k = lane * 8; k < K; k += 256) {
    const uint4 xv = *reinterpret_cast<const uint4*>(xr + k);
    const uint4 wv = *reinterpret_cast<const uint4*>(wr + k);
    const bf16* xp = reinterpret_cast<const bf16*>(&xv);
    const bf16* wp = reinterpret_cast<const bf16*>(&wv);
#pragma unroll
    for (int i = 0; i < 8; ++i) acc += B2F(xp[i]) * B2F(wp[i]);
  }
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
  if (lane == 0) scores[static_cast<int64_t>(r) * E + e] = 1.0f / (1.0f + expf(-acc));
}

// one warp per row (E <= 512): iterative warp arg-max over bias-corrected scores
__global__ void RouterTopkKernel(const float* __restrict__ scores, const float* __restrict__ bias, int E, int topk,
                                 float scale, int* __restrict__ idx_out, float* __restrict__ w_out) {
  const int r = blockIdx.x;
  const int lane = threadIdx.x;
  float val[16], sc[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int e = lane + 32 * i;
    if (e < E) {
      sc[i] = scores[static_cast<int64_t>(r) * E + e];
      val[i] = sc[i] + bias[e];
    } else {
      sc[i] = 0.f;
      val[i] = -INFINITY;
    }
  }
  int chosen[32];
  float cs[32];
  float sum = 0.f;
  for (int j = 0; j < topk; ++j) {
    float bv = -INFINITY;
    int bi = -1, bslot = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i)
      if (val[i] > bv) { bv = val[i]; bi = lane + 32 * i; bslot = i; }
    float wv = bv;
    int wi = bi;
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffff, wv, o);
      const int oi = __shfl_xor_sync(0xffffffff, wi, o);
      if (ov > wv || (ov == wv && oi >= 0 && (wi < 0 || oi < wi))) { wv = ov; wi = oi; }
    }
    float s = 0.f;
    if (wi >= 0 && (wi & 31) == lane) {
      s = sc[wi >> 5];
#pragma unroll
      for (int i = 0; i < 16; ++i)
        if (i == (wi >> 5)) val[i] = -INFINITY;
    }
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
    chosen[j] = wi;
    cs[j] = s;
    sum += s;
    (void)bslot;
  }
  const float denom = sum + 1e-20f;
  if (lane == 0) {
    for (int j = 0; j < topk; ++j) {
      idx_out[r * topk + j] = chosen[j];
      w_out[r * topk + j] = cs[j] / denom * scale;
    }
  }
}

struct MoeRouteState {
  const OrtApi& api;
  int topk;
  float scale;
  MoeRouteState(const OrtApi& a, const OrtKernelInfo* info) : api(a) {
    Ort::ConstKernelInfo k(info);
    topk = static_cast<int>(k.GetAttribute<int64_t>("topk"));
    scale = k.GetAttribute<float>("scale");
  }
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      auto x = c.GetInput(0);
      const int R = static_cast<int>(Dim(x, 0)), K = static_cast<int>(Dim(x, 1));
      const int E = static_cast<int>(Dim(c.GetInput(1), 0));
      if (E > 512 || K % 256 != 0 || topk > 32) throw std::invalid_argument("MoeRoute: unsupported shape");
      auto idx = c.GetOutput(0, std::vector<int64_t>{R, topk});
      auto wt = c.GetOutput(1, std::vector<int64_t>{R, topk});
      auto scores = c.GetOutput(2, std::vector<int64_t>{R, E});
      cudaStream_t s = Stream(api, raw);
      RouterLogitsKernel<<<dim3(R, (E + 7) / 8), 256, 0, s>>>(In<bf16>(c, 0), In<bf16>(c, 1), K, E, Out<float>(scores));
      RouterTopkKernel<<<R, 32, 0, s>>>(Out<float>(scores), In<float>(c, 2), E, topk, scale, Out<int>(idx),
                                        Out<float>(wt));
      CheckLaunch();
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_FAIL, e.what());
    }
  }
};

// ------------------------------------------------------------------------------------------------------------------
// MoeDecode: routed experts (int4 block-32, zero point 8, nibble-low-first) for R*topk (token, expert) pairs.
//   FC1: h = bf16(relu2(bf16(z @ W1[e]^T)));  FC2: p = bf16(h @ W2[e]^T);  out = bf16(sum_k w_k * p_k)
// ------------------------------------------------------------------------------------------------------------------

// dot of 32 int4 (one 16-byte chunk) with x[32] floats
__device__ __forceinline__ float DotInt4x32(const uint4 wv, const float* x) {
  const uint32_t w[4] = {wv.x, wv.y, wv.z, wv.w};
  float acc = 0.f;
#pragma unroll
  for (int q = 0; q < 4; ++q) {
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      const uint32_t byte = (w[q] >> (8 * b)) & 0xFFu;
      const float lo = __uint_as_float(0x4B000000u | (byte & 0xFu)) - 8388608.0f - 8.0f;
      const float hi = __uint_as_float(0x4B000000u | (byte >> 4)) - 8388608.0f - 8.0f;
      acc += lo * x[q * 8 + b * 2] + hi * x[q * 8 + b * 2 + 1];
    }
  }
  return acc;
}

// grid (R*topk, I/8), block 256; K1 must be 1024 (one 16B chunk per lane)
__global__ void MoeFc1Kernel(const bf16* __restrict__ z, const int* __restrict__ idx, int topk, int I,
                             const uint8_t* __restrict__ w1, const bf16* __restrict__ s1, bf16* __restrict__ h) {
  const int pair = blockIdx.x;
  const int t = pair / topk;
  const int e = idx[pair];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int j = blockIdx.y * 8 + warp;
  if (j >= I) return;
  float x[32];
  const uint4* zp = reinterpret_cast<const uint4*>(z + static_cast<int64_t>(t) * 1024 + lane * 32);
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const uint4 v = zp[q];
    const bf16* b = reinterpret_cast<const bf16*>(&v);
#pragma unroll
    for (int i = 0; i < 8; ++i) x[q * 8 + i] = B2F(b[i]);
  }
  const int64_t row = static_cast<int64_t>(e) * I + j;
  const uint4 wv = *reinterpret_cast<const uint4*>(w1 + row * 512 + lane * 16);
  const float sc = B2F(s1[row * 32 + lane]);
  const float tot = WarpSum(DotInt4x32(wv, x) * sc);
  if (lane == 0) {
    const float r = fmaxf(B2F(__float2bfloat16(tot)), 0.f);
    h[static_cast<int64_t>(pair) * I + j] = __float2bfloat16(r * r);
  }
}

// grid (R*topk, 1024/8), block 256; K2 = I (multiple of 32)
__global__ void MoeFc2Kernel(const bf16* __restrict__ h, const int* __restrict__ idx, int I,
                             const uint8_t* __restrict__ w2, const bf16* __restrict__ s2, float* __restrict__ part) {
  const int pair = blockIdx.x;
  const int e = idx[pair];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n = blockIdx.y * 8 + warp;
  const int nchunks = I / 32;
  const int64_t row = static_cast<int64_t>(e) * 1024 + n;
  const bf16* hp = h + static_cast<int64_t>(pair) * I;
  float acc = 0.f;
  for (int c = lane; c < nchunks; c += 32) {
    float x[32];
    const uint4* xp = reinterpret_cast<const uint4*>(hp + c * 32);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const uint4 v = xp[q];
      const bf16* b = reinterpret_cast<const bf16*>(&v);
#pragma unroll
      for (int i = 0; i < 8; ++i) x[q * 8 + i] = B2F(b[i]);
    }
    const uint4 wv = *reinterpret_cast<const uint4*>(w2 + row * (I / 2) + c * 16);
    acc += DotInt4x32(wv, x) * B2F(s2[row * nchunks + c]);
  }
  acc = WarpSum(acc);
  if (lane == 0) part[static_cast<int64_t>(pair) * 1024 + n] = B2F(__float2bfloat16(acc));
}

__global__ void MoeFinalizeKernel(const float* __restrict__ part, const float* __restrict__ w, int topk,
                                  bf16* __restrict__ out) {
  const int t = blockIdx.x;
  for (int n = threadIdx.x; n < 1024; n += blockDim.x) {
    float a = 0.f;
    for (int k = 0; k < topk; ++k) a += w[t * topk + k] * part[(static_cast<int64_t>(t) * topk + k) * 1024 + n];
    out[static_cast<int64_t>(t) * 1024 + n] = __float2bfloat16(a);
  }
}

struct MoeDecodeState {
  const OrtApi& api;
  int topk;
  MoeDecodeState(const OrtApi& a, const OrtKernelInfo* info) : api(a) {
    Ort::ConstKernelInfo k(info);
    topk = static_cast<int>(k.GetAttribute<int64_t>("topk"));
  }
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      auto z = c.GetInput(0);
      const int R = static_cast<int>(Dim(z, 0));
      if (Dim(z, 1) != 1024) throw std::invalid_argument("MoeDecode: latent size must be 1024");
      const int I = static_cast<int>(Dim(c.GetInput(3), 1));  // fc1_w [E, I, 512]
      if (I % 32 != 0 || I % 8 != 0) throw std::invalid_argument("MoeDecode: bad inter size");
      auto out = c.GetOutput(0, std::vector<int64_t>{R, 1024});
      auto h = c.GetOutput(1, std::vector<int64_t>{static_cast<int64_t>(R) * topk, I});
      auto part = c.GetOutput(2, std::vector<int64_t>{static_cast<int64_t>(R) * topk, 1024});
      cudaStream_t s = Stream(api, raw);
      MoeFc1Kernel<<<dim3(R * topk, I / 8), 256, 0, s>>>(In<bf16>(c, 0), In<int>(c, 1), topk, I, In<uint8_t>(c, 3),
                                                         In<bf16>(c, 4), Out<bf16>(h));
      MoeFc2Kernel<<<dim3(R * topk, 1024 / 8), 256, 0, s>>>(Out<bf16>(h), In<int>(c, 1), I, In<uint8_t>(c, 5),
                                                           In<bf16>(c, 6), Out<float>(part));
      MoeFinalizeKernel<<<R, 256, 0, s>>>(Out<float>(part), In<float>(c, 2), topk, Out<bf16>(out));
      CheckLaunch();
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_FAIL, e.what());
    }
  }
};

// ------------------------------------------------------------------------------------------------------------------
// Relu2
// ------------------------------------------------------------------------------------------------------------------
__global__ void Relu2Kernel(const bf16* x, bf16* y, int64_t n) {
  int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= n) return;
  const float r = fmaxf(B2F(x[i]), 0.f);
  y[i] = __float2bfloat16(r * r);
}
struct Relu2State {
  const OrtApi& api;
  Relu2State(const OrtApi& a, const OrtKernelInfo*) : api(a) {}
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      auto x = c.GetInput(0);
      auto out = c.GetOutput(0, x.GetTensorTypeAndShapeInfo().GetShape());
      const int64_t n = Count(x);
      Relu2Kernel<<<(n + 255) / 256, 256, 0, Stream(api, raw)>>>(In<bf16>(c, 0), Out<bf16>(out), n);
      CheckLaunch();
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_FAIL, e.what());
    }
  }
};

// ------------------------------------------------------------------------------------------------------------------
// Nvfp4Gemv: a warp owns ROWS weight rows; lanes stride over 16-byte (32-value) chunks of K.
// ------------------------------------------------------------------------------------------------------------------
// e2m1 nibble pairs -> fp16x2 (value * 2^-14); the caller rescales by 16384 (exact).
__device__ __forceinline__ uint32_t Fp4PairPattern(uint32_t s) {
  return ((s & 0x00070007u) << 9) | ((s & 0x00080008u) << 12);
}
__device__ __forceinline__ float Fp8E4M3ToFloat(uint32_t byte) {
  return __half2float(__half(__nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(byte), __NV_E4M3)));
}

// A warp owns ROWS weight rows; lane l handles 16-byte (32-value) chunks l + 32*it. All weight loads are issued up front.
template <int R, int ROWS, int IT>
__global__ void Nvfp4GemvKernel(const bf16* __restrict__ x, const uint8_t* __restrict__ w,
                                const uint8_t* __restrict__ ws, const float* __restrict__ gscale, int N, int K,
                                bf16* __restrict__ y) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n0 = (blockIdx.x * (blockDim.x >> 5) + warp) * ROWS;
  if (n0 >= N) return;
  const int kbytes = K / 2;
  const int kscale = K / 16;
  uint4 wv[ROWS][IT];
  uint32_t sc[ROWS][IT];
#pragma unroll
  for (int i = 0; i < ROWS; ++i) {
    const int n = min(n0 + i, N - 1);
#pragma unroll
    for (int it = 0; it < IT; ++it) {
      const int cb = it * 512 + lane * 16;
      wv[i][it] = *reinterpret_cast<const uint4*>(w + static_cast<int64_t>(n) * kbytes + cb);
      sc[i][it] = *reinterpret_cast<const uint16_t*>(ws + static_cast<int64_t>(n) * kscale + (cb >> 3));
    }
  }
  float acc[ROWS][R];
#pragma unroll
  for (int i = 0; i < ROWS; ++i)
#pragma unroll
    for (int r = 0; r < R; ++r) acc[i][r] = 0.f;
#pragma unroll
  for (int it = 0; it < IT; ++it) {
    const int k0 = (it * 512 + lane * 16) * 2;
#pragma unroll
    for (int r = 0; r < R; ++r) {
      __half2 xp[16];
      const uint4* xq = reinterpret_cast<const uint4*>(x + static_cast<int64_t>(r) * K + k0);
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        const uint4 v = xq[g];
        const bf16* b = reinterpret_cast<const bf16*>(&v);
#pragma unroll
        for (int p = 0; p < 4; ++p) xp[g * 4 + p] = __floats2half2_rn(B2F(b[p]), B2F(b[p + 4]));
      }
#pragma unroll
      for (int i = 0; i < ROWS; ++i) {
        const uint32_t words[4] = {wv[i][it].x, wv[i][it].y, wv[i][it].z, wv[i][it].w};
        float tot = 0.f;
#pragma unroll
        for (int blk = 0; blk < 2; ++blk) {
          __half2 a = __float2half2_rn(0.f);
#pragma unroll
          for (int wd = 0; wd < 2; ++wd) {
            const uint32_t word = words[blk * 2 + wd];
#pragma unroll
            for (int p = 0; p < 4; ++p) {
              const uint32_t pb = Fp4PairPattern(word >> (4 * p));
              __half2 pat = *reinterpret_cast<const __half2*>(&pb);
              pat = __hmul2(pat, __float2half2_rn(16384.f));
              a = __hfma2(pat, xp[(blk * 2 + wd) * 4 + p], a);
            }
          }
          const float f = __low2float(a) + __high2float(a);
          tot += f * Fp8E4M3ToFloat((sc[i][it] >> (8 * blk)) & 0xFFu);
        }
        acc[i][r] += tot;
      }
    }
  }
  const float g = gscale[0];
#pragma unroll
  for (int i = 0; i < ROWS; ++i)
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const float v = WarpSum(acc[i][r]);
      if (lane == 0 && n0 + i < N) y[static_cast<int64_t>(r) * N + n0 + i] = __float2bfloat16(v * g);
    }
}

template <int R, int IT>
void LaunchNvfp4Impl(const bf16* x, const uint8_t* w, const uint8_t* ws, const float* g, int N, int K, bf16* y,
                     cudaStream_t s) {
  if (N <= 8192) {
    constexpr int ROWS = 1, WARPS = 4;
    Nvfp4GemvKernel<R, ROWS, IT><<<(N + WARPS - 1) / WARPS, WARPS * 32, 0, s>>>(x, w, ws, g, N, K, y);
  } else {
    constexpr int ROWS = 2, WARPS = 8;
    const int per_block = ROWS * WARPS;
    Nvfp4GemvKernel<R, ROWS, IT><<<(N + per_block - 1) / per_block, WARPS * 32, 0, s>>>(x, w, ws, g, N, K, y);
  }
}
template <int R>
void LaunchNvfp4(const bf16* x, const uint8_t* w, const uint8_t* ws, const float* g, int N, int K, bf16* y,
                 cudaStream_t s) {
  if (K == 4096) LaunchNvfp4Impl<R, 4>(x, w, ws, g, N, K, y, s);
  else if (K == 8192) LaunchNvfp4Impl<R, 8>(x, w, ws, g, N, K, y, s);
  else throw std::invalid_argument("Nvfp4Gemv: K must be 4096 or 8192");
}

struct Nvfp4GemvState {
  const OrtApi& api;
  Nvfp4GemvState(const OrtApi& a, const OrtKernelInfo*) : api(a) {}
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      auto x = c.GetInput(0);
      auto w = c.GetInput(1);
      auto out_shape = x.GetTensorTypeAndShapeInfo().GetShape();
      const int K = static_cast<int>(out_shape.back());
      const int R = static_cast<int>(Count(x) / K);
      const int N = static_cast<int>(Dim(w, 0));
      if (Dim(w, 1) * 2 != K || K % 1024 != 0) throw std::invalid_argument("Nvfp4Gemv: K mismatch (K % 1024 required)");
      out_shape.back() = N;
      auto y = c.GetOutput(0, out_shape);
      cudaStream_t s = Stream(api, raw);
      const bf16* xp = In<bf16>(c, 0);
      const uint8_t* wp = In<uint8_t>(c, 1);
      const uint8_t* sp = In<uint8_t>(c, 2);
      const float* gp = In<float>(c, 3);
      bf16* yp = Out<bf16>(y);
      switch (R) {
        case 1: LaunchNvfp4<1>(xp, wp, sp, gp, N, K, yp, s); break;
        case 2: LaunchNvfp4<2>(xp, wp, sp, gp, N, K, yp, s); break;
        case 3: case 4: {
          // pad rows up to 4 is not possible without copying; run 4-row variant only when R == 4
          if (R == 4) LaunchNvfp4<4>(xp, wp, sp, gp, N, K, yp, s);
          else LaunchNvfp4<3>(xp, wp, sp, gp, N, K, yp, s);
          break;
        }
        case 5: LaunchNvfp4<5>(xp, wp, sp, gp, N, K, yp, s); break;
        case 6: LaunchNvfp4<6>(xp, wp, sp, gp, N, K, yp, s); break;
        case 7: LaunchNvfp4<7>(xp, wp, sp, gp, N, K, yp, s); break;
        case 8: LaunchNvfp4<8>(xp, wp, sp, gp, N, K, yp, s); break;
        default: throw std::invalid_argument("Nvfp4Gemv: rows must be in [1,8]");
      }
      CheckLaunch();
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_FAIL, e.what());
    }
  }
};

// ------------------------------------------------------------------------------------------------------------------
// op registration
// ------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------------
// QGemv: block-32 symmetric int8 (zero point 128) / int4 (zero point 8, nibble-low-first) GEMV, bf16 scales,
//        weights in the MatMulNBits layout [N, K/32, 32 or 16].  y[R,N] = x[R,K] @ W^T
// ------------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ float DotU8x16(const uint4 wv, const float* x) {
  const uint32_t w[4] = {wv.x, wv.y, wv.z, wv.w};
  float acc = 0.f;
#pragma unroll
  for (int q = 0; q < 4; ++q)
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      const uint32_t byte = (w[q] >> (8 * b)) & 0xFFu;
      acc += (__uint_as_float(0x4B000000u | byte) - 8388608.0f - 128.0f) * x[q * 4 + b];
    }
  return acc;
}

template <int BITS, int R, int ROWS, int IT>
__global__ void QGemvKernel(const bf16* __restrict__ x, const uint8_t* __restrict__ w, const bf16* __restrict__ sc,
                            int N, int K, bf16* __restrict__ y) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n0 = (blockIdx.x * (blockDim.x >> 5) + warp) * ROWS;
  if (n0 >= N) return;
  const int nb = K / 32;
  constexpr int CH = BITS == 8 ? 2 : 1;  // uint4 per 32-value block
  uint4 wv[ROWS][IT][CH];
  float sv[ROWS][IT];
#pragma unroll
  for (int i = 0; i < ROWS; ++i) {
    const int n = min(n0 + i, N - 1);
#pragma unroll
    for (int it = 0; it < IT; ++it) {
      const int b = lane + 32 * it;
      if (b < nb) {
        const uint4* wp = reinterpret_cast<const uint4*>(w + (static_cast<int64_t>(n) * nb + b) * (BITS * 4));
#pragma unroll
        for (int c = 0; c < CH; ++c) wv[i][it][c] = wp[c];
        sv[i][it] = B2F(sc[static_cast<int64_t>(n) * nb + b]);
      } else {
        sv[i][it] = 0.f;
#pragma unroll
        for (int c = 0; c < CH; ++c) wv[i][it][c] = make_uint4(0, 0, 0, 0);
      }
    }
  }
  float acc[ROWS][R];
#pragma unroll
  for (int i = 0; i < ROWS; ++i)
#pragma unroll
    for (int r = 0; r < R; ++r) acc[i][r] = 0.f;
#pragma unroll
  for (int it = 0; it < IT; ++it) {
    const int b = lane + 32 * it;
    if (b >= nb) continue;
#pragma unroll
    for (int r = 0; r < R; ++r) {
      float xf[32];
      const uint4* xq = reinterpret_cast<const uint4*>(x + static_cast<int64_t>(r) * K + b * 32);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const uint4 v = xq[q];
        const bf16* bb = reinterpret_cast<const bf16*>(&v);
#pragma unroll
        for (int i = 0; i < 8; ++i) xf[q * 8 + i] = B2F(bb[i]);
      }
#pragma unroll
      for (int i = 0; i < ROWS; ++i) {
        float d;
        if constexpr (BITS == 4) d = DotInt4x32(wv[i][it][0], xf);
        else d = DotU8x16(wv[i][it][0], xf) + DotU8x16(wv[i][it][1], xf + 16);
        acc[i][r] += d * sv[i][it];
      }
    }
  }
#pragma unroll
  for (int i = 0; i < ROWS; ++i)
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const float v = WarpSum(acc[i][r]);
      if (lane == 0 && n0 + i < N) y[static_cast<int64_t>(r) * N + n0 + i] = __float2bfloat16(v);
    }
}

template <int BITS, int R, int IT>
void LaunchQGemvImpl(const bf16* x, const uint8_t* w, const bf16* sc, int N, int K, bf16* y, cudaStream_t s) {
  if (N <= 4096) {  // small N: one row per warp to fill the GPU
    constexpr int ROWS = 1, WARPS = 4;
    QGemvKernel<BITS, R, ROWS, IT><<<(N + WARPS - 1) / WARPS, WARPS * 32, 0, s>>>(x, w, sc, N, K, y);
  } else {
    constexpr int ROWS = 2, WARPS = 8;
    const int per_block = ROWS * WARPS;
    QGemvKernel<BITS, R, ROWS, IT><<<(N + per_block - 1) / per_block, WARPS * 32, 0, s>>>(x, w, sc, N, K, y);
  }
}
template <int BITS, int R>
void LaunchQGemvR(const bf16* x, const uint8_t* w, const bf16* sc, int N, int K, bf16* y, cudaStream_t s) {
  const int nb = K / 32;
  if (nb <= 32) LaunchQGemvImpl<BITS, R, 1>(x, w, sc, N, K, y, s);
  else if (nb <= 64) LaunchQGemvImpl<BITS, R, 2>(x, w, sc, N, K, y, s);
  else if (nb <= 128) LaunchQGemvImpl<BITS, R, 4>(x, w, sc, N, K, y, s);
  else if (nb <= 192) LaunchQGemvImpl<BITS, R, 6>(x, w, sc, N, K, y, s);
  else throw std::invalid_argument("QGemv: K too large");
}
template <int BITS>
void LaunchQGemv(int R, const bf16* x, const uint8_t* w, const bf16* sc, int N, int K, bf16* y, cudaStream_t s) {
  switch (R) {
    case 1: LaunchQGemvR<BITS, 1>(x, w, sc, N, K, y, s); break;
    case 2: LaunchQGemvR<BITS, 2>(x, w, sc, N, K, y, s); break;
    case 3: LaunchQGemvR<BITS, 3>(x, w, sc, N, K, y, s); break;
    case 4: LaunchQGemvR<BITS, 4>(x, w, sc, N, K, y, s); break;
    case 5: LaunchQGemvR<BITS, 5>(x, w, sc, N, K, y, s); break;
    case 6: LaunchQGemvR<BITS, 6>(x, w, sc, N, K, y, s); break;
    case 7: LaunchQGemvR<BITS, 7>(x, w, sc, N, K, y, s); break;
    case 8: LaunchQGemvR<BITS, 8>(x, w, sc, N, K, y, s); break;
    default: throw std::invalid_argument("QGemv: rows must be in [1,8]");
  }
}

template <int BITS>
struct QGemvState {
  const OrtApi& api;
  QGemvState(const OrtApi& a, const OrtKernelInfo*) : api(a) {}
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      auto x = c.GetInput(0);
      auto w = c.GetInput(1);
      auto shape = x.GetTensorTypeAndShapeInfo().GetShape();
      const int K = static_cast<int>(shape.back());
      const int R = static_cast<int>(Count(x) / K);
      const int N = static_cast<int>(Dim(w, 0));
      if (K % 32 != 0 || Dim(w, 1) * 32 != K) throw std::invalid_argument("QGemv: K mismatch");
      shape.back() = N;
      auto y = c.GetOutput(0, shape);
      LaunchQGemv<BITS>(R, In<bf16>(c, 0), In<uint8_t>(c, 1), In<bf16>(c, 2), N, K, Out<bf16>(y), Stream(api, raw));
      CheckLaunch();
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_FAIL, e.what());
    }
  }
};
template <typename Derived, typename State>
struct BoltOp : Ort::CustomOpBase<Derived, State, true> {
  const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; }
  OrtStatus* CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void** kernel) const noexcept {
    try {
      *kernel = new State(api, info);
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_INVALID_ARGUMENT, e.what());
    }
  }
  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t) const { return INPUT_OUTPUT_REQUIRED; }
  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t) const { return INPUT_OUTPUT_REQUIRED; }
};

constexpr auto BF = ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
constexpr auto F32 = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
constexpr auto U8 = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
constexpr auto I32 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;

struct MambaStepOp : BoltOp<MambaStepOp, MambaStepState> {
  const char* GetName() const { return "MambaStep"; }
  size_t GetInputTypeCount() const { return 9; }
  ONNXTensorElementDataType GetInputType(size_t i) const {
    // proj, conv_state, ssm_state, conv_w, conv_b, dt_bias, A_neg, D, norm_w
    static const ONNXTensorElementDataType t[9] = {BF, BF, F32, BF, BF, F32, F32, F32, F32};
    return t[i];
  }
  size_t GetOutputTypeCount() const { return 5; }
  ONNXTensorElementDataType GetOutputType(size_t i) const {
    static const ONNXTensorElementDataType t[5] = {BF, BF, F32, BF, F32};
    return t[i];
  }
};
struct MoeRouteOp : BoltOp<MoeRouteOp, MoeRouteState> {
  const char* GetName() const { return "MoeRoute"; }
  size_t GetInputTypeCount() const { return 3; }
  ONNXTensorElementDataType GetInputType(size_t i) const { return i < 2 ? BF : F32; }
  size_t GetOutputTypeCount() const { return 3; }
  ONNXTensorElementDataType GetOutputType(size_t i) const {
    static const ONNXTensorElementDataType t[3] = {I32, F32, F32};
    return t[i];
  }
};
struct MoeDecodeOp : BoltOp<MoeDecodeOp, MoeDecodeState> {
  const char* GetName() const { return "MoeDecode"; }
  size_t GetInputTypeCount() const { return 7; }
  ONNXTensorElementDataType GetInputType(size_t i) const {
    // z, idx, w, fc1_w, fc1_s, fc2_w, fc2_s
    static const ONNXTensorElementDataType t[7] = {BF, I32, F32, U8, BF, U8, BF};
    return t[i];
  }
  size_t GetOutputTypeCount() const { return 3; }
  ONNXTensorElementDataType GetOutputType(size_t i) const {
    static const ONNXTensorElementDataType t[3] = {BF, BF, F32};
    return t[i];
  }
};
struct Relu2Op : BoltOp<Relu2Op, Relu2State> {
  const char* GetName() const { return "Relu2"; }
  size_t GetInputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetInputType(size_t) const { return BF; }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return BF; }
};
struct Nvfp4GemvOp : BoltOp<Nvfp4GemvOp, Nvfp4GemvState> {
  const char* GetName() const { return "Nvfp4Gemv"; }
  size_t GetInputTypeCount() const { return 4; }
  ONNXTensorElementDataType GetInputType(size_t i) const {
    static const ONNXTensorElementDataType t[4] = {BF, U8, U8, F32};
    return t[i];
  }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return BF; }
};

struct Q8GemvOp : BoltOp<Q8GemvOp, QGemvState<8>> {
  const char* GetName() const { return "Q8Gemv"; }
  size_t GetInputTypeCount() const { return 3; }
  ONNXTensorElementDataType GetInputType(size_t i) const { return i == 1 ? U8 : BF; }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return BF; }
};
struct Q4GemvOp : BoltOp<Q4GemvOp, QGemvState<4>> {
  const char* GetName() const { return "Q4Gemv"; }
  size_t GetInputTypeCount() const { return 3; }
  ONNXTensorElementDataType GetInputType(size_t i) const { return i == 1 ? U8 : BF; }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return BF; }
};

}  // namespace

extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api_base) {
  const auto api = api_base->GetApi(ORT_API_VERSION);
  if (api == nullptr) return api_base->GetApi(1)->CreateStatus(ORT_FAIL, "Unsupported ORT API");
  Ort::InitApi(api);
  try {
    static MambaStepOp mamba;
    static MoeRouteOp route;
    static MoeDecodeOp moe_decode;
    static Relu2Op relu2;
    static Nvfp4GemvOp fp4;
    static Q8GemvOp q8;
    static Q4GemvOp q4;
    static Ort::CustomOpDomain domain("bolt");
    static std::once_flag registered;
    std::call_once(registered, [&]() {
      domain.Add(&mamba);
      domain.Add(&route);
      domain.Add(&moe_decode);
      domain.Add(&relu2);
      domain.Add(&fp4);
      domain.Add(&q8);
      domain.Add(&q4);
    });
    Ort::UnownedSessionOptions(options).Add(domain);
    return nullptr;
  } catch (const std::exception& e) {
    return api->CreateStatus(ORT_FAIL, e.what());
  }
}
