# nemotron_ops

CUDA custom-op library for **NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB** (Mamba2 + attention + LatentMoE hybrid) decode on
ONNX Runtime's CUDA EP. Same layout as `qwen38-flash-ops`: plain ORT custom-op ABI, `sm_90`, ops run on the provider's compute
stream (CUDA-graph capturable), one shared library, domain `bolt`.

Build:

```bash
export PATH=$CUDA_HOME/bin:$PATH
cmake -S . -B build -DORT_INCLUDE_DIR=<ort>/include/onnxruntime/core/session -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc
cmake --build build -j
# build/lib/libbolt_ops.so; load with SessionOptions.register_custom_ops_library
```

Test (every op against a torch reference, random data, real shapes of the model):

```bash
BOLT_PLUGIN=<libonnxruntime_providers_cuda.so of the CUDA plugin EP> BOLT_OPS=build/lib/libbolt_ops.so python tests/test_bolt_ops.py
```

| op | what it replaces | inputs -> outputs |
| --- | --- | --- |
| `MambaStep` | the ~35-node chain between `in_proj` and `out_proj` of a Mamba2 layer (Transpose, `CausalConvWithState`, Split, dt/softplus/clamp, `LinearAttention[gated]`, D skip, gated grouped RMSNorm, casts), R <= 16 rows | proj `[R,18560]`, conv_state `[1,10240,3]`, ssm_state `[1,128,128,64]` f32, conv_w, conv_b, dt_bias, A_neg, D, norm_w -> y `[R,8192]`, conv_state', ssm_state' (+2 scratch). State outputs may alias the state inputs. |
| `MoeRoute` | fp32 router (`MatMul`, sigmoid, bias-corrected `TopK`, `GatherElements`, normalise, x5, scatter to the dense QMoE inputs) | x `[R,4096]`, gate_w `[E,4096]` bf16, bias `[E]` f32 -> idx `[R,k]` i32, w `[R,k]` f32 (+scores) |
| `MoeDecode` | `QMoE` for R <= 8 rows (relu2 experts, latent size 1024, int4 block-32 raw `[E,N,K/2]` weights + bf16 scales, zero point 8) | z `[R,1024]`, idx, w, fc1_w, fc1_s, fc2_w, fc2_s -> out `[R,1024]` (+2 scratch) |
| `Nvfp4Gemv` | modelopt NVFP4 linear (u8 `[N,K/2]`, fp8 e4m3 block-16 scales, fp32 global scale), R <= 8 rows, K in {4096, 8192}; fp16x2 FMA on rescaled e2m1 patterns | x, w, w_scale (u8 view), global scale -> y |
| `Q8Gemv` / `Q4Gemv` | `MatMulNBits` bits=8 / bits=4, block 32, symmetric (Q8_0 / IQ2_XXS-derived weights), bf16 scales, R <= 8 | x, B `[N,K/32,32 or 16]`, scales `[N,K/32]` -> y |
| `Relu2` | `Relu` + `Mul` | x -> bf16(relu(x)^2) |

All numerics follow the bf16/fp32 rounding points of the ONNX chains they replace (see the header of `src/bolt_ops.cu`).
