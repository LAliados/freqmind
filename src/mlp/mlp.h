#ifndef MLP_FXP_H
#define MLP_FXP_H

/*
 * mlp_fxp: fast small-MLP primitives for Q48.16 fixed-point numbers.
 *
 * Assumptions:
 *   - fxp_t is compatible with the fxp48_16_t API shown in fxp48_16.h.
 *   - Matrices are row-major and contiguous.
 *   - No floating point and no SIMD/vector instructions are used.
 *   - Output buffers must not overlap input/weight/bias buffers unless a
 *     function explicitly says otherwise.
 *
 * Dense layer weight layout:
 *   weights[out_dim][in_dim], row-major.
 *
 * Generic matrix multiply layout:
 *   C[M][N] = A[M][K] * B[K][N], all row-major.
 */


#ifdef MLP_FXP_FXP_HEADER
#include MLP_FXP_FXP_HEADER
#else
#include "utility/fixed_point.h"
#endif

#ifndef __SIZEOF_INT128__
#error "mlp_fxp requires __int128 support"
#endif

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L)
#define MLP_FXP_RESTRICT restrict
#elif defined(__GNUC__) || defined(__clang__)
#define MLP_FXP_RESTRICT __restrict__
#else
#define MLP_FXP_RESTRICT
#endif

#ifndef MLP_FXP_INLINE
#define MLP_FXP_INLINE static inline
#endif

/*
 * Fast downscale uses shifts instead of signed __int128 division by 65536.
 * It keeps truncation-toward-zero semantics for negative values, matching C
 * integer division. Set to 0 if you prefer the most literal implementation.
 */
#ifndef MLP_FXP_FAST_ACC_SHIFT
#define MLP_FXP_FAST_ACC_SHIFT 1
#endif

typedef enum { MLP_FXP_OK = 0, MLP_FXP_ERR_ARG = -1, MLP_FXP_ERR_SHAPE = -2, MLP_FXP_ERR_WORK = -3 } mlp_fxp_status_t;

typedef enum { MLP_FXP_ACT_NONE = 0, MLP_FXP_ACT_RELU = 1 } mlp_fxp_activation_t;

typedef struct {
    size_t in_dim;
    size_t out_dim;

    /* weights[out_dim][in_dim], row-major */
    const fxp_t* weights;

    /* optional bias[out_dim], may be NULL */
    const fxp_t* bias;

    mlp_fxp_activation_t activation;
} mlp_fxp_layer_t;

MLP_FXP_INLINE fxp_t mlp_fxp_from_raw(s64 raw) {
    return FXP48_16_RAW(raw);
}

MLP_FXP_INLINE s64 mlp_fxp_relu_raw(s64 v) {
    /*
     * Branchless ReLU for two's-complement machines:
     * mask is all ones if v > 0, otherwise 0.
     */
    u64 mask = (u64) - (s64)(v > 0);
    return (s64)((u64)v & mask);
}

MLP_FXP_INLINE fxp_t mlp_fxp_relu_scalar(fxp_t x) {
    return mlp_fxp_from_raw(mlp_fxp_relu_raw(x._val));
}

const char* mlp_fxp_status_string(int status);

/*
 * Work length for mlp_fxp_forward().
 * For N <= 1 layers this is 0.
 * For N > 1 layers this is 2 * max(out_dim of all non-final layers).
 */
size_t mlp_fxp_required_work_len(const mlp_fxp_layer_t* layers, size_t n_layers);

/*
 * Very fast ReLU over vectors. In-place is allowed only for
 * mlp_fxp_relu_inplace().
 */
void mlp_fxp_relu(const fxp_t* MLP_FXP_RESTRICT src, fxp_t* MLP_FXP_RESTRICT dst, size_t n);

void mlp_fxp_relu_inplace(fxp_t* x, size_t n);

/*
 * Dense layer:
 *   output[out_dim] = activation(weights[out_dim][in_dim] * input[in_dim] + bias)
 *
 * This is the main primitive for single-sample small MLP inference.
 */
void mlp_fxp_dense(const fxp_t* MLP_FXP_RESTRICT weights, const fxp_t* MLP_FXP_RESTRICT bias,
                   const fxp_t* MLP_FXP_RESTRICT input, fxp_t* MLP_FXP_RESTRICT output, size_t out_dim, size_t in_dim,
                   mlp_fxp_activation_t activation);

/*
 * Same as mlp_fxp_dense(..., MLP_FXP_ACT_NONE).
 */
void mlp_fxp_matvec(const fxp_t* MLP_FXP_RESTRICT weights, const fxp_t* MLP_FXP_RESTRICT bias,
                    const fxp_t* MLP_FXP_RESTRICT input, fxp_t* MLP_FXP_RESTRICT output, size_t out_dim, size_t in_dim);

/*
 * Same as mlp_fxp_dense(..., MLP_FXP_ACT_RELU).
 */
void mlp_fxp_matvec_relu(const fxp_t* MLP_FXP_RESTRICT weights, const fxp_t* MLP_FXP_RESTRICT bias,
                         const fxp_t* MLP_FXP_RESTRICT input, fxp_t* MLP_FXP_RESTRICT output, size_t out_dim,
                         size_t in_dim);

/*
 * Batched dense layer with the same weight layout as mlp_fxp_dense:
 *   output[batch][out_dim] =
 *       activation(input[batch][in_dim] * transpose(weights[out_dim][in_dim]) + bias)
 */
void mlp_fxp_dense_batch(const fxp_t* MLP_FXP_RESTRICT input, const fxp_t* MLP_FXP_RESTRICT weights,
                         const fxp_t* MLP_FXP_RESTRICT bias, fxp_t* MLP_FXP_RESTRICT output, size_t batch,
                         size_t out_dim, size_t in_dim, mlp_fxp_activation_t activation);

/*
 * Full MLP inference. Work buffer may be NULL only when required_work_len == 0.
 */
int mlp_fxp_forward(const mlp_fxp_layer_t* layers, size_t n_layers, const fxp_t* input, fxp_t* output, fxp_t* work,
                    size_t work_len);

/*
 * Generic row-major fixed-point matrix multiplication:
 *   C[M][N] = A[M][K] * B[K][N]
 *
 * Cache behavior:
 *   - A is streamed by row.
 *   - B is consumed in small contiguous column groups.
 *   - Each dot product is accumulated in __int128 and downscaled once.
 */
void mlp_fxp_matmul(const fxp_t* MLP_FXP_RESTRICT A, const fxp_t* MLP_FXP_RESTRICT B, fxp_t* MLP_FXP_RESTRICT C,
                    size_t M, size_t N, size_t K);

/*
 * Generic matrix multiplication plus optional per-column bias and activation:
 *   C[M][N] = activation(A[M][K] * B[K][N] + bias[N])
 */
void mlp_fxp_matmul_bias_act(const fxp_t* MLP_FXP_RESTRICT A, const fxp_t* MLP_FXP_RESTRICT B,
                             const fxp_t* MLP_FXP_RESTRICT bias, fxp_t* MLP_FXP_RESTRICT C, size_t M, size_t N,
                             size_t K, mlp_fxp_activation_t activation);

/*
 * Row-major transpose:
 *   dst[cols][rows] = transpose(src[rows][cols])
 */
void mlp_fxp_transpose(const fxp_t* MLP_FXP_RESTRICT src, fxp_t* MLP_FXP_RESTRICT dst, size_t rows, size_t cols);

#if defined(__cplusplus)
}
#endif

#endif /* MLP_FXP_H */