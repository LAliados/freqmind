#include "mlp.h"

#ifndef MLP_FXP_UNROLL
#define MLP_FXP_UNROLL 4
#endif

#if defined(__GNUC__) || defined(__clang__)
#define MLP_FXP_PREFETCH_R(p) __builtin_prefetch((p), 0, 1)
#else
#define MLP_FXP_PREFETCH_R(p) ((void)0)
#endif

static inline s64 mlp_fxp_downscale_acc(__int128 acc) {
#if MLP_FXP_FAST_ACC_SHIFT
    if (acc >= 0) {
        return (s64)((unsigned __int128)acc >> FXP48_16_FRAC_BITS);
    } else {
        /*
         * Magnitude without undefined behavior on the theoretical minimum
         * signed __int128 value: mag = -acc.
         */
        unsigned __int128 mag = (unsigned __int128)(-(acc + 1)) + 1u;
        return -(s64)(mag >> FXP48_16_FRAC_BITS);
    }
#else
    return (s64)(acc / (__int128)FXP48_16_ONE);
#endif
}

static inline __int128 mlp_fxp_dot_raw(const fxp_t* MLP_FXP_RESTRICT a, const fxp_t* MLP_FXP_RESTRICT b, size_t n) {
    size_t i = 0;
    __int128 acc0 = 0;
    __int128 acc1 = 0;
    __int128 acc2 = 0;
    __int128 acc3 = 0;

    for (; i + 4 <= n; i += 4) {
        acc0 += (__int128)a[i + 0]._val * (__int128)b[i + 0]._val;
        acc1 += (__int128)a[i + 1]._val * (__int128)b[i + 1]._val;
        acc2 += (__int128)a[i + 2]._val * (__int128)b[i + 2]._val;
        acc3 += (__int128)a[i + 3]._val * (__int128)b[i + 3]._val;
    }

    acc0 += acc1 + acc2 + acc3;

    for (; i < n; ++i) {
        acc0 += (__int128)a[i]._val * (__int128)b[i]._val;
    }

    return acc0;
}

static inline fxp_t mlp_fxp_finish_acc(__int128 acc, const fxp_t* bias, size_t bias_index,
                                       mlp_fxp_activation_t activation) {
    s64 raw = mlp_fxp_downscale_acc(acc);

    if (bias != 0) {
        raw += bias[bias_index]._val;
    }

    if (activation == MLP_FXP_ACT_RELU) {
        raw = mlp_fxp_relu_raw(raw);
    }

    return mlp_fxp_from_raw(raw);
}

const char* mlp_fxp_status_string(int status) {
    switch (status) {
        case MLP_FXP_OK:
            return "MLP_FXP_OK";
        case MLP_FXP_ERR_ARG:
            return "MLP_FXP_ERR_ARG";
        case MLP_FXP_ERR_SHAPE:
            return "MLP_FXP_ERR_SHAPE";
        case MLP_FXP_ERR_WORK:
            return "MLP_FXP_ERR_WORK";
        default:
            return "MLP_FXP_ERR_UNKNOWN";
    }
}

size_t mlp_fxp_required_work_len(const mlp_fxp_layer_t* layers, size_t n_layers) {
    size_t i;
    size_t max_hidden = 0;

    if (layers == 0 || n_layers <= 1) {
        return 0;
    }

    for (i = 0; i + 1 < n_layers; ++i) {
        if (layers[i].out_dim > max_hidden) {
            max_hidden = layers[i].out_dim;
        }
    }

    return 2u * max_hidden;
}

void mlp_fxp_relu(const fxp_t* MLP_FXP_RESTRICT src, fxp_t* MLP_FXP_RESTRICT dst, size_t n) {
    size_t i = 0;

    if (src == 0 || dst == 0) {
        return;
    }

    for (; i + 8 <= n; i += 8) {
        dst[i + 0] = mlp_fxp_from_raw(mlp_fxp_relu_raw(src[i + 0]._val));
        dst[i + 1] = mlp_fxp_from_raw(mlp_fxp_relu_raw(src[i + 1]._val));
        dst[i + 2] = mlp_fxp_from_raw(mlp_fxp_relu_raw(src[i + 2]._val));
        dst[i + 3] = mlp_fxp_from_raw(mlp_fxp_relu_raw(src[i + 3]._val));
        dst[i + 4] = mlp_fxp_from_raw(mlp_fxp_relu_raw(src[i + 4]._val));
        dst[i + 5] = mlp_fxp_from_raw(mlp_fxp_relu_raw(src[i + 5]._val));
        dst[i + 6] = mlp_fxp_from_raw(mlp_fxp_relu_raw(src[i + 6]._val));
        dst[i + 7] = mlp_fxp_from_raw(mlp_fxp_relu_raw(src[i + 7]._val));
    }

    for (; i < n; ++i) {
        dst[i] = mlp_fxp_from_raw(mlp_fxp_relu_raw(src[i]._val));
    }
}

void mlp_fxp_relu_inplace(fxp_t* x, size_t n) {
    size_t i = 0;

    if (x == 0) {
        return;
    }

    for (; i + 8 <= n; i += 8) {
        x[i + 0] = mlp_fxp_from_raw(mlp_fxp_relu_raw(x[i + 0]._val));
        x[i + 1] = mlp_fxp_from_raw(mlp_fxp_relu_raw(x[i + 1]._val));
        x[i + 2] = mlp_fxp_from_raw(mlp_fxp_relu_raw(x[i + 2]._val));
        x[i + 3] = mlp_fxp_from_raw(mlp_fxp_relu_raw(x[i + 3]._val));
        x[i + 4] = mlp_fxp_from_raw(mlp_fxp_relu_raw(x[i + 4]._val));
        x[i + 5] = mlp_fxp_from_raw(mlp_fxp_relu_raw(x[i + 5]._val));
        x[i + 6] = mlp_fxp_from_raw(mlp_fxp_relu_raw(x[i + 6]._val));
        x[i + 7] = mlp_fxp_from_raw(mlp_fxp_relu_raw(x[i + 7]._val));
    }

    for (; i < n; ++i) {
        x[i] = mlp_fxp_from_raw(mlp_fxp_relu_raw(x[i]._val));
    }
}

void mlp_fxp_dense(const fxp_t* MLP_FXP_RESTRICT weights, const fxp_t* MLP_FXP_RESTRICT bias,
                   const fxp_t* MLP_FXP_RESTRICT input, fxp_t* MLP_FXP_RESTRICT output, size_t out_dim, size_t in_dim,
                   mlp_fxp_activation_t activation) {
    size_t r;

    if (weights == 0 || input == 0 || output == 0) {
        return;
    }

    for (r = 0; r < out_dim; ++r) {
        const fxp_t* wrow = weights + r * in_dim;
        __int128 acc = mlp_fxp_dot_raw(wrow, input, in_dim);
        output[r] = mlp_fxp_finish_acc(acc, bias, r, activation);
    }
}

void mlp_fxp_matvec(const fxp_t* MLP_FXP_RESTRICT weights, const fxp_t* MLP_FXP_RESTRICT bias,
                    const fxp_t* MLP_FXP_RESTRICT input, fxp_t* MLP_FXP_RESTRICT output, size_t out_dim,
                    size_t in_dim) {
    mlp_fxp_dense(weights, bias, input, output, out_dim, in_dim, MLP_FXP_ACT_NONE);
}

void mlp_fxp_matvec_relu(const fxp_t* MLP_FXP_RESTRICT weights, const fxp_t* MLP_FXP_RESTRICT bias,
                         const fxp_t* MLP_FXP_RESTRICT input, fxp_t* MLP_FXP_RESTRICT output, size_t out_dim,
                         size_t in_dim) {
    mlp_fxp_dense(weights, bias, input, output, out_dim, in_dim, MLP_FXP_ACT_RELU);
}

void mlp_fxp_dense_batch(const fxp_t* MLP_FXP_RESTRICT input, const fxp_t* MLP_FXP_RESTRICT weights,
                         const fxp_t* MLP_FXP_RESTRICT bias, fxp_t* MLP_FXP_RESTRICT output, size_t batch,
                         size_t out_dim, size_t in_dim, mlp_fxp_activation_t activation) {
    size_t b;

    if (input == 0 || weights == 0 || output == 0) {
        return;
    }

    for (b = 0; b < batch; ++b) {
        mlp_fxp_dense(weights, bias, input + b * in_dim, output + b * out_dim, out_dim, in_dim, activation);
    }
}

int mlp_fxp_forward(const mlp_fxp_layer_t* layers, size_t n_layers, const fxp_t* input, fxp_t* output, fxp_t* work,
                    size_t work_len) {
    size_t i;
    size_t need;
    size_t max_hidden;
    const fxp_t* src;
    fxp_t* buf0;
    fxp_t* buf1;

    if (layers == 0 || n_layers == 0 || input == 0 || output == 0) {
        return MLP_FXP_ERR_ARG;
    }

    for (i = 0; i < n_layers; ++i) {
        if (layers[i].weights == 0 || layers[i].in_dim == 0 || layers[i].out_dim == 0) {
            return MLP_FXP_ERR_ARG;
        }

        if (i > 0 && layers[i].in_dim != layers[i - 1].out_dim) {
            return MLP_FXP_ERR_SHAPE;
        }

        if (layers[i].activation != MLP_FXP_ACT_NONE && layers[i].activation != MLP_FXP_ACT_RELU) {
            return MLP_FXP_ERR_ARG;
        }
    }

    need = mlp_fxp_required_work_len(layers, n_layers);
    if (work_len < need || (need != 0 && work == 0)) {
        return MLP_FXP_ERR_WORK;
    }

    if (n_layers == 1) {
        mlp_fxp_dense(layers[0].weights, layers[0].bias, input, output, layers[0].out_dim, layers[0].in_dim,
                      layers[0].activation);
        return MLP_FXP_OK;
    }

    max_hidden = need / 2u;
    buf0 = work;
    buf1 = work + max_hidden;
    src = input;

    for (i = 0; i < n_layers; ++i) {
        fxp_t* dst;

        if (i + 1 == n_layers) {
            dst = output;
        } else {
            dst = (i & 1u) ? buf1 : buf0;
        }

        mlp_fxp_dense(layers[i].weights, layers[i].bias, src, dst, layers[i].out_dim, layers[i].in_dim,
                      layers[i].activation);

        src = dst;
    }

    return MLP_FXP_OK;
}

void mlp_fxp_matmul(const fxp_t* MLP_FXP_RESTRICT A, const fxp_t* MLP_FXP_RESTRICT B, fxp_t* MLP_FXP_RESTRICT C,
                    size_t M, size_t N, size_t K) {
    mlp_fxp_matmul_bias_act(A, B, 0, C, M, N, K, MLP_FXP_ACT_NONE);
}

void mlp_fxp_matmul_bias_act(const fxp_t* MLP_FXP_RESTRICT A, const fxp_t* MLP_FXP_RESTRICT B,
                             const fxp_t* MLP_FXP_RESTRICT bias, fxp_t* MLP_FXP_RESTRICT C, size_t M, size_t N,
                             size_t K, mlp_fxp_activation_t activation) {
    size_t i;

    if (A == 0 || B == 0 || C == 0) {
        return;
    }

    for (i = 0; i < M; ++i) {
        const fxp_t* arow = A + i * K;
        fxp_t* crow = C + i * N;
        size_t j = 0;

        /*
         * Four output columns at a time:
         * for each k, B[k][j..j+3] is contiguous, which is friendlier to cache
         * than computing one strided column at a time.
         */
        for (; j + 4 <= N; j += 4) {
            __int128 acc0 = 0;
            __int128 acc1 = 0;
            __int128 acc2 = 0;
            __int128 acc3 = 0;
            size_t k;

            for (k = 0; k < K; ++k) {
                s64 av = arow[k]._val;
                const fxp_t* brow = B + k * N + j;

                if (k + 4 < K) {
                    MLP_FXP_PREFETCH_R(B + (k + 4) * N + j);
                }

                acc0 += (__int128)av * (__int128)brow[0]._val;
                acc1 += (__int128)av * (__int128)brow[1]._val;
                acc2 += (__int128)av * (__int128)brow[2]._val;
                acc3 += (__int128)av * (__int128)brow[3]._val;
            }

            crow[j + 0] = mlp_fxp_finish_acc(acc0, bias, j + 0, activation);
            crow[j + 1] = mlp_fxp_finish_acc(acc1, bias, j + 1, activation);
            crow[j + 2] = mlp_fxp_finish_acc(acc2, bias, j + 2, activation);
            crow[j + 3] = mlp_fxp_finish_acc(acc3, bias, j + 3, activation);
        }

        for (; j < N; ++j) {
            __int128 acc = 0;
            size_t k;

            for (k = 0; k < K; ++k) {
                acc += (__int128)arow[k]._val * (__int128)B[k * N + j]._val;
            }

            crow[j] = mlp_fxp_finish_acc(acc, bias, j, activation);
        }
    }
}

void mlp_fxp_transpose(const fxp_t* MLP_FXP_RESTRICT src, fxp_t* MLP_FXP_RESTRICT dst, size_t rows, size_t cols) {
    size_t r;

    if (src == 0 || dst == 0) {
        return;
    }

    for (r = 0; r < rows; ++r) {
        size_t c;
        const fxp_t* srow = src + r * cols;

        for (c = 0; c < cols; ++c) {
            dst[c * rows + r] = srow[c];
        }
    }
}