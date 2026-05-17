#include "ga.h"

#include <linux/errno.h>
#include <linux/gfp.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/string.h>
#include <linux/types.h>

/*
 * Fixed-point GA implementation for Linux kernel modules.
 *
 * No libc, no floating point, no libm.
 *
 * Important:
 *     This file intentionally does not use:
 *         - compiler-emitted 128-bit arithmetic helpers
 *         - FXP_MUL
 *         - FXP_DIV
 *         - FXP_DIV_INT
 *
 * Reason:
 *     Some kernel-module builds cannot link compiler runtime helpers for
 *     wide integer division.
 */

#define GA_U64_MAX_VALUE (~0ULL)
#define GA_S64_MAX_VALUE ((s64)((~0ULL) >> 1))
#define GA_S64_MIN_VALUE (-GA_S64_MAX_VALUE - 1)

#ifndef GA_INCEST_RATE_RAW
#define GA_INCEST_RATE_RAW ((s64)(FXP48_16_ONE / 4)) /* 0.25 */
#endif

#ifndef GA_INCEST_NOISE_RAW
#define GA_INCEST_NOISE_RAW ((s64)((FXP48_16_ONE * 35) / 100)) /* 0.35 */
#endif

#ifndef GA_INCEST_BLEND_NOISE_RAW
#define GA_INCEST_BLEND_NOISE_RAW ((s64)((FXP48_16_ONE * 10) / 100)) /* 0.10 */
#endif

static fxp_t ga_fxp_zero(void) {
    return FXP_FROM_INT(0);
}

static fxp_t ga_fxp_one(void) {
    return FXP_FROM_INT(1);
}

static fxp_t ga_fxp_raw(s64 raw) {
    return FXP48_16_RAW(raw);
}

static fxp_t ga_fxp_neg_inf(void) {
    return ga_fxp_raw((s64)(GA_S64_MIN_VALUE / 4));
}

static u64 ga_abs_s64_to_u64(s64 x) {
    if (x >= 0)
        return (u64)x;

    if (x == GA_S64_MIN_VALUE)
        return ((u64)1) << 63;

    return (u64)(-x);
}

static fxp_t ga_fxp_from_mag(int negative, u64 mag) {
    if (negative) {
        if (mag >= (((u64)1) << 63))
            return ga_fxp_raw(GA_S64_MIN_VALUE);

        return ga_fxp_raw(-(s64)mag);
    }

    if (mag > (u64)GA_S64_MAX_VALUE)
        return ga_fxp_raw(GA_S64_MAX_VALUE);

    return ga_fxp_raw((s64)mag);
}

static u64 ga_u64_add_sat(u64 a, u64 b) {
    if (a > GA_U64_MAX_VALUE - b)
        return GA_U64_MAX_VALUE;

    return a + b;
}

static u64 ga_u64_shl_sat(u64 v, unsigned int shift) {
    if (shift >= 64)
        return v ? GA_U64_MAX_VALUE : 0;

    if (v > (GA_U64_MAX_VALUE >> shift))
        return GA_U64_MAX_VALUE;

    return v << shift;
}

/*
 * Compute saturating ((a * b) >> 16) using only 32x32->64 products.
 */
static u64 ga_umul_shr16_sat(u64 a, u64 b) {
    u64 ah = a >> 32;
    u64 al = a & 0xffffffffULL;
    u64 bh = b >> 32;
    u64 bl = b & 0xffffffffULL;
    u64 acc = 0;
    u64 part;

    part = al * bl;
    acc = ga_u64_add_sat(acc, part >> 16);

    part = ah * bl;
    acc = ga_u64_add_sat(acc, ga_u64_shl_sat(part, 16));

    part = al * bh;
    acc = ga_u64_add_sat(acc, ga_u64_shl_sat(part, 16));

    part = ah * bh;
    acc = ga_u64_add_sat(acc, ga_u64_shl_sat(part, 48));

    return acc;
}

static fxp_t ga_fxp_mul_safe(fxp_t a, fxp_t b) {
    s64 ar = FXP_RAW(a);
    s64 br = FXP_RAW(b);
    int negative = ((ar < 0) != (br < 0));
    u64 amag = ga_abs_s64_to_u64(ar);
    u64 bmag = ga_abs_s64_to_u64(br);
    u64 mag = ga_umul_shr16_sat(amag, bmag);

    return ga_fxp_from_mag(negative, mag);
}

/*
 * Compute saturating ((num_abs << 16) / den_abs) without 128-bit division.
 *
 * The virtual numerator is up to 80 bits. Binary long division is used over
 * bit positions 79..0.
 */
static u64 ga_udiv_scaled16_sat(u64 num_abs, u64 den_abs) {
    u64 q = 0;
    u64 rem = 0;
    int pos;

    if (den_abs == 0)
        return GA_U64_MAX_VALUE;

    for (pos = 79; pos >= 0; --pos) {
        u64 bit = 0;

        if (pos >= 16)
            bit = (num_abs >> (pos - 16)) & 1ULL;

        /*
         * rem is always less than den_abs before this shift.
         * den_abs is at most 2^63, so rem << 1 cannot exceed u64 max.
         */
        rem = (rem << 1) | bit;

        if (rem >= den_abs) {
            rem -= den_abs;

            if (pos >= 64)
                return GA_U64_MAX_VALUE;

            q |= 1ULL << pos;
        }
    }

    return q;
}

static fxp_t ga_fxp_div_raw_safe(fxp_t a, s64 b_raw, fxp_t fallback) {
    s64 ar = FXP_RAW(a);
    int negative;
    u64 amag;
    u64 bmag;
    u64 mag;

    if (b_raw == 0)
        return fallback;

    negative = ((ar < 0) != (b_raw < 0));
    amag = ga_abs_s64_to_u64(ar);
    bmag = ga_abs_s64_to_u64(b_raw);

    mag = ga_udiv_scaled16_sat(amag, bmag);

    return ga_fxp_from_mag(negative, mag);
}

static fxp_t ga_fxp_div_safe(fxp_t a, fxp_t b, fxp_t fallback) {
    return ga_fxp_div_raw_safe(a, FXP_RAW(b), fallback);
}

static fxp_t ga_fxp_div_int_safe(fxp_t a, s64 b, fxp_t fallback) {
    s64 ar = FXP_RAW(a);

    if (b == 0)
        return fallback;

    return ga_fxp_raw(ar / b);
}

static fxp_t ga_fxp_ratio(s64 num, s64 den) {
    int negative;
    u64 nmag;
    u64 dmag;
    u64 mag;

    if (den == 0)
        return ga_fxp_zero();

    negative = ((num < 0) != (den < 0));
    nmag = ga_abs_s64_to_u64(num);
    dmag = ga_abs_s64_to_u64(den);

    mag = ga_udiv_scaled16_sat(nmag, dmag);

    return ga_fxp_from_mag(negative, mag);
}

static int ga_fxp_gt(fxp_t a, fxp_t b) {
    return FXP_RAW(a) > FXP_RAW(b);
}

static int ga_fxp_lt(fxp_t a, fxp_t b) {
    return FXP_RAW(a) < FXP_RAW(b);
}

static fxp_t ga_fxp_abs(fxp_t x) {
    s64 raw = FXP_RAW(x);

    if (raw >= 0)
        return x;

    if (raw == GA_S64_MIN_VALUE)
        return ga_fxp_raw(GA_S64_MAX_VALUE);

    return ga_fxp_raw(-raw);
}

static fxp_t ga_fxp_clamp_value(fxp_t x, fxp_t lo, fxp_t hi) {
    if (ga_fxp_lt(x, lo))
        return lo;

    if (ga_fxp_gt(x, hi))
        return hi;

    return x;
}

static fxp_t* ga_alloc_copy(const fxp_t* src, size_t n, gfp_t flags) {
    fxp_t* dst;

    dst = kmalloc_array(n, sizeof(*dst), flags);
    if (!dst)
        return NULL;

    memcpy(dst, src, n * sizeof(*dst));
    return dst;
}

static fxp_t* ga_alloc_zero_vecs(size_t count, size_t n, gfp_t flags) {
    if (count != 0 && n > ((size_t)-1) / count)
        return NULL;

    return kcalloc(count * n, sizeof(fxp_t), flags);
}

void ga_default_config(ga_config* c, size_t n_params) {
    if (!c)
        return;

    memset(c, 0, sizeof(*c));

    c->n_params = n_params;

    c->offspring_count = 20;
    c->elite_count = 3;

    c->max_generations = 1;
    c->max_evaluations = 0;
    c->patience_generations = 0;

    c->alloc_flags = GFP_KERNEL;

    c->sigma_init = ga_fxp_ratio(1, 4);     /* 0.25 */
    c->sigma_min = ga_fxp_ratio(1, 10000);  /* 0.0001 */
    c->sigma_decay = ga_fxp_ratio(72, 100); /* 0.72 */
    c->sigma_grow = ga_fxp_ratio(105, 100); /* 1.05 */
    c->sigma_max = ga_fxp_ratio(1, 2);      /* 0.50 */

    c->lambda_pretrained = ga_fxp_ratio(20, 100); /* 0.20 */
    c->lambda_start = ga_fxp_ratio(12, 100);      /* 0.12 */
    c->lambda_step = ga_fxp_ratio(3, 100);        /* 0.03 */
    c->lambda_radius = ga_fxp_ratio(25, 100);     /* 0.25 */
    c->trust_radius = FXP_FROM_INT(4);

    c->anchor_pull = ga_fxp_ratio(5, 100); /* 0.05 */

    c->differential_rate = ga_fxp_ratio(25, 100);   /* 0.25 */
    c->differential_weight = ga_fxp_ratio(60, 100); /* 0.60 */

    c->min_improvement = ga_fxp_zero();
    c->seed = 88172645463393265ULL;
}

static u64 ga_rng_next(struct ga* g) {
    u64 x = g->rng;

    if (x == 0)
        x = 88172645463393265ULL;

    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;

    g->rng = x;

    return x * 2685821657736338717ULL;
}

static fxp_t ga_rand01(struct ga* g) {
    s64 raw = (s64)(ga_rng_next(g) & (u64)(FXP48_16_ONE - 1));

    return ga_fxp_raw(raw);
}

static size_t ga_rand_index(struct ga* g, size_t n) {
    if (n == 0)
        return 0;

    return (size_t)(ga_rng_next(g) % (u64)n);
}

static fxp_t ga_randn(struct ga* g) {
    s64 acc = 0;
    size_t i;

    /*
     * Irwin-Hall approximation of N(0, 1):
     *     sum_{i=1..12} U(0, 1) - 6
     */
    for (i = 0; i < 12; ++i)
        acc += FXP_RAW(ga_rand01(g)) - (FXP48_16_ONE / 2);

    return ga_fxp_raw(acc);
}

static fxp_t ga_scale_at(const struct ga* g, size_t i) {
    fxp_t s;

    if (!g || !g->scale)
        return ga_fxp_one();

    s = ga_fxp_abs(g->scale[i]);

    if (FXP_RAW(s) <= 0)
        return ga_fxp_one();

    return s;
}

static s64 ga_scale_raw_at(const struct ga* g, size_t i) {
    s64 raw = FXP_RAW(ga_scale_at(g, i));

    if (raw == 0)
        return FXP48_16_ONE;

    return raw;
}

static void ga_clamp_vec(struct ga* g, fxp_t* x) {
    size_t i;

    if (!g || !x)
        return;

    for (i = 0; i < g->n; ++i) {
        if (g->lower && ga_fxp_lt(x[i], g->lower[i]))
            x[i] = g->lower[i];

        if (g->upper && ga_fxp_gt(x[i], g->upper[i]))
            x[i] = g->upper[i];
    }
}

static fxp_t ga_norm2(const struct ga* g, const fxp_t* a, const fxp_t* b) {
    fxp_t acc = ga_fxp_zero();
    size_t i;

    if (!g || !a || !b || g->n == 0)
        return acc;

    for (i = 0; i < g->n; ++i) {
        fxp_t diff = FXP_SUB(a[i], b[i]);
        fxp_t d = ga_fxp_div_raw_safe(diff, ga_scale_raw_at(g, i), diff);
        fxp_t d2 = ga_fxp_mul_safe(d, d);

        acc = FXP_ADD(acc, d2);
    }

    return ga_fxp_div_int_safe(acc, (s64)g->n, acc);
}

static fxp_t ga_penalty(struct ga* g, const fxp_t* x, const fxp_t* parent) {
    fxp_t p = ga_fxp_zero();
    fxp_t n_pretrained;
    fxp_t n_start;
    fxp_t trust2;

    if (!g || !x || !g->pretrained || !g->start || g->n == 0)
        return p;

    n_pretrained = ga_norm2(g, x, g->pretrained);
    n_start = ga_norm2(g, x, g->start);

    p = FXP_ADD(p, ga_fxp_mul_safe(g->lambda_pretrained, n_pretrained));
    p = FXP_ADD(p, ga_fxp_mul_safe(g->lambda_start, n_start));

    if (parent) {
        fxp_t n_parent = ga_norm2(g, x, parent);

        p = FXP_ADD(p, ga_fxp_mul_safe(g->lambda_step, n_parent));
    }

    /*
     * No sqrt is used here.
     *
     * Previous formula:
     *     if sqrt(n_start) > trust_radius:
     *         penalty += lambda_radius * (sqrt(n_start) - trust_radius)^2
     *
     * Kernel-safe formula:
     *     if n_start > trust_radius^2:
     *         penalty += lambda_radius * (n_start - trust_radius^2)
     */
    trust2 = ga_fxp_mul_safe(g->trust_radius, g->trust_radius);

    if (FXP_RAW(g->trust_radius) > 0 && ga_fxp_gt(n_start, trust2)) {
        fxp_t over2 = FXP_SUB(n_start, trust2);

        p = FXP_ADD(p, ga_fxp_mul_safe(g->lambda_radius, over2));
    }

    return p;
}

static int ga_same_vec(const struct ga* g, const fxp_t* a, const fxp_t* b) {
    size_t i;

    if (!g || !a || !b || g->n == 0)
        return 0;

    for (i = 0; i < g->n; ++i) {
        fxp_t diff = ga_fxp_abs(FXP_SUB(a[i], b[i]));
        fxp_t eps = ga_fxp_div_int_safe(ga_scale_at(g, i), 65536, ga_fxp_raw(1));

        if (ga_fxp_gt(diff, eps))
            return 0;
    }

    return 1;
}

static size_t ga_find_closest_elite(const struct ga* g, size_t parent_idx) {
    size_t best_idx = parent_idx;
    fxp_t best_d = ga_fxp_raw(GA_S64_MAX_VALUE);
    const fxp_t* parent;
    size_t j;

    if (!g || g->elite_valid < 2 || parent_idx >= g->elite_valid)
        return best_idx;

    parent = g->elites + parent_idx * g->n;

    for (j = 0; j < g->elite_valid; ++j) {
        const fxp_t* candidate;
        fxp_t d;

        if (j == parent_idx)
            continue;

        candidate = g->elites + j * g->n;
        d = ga_norm2(g, parent, candidate);

        if (ga_fxp_lt(d, best_d)) {
            best_d = d;
            best_idx = j;
        }
    }

    return best_idx;
}

static void ga_make_inbred_child(struct ga* g, size_t parent_idx, fxp_t* child) {
    size_t mate_idx = ga_find_closest_elite(g, parent_idx);
    const fxp_t* parent = g->elites + parent_idx * g->n;
    const fxp_t* mate = g->elites + mate_idx * g->n;
    fxp_t alpha;
    size_t i;

    alpha = FXP_ADD(ga_fxp_ratio(1, 2), ga_fxp_mul_safe(ga_randn(g), ga_fxp_raw(GA_INCEST_BLEND_NOISE_RAW)));

    alpha = ga_fxp_clamp_value(alpha, ga_fxp_ratio(15, 100), ga_fxp_ratio(85, 100));

    for (i = 0; i < g->n; ++i) {
        fxp_t blended =
            FXP_ADD(ga_fxp_mul_safe(alpha, parent[i]), ga_fxp_mul_safe(FXP_SUB(ga_fxp_one(), alpha), mate[i]));

        fxp_t noise =
            ga_fxp_mul_safe(ga_fxp_mul_safe(ga_fxp_mul_safe(ga_randn(g), g->sigma), ga_fxp_raw(GA_INCEST_NOISE_RAW)),
                            ga_scale_at(g, i));

        child[i] = FXP_ADD(blended, noise);
    }
}

static void ga_apply_anchor_pull(struct ga* g, fxp_t* child) {
    fxp_t a;
    const fxp_t* anchor;
    size_t i;

    if (!g || !child || FXP_RAW(g->anchor_pull) <= 0)
        return;

    anchor = ga_fxp_lt(ga_rand01(g), ga_fxp_ratio(1, 2)) ? g->start : g->pretrained;

    a = ga_fxp_gt(g->anchor_pull, ga_fxp_one()) ? ga_fxp_one() : g->anchor_pull;

    for (i = 0; i < g->n; ++i) {
        child[i] = FXP_ADD(ga_fxp_mul_safe(FXP_SUB(ga_fxp_one(), a), child[i]), ga_fxp_mul_safe(a, anchor[i]));
    }
}

static void ga_generate_initial(struct ga* g) {
    size_t j;

    g->active_index = 0;
    g->evaluated_in_generation = 0;
    g->has_pending = 0;

    memcpy(g->offspring, g->start, g->n * sizeof(fxp_t));
    g->offspring_penalty[0] = ga_penalty(g, g->offspring, NULL);

    for (j = 1; j < g->offspring_count; ++j) {
        fxp_t* child = g->offspring + j * g->n;

        if (j == 1 && !ga_same_vec(g, g->start, g->pretrained)) {
            memcpy(child, g->pretrained, g->n * sizeof(fxp_t));
        } else {
            const fxp_t* base = ga_fxp_lt(ga_rand01(g), ga_fxp_ratio(75, 100)) ? g->start : g->pretrained;

            size_t i;

            for (i = 0; i < g->n; ++i) {
                fxp_t noise = ga_fxp_mul_safe(ga_fxp_mul_safe(ga_randn(g), g->sigma), ga_scale_at(g, i));

                child[i] = FXP_ADD(base[i], noise);
            }
        }

        ga_clamp_vec(g, child);
        g->offspring_penalty[j] = ga_penalty(g, child, NULL);
    }
}

static void ga_generate_next_offspring(struct ga* g) {
    size_t j;

    g->active_index = 0;
    g->evaluated_in_generation = 0;
    g->has_pending = 0;

    if (g->elite_valid == 0) {
        ga_generate_initial(g);
        return;
    }

    for (j = 0; j < g->offspring_count; ++j) {
        size_t parent_idx = ga_rand_index(g, g->elite_valid);
        const fxp_t* parent = g->elites + parent_idx * g->n;
        fxp_t* child = g->offspring + j * g->n;

        if (g->elite_valid >= 2 && ga_fxp_lt(ga_rand01(g), ga_fxp_raw(GA_INCEST_RATE_RAW))) {
            ga_make_inbred_child(g, parent_idx, child);
        } else if (g->elite_valid >= 2 && ga_fxp_lt(ga_rand01(g), g->differential_rate)) {
            size_t a = ga_rand_index(g, g->elite_valid);
            size_t b = ga_rand_index(g, g->elite_valid);
            const fxp_t* ea;
            const fxp_t* eb;
            const fxp_t* best = g->elites;
            size_t i;

            if (b == a)
                b = (b + 1) % g->elite_valid;

            ea = g->elites + a * g->n;
            eb = g->elites + b * g->n;

            for (i = 0; i < g->n; ++i) {
                fxp_t diff = FXP_SUB(ea[i], eb[i]);
                fxp_t step = ga_fxp_mul_safe(g->differential_weight, diff);

                fxp_t noise = ga_fxp_mul_safe(ga_fxp_mul_safe(ga_randn(g), g->sigma), ga_scale_at(g, i));

                child[i] = FXP_ADD(FXP_ADD(best[i], step), noise);
            }
        } else {
            size_t i;

            for (i = 0; i < g->n; ++i) {
                fxp_t noise = ga_fxp_mul_safe(ga_fxp_mul_safe(ga_randn(g), g->sigma), ga_scale_at(g, i));

                child[i] = FXP_ADD(parent[i], noise);
            }
        }

        ga_apply_anchor_pull(g, child);

        ga_clamp_vec(g, child);
        g->offspring_penalty[j] = ga_penalty(g, child, parent);
    }
}

static void ga_finish_generation(struct ga* g) {
    size_t total = 0;
    fxp_t mean = ga_fxp_zero();
    fxp_t norm_scale = ga_fxp_zero();
    int improved = 0;
    size_t j;
    size_t new_elites;
    size_t i;

    for (j = 0; j < g->elite_valid; ++j) {
        memcpy(g->pool + total * g->n, g->elites + j * g->n, g->n * sizeof(fxp_t));

        g->pool_raw[total] = g->elite_raw[j];
        g->pool_penalty[total] = g->elite_penalty[j];
        total++;
    }

    for (j = 0; j < g->evaluated_in_generation; ++j) {
        memcpy(g->pool + total * g->n, g->offspring + j * g->n, g->n * sizeof(fxp_t));

        g->pool_raw[total] = g->offspring_raw[j];
        g->pool_penalty[total] = g->offspring_penalty[j];
        total++;
    }

    if (total == 0) {
        g->done = 1;
        return;
    }

    for (j = 0; j < total; ++j)
        mean = FXP_ADD(mean, g->pool_raw[j]);

    mean = ga_fxp_div_int_safe(mean, (s64)total, ga_fxp_zero());

    /*
     * Mean absolute deviation is used instead of variance/stddev.
     * This avoids fixed-point square overflow and keeps divisor checks simple.
     */
    for (j = 0; j < total; ++j) {
        fxp_t d = ga_fxp_abs(FXP_SUB(g->pool_raw[j], mean));

        norm_scale = FXP_ADD(norm_scale, d);
    }

    norm_scale = ga_fxp_div_int_safe(norm_scale, (s64)total, ga_fxp_one());

    if (FXP_RAW(norm_scale) <= 0)
        norm_scale = ga_fxp_one();

    for (j = 0; j < total; ++j) {
        fxp_t normalized;

        normalized = ga_fxp_div_safe(FXP_SUB(g->pool_raw[j], mean), norm_scale, ga_fxp_zero());

        g->pool_fitness[j] = FXP_SUB(normalized, g->pool_penalty[j]);

        if (!g->has_best_observed || ga_fxp_gt(g->pool_raw[j], FXP_ADD(g->best_observed_raw, g->min_improvement))) {
            g->best_observed_raw = g->pool_raw[j];

            memcpy(g->best_observed, g->pool + j * g->n, g->n * sizeof(fxp_t));

            g->has_best_observed = 1;
            improved = 1;
        }
    }

    memset(g->pool_selected, 0, (g->offspring_count + g->elite_count) * sizeof(unsigned char));

    new_elites = g->elite_count < total ? g->elite_count : total;

    for (i = 0; i < new_elites; ++i) {
        size_t best = 0;
        fxp_t best_fit = ga_fxp_neg_inf();
        int found = 0;

        for (j = 0; j < total; ++j) {
            if (!g->pool_selected[j] && (!found || ga_fxp_gt(g->pool_fitness[j], best_fit))) {
                best_fit = g->pool_fitness[j];
                best = j;
                found = 1;
            }
        }

        g->pool_selected[best] = 1;

        memcpy(g->elites + i * g->n, g->pool + best * g->n, g->n * sizeof(fxp_t));

        g->elite_raw[i] = g->pool_raw[best];
        g->elite_penalty[i] = g->pool_penalty[best];
        g->elite_fitness[i] = g->pool_fitness[best];
    }

    g->elite_valid = new_elites;

    if (g->elite_valid > 0) {
        memcpy(g->best_regularized, g->elites, g->n * sizeof(fxp_t));

        g->best_regularized_raw = g->elite_raw[0];
        g->best_regularized_fitness = g->elite_fitness[0];
        g->has_best_regularized = 1;
    }

    if (improved) {
        g->no_improve_generations = 0;

        g->sigma = ga_fxp_mul_safe(g->sigma, g->sigma_grow);

        if (ga_fxp_gt(g->sigma, g->sigma_max))
            g->sigma = g->sigma_max;
    } else {
        g->no_improve_generations++;

        g->sigma = ga_fxp_mul_safe(g->sigma, g->sigma_decay);

        if (ga_fxp_lt(g->sigma, g->sigma_min))
            g->sigma = g->sigma_min;
    }

    g->generation++;

    if (g->max_generations > 0 && g->generation >= g->max_generations)
        g->done = 1;

    if (g->max_evaluations > 0 && g->evaluations >= g->max_evaluations)
        g->done = 1;

    if (g->patience_generations > 0 && g->no_improve_generations >= g->patience_generations) {
        g->done = 1;
    }

    if (!g->done)
        ga_generate_next_offspring(g);
}

int ga_init(struct ga* g, const fxp_t* pretrained_params, const ga_config* cfg) {
    const fxp_t* start;
    size_t pool_cap;
    gfp_t flags;

    if (!g || !pretrained_params || !cfg || cfg->n_params == 0)
        return -EINVAL;

    memset(g, 0, sizeof(*g));

    flags = cfg->alloc_flags ? cfg->alloc_flags : GFP_KERNEL;

    g->n = cfg->n_params;

    g->offspring_count = cfg->offspring_count ? cfg->offspring_count : 20;
    g->elite_count = cfg->elite_count ? cfg->elite_count : 3;

    if (g->elite_count > g->offspring_count)
        g->elite_count = g->offspring_count;

    g->max_generations = cfg->max_generations;
    g->max_evaluations = cfg->max_evaluations;
    g->patience_generations = cfg->patience_generations;

    g->sigma_init = FXP_RAW(cfg->sigma_init) > 0 ? cfg->sigma_init : ga_fxp_ratio(1, 4);

    g->sigma = g->sigma_init;

    g->sigma_min = FXP_RAW(cfg->sigma_min) > 0 ? cfg->sigma_min : ga_fxp_ratio(1, 10000);

    g->sigma_decay = FXP_RAW(cfg->sigma_decay) > 0 && ga_fxp_lt(cfg->sigma_decay, ga_fxp_one()) ? cfg->sigma_decay
                                                                                                : ga_fxp_ratio(72, 100);

    g->sigma_grow = ga_fxp_gt(cfg->sigma_grow, ga_fxp_one()) ? cfg->sigma_grow : ga_fxp_ratio(105, 100);

    g->sigma_max = FXP_RAW(cfg->sigma_max) > 0 ? cfg->sigma_max : g->sigma;

    if (ga_fxp_lt(g->sigma_max, g->sigma))
        g->sigma_max = g->sigma;

    g->lambda_pretrained = cfg->lambda_pretrained;
    g->lambda_start = cfg->lambda_start;
    g->lambda_step = cfg->lambda_step;
    g->lambda_radius = cfg->lambda_radius;
    g->trust_radius = cfg->trust_radius;
    g->anchor_pull = cfg->anchor_pull;
    g->differential_rate = cfg->differential_rate;
    g->differential_weight = cfg->differential_weight;
    g->min_improvement = cfg->min_improvement;
    g->rng = cfg->seed ? cfg->seed : 88172645463393265ULL;

    start = cfg->start_params ? cfg->start_params : pretrained_params;

    g->pretrained = ga_alloc_copy(pretrained_params, g->n, flags);
    g->start = ga_alloc_copy(start, g->n, flags);

    if (cfg->param_scale)
        g->scale = ga_alloc_copy(cfg->param_scale, g->n, flags);

    if (cfg->lower_bound)
        g->lower = ga_alloc_copy(cfg->lower_bound, g->n, flags);

    if (cfg->upper_bound)
        g->upper = ga_alloc_copy(cfg->upper_bound, g->n, flags);

    g->offspring = ga_alloc_zero_vecs(g->offspring_count, g->n, flags);
    g->offspring_raw = kcalloc(g->offspring_count, sizeof(fxp_t), flags);
    g->offspring_penalty = kcalloc(g->offspring_count, sizeof(fxp_t), flags);

    g->elites = ga_alloc_zero_vecs(g->elite_count, g->n, flags);
    g->elite_raw = kcalloc(g->elite_count, sizeof(fxp_t), flags);
    g->elite_penalty = kcalloc(g->elite_count, sizeof(fxp_t), flags);
    g->elite_fitness = kcalloc(g->elite_count, sizeof(fxp_t), flags);

    pool_cap = g->offspring_count + g->elite_count;

    if (pool_cap < g->offspring_count) {
        ga_free(g);
        return -ENOMEM;
    }

    g->pool = ga_alloc_zero_vecs(pool_cap, g->n, flags);
    g->pool_raw = kcalloc(pool_cap, sizeof(fxp_t), flags);
    g->pool_penalty = kcalloc(pool_cap, sizeof(fxp_t), flags);
    g->pool_fitness = kcalloc(pool_cap, sizeof(fxp_t), flags);
    g->pool_selected = kcalloc(pool_cap, sizeof(unsigned char), flags);

    g->best_regularized = kcalloc(g->n, sizeof(fxp_t), flags);
    g->best_observed = kcalloc(g->n, sizeof(fxp_t), flags);

    g->best_regularized_raw = ga_fxp_neg_inf();
    g->best_regularized_fitness = ga_fxp_neg_inf();
    g->best_observed_raw = ga_fxp_neg_inf();
    g->has_best_regularized = 0;
    g->has_best_observed = 0;

    if (!g->pretrained || !g->start || (cfg->param_scale && !g->scale) || (cfg->lower_bound && !g->lower) ||
        (cfg->upper_bound && !g->upper) || !g->offspring || !g->offspring_raw || !g->offspring_penalty || !g->elites ||
        !g->elite_raw || !g->elite_penalty || !g->elite_fitness || !g->pool || !g->pool_raw || !g->pool_penalty ||
        !g->pool_fitness || !g->pool_selected || !g->best_regularized || !g->best_observed) {
        ga_free(g);
        return -ENOMEM;
    }

    ga_generate_initial(g);

    return 0;
}

static void ga_record_result(struct ga* g, fxp_t result) {
    if (!g->has_pending || g->done)
        return;

    g->offspring_raw[g->active_index] = result;
    g->evaluated_in_generation++;
    g->evaluations++;
    g->has_pending = 0;
}

fxp_t* ga_get_new_parameters(struct ga* g, fxp_t previous_result) {
    if (!g || g->done)
        return NULL;

    if (g->has_pending)
        ga_record_result(g, previous_result);

    if (g->max_evaluations > 0 && g->evaluations >= g->max_evaluations) {
        ga_finish_generation(g);
        return NULL;
    }

    if (g->evaluated_in_generation >= g->offspring_count) {
        ga_finish_generation(g);

        if (g->done)
            return NULL;
    }

    g->active_index = g->evaluated_in_generation;
    g->has_pending = 1;

    return g->offspring + g->active_index * g->n;
}

int ga_is_done(const struct ga* g) {
    return !g || g->done;
}

int ga_begin_new_argument_set(struct ga* g, const fxp_t* new_start_params) {
    if (!g)
        return -EINVAL;

    if (!g->done)
        return -EBUSY;

    if (g->has_pending)
        return -EBUSY;

    if (new_start_params) {
        memmove(g->start, new_start_params, g->n * sizeof(fxp_t));
    } else if (g->has_best_regularized) {
        memmove(g->start, g->best_regularized, g->n * sizeof(fxp_t));
    }

    g->active_index = 0;
    g->evaluated_in_generation = 0;
    g->has_pending = 0;

    g->elite_valid = 0;

    g->best_regularized_raw = ga_fxp_neg_inf();
    g->best_regularized_fitness = ga_fxp_neg_inf();
    g->best_observed_raw = ga_fxp_neg_inf();
    g->has_best_regularized = 0;
    g->has_best_observed = 0;

    g->generation = 0;
    g->evaluations = 0;
    g->no_improve_generations = 0;

    g->sigma = g->sigma_init;

    g->done = 0;

    ga_generate_initial(g);

    return 0;
}

const fxp_t* ga_best_parameters(const struct ga* g) {
    if (!g)
        return NULL;

    if (!g->has_best_regularized || g->elite_valid == 0)
        return g->start;

    return g->best_regularized;
}

fxp_t ga_best_result(const struct ga* g) {
    if (!g || !g->has_best_regularized)
        return ga_fxp_neg_inf();

    return g->best_regularized_raw;
}

const fxp_t* ga_best_observed_parameters(const struct ga* g) {
    if (!g || !g->has_best_observed)
        return NULL;

    return g->best_observed;
}

fxp_t ga_best_observed_result(const struct ga* g) {
    if (!g || !g->has_best_observed)
        return ga_fxp_neg_inf();

    return g->best_observed_raw;
}

size_t ga_generation(const struct ga* g) {
    return g ? g->generation : 0;
}

size_t ga_evaluations(const struct ga* g) {
    return g ? g->evaluations : 0;
}

void ga_free(struct ga* g) {
    if (!g)
        return;

    kfree(g->pretrained);
    kfree(g->start);
    kfree(g->scale);
    kfree(g->lower);
    kfree(g->upper);

    kfree(g->offspring);
    kfree(g->offspring_raw);
    kfree(g->offspring_penalty);

    kfree(g->elites);
    kfree(g->elite_raw);
    kfree(g->elite_penalty);
    kfree(g->elite_fitness);

    kfree(g->pool);
    kfree(g->pool_raw);
    kfree(g->pool_penalty);
    kfree(g->pool_fitness);
    kfree(g->pool_selected);

    kfree(g->best_regularized);
    kfree(g->best_observed);

    memset(g, 0, sizeof(*g));
}