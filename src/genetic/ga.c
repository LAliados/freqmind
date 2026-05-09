#include "ga.h"

#include <linux/errno.h>
#include <linux/kernel.h>
#include <linux/limits.h>
#include <linux/math64.h>
#include <linux/overflow.h>
#include <linux/random.h>
#include <linux/slab.h>
#include <linux/string.h>
#include <linux/vmalloc.h>

#ifndef S64_MIN
#define S64_MIN ((s64)(-9223372036854775807LL - 1))
#endif

#ifndef S64_MAX
#define S64_MAX ((s64)9223372036854775807LL)
#endif

#define KGA_NEG_INF (S64_MIN / 4)
#define KGA_MAX_NORM_PPM (1000LL * (s64)KGA_PPM)

struct kga_one_case_result {
    s64 best_score;
    u32 evaluations;
};

struct kga_regularized_ctx {
    kga_score_fn raw_score_fn;
    void* raw_user_data;

    const s64* base_params;
    const s64* previous_params;

    const struct kga_config* cfg;
};

void kga_default_config(struct kga_config* cfg) {
    memset(cfg, 0, sizeof(*cfg));

    cfg->population_size = 8;
    cfg->epochs = 4;
    cfg->offspring = 4;
    cfg->tournament_size = 3;

    cfg->mutation_rate_ppm = 0;

    cfg->sigma_ppm = 25000;
    cfg->min_sigma_ppm = 1;
    cfg->max_sigma_ppm = 250000;

    cfg->sigma_decay_ppm = 970000;
    cfg->sigma_growth_ppm = 1050000;

    cfg->blend_alpha_ppm = 200000;
    cfg->elite_gene_rate_ppm = 100000;

    cfg->gradient_child_rate_ppm = 350000;
    cfg->gradient_lr_ppm = 15000;
    cfg->gradient_eps_ppm = 20000;
    cfg->gradient_noise_ppm = 100000;
    cfg->gradient_clip_ppm = 3000000;

    cfg->base_penalty = 1000;
    cfg->step_penalty = 10000;
    cfg->max_step_drift_ppm = 250000;

    cfg->accept_rate_ppm = 200000;

    cfg->seed = 0;

    cfg->lower = NULL;
    cfg->upper = NULL;
    cfg->step = NULL;

    cfg->gfp = GFP_KERNEL;
}

static void* kga_alloc_array(size_t count, size_t size, gfp_t gfp) {
    if (size != 0 && count > SIZE_MAX / size)
        return NULL;

    return kvmalloc(count * size, gfp);
}

static s64 kga_abs64_safe(s64 v) {
    if (v == S64_MIN)
        return S64_MAX;

    return v < 0 ? -v : v;
}

static s64 kga_saturating_add_s64(s64 a, s64 b) {
    s64 out;

    if (check_add_overflow(a, b, &out))
        return b < 0 ? -S64_MAX : S64_MAX;

    return out;
}

static s64 kga_saturating_sub_s64(s64 a, s64 b) {
    s64 out;

    if (check_sub_overflow(a, b, &out))
        return b > 0 ? -S64_MAX : S64_MAX;

    return out;
}

static s64 kga_mul_scaled_s64(s64 value, s64 scaled) {
    s64 product;
    s64 result;

    if (!value || !scaled)
        return 0;

    if (check_mul_overflow(value, scaled, &product)) {
        bool neg = (value < 0) ^ (scaled < 0);

        return neg ? -S64_MAX : S64_MAX;
    }

    result = div64_s64(product, KGA_PPM);

    return result;
}

static s64 kga_mul_ppm(s64 value, u32 ppm) {
    return kga_mul_scaled_s64(value, ppm);
}

static void kga_clamp_param(s64* v, u32 j, const struct kga_config* cfg) {
    if (cfg->lower && *v < cfg->lower[j])
        *v = cfg->lower[j];

    if (cfg->upper && *v > cfg->upper[j])
        *v = cfg->upper[j];
}

static s64 kga_param_step(s64 value, u32 j, const struct kga_config* cfg) {
    s64 s;

    if (cfg->step) {
        s = kga_abs64_safe(cfg->step[j]);
        return s > 0 ? s : 1;
    }

    if (cfg->lower && cfg->upper) {
        s = kga_abs64_safe(cfg->upper[j] - cfg->lower[j]);
        return s > 0 ? s : 1;
    }

    s = kga_abs64_safe(value);

    return s > 0 ? s : 1;
}

static s64 kga_norm_ppm(s64 abs_delta, s64 scale) {
    u64 a;
    u64 product;
    u64 result;

    if (abs_delta <= 0)
        return 0;

    if (scale <= 0)
        return KGA_MAX_NORM_PPM;

    a = (u64)abs_delta;

    if (a > U64_MAX / KGA_PPM)
        return KGA_MAX_NORM_PPM;

    product = a * KGA_PPM;
    result = div64_u64(product, (u64)scale);

    if (result > KGA_MAX_NORM_PPM)
        return KGA_MAX_NORM_PPM;

    return (s64)result;
}

static u32 kga_rng_next(u32* state) {
    u32 x = *state;

    if (!x)
        x = get_random_u32();

    if (!x)
        x = 2463534242U;

    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;

    *state = x;

    return x;
}

static u32 kga_rand_ppm(u32* state) {
    return kga_rng_next(state) % (KGA_PPM + 1);
}

static u32 kga_rand_int(u32* state, u32 n) {
    if (!n)
        return 0;

    return kga_rng_next(state) % n;
}

static s64 kga_rand_normalish_ppm(u32* state) {
    s64 s = 0;

    s += kga_rand_ppm(state);
    s += kga_rand_ppm(state);
    s += kga_rand_ppm(state);
    s += kga_rand_ppm(state);
    s += kga_rand_ppm(state);
    s += kga_rand_ppm(state);

    return s - 3 * (s64)KGA_PPM;
}

static s64 kga_safe_score(kga_score_fn fn, const s64* params, u32 n, void* user_data) {
    s64 s = fn(params, n, user_data);

    if (s <= KGA_NEG_INF)
        return KGA_NEG_INF;

    return s;
}

static s64 kga_regularized_score(const s64* params, u32 n, void* user_data) {
    struct kga_regularized_ctx* ctx = user_data;
    const struct kga_config* cfg = ctx->cfg;
    s64 raw;
    s64 base_drift_ppm = 0;
    s64 step_drift_ppm = 0;
    s64 penalty = 0;
    u32 j;

    raw = ctx->raw_score_fn(params, n, ctx->raw_user_data);
    if (raw <= KGA_NEG_INF)
        return KGA_NEG_INF;

    for (j = 0; j < n; ++j) {
        s64 scale = kga_param_step(params[j], j, cfg);

        if (ctx->base_params) {
            s64 d = kga_abs64_safe(params[j] - ctx->base_params[j]);
            s64 norm_ppm = kga_norm_ppm(d, scale);
            s64 sq = kga_mul_scaled_s64(norm_ppm, norm_ppm);

            base_drift_ppm = kga_saturating_add_s64(base_drift_ppm, sq);
        }

        if (ctx->previous_params) {
            s64 d = kga_abs64_safe(params[j] - ctx->previous_params[j]);
            s64 norm_ppm = kga_norm_ppm(d, scale);
            s64 sq;

            if (cfg->max_step_drift_ppm > 0 && norm_ppm > cfg->max_step_drift_ppm)
                return KGA_NEG_INF;

            sq = kga_mul_scaled_s64(norm_ppm, norm_ppm);

            step_drift_ppm = kga_saturating_add_s64(step_drift_ppm, sq);
        }
    }

    base_drift_ppm = div64_s64(base_drift_ppm, n);
    step_drift_ppm = div64_s64(step_drift_ppm, n);

    if (cfg->base_penalty > 0) {
        penalty = kga_saturating_add_s64(penalty, kga_mul_scaled_s64(cfg->base_penalty, base_drift_ppm));
    }

    if (cfg->step_penalty > 0) {
        penalty = kga_saturating_add_s64(penalty, kga_mul_scaled_s64(cfg->step_penalty, step_drift_ppm));
    }

    if (raw <= KGA_NEG_INF + penalty)
        return KGA_NEG_INF;

    return raw - penalty;
}

static u32 kga_best_index(const s64* scores, u32 count) {
    u32 best = 0;
    u32 i;

    for (i = 1; i < count; ++i) {
        if (scores[i] > scores[best])
            best = i;
    }

    return best;
}

static u32 kga_worst_index(const s64* scores, u32 count) {
    u32 worst = 0;
    u32 i;

    for (i = 1; i < count; ++i) {
        if (scores[i] < scores[worst])
            worst = i;
    }

    return worst;
}

static u32 kga_tournament_select(const s64* scores, u32 population_size, u32 tournament_size, u32* rng) {
    u32 best = kga_rand_int(rng, population_size);
    u32 i;

    for (i = 1; i < tournament_size; ++i) {
        u32 candidate = kga_rand_int(rng, population_size);

        if (scores[candidate] > scores[best])
            best = candidate;
    }

    return best;
}

static void kga_insert_if_better_than_worst(s64* population, s64* scores, u32 population_size, u32 n,
                                            const s64* candidate, s64 candidate_score) {
    u32 worst = kga_worst_index(scores, population_size);

    if (candidate_score > scores[worst]) {
        memcpy(population + (size_t)worst * n, candidate, (size_t)n * sizeof(s64));

        scores[worst] = candidate_score;
    }
}

static void kga_mutate_one_gene(s64* child, u32 j, s64 parent_distance, u32 sigma_ppm, const struct kga_config* cfg,
                                u32* rng) {
    s64 base = kga_param_step(child[j], j, cfg);
    s64 adaptive_step = base;
    s64 noise_ppm;
    s64 amp;
    s64 delta;

    if (parent_distance > 0 && parent_distance < base)
        adaptive_step = (base + parent_distance) / 2;

    noise_ppm = kga_rand_normalish_ppm(rng);

    amp = kga_mul_ppm(adaptive_step, sigma_ppm);
    delta = kga_mul_scaled_s64(amp, noise_ppm);

    child[j] = kga_saturating_add_s64(child[j], delta);

    kga_clamp_param(&child[j], j, cfg);
}

static void kga_make_random_child(s64* child, const s64* parent_a, const s64* parent_b, const s64* best, u32 n,
                                  u32 mutation_rate_ppm, u32 sigma_ppm, const struct kga_config* cfg, u32* rng) {
    bool changed = false;
    u32 j;

    for (j = 0; j < n; ++j) {
        s64 a = parent_a[j];
        s64 b = parent_b[j];

        if (kga_rand_ppm(rng) < cfg->elite_gene_rate_ppm) {
            child[j] = best[j];
        } else {
            s64 lo = a < b ? a : b;
            s64 hi = a > b ? a : b;
            s64 d = hi - lo;

            if (d > 0) {
                s64 extra = kga_mul_ppm(d, cfg->blend_alpha_ppm);
                s64 left = kga_saturating_sub_s64(lo, extra);
                s64 right = kga_saturating_add_s64(hi, extra);
                s64 range = right - left;
                s64 offset = kga_mul_ppm(range, kga_rand_ppm(rng));

                child[j] = kga_saturating_add_s64(left, offset);
            } else {
                child[j] = a;
            }
        }

        kga_clamp_param(&child[j], j, cfg);

        if (kga_rand_ppm(rng) < mutation_rate_ppm) {
            kga_mutate_one_gene(child, j, kga_abs64_safe(a - b), sigma_ppm, cfg, rng);

            changed = true;
        }
    }

    if (!changed) {
        u32 j2 = kga_rand_int(rng, n);

        kga_mutate_one_gene(child, j2, kga_abs64_safe(parent_a[j2] - parent_b[j2]), sigma_ppm, cfg, rng);
    }
}

static void kga_estimate_spsa_direction(s8* direction, s64* plus, s64* minus, const s64* center, u32 n,
                                        kga_score_fn score_fn, void* score_user_data, const struct kga_config* cfg,
                                        u32* rng, s64* score_plus_out, s64* score_minus_out) {
    u32 j;
    s64 score_plus;
    s64 score_minus;

    for (j = 0; j < n; ++j) {
        s64 scale = kga_param_step(center[j], j, cfg);
        s64 delta = kga_mul_ppm(scale, cfg->gradient_eps_ppm);
        s8 sign = kga_rand_ppm(rng) < 500000 ? -1 : 1;

        direction[j] = sign;

        plus[j] = kga_saturating_add_s64(center[j], sign * delta);
        minus[j] = kga_saturating_sub_s64(center[j], sign * delta);

        kga_clamp_param(&plus[j], j, cfg);
        kga_clamp_param(&minus[j], j, cfg);
    }

    score_plus = kga_safe_score(score_fn, plus, n, score_user_data);
    score_minus = kga_safe_score(score_fn, minus, n, score_user_data);

    if (score_minus > score_plus) {
        for (j = 0; j < n; ++j)
            direction[j] = -direction[j];
    }

    *score_plus_out = score_plus;
    *score_minus_out = score_minus;
}

static void kga_make_gradient_child(s64* child, const s64* parent, const s8* direction, u32 n, u32 sigma_ppm,
                                    const struct kga_config* cfg, u32* rng) {
    u32 j;

    for (j = 0; j < n; ++j) {
        s64 scale = kga_param_step(parent[j], j, cfg);
        s64 grad_step = kga_mul_ppm(scale, cfg->gradient_lr_ppm);
        s64 noise_amp;
        s64 noise;
        s64 noise_ppm;

        if (direction[j] < 0)
            grad_step = -grad_step;

        noise_ppm = kga_rand_normalish_ppm(rng);

        noise_amp = kga_mul_ppm(scale, sigma_ppm);
        noise_amp = kga_mul_ppm(noise_amp, cfg->gradient_noise_ppm);
        noise = kga_mul_scaled_s64(noise_amp, noise_ppm);

        child[j] = kga_saturating_add_s64(parent[j], grad_step);
        child[j] = kga_saturating_add_s64(child[j], noise);

        kga_clamp_param(&child[j], j, cfg);
    }
}

static void kga_sanitize_config(struct kga_config* cfg) {
    if (cfg->population_size < 2)
        cfg->population_size = 2;

    if (cfg->epochs == 0)
        cfg->epochs = 1;

    if (cfg->offspring == 0)
        cfg->offspring = 1;

    if (cfg->tournament_size == 0)
        cfg->tournament_size = 2;

    if (cfg->tournament_size > cfg->population_size)
        cfg->tournament_size = cfg->population_size;

    if (cfg->mutation_rate_ppm > KGA_PPM)
        cfg->mutation_rate_ppm = KGA_PPM;

    if (cfg->sigma_ppm == 0)
        cfg->sigma_ppm = 1;

    if (cfg->min_sigma_ppm == 0)
        cfg->min_sigma_ppm = 1;

    if (cfg->max_sigma_ppm < cfg->min_sigma_ppm)
        cfg->max_sigma_ppm = cfg->min_sigma_ppm;

    if (cfg->sigma_decay_ppm == 0 || cfg->sigma_decay_ppm > KGA_PPM)
        cfg->sigma_decay_ppm = KGA_PPM;

    if (cfg->sigma_growth_ppm < KGA_PPM)
        cfg->sigma_growth_ppm = KGA_PPM;

    if (cfg->blend_alpha_ppm > KGA_PPM)
        cfg->blend_alpha_ppm = KGA_PPM;

    if (cfg->elite_gene_rate_ppm > KGA_PPM)
        cfg->elite_gene_rate_ppm = KGA_PPM;

    if (cfg->gradient_child_rate_ppm > KGA_PPM)
        cfg->gradient_child_rate_ppm = KGA_PPM;

    if (cfg->gradient_eps_ppm == 0)
        cfg->gradient_eps_ppm = 1;

    if (cfg->gradient_noise_ppm > KGA_PPM)
        cfg->gradient_noise_ppm = KGA_PPM;

    if (cfg->accept_rate_ppm == 0 || cfg->accept_rate_ppm > KGA_PPM)
        cfg->accept_rate_ppm = KGA_PPM;

    if (!cfg->gfp)
        cfg->gfp = GFP_KERNEL;
}

static struct kga_one_case_result kga_optimize_current_case_small_step(s64* params, u32 n, kga_score_fn score_fn,
                                                                       void* score_user_data,
                                                                       const struct kga_config* cfg, u32* rng) {
    struct kga_one_case_result result;
    const size_t row_bytes = (size_t)n * sizeof(s64);
    const size_t pop_bytes = (size_t)cfg->population_size * row_bytes;
    s64* population = NULL;
    s64* scores = NULL;
    s64* child = NULL;
    s64* plus = NULL;
    s64* minus = NULL;
    s8* direction = NULL;
    u32 mutation_rate_ppm;
    u32 sigma_ppm;
    u32 i;
    u32 epoch;

    result.best_score = KGA_NEG_INF;
    result.evaluations = 0;

    population = kga_alloc_array(1, pop_bytes, cfg->gfp);
    scores = kga_alloc_array(cfg->population_size, sizeof(s64), cfg->gfp);
    child = kga_alloc_array(n, sizeof(s64), cfg->gfp);
    plus = kga_alloc_array(n, sizeof(s64), cfg->gfp);
    minus = kga_alloc_array(n, sizeof(s64), cfg->gfp);
    direction = kga_alloc_array(n, sizeof(s8), cfg->gfp);

    if (!population || !scores || !child || !plus || !minus || !direction)
        goto out;

    mutation_rate_ppm = cfg->mutation_rate_ppm;

    if (mutation_rate_ppm == 0) {
        mutation_rate_ppm = div64_s64(3LL * KGA_PPM, n);

        if (mutation_rate_ppm > 300000)
            mutation_rate_ppm = 300000;
    }

    if (mutation_rate_ppm > KGA_PPM)
        mutation_rate_ppm = KGA_PPM;

    sigma_ppm = cfg->sigma_ppm;

    memcpy(population, params, row_bytes);

    scores[0] = kga_safe_score(score_fn, population, n, score_user_data);

    result.evaluations++;

    for (i = 1; i < cfg->population_size; ++i) {
        s64* ind = population + (size_t)i * n;
        bool changed = false;
        u32 j;

        memcpy(ind, params, row_bytes);

        for (j = 0; j < n; ++j) {
            if (kga_rand_ppm(rng) < mutation_rate_ppm) {
                kga_mutate_one_gene(ind, j, 0, sigma_ppm, cfg, rng);

                changed = true;
            }
        }

        if (!changed) {
            u32 j2 = kga_rand_int(rng, n);

            kga_mutate_one_gene(ind, j2, 0, sigma_ppm, cfg, rng);
        }

        scores[i] = kga_safe_score(score_fn, ind, n, score_user_data);

        result.evaluations++;
    }

    i = kga_best_index(scores, cfg->population_size);
    result.best_score = scores[i];

    for (epoch = 0; epoch < cfg->epochs; ++epoch) {
        bool improved = false;
        u32 best_idx;
        u32 k;

        best_idx = kga_best_index(scores, cfg->population_size);

        if (cfg->gradient_child_rate_ppm > 0 && cfg->gradient_lr_ppm > 0) {
            const s64* current_best;
            s64 score_plus;
            s64 score_minus;

            current_best = population + (size_t)best_idx * n;

            kga_estimate_spsa_direction(direction, plus, minus, current_best, n, score_fn, score_user_data, cfg, rng,
                                        &score_plus, &score_minus);

            result.evaluations += 2;

            kga_insert_if_better_than_worst(population, scores, cfg->population_size, n, plus, score_plus);

            kga_insert_if_better_than_worst(population, scores, cfg->population_size, n, minus, score_minus);

            if (score_plus > result.best_score) {
                result.best_score = score_plus;
                improved = true;
            }

            if (score_minus > result.best_score) {
                result.best_score = score_minus;
                improved = true;
            }
        }

        for (k = 0; k < cfg->offspring; ++k) {
            u32 best_idx2 = kga_best_index(scores, cfg->population_size);
            const s64* best = population + (size_t)best_idx2 * n;
            s64 child_score;
            u32 worst_idx;
            bool use_gradient;

            use_gradient = cfg->gradient_child_rate_ppm > 0 && cfg->gradient_lr_ppm > 0 &&
                           kga_rand_ppm(rng) < cfg->gradient_child_rate_ppm;

            if (use_gradient) {
                u32 parent_idx;
                const s64* parent;

                parent_idx = kga_tournament_select(scores, cfg->population_size, cfg->tournament_size, rng);

                parent = population + (size_t)parent_idx * n;

                kga_make_gradient_child(child, parent, direction, n, sigma_ppm, cfg, rng);
            } else {
                u32 parent_a_idx;
                u32 parent_b_idx;
                const s64* parent_a;
                const s64* parent_b;

                parent_a_idx = kga_tournament_select(scores, cfg->population_size, cfg->tournament_size, rng);

                parent_b_idx = kga_tournament_select(scores, cfg->population_size, cfg->tournament_size, rng);

                if (parent_b_idx == parent_a_idx && cfg->population_size > 1)
                    parent_b_idx = kga_rand_int(rng, cfg->population_size);

                parent_a = population + (size_t)parent_a_idx * n;
                parent_b = population + (size_t)parent_b_idx * n;

                kga_make_random_child(child, parent_a, parent_b, best, n, mutation_rate_ppm, sigma_ppm, cfg, rng);
            }

            child_score = kga_safe_score(score_fn, child, n, score_user_data);

            result.evaluations++;

            worst_idx = kga_worst_index(scores, cfg->population_size);

            if (child_score > scores[worst_idx]) {
                memcpy(population + (size_t)worst_idx * n, child, row_bytes);

                scores[worst_idx] = child_score;
                improved = true;

                if (child_score > result.best_score)
                    result.best_score = child_score;
            }
        }

        if (improved) {
            sigma_ppm = kga_mul_ppm(sigma_ppm, cfg->sigma_decay_ppm);

            if (sigma_ppm < cfg->min_sigma_ppm)
                sigma_ppm = cfg->min_sigma_ppm;
        } else {
            sigma_ppm = kga_mul_ppm(sigma_ppm, cfg->sigma_growth_ppm);

            if (sigma_ppm > cfg->max_sigma_ppm)
                sigma_ppm = cfg->max_sigma_ppm;
        }
    }

    i = kga_best_index(scores, cfg->population_size);
    result.best_score = scores[i];

    memcpy(params, population + (size_t)i * n, row_bytes);

out:
    kvfree(population);
    kvfree(scores);
    kvfree(child);
    kvfree(plus);
    kvfree(minus);
    kvfree(direction);

    return result;
}

static void kga_smooth_accept_params(s64* params, const s64* previous_params, u32 n, const struct kga_config* cfg) {
    u32 j;

    for (j = 0; j < n; ++j) {
        s64 diff = params[j] - previous_params[j];

        params[j] = previous_params[j] + kga_mul_ppm(diff, cfg->accept_rate_ppm);

        kga_clamp_param(&params[j], j, cfg);
    }
}

int kga_state_init(struct kga_state* state, const s64* initial_params, u32 n, const struct kga_config* user_cfg) {
    struct kga_config cfg;
    size_t bytes;

    if (!state || !initial_params || !n)
        return -EINVAL;

    if (user_cfg)
        cfg = *user_cfg;
    else
        kga_default_config(&cfg);

    kga_sanitize_config(&cfg);

    memset(state, 0, sizeof(*state));

    bytes = (size_t)n * sizeof(s64);

    state->base_params = kga_alloc_array(n, sizeof(s64), cfg.gfp);
    state->previous_params = kga_alloc_array(n, sizeof(s64), cfg.gfp);

    if (!state->base_params || !state->previous_params) {
        kga_state_free(state);
        return -ENOMEM;
    }

    memcpy(state->base_params, initial_params, bytes);
    memcpy(state->previous_params, initial_params, bytes);

    state->n = n;
    state->gfp = cfg.gfp;
    state->rng = cfg.seed ? cfg.seed : get_random_u32();

    if (!state->rng)
        state->rng = 2463534242U;

    state->total_evaluations = 0;
    state->train_calls = 0;

    return 0;
}

int kga_state_reset_base(struct kga_state* state, const s64* params, u32 n) {
    if (!state || !params || !state->base_params || state->n != n)
        return -EINVAL;

    memcpy(state->base_params, params, (size_t)n * sizeof(s64));

    return 0;
}

void kga_state_free(struct kga_state* state) {
    if (!state)
        return;

    kvfree(state->base_params);
    kvfree(state->previous_params);

    memset(state, 0, sizeof(*state));
}

struct kga_result kga_train(struct kga_state* state, s64* params, u32 n, kga_score_fn raw_score_fn, void* user_data,
                            const struct kga_config* user_cfg) {
    struct kga_result result;
    struct kga_config cfg;
    struct kga_regularized_ctx reg_ctx;
    struct kga_one_case_result step_result;
    size_t bytes;

    result.best_regularized_score = KGA_NEG_INF;
    result.evaluations = 0;
    result.total_evaluations = state ? state->total_evaluations : 0;
    result.train_calls = state ? state->train_calls : 0;

    if (!state || !params || !n || !raw_score_fn)
        return result;

    if (!state->base_params || !state->previous_params)
        return result;

    if (state->n != n)
        return result;

    if (user_cfg)
        cfg = *user_cfg;
    else
        kga_default_config(&cfg);

    kga_sanitize_config(&cfg);

    bytes = (size_t)n * sizeof(s64);

    memcpy(state->previous_params, params, bytes);

    reg_ctx.raw_score_fn = raw_score_fn;
    reg_ctx.raw_user_data = user_data;
    reg_ctx.base_params = state->base_params;
    reg_ctx.previous_params = state->previous_params;
    reg_ctx.cfg = &cfg;

    step_result = kga_optimize_current_case_small_step(params, n, kga_regularized_score, &reg_ctx, &cfg, &state->rng);

    kga_smooth_accept_params(params, state->previous_params, n, &cfg);

    state->total_evaluations += step_result.evaluations;
    state->train_calls++;

    result.best_regularized_score = step_result.best_score;
    result.evaluations = step_result.evaluations;
    result.total_evaluations = state->total_evaluations;
    result.train_calls = state->train_calls;

    return result;
}