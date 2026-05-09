#ifndef KGA_H
#define KGA_H

#include <linux/gfp.h>
#include <linux/types.h>

#define KGA_PPM 1000000U

typedef s64 (*kga_score_fn)(const s64* params, u32 n, void* user_data);

struct kga_config {
    u32 population_size;
    u32 epochs;
    u32 offspring;
    u32 tournament_size;

    /*
	 * Probability in PPM.
	 * 0 means auto: roughly 3 genes per candidate.
	 */
    u32 mutation_rate_ppm;

    /*
	 * Mutation strength in PPM.
	 *
	 * Example:
	 *   25000 = 0.025
	 */
    u32 sigma_ppm;
    u32 min_sigma_ppm;
    u32 max_sigma_ppm;

    /*
	 * Multipliers in PPM.
	 *
	 * Example:
	 *   970000  = *0.97
	 *   1050000 = *1.05
	 */
    u32 sigma_decay_ppm;
    u32 sigma_growth_ppm;

    /*
	 * BLX-alpha crossover.
	 *
	 * Example:
	 *   200000 = 0.20
	 */
    u32 blend_alpha_ppm;

    /*
	 * Probability that a child gene is copied from the best individual.
	 */
    u32 elite_gene_rate_ppm;

    /*
	 * Part of children that use SPSA-like direction.
	 */
    u32 gradient_child_rate_ppm;
    u32 gradient_lr_ppm;
    u32 gradient_eps_ppm;
    u32 gradient_noise_ppm;
    u32 gradient_clip_ppm;

    /*
	 * Regularization.
	 *
	 * Penalty values are in score units.
	 */
    s64 base_penalty;
    s64 step_penalty;

    /*
	 * Hard limit for one parameter movement per kga_train() call.
	 *
	 * Example:
	 *   250000 = 0.25 * step[i]
	 */
    u32 max_step_drift_ppm;

    /*
	 * Smooth accept:
	 *
	 *   params = previous + accept_rate * (optimized - previous)
	 */
    u32 accept_rate_ppm;

    u32 seed;

    const s64* lower;
    const s64* upper;
    const s64* step;

    gfp_t gfp;
};

struct kga_state {
    s64* base_params;
    s64* previous_params;

    u32 n;
    u32 rng;
    gfp_t gfp;

    u64 total_evaluations;
    u32 train_calls;
};

struct kga_result {
    s64 best_regularized_score;

    u32 evaluations;
    u64 total_evaluations;

    u32 train_calls;
};

void kga_default_config(struct kga_config* cfg);

int kga_state_init(struct kga_state* state, const s64* initial_params, u32 n, const struct kga_config* cfg);

int kga_state_reset_base(struct kga_state* state, const s64* params, u32 n);

void kga_state_free(struct kga_state* state);

/*
 * Call this after external arguments have already been switched.
 *
 * params:
 *   input  - current trained params
 *   output - updated params
 *
 * raw_score_fn:
 *   must evaluate only currently active external arguments.
 *
 * Bigger score is better.
 * If you minimize loss, return -loss.
 */
struct kga_result kga_train(struct kga_state* state, s64* params, u32 n, kga_score_fn raw_score_fn, void* user_data,
                            const struct kga_config* cfg);

#endif /* KGA_H */