#ifndef GA_H
#define GA_H

#include <linux/gfp.h>
#include <linux/types.h>

#include "utility/fixed_point.h"

/*
 * Sequential black-box genetic optimizer for Linux kernel modules.
 *
 * Numeric type:
 *     fxp_t from fixed_point.h, Q48.16.
 *
 * Optimization direction:
 *     the optimizer maximizes the objective result.
 *
 * To minimize a loss:
 *     pass result = -loss.
 *
 * Synchronization:
 *     this object does not lock internally. If one struct ga is used from
 *     several execution paths, protect it outside with a suitable lock.
 */

typedef struct ga_config {
    size_t n_params;

    size_t offspring_count;
    size_t elite_count;

    size_t max_generations;
    size_t max_evaluations;
    size_t patience_generations;

    const fxp_t* start_params;
    const fxp_t* param_scale;
    const fxp_t* lower_bound;
    const fxp_t* upper_bound;

    gfp_t alloc_flags;

    fxp_t sigma_init;
    fxp_t sigma_min;
    fxp_t sigma_decay;
    fxp_t sigma_grow;
    fxp_t sigma_max;

    fxp_t lambda_pretrained;
    fxp_t lambda_start;
    fxp_t lambda_step;
    fxp_t lambda_radius;
    fxp_t trust_radius;

    fxp_t anchor_pull;

    fxp_t differential_rate;
    fxp_t differential_weight;

    fxp_t min_improvement;

    u64 seed;
} ga_config;

struct ga {
    size_t n;
    size_t offspring_count;
    size_t elite_count;
    size_t max_generations;
    size_t max_evaluations;
    size_t patience_generations;

    fxp_t sigma;
    fxp_t sigma_init;
    fxp_t sigma_min;
    fxp_t sigma_decay;
    fxp_t sigma_grow;
    fxp_t sigma_max;

    fxp_t lambda_pretrained;
    fxp_t lambda_start;
    fxp_t lambda_step;
    fxp_t lambda_radius;
    fxp_t trust_radius;
    fxp_t anchor_pull;
    fxp_t differential_rate;
    fxp_t differential_weight;
    fxp_t min_improvement;

    u64 rng;

    fxp_t* pretrained;
    fxp_t* start;
    fxp_t* scale;
    fxp_t* lower;
    fxp_t* upper;

    fxp_t* offspring;
    fxp_t* offspring_raw;
    fxp_t* offspring_penalty;
    size_t active_index;
    size_t evaluated_in_generation;
    int has_pending;

    fxp_t* elites;
    fxp_t* elite_raw;
    fxp_t* elite_penalty;
    fxp_t* elite_fitness;
    size_t elite_valid;

    fxp_t* pool;
    fxp_t* pool_raw;
    fxp_t* pool_penalty;
    fxp_t* pool_fitness;
    unsigned char* pool_selected;

    fxp_t* best_regularized;
    fxp_t best_regularized_raw;
    fxp_t best_regularized_fitness;
    int has_best_regularized;

    fxp_t* best_observed;
    fxp_t best_observed_raw;
    int has_best_observed;

    size_t generation;
    size_t evaluations;
    size_t no_improve_generations;

    /*
     * done == 1 means the active argument set is done.
     * Call ga_begin_new_argument_set() to continue with another argument set.
     */
    int done;
};

void ga_default_config(ga_config* c, size_t n_params);

int ga_init(struct ga* g, const fxp_t* pretrained_params, const ga_config* cfg);

fxp_t* ga_get_new_parameters(struct ga* g, fxp_t previous_result);

int ga_is_done(const struct ga* g);

int ga_begin_new_argument_set(struct ga* g, const fxp_t* new_start_params);

const fxp_t* ga_best_parameters(const struct ga* g);
fxp_t ga_best_result(const struct ga* g);

const fxp_t* ga_best_observed_parameters(const struct ga* g);
fxp_t ga_best_observed_result(const struct ga* g);

size_t ga_generation(const struct ga* g);
size_t ga_evaluations(const struct ga* g);

void ga_free(struct ga* g);

#endif /* GA_H */