// SPDX-License-Identifier: GPL-2.0

#include "pmu/pmu.h"
#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt

#include <linux/cpufreq.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include "utility/fixed_point.h"
#include "utility/global_variables.h"


#define GOV_NAME "freqmind"

/*
===================================================
================   CONSTANTS   ====================
===================================================
*/
static const unsigned long MEMORY_LATENCY_NS = 80;

/*
===================================================
=============   GLOBAL VARIABLES   ================
===================================================
*/

unsigned long current_frequency_khz = 1000000;

/*
===================================================
=============   LOCAL VARIABLES   =================
===================================================
*/

static pmu_counter pmu_metrics_counter;



/*
===================================================
================   CALLBACKS   ====================
===================================================
*/

static unsigned int freqmind_choose_freq(struct cpufreq_policy* policy) {
    unsigned int min = policy->min;
    unsigned int max = policy->max;

    pmu_events_count metrics_count;

    int ret = pmu_read_events(&pmu_metrics_counter, &metrics_count);
    if (ret) {
        pr_err("pmu wasn't read: %d", ret);
        return 0;
    }
    unsigned long lost_time_ns = metrics_count.branch_misses * MEMORY_LATENCY_NS;

    unsigned long lost_cycles = (lost_time_ns * (current_frequency_khz * 1000)) / 1000000000;

    fxp_t ratio = FXP_DIV_INT(FXP_FROM_INT((long)(metrics_count.cycles - lost_cycles)), metrics_count.cycles);

    if (min >= max)
        return min;

    unsigned long frequency = min + FXP_TO_INT(FXP_INT_MUL((max - min), ratio));
    return frequency;
}

static void freqmind_apply(struct cpufreq_policy* policy) {
    unsigned int target = freqmind_choose_freq(policy);

    if (target == 0) {
        pr_err(GOV_NAME " governor error. Stopping governor...");
        // TODO: Add driver stopping
        return;
    }

    pr_info("CPU%u: target=%u kHz min=%u kHz max=%u kHz cur=%u kHz\n", policy->cpu, target, policy->min, policy->max,
            policy->cur);

    current_frequency_khz = target;
    __cpufreq_driver_target(policy, target, CPUFREQ_RELATION_C);
}

static int freqmind_init_policy(struct cpufreq_policy* policy) {
    pr_info("init policy for CPU%u\n", policy->cpu);
    return 0;
}

static void freqmind_exit_policy(struct cpufreq_policy* policy) {
    pr_info("exit policy for CPU%u\n", policy->cpu);
}

static int freqmind_start(struct cpufreq_policy* policy) {
    pr_info("start policy for CPU%u\n", policy->cpu);

    int ret = pmu_create_counter(&pmu_metrics_counter);
    if (ret) {
        pr_err("can't create pmu counter: %d", ret);
        return ret;
    }

    current_frequency_khz = (policy->max + policy->min) / 2;
    __cpufreq_driver_target(policy, current_frequency_khz, CPUFREQ_RELATION_C);

    return 0;
}

static void freqmind_stop(struct cpufreq_policy* policy) {
    pmu_delete_counter(&pmu_metrics_counter);
    pr_info("stop policy for CPU%u\n", policy->cpu);
}

static void freqmind_limits(struct cpufreq_policy* policy) {
    pr_info("limits changed for CPU%u: min=%u kHz max=%u kHz\n", policy->cpu, policy->min, policy->max);

    freqmind_apply(policy);
}

static struct cpufreq_governor freqmind_governor = {
    .name = GOV_NAME,
    .owner = THIS_MODULE,
    .init = freqmind_init_policy,
    .exit = freqmind_exit_policy,
    .start = freqmind_start,
    .stop = freqmind_stop,
    .limits = freqmind_limits,
};

static int __init freqmind_init(void) {
    int ret;

    ret = cpufreq_register_governor(&freqmind_governor);
    if (ret) {
        pr_err("failed to register governor: %d\n", ret);
        return ret;
    }

    pr_info("governor '%s' registered\n", GOV_NAME);
    return 0;
}

static void __exit freqmind_exit(void) {
    cpufreq_unregister_governor(&freqmind_governor);
    pr_info("governor '%s' unregistered\n", GOV_NAME);
}

module_init(freqmind_init);
module_exit(freqmind_exit);

MODULE_AUTHOR("Igor Rasskazchikov");
MODULE_DESCRIPTION("Advanced Linux governor based on PMU events and a little bit ML");
MODULE_LICENSE("GPL");
MODULE_VERSION("0.1");