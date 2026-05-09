// SPDX-License-Identifier: GPL-2.0

#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt
#include "pmu/pmu.h"

#include <linux/cpufreq.h>
#include <linux/init.h>
#include <linux/jiffies.h>
#include <linux/kernel.h>
#include <linux/kthread.h>
#include <linux/math64.h>
#include <linux/module.h>
#include <linux/slab.h>
#include <linux/wait.h>
#include "utility/fixed_point.h"
#include "utility/global_variables.h"


#define PERIOD_MS 1000
#define GOV_NAME "freqmind_train"

/*
===================================================
================   CONSTANTS   ====================
===================================================
*/
static const unsigned long MEMORY_LATENCY_NS = 40;

/*
===================================================
=============   GLOBAL VARIABLES   ================
===================================================
*/


/*
===================================================
=============   LOCAL VARIABLES   =================
===================================================
*/

typedef struct {
    struct task_struct* worker_thread;
    wait_queue_head_t wq;
    bool running;
} freqmind_worker_t;

typedef struct {
    struct cpufreq_policy* policy;
    pmu_counter pmu_metrics_counter;
    freqmind_worker_t worker;
    unsigned long current_frequency_khz;
} freqmind_policy_t;


/*
===================================================
================   CALLBACKS   ====================
===================================================
*/

static unsigned int freqmind_choose_freq(struct cpufreq_policy* policy) {
    unsigned int min = policy->min;
    unsigned int max = policy->max;

    freqmind_policy_t* fp = (freqmind_policy_t*)policy->governor_data;

    pmu_events_count metrics_count;

    int ret = pmu_read_events(&fp->pmu_metrics_counter, &metrics_count);
    if (ret) {
        pr_err("pmu wasn't read: %d", ret);
        return 0;
    }
    pr_info("PMU: cycles=%llu, instructions=%llu, cache_misses=%llu,branch_misses=%llu", metrics_count.cycles,
            metrics_count.instructions, metrics_count.cache_misses, metrics_count.branch_misses);
    if (metrics_count.cycles == 0)
        return policy->cur;

    u64 lost_time_ns;
    u64 lost_cycles;

    lost_time_ns = (u64)metrics_count.cache_misses * MEMORY_LATENCY_NS;
    lost_cycles = lost_time_ns * fp->current_frequency_khz / 1000000;

    pr_info("Lost cycles: %llu", lost_cycles);

    fxp_t ratio = FXP_DIV_INT(FXP_FROM_INT((long)(metrics_count.cycles - lost_cycles)), metrics_count.cycles);


    fxp_pr_info("Ratio: %u.%u", ratio);

    if (lost_cycles >= metrics_count.cycles)
        ratio = FXP_FROM_INT(0);
    if (min >= max)
        return min;

    unsigned long frequency = min + FXP_TO_INT(FXP_INT_MUL((max - min), ratio));
    frequency = clamp_t(unsigned long, frequency, min, max);

    return frequency;
}


static void freqmind_apply(struct cpufreq_policy* policy) {
    unsigned int target = freqmind_choose_freq(policy);
    freqmind_policy_t* fp = policy->governor_data;

    if (target == 0) {
        pr_err(GOV_NAME " governor error. Stopping governor...");
        // TODO: Add driver stopping
        return;
    }

    pr_info("CPU%u: target=%u kHz min=%u kHz max=%u kHz cur=%u kHz\n", policy->cpu, target, policy->min, policy->max,
            policy->cur);

    fp->current_frequency_khz = target;
    __cpufreq_driver_target(policy, target, CPUFREQ_RELATION_C);
}

static int freqmind_worker(void* data) {
    freqmind_policy_t* fp = (freqmind_policy_t*)data;
    while (!kthread_should_stop()) {
        WRITE_ONCE(fp->worker.running, false);
        freqmind_apply(fp->policy);

        /*
         * Ждём PERIOD_MS или просыпаемся раньше,
         * если поток попросили остановиться.
         */
        wait_event_interruptible_timeout(fp->worker.wq, kthread_should_stop() || READ_ONCE(fp->worker.running),
                                         msecs_to_jiffies(PERIOD_MS));
    }

    pr_info("my_module: worker thread stopped\n");
    return 0;
}


static int freqmind_init_policy(struct cpufreq_policy* policy) {
    pr_info("init policy for CPU%u\n", policy->cpu);

    freqmind_policy_t* fp;

    fp = kzalloc(sizeof(*fp), GFP_KERNEL);
    if (!fp)
        return -ENOMEM;

    fp->policy = policy;
    init_waitqueue_head(&fp->worker.wq);

    policy->governor_data = fp;

    return 0;
}

static void freqmind_exit_policy(struct cpufreq_policy* policy) {
    pr_info("exit policy for CPU%u\n", policy->cpu);

    freqmind_policy_t* fp = policy->governor_data;

    if (!fp)
        return;

    if (fp->worker.worker_thread) {
        kthread_stop(fp->worker.worker_thread);
        fp->worker.worker_thread = NULL;
    }

    policy->governor_data = NULL;
    kfree(fp);
}

static int freqmind_start(struct cpufreq_policy* policy) {
    pr_info("start policy for CPU%u\n", policy->cpu);

    freqmind_policy_t* fp = policy->governor_data;

    if (!fp)
        return -EINVAL;

    int ret = pmu_create_counter(&fp->pmu_metrics_counter, (int)policy->cpu);
    if (ret) {
        pr_err("can't create pmu counter: %d", ret);
        return ret;
    }
    fp->current_frequency_khz = policy->cur;
    WRITE_ONCE(fp->worker.running, false);
    if (!fp->worker.worker_thread) {
        fp->worker.worker_thread = kthread_create(freqmind_worker, fp, "freqmind/%u", policy->cpu);
        if (IS_ERR(fp->worker.worker_thread)) {
            ret = PTR_ERR(fp->worker.worker_thread);
            pr_err("can't create worker: %d\n", ret);
            fp->worker.worker_thread = NULL;
            pmu_delete_counter(&fp->pmu_metrics_counter);
            return ret;
        }

        kthread_bind(fp->worker.worker_thread, policy->cpu);
        wake_up_process(fp->worker.worker_thread);
    }

    return 0;
}

static void freqmind_stop(struct cpufreq_policy* policy) {
    freqmind_policy_t* fp = policy->governor_data;

    if (!fp)
        return;

    if (fp->worker.worker_thread) {
        kthread_stop(fp->worker.worker_thread);
        fp->worker.worker_thread = NULL;
    }

    pmu_delete_counter(&fp->pmu_metrics_counter);
    pr_info("stop policy for CPU%u\n", policy->cpu);
}

static void freqmind_limits(struct cpufreq_policy* policy) {
    pr_info("limits changed for CPU%u: min=%u kHz max=%u kHz\n", policy->cpu, policy->min, policy->max);
    freqmind_policy_t* fp = policy->governor_data;
    if (!fp)
        return;
    WRITE_ONCE(fp->worker.running, true);
    wake_up_interruptible(&fp->worker.wq);
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