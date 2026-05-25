#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt

#include "pmu/pmu.h"

#include <linux/cpufreq.h>
#include <linux/init.h>
#include <linux/jiffies.h>
#include <linux/kernel.h>
#include <linux/kthread.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/slab.h>
#include <linux/string.h>
#include <linux/wait.h>

#include "perf_mlp_params.h"
#include "utility/fixed_point.h"
#include "utility/global_variables.h"

#if defined(__has_include)
#if __has_include("freqmind_tuned_params.h")
#include "freqmind_tuned_params.h"
#define FREQMIND_HAS_TUNED_PARAMS 1
#endif
#endif

#ifndef FREQMIND_HAS_TUNED_PARAMS
#define FREQMIND_HAS_TUNED_PARAMS 0
#endif

#define PERIOD_MS 200
#define GOV_NAME "freqmind"

static DEFINE_MUTEX(freqmind_mlp_lock);
static fxp_t freqmind_inference_params[PERF_MLP_PARAM_COUNT];

typedef struct freqmind_worker {
    struct task_struct* task;
    wait_queue_head_t wq;
    bool stop;
} freqmind_worker_t;

typedef struct freqmind_policy {
    struct cpufreq_policy* policy;

    pmu_counter pmu_metrics_counter;
    bool pmu_created;

    fxp_t active_mlp_params[PERF_MLP_PARAM_COUNT];
    freqmind_worker_t worker;

    long predicted_frequency_khz;
    unsigned long current_frequency_khz;
} freqmind_policy_t;

static void freqmind_copy_params(fxp_t* dst, const fxp_t* src) {
    memcpy(dst, src, sizeof(fxp_t) * PERF_MLP_PARAM_COUNT);
}

#if FREQMIND_HAS_TUNED_PARAMS
static const fxp_t freqmind_tuned_params[PERF_MLP_PARAM_COUNT] = FREQMIND_TUNED_PARAMS;
#endif

static int freqmind_load_params(void) {
    perf_mlp_copy_params(freqmind_inference_params);

#if FREQMIND_HAS_TUNED_PARAMS
    BUILD_BUG_ON(FREQMIND_TUNED_PARAM_COUNT != PERF_MLP_PARAM_COUNT);
    freqmind_copy_params(freqmind_inference_params, freqmind_tuned_params);
    pr_info("loaded tuned parameters from freqmind_tuned_params.h\n");
#else
    pr_info("freqmind_tuned_params.h was not found, using pretrained parameters\n");
#endif

    return 0;
}

static long freqmind_predict_freq(freqmind_policy_t* fp, const pmu_events_count* metrics_count) {
    fxp_t output[PERF_MLP_OUTPUT_DIM];
    fxp_t work[PERF_MLP_WORK_LEN];
    long predicted_khz;

    if (!fp || !metrics_count || !fp->policy) {
        return 0;
    }

    if (!metrics_count->cycles) {
        return fp->policy->cur;
    }

    mutex_lock(&freqmind_mlp_lock);
    perf_mlp_set_params(fp->active_mlp_params);
    perf_mlp_predict(metrics_count->cycles, metrics_count->instructions, metrics_count->cache_misses,
                     metrics_count->stalled_cycles_frontend, metrics_count->branch_misses, output, work);
    mutex_unlock(&freqmind_mlp_lock);

    predicted_khz = (long)FXP_TO_S64(output[0]);

    if (predicted_khz < (long)fp->policy->min) {
        predicted_khz = fp->policy->min;
    } else if (predicted_khz > (long)fp->policy->max) {
        predicted_khz = fp->policy->max;
    }

    fp->predicted_frequency_khz = predicted_khz;
    return predicted_khz;
}

static unsigned int freqmind_choose_freq(freqmind_policy_t* fp) {
    pmu_events_count metrics_count;
    long predicted_frequency_khz;
    int ret;

    if (!fp || !fp->policy || !fp->pmu_created) {
        return fp && fp->policy ? fp->policy->cur : 0;
    }

    ret = pmu_read_events(&fp->pmu_metrics_counter, &metrics_count);
    if (ret) {
        pr_err("failed to read PMU events: %d\n", ret);
        return fp->policy->cur;
    }

    fp->current_frequency_khz = fp->policy->cur;
    predicted_frequency_khz = freqmind_predict_freq(fp, &metrics_count);

    return predicted_frequency_khz ? predicted_frequency_khz : fp->policy->cur;
}

static void freqmind_apply(freqmind_policy_t* fp) {
    unsigned int target_freq;

    if (!fp || !fp->policy) {
        return;
    }

    target_freq = freqmind_choose_freq(fp);
    if (!target_freq) {
        return;
    }

    __cpufreq_driver_target(fp->policy, target_freq, CPUFREQ_RELATION_L);
}

static int freqmind_thread(void* data) {
    freqmind_policy_t* fp = data;

    while (!kthread_should_stop()) {
        wait_event_interruptible_timeout(fp->worker.wq, fp->worker.stop, msecs_to_jiffies(PERIOD_MS));
        if (fp->worker.stop) {
            break;
        }
        freqmind_apply(fp);
    }

    return 0;
}

static int freqmind_worker_start(freqmind_policy_t* fp) {
    if (!fp) {
        return -EINVAL;
    }

    init_waitqueue_head(&fp->worker.wq);
    fp->worker.stop = false;
    fp->worker.task = kthread_run(freqmind_thread, fp, "freqmind/%u", fp->policy->cpu);
    if (IS_ERR(fp->worker.task)) {
        int ret = PTR_ERR(fp->worker.task);

        fp->worker.task = NULL;
        return ret;
    }

    return 0;
}

static void freqmind_worker_stop(freqmind_policy_t* fp) {
    if (!fp || !fp->worker.task) {
        return;
    }

    fp->worker.stop = true;
    wake_up_interruptible(&fp->worker.wq);
    kthread_stop(fp->worker.task);
    fp->worker.task = NULL;
}

static int freqmind_init_policy(struct cpufreq_policy* policy) {
    freqmind_policy_t* fp;

    if (!policy) {
        return -EINVAL;
    }

    fp = kzalloc(sizeof(*fp), GFP_KERNEL);
    if (!fp) {
        return -ENOMEM;
    }

    fp->policy = policy;
    fp->predicted_frequency_khz = policy->cur;
    fp->current_frequency_khz = policy->cur;
    freqmind_copy_params(fp->active_mlp_params, freqmind_inference_params);

    policy->governor_data = fp;
    return 0;
}

static void freqmind_exit_policy(struct cpufreq_policy* policy) {
    freqmind_policy_t* fp;

    if (!policy) {
        return;
    }

    fp = policy->governor_data;
    if (!fp) {
        return;
    }

    freqmind_worker_stop(fp);

    if (fp->pmu_created) {
        pmu_delete_counter(&fp->pmu_metrics_counter);
        fp->pmu_created = false;
    }

    policy->governor_data = NULL;
    kfree(fp);
}

static int freqmind_start(struct cpufreq_policy* policy) {
    freqmind_policy_t* fp;
    int ret;

    if (!policy) {
        return -EINVAL;
    }

    fp = policy->governor_data;
    if (!fp) {
        return -EINVAL;
    }

    ret = pmu_create_counter(&fp->pmu_metrics_counter, policy->cpu);
    if (ret) {
        pr_err("failed to create PMU counter for CPU %u: %d\n", policy->cpu, ret);
        return ret;
    }
    fp->pmu_created = true;

    ret = freqmind_worker_start(fp);
    if (ret) {
        pmu_delete_counter(&fp->pmu_metrics_counter);
        fp->pmu_created = false;
        return ret;
    }

    pr_info("governor started on CPU %u\n", policy->cpu);
    return 0;
}

static void freqmind_stop(struct cpufreq_policy* policy) {
    freqmind_policy_t* fp;

    if (!policy) {
        return;
    }

    fp = policy->governor_data;
    if (!fp) {
        return;
    }

    freqmind_worker_stop(fp);

    if (fp->pmu_created) {
        pmu_delete_counter(&fp->pmu_metrics_counter);
        fp->pmu_created = false;
    }

    pr_info("governor stopped on CPU %u\n", policy->cpu);
}

static void freqmind_limits(struct cpufreq_policy* policy) {
    freqmind_policy_t* fp;

    if (!policy) {
        return;
    }

    fp = policy->governor_data;
    if (!fp) {
        return;
    }

    if (policy->cur < policy->min || policy->cur > policy->max) {
        __cpufreq_driver_target(policy, policy->cur, CPUFREQ_RELATION_L);
    }
}

static struct cpufreq_governor freqmind_gov = {
    .name = GOV_NAME,
    .owner = THIS_MODULE,
    .init = freqmind_init_policy,
    .exit = freqmind_exit_policy,
    .start = freqmind_start,
    .stop = freqmind_stop,
    .limits = freqmind_limits,
};

static int __init freqmind_module_init(void) {
    int ret;

    freqmind_load_params();

    ret = cpufreq_register_governor(&freqmind_gov);
    if (ret) {
        pr_err("failed to register cpufreq governor: %d\n", ret);
        return ret;
    }

    pr_info("registered governor %s\n", GOV_NAME);
    return 0;
}

static void __exit freqmind_module_exit(void) {
    cpufreq_unregister_governor(&freqmind_gov);
    pr_info("unregistered governor %s\n", GOV_NAME);
}

module_init(freqmind_module_init);
module_exit(freqmind_module_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("LAliados");
MODULE_DESCRIPTION("Inference Linux CPUFreq governor for trained freqmind parameters");
