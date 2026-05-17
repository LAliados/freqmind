// SPDX-License-Identifier: GPL-2.0

#include "genetic/ga.h"
#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt
#include "pmu/pmu.h"

#include <linux/cpufreq.h>
#include <linux/init.h>
#include <linux/jiffies.h>
#include <linux/kernel.h>
#include <linux/kthread.h>
#include <linux/math64.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/slab.h>
#include <linux/string.h>
#include <linux/wait.h>
#include "perf_mlp_params.h"
#include "power/amd_core_energy.h"
#include "utility/fixed_point.h"
#include "utility/global_variables.h"


#ifndef BENCHMARK_DIR_PATH
#error Benchmark path is not provided. Use -DBENCHMARK_DIR_PATH=<absolute path to benchmark dir>
#define BENCHMARK_DIR_PATH ""
#endif

#ifndef BENCHMARK_COUNT
#error Benchmark count is not provided. Use -DBENCHMARK_COUNT=<count>. Names must be test_<num from 0 to BENCHMARK_COUNT-1>
#define BENCHMARK_COUNT 120
#endif


#define PERIOD_MS 200
#define GOV_NAME "freqmind_train"

/*
===================================================
================   CONSTANTS   ====================
===================================================
*/
static const unsigned long MEMORY_LATENCY_NS = 10;

/*
 * amd_core_energy_delta_uj() returns energy in microjoules.  Scaling the
 * objective by 1e6 keeps P / E numerically visible in Q48.16 and makes the
 * optimized value equivalent to performance per joule.
 */
#define FREQMIND_OBJECTIVE_ENERGY_SCALE_UJ 1000000ULL

/*
===================================================
=============   GLOBAL VARIABLES   ================
===================================================
*/

static DEFINE_MUTEX(freqmind_mlp_lock);
static fxp_t freqmind_pretrained_params[PERF_MLP_PARAM_COUNT];

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
    bool pmu_created;
    struct ga ga_optimizer;
    struct ga_config ga_config;
    bool ga_initialized;
    fxp_t active_mlp_params[PERF_MLP_PARAM_COUNT];
    size_t completed_generations;
    freqmind_worker_t worker;
    fxp_t max_objective;
    unsigned long current_benchmark_id;
    struct pid* current_benchmark_pid;
    long predicted_frequency_khz;
    unsigned long current_frequency_khz;
} freqmind_policy_t;

/*
===================================================
================   HELPERS   ======================
===================================================
*/

static int helper_init_cb(struct subprocess_info* info, struct cred* new) {
    struct pid* pid;
    freqmind_policy_t* fp = (freqmind_policy_t*)info->data;

    pid = get_task_pid(current, PIDTYPE_PID);
    if (!pid)
        return -ESRCH;

    if (fp->current_benchmark_pid)
        put_pid(fp->current_benchmark_pid);
    fp->current_benchmark_pid = pid;

    return 0;
}

static int freqmind_run_new_process(freqmind_policy_t* fp, unsigned long benchmark_id) {
    pr_info("Starting benchmark %lu", benchmark_id);
    struct subprocess_info* info;
    int ret;
    char path[sizeof(BENCHMARK_DIR_PATH "/test_") + 20];
    char* argv[] = {path, NULL};
    char* envp[] = {"HOME=/", "PATH=/sbin:/bin:/usr/sbin:/usr/bin", NULL};

    scnprintf(path, sizeof(path), BENCHMARK_DIR_PATH "/test_%lu", benchmark_id);

    info = call_usermodehelper_setup(argv[0], argv, envp, GFP_KERNEL, helper_init_cb, NULL, fp);
    if (!info)
        return -ENOMEM;

    ret = call_usermodehelper_exec(info, UMH_WAIT_EXEC | UMH_KILLABLE);
    /*
     * Если exec не удался, PID мог быть уже сохранён в init_cb,
     * но живого userspace helper'а нет. Чистим ссылку.
     */
    if (ret) {
        struct pid* pid = NULL;

        pid = fp->current_benchmark_pid;
        fp->current_benchmark_pid = NULL;

        if (pid)
            put_pid(pid);
    } else {
        fp->current_benchmark_id = benchmark_id;
    }
    return ret;
}
static void freqmind_kill_current_process(freqmind_policy_t* fp) {
    struct pid* pid;

    pid = fp->current_benchmark_pid;
    fp->current_benchmark_pid = NULL;

    if (!pid)
        return;

    kill_pid(pid, SIGKILL, 1);
}

static void freqmind_log_fxp_metric(unsigned int cpu, size_t generation, fxp_t metric) {
    s64 raw = metric._val;
    s64 ipart;
    s64 rem;
    u64 fpart;

    ipart = raw / FXP48_16_ONE;
    rem = raw % FXP48_16_ONE;
    if (rem < 0)
        rem = -rem;

    fpart = ((u64)rem * FXP_DEC_SCALE) / FXP48_16_ONE;

    if (raw < 0 && ipart == 0)
        pr_info("CPU%u: GA generation=%zu metric=-0.%05llu\n", cpu, generation, (unsigned long long)fpart);
    else
        pr_info("CPU%u: GA generation=%zu metric=%lld.%05llu\n", cpu, generation, (long long)ipart,
                (unsigned long long)fpart);
}

static fxp_t freqmind_calc_objective(const pmu_events_count* metrics_count, u64 lost_cycles, u64 energy_uj) {
    fxp_t performance;

    if (!metrics_count || metrics_count->cycles == 0 || energy_uj == 0)
        return FXP_FROM_INT(0);

    if (lost_cycles >= metrics_count->cycles) {
        performance = FXP_FROM_INT(0);
    } else {
        u64 useful_cycles = metrics_count->cycles - lost_cycles;

        performance = FXP_FROM_INT(useful_cycles);
    }


    s64 raw = performance._val;
    s64 ipart;
    s64 rem;
    u64 fpart;
    ipart = raw / FXP48_16_ONE;
    rem = raw % FXP48_16_ONE;
    if (rem < 0)
        rem = -rem;

    fpart = ((u64)rem * FXP_DEC_SCALE) / FXP48_16_ONE;

    pr_info("Performance: %lld.%llu, Energy: %lluuj", ipart, fpart, energy_uj);

    if (performance._val == 0) {
        performance = FXP_ADD_INT(performance, 1);
        performance = FXP_MUL_INT(performance, (s64)FREQMIND_OBJECTIVE_ENERGY_SCALE_UJ);
    }
    return FXP_DIV_INT(performance, (s64)energy_uj);
}

static void freqmind_copy_params(fxp_t* dst, const fxp_t* src) {
    memcpy(dst, src, sizeof(fxp_t) * PERF_MLP_PARAM_COUNT);
}

static void freqmind_ga_free_policy(freqmind_policy_t* fp);

static int freqmind_ga_init_policy(freqmind_policy_t* fp) {
    int ret;

    if (!fp)
        return -EINVAL;

    freqmind_ga_free_policy(fp);

    ga_default_config(&fp->ga_config, PERF_MLP_PARAM_COUNT);
    fp->ga_config.start_params = fp->active_mlp_params;
    fp->ga_config.alloc_flags = GFP_KERNEL;
    fp->ga_config.max_generations = 10;
    fp->ga_config.max_evaluations = 0;
    fp->ga_config.patience_generations = 0;
    fp->ga_config.lambda_pretrained = FXP_DIV_INT(FXP_FROM_INT(1), 1000);
    fp->ga_config.lambda_start = FXP_DIV_INT(FXP_FROM_INT(1), 1000);
    fp->ga_config.lambda_step = FXP_DIV_INT(FXP_FROM_INT(1), 1000);
    fp->ga_config.lambda_radius = FXP_FROM_INT(0);
    fp->ga_config.sigma_init = FXP_FROM_INT(100000);
    ret = ga_init(&fp->ga_optimizer, freqmind_pretrained_params, &fp->ga_config);
    if (ret)
        return ret;

    fp->ga_initialized = true;
    fp->completed_generations = 0;

    return 0;
}

static void freqmind_ga_free_policy(freqmind_policy_t* fp) {
    if (!fp || !fp->ga_initialized)
        return;

    ga_free(&fp->ga_optimizer);
    fp->ga_initialized = false;
}

static int freqmind_update_mlp_params_from_ga(freqmind_policy_t* fp, fxp_t objective) {
    fxp_t* new_params;
    size_t generation_before;
    size_t generation_after;
    int ret;

    if (!fp || !fp->ga_initialized)
        return -EINVAL;
    if (FXP_SUB(objective, fp->max_objective)._val > 0) {
        fp->max_objective = objective;
    }
    generation_before = ga_generation(&fp->ga_optimizer);
    new_params = ga_get_new_parameters(&fp->ga_optimizer, objective);
    generation_after = ga_generation(&fp->ga_optimizer);

    if (generation_after > generation_before) {
        fp->completed_generations += generation_after - generation_before;
        fxp_pr_info("\t\t\t\tGeneration best metric", ga_best_result(&fp->ga_optimizer));
        //freqmind_log_fxp_metric(fp->policy->cpu, fp->completed_generations, fp->max_objective);
        fp->max_objective = FXP_FROM_INT(0);
    }

    if (ga_is_done(&fp->ga_optimizer)) {
        freqmind_kill_current_process(fp);
        if (fp->current_benchmark_id + 1 >= BENCHMARK_COUNT) {
            ret = freqmind_run_new_process(fp, 0);
        } else {
            ret = freqmind_run_new_process(fp, fp->current_benchmark_id + 1);
        }
        if (ret) {
            pr_err("Can't run new benchmark: %d\n", ret);
            return ret;
        }
        if (fp->current_benchmark_id + 1 >= BENCHMARK_COUNT) {
            fxp_pr_info("Cycle best metric", ga_best_result(&fp->ga_optimizer));
            freqmind_copy_params(fp->active_mlp_params, ga_best_parameters(&fp->ga_optimizer));
            ret = freqmind_ga_init_policy(fp);
            if (ret) {
                pr_err("CPU%u: can't initialize GA: %d\n", fp->policy->cpu, ret);
                pmu_delete_counter(&fp->pmu_metrics_counter);
                fp->pmu_created = false;
                return ret;
            }
        } else {
            fxp_pr_info("\t\tPoint best metric", ga_best_result(&fp->ga_optimizer));
            ret = ga_begin_new_argument_set(&fp->ga_optimizer, NULL);
            if (ret) {
                pr_err("CPU%u: can't begin new GA argument set: %d\n", fp->policy->cpu, ret);
                return ret;
            }
        }

        new_params = ga_get_new_parameters(&fp->ga_optimizer, FXP_FROM_INT(0));
    }

    if (!new_params) {
        pr_err("CPU%u: GA didn't return new parameters\n", fp->policy->cpu);
        return -EINVAL;
    }

    freqmind_copy_params(fp->active_mlp_params, new_params);

    return 0;
}

static unsigned long freqmind_predict_freq(freqmind_policy_t* fp, const pmu_events_count* metrics_count,
                                           unsigned int min, unsigned int max) {
    fxp_t output[PERF_MLP_OUTPUT_DIM];
    fxp_t work[PERF_MLP_WORK_LEN];
    long predicted_khz;
    int ret;

    if (!fp || !metrics_count)
        return min;

    mutex_lock(&freqmind_mlp_lock);
    perf_mlp_set_params(fp->active_mlp_params);
    ret =
        perf_mlp_predict((s64)metrics_count->cycles, (s64)metrics_count->instructions, (s64)metrics_count->cache_misses,
                         (s64)metrics_count->stalled_cycles_frontend, (s64)metrics_count->branch_misses, output, work);
    mutex_unlock(&freqmind_mlp_lock);

    if (ret) {
        pr_err("CPU%u: perf_mlp_predict failed: %d\n", fp->policy->cpu, ret);
        return fp->current_frequency_khz;
    }

    predicted_khz = (long)FXP_TO_S64(output[0]);

    pr_info("Predicted frequency: %ld khz", predicted_khz);
    fp->predicted_frequency_khz = predicted_khz;

    if (predicted_khz <= 0)
        predicted_khz = min;


    return clamp_t(unsigned long, (unsigned long)predicted_khz, min, max);
}



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
    u64 energy_uj;
    u64 lost_time_ns;
    u64 lost_cycles;
    fxp_t objective;
    int ret;

    if (!fp || !fp->ga_initialized)
        return policy->cur;

    ret = pmu_read_events(&fp->pmu_metrics_counter, &metrics_count);
    if (ret) {
        pr_err("pmu wasn't read: %d", ret);
        return 0;
    }

    ret = amd_core_energy_delta_uj((int)policy->cpu, &energy_uj);
    if (ret) {
        pr_err("CPU%u: energy wasn't read: %d\n", policy->cpu, ret);
        return 0;
    }

    pr_debug(
        "PMU: cycles=%llu, instructions=%llu, cache_misses=%llu, stalled_frontend=%llu, branch_misses=%llu, "
        "energy_uj=%llu\n",
        metrics_count.cycles, metrics_count.instructions, metrics_count.cache_misses,
        metrics_count.stalled_cycles_frontend, metrics_count.branch_misses, energy_uj);

    if (metrics_count.cycles == 0)
        return policy->cur;

    lost_time_ns = (u64)metrics_count.cache_misses * MEMORY_LATENCY_NS;
    lost_cycles = lost_time_ns * fp->current_frequency_khz / 1000000;

    objective = freqmind_calc_objective(&metrics_count, lost_cycles, energy_uj);
    if (fp->predicted_frequency_khz < policy->min) {
        objective = FXP_SUB(objective, FXP_FROM_INT((policy->min - fp->predicted_frequency_khz)));
    } else if (fp->predicted_frequency_khz > policy->max) {
        objective = FXP_SUB(objective, FXP_FROM_INT((fp->predicted_frequency_khz - policy->max)));
    }

    if (energy_uj != 0) {
        ret = freqmind_update_mlp_params_from_ga(fp, objective);
        if (ret)
            return 0;
    } else {
        pr_debug("CPU%u: skip GA update because energy delta is zero\n", policy->cpu);
    }

    if (min >= max)
        return min;
    ret = pmu_read_events(&fp->pmu_metrics_counter, &metrics_count);
    ret = amd_core_energy_delta_uj((int)policy->cpu, &energy_uj);


    return freqmind_predict_freq(fp, &metrics_count, min, max);
}


static void freqmind_apply(struct cpufreq_policy* policy) {
    unsigned int target = freqmind_choose_freq(policy);
    freqmind_policy_t* fp = policy->governor_data;

    if (target == 0) {
        pr_err(GOV_NAME " governor error. Stopping governor...");
        // TODO: Add driver stopping
        return;
    }

    //pr_info("CPU%u: target=%u kHz min=%u kHz max=%u kHz cur=%u kHz\n", policy->cpu, target, policy->min, policy->max,
    //        policy->cur);

    fp->current_frequency_khz = target;
    __cpufreq_driver_target(policy, target, CPUFREQ_RELATION_C);
}

static int freqmind_worker(void* data) {
    freqmind_policy_t* fp = (freqmind_policy_t*)data;
    while (!kthread_should_stop()) {
        WRITE_ONCE(fp->worker.running, false);
        freqmind_apply(fp->policy);


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
    freqmind_copy_params(fp->active_mlp_params, freqmind_pretrained_params);

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

    if (fp->pmu_created) {
        pmu_delete_counter(&fp->pmu_metrics_counter);
        fp->pmu_created = false;
    }

    freqmind_ga_free_policy(fp);

    policy->governor_data = NULL;
    kfree(fp);
}

static int freqmind_start(struct cpufreq_policy* policy) {
    pr_info("start policy for CPU%u\n", policy->cpu);

    freqmind_policy_t* fp = policy->governor_data;
    u64 energy_uj;
    int ret;

    if (!fp)
        return -EINVAL;

    ret = pmu_create_counter(&fp->pmu_metrics_counter, (int)policy->cpu);
    if (ret) {
        pr_err("can't create pmu counter: %d", ret);
        return ret;
    }
    fp->pmu_created = true;

    ret = amd_core_energy_delta_uj((int)policy->cpu, &energy_uj);
    if (ret) {
        pr_err("CPU%u: can't initialize energy baseline: %d\n", policy->cpu, ret);
        pmu_delete_counter(&fp->pmu_metrics_counter);
        fp->pmu_created = false;
        return ret;
    }
    ret = freqmind_ga_init_policy(fp);
    if (ret) {
        pr_err("CPU%u: can't initialize GA: %d\n", fp->policy->cpu, ret);
        pmu_delete_counter(&fp->pmu_metrics_counter);
        fp->pmu_created = false;
        return ret;
    }
    ret = freqmind_run_new_process(fp, 0);

    if (ret) {
        pr_err("Can't run new benchmark: %d\n", ret);
        pmu_delete_counter(&fp->pmu_metrics_counter);
        fp->pmu_created = false;
        return ret;
    }

    freqmind_copy_params(fp->active_mlp_params, freqmind_pretrained_params);
    ret = freqmind_ga_init_policy(fp);
    if (ret) {
        pr_err("CPU%u: can't initialize GA: %d\n", policy->cpu, ret);
        pmu_delete_counter(&fp->pmu_metrics_counter);
        fp->pmu_created = false;
        freqmind_kill_current_process(fp);
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
            freqmind_ga_free_policy(fp);
            pmu_delete_counter(&fp->pmu_metrics_counter);
            fp->pmu_created = false;
            freqmind_kill_current_process(fp);
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

    if (fp->pmu_created) {
        pmu_delete_counter(&fp->pmu_metrics_counter);
        fp->pmu_created = false;
    }

    freqmind_ga_free_policy(fp);
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

    perf_mlp_copy_params(freqmind_pretrained_params);

    ret = amd_energy_init_state();
    if (ret) {
        pr_err("failed to initialize AMD energy reader: %d\n", ret);
        return ret;
    }

    ret = cpufreq_register_governor(&freqmind_governor);
    if (ret) {
        pr_err("failed to register governor: %d\n", ret);
        amd_energy_free_state();
        return ret;
    }

    pr_info("governor '%s' registered\n", GOV_NAME);
    return 0;
}

static void __exit freqmind_exit(void) {
    cpufreq_unregister_governor(&freqmind_governor);
    amd_energy_free_state();
    pr_info("governor '%s' unregistered\n", GOV_NAME);
}

module_init(freqmind_init);
module_exit(freqmind_exit);

MODULE_AUTHOR("Igor Rasskazchikov");
MODULE_DESCRIPTION("Advanced Linux governor based on PMU events and a little bit ML");
MODULE_LICENSE("GPL");
MODULE_VERSION("0.1");