#include "genetic/ga.h"

#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt

#include "pmu/pmu.h"

#include <linux/cpufreq.h>
#include <linux/cred.h>
#include <linux/err.h>
#include <linux/fcntl.h>
#include <linux/fs.h>
#include <linux/init.h>
#include <linux/jiffies.h>
#include <linux/kernel.h>
#include <linux/kmod.h>
#include <linux/kthread.h>
#include <linux/math64.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/pid.h>
#include <linux/sched/signal.h>
#include <linux/slab.h>
#include <linux/string.h>
#include <linux/wait.h>

#include "perf_mlp_params.h"
#include "power/amd_core_energy.h"
#include "utility/fixed_point.h"
#include "utility/global_variables.h"

#ifndef BENCHMARK_DIR_PATH
#error "BENCHMARK_DIR_PATH must be defined by the Makefile"
#endif

#ifndef BENCHMARK_COUNT
#error "BENCHMARK_COUNT must be defined by the Makefile"
#endif

#ifndef TRAIN_RESULTS_PATH
#define TRAIN_RESULTS_PATH "/tmp/freqmind_train_cycles.csv"
#endif

#ifndef FREQMIND_TUNED_PARAMS_PATH
#define FREQMIND_TUNED_PARAMS_PATH "/tmp/freqmind_tuned_params.h"
#endif

#ifndef FREQMIND_EVAL_SAMPLES_PER_BENCHMARK
#define FREQMIND_EVAL_SAMPLES_PER_BENCHMARK 20
#endif

#define PERIOD_MS 200
#define GOV_NAME "freqmind_train"
#define MEMORY_LATENCY_NS 10ULL
#define FREQMIND_OBJECTIVE_ENERGY_SCALE_UJ 1000000ULL

static DEFINE_MUTEX(freqmind_mlp_lock);
static fxp_t freqmind_pretrained_params[PERF_MLP_PARAM_COUNT];

typedef struct freqmind_worker {
    struct task_struct* task;
    wait_queue_head_t wq;
    bool stop;
} freqmind_worker_t;

enum freqmind_phase {
    FREQMIND_PHASE_TRAIN = 0,
    FREQMIND_PHASE_EVAL = 1,
};

typedef struct freqmind_policy {
    struct cpufreq_policy* policy;

    pmu_counter pmu_metrics_counter;
    bool pmu_created;

    struct ga ga_optimizer;
    struct ga_config ga_config;
    bool ga_initialized;

    fxp_t active_mlp_params[PERF_MLP_PARAM_COUNT];
    fxp_t best_mlp_params[PERF_MLP_PARAM_COUNT];
    bool best_mlp_params_valid;
    fxp_t best_training_metric;
    bool best_training_metric_valid;

    size_t completed_generations;
    freqmind_worker_t worker;
    fxp_t max_objective;

    enum freqmind_phase phase;
    unsigned long cycle_id;
    unsigned long eval_benchmark_id;
    unsigned int eval_samples;
    fxp_t eval_metric_sum;

    unsigned long current_benchmark_id;
    struct pid* current_benchmark_pid;

    long predicted_frequency_khz;
    unsigned long current_frequency_khz;
} freqmind_policy_t;

static void freqmind_copy_params(fxp_t* dst, const fxp_t* src) {
    memcpy(dst, src, sizeof(fxp_t) * PERF_MLP_PARAM_COUNT);
}

static int helper_init_cb(struct subprocess_info* info, struct cred* new) {
    struct pid** pid_storage = info->data;

    if (!pid_storage) {
        return -EINVAL;
    }

    *pid_storage = get_pid(task_pid(current));
    return 0;
}

static int freqmind_run_new_process(freqmind_policy_t* fp, unsigned long benchmark_id) {
    char path[256];
    char* argv[] = {path, NULL};
    static char* envp[] = {
        "HOME=/",
        "PATH=/sbin:/bin:/usr/sbin:/usr/bin",
        NULL,
    };
    struct subprocess_info* sub_info;
    int ret;

    if (!fp) {
        return -EINVAL;
    }

    snprintf(path, sizeof(path), BENCHMARK_DIR_PATH "/test_%lu", benchmark_id);

    put_pid(fp->current_benchmark_pid);
    fp->current_benchmark_pid = NULL;

    sub_info =
        call_usermodehelper_setup(path, argv, envp, GFP_KERNEL, helper_init_cb, NULL, &fp->current_benchmark_pid);
    if (!sub_info) {
        pr_err("failed to setup benchmark helper: %s\n", path);
        return -ENOMEM;
    }

    ret = call_usermodehelper_exec(sub_info, UMH_WAIT_EXEC | UMH_KILLABLE);
    if (ret) {
        pr_err("failed to start benchmark %lu: %d\n", benchmark_id, ret);
        put_pid(fp->current_benchmark_pid);
        fp->current_benchmark_pid = NULL;
        return ret;
    }

    fp->current_benchmark_id = benchmark_id;
    pr_info("started benchmark %lu: %s\n", benchmark_id, path);

    return 0;
}

static void freqmind_kill_current_process(freqmind_policy_t* fp) {
    if (!fp || !fp->current_benchmark_pid) {
        return;
    }

    kill_pid(fp->current_benchmark_pid, SIGKILL, 1);
    put_pid(fp->current_benchmark_pid);
    fp->current_benchmark_pid = NULL;
}

static void freqmind_reset_measurement_baseline(freqmind_policy_t* fp) {
    pmu_events_count metrics_count;
    u64 energy_uj;

    if (!fp || !fp->pmu_created || !fp->policy) {
        return;
    }

    pmu_read_events(&fp->pmu_metrics_counter, &metrics_count);
    amd_core_energy_delta_uj(fp->policy->cpu, &energy_uj);
}

static int freqmind_start_benchmark(freqmind_policy_t* fp, unsigned long benchmark_id) {
    int ret;

    ret = freqmind_run_new_process(fp, benchmark_id);
    if (!ret) {
        freqmind_reset_measurement_baseline(fp);
    }

    return ret;
}

static int freqmind_write_all(struct file* file, loff_t* pos, const char* buf, size_t len) {
    ssize_t written;
    size_t off = 0;

    while (off < len) {
        written = kernel_write(file, buf + off, len - off, pos);
        if (written < 0) {
            return (int)written;
        }
        if (written == 0) {
            return -EIO;
        }
        off += written;
    }

    return 0;
}

static int freqmind_write_cstr(struct file* file, loff_t* pos, const char* buf) {
    return freqmind_write_all(file, pos, buf, strlen(buf));
}

static void freqmind_fxp_to_decimal(fxp_t value, char* buf, size_t len) {
    s64 raw = value._val;
    s64 int_part = raw / FXP48_16_ONE;
    s64 frac_part = raw % FXP48_16_ONE;

    if (frac_part < 0) {
        frac_part = -frac_part;
    }

    frac_part = (frac_part * FXP_DEC_SCALE) / FXP48_16_ONE;
    scnprintf(buf, len, "%lld.%05lld", int_part, frac_part);
}

static int freqmind_prepare_results_file(void) {
    struct file* file;
    loff_t pos = 0;
    int ret;

    file = filp_open(TRAIN_RESULTS_PATH, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (IS_ERR(file)) {
        ret = PTR_ERR(file);
        pr_err("failed to create training results file %s: %d\n", TRAIN_RESULTS_PATH, ret);
        return ret;
    }

    ret = freqmind_write_cstr(file, &pos, "cycle,benchmark_id,samples,pe_no_penalty_raw,pe_no_penalty\n");
    filp_close(file, NULL);

    return ret;
}

static int freqmind_append_cycle_result(unsigned long cycle_id, unsigned long benchmark_id, unsigned int samples,
                                        fxp_t avg_metric) {
    struct file* file;
    loff_t pos;
    char metric_buf[48];
    char line[192];
    int ret;

    file = filp_open(TRAIN_RESULTS_PATH, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (IS_ERR(file)) {
        ret = PTR_ERR(file);
        pr_err("failed to append training result to %s: %d\n", TRAIN_RESULTS_PATH, ret);
        return ret;
    }

    pos = i_size_read(file_inode(file));
    freqmind_fxp_to_decimal(avg_metric, metric_buf, sizeof(metric_buf));
    scnprintf(line, sizeof(line), "%lu,%lu,%u,%lld,%s\n", cycle_id, benchmark_id, samples, (long long)avg_metric._val,
              metric_buf);

    ret = freqmind_write_cstr(file, &pos, line);
    filp_close(file, NULL);

    if (!ret) {
        pr_info("cycle %lu benchmark %lu P/E without penalties: %s\n", cycle_id, benchmark_id, metric_buf);
    }

    return ret;
}

static int freqmind_save_params_header(const fxp_t* params) {
    struct file* file;
    loff_t pos = 0;
    char line[160];
    int ret = 0;
    size_t i;

    if (!params) {
        return -EINVAL;
    }

    file = filp_open(FREQMIND_TUNED_PARAMS_PATH, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (IS_ERR(file)) {
        ret = PTR_ERR(file);
        pr_err("failed to create tuned params header %s: %d\n", FREQMIND_TUNED_PARAMS_PATH, ret);
        return ret;
    }

    ret = freqmind_write_cstr(file, &pos,
                              "#ifndef FREQMIND_TUNED_PARAMS_H\n"
                              "#define FREQMIND_TUNED_PARAMS_H\n\n"
                              "#define FREQMIND_TUNED_PARAM_COUNT PERF_MLP_PARAM_COUNT\n"
                              "#define FREQMIND_TUNED_PARAMS { \\\n");
    if (ret) {
        goto out;
    }

    for (i = 0; i < PERF_MLP_PARAM_COUNT; ++i) {
        scnprintf(line, sizeof(line), "\tFXP48_16_RAW((s64)%lldLL)%s \\\n", (long long)params[i]._val,
                  i + 1 == PERF_MLP_PARAM_COUNT ? "" : ",");
        ret = freqmind_write_cstr(file, &pos, line);
        if (ret) {
            goto out;
        }
    }

    ret = freqmind_write_cstr(file, &pos, "}\n\n#endif\n");

out:
    filp_close(file, NULL);
    if (!ret) {
        pr_info("saved tuned parameters to %s\n", FREQMIND_TUNED_PARAMS_PATH);
    }
    return ret;
}

static void freqmind_log_fxp_metric(const char* metric_name, fxp_t metric) {
    char metric_buf[48];

    freqmind_fxp_to_decimal(metric, metric_buf, sizeof(metric_buf));
    pr_info("%s: %s\n", metric_name, metric_buf);
}

static fxp_t freqmind_calc_objective(const pmu_events_count* metrics_count, u64 lost_cycles, u64 energy_uj) {
    fxp_t performance;
    fxp_t result;

    if (!metrics_count || energy_uj == 0) {
        return FXP_FROM_INT(0);
    }

    if (lost_cycles >= metrics_count->cycles) {
        performance = FXP_FROM_INT(0);
    } else {
        performance = FXP_FROM_INT(metrics_count->cycles - lost_cycles);
    }

    if (!performance._val) {
        performance = FXP_MUL_INT(performance, FREQMIND_OBJECTIVE_ENERGY_SCALE_UJ);
    }

    result = FXP_DIV_INT(performance, energy_uj);
    return result;
}

static fxp_t freqmind_calc_pe_no_penalty(const pmu_events_count* metrics_count, u64 energy_uj) {
    fxp_t performance;

    if (!metrics_count || !metrics_count->cycles || energy_uj == 0) {
        return FXP_FROM_INT(0);
    }

    performance = FXP_FROM_INT(metrics_count->cycles);
    return FXP_DIV_INT(performance, energy_uj);
}

static int freqmind_ga_init_policy(freqmind_policy_t* fp) {
    int ret;

    if (!fp) {
        return -EINVAL;
    }

    if (fp->ga_initialized) {
        ga_free(&fp->ga_optimizer);
        fp->ga_initialized = false;
    }

    perf_mlp_ga_default_config(&fp->ga_config);

    fp->ga_config.start_params = fp->active_mlp_params;
    fp->ga_config.max_generations = 10;
    fp->ga_config.lambda_start = FXP_DIV_INT(FXP48_16_FROM_INT(1), 1000);
    fp->ga_config.sigma_init = FXP48_16_RAW(100000);

    ret = ga_init(&fp->ga_optimizer, fp->active_mlp_params, &fp->ga_config);
    if (ret) {
        pr_err("failed to initialize genetic optimizer: %d\n", ret);
        return ret;
    }

    fp->ga_initialized = true;
    fp->completed_generations = ga_generation(&fp->ga_optimizer);
    fp->max_objective = FXP_FROM_INT(0);

    return 0;
}

static void freqmind_ga_free_policy(freqmind_policy_t* fp) {
    if (!fp || !fp->ga_initialized) {
        return;
    }

    ga_free(&fp->ga_optimizer);
    fp->ga_initialized = false;
}

static void freqmind_select_best_params_from_ga(freqmind_policy_t* fp) {
    const fxp_t* params;
    fxp_t metric;

    if (!fp || !fp->ga_initialized) {
        return;
    }

    params = ga_best_observed_parameters(&fp->ga_optimizer);
    if (params) {
        metric = ga_best_observed_result(&fp->ga_optimizer);
    } else {
        params = ga_best_parameters(&fp->ga_optimizer);
        metric = ga_best_result(&fp->ga_optimizer);
    }

    if (!params) {
        return;
    }

    freqmind_copy_params(fp->active_mlp_params, params);

    if (!fp->best_training_metric_valid || FXP_SUB(metric, fp->best_training_metric)._val > 0) {
        freqmind_copy_params(fp->best_mlp_params, params);
        fp->best_mlp_params_valid = true;
        fp->best_training_metric = metric;
        fp->best_training_metric_valid = true;
    }
}

static int freqmind_start_training_cycle(freqmind_policy_t* fp) {
    int ret;

    if (!fp) {
        return -EINVAL;
    }

    freqmind_kill_current_process(fp);
    fp->phase = FREQMIND_PHASE_TRAIN;
    fp->eval_benchmark_id = 0;
    fp->eval_samples = 0;
    fp->eval_metric_sum = FXP_FROM_INT(0);

    if (fp->best_mlp_params_valid) {
        freqmind_copy_params(fp->active_mlp_params, fp->best_mlp_params);
    }

    ret = freqmind_ga_init_policy(fp);
    if (ret) {
        return ret;
    }

    return freqmind_start_benchmark(fp, 0);
}

static int freqmind_start_cycle_evaluation(freqmind_policy_t* fp) {
    if (!fp) {
        return -EINVAL;
    }

    freqmind_kill_current_process(fp);
    freqmind_ga_free_policy(fp);

    fp->phase = FREQMIND_PHASE_EVAL;
    fp->eval_benchmark_id = 0;
    fp->eval_samples = 0;
    fp->eval_metric_sum = FXP_FROM_INT(0);

    pr_info("cycle %lu finished, starting penalty-free evaluation\n", fp->cycle_id);

    return freqmind_start_benchmark(fp, 0);
}

static int freqmind_finish_eval_benchmark(freqmind_policy_t* fp) {
    fxp_t avg_metric;
    int ret;

    if (!fp || !fp->eval_samples) {
        return -EINVAL;
    }

    avg_metric = FXP_DIV_INT(fp->eval_metric_sum, fp->eval_samples);
    ret = freqmind_append_cycle_result(fp->cycle_id, fp->eval_benchmark_id, fp->eval_samples, avg_metric);
    if (ret) {
        pr_err("failed to write evaluation result: %d\n", ret);
    }

    freqmind_kill_current_process(fp);

    if (fp->eval_benchmark_id + 1 >= BENCHMARK_COUNT) {
        fp->cycle_id++;
        pr_info("completed penalty-free evaluation, next cycle is %lu\n", fp->cycle_id);
        return freqmind_start_training_cycle(fp);
    }

    fp->eval_benchmark_id++;
    fp->eval_samples = 0;
    fp->eval_metric_sum = FXP_FROM_INT(0);

    return freqmind_start_benchmark(fp, fp->eval_benchmark_id);
}

static int freqmind_handle_eval_metric(freqmind_policy_t* fp, fxp_t pe_no_penalty) {
    if (!fp) {
        return -EINVAL;
    }

    fp->eval_metric_sum._val += pe_no_penalty._val;
    fp->eval_samples++;

    if (fp->eval_samples >= FREQMIND_EVAL_SAMPLES_PER_BENCHMARK) {
        return freqmind_finish_eval_benchmark(fp);
    }

    return 0;
}

static int freqmind_update_mlp_params_from_ga(freqmind_policy_t* fp, fxp_t objective) {
    fxp_t* new_params;
    size_t current_generation;
    bool cycle_finished;
    unsigned long next_benchmark_id;
    int ret;

    if (!fp || !fp->ga_initialized) {
        return -EINVAL;
    }

    new_params = ga_get_new_parameters(&fp->ga_optimizer, objective);

    current_generation = ga_generation(&fp->ga_optimizer);
    if (current_generation > fp->completed_generations) {
        fp->completed_generations = current_generation;
        freqmind_log_fxp_metric("generation best metric", ga_best_result(&fp->ga_optimizer));
        fp->max_objective = FXP_FROM_INT(0);
    }

    if (ga_is_done(&fp->ga_optimizer)) {
        cycle_finished = fp->current_benchmark_id + 1 >= BENCHMARK_COUNT;
        freqmind_select_best_params_from_ga(fp);

        if (cycle_finished) {
            freqmind_log_fxp_metric("cycle best training metric", fp->best_training_metric);
            return freqmind_start_cycle_evaluation(fp);
        }

        next_benchmark_id = fp->current_benchmark_id + 1;
        freqmind_kill_current_process(fp);

        freqmind_log_fxp_metric("benchmark best metric", ga_best_result(&fp->ga_optimizer));

        ret = ga_begin_new_argument_set(&fp->ga_optimizer, NULL);
        if (ret) {
            pr_err("failed to begin new GA argument set: %d\n", ret);
            return ret;
        }

        new_params = ga_get_new_parameters(&fp->ga_optimizer, FXP_FROM_INT(0));
        if (!new_params) {
            pr_err("genetic optimizer returned no parameters\n");
            return -EINVAL;
        }

        freqmind_copy_params(fp->active_mlp_params, new_params);
        return freqmind_start_benchmark(fp, next_benchmark_id);
    }

    if (!new_params) {
        pr_err("genetic optimizer returned no parameters\n");
        return -EINVAL;
    }

    freqmind_copy_params(fp->active_mlp_params, new_params);
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
    u64 energy_uj;
    u64 lost_cycles;
    fxp_t objective;
    fxp_t pe_no_penalty;
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

    ret = amd_core_energy_delta_uj(fp->policy->cpu, &energy_uj);
    if (ret) {
        pr_err("failed to read energy counter: %d\n", ret);
        return fp->policy->cur;
    }

    if (!metrics_count.cycles || !energy_uj) {
        return fp->policy->cur;
    }

    fp->current_frequency_khz = fp->policy->cur;

    lost_cycles = metrics_count.cache_misses * MEMORY_LATENCY_NS * fp->current_frequency_khz;
    lost_cycles = div64_u64(lost_cycles, 1000000ULL);

    pe_no_penalty = freqmind_calc_pe_no_penalty(&metrics_count, energy_uj);

    if (fp->phase == FREQMIND_PHASE_EVAL) {
        ret = freqmind_handle_eval_metric(fp, pe_no_penalty);
        if (ret) {
            pr_err("failed to handle evaluation metric: %d\n", ret);
        }
        predicted_frequency_khz = freqmind_predict_freq(fp, &metrics_count);
        return predicted_frequency_khz ? predicted_frequency_khz : fp->policy->cur;
    }

    objective = freqmind_calc_objective(&metrics_count, lost_cycles, energy_uj);

    predicted_frequency_khz = fp->predicted_frequency_khz;
    if (predicted_frequency_khz < (long)fp->policy->min) {
        objective = FXP_SUB(objective, FXP_FROM_INT(fp->policy->min - predicted_frequency_khz));
    } else if (predicted_frequency_khz > (long)fp->policy->max) {
        objective = FXP_SUB(objective, FXP_FROM_INT(predicted_frequency_khz - fp->policy->max));
    }

    if (FXP_SUB(objective, fp->max_objective)._val > 0) {
        fp->max_objective = objective;
    }

    ret = freqmind_update_mlp_params_from_ga(fp, objective);
    if (ret) {
        pr_err("failed to update MLP parameters: %d\n", ret);
    }

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
    fp->worker.task = kthread_run(freqmind_thread, fp, "freqmind_train/%u", fp->policy->cpu);
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
    fp->phase = FREQMIND_PHASE_TRAIN;
    fp->current_benchmark_pid = NULL;
    fp->cycle_id = 0;
    fp->eval_benchmark_id = 0;
    fp->eval_samples = 0;
    fp->eval_metric_sum = FXP_FROM_INT(0);
    fp->best_mlp_params_valid = false;
    fp->best_training_metric_valid = false;
    fp->predicted_frequency_khz = policy->cur;
    fp->current_frequency_khz = policy->cur;

    freqmind_copy_params(fp->active_mlp_params, freqmind_pretrained_params);
    freqmind_copy_params(fp->best_mlp_params, freqmind_pretrained_params);
    fp->best_mlp_params_valid = true;

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
    freqmind_kill_current_process(fp);
    freqmind_ga_free_policy(fp);

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
    u64 energy_uj;

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

    ret = amd_core_energy_delta_uj(policy->cpu, &energy_uj);
    if (ret) {
        pr_err("failed to initialize energy counter for CPU %u: %d\n", policy->cpu, ret);
        pmu_delete_counter(&fp->pmu_metrics_counter);
        fp->pmu_created = false;
        return ret;
    }

    freqmind_prepare_results_file();

    ret = freqmind_start_training_cycle(fp);
    if (ret) {
        pmu_delete_counter(&fp->pmu_metrics_counter);
        fp->pmu_created = false;
        return ret;
    }

    ret = freqmind_worker_start(fp);
    if (ret) {
        freqmind_kill_current_process(fp);
        freqmind_ga_free_policy(fp);
        pmu_delete_counter(&fp->pmu_metrics_counter);
        fp->pmu_created = false;
        return ret;
    }

    pr_info("governor started on CPU %u\n", policy->cpu);
    return 0;
}

static void freqmind_stop(struct cpufreq_policy* policy) {
    freqmind_policy_t* fp;
    const fxp_t* params_to_save;

    if (!policy) {
        return;
    }

    fp = policy->governor_data;
    if (!fp) {
        return;
    }

    freqmind_worker_stop(fp);
    freqmind_kill_current_process(fp);

    if (fp->ga_initialized) {
        freqmind_select_best_params_from_ga(fp);
    }

    params_to_save = fp->best_mlp_params_valid ? fp->best_mlp_params : fp->active_mlp_params;
    freqmind_save_params_header(params_to_save);

    freqmind_ga_free_policy(fp);

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

    perf_mlp_copy_params(freqmind_pretrained_params);

    ret = amd_energy_init_state();
    if (ret) {
        pr_err("failed to initialize AMD energy reader: %d\n", ret);
        return ret;
    }

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
MODULE_DESCRIPTION("Training Linux CPUFreq governor with a genetic algorithm");
