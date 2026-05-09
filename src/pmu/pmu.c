#include "pmu.h"

#include <linux/cpu.h>
#include <linux/cpumask.h>
#include <linux/err.h>
#include <linux/errno.h>
#include <linux/string.h>

static void pmu_init_attr(struct perf_event_attr* attr, u32 type, u64 config) {
    memset(attr, 0, sizeof(*attr));

    attr->type = type;
    attr->size = sizeof(*attr);
    attr->config = config;

    /*
     * Create counters disabled first. pmu_create_counter() enables all events
     * only after every event has been successfully created.
     */
    attr->disabled = 1;

    /*
     * pinned=1 avoids multiplexing. If the PMU cannot schedule all pinned
     * events at the same time, the event may fail to run.
     */
    attr->pinned = 1;

    /*
     * Per-CPU mode counts activity on the selected CPU regardless of task.
     *
     * Set exclude_user = 0 to count both user and kernel execution on that CPU.
     * If you only want kernel-mode activity on the selected CPU, change this
     * field to 1.
     */
    attr->exclude_user = 0;
    attr->exclude_kernel = 0;
    attr->exclude_hv = 1;

    /*
     * Pure counting mode. No sampling or overflow callback is needed.
     */
    attr->sample_period = 0;
}

static int pmu_create_one(struct perf_event** event, int cpu, u32 type, u64 config) {
    struct perf_event_attr attr;
    struct perf_event* created;

    pmu_init_attr(&attr, type, config);

    /*
     * cpu = cpu, task = NULL:
     * create a per-CPU counter bound to the selected CPU.
     */
    created = perf_event_create_kernel_counter(&attr, cpu, NULL, NULL, NULL);
    if (IS_ERR(created)) {
        *event = NULL;
        return PTR_ERR(created);
    }

    *event = created;
    return 0;
}

static u64 pmu_read_raw(struct perf_event* event) {
    u64 enabled;
    u64 running;

    if (!event)
        return 0;

    return perf_event_read_value(event, &enabled, &running);
}

static u64 pmu_read_delta(pmu_counter* counter, enum pmu_event_id id) {
    u64 current_value;
    u64 delta;

    if (!counter->_events[id])
        return 0;

    current_value = pmu_read_raw(counter->_events[id]);

    /*
     * perf_event_read_value() returns a virtual cumulative 64-bit value.
     * Unsigned subtraction intentionally handles u64 wraparound as long as the
     * interval between reads is shorter than one full 2^64 counter period.
     */
    delta = current_value - counter->_prev_values[id];
    counter->_prev_values[id] = current_value;

    return delta;
}

static void pmu_enable_all(pmu_counter* counter) {
#define PMU_ENABLE_EVENT(_id, _field, _type, _config)             \
    do {                                                          \
        if (counter->_events[PMU_EVENT_##_id])                    \
            perf_event_enable(counter->_events[PMU_EVENT_##_id]); \
    } while (0);

    PMU_EVENT_LIST(PMU_ENABLE_EVENT)

#undef PMU_ENABLE_EVENT
}

int pmu_create_counter(pmu_counter* counter, int cpu) {
    int ret;

    if (!counter)
        return -EINVAL;

    if (cpu < 0 || cpu >= nr_cpu_ids)
        return -EINVAL;

    if (!cpu_online(cpu))
        return -ENODEV;

    memset(counter, 0, sizeof(*counter));
    mutex_init(&counter->_lock);
    counter->_cpu = cpu;

#define PMU_CREATE_EVENT(_id, _field, _type, _config)                                      \
    do {                                                                                   \
        ret = pmu_create_one(&counter->_events[PMU_EVENT_##_id], cpu, (_type), (_config)); \
        if (ret)                                                                           \
            goto err;                                                                      \
    } while (0);

    PMU_EVENT_LIST(PMU_CREATE_EVENT)

#undef PMU_CREATE_EVENT

    counter->_created = true;

    pmu_enable_all(counter);

    /*
     * Initialize baseline after all events have been enabled. The first
     * pmu_read_events() call will return deltas since this point.
     */
    ret = pmu_reset_counter(counter);
    if (ret)
        goto err;

    return 0;

err:
    pmu_delete_counter(counter);
    return ret;
}

int pmu_delete_counter(pmu_counter* counter) {
    bool locked;

    if (!counter)
        return -EINVAL;

    locked = counter->_created;
    if (locked)
        mutex_lock(&counter->_lock);

#define PMU_DELETE_EVENT(_id, _field, _type, _config)                   \
    do {                                                                \
        struct perf_event** event = &counter->_events[PMU_EVENT_##_id]; \
                                                                        \
        if (*event) {                                                   \
            perf_event_release_kernel(*event);                          \
            *event = NULL;                                              \
        }                                                               \
    } while (0);

    PMU_EVENT_LIST(PMU_DELETE_EVENT)

#undef PMU_DELETE_EVENT

    memset(counter->_prev_values, 0, sizeof(counter->_prev_values));
    counter->_cpu = -1;
    counter->_created = false;

    if (locked)
        mutex_unlock(&counter->_lock);

    return 0;
}

int pmu_read_events(pmu_counter* counter, pmu_events_count* events_count) {
    if (!counter || !events_count)
        return -EINVAL;

    if (!counter->_created)
        return -EINVAL;

    mutex_lock(&counter->_lock);

    memset(events_count, 0, sizeof(*events_count));

#define PMU_READ_EVENT(_id, _field, _type, _config)                      \
    do {                                                                 \
        events_count->_field = pmu_read_delta(counter, PMU_EVENT_##_id); \
    } while (0);

    PMU_EVENT_LIST(PMU_READ_EVENT)

#undef PMU_READ_EVENT

    mutex_unlock(&counter->_lock);

    return 0;
}

int pmu_reset_counter(pmu_counter* counter) {
    if (!counter)
        return -EINVAL;

    if (!counter->_created)
        return -EINVAL;

    mutex_lock(&counter->_lock);

#define PMU_RESET_EVENT(_id, _field, _type, _config)                                              \
    do {                                                                                          \
        counter->_prev_values[PMU_EVENT_##_id] = pmu_read_raw(counter->_events[PMU_EVENT_##_id]); \
    } while (0);

    PMU_EVENT_LIST(PMU_RESET_EVENT)

#undef PMU_RESET_EVENT

    mutex_unlock(&counter->_lock);

    return 0;
}