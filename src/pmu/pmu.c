#include "pmu.h"

#include <linux/err.h>
#include <linux/errno.h>
#include <linux/sched.h>
#include <linux/string.h>

static void pmu_init_attr(struct perf_event_attr* attr, u32 type, u64 config) {
    memset(attr, 0, sizeof(*attr));

    attr->type = type;
    attr->size = sizeof(*attr);
    attr->config = config;

    /*
     * Count immediately after creation.
     *
     * pinned=1 makes measurements more predictable because the kernel should
     * not multiplex this event with other events. If the PMU cannot schedule
     * all pinned events, creation may fail or the event may not run.
     */
    attr->disabled = 0;
    attr->pinned = 1;

    /*
     * This helper is intended to measure kernel-side execution.
     * User-space execution of the same task is ignored.
     */
    attr->exclude_user = 1;
    attr->exclude_kernel = 0;
    attr->exclude_hv = 1;

    /*
     * Pure counting mode. No sampling or overflow callback is needed.
     */
    attr->sample_period = 0;
}

static int pmu_create_one(struct perf_event** event, u32 type, u64 config) {
    struct perf_event_attr attr;
    struct perf_event* created;

    pmu_init_attr(&attr, type, config);

    /*
     * cpu = -1, task = current:
     * create a task-bound counter that follows the current task across CPUs.
     */
    created = perf_event_create_kernel_counter(&attr, -1, current, NULL, NULL);
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
     * Unsigned subtraction is intentional. If a counter ever wraps, the delta
     * is still correct modulo 2^64 for intervals shorter than one full wrap.
     */
    delta = current_value - counter->_prev_values[id];
    counter->_prev_values[id] = current_value;

    return delta;
}

int pmu_create_counter(pmu_counter* counter) {
    int ret;

    if (!counter)
        return -EINVAL;

    memset(counter, 0, sizeof(*counter));
    mutex_init(&counter->_lock);

#define PMU_CREATE_EVENT(_id, _field, _type, _config)                                 \
    do {                                                                              \
        ret = pmu_create_one(&counter->_events[PMU_EVENT_##_id], (_type), (_config)); \
        if (ret)                                                                      \
            goto err;                                                                 \
    } while (0);

    PMU_EVENT_LIST(PMU_CREATE_EVENT)

#undef PMU_CREATE_EVENT

    counter->_created = true;

    /*
     * Initialize the baseline. The first pmu_read_events() call will return
     * deltas since this point, not since each individual event creation.
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