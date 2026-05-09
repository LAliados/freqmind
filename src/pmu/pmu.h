#pragma once

#include <linux/mutex.h>
#include <linux/perf_event.h>
#include <linux/types.h>

/*
 * PMU_EVENT_LIST is the single source of truth for all collected events.
 *
 * Arguments:
 *   _id     - internal enum suffix.
 *   _field  - field name in pmu_events_count.
 *   _type   - perf event type, for example PERF_TYPE_HARDWARE,
 *             PERF_TYPE_SOFTWARE, PERF_TYPE_HW_CACHE, PERF_TYPE_RAW.
 *   _config - event-specific config value.
 */
#define PMU_EVENT_LIST(_X)                                                         \
    _X(CYCLES, cycles, PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES)               \
    _X(INSTRUCTIONS, instructions, PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS) \
    _X(CACHE_MISSES, cache_misses, PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES) \
    _X(BRANCH_MISSES, branch_misses, PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES)

/**
 * struct pmu_events_count - Delta snapshot of all events collected by pmu_counter.
 *
 * The values are deltas between the previous successful pmu_read_events() call
 * and the current one. For the first read after pmu_create_counter(), values
 * are deltas since pmu_create_counter() initialized the baseline.
 */
typedef struct {
#define PMU_DECLARE_COUNT_FIELD(_id, _field, _type, _config) u64 _field;
    PMU_EVENT_LIST(PMU_DECLARE_COUNT_FIELD)
#undef PMU_DECLARE_COUNT_FIELD
} pmu_events_count;

enum pmu_event_id {
#define PMU_DECLARE_EVENT_ID(_id, _field, _type, _config) PMU_EVENT_##_id,
    PMU_EVENT_LIST(PMU_DECLARE_EVENT_ID)
#undef PMU_DECLARE_EVENT_ID

        PMU_EVENT_COUNT,
};

/**
 * struct pmu_counter - Private per-CPU PMU counter set.
 *
 * All fields are private implementation details. They intentionally start with
 * an underscore to make it clear that users of this module must not access or
 * modify them directly.
 */
typedef struct {
    struct perf_event* _events[PMU_EVENT_COUNT];
    u64 _prev_values[PMU_EVENT_COUNT];
    int _cpu;
    struct mutex _lock;
    bool _created;
} pmu_counter;

/**
 * pmu_create_counter() - Create, start, and initialize PMU counters on one CPU.
 * @counter: Counter object to initialize.
 * @cpu: CPU number on which all counters should be created.
 *
 * Creates one per-CPU kernel perf event for every entry in PMU_EVENT_LIST.
 * The events are bound to @cpu and are not bound to any particular task.
 *
 * This means the counters measure activity on the selected CPU regardless of
 * which task is running there. The exact privilege levels being counted depend
 * on the exclude_* fields configured in pmu.c.
 *
 * After all events are created and enabled, this function reads their current
 * cumulative values and stores them as the initial baseline. Therefore, the
 * first pmu_read_events() call returns the delta since pmu_create_counter().
 *
 * The function is all-or-nothing: if any event cannot be created, all already
 * created events are released before returning an error.
 *
 * Context: process context. Do not call from hard IRQ/NMI context.
 *
 * Return: 0 on success, -EINVAL for invalid arguments, -ENODEV if @cpu is not
 * online, or another negative errno from perf event creation.
 */
int pmu_create_counter(pmu_counter* counter, int cpu);

/**
 * pmu_delete_counter() - Stop and release all events owned by a counter.
 * @counter: Counter object previously initialized by pmu_create_counter().
 *
 * Releases every perf event stored in @counter and clears internal pointers.
 *
 * Context: process context. Do not call from hard IRQ/NMI context.
 *
 * Return: 0 on success, or -EINVAL if @counter is NULL.
 */
int pmu_delete_counter(pmu_counter* counter);

/**
 * pmu_read_events() - Read event deltas since the previous read.
 * @counter: Counter object created by pmu_create_counter().
 * @events_count: Output delta snapshot.
 *
 * For each configured event, this function reads the current cumulative perf
 * value, subtracts the previous value stored in @counter, writes the delta into
 * @events_count, and then updates the stored previous value.
 *
 * This means each successful call consumes the interval: the next call will
 * measure from this call to the next one.
 *
 * Context: process context. Do not call from hard IRQ/NMI context.
 *
 * Return: 0 on success, or a negative errno on invalid arguments.
 */
int pmu_read_events(pmu_counter* counter, pmu_events_count* events_count);

/**
 * pmu_reset_counter() - Reset the delta baseline without returning values.
 * @counter: Counter object created by pmu_create_counter().
 *
 * Reads current cumulative values and stores them as the new baseline. The next
 * pmu_read_events() call will return deltas since this reset.
 *
 * Context: process context. Do not call from hard IRQ/NMI context.
 *
 * Return: 0 on success, or a negative errno on invalid arguments.
 */
int pmu_reset_counter(pmu_counter* counter);