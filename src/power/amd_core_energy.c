// SPDX-License-Identifier: GPL-2.0

#include <asm/msr.h>
#include <linux/cpu.h>
#include <linux/errno.h>
#include <linux/kernel.h>
#include <linux/math64.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/slab.h>

#include "amd_core_energy.h"

#define AMD_MSR_RAPL_POWER_UNIT 0xC0010299
#define AMD_MSR_CORE_ENERGY 0xC001029A

#define AMD_ENERGY_UNIT_MASK 0x00001F00ULL
#define AMD_ENERGY_UNIT_SHIFT 8
#define AMD_ENERGY_MASK 0xFFFFFFFFULL

struct amd_core_energy_state {
    bool initialized;
    u32 last_raw;
};

static DEFINE_MUTEX(amd_energy_lock);

static struct amd_core_energy_state* amd_energy_state;
static unsigned int amd_energy_esu;

int amd_energy_init_state(void) {
    u64 units;
    int ret;

    ret = rdmsrl_safe(AMD_MSR_RAPL_POWER_UNIT, &units);
    if (ret)
        return ret;

    amd_energy_esu = (unsigned int)((units & AMD_ENERGY_UNIT_MASK) >> AMD_ENERGY_UNIT_SHIFT);

    amd_energy_state = kcalloc(nr_cpu_ids, sizeof(*amd_energy_state), GFP_KERNEL);
    if (!amd_energy_state)
        return -ENOMEM;

    return 0;
}

void amd_energy_free_state(void) {
    kfree(amd_energy_state);
    amd_energy_state = NULL;
    amd_energy_esu = 0;
}

int amd_core_energy_delta_uj(int cpu, u64* delta_uj) {
    u64 msr_value;
    u32 raw;
    u32 prev;
    u32 delta_raw;
    int ret;

    if (!delta_uj)
        return -EINVAL;

    if (!amd_energy_state)
        return -EINVAL;

    if (cpu < 0 || cpu >= nr_cpu_ids)
        return -EINVAL;

    if (!cpu_online(cpu))
        return -ENODEV;

    ret = rdmsrl_safe_on_cpu(cpu, AMD_MSR_CORE_ENERGY, &msr_value);
    if (ret)
        return ret;

    raw = (u32)(msr_value & AMD_ENERGY_MASK);

    mutex_lock(&amd_energy_lock);

    if (!amd_energy_state[cpu].initialized) {
        amd_energy_state[cpu].last_raw = raw;
        amd_energy_state[cpu].initialized = true;
        *delta_uj = 0;

        mutex_unlock(&amd_energy_lock);
        return 0;
    }

    prev = amd_energy_state[cpu].last_raw;

    /*
     * Счетчик 32-битный и может переполниться.
     * Арифметика u32 корректно обрабатывает один wrap-around.
     */
    delta_raw = raw - prev;

    amd_energy_state[cpu].last_raw = raw;

    mutex_unlock(&amd_energy_lock);

    /*
     * joules = raw / 2^ESU
     * microjoules = raw * 1000000 / 2^ESU
     */
    *delta_uj = div64_u64((u64)delta_raw * 1000000ULL, 1ULL << amd_energy_esu);

    return 0;
}