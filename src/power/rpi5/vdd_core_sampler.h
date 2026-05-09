#ifndef VDD_CORE_SAMPLER_H
#define VDD_CORE_SAMPLER_H

#include <linux/types.h>

#define VDD_CORE_SAMPLE_PERIOD_US 10000ULL

typedef unsigned long long u64;

int vdd_core_sampler_start(void);
void vdd_core_sampler_stop(void);

/*
 * Возвращает энергию в nanojoule и атомарно сбрасывает аккумулятор.
 *
 * Внутри:
 *   accumulated = P1 + P2 + ... + Pn
 *   E = accumulated * dt
 *
 * Размерности:
 *   power: uW
 *   dt:    us
 *   uW * us = pJ
 *   pJ / 1000 = nJ
 */
u64 vdd_core_take_energy_nj(void);

/*
 * То же самое, но дополнительно возвращает количество успешных сэмплов.
 */
u64 vdd_core_take_energy_nj_and_samples(u64 *samples);

#endif /* VDD_CORE_SAMPLER_H */