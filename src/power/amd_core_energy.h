/* SPDX-License-Identifier: GPL-2.0 */
#ifndef AMD_CORE_ENERGY_H
#define AMD_CORE_ENERGY_H

#include <linux/types.h>

/*
 * Инициализация подсистемы чтения AMD core energy.
 *
 * Вызывать один раз из module_init().
 */
int amd_energy_init_state(void);

/*
 * Освобождение ресурсов.
 *
 * Вызывать из module_exit().
 */
void amd_energy_free_state(void);

/*
 * Возвращает энергию, потребленную core, на котором расположен logical CPU `cpu`,
 * с момента прошлого вызова этой функции.
 *
 * Результат возвращается в микроджоулях через delta_uj.
 *
 * Первый вызов для конкретного cpu возвращает 0, потому что он только
 * запоминает начальное значение счетчика.
 */
int amd_core_energy_delta_uj(int cpu, u64* delta_uj);

#endif /* AMD_CORE_ENERGY_H */