/* SPDX-License-Identifier: GPL-2.0 */
#ifndef ENERGY_METER_H
#define ENERGY_METER_H

#include <linux/types.h>

enum energy_meter_backend {
    ENERGY_METER_BACKEND_NONE = 0,
    ENERGY_METER_BACKEND_RPI5_VDD_CORE,
    ENERGY_METER_BACKEND_AMD_ZEN2_RAPL,
    ENERGY_METER_BACKEND_LINUX_POWERCAP,
};

/*
 * Инициализирует backend и запоминает текущий счётчик как baseline.
 *
 * Возвращает:
 *   0        - успех
 *   -ENODEV  - нет подходящего источника энергии
 *   другие   - ошибка чтения backend-а
 */
int energy_meter_init(void);

/*
 * Деинициализация. После неё energy_meter_read_delta_uj() вернёт -ENODEV.
 */
void energy_meter_deinit(void);

/*
 * Возвращает энергию, потреблённую с прошлого вызова energy_meter_read_delta_uj()
 * или с energy_meter_init(), если это первый вызов.
 *
 * Единицы: микроджоули.
 */
int energy_meter_read_delta_uj(u64* energy_uj);

enum energy_meter_backend energy_meter_get_backend(void);
const char* energy_meter_backend_name(void);

#endif /* ENERGY_METER_H */