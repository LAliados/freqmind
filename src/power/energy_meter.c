// SPDX-License-Identifier: GPL-2.0
#include "energy_meter.h"

#include <linux/cpu.h>
#include <linux/err.h>
#include <linux/export.h>
#include <linux/fs.h>
#include <linux/kernel.h>
#include <linux/limits.h>
#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/mutex.h>
#include <linux/string.h>
#include <linux/types.h>

#if IS_ENABLED(CONFIG_OF)
#include <linux/of.h>
#endif

#if defined(CONFIG_X86)
#include <asm/msr.h>
#include <asm/processor.h>
#endif

/*
 * Для Raspberry Pi 5 пользовательская функция должна быть слинкована
 * в этот же модуль или быть доступна на этапе линковки.
 *
 * Если helper всегда присутствует, можно убрать __weak.
 */
#if defined(CONFIG_ARM64)
extern u64 vdd_core_take_energy_nj(void) __weak;
#endif

/* ======================= Общая часть API ======================= */

static DEFINE_MUTEX(energy_meter_lock);

static enum energy_meter_backend current_backend = ENERGY_METER_BACKEND_NONE;
static bool energy_meter_initialized;

/* ======================= Raspberry Pi 5 backend ======================= */
/* Здесь находится код для Raspberry Pi 5. */

#if defined(CONFIG_ARM64)

static bool rpi5_is_supported(void) {
#if IS_ENABLED(CONFIG_OF)
    if (!vdd_core_take_energy_nj)
        return false;

    return of_machine_is_compatible("raspberrypi,5-model-b") || of_machine_is_compatible("brcm,bcm2712");
#else
    return false;
#endif
}

static int rpi5_energy_init(void) {
    /*
	 * Предполагается, что vdd_core_take_energy_nj() возвращает энергию
	 * за период с прошлого вызова и внутри себя сбрасывает/обновляет
	 * baseline. Поэтому первый вызов здесь нужен для обнуления интервала.
	 *
	 * Если ваша функция возвращает накопительный счётчик, замените этот
	 * backend на хранение last_nj и вычисление delta.
	 */
    (void)vdd_core_take_energy_nj();

    current_backend = ENERGY_METER_BACKEND_RPI5_VDD_CORE;
    return 0;
}

static int rpi5_energy_read_delta_uj(u64* energy_uj) {
    u64 energy_nj;

    if (!vdd_core_take_energy_nj)
        return -ENODEV;

    energy_nj = vdd_core_take_energy_nj();

    /* nJ -> uJ, с округлением до ближайшего целого. */
    *energy_uj = (energy_nj + 500ULL) / 1000ULL;

    return 0;
}

#else

static bool rpi5_is_supported(void) {
    return false;
}

static int rpi5_energy_init(void) {
    return -ENODEV;
}

static int rpi5_energy_read_delta_uj(u64* energy_uj) {
    return -ENODEV;
}

#endif /* CONFIG_ARM64 */

/* ======================= AMD Zen2 backend ======================= */
/* Здесь находится код для AMD Zen2. */

#if defined(CONFIG_X86)

/*
 * AMD Family 17h RAPL MSRs:
 *   C001_0299 - Power, Energy and Time Units
 *   C001_029B - Package/Socket Energy Status
 */
#define AMD_MSR_RAPL_POWER_UNIT 0xC0010299
#define AMD_MSR_PACKAGE_ENERGY_STATUS 0xC001029B

#define RAPL_ENERGY_UNIT_SHIFT 8
#define RAPL_ENERGY_UNIT_MASK 0x1f

static unsigned int amd_zen2_cpu;
static u8 amd_zen2_energy_unit;
static u32 amd_zen2_last_raw;

static bool amd_zen2_is_supported(void) {
    const struct cpuinfo_x86* c = &boot_cpu_data;

    if (c->x86_vendor != X86_VENDOR_AMD)
        return false;

    if (c->x86 != 0x17)
        return false;

    /*
	 * Zen2 в основном находится в Family 17h Model 30h-3Fh,
	 * 60h-6Fh и 70h-7Fh. При необходимости сузьте список моделей
	 * под конкретную линейку CPU.
	 */
    return (c->x86_model >= 0x30 && c->x86_model <= 0x3f) || (c->x86_model >= 0x60 && c->x86_model <= 0x7f);
}

static int amd_zen2_read_package_raw(u32* raw) {
    u64 val;
    int ret;

    if (!cpu_online(amd_zen2_cpu))
        return -ENODEV;

    ret = rdmsrl_safe_on_cpu(amd_zen2_cpu, AMD_MSR_PACKAGE_ENERGY_STATUS, &val);
    if (ret)
        return ret;

    /*
	 * Документированный energy status register — 32-битный накопитель.
	 * Переполнение корректно обрабатывается unsigned subtraction ниже.
	 */
    *raw = (u32)val;

    return 0;
}

static int amd_zen2_energy_init(void) {
    u64 unit_msr;
    u32 raw;
    int ret;
    unsigned int cpu;

    cpus_read_lock();

    cpu = cpumask_first(cpu_online_mask);
    if (cpu >= nr_cpu_ids) {
        cpus_read_unlock();
        return -ENODEV;
    }

    ret = rdmsrl_safe_on_cpu(cpu, AMD_MSR_RAPL_POWER_UNIT, &unit_msr);
    if (ret) {
        cpus_read_unlock();
        return ret;
    }

    amd_zen2_cpu = cpu;
    amd_zen2_energy_unit = (unit_msr >> RAPL_ENERGY_UNIT_SHIFT) & RAPL_ENERGY_UNIT_MASK;

    ret = amd_zen2_read_package_raw(&raw);
    if (ret) {
        cpus_read_unlock();
        return ret;
    }

    amd_zen2_last_raw = raw;

    cpus_read_unlock();

    current_backend = ENERGY_METER_BACKEND_AMD_ZEN2_RAPL;
    return 0;
}

static int amd_zen2_energy_read_delta_uj(u64* energy_uj) {
    u32 raw;
    u32 delta_raw;
    u64 scaled;
    int ret;

    ret = amd_zen2_read_package_raw(&raw);
    if (ret)
        return ret;

    /*
	 * Для u32 subtraction переполнение даёт корректную delta по модулю 2^32.
	 */
    delta_raw = raw - amd_zen2_last_raw;
    amd_zen2_last_raw = raw;

    /*
	 * uJ = raw_delta * 1_000_000 / 2^energy_unit
	 */
    scaled = (u64)delta_raw * 1000000ULL;
    *energy_uj = scaled >> amd_zen2_energy_unit;

    return 0;
}

#else

static bool amd_zen2_is_supported(void) {
    return false;
}

static int amd_zen2_energy_init(void) {
    return -ENODEV;
}

static int amd_zen2_energy_read_delta_uj(u64* energy_uj) {
    return -ENODEV;
}

#endif /* CONFIG_X86 */

/* ======================= Linux fallback backend ======================= */
/*
 * Здесь находится код для остальных процессоров.
 *
 * Это fallback через стандартный Linux powercap sysfs.
 *
 * Важно: чтение sysfs из kernel-space — компромиссный вариант. Для production
 * обычно лучше либо читать powercap из userspace, либо добавить в нужный
 * драйвер явный exported API. Но если модуль обязан сам получить fallback,
 * этот код работает как практичная интеграционная точка.
 */

static char linux_std_energy_uj_path[PATH_MAX] = "/sys/class/powercap/intel-rapl:0/energy_uj";

static char linux_std_max_energy_range_uj_path[PATH_MAX] = "/sys/class/powercap/intel-rapl:0/max_energy_range_uj";

module_param_string(linux_std_energy_uj_path, linux_std_energy_uj_path, sizeof(linux_std_energy_uj_path), 0444);
MODULE_PARM_DESC(linux_std_energy_uj_path, "Path to standard Linux powercap energy_uj file");

module_param_string(linux_std_max_energy_range_uj_path, linux_std_max_energy_range_uj_path,
                    sizeof(linux_std_max_energy_range_uj_path), 0444);
MODULE_PARM_DESC(linux_std_max_energy_range_uj_path, "Path to standard Linux powercap max_energy_range_uj file");

static u64 linux_std_last_uj;
static u64 linux_std_max_range_uj;

static int linux_std_read_u64_file(const char* path, u64* value) {
    struct file* file;
    char buf[64];
    loff_t pos = 0;
    ssize_t len;
    int ret;

    file = filp_open(path, O_RDONLY, 0);
    if (IS_ERR(file))
        return PTR_ERR(file);

    len = kernel_read(file, buf, sizeof(buf) - 1, &pos);
    filp_close(file, NULL);

    if (len < 0)
        return (int)len;

    if (len == 0)
        return -EIO;

    buf[len] = '\0';

    ret = kstrtou64(strim(buf), 10, value);
    if (ret)
        return ret;

    return 0;
}

static int linux_std_energy_init(void) {
    u64 now_uj;
    u64 max_range_uj = 0;
    int ret;

    ret = linux_std_read_u64_file(linux_std_energy_uj_path, &now_uj);
    if (ret)
        return ret;

    /*
	 * max_energy_range_uj нужен для обработки wraparound.
	 * Если файла нет, backend всё равно может работать, но wraparound
	 * будет возвращать -EOVERFLOW.
	 */
    (void)linux_std_read_u64_file(linux_std_max_energy_range_uj_path, &max_range_uj);

    linux_std_last_uj = now_uj;
    linux_std_max_range_uj = max_range_uj;

    current_backend = ENERGY_METER_BACKEND_LINUX_POWERCAP;
    return 0;
}

static int linux_std_energy_read_delta_uj(u64* energy_uj) {
    u64 now_uj;
    u64 delta_uj;
    int ret;

    ret = linux_std_read_u64_file(linux_std_energy_uj_path, &now_uj);
    if (ret)
        return ret;

    if (now_uj >= linux_std_last_uj) {
        delta_uj = now_uj - linux_std_last_uj;
    } else {
        if (!linux_std_max_range_uj)
            return -EOVERFLOW;

        /*
		 * powercap energy_uj — накопительный счётчик с диапазоном
		 * max_energy_range_uj.
		 */
        delta_uj = (linux_std_max_range_uj - linux_std_last_uj) + now_uj;
    }

    linux_std_last_uj = now_uj;
    *energy_uj = delta_uj;

    return 0;
}

/* ======================= Публичный API ======================= */

int energy_meter_init(void) {
    int ret;

    mutex_lock(&energy_meter_lock);

    if (energy_meter_initialized) {
        mutex_unlock(&energy_meter_lock);
        return 0;
    }

    current_backend = ENERGY_METER_BACKEND_NONE;

    if (rpi5_is_supported()) {
        ret = rpi5_energy_init();
        goto out;
    }

    if (amd_zen2_is_supported()) {
        ret = amd_zen2_energy_init();
        goto out;
    }

    ret = linux_std_energy_init();

out:
    if (!ret)
        energy_meter_initialized = true;
    else
        current_backend = ENERGY_METER_BACKEND_NONE;

    mutex_unlock(&energy_meter_lock);
    return ret;
}
EXPORT_SYMBOL_GPL(energy_meter_init);

void energy_meter_deinit(void) {
    mutex_lock(&energy_meter_lock);

    energy_meter_initialized = false;
    current_backend = ENERGY_METER_BACKEND_NONE;

    amd_zen2_cpu = 0;
    amd_zen2_energy_unit = 0;
    amd_zen2_last_raw = 0;

    linux_std_last_uj = 0;
    linux_std_max_range_uj = 0;

    mutex_unlock(&energy_meter_lock);
}
EXPORT_SYMBOL_GPL(energy_meter_deinit);

int energy_meter_read_delta_uj(u64* energy_uj) {
    int ret;

    if (!energy_uj)
        return -EINVAL;

    mutex_lock(&energy_meter_lock);

    if (!energy_meter_initialized) {
        ret = -ENODEV;
        goto out;
    }

    switch (current_backend) {
        case ENERGY_METER_BACKEND_RPI5_VDD_CORE:
            ret = rpi5_energy_read_delta_uj(energy_uj);
            break;

        case ENERGY_METER_BACKEND_AMD_ZEN2_RAPL:
            ret = amd_zen2_energy_read_delta_uj(energy_uj);
            break;

        case ENERGY_METER_BACKEND_LINUX_POWERCAP:
            ret = linux_std_energy_read_delta_uj(energy_uj);
            break;

        default:
            ret = -ENODEV;
            break;
    }

out:
    mutex_unlock(&energy_meter_lock);
    return ret;
}
EXPORT_SYMBOL_GPL(energy_meter_read_delta_uj);

enum energy_meter_backend energy_meter_get_backend(void) {
    return current_backend;
}
EXPORT_SYMBOL_GPL(energy_meter_get_backend);

const char* energy_meter_backend_name(void) {
    switch (current_backend) {
        case ENERGY_METER_BACKEND_RPI5_VDD_CORE:
            return "raspberry-pi-5-vdd-core";

        case ENERGY_METER_BACKEND_AMD_ZEN2_RAPL:
            return "amd-zen2-rapl-msr";

        case ENERGY_METER_BACKEND_LINUX_POWERCAP:
            return "linux-powercap-sysfs";

        case ENERGY_METER_BACKEND_NONE:
        default:
            return "none";
    }
}
EXPORT_SYMBOL_GPL(energy_meter_backend_name);