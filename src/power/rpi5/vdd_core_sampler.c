#include <linux/atomic.h>
#include <linux/hrtimer.h>
#include <linux/kernel.h>
#include <linux/kthread.h>
#include <linux/ktime.h>
#include <linux/math64.h>
#include <linux/module.h>
#include <linux/of.h>
#include <linux/sched.h>
#include <linux/sched/rt.h>
#include <linux/string.h>
#include <linux/wait.h>
#include <soc/bcm2835/raspberrypi-firmware.h>

#include "vdd_core_sampler.h"

#define RPI_FIRMWARE_GET_GENCMD_RESULT 0x00030080

#define VDD_CORE_THREAD_NAME "vdd_core_sampler"

/*
 * Ограничение, чтобы при слишком медленном firmware-вызове поток
 * не пытался бесконечно догонять накопившиеся тики.
 */
#define VDD_CORE_MAX_PENDING_TICKS 1

struct rpi_gencmd {
  u32 error;
  char text[1024];
} __packed __aligned(4);

static struct rpi_firmware *vdd_core_fw;
static struct task_struct *vdd_core_thread;
static struct hrtimer vdd_core_timer;

static DECLARE_WAIT_QUEUE_HEAD(vdd_core_wq);

static atomic_t vdd_core_pending_ticks = ATOMIC_INIT(0);
static atomic64_t vdd_core_power_sum_uw = ATOMIC64_INIT(0);
static atomic64_t vdd_core_sample_count = ATOMIC64_INIT(0);

static int parse_fixed_decimal_to_micro(const char *p, char end_ch, u64 *out) {
  u64 int_part = 0;
  u64 frac_part = 0;
  u64 scale = 1000000;
  int frac_digits = 0;

  if (!p || !out)
    return -EINVAL;

  while (*p >= '0' && *p <= '9') {
    int_part = int_part * 10 + (*p - '0');
    p++;
  }

  if (*p == '.') {
    p++;

    while (*p >= '0' && *p <= '9' && frac_digits < 6) {
      scale /= 10;
      frac_part += (*p - '0') * scale;
      frac_digits++;
      p++;
    }

    while (*p >= '0' && *p <= '9')
      p++;
  }

  if (*p != end_ch)
    return -EINVAL;

  *out = int_part * 1000000 + frac_part;
  return 0;
}

static int parse_named_adc_value_micro(const char *buf, const char *name,
                                       char unit, u64 *out_micro) {
  const char *p;

  p = strstr(buf, name);
  if (!p)
    return -ENOENT;

  p = strchr(p, '=');
  if (!p)
    return -EINVAL;

  p++;

  return parse_fixed_decimal_to_micro(p, unit, out_micro);
}

static int rpi_read_vdd_core_power_uw(struct rpi_firmware *fw, u64 *power_uw) {
  struct rpi_gencmd cmd;
  u64 current_ua;
  u64 voltage_uv;
  int ret;

  if (!fw || !power_uw)
    return -EINVAL;

  memset(&cmd, 0, sizeof(cmd));
  strscpy(cmd.text, "pmic_read_adc", sizeof(cmd.text));

  ret = rpi_firmware_property(fw, RPI_FIRMWARE_GET_GENCMD_RESULT, &cmd,
                              sizeof(cmd));
  if (ret)
    return ret;

  if (cmd.error)
    return -EIO;

  ret = parse_named_adc_value_micro(cmd.text, "VDD_CORE_A current", 'A',
                                    &current_ua);
  if (ret)
    return ret;

  ret = parse_named_adc_value_micro(cmd.text, "VDD_CORE_V volt", 'V',
                                    &voltage_uv);
  if (ret)
    return ret;

  /*
   * uA * uV = 1e-12 W.
   * Делим на 1e6, получаем uW.
   */
  *power_uw = div64_u64(current_ua * voltage_uv, 1000000ULL);

  return 0;
}

static enum hrtimer_restart vdd_core_timer_cb(struct hrtimer *timer) {
  int pending;

  pending = atomic_read(&vdd_core_pending_ticks);

  if (pending < VDD_CORE_MAX_PENDING_TICKS)
    atomic_inc(&vdd_core_pending_ticks);

  wake_up_interruptible(&vdd_core_wq);

  hrtimer_forward_now(timer, ns_to_ktime(VDD_CORE_SAMPLE_PERIOD_US * 1000ULL));

  return HRTIMER_RESTART;
}

static int vdd_core_sampler_thread_fn(void *arg) {
  struct rpi_firmware *fw = arg;
  u64 power_uw;
  int ret;

  while (!kthread_should_stop()) {
    wait_event_interruptible(vdd_core_wq,
                             kthread_should_stop() ||
                                 atomic_read(&vdd_core_pending_ticks) > 0);

    if (kthread_should_stop())
      break;

    if (atomic_dec_if_positive(&vdd_core_pending_ticks) < 0)
      continue;

    ret = rpi_read_vdd_core_power_uw(fw, &power_uw);
    if (!ret) {
      atomic64_add((s64)power_uw, &vdd_core_power_sum_uw);
      atomic64_inc(&vdd_core_sample_count);
    }
  }

  return 0;
}

int vdd_core_sampler_start(void) {
  struct device_node *fw_np;
  struct sched_param param = {
      .sched_priority = MAX_RT_PRIO - 2,
  };
  int ret = 0;

  if (vdd_core_fw || vdd_core_thread)
    return -EBUSY;

  atomic_set(&vdd_core_pending_ticks, 0);
  atomic64_set(&vdd_core_power_sum_uw, 0);
  atomic64_set(&vdd_core_sample_count, 0);

  fw_np = rpi_firmware_find_node();
  if (!fw_np)
    return -ENODEV;

  vdd_core_fw = rpi_firmware_get(fw_np);
  of_node_put(fw_np);

  if (!vdd_core_fw)
    return -EPROBE_DEFER;

  vdd_core_thread = kthread_run(vdd_core_sampler_thread_fn, vdd_core_fw,
                                VDD_CORE_THREAD_NAME);
  if (IS_ERR(vdd_core_thread)) {
    ret = PTR_ERR(vdd_core_thread);
    vdd_core_thread = NULL;
    goto err_put_fw;
  }

  /*
   * Уменьшает latency пробуждения.
   * Если модуль не должен вмешиваться в RT-планирование, этот блок
   * можно удалить.
   */
  sched_setscheduler_nocheck(vdd_core_thread, SCHED_FIFO, &param);

  hrtimer_init(&vdd_core_timer, CLOCK_MONOTONIC, HRTIMER_MODE_REL_PINNED);

  vdd_core_timer.function = vdd_core_timer_cb;

  hrtimer_start(&vdd_core_timer,
                ns_to_ktime(VDD_CORE_SAMPLE_PERIOD_US * 1000ULL),
                HRTIMER_MODE_REL_PINNED);

  return 0;

err_put_fw:
  rpi_firmware_put(vdd_core_fw);
  vdd_core_fw = NULL;
  return ret;
}
EXPORT_SYMBOL_GPL(vdd_core_sampler_start);

void vdd_core_sampler_stop(void) {
  hrtimer_cancel(&vdd_core_timer);

  if (vdd_core_thread) {
    wake_up_interruptible(&vdd_core_wq);
    kthread_stop(vdd_core_thread);
    vdd_core_thread = NULL;
  }

  if (vdd_core_fw) {
    rpi_firmware_put(vdd_core_fw);
    vdd_core_fw = NULL;
  }

  atomic_set(&vdd_core_pending_ticks, 0);
}
EXPORT_SYMBOL_GPL(vdd_core_sampler_stop);

u64 vdd_core_take_energy_nj(void) {
  s64 sum_uw;

  sum_uw = atomic64_xchg(&vdd_core_power_sum_uw, 0);

  if (sum_uw <= 0)
    return 0;

  /*
   * uW * us = pJ
   * pJ / 1000 = nJ
   */
  return div64_u64((u64)sum_uw * VDD_CORE_SAMPLE_PERIOD_US, 1000ULL);
}
EXPORT_SYMBOL_GPL(vdd_core_take_energy_nj);

u64 vdd_core_take_energy_nj_and_samples(u64 *samples) {
  s64 sum_uw;
  s64 n;

  sum_uw = atomic64_xchg(&vdd_core_power_sum_uw, 0);
  n = atomic64_xchg(&vdd_core_sample_count, 0);

  if (samples)
    *samples = n > 0 ? (u64)n : 0;

  if (sum_uw <= 0)
    return 0;

  return div64_u64((u64)sum_uw * VDD_CORE_SAMPLE_PERIOD_US, 1000ULL);
}
EXPORT_SYMBOL_GPL(vdd_core_take_energy_nj_and_samples);