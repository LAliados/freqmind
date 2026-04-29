// SPDX-License-Identifier: GPL-2.0

#include <linux/fs.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/miscdevice.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/slab.h>
#include <linux/uaccess.h>

#define DEVICE_NAME "test"
#define MAX_STORE_SIZE 128

static char* stored_msg;
static size_t stored_len;
static DEFINE_MUTEX(stored_lock);

#ifdef TEST_ANALYZERS_WITH_BUGS
static void analyzer_traps(const char __user* user_buf, size_t count) {
    char stack_buf[8];
    char* leak;

    /*
	 * Намеренная утечка памяти.
	 * Некоторые анализаторы должны заметить, что leak не освобождается.
	 */
    leak = kmalloc(32, GFP_KERNEL);
    if (leak)
        strcpy(leak, "memory leak");

    /*
	 * Намеренное прямое обращение к __user-указателю.
	 * sparse должен ругаться на dereference of noderef expression.
	 */
    if (count > 0 && user_buf[0] == 'X')
        pr_info("direct user pointer dereference\n");

    /*
	 * Намеренный потенциальный overflow.
	 * Не загружайте BUGS=1 модуль на рабочей системе.
	 */
    if (count > sizeof(stack_buf))
        memcpy(stack_buf, "0123456789abcdef", count);

    if (stack_buf[0] == '\0')
        pr_info("unreachable-ish branch\n");
}
#endif

static ssize_t analyzer_demo_read(struct file* file, char __user* user_buf, size_t count, loff_t* ppos) {
    ssize_t ret;

    mutex_lock(&stored_lock);

    if (!stored_msg) {
        mutex_unlock(&stored_lock);
        return 0;
    }

    ret = simple_read_from_buffer(user_buf, count, ppos, stored_msg, stored_len);

    mutex_unlock(&stored_lock);

    return ret;
}

static ssize_t analyzer_demo_write(struct file* file, const char __user* user_buf, size_t count, loff_t* ppos) {
    char* new_msg;
    size_t len;

#ifdef TEST_ANALYZERS_WITH_BUGS
    analyzer_traps(user_buf, count);
#endif

    if (count == 0)
        return 0;

    len = min_t(size_t, count, MAX_STORE_SIZE);

    new_msg = kmalloc(len + 1, GFP_KERNEL);
    if (!new_msg)
        return -ENOMEM;

    if (copy_from_user(new_msg, user_buf, len)) {
        kfree(new_msg);
        return -EFAULT;
    }

    new_msg[len] = '\0';

    mutex_lock(&stored_lock);

    kfree(stored_msg);
    stored_msg = new_msg;
    stored_len = len;

    mutex_unlock(&stored_lock);

    return len;
}

static const struct file_operations analyzer_demo_fops = {
    .owner = THIS_MODULE,
    .read = analyzer_demo_read,
    .write = analyzer_demo_write,
    .llseek = default_llseek,
};

static struct miscdevice analyzer_demo_device = {
    .minor = MISC_DYNAMIC_MINOR,
    .name = DEVICE_NAME,
    .fops = &analyzer_demo_fops,
    .mode = 0666,
};

static int __init analyzer_demo_init(void) {
    int ret;

    ret = misc_register(&analyzer_demo_device);
    if (ret) {
        pr_err("failed to register misc device: %d\n", ret);
        return ret;
    }

    pr_info("/dev/%s registered\n", DEVICE_NAME);
    return 0;
}

static void __exit analyzer_demo_exit(void) {
    misc_deregister(&analyzer_demo_device);

    mutex_lock(&stored_lock);
    kfree(stored_msg);
    stored_msg = NULL;
    stored_len = 0;
    mutex_unlock(&stored_lock);

    pr_info("module unloaded\n");
}

module_init(analyzer_demo_init);
module_exit(analyzer_demo_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Static analyzer test");
MODULE_DESCRIPTION("Very small Linux driver for static analyzer checks");
MODULE_VERSION("0.1");