#define _GNU_SOURCE

#include <errno.h>
#include <inttypes.h>
#include <sched.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#if !defined(__x86_64__)
#error "This file uses x86-64 GNU inline assembly and must be built for x86-64."
#endif

#define STR2(x) #x
#define STR(x) STR2(x)

/*
 * ============================================================
 * User-tunable parameters
 * ============================================================
 *
 * The program does not configure PMU and does not read PMU.
 * It only creates a repeatable instruction/memory workload.
 */

/* CPU pinning. Use -1 to disable. */
#ifndef PIN_TO_CPU
#define PIN_TO_CPU 2
#endif

/* Realtime scheduling. May require root or CAP_SYS_NICE. */
//#ifndef USE_SCHED_FIFO
#define USE_SCHED_FIFO 0
//#endif

//#ifndef SCHED_FIFO_PRIORITY
#define SCHED_FIFO_PRIORITY 1
//#endif

/* Lock memory. May require proper ulimit/root privileges. */
#ifndef USE_MLOCK
#define USE_MLOCK 1
#endif

/* Must be a power of two and at least 8 bytes. */
#ifndef WORKING_SET_BYTES
#define WORKING_SET_BYTES (256 * 1024)
#endif

/* Stride in uint64_t elements. */
#ifndef MEM_STRIDE_WORDS
#define MEM_STRIDE_WORDS 16
#endif

/* One C-level batch contains this many asm rounds. */
#ifndef ROUNDS_PER_BATCH
#define ROUNDS_PER_BATCH 4096
#endif

/* Number of exact asm operations/groups per round. */
#ifndef ASM_ALU_GROUPS_PER_ROUND
#define ASM_ALU_GROUPS_PER_ROUND 32
#endif

#ifndef ASM_IMULS_PER_ROUND
#define ASM_IMULS_PER_ROUND 8
#endif

#ifndef ASM_IDIVS_PER_ROUND
#define ASM_IDIVS_PER_ROUND 2
#endif

#ifndef ASM_LOADS_PER_ROUND
#define ASM_LOADS_PER_ROUND 8
#endif

#ifndef ASM_STORES_PER_ROUND
#define ASM_STORES_PER_ROUND 2
#endif

#ifndef ASM_BRANCHES_PER_ROUND
#define ASM_BRANCHES_PER_ROUND 8
#endif

#ifndef ASM_NOPS_PER_ROUND
#define ASM_NOPS_PER_ROUND 16
#endif

/* 0 = predictable branch body; 1 = branch condition depends on changing x. */
#ifndef ASM_DATA_DEP_BRANCHES
#define ASM_DATA_DEP_BRANCHES 0
#endif

#if WORKING_SET_BYTES < 8
#error "WORKING_SET_BYTES must be at least 8."
#endif

#if (WORKING_SET_BYTES & (WORKING_SET_BYTES - 1)) != 0
#error "WORKING_SET_BYTES must be a power of two."
#endif

static volatile sig_atomic_t keep_running = 1;
static volatile uint64_t global_sink = 0;

static void handle_signal(int signo) {
    (void)signo;
    keep_running = 0;
}

static void install_signal_handlers(void) {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = handle_signal;

    if (sigaction(SIGINT, &sa, NULL) != 0) {
        perror("sigaction(SIGINT)");
        exit(EXIT_FAILURE);
    }

    if (sigaction(SIGTERM, &sa, NULL) != 0) {
        perror("sigaction(SIGTERM)");
        exit(EXIT_FAILURE);
    }
}

static void pin_to_cpu_if_requested(void) {
#if PIN_TO_CPU >= 0
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(PIN_TO_CPU, &set);

    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        fprintf(stderr, "warning: sched_setaffinity(cpu=%d) failed: %s\n", PIN_TO_CPU, strerror(errno));
    }
#endif
}

static void enable_realtime_if_requested(void) {
#if USE_SCHED_FIFO
    struct sched_param sp;
    memset(&sp, 0, sizeof(sp));
    sp.sched_priority = SCHED_FIFO_PRIORITY;

    if (sched_setscheduler(0, SCHED_FIFO, &sp) != 0) {
        fprintf(stderr, "warning: sched_setscheduler(SCHED_FIFO) failed: %s\n", strerror(errno));
    }
#endif
}

static void lock_memory_if_requested(void) {
#if USE_MLOCK
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        fprintf(stderr, "warning: mlockall() failed: %s\n", strerror(errno));
    }
#endif
}

static uint64_t xorshift64(uint64_t x) {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return x;
}

static uint64_t* allocate_working_set(size_t bytes) {
    void* ptr = NULL;

    if (posix_memalign(&ptr, 4096, bytes) != 0) {
        fprintf(stderr, "posix_memalign failed\n");
        exit(EXIT_FAILURE);
    }

    return (uint64_t*)ptr;
}

static void initialize_working_set(uint64_t* buf, size_t words) {
    uint64_t x = 0x123456789abcdef0ULL;

    for (size_t i = 0; i < words; ++i) {
        x = xorshift64(x + i);
        buf[i] = x;
    }
}

__attribute__((noinline)) static void do_work_asm(uint64_t* buf, size_t words) {
    uint64_t x = global_sink + 0x9e3779b97f4a7c15ULL;
    uint64_t idx = global_sink & (words - 1);
    const uint64_t mask = words - 1;

    for (uint64_t round = 0; round < ROUNDS_PER_BATCH; ++round) {
        __asm__ __volatile__(
            /*
             * ALU group: 3 fixed integer ALU instructions per group.
             */
            ".rept " STR(ASM_ALU_GROUPS_PER_ROUND) "\n\t"
            "addq $0x1234, %[x]\n\t"
            "xorq $0x5678, %[x]\n\t"
            "rolq $7, %[x]\n\t"
            ".endr\n\t"

            /*
             * Fixed integer multiplications.
             */
            ".rept " STR(ASM_IMULS_PER_ROUND) "\n\t"
            "imulq $1103515245, %[x], %[x]\n\t"
            "addq $12345, %[x]\n\t"
            ".endr\n\t"

            /*
             * Fixed signed integer divisions.
             *
             * idivq divides RDX:RAX by r/m64.
             * Quotient -> RAX, remainder -> RDX.
             * r11 is used as a nonzero positive divisor.
             */
            ".rept " STR(ASM_IDIVS_PER_ROUND) "\n\t"
            "movq %[x], %%rax\n\t"
            "movq %[x], %%r11\n\t"
            "sarq $17, %%r11\n\t"
            "andq $0x7fff, %%r11\n\t"
            "orq $1, %%r11\n\t"
            "cqto\n\t"
            "idivq %%r11\n\t"
            "addq %%rdx, %%rax\n\t"
            "addq $0x12345678, %%rax\n\t"
            "movq %%rax, %[x]\n\t"
            ".endr\n\t"

            /*
             * Loads from the working set.
             */
            ".rept " STR(ASM_LOADS_PER_ROUND) "\n\t"
            "addq $" STR(MEM_STRIDE_WORDS) ", %[idx]\n\t"
            "andq %[mask], %[idx]\n\t"
            "movq (%[buf], %[idx], 8), %%r10\n\t"
            "addq %%r10, %[x]\n\t"
            ".endr\n\t"

            /*
             * Stores to the working set.
             */
            ".rept " STR(ASM_STORES_PER_ROUND) "\n\t"
            "addq $" STR(MEM_STRIDE_WORDS) ", %[idx]\n\t"
            "andq %[mask], %[idx]\n\t"
            "movq %[x], (%[buf], %[idx], 8)\n\t"
            ".endr\n\t"

            /*
             * Branches. These are intentionally simple and repeatable.
             */
            ".rept " STR(ASM_BRANCHES_PER_ROUND) "\n\t"
#if ASM_DATA_DEP_BRANCHES
            "testq $1, %[x]\n\t"
#else
            "cmpq %[idx], %[idx]\n\t"
#endif
            "jz 1f\n\t"
            "addq $3, %[x]\n\t"
            "1:\n\t"
            ".endr\n\t"

            /*
             * NOPs.
             */
            ".rept " STR(ASM_NOPS_PER_ROUND) "\n\t"
            "nop\n\t"
            ".endr\n\t"

            : [x] "+&r"(x),
              [idx] "+&r"(idx)
            : [buf] "r"(buf),
              [mask] "r"(mask)
            : "rax", "rdx", "r10", "r11", "cc", "memory"
        );
    }

    global_sink = x + idx;
}

int main(void) {
    const size_t words = WORKING_SET_BYTES / sizeof(uint64_t);

    install_signal_handlers();
    pin_to_cpu_if_requested();
    enable_realtime_if_requested();

    uint64_t* buf = allocate_working_set(WORKING_SET_BYTES);
    initialize_working_set(buf, words);
    lock_memory_if_requested();

    fprintf(stderr, "pid=%ld\n", (long)getpid());
    fprintf(stderr, "working set: %zu bytes\n", (size_t)WORKING_SET_BYTES);
#if PIN_TO_CPU >= 0
    fprintf(stderr, "pinned cpu: %d\n", PIN_TO_CPU);
#endif
    fprintf(stderr, "rounds per batch: %u\n", (unsigned)ROUNDS_PER_BATCH);
    fprintf(stderr, "asm alu groups: %u\n", (unsigned)ASM_ALU_GROUPS_PER_ROUND);
    fprintf(stderr, "asm imuls: %u\n", (unsigned)ASM_IMULS_PER_ROUND);
    fprintf(stderr, "asm idivs: %u\n", (unsigned)ASM_IDIVS_PER_ROUND);
    fprintf(stderr, "asm loads: %u\n", (unsigned)ASM_LOADS_PER_ROUND);
    fprintf(stderr, "asm stores: %u\n", (unsigned)ASM_STORES_PER_ROUND);
    fprintf(stderr, "asm branches: %u\n", (unsigned)ASM_BRANCHES_PER_ROUND);
    fprintf(stderr, "asm nops: %u\n", (unsigned)ASM_NOPS_PER_ROUND);
    fprintf(stderr, "running; stop with Ctrl+C or SIGTERM\n");

    while (keep_running) {
        do_work_asm(buf, words);
    }

    fprintf(stderr, "stopped, sink=%" PRIu64 "\n", global_sink);

    free(buf);
    return 0;
}