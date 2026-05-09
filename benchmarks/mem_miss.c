#define _GNU_SOURCE

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__linux__)
#include <sched.h>
#include <unistd.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#include <x86intrin.h>
#define ARCH_X86_64 1
#else
#define ARCH_X86_64 0
#endif

#if defined(__aarch64__)
#define ARCH_AARCH64 1
#else
#define ARCH_AARCH64 0
#endif

#define CACHE_LINE 64

typedef struct Node {
    struct Node* next;
    unsigned char pad[CACHE_LINE - sizeof(void*)];
} Node;

_Static_assert(sizeof(Node) == CACHE_LINE, "Node size must be equal to CACHE_LINE");

typedef struct ChaseResult {
    size_t bytes;
    size_t nodes;
    uint64_t iterations;
    double p50_ns;
    double p90_ns;
    double p99_ns;
} ChaseResult;

typedef enum OutputMode { OUTPUT_VALUE_ONLY = 0, OUTPUT_VERBOSE = 1, OUTPUT_CSV = 2 } OutputMode;

typedef enum MetricKind { METRIC_P50 = 50, METRIC_P90 = 90, METRIC_P99 = 99 } MetricKind;

typedef struct Config {
    uint64_t repeat_count;
    uint64_t min_iters_millions;
    uint64_t max_iters_millions;
    uint64_t x86_samples;
    double ratio;
    size_t max_size_mib;
    int pin_cpu;
    int run_x86_clflush;
    OutputMode output_mode;
    MetricKind metric;
} Config;

static volatile uint64_t sink_u64 = 0;
static volatile Node* sink_node = 0;

static uint64_t now_ns(void) {
    struct timespec ts;

#ifdef CLOCK_MONOTONIC_RAW
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
#else
    clock_gettime(CLOCK_MONOTONIC, &ts);
#endif

    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void print_help(const char* prog) {
    printf(
        "Usage:\n"
        "  %s [options]\n\n"
        "Default output is one number: estimated RAM minus LLC latency in ns.\n\n"
        "Options:\n"
        "  --repeat N              Repetitions per working set, default 9. Even N is rounded up.\n"
        "  --min-iters-m M         Minimum iterations per repetition, in millions, default 10.\n"
        "  --max-iters-m M         Maximum iterations per repetition, in millions, default 100.\n"
        "  --max-size-mib M        Maximum working set size in MiB, default 128.\n"
        "  --ratio R               Boundary ratio, default 1.3.\n"
        "                          RAM latency is max metric; LLC candidate is the largest\n"
        "                          lower metric where RAM / candidate > R.\n"
        "  --metric p50|p90|p99    Metric used for final estimate, default p50.\n"
        "  --verbose               Print detailed human-readable output.\n"
        "  --csv                   Print CSV table plus final estimate row.\n"
        "  --cpu N                 Pin process to CPU N on Linux. Default 0.\n"
        "  --no-pin                Do not pin process to a CPU.\n"
        "  --x86-clflush           Also run explicit x86_64 clflush test when available.\n"
        "  --x86-samples N         Samples for x86_64 clflush test, default 100000.\n"
        "  --quick                 Shortcut: --repeat 5 --max-iters-m 30 --x86-samples 30000.\n"
        "  --long                  Shortcut: --repeat 15 --max-iters-m 300 --x86-samples 300000.\n"
        "  --help                  Show this help.\n\n"
        "Examples:\n"
        "  %s\n"
        "  %s --verbose\n"
        "  %s --repeat 15 --max-iters-m 300 --ratio 1.3\n"
        "  %s --csv --metric p90\n",
        prog, prog, prog, prog, prog);
}

static uint64_t parse_u64_value(const char* s, int* ok) {
    char* end = NULL;
    errno = 0;

    unsigned long long v = strtoull(s, &end, 10);

    if (errno != 0 || end == s || *end != '\0') {
        *ok = 0;
        return 0;
    }

    *ok = 1;
    return (uint64_t)v;
}

static double parse_double_value(const char* s, int* ok) {
    char* end = NULL;
    errno = 0;

    double v = strtod(s, &end);

    if (errno != 0 || end == s || *end != '\0') {
        *ok = 0;
        return 0.0;
    }

    *ok = 1;
    return v;
}

static const char* require_value(int argc, char** argv, int* i) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "Missing value after %s\n", argv[*i]);
        exit(2);
    }

    ++(*i);
    return argv[*i];
}

static void parse_args(int argc, char** argv, Config* cfg) {
    cfg->repeat_count = 9;
    cfg->min_iters_millions = 10;
    cfg->max_iters_millions = 100;
    cfg->x86_samples = 100000;
    cfg->ratio = 1.3;
    cfg->max_size_mib = 128;
    cfg->pin_cpu = 0;
    cfg->run_x86_clflush = 0;
    cfg->output_mode = OUTPUT_VALUE_ONLY;
    cfg->metric = METRIC_P50;

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];

        if (strcmp(a, "--help") == 0 || strcmp(a, "-h") == 0) {
            print_help(argv[0]);
            exit(0);
        } else if (strcmp(a, "--verbose") == 0 || strcmp(a, "-v") == 0) {
            cfg->output_mode = OUTPUT_VERBOSE;
        } else if (strcmp(a, "--csv") == 0) {
            cfg->output_mode = OUTPUT_CSV;
        } else if (strcmp(a, "--no-pin") == 0) {
            cfg->pin_cpu = -1;
        } else if (strcmp(a, "--x86-clflush") == 0) {
            cfg->run_x86_clflush = 1;
        } else if (strcmp(a, "--quick") == 0) {
            cfg->repeat_count = 5;
            cfg->max_iters_millions = 30;
            cfg->x86_samples = 30000;
        } else if (strcmp(a, "--long") == 0) {
            cfg->repeat_count = 15;
            cfg->max_iters_millions = 300;
            cfg->x86_samples = 300000;
        } else if (strcmp(a, "--repeat") == 0) {
            int ok = 0;
            cfg->repeat_count = parse_u64_value(require_value(argc, argv, &i), &ok);
            if (!ok) {
                fprintf(stderr, "Invalid --repeat value\n");
                exit(2);
            }
        } else if (strcmp(a, "--min-iters-m") == 0) {
            int ok = 0;
            cfg->min_iters_millions = parse_u64_value(require_value(argc, argv, &i), &ok);
            if (!ok) {
                fprintf(stderr, "Invalid --min-iters-m value\n");
                exit(2);
            }
        } else if (strcmp(a, "--max-iters-m") == 0) {
            int ok = 0;
            cfg->max_iters_millions = parse_u64_value(require_value(argc, argv, &i), &ok);
            if (!ok) {
                fprintf(stderr, "Invalid --max-iters-m value\n");
                exit(2);
            }
        } else if (strcmp(a, "--max-size-mib") == 0) {
            int ok = 0;
            uint64_t v = parse_u64_value(require_value(argc, argv, &i), &ok);
            if (!ok) {
                fprintf(stderr, "Invalid --max-size-mib value\n");
                exit(2);
            }
            cfg->max_size_mib = (size_t)v;
        } else if (strcmp(a, "--x86-samples") == 0) {
            int ok = 0;
            cfg->x86_samples = parse_u64_value(require_value(argc, argv, &i), &ok);
            if (!ok) {
                fprintf(stderr, "Invalid --x86-samples value\n");
                exit(2);
            }
        } else if (strcmp(a, "--ratio") == 0) {
            int ok = 0;
            cfg->ratio = parse_double_value(require_value(argc, argv, &i), &ok);
            if (!ok || cfg->ratio <= 1.0) {
                fprintf(stderr, "Invalid --ratio value; it must be > 1.0\n");
                exit(2);
            }
        } else if (strcmp(a, "--cpu") == 0) {
            int ok = 0;
            uint64_t v = parse_u64_value(require_value(argc, argv, &i), &ok);
            if (!ok) {
                fprintf(stderr, "Invalid --cpu value\n");
                exit(2);
            }
            cfg->pin_cpu = (int)v;
        } else if (strcmp(a, "--metric") == 0) {
            const char* v = require_value(argc, argv, &i);
            if (strcmp(v, "p50") == 0 || strcmp(v, "50") == 0) {
                cfg->metric = METRIC_P50;
            } else if (strcmp(v, "p90") == 0 || strcmp(v, "90") == 0) {
                cfg->metric = METRIC_P90;
            } else if (strcmp(v, "p99") == 0 || strcmp(v, "99") == 0) {
                cfg->metric = METRIC_P99;
            } else {
                fprintf(stderr, "Invalid --metric value; use p50, p90, or p99\n");
                exit(2);
            }
        } else {
            fprintf(stderr, "Unknown option: %s\n", a);
            fprintf(stderr, "Use --help for usage.\n");
            exit(2);
        }
    }

    if (cfg->repeat_count < 3) {
        cfg->repeat_count = 3;
    }

    if ((cfg->repeat_count % 2) == 0) {
        cfg->repeat_count++;
    }

    if (cfg->min_iters_millions < 1) {
        cfg->min_iters_millions = 1;
    }

    if (cfg->max_iters_millions < cfg->min_iters_millions) {
        cfg->max_iters_millions = cfg->min_iters_millions;
    }

    if (cfg->x86_samples < 1000) {
        cfg->x86_samples = 1000;
    }

    if (cfg->max_size_mib < 1) {
        cfg->max_size_mib = 1;
    }
}

static void pin_to_cpu(int cpu) {
#if defined(__linux__)
    if (cpu < 0) {
        return;
    }

    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    (void)sched_setaffinity(0, sizeof(set), &set);
#else
    (void)cpu;
#endif
}

static uint64_t rng_next(uint64_t* state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 2685821657736338717ull;
}

static void shuffle_size_t(size_t* a, size_t n, uint64_t seed) {
    uint64_t s = seed ? seed : 0x123456789abcdef0ull;

    for (size_t i = n - 1; i > 0; --i) {
        size_t j = (size_t)(rng_next(&s) % (i + 1));
        size_t t = a[i];
        a[i] = a[j];
        a[j] = t;
    }
}

static int cmp_u64(const void* pa, const void* pb) {
    uint64_t a = *(const uint64_t*)pa;
    uint64_t b = *(const uint64_t*)pb;
    return (a > b) - (a < b);
}

static int cmp_double(const void* pa, const void* pb) {
    double a = *(const double*)pa;
    double b = *(const double*)pb;
    return (a > b) - (a < b);
}

static uint64_t sat_sub_u64(uint64_t a, uint64_t b) {
    return a > b ? a - b : 0;
}

static size_t percentile_idx(size_t n, unsigned p) {
    if (n == 0) {
        return 0;
    }

    size_t idx = (size_t)(((uint64_t)n * p + 99) / 100);

    if (idx == 0) {
        idx = 1;
    }

    idx -= 1;

    if (idx >= n) {
        idx = n - 1;
    }

    return idx;
}

#if ARCH_X86_64

static uint64_t x86_measure_empty_cycles(void) {
    unsigned aux;
    uint64_t t0, t1;

    _mm_mfence();
    _mm_lfence();

    t0 = __rdtscp(&aux);
    _mm_lfence();

    __asm__ __volatile__("" ::: "memory");

    _mm_lfence();
    t1 = __rdtscp(&aux);
    _mm_lfence();

    return t1 - t0;
}

static uint64_t x86_measure_load_cycles(volatile uint64_t* p) {
    unsigned aux;
    uint64_t t0, t1;
    uint64_t v;

    _mm_mfence();
    _mm_lfence();

    t0 = __rdtscp(&aux);
    _mm_lfence();

    v = *p;

    _mm_lfence();
    t1 = __rdtscp(&aux);
    _mm_lfence();

    sink_u64 += v;
    return t1 - t0;
}

static void x86_clflush_test(uint64_t samples, OutputMode mode) {
    uint64_t* p = NULL;
    uint64_t* overhead = malloc(sizeof(uint64_t) * samples);
    uint64_t* cached = malloc(sizeof(uint64_t) * samples);
    uint64_t* flushed = malloc(sizeof(uint64_t) * samples);

    if (!overhead || !cached || !flushed) {
        fprintf(stderr, "malloc failed\n");
        exit(1);
    }

    if (posix_memalign((void**)&p, CACHE_LINE, CACHE_LINE) != 0) {
        fprintf(stderr, "posix_memalign failed\n");
        exit(1);
    }

    *p = 0x1122334455667788ull;

    for (int i = 0; i < 10000; ++i) {
        sink_u64 += *p;
    }

    for (uint64_t i = 0; i < samples; ++i) {
        overhead[i] = x86_measure_empty_cycles();

        sink_u64 += *p;
        cached[i] = x86_measure_load_cycles((volatile uint64_t*)p);

        _mm_clflush((const void*)p);
        _mm_mfence();

        flushed[i] = x86_measure_load_cycles((volatile uint64_t*)p);
    }

    qsort(overhead, samples, sizeof(uint64_t), cmp_u64);
    qsort(cached, samples, sizeof(uint64_t), cmp_u64);
    qsort(flushed, samples, sizeof(uint64_t), cmp_u64);

    uint64_t oh_med = overhead[samples / 2];

    uint64_t cached_p50_raw = cached[samples / 2];
    uint64_t cached_p90_raw = cached[percentile_idx((size_t)samples, 90)];
    uint64_t cached_p99_raw = cached[percentile_idx((size_t)samples, 99)];

    uint64_t miss_p50_raw = flushed[samples / 2];
    uint64_t miss_p90_raw = flushed[percentile_idx((size_t)samples, 90)];
    uint64_t miss_p99_raw = flushed[percentile_idx((size_t)samples, 99)];

    uint64_t cached_p50 = sat_sub_u64(cached_p50_raw, oh_med);
    uint64_t miss_p50 = sat_sub_u64(miss_p50_raw, oh_med);

    if (mode == OUTPUT_CSV) {
        printf(
            "x86_clflush,samples,overhead_cycles,cached_p50_raw,cached_p90_raw,cached_p99_raw,miss_p50_raw,miss_p90_"
            "raw,miss_p99_raw,cached_p50_adj,miss_p50_adj,extra_miss_cycles\n");
        printf("x86_clflush,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64
               ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
               samples, oh_med, cached_p50_raw, cached_p90_raw, cached_p99_raw, miss_p50_raw, miss_p90_raw,
               miss_p99_raw, cached_p50, miss_p50, sat_sub_u64(miss_p50, cached_p50));
    } else if (mode == OUTPUT_VERBOSE) {
        printf("\n=== x86_64 explicit clflush test ===\n");
        printf("samples:                    %" PRIu64 "\n", samples);
        printf("median timer overhead:      %" PRIu64 " cycles\n", oh_med);
        printf("cached load p50 raw:        %" PRIu64 " cycles\n", cached_p50_raw);
        printf("cached load p90 raw:        %" PRIu64 " cycles\n", cached_p90_raw);
        printf("cached load p99 raw:        %" PRIu64 " cycles\n", cached_p99_raw);
        printf("flushed load p50 raw:       %" PRIu64 " cycles\n", miss_p50_raw);
        printf("flushed load p90 raw:       %" PRIu64 " cycles\n", miss_p90_raw);
        printf("flushed load p99 raw:       %" PRIu64 " cycles\n", miss_p99_raw);
        printf("cached load p50 adjusted:   %" PRIu64 " cycles\n", cached_p50);
        printf("cache-miss p50 adjusted:    %" PRIu64 " cycles\n", miss_p50);
        printf("extra miss penalty approx:  %" PRIu64 " cycles\n", sat_sub_u64(miss_p50, cached_p50));
    }

    free(p);
    free(overhead);
    free(cached);
    free(flushed);
}

#else

static void x86_clflush_test(uint64_t samples, OutputMode mode) {
    (void)samples;

    if (mode == OUTPUT_VERBOSE) {
        printf("\n=== x86_64 explicit clflush test ===\n");
        printf("not available on this architecture\n");
    }
}

#endif

#if ARCH_AARCH64

static uint64_t arm_read_cntvct(void) {
    uint64_t v;
    __asm__ __volatile__("isb; mrs %0, cntvct_el0" : "=r"(v));
    return v;
}

static uint64_t arm_read_cntfrq(void) {
    uint64_t v;
    __asm__ __volatile__("mrs %0, cntfrq_el0" : "=r"(v));
    return v;
}

static void arm_timer_info(OutputMode mode) {
    if (mode != OUTPUT_VERBOSE) {
        return;
    }

    uint64_t f = arm_read_cntfrq();
    uint64_t t0 = arm_read_cntvct();
    uint64_t n0 = now_ns();

    while (now_ns() - n0 < 1000000ull) {
        __asm__ __volatile__("" ::: "memory");
    }

    uint64_t t1 = arm_read_cntvct();

    printf("\n=== ARMv8 generic timer ===\n");
    printf("cntfrq_el0:                 %" PRIu64 " Hz\n", f);
    printf("ticks in about 1 ms:        %" PRIu64 "\n", t1 - t0);
    printf("Note: direct cache-line flush from user-space is often unavailable on ARMv8/Linux.\n");
}

#else

static void arm_timer_info(OutputMode mode) {
    (void)mode;
}

#endif

static Node* make_pointer_chase_ring(size_t bytes, size_t* out_nodes, Node** out_base) {
    size_t n = bytes / sizeof(Node);

    if (n < 2) {
        n = 2;
    }

    Node* nodes = NULL;
    size_t* idx = malloc(n * sizeof(size_t));

    if (!idx) {
        fprintf(stderr, "malloc idx failed\n");
        exit(1);
    }

    if (posix_memalign((void**)&nodes, CACHE_LINE, n * sizeof(Node)) != 0) {
        fprintf(stderr, "posix_memalign nodes failed\n");
        exit(1);
    }

    memset(nodes, 0, n * sizeof(Node));

    for (size_t i = 0; i < n; ++i) {
        idx[i] = i;
    }

    shuffle_size_t(idx, n, 0x5eed12345678abcdull ^ (uint64_t)n);

    for (size_t i = 0; i < n; ++i) {
        nodes[idx[i]].next = &nodes[idx[(i + 1) % n]];
    }

    Node* start = &nodes[idx[0]];

    free(idx);

    *out_nodes = n;
    *out_base = nodes;

    return start;
}

static double measure_pointer_chase_ns_per_load(Node* start, uint64_t iters) {
    Node* p = start;

    for (uint64_t i = 0; i < 1000000ull; ++i) {
        p = *(Node* volatile*)&p->next;
    }

    uint64_t t0 = now_ns();

    for (uint64_t i = 0; i < iters; ++i) {
        p = *(Node* volatile*)&p->next;
    }

    uint64_t t1 = now_ns();

    sink_node = p;

    return (double)(t1 - t0) / (double)iters;
}

static size_t build_working_set_sizes(size_t max_size_mib, size_t* out, size_t capacity) {
    const size_t defaults[] = {32ull * 1024ull,
                               256ull * 1024ull,
                               1ull * 1024ull * 1024ull,
                               4ull * 1024ull * 1024ull,
                               16ull * 1024ull * 1024ull,
                               64ull * 1024ull * 1024ull,
                               128ull * 1024ull * 1024ull,
                               256ull * 1024ull * 1024ull,
                               512ull * 1024ull * 1024ull,
                               1024ull * 1024ull * 1024ull};

    size_t max_bytes = max_size_mib * 1024ull * 1024ull;
    size_t n = 0;

    for (size_t i = 0; i < sizeof(defaults) / sizeof(defaults[0]); ++i) {
        if (defaults[i] <= max_bytes && n < capacity) {
            out[n++] = defaults[i];
        }
    }

    if (n == 0 && capacity > 0) {
        out[n++] = 32ull * 1024ull;
    }

    return n;
}

static void run_pointer_chase(const Config* cfg, ChaseResult* results, size_t* out_count) {
    size_t sizes[16];
    size_t count = build_working_set_sizes(cfg->max_size_mib, sizes, sizeof(sizes) / sizeof(sizes[0]));

    const uint64_t min_iters = cfg->min_iters_millions * 1000000ull;
    const uint64_t max_iters = cfg->max_iters_millions * 1000000ull;

    double* reps = malloc(sizeof(double) * cfg->repeat_count);

    if (!reps) {
        fprintf(stderr, "malloc reps failed\n");
        exit(1);
    }

    for (size_t si = 0; si < count; ++si) {
        size_t nodes_count = 0;
        Node* base = NULL;
        Node* start = make_pointer_chase_ring(sizes[si], &nodes_count, &base);

        uint64_t iters = (uint64_t)nodes_count * 64ull;

        if (iters < min_iters) {
            iters = min_iters;
        }

        if (iters > max_iters) {
            iters = max_iters;
        }

        for (uint64_t r = 0; r < cfg->repeat_count; ++r) {
            reps[r] = measure_pointer_chase_ns_per_load(start, iters);
        }

        qsort(reps, cfg->repeat_count, sizeof(double), cmp_double);

        results[si].bytes = sizes[si];
        results[si].nodes = nodes_count;
        results[si].iterations = iters;
        results[si].p50_ns = reps[percentile_idx((size_t)cfg->repeat_count, 50)];
        results[si].p90_ns = reps[percentile_idx((size_t)cfg->repeat_count, 90)];
        results[si].p99_ns = reps[percentile_idx((size_t)cfg->repeat_count, 99)];

        free(base);
    }

    free(reps);
    *out_count = count;
}

static double result_metric(const ChaseResult* r, MetricKind metric) {
    switch (metric) {
        case METRIC_P50:
            return r->p50_ns;
        case METRIC_P90:
            return r->p90_ns;
        case METRIC_P99:
            return r->p99_ns;
        default:
            return r->p50_ns;
    }
}

static const char* metric_name(MetricKind metric) {
    switch (metric) {
        case METRIC_P50:
            return "p50";
        case METRIC_P90:
            return "p90";
        case METRIC_P99:
            return "p99";
        default:
            return "p50";
    }
}

static int estimate_ram_minus_llc(const ChaseResult* results, size_t count, MetricKind metric, double ratio,
                                  double* out_ram_ns, double* out_llc_ns, double* out_penalty_ns, size_t* out_ram_idx,
                                  size_t* out_llc_idx) {
    if (count < 2) {
        return 0;
    }

    size_t ram_idx = 0;
    double ram_ns = result_metric(&results[0], metric);

    for (size_t i = 1; i < count; ++i) {
        double v = result_metric(&results[i], metric);

        if (v > ram_ns) {
            ram_ns = v;
            ram_idx = i;
        }
    }

    int found = 0;
    size_t llc_idx = 0;
    double llc_ns = 0.0;

    for (size_t i = 0; i < count; ++i) {
        double v = result_metric(&results[i], metric);

        if (i == ram_idx || v <= 0.0) {
            continue;
        }

        if ((ram_ns / v) > ratio) {
            if (!found || v > llc_ns) {
                found = 1;
                llc_ns = v;
                llc_idx = i;
            }
        }
    }

    if (!found) {
        return 0;
    }

    *out_ram_ns = ram_ns;
    *out_llc_ns = llc_ns;
    *out_penalty_ns = ram_ns - llc_ns;
    *out_ram_idx = ram_idx;
    *out_llc_idx = llc_idx;

    return 1;
}

static void print_verbose_results(const Config* cfg, const ChaseResult* results, size_t count, int estimate_ok,
                                  double ram_ns, double llc_ns, double penalty_ns, size_t ram_idx, size_t llc_idx) {
    printf("\n=== dependent pointer-chasing latency ===\n");
    printf("repeat count: %" PRIu64 "\n", cfg->repeat_count);
    printf("min iterations: %" PRIu64 " million\n", cfg->min_iters_millions);
    printf("max iterations: %" PRIu64 " million\n", cfg->max_iters_millions);
    printf("max working set: %zu MiB\n", cfg->max_size_mib);
    printf("boundary ratio: %.3f\n", cfg->ratio);
    printf("final metric: %s\n", metric_name(cfg->metric));
    printf("\n");
    printf("Working set       Nodes        Iterations      p50 ns/load   p90 ns/load   p99 ns/load\n");

    for (size_t i = 0; i < count; ++i) {
        printf("%10.3f MiB  %10zu  %14" PRIu64 "  %12.3f  %12.3f  %12.3f\n",
               (double)results[i].bytes / (1024.0 * 1024.0), results[i].nodes, results[i].iterations, results[i].p50_ns,
               results[i].p90_ns, results[i].p99_ns);
    }

    if (estimate_ok) {
        printf("\n=== estimate ===\n");
        printf("RAM candidate:    %.3f ns/load at %.3f MiB\n", ram_ns,
               (double)results[ram_idx].bytes / (1024.0 * 1024.0));
        printf("LLC candidate:    %.3f ns/load at %.3f MiB\n", llc_ns,
               (double)results[llc_idx].bytes / (1024.0 * 1024.0));
        printf("RAM - LLC:        %.3f ns\n", penalty_ns);
    } else {
        printf("\nCould not find LLC candidate with RAM / candidate > %.3f\n", cfg->ratio);
    }
}

static void print_csv_results(const Config* cfg, const ChaseResult* results, size_t count, int estimate_ok,
                              double ram_ns, double llc_ns, double penalty_ns, size_t ram_idx, size_t llc_idx) {
    printf("type,working_set_mib,nodes,iterations,p50_ns,p90_ns,p99_ns,metric\n");

    for (size_t i = 0; i < count; ++i) {
        printf("result,%.6f,%zu,%" PRIu64 ",%.6f,%.6f,%.6f,%s\n", (double)results[i].bytes / (1024.0 * 1024.0),
               results[i].nodes, results[i].iterations, results[i].p50_ns, results[i].p90_ns, results[i].p99_ns,
               metric_name(cfg->metric));
    }

    if (estimate_ok) {
        printf("estimate,%.6f,%zu,0,%.6f,%.6f,%.6f,RAM\n", (double)results[ram_idx].bytes / (1024.0 * 1024.0),
               results[ram_idx].nodes, ram_ns, ram_ns, ram_ns);
        printf("estimate,%.6f,%zu,0,%.6f,%.6f,%.6f,LLC\n", (double)results[llc_idx].bytes / (1024.0 * 1024.0),
               results[llc_idx].nodes, llc_ns, llc_ns, llc_ns);
        printf("summary,0,0,0,%.6f,%.6f,%.6f,ram_minus_llc_ns\n", penalty_ns, penalty_ns, penalty_ns);
    } else {
        printf("summary,0,0,0,nan,nan,nan,no_candidate_for_ratio_%.6f\n", cfg->ratio);
    }
}

int main(int argc, char** argv) {
    Config cfg;
    parse_args(argc, argv, &cfg);

    pin_to_cpu(cfg.pin_cpu);

    arm_timer_info(cfg.output_mode);

    if (cfg.run_x86_clflush) {
        x86_clflush_test(cfg.x86_samples, cfg.output_mode);
    }

    ChaseResult results[16];
    size_t result_count = 0;

    run_pointer_chase(&cfg, results, &result_count);

    double ram_ns = 0.0;
    double llc_ns = 0.0;
    double penalty_ns = 0.0;
    size_t ram_idx = 0;
    size_t llc_idx = 0;

    int estimate_ok = estimate_ram_minus_llc(results, result_count, cfg.metric, cfg.ratio, &ram_ns, &llc_ns,
                                             &penalty_ns, &ram_idx, &llc_idx);

    if (cfg.output_mode == OUTPUT_VALUE_ONLY) {
        if (estimate_ok) {
            printf("%.3f\n", penalty_ns);
        } else {
            printf("nan\n");
            return 3;
        }
    } else if (cfg.output_mode == OUTPUT_VERBOSE) {
        print_verbose_results(&cfg, results, result_count, estimate_ok, ram_ns, llc_ns, penalty_ns, ram_idx, llc_idx);
    } else if (cfg.output_mode == OUTPUT_CSV) {
        print_csv_results(&cfg, results, result_count, estimate_ok, ram_ns, llc_ns, penalty_ns, ram_idx, llc_idx);
    }

    if (sink_u64 == 0xdeadbeefull) {
        printf("%" PRIu64 "\n", sink_u64);
    }

    if (sink_node == (void*)0xdeadbeef) {
        printf("%p\n", (void*)sink_node);
    }

    return 0;
}