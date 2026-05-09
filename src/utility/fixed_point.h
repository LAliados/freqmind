#ifndef FXP48_16_H
#define FXP48_16_H

#include <linux/kernel.h>
#include <linux/types.h>

typedef struct {
    s64 _val;
} fxp48_16_t;

typedef fxp48_16_t fxp_t;

#define FXP48_16_FRAC_BITS 16
#define FXP48_16_ONE ((s64)1 << FXP48_16_FRAC_BITS)

/*
 * Q48.16 signed fixed-point:
 *
 *   real_value = _val / 2^16
 *
 * Переполнение и деление на ноль здесь не проверяются.
 * Для fixed-point умножения и деления используется __int128.
 */

#ifndef __SIZEOF_INT128__
#error "fxp48_16_t requires __int128 support"
#endif

#define FXP48_16_RAW(x) ((fxp48_16_t){._val = (s64)(x)})

/* conversions */

static inline fxp48_16_t fxp48_16_from_int(s64 x) {
    return FXP48_16_RAW(x * FXP48_16_ONE);
}

static inline int fxp48_16_to_int(fxp48_16_t x) {
    return (int)(x._val / FXP48_16_ONE);
}

static inline s64 fxp48_16_to_s64(fxp48_16_t x) {
    return x._val / FXP48_16_ONE;
}

static inline s64 fxp48_16_raw(fxp48_16_t x) {
    return x._val;
}

/* fixed <op> fixed */

static inline fxp48_16_t fxp48_16_add(fxp48_16_t a, fxp48_16_t b) {
    return FXP48_16_RAW(a._val + b._val);
}

static inline fxp48_16_t fxp48_16_sub(fxp48_16_t a, fxp48_16_t b) {
    return FXP48_16_RAW(a._val - b._val);
}

static inline fxp48_16_t fxp48_16_mul(fxp48_16_t a, fxp48_16_t b) {
    __int128 v;

    v = (__int128)a._val * (__int128)b._val;
    v /= FXP48_16_ONE;

    return FXP48_16_RAW((s64)v);
}

static inline fxp48_16_t fxp48_16_div(fxp48_16_t a, fxp48_16_t b) {
    __int128 v;

    v = (__int128)a._val * FXP48_16_ONE;
    v /= b._val;

    return FXP48_16_RAW((s64)v);
}

/* fixed <op> int */

static inline fxp48_16_t fxp48_16_add_int(fxp48_16_t a, s64 b) {
    return fxp48_16_add(a, fxp48_16_from_int(b));
}

static inline fxp48_16_t fxp48_16_sub_int(fxp48_16_t a, s64 b) {
    return fxp48_16_sub(a, fxp48_16_from_int(b));
}

static inline fxp48_16_t fxp48_16_mul_int(fxp48_16_t a, s64 b) {
    return FXP48_16_RAW(a._val * b);
}

static inline fxp48_16_t fxp48_16_div_int(fxp48_16_t a, s64 b) {
    return FXP48_16_RAW(a._val / b);
}

/* int <op> fixed */

static inline fxp48_16_t fxp48_16_int_add(s64 a, fxp48_16_t b) {
    return fxp48_16_add(fxp48_16_from_int(a), b);
}

static inline fxp48_16_t fxp48_16_int_sub(s64 a, fxp48_16_t b) {
    return fxp48_16_sub(fxp48_16_from_int(a), b);
}

static inline fxp48_16_t fxp48_16_int_mul(s64 a, fxp48_16_t b) {
    return FXP48_16_RAW(a * b._val);
}

static inline fxp48_16_t fxp48_16_int_div(s64 a, fxp48_16_t b) {
    return fxp48_16_div(fxp48_16_from_int(a), b);
}

/* full macro API */

#define FXP48_16_FROM_INT(x) fxp48_16_from_int((x))
#define FXP48_16_TO_INT(x) fxp48_16_to_int((x))
#define FXP48_16_TO_S64(x) fxp48_16_to_s64((x))
#define FXP48_16_RAW_VALUE(x) fxp48_16_raw((x))

#define FXP48_16_ADD(a, b) fxp48_16_add((a), (b))
#define FXP48_16_SUB(a, b) fxp48_16_sub((a), (b))
#define FXP48_16_MUL(a, b) fxp48_16_mul((a), (b))
#define FXP48_16_DIV(a, b) fxp48_16_div((a), (b))

#define FXP48_16_ADD_INT(a, b) fxp48_16_add_int((a), (b))
#define FXP48_16_SUB_INT(a, b) fxp48_16_sub_int((a), (b))
#define FXP48_16_MUL_INT(a, b) fxp48_16_mul_int((a), (b))
#define FXP48_16_DIV_INT(a, b) fxp48_16_div_int((a), (b))

#define FXP48_16_INT_ADD(a, b) fxp48_16_int_add((a), (b))
#define FXP48_16_INT_SUB(a, b) fxp48_16_int_sub((a), (b))
#define FXP48_16_INT_MUL(a, b) fxp48_16_int_mul((a), (b))
#define FXP48_16_INT_DIV(a, b) fxp48_16_int_div((a), (b))


/* uppercase short alias API */

#define FXP_RAW(x) FXP48_16_RAW_VALUE((x))
#define FXP_FROM_INT(x) FXP48_16_FROM_INT((x))
#define FXP_TO_INT(x) FXP48_16_TO_INT((x))
#define FXP_TO_S64(x) FXP48_16_TO_S64((x))

#define FXP_ADD(a, b) FXP48_16_ADD((a), (b))
#define FXP_SUB(a, b) FXP48_16_SUB((a), (b))
#define FXP_MUL(a, b) FXP48_16_MUL((a), (b))
#define FXP_DIV(a, b) FXP48_16_DIV((a), (b))

#define FXP_ADD_INT(a, b) FXP48_16_ADD_INT((a), (b))
#define FXP_SUB_INT(a, b) FXP48_16_SUB_INT((a), (b))
#define FXP_MUL_INT(a, b) FXP48_16_MUL_INT((a), (b))
#define FXP_DIV_INT(a, b) FXP48_16_DIV_INT((a), (b))

#define FXP_INT_ADD(a, b) FXP48_16_INT_ADD((a), (b))
#define FXP_INT_SUB(a, b) FXP48_16_INT_SUB((a), (b))
#define FXP_INT_MUL(a, b) FXP48_16_INT_MUL((a), (b))
#define FXP_INT_DIV(a, b) FXP48_16_INT_DIV((a), (b))
#define FXP_DEC_SCALE 100000

static inline void fxp_pr_info(const char* name, fxp_t x) {
    s64 raw = x._val;
    s64 ipart;
    s64 rem;
    u64 fpart;

    ipart = raw / FXP48_16_ONE;
    rem = raw % FXP48_16_ONE;

    if (rem < 0)
        rem = -rem;

    fpart = ((u64)rem * FXP_DEC_SCALE) / FXP48_16_ONE;

    if (raw < 0 && ipart == 0)
        pr_info("%s = -0.%05llu\n", name, fpart);
    else
        pr_info("%s = %lld.%05llu\n", name, (long long)ipart, (unsigned long long)fpart);
}
#endif /* FXP48_16_H */