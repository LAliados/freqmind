#ifndef FXP32_32_H
#define FXP32_32_H

#include <linux/types.h>

typedef struct {
    s64 _val;
} fxp32_32_t;

#define FXP32_32_FRAC_BITS 32
#define FXP32_32_ONE ((s64)1 << FXP32_32_FRAC_BITS)

/*
 * Q32.32 signed fixed-point:
 *
 *   real_value = _val / 2^32
 *
 * Переполнение и деление на ноль здесь не проверяются.
 * Для умножения и деления используется __int128.
 */

#ifndef __SIZEOF_INT128__
#error "fxp32_32_t requires __int128 support"
#endif

#define FXP32_32_RAW(x) ((fxp32_32_t){._val = (s64)(x)})

/* conversions */

static inline fxp32_32_t fxp32_32_from_int(s64 x) {
    return FXP32_32_RAW(x * FXP32_32_ONE);
}

static inline int fxp32_32_to_int(fxp32_32_t x) {
    return (int)(x._val / FXP32_32_ONE);
}

static inline s64 fxp32_32_raw(fxp32_32_t x) {
    return x._val;
}

/* fixed <op> fixed */

static inline fxp32_32_t fxp32_32_add(fxp32_32_t a, fxp32_32_t b) {
    return FXP32_32_RAW(a._val + b._val);
}

static inline fxp32_32_t fxp32_32_sub(fxp32_32_t a, fxp32_32_t b) {
    return FXP32_32_RAW(a._val - b._val);
}

static inline fxp32_32_t fxp32_32_mul(fxp32_32_t a, fxp32_32_t b) {
    __int128 v;

    v = (__int128)a._val * (__int128)b._val;
    v /= FXP32_32_ONE;

    return FXP32_32_RAW((s64)v);
}

static inline fxp32_32_t fxp32_32_div(fxp32_32_t a, fxp32_32_t b) {
    __int128 v;

    v = (__int128)a._val * FXP32_32_ONE;
    v /= b._val;

    return FXP32_32_RAW((s64)v);
}

/* fixed <op> int */

static inline fxp32_32_t fxp32_32_add_int(fxp32_32_t a, s64 b) {
    return fxp32_32_add(a, fxp32_32_from_int(b));
}

static inline fxp32_32_t fxp32_32_sub_int(fxp32_32_t a, s64 b) {
    return fxp32_32_sub(a, fxp32_32_from_int(b));
}

static inline fxp32_32_t fxp32_32_mul_int(fxp32_32_t a, s64 b) {
    return FXP32_32_RAW(a._val * b);
}

static inline fxp32_32_t fxp32_32_div_int(fxp32_32_t a, s64 b) {
    return FXP32_32_RAW(a._val / b);
}

/* int <op> fixed */

static inline fxp32_32_t fxp32_32_int_add(s64 a, fxp32_32_t b) {
    return fxp32_32_add(fxp32_32_from_int(a), b);
}

static inline fxp32_32_t fxp32_32_int_sub(s64 a, fxp32_32_t b) {
    return fxp32_32_sub(fxp32_32_from_int(a), b);
}

static inline fxp32_32_t fxp32_32_int_mul(s64 a, fxp32_32_t b) {
    return FXP32_32_RAW(a * b._val);
}

static inline fxp32_32_t fxp32_32_int_div(s64 a, fxp32_32_t b) {
    return fxp32_32_div(fxp32_32_from_int(a), b);
}

/* macro API */

#define FXP32_32_FROM_INT(x) fxp32_32_from_int((x))
#define FXP32_32_TO_INT(x) fxp32_32_to_int((x))
#define FXP32_32_RAW_VALUE(x) fxp32_32_raw((x))

#define FXP32_32_ADD(a, b) fxp32_32_add((a), (b))
#define FXP32_32_SUB(a, b) fxp32_32_sub((a), (b))
#define FXP32_32_MUL(a, b) fxp32_32_mul((a), (b))
#define FXP32_32_DIV(a, b) fxp32_32_div((a), (b))

#define FXP32_32_ADD_INT(a, b) fxp32_32_add_int((a), (b))
#define FXP32_32_SUB_INT(a, b) fxp32_32_sub_int((a), (b))
#define FXP32_32_MUL_INT(a, b) fxp32_32_mul_int((a), (b))
#define FXP32_32_DIV_INT(a, b) fxp32_32_div_int((a), (b))

#define FXP32_32_INT_ADD(a, b) fxp32_32_int_add((a), (b))
#define FXP32_32_INT_SUB(a, b) fxp32_32_int_sub((a), (b))
#define FXP32_32_INT_MUL(a, b) fxp32_32_int_mul((a), (b))
#define FXP32_32_INT_DIV(a, b) fxp32_32_int_div((a), (b))

/*
 * Short aliases.
 *
 * fxp == fxp32_32_t
 */

typedef fxp32_32_t fxp_t;

#define FXP_RAW(x) FXP32_32_RAW_VALUE((x))
#define FXP_FROM_INT(x) FXP32_32_FROM_INT((x))
#define FXP_TO_INT(x) FXP32_32_TO_INT((x))

#define FXP_ADD(a, b) FXP32_32_ADD((a), (b))
#define FXP_SUB(a, b) FXP32_32_SUB((a), (b))
#define FXP_MUL(a, b) FXP32_32_MUL((a), (b))
#define FXP_DIV(a, b) FXP32_32_DIV((a), (b))

#define FXP_ADD_INT(a, b) FXP32_32_ADD_INT((a), (b))
#define FXP_SUB_INT(a, b) FXP32_32_SUB_INT((a), (b))
#define FXP_MUL_INT(a, b) FXP32_32_MUL_INT((a), (b))
#define FXP_DIV_INT(a, b) FXP32_32_DIV_INT((a), (b))

#define FXP_INT_ADD(a, b) FXP32_32_INT_ADD((a), (b))
#define FXP_INT_SUB(a, b) FXP32_32_INT_SUB((a), (b))
#define FXP_INT_MUL(a, b) FXP32_32_INT_MUL((a), (b))
#define FXP_INT_DIV(a, b) FXP32_32_INT_DIV((a), (b))

#endif /* FXP32_32_H */