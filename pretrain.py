#!/usr/bin/env python3
import argparse
import re
import sys
import numpy as np
import pandas as pd
import tensorflow as tf


RAW_FEATURES = [
    "cpu_cycles",
    "instructions",
    "cache_misses",
    "stalls_forward",
    "branch_misses",
]

FXP_FRAC_BITS = 16
FXP_SCALE = 1 << FXP_FRAC_BITS
S64_MIN = -(1 << 63)
S64_MAX = (1 << 63) - 1


def target_function(cpu_cycles, instructions, cache_misses, stalls_forward, branch_misses):
    """
    Замени эту функцию на свою.

    Важно:
    target_function получает исходные значения из CSV,
    а не признаки, уже поделенные на cpu_cycles.
    """

    ipc = instructions / (cpu_cycles + 1.0)
    cache_miss_rate = cache_misses / (instructions + 1.0)
    branch_miss_rate = branch_misses / (instructions + 1.0)
    stall_rate = stalls_forward / (cpu_cycles + 1.0)
    min = 400000
    max = 4000000
    y = (
        (1-cache_misses/instructions)*(max-min) + min
    )

    return y.astype(np.float32)


def parse_int_list(s):
    values = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not values:
        raise ValueError("Empty layer dimension list")
    if any(v <= 0 for v in values):
        raise ValueError(f"All layer dimensions must be positive: {values}")
    return values


def parse_activations(s):
    values = [x.strip().lower() for x in s.split(",") if x.strip()]
    allowed = {"relu", "none"}

    bad = [x for x in values if x not in allowed]
    if bad:
        raise ValueError(f"Unsupported activations: {bad}. Allowed: relu, none")

    return values


def make_model_input_features(df):
    """
    Признаки для MLP:

        x[0] = instructions / cpu_cycles
        x[1] = cache_misses / cpu_cycles
        x[2] = stalls_forward / cpu_cycles
        x[3] = branch_misses / cpu_cycles

    Сам cpu_cycles в модель НЕ подаётся.
    Он используется только как делитель.
    """

    cpu_cycles = df["cpu_cycles"].to_numpy(dtype=np.float32)
    denom = np.where(cpu_cycles == 0.0, 1.0, cpu_cycles)

    instructions = df["instructions"].to_numpy(dtype=np.float32)
    cache_misses = df["cache_misses"].to_numpy(dtype=np.float32)
    stalls_forward = df["stalls_forward"].to_numpy(dtype=np.float32)
    branch_misses = df["branch_misses"].to_numpy(dtype=np.float32)

    x = np.stack(
        [
            instructions / denom,
            cache_misses / denom,
            stalls_forward / denom,
            branch_misses / denom,
        ],
        axis=1,
    )

    return x.astype(np.float32)


def build_model(input_dim, layer_dims, activations):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_dim,)))

    for i, (out_dim, activation) in enumerate(zip(layer_dims, activations)):
        keras_activation = None if activation == "none" else activation

        model.add(
            tf.keras.layers.Dense(
                out_dim,
                activation=keras_activation,
                name=f"dense_{i}",
            )
        )

    return model


def validate_c_identifier(name):
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
        raise ValueError(f"Invalid C identifier: {name}")


def float_to_q48_16_raw(x):
    if not np.isfinite(x):
        raise ValueError(f"Non-finite value during fxp export: {x}")

    raw = int(np.round(float(x) * FXP_SCALE))

    if raw < S64_MIN or raw > S64_MAX:
        raise OverflowError(f"Q48.16 raw value out of s64 range: value={x}, raw={raw}")

    return raw


def c_fxp_literal(x):
    raw = float_to_q48_16_raw(x)
    return f"FXP48_16_RAW((s64){raw}LL)"


def activation_to_c(activation):
    if activation == "relu":
        return "MLP_FXP_ACT_RELU"
    if activation == "none":
        return "MLP_FXP_ACT_NONE"
    raise ValueError(f"Unsupported activation: {activation}")


def export_c_for_mlp_fxp_ga(
    model,
    activations,
    x_mean,
    x_std,
    y_mean,
    y_std,
    c_prefix,
    file,
):
    validate_c_identifier(c_prefix)

    macro_prefix = c_prefix.upper()

    dense_layers = [
        layer for layer in model.layers
        if isinstance(layer, tf.keras.layers.Dense)
    ]

    if len(dense_layers) != len(activations):
        raise ValueError("Internal error: layer/activation count mismatch")

    weights = []
    biases = []

    for layer in dense_layers:
        kernel, bias = layer.get_weights()

        # TensorFlow Dense:
        #   kernel[in_dim][out_dim]
        #
        # mlp_fxp:
        #   weights[out_dim][in_dim]
        w = kernel.T.astype(np.float64)
        b = bias.astype(np.float64)

        weights.append(w)
        biases.append(b)

    # Fold input normalization into first layer.
    #
    # z = W * ((x - mean) / std) + b
    #   = (W / std) * x + (b - W * mean / std)
    original_w0 = weights[0].copy()

    weights[0] = weights[0] / x_std[None, :]
    biases[0] = biases[0] - np.sum(
        original_w0 * (x_mean / x_std)[None, :],
        axis=1,
    )

    # Fold output denormalization into final layer.
    #
    # y = y_norm * y_std + y_mean
    weights[-1] = weights[-1] * y_std
    biases[-1] = biases[-1] * y_std + y_mean

    layer_shapes = [(w.shape[1], w.shape[0]) for w in weights]
    n_layers = len(layer_shapes)
    input_dim = layer_shapes[0][0]
    output_dim = layer_shapes[-1][1]

    non_final_out_dims = [out_dim for _, out_dim in layer_shapes[:-1]]
    work_len = 0 if n_layers <= 1 else 2 * max(non_final_out_dims)

    param_offsets = []
    param_values = []

    offset = 0
    for w, b in zip(weights, biases):
        weight_offset = offset
        flat_w = w.reshape(-1)
        param_values.extend(flat_w)
        offset += flat_w.size

        bias_offset = offset
        param_values.extend(b.reshape(-1))
        offset += b.size

        param_offsets.append((weight_offset, bias_offset))

    param_count = len(param_values)
    guard = f"{macro_prefix}_PARAMS_H"

    print(f"#ifndef {guard}", file=file)
    print(f"#define {guard}", file=file)
    print("", file=file)
    print('#include "mlp/mlp.h"', file=file)
    print('#include "genetic/ga.h"', file=file)
    print("", file=file)

    print("/*", file=file)
    print(" * Auto-generated MLP parameters for mlp_fxp + GA fine-tuning.", file=file)
    print(" *", file=file)
    print(" * Input feature order:", file=file)
    print(" *   input[0] = instructions / cpu_cycles", file=file)
    print(" *   input[1] = cache_misses / cpu_cycles", file=file)
    print(" *   input[2] = stalls_forward / cpu_cycles", file=file)
    print(" *   input[3] = branch_misses / cpu_cycles", file=file)
    print(" *", file=file)
    print(" * Parameter layout:", file=file)
    print(" *   All weights and biases are stored sequentially in one mutable array.", file=file)
    print(" *   For each layer:", file=file)
    print(" *     1. weights[out_dim][in_dim], row-major", file=file)
    print(" *     2. bias[out_dim]", file=file)
    print(" *", file=file)
    print(" * Input normalization is folded into layer 0.", file=file)
    print(" * Output denormalization is folded into the final layer.", file=file)
    print(" */", file=file)
    print("", file=file)

    print("enum {", file=file)
    print(f"    {macro_prefix}_INPUT_DIM = {input_dim},", file=file)

    for i, (in_dim, out_dim) in enumerate(layer_shapes):
        print(f"    {macro_prefix}_LAYER_{i}_IN_DIM = {in_dim},", file=file)
        print(f"    {macro_prefix}_LAYER_{i}_OUT_DIM = {out_dim},", file=file)
        print(f"    {macro_prefix}_LAYER_{i}_WEIGHT_COUNT = {in_dim * out_dim},", file=file)
        print(f"    {macro_prefix}_LAYER_{i}_BIAS_COUNT = {out_dim},", file=file)
        print(f"    {macro_prefix}_LAYER_{i}_PARAM_COUNT = {in_dim * out_dim + out_dim},", file=file)

    print(f"    {macro_prefix}_OUTPUT_DIM = {output_dim},", file=file)
    print(f"    {macro_prefix}_NUM_LAYERS = {n_layers},", file=file)
    print(f"    {macro_prefix}_PARAM_COUNT = {param_count},", file=file)
    print(f"    {macro_prefix}_WORK_LEN = {work_len}", file=file)
    print("};", file=file)
    print("", file=file)

    print(f"enum {macro_prefix}_PARAM_OFFSETS {{", file=file)
    for i, (weight_offset, bias_offset) in enumerate(param_offsets):
        print(f"    {macro_prefix}_LAYER_{i}_WEIGHTS_OFFSET = {weight_offset},", file=file)
        print(f"    {macro_prefix}_LAYER_{i}_BIAS_OFFSET = {bias_offset},", file=file)
    print("};", file=file)
    print("", file=file)

    print(f"static fxp_t {c_prefix}_params[{macro_prefix}_PARAM_COUNT] = {{", file=file)

    idx = 0
    for layer_i, (w, b) in enumerate(zip(weights, biases)):
        print(f"    /* layer {layer_i} weights[out_dim][in_dim] */", file=file)
        flat_w = w.reshape(-1)

        for row_start in range(0, len(flat_w), 4):
            chunk = flat_w[row_start:row_start + 4]
            values = ", ".join(c_fxp_literal(v) for v in chunk)
            print(f"    {values},", file=file)
            idx += len(chunk)

        print(f"    /* layer {layer_i} bias[out_dim] */", file=file)
        flat_b = b.reshape(-1)

        for row_start in range(0, len(flat_b), 4):
            chunk = flat_b[row_start:row_start + 4]
            values = ", ".join(c_fxp_literal(v) for v in chunk)
            print(f"    {values},", file=file)
            idx += len(chunk)

    print("};", file=file)
    print("", file=file)

    print(f"static mlp_fxp_layer_t {c_prefix}_layers[{macro_prefix}_NUM_LAYERS] = {{", file=file)

    for i, (in_dim, out_dim) in enumerate(layer_shapes):
        weight_offset, bias_offset = param_offsets[i]
        act = activation_to_c(activations[i])

        print("    {", file=file)
        print(f"        .in_dim = {in_dim},", file=file)
        print(f"        .out_dim = {out_dim},", file=file)
        print(f"        .weights = &{c_prefix}_params[{weight_offset}],", file=file)
        print(f"        .bias = &{c_prefix}_params[{bias_offset}],", file=file)
        print(f"        .activation = {act},", file=file)
        print("    },", file=file)

    print("};", file=file)
    print("", file=file)

    print(f"static inline void {c_prefix}_set_params(const fxp_t* params)", file=file)
    print("{", file=file)
    print("    size_t i;", file=file)
    print("", file=file)
    print("    if (!params)", file=file)
    print("        return;", file=file)
    print("", file=file)
    print(f"    for (i = 0; i < {macro_prefix}_PARAM_COUNT; ++i)", file=file)
    print(f"        {c_prefix}_params[i] = params[i];", file=file)
    print("}", file=file)
    print("", file=file)

    print(f"static inline void {c_prefix}_copy_params(fxp_t* dst)", file=file)
    print("{", file=file)
    print("    size_t i;", file=file)
    print("", file=file)
    print("    if (!dst)", file=file)
    print("        return;", file=file)
    print("", file=file)
    print(f"    for (i = 0; i < {macro_prefix}_PARAM_COUNT; ++i)", file=file)
    print(f"        dst[i] = {c_prefix}_params[i];", file=file)
    print("}", file=file)
    print("", file=file)

    print(f"static inline void {c_prefix}_make_input(", file=file)
    print("    s64 cpu_cycles,", file=file)
    print("    s64 instructions,", file=file)
    print("    s64 cache_misses,", file=file)
    print("    s64 stalls_forward,", file=file)
    print("    s64 branch_misses,", file=file)
    print(f"    fxp_t input[{macro_prefix}_INPUT_DIM])", file=file)
    print("{", file=file)
    print("    s64 denom_cycles = cpu_cycles == 0 ? 1 : cpu_cycles;", file=file)
    print("    fxp_t denom = FXP48_16_FROM_INT(denom_cycles);", file=file)
    print("", file=file)
    print("    input[0] = FXP48_16_DIV(FXP48_16_FROM_INT(instructions), denom);", file=file)
    print("    input[1] = FXP48_16_DIV(FXP48_16_FROM_INT(cache_misses), denom);", file=file)
    print("    input[2] = FXP48_16_DIV(FXP48_16_FROM_INT(stalls_forward), denom);", file=file)
    print("    input[3] = FXP48_16_DIV(FXP48_16_FROM_INT(branch_misses), denom);", file=file)
    print("}", file=file)
    print("", file=file)

    print(f"static inline int {c_prefix}_predict_from_input(", file=file)
    print(f"    const fxp_t input[{macro_prefix}_INPUT_DIM],", file=file)
    print(f"    fxp_t output[{macro_prefix}_OUTPUT_DIM],", file=file)
    print(f"    fxp_t work[{macro_prefix}_WORK_LEN])", file=file)
    print("{", file=file)
    print("    return mlp_fxp_forward(", file=file)
    print(f"        {c_prefix}_layers,", file=file)
    print(f"        {macro_prefix}_NUM_LAYERS,", file=file)
    print("        input,", file=file)
    print("        output,", file=file)
    print("        work,", file=file)
    print(f"        {macro_prefix}_WORK_LEN);", file=file)
    print("}", file=file)
    print("", file=file)

    print(f"static inline int {c_prefix}_predict(", file=file)
    print("    s64 cpu_cycles,", file=file)
    print("    s64 instructions,", file=file)
    print("    s64 cache_misses,", file=file)
    print("    s64 stalls_forward,", file=file)
    print("    s64 branch_misses,", file=file)
    print(f"    fxp_t output[{macro_prefix}_OUTPUT_DIM],", file=file)
    print(f"    fxp_t work[{macro_prefix}_WORK_LEN])", file=file)
    print("{", file=file)
    print(f"    fxp_t input[{macro_prefix}_INPUT_DIM];", file=file)
    print("", file=file)
    print(f"    {c_prefix}_make_input(", file=file)
    print("        cpu_cycles,", file=file)
    print("        instructions,", file=file)
    print("        cache_misses,", file=file)
    print("        stalls_forward,", file=file)
    print("        branch_misses,", file=file)
    print("        input);", file=file)
    print("", file=file)
    print(f"    return {c_prefix}_predict_from_input(input, output, work);", file=file)
    print("}", file=file)
    print("", file=file)

    print(f"static inline void {c_prefix}_ga_default_config(ga_config* cfg)", file=file)
    print("{", file=file)
    print("    if (!cfg)", file=file)
    print("        return;", file=file)
    print("", file=file)
    print(f"    ga_default_config(cfg, {macro_prefix}_PARAM_COUNT);", file=file)
    print(f"    cfg->start_params = {c_prefix}_params;", file=file)
    print("}", file=file)
    print("", file=file)

    print(f"static inline int {c_prefix}_ga_init(struct ga* g, ga_config* cfg)", file=file)
    print("{", file=file)
    print("    if (!g || !cfg)", file=file)
    print("        return -EINVAL;", file=file)
    print("", file=file)
    print(f"    {c_prefix}_ga_default_config(cfg);", file=file)
    print(f"    return ga_init(g, {c_prefix}_params, cfg);", file=file)
    print("}", file=file)
    print("", file=file)

    print(f"#endif /* {guard} */", file=file)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("csv_path", help="Path to CSV dataset")

    parser.add_argument(
        "--layer-dims",
        default="16,8,1",
        help="Comma-separated output dimensions for all Dense layers, e.g. 16,8,1",
    )

    parser.add_argument(
        "--activations",
        default="relu,relu,none",
        help="Comma-separated activations for all layers: relu or none, e.g. relu,relu,none",
    )

    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--c-prefix",
        default="perf_mlp",
        help="Prefix for generated C symbols",
    )

    parser.add_argument(
        "--c-output",
        default=None,
        help="Optional path for generated C header. If omitted, C code is printed to stdout.",
    )

    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    validate_c_identifier(args.c_prefix)

    layer_dims = parse_int_list(args.layer_dims)
    activations = parse_activations(args.activations)

    if len(layer_dims) != len(activations):
        raise ValueError(
            "--layer-dims and --activations must have the same length. "
            f"Got {len(layer_dims)} dims and {len(activations)} activations."
        )

    if layer_dims[-1] != 1:
        raise ValueError(
            "The final layer must have out_dim=1 because the model output is one number. "
            f"Got final out_dim={layer_dims[-1]}"
        )

    df = pd.read_csv(args.csv_path)

    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    x = make_model_input_features(df)

    y = target_function(
        df["cpu_cycles"].to_numpy(dtype=np.float32),
        df["instructions"].to_numpy(dtype=np.float32),
        df["cache_misses"].to_numpy(dtype=np.float32),
        df["stalls_forward"].to_numpy(dtype=np.float32),
        df["branch_misses"].to_numpy(dtype=np.float32),
    ).reshape(-1, 1)

    x_mean = x.mean(axis=0).astype(np.float64)
    x_std = x.std(axis=0).astype(np.float64) + 1e-8

    y_mean = float(y.mean())
    y_std = float(y.std() + 1e-8)

    x_norm = ((x.astype(np.float64) - x_mean) / x_std).astype(np.float32)
    y_norm = ((y.astype(np.float64) - y_mean) / y_std).astype(np.float32)

    model = build_model(
        input_dim=x_norm.shape[1],
        layer_dims=layer_dims,
        activations=activations,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="mse",
        metrics=["mae"],
    )

    print("Training MLP...")
    print(f"Layer dims: {layer_dims}")
    print(f"Activations: {activations}")

    model.fit(
        x_norm,
        y_norm,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
        shuffle=True,
    )

    pred_norm = model.predict(x_norm, verbose=0)
    pred = pred_norm * y_std + y_mean

    mse = np.mean((pred - y) ** 2)
    mae = np.mean(np.abs(pred - y))

    print(f"\nFinal MSE in original scale: {mse:.8f}")
    print(f"Final MAE in original scale: {mae:.8f}")

    if args.c_output:
        with open(args.c_output, "w", encoding="utf-8") as f:
            export_c_for_mlp_fxp_ga(
                model=model,
                activations=activations,
                x_mean=x_mean,
                x_std=x_std,
                y_mean=y_mean,
                y_std=y_std,
                c_prefix=args.c_prefix,
                file=f,
            )

        print(f"\nWrote C export to: {args.c_output}")
    else:
        print("\n/* BEGIN GENERATED C EXPORT */\n")
        export_c_for_mlp_fxp_ga(
            model=model,
            activations=activations,
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
            c_prefix=args.c_prefix,
            file=sys.stdout,
        )
        print("\n/* END GENERATED C EXPORT */")


if __name__ == "__main__":
    main()