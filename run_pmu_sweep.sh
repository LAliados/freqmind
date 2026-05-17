#!/usr/bin/env bash
set -euo pipefail

# Sweep compiler -D parameters for pmu_asm_load.c, run each binary,
# sample PMU counters with perf stat every 5 ms, and write the sample
# closest to 25 ms to CSV.
#
# Usage:
#   sudo ./run_pmu_sweep.sh [source_c_file] [output_csv]
#
# Example:
#   sudo ./run_pmu_sweep.sh pmu_asm_load.c results.csv
#
# Important:
#   stalls_forward is not a generic perf event on all CPUs.
#   Override it if your machine uses a different event name:
#     sudo EVENT_STALLS_FORWARD='ld_blocks.store_forward:u' ./run_pmu_sweep.sh
#     sudo EVENT_STALLS_FORWARD='rNNN:u' ./run_pmu_sweep.sh

SOURCE_FILE="${1:-pmu_asm_load.c}"
OUT_CSV="${2:-pmu_results.csv}"
VARIANT_CSV="${VARIANT_CSV:-pmu_variants.csv}"
BUILD_DIR="${BUILD_DIR:-pmu_build}"
LOG_DIR="${LOG_DIR:-pmu_logs}"

CC="${CC:-gcc}"
CFLAGS="${CFLAGS:--O2 -std=c11 -Wall -Wextra -march=native}"

# Number of generated programs. Keep this around 100-150.
MAX_VARIANTS="${MAX_VARIANTS:-120}"

# Runtime / perf settings.
PIN_TO_CPU="${PIN_TO_CPU:-2}"
RUN_WARMUP_SECONDS="${RUN_WARMUP_SECONDS:-0.20}"
PERF_INTERVAL_MS="${PERF_INTERVAL_MS:-60}"
TARGET_MS="${TARGET_MS:-120}"
PERF_INTERVAL_COUNT="${PERF_INTERVAL_COUNT:-3}"

# PMU events. The CSV header is fixed to the user's requested format.
EVENT_CYCLES="${EVENT_CYCLES:-cycles:u}"
EVENT_INSTRUCTIONS="${EVENT_INSTRUCTIONS:-instructions:u}"
EVENT_CACHE_MISSES="${EVENT_CACHE_MISSES:-cache-misses:u}"
EVENT_STALLS_FORWARD="${EVENT_STALLS_FORWARD:-stalled-cycles-frontend:u}"
EVENT_BRANCH_MISSES="${EVENT_BRANCH_MISSES:-branch-misses:u}"
PERF_EVENTS="${EVENT_CYCLES},${EVENT_INSTRUCTIONS},${EVENT_CACHE_MISSES},${EVENT_STALLS_FORWARD},${EVENT_BRANCH_MISSES}"

# Sweep grid. Edit these arrays to change the generated workloads.
WORKING_SET_BYTES_LIST=(
  $((256 * 1024))
  $((1 * 1024 * 1024))
  $((8 * 1024 * 1024))
  $((64 * 1024 * 1024))
)
MEM_STRIDE_WORDS_LIST=(8 16 64 256)
ASM_ALU_GROUPS_PER_ROUND_LIST=(1 8 32 64)
ASM_IMULS_PER_ROUND_LIST=(0 2 8)
ASM_IDIVS_PER_ROUND_LIST=(0 1 2)
ASM_LOADS_PER_ROUND_LIST=(4 16 64 128)
ASM_STORES_PER_ROUND_LIST=(0 4 16)
ASM_BRANCHES_PER_ROUND_LIST=(0 4 16)
ASM_NOPS_PER_ROUND_LIST=(0 8 32)
ROUNDS_PER_BATCH_LIST=(512 2048 8192)

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "error: required command not found: $1" >&2
    exit 1
  fi
}

check_perf_event() {
  local event="$1"
  local log_file="$2"

  # This may fail because of permissions or because the event is not supported.
  # The main reason for checking here is to catch unsupported stalls_forward early.
  if ! perf stat -e "$event" -- true >/dev/null 2>"$log_file"; then
    echo "error: perf event does not work: $event" >&2
    echo "perf output:" >&2
    sed -n '1,40p' "$log_file" >&2
    echo >&2
    echo "Set the correct event with, for example:" >&2
    echo "  EVENT_STALLS_FORWARD='your_event:u' sudo ./run_pmu_sweep.sh $SOURCE_FILE $OUT_CSV" >&2
    exit 1
  fi
}

compile_variant() {
  local bin="$1"
  local working_set_bytes="$2"
  local mem_stride_words="$3"
  local alu_groups="$4"
  local imuls="$5"
  local idivs="$6"
  local loads="$7"
  local stores="$8"
  local branches="$9"
  local nops="${10}"
  local rounds="${11}"

  # shellcheck disable=SC2086
  "$CC" $CFLAGS \
    -DPIN_TO_CPU="$PIN_TO_CPU" \
    -DWORKING_SET_BYTES="$working_set_bytes" \
    -DMEM_STRIDE_WORDS="$mem_stride_words" \
    -DASM_ALU_GROUPS_PER_ROUND="$alu_groups" \
    -DASM_IMULS_PER_ROUND="$imuls" \
    -DASM_IDIVS_PER_ROUND="$idivs" \
    -DASM_LOADS_PER_ROUND="$loads" \
    -DASM_STORES_PER_ROUND="$stores" \
    -DASM_BRANCHES_PER_ROUND="$branches" \
    -DASM_NOPS_PER_ROUND="$nops" \
    -DROUNDS_PER_BATCH="$rounds" \
    "$SOURCE_FILE" -o "$bin"
}

run_and_measure() {
  local bin="$1"
  local perf_file="$2"

  "$bin" >"${perf_file}.program.stdout" 2>"${perf_file}.program.stderr" &
  local pid=$!

  sleep "$RUN_WARMUP_SECONDS"

  # perf stat writes statistics to stderr.
  set +e
  perf stat \
    -x ';' \
    -I "$PERF_INTERVAL_MS" \
    --interval-count "$PERF_INTERVAL_COUNT" \
    -p "$pid" \
    -e "$PERF_EVENTS" \
    >/dev/null 2>"$perf_file"
  local perf_rc=$?
  set -e

  kill -TERM "$pid" >/dev/null 2>&1 || true
  wait "$pid" >/dev/null 2>&1 || true

  if [[ "$perf_rc" -ne 0 ]]; then
    echo "warning: perf returned $perf_rc for $bin; see $perf_file" >&2
  fi
}

extract_target_sample() {
  local perf_file="$1"

  awk -F ';' \
    -v target_ms="$TARGET_MS" \
    -v e_cycles="$EVENT_CYCLES" \
    -v e_instructions="$EVENT_INSTRUCTIONS" \
    -v e_cache_misses="$EVENT_CACHE_MISSES" \
    -v e_stalls_forward="$EVENT_STALLS_FORWARD" \
    -v e_branch_misses="$EVENT_BRANCH_MISSES" '
    function trim(s) {
      gsub(/^[ \t]+|[ \t]+$/, "", s)
      return s
    }

    function bare_event(s) {
      sub(/:.*/, "", s)
      return s
    }

    function event_match(actual, expected) {
      actual = trim(actual)
      expected = trim(expected)
      return actual == expected || actual == bare_event(expected) || bare_event(actual) == bare_event(expected)
    }

    function clean_value(v) {
      v = trim(v)
      gsub(/,/, "", v)
      if (v == "" || v ~ /not supported/ || v ~ /not counted/ || v ~ /<not/) {
        return "nan"
      }
      return v
    }

    function abs(x) {
      return x < 0 ? -x : x
    }

    BEGIN {
      target_s = target_ms / 1000.0
      best_diff = 1e100
      best_key = ""
    }

    # Expected perf -x output shape with interval:
    # timestamp;counter;unit;event;run_percent;...
    $1 ~ /^[[:space:]]*[0-9]+(\.[0-9]+)?[[:space:]]*$/ {
      t = trim($1) + 0.0
      event = trim($4)
      value = clean_value($2)
      key = trim($1)

      diff = abs(t - target_s)
      if (diff < best_diff) {
        best_diff = diff
        best_key = key
      }

      if (event_match(event, e_cycles)) {
        cycles[key] = value
      } else if (event_match(event, e_instructions)) {
        instructions[key] = value
      } else if (event_match(event, e_cache_misses)) {
        cache_misses[key] = value
      } else if (event_match(event, e_stalls_forward)) {
        stalls_forward[key] = value
      } else if (event_match(event, e_branch_misses)) {
        branch_misses[key] = value
      }
    }

    END {
      if (best_key == "") {
        print "nan,nan,nan,nan,nan"
        exit 0
      }

      c  = (best_key in cycles)         ? cycles[best_key]         : "nan"
      i  = (best_key in instructions)   ? instructions[best_key]   : "nan"
      cm = (best_key in cache_misses)   ? cache_misses[best_key]   : "nan"
      sf = (best_key in stalls_forward) ? stalls_forward[best_key] : "nan"
      bm = (best_key in branch_misses)  ? branch_misses[best_key]  : "nan"

      print c "," i "," cm "," sf "," bm
    }
  ' "$perf_file"
}

main() {
  need_cmd "$CC"
  need_cmd perf
  need_cmd awk

  if [[ ! -f "$SOURCE_FILE" ]]; then
    echo "error: source file not found: $SOURCE_FILE" >&2
    exit 1
  fi

  if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    echo "warning: not running as root; perf, SCHED_FIFO, or mlockall may fail" >&2
  fi

  mkdir -p "$BUILD_DIR" "$LOG_DIR"

  check_perf_event "$EVENT_CYCLES" "$LOG_DIR/check_cycles.log"
  check_perf_event "$EVENT_INSTRUCTIONS" "$LOG_DIR/check_instructions.log"
  check_perf_event "$EVENT_CACHE_MISSES" "$LOG_DIR/check_cache_misses.log"
  check_perf_event "$EVENT_STALLS_FORWARD" "$LOG_DIR/check_stalls_forward.log"
  check_perf_event "$EVENT_BRANCH_MISSES" "$LOG_DIR/check_branch_misses.log"

  echo "cpu_cycles,instructions,cache_misses,stalls_forward,branch_misses" >"$OUT_CSV"
  echo "variant,binary,working_set_bytes,mem_stride_words,asm_alu_groups_per_round,asm_imuls_per_round,asm_idivs_per_round,asm_loads_per_round,asm_stores_per_round,asm_branches_per_round,asm_nops_per_round,rounds_per_batch" >"$VARIANT_CSV"

  local variant=0

  
    for mem_stride_words in "${MEM_STRIDE_WORDS_LIST[@]}"; do
      for alu_groups in "${ASM_ALU_GROUPS_PER_ROUND_LIST[@]}"; do
        for imuls in "${ASM_IMULS_PER_ROUND_LIST[@]}"; do
          for idivs in "${ASM_IDIVS_PER_ROUND_LIST[@]}"; do
            for loads in "${ASM_LOADS_PER_ROUND_LIST[@]}"; do
              for stores in "${ASM_STORES_PER_ROUND_LIST[@]}"; do
                for branches in "${ASM_BRANCHES_PER_ROUND_LIST[@]}"; do
                  for nops in "${ASM_NOPS_PER_ROUND_LIST[@]}"; do
                    for rounds in "${ROUNDS_PER_BATCH_LIST[@]}"; do
                    for working_set_bytes in "${WORKING_SET_BYTES_LIST[@]}"; do
                      if (( variant >= MAX_VARIANTS )); then
                        echo "done: wrote $variant samples to $OUT_CSV" >&2
                        echo "variant parameters are in $VARIANT_CSV" >&2
                        return 0
                      fi

                      local bin="$BUILD_DIR/test_${variant}"
                      local perf_file="$LOG_DIR/perf_variant_${variant}.log"

                      echo "[$((variant + 1))/$MAX_VARIANTS] build/run $bin" >&2

                      compile_variant \
                        "$bin" \
                        "$working_set_bytes" \
                        "$mem_stride_words" \
                        "$alu_groups" \
                        "$imuls" \
                        "$idivs" \
                        "$loads" \
                        "$stores" \
                        "$branches" \
                        "$nops" \
                        "$rounds"

                      run_and_measure "$bin" "$perf_file"
                      extract_target_sample "$perf_file" >>"$OUT_CSV"

                      echo "$variant,$bin,$working_set_bytes,$mem_stride_words,$alu_groups,$imuls,$idivs,$loads,$stores,$branches,$nops,$rounds" >>"$VARIANT_CSV"

                      variant=$((variant + 1))
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done

  echo "done: wrote $variant samples to $OUT_CSV" >&2
  echo "variant parameters are in $VARIANT_CSV" >&2
}

main "$@"
