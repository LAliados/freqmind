MODULE := freqmind
ROOT_DIR := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))
KDIR ?= /lib/modules/$(shell uname -r)/build

SRC_DIR := $(ROOT_DIR)/src
BUILD_DIR := $(ROOT_DIR)/build

BENCHMARK_DIR_PATH := $(ROOT_DIR)/benchmarks/pmu_build
BENCHMARK_COUNT := 5

BUILD_MODE ?= inference
EVAL_SAMPLES_PER_BENCHMARK ?= 20
TRAIN_RESULTS_PATH := $(BUILD_DIR)/freqmind_train_cycles.csv
TUNED_PARAMS_PATH := $(SRC_DIR)/freqmind_tuned_params.h

ALL_SRCS := $(shell find $(SRC_DIR) -type f -name '*.c' | sort)

ifeq ($(BUILD_MODE),train)
EXCLUDED_SRCS := $(SRC_DIR)/inference.c
else ifeq ($(BUILD_MODE),inference)
EXCLUDED_SRCS := $(SRC_DIR)/train.c
else
$(error unknown BUILD_MODE '$(BUILD_MODE)': expected train or inference)
endif

SRCS := $(filter-out $(EXCLUDED_SRCS),$(ALL_SRCS))
OBJS := $(patsubst $(ROOT_DIR)/%.c,%.o,$(SRCS))

ifneq ($(KERNELRELEASE),)
obj-m := $(MODULE).o
$(MODULE)-y := $(OBJS)

ccflags-y += -I$(ROOT_DIR)/src
ccflags-y += -Wall -Wextra
ccflags-y += -DBENCHMARK_DIR_PATH=\"$(BENCHMARK_DIR_PATH)\"
ccflags-y += -DBENCHMARK_COUNT=$(BENCHMARK_COUNT)
ccflags-y += -DTRAIN_RESULTS_PATH=\"$(TRAIN_RESULTS_PATH)\"
ccflags-y += -DFREQMIND_TUNED_PARAMS_PATH=\"$(TUNED_PARAMS_PATH)\"
ccflags-y += -DFREQMIND_EVAL_SAMPLES_PER_BENCHMARK=$(EVAL_SAMPLES_PER_BENCHMARK)
else

.PHONY: all train inference pretrain build-module cc clean load unload print move-artifacts

all: inference

pretrain:
	cd benchmarks && sudo ../run_pmu_sweep.sh pmu_x86_load.c
	python3 pretrain.py benchmarks/pmu_results.csv \
		--layer-dims 1 \
		--activations none \
		--epochs 2000 \
		--batch-size 32 \
		--c-prefix perf_mlp \
		--c-output src/perf_mlp_params.h

train: pretrain
	$(MAKE) BUILD_MODE=train build-module

inference:
	$(MAKE) BUILD_MODE=inference build-module

build-module:
	$(MAKE) -C $(KDIR) M=$(ROOT_DIR) BUILD_MODE=$(BUILD_MODE) modules
	mkdir -p $(BUILD_DIR)
	python3 $(KDIR)/scripts/clang-tools/gen_compile_commands.py \
		-d $(KDIR) \
		-o $(ROOT_DIR)/compile_commands.json \
		$(BUILD_DIR)
	$(MAKE) move-artifacts
	mv compile_commands.json $(BUILD_DIR)

cc: all

move-artifacts:
	rm -rf $(BUILD_DIR)
	mkdir -p $(BUILD_DIR)
	find $(ROOT_DIR) -path $(BUILD_DIR) -prune -o -type f \( \
		-name '*.o' -o \
		-name '*.ko' -o \
		-name '*.mod' -o \
		-name '*.mod.c' -o \
		-name '*.mod.o' -o \
		-name '*.cmd' -o \
		-name '*.symvers' -o \
		-name 'modules.order' \) \
		-exec mv {} $(BUILD_DIR)/ \;
	find $(ROOT_DIR) -type f -name '*.cmd' -delete

clean:
	$(MAKE) -C $(KDIR) M=$(ROOT_DIR) clean
	rm -rf $(BUILD_DIR)

load:
	sudo insmod $(BUILD_DIR)/$(MODULE).ko

unload:
	sudo rmmod $(MODULE)

print:
	@echo MODULE=$(MODULE)
	@echo ROOT_DIR=$(ROOT_DIR)
	@echo KDIR=$(KDIR)
	@echo BUILD_MODE=$(BUILD_MODE)
	@echo SRC_DIR=$(SRC_DIR)
	@echo BUILD_DIR=$(BUILD_DIR)
	@echo BENCHMARK_DIR_PATH=$(BENCHMARK_DIR_PATH)
	@echo BENCHMARK_COUNT=$(BENCHMARK_COUNT)
	@echo EVAL_SAMPLES_PER_BENCHMARK=$(EVAL_SAMPLES_PER_BENCHMARK)
	@echo TRAIN_RESULTS_PATH=$(TRAIN_RESULTS_PATH)
	@echo TUNED_PARAMS_PATH=$(TUNED_PARAMS_PATH)
	@echo EXCLUDED_SRCS=$(EXCLUDED_SRCS)
	@echo SRCS=$(SRCS)
	@echo OBJS=$(OBJS)

endif
