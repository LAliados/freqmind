MODULE := freqmind

ROOT_DIR := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))
KDIR ?= /lib/modules/$(shell uname -r)/build

SRC_DIR   := $(ROOT_DIR)/src
BUILD_DIR := $(ROOT_DIR)/build

SRCS := $(shell find $(SRC_DIR) -type f -name '*.c' | sort)
OBJS := $(patsubst $(ROOT_DIR)/%.c,%.o,$(SRCS))

ifeq ($(strip $(SRCS)),)
$(error no .c files found in $(SRC_DIR))
endif

ifneq ($(KERNELRELEASE),)

obj-m := $(MODULE).o
$(MODULE)-y := $(OBJS)

ccflags-y += -I$(ROOT_DIR)/src
ccflags-y += -Wall -Wextra

else

.PHONY: all clean cc load unload print move-artifacts

all:
	$(MAKE) -C $(KDIR) M=$(ROOT_DIR) modules
	$(MAKE) move-artifacts

cc: all
	python3 $(KDIR)/scripts/clang-tools/gen_compile_commands.py \
		-d $(KDIR) \
		-o $(ROOT_DIR)/compile_commands.json \
		$(BUILD_DIR)
	mv compile_commands.json $(BUILD_DIR)

move-artifacts:
	rm -rf $(BUILD_DIR)
	mkdir -p $(BUILD_DIR)
	find $(ROOT_DIR) -path $(BUILD_DIR) -prune -o -type f \( \
		-name '*.o' -o \
		-name '*.ko' -o \
		-name '*.mod' -o \
		-name '*.mod.c' -o \
		-name '.*.cmd' -o \
		-name '*.o.d' -o \
		-name 'Module.symvers' -o \
		-name 'modules.order' -o \
		-name 'modules.nsdeps' \
	\) -exec sh -c '\
		root="'"$(ROOT_DIR)"'"; \
		build="'"$(BUILD_DIR)"'"; \
		for f do \
			rel=$${f#$$root/}; \
			dst="$$build/$$(dirname "$$rel")"; \
			mkdir -p "$$dst"; \
			mv "$$f" "$$dst/"; \
		done \
	' sh {} +
	if [ -d $(ROOT_DIR)/.tmp_versions ]; then \
		mv $(ROOT_DIR)/.tmp_versions $(BUILD_DIR)/; \
	fi

clean:
	$(MAKE) -C $(KDIR) M=$(ROOT_DIR) clean
	rm -rf $(BUILD_DIR)
	rm -f $(ROOT_DIR)/compile_commands.json
	

load:
	sudo insmod $(BUILD_DIR)/$(MODULE).ko

unload:
	sudo rmmod $(MODULE)

print:
	@echo "MODULE    = $(MODULE)"
	@echo "ROOT_DIR  = $(ROOT_DIR)"
	@echo "SRC_DIR   = $(SRC_DIR)"
	@echo "BUILD_DIR = $(BUILD_DIR)"
	@echo "SRCS      = $(SRCS)"
	@echo "OBJS      = $(OBJS)"

endif