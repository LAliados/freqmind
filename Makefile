MODULE := analyzer_demo

KDIR ?= /lib/modules/$(shell uname -r)/build
PWD  := $(CURDIR)

BUILD_DIR := $(PWD)/build
SRC_DIR   := src

SRCS := $(shell find $(SRC_DIR) -type f -name '*.c' | sort)
OBJS := $(patsubst %.c,%.o,$(SRCS))

ifneq ($(KERNELRELEASE),)

obj-m := $(MODULE).o
$(MODULE)-y := $(OBJS)

ccflags-y += -I$(src)/src
ccflags-y += -Wall -Wextra

else

.PHONY: all clean cc load unload

all:
	mkdir -p $(BUILD_DIR)
	$(MAKE) -C $(KDIR) M=$(PWD) MO=$(BUILD_DIR) modules

cc: all
	python3 $(KDIR)/scripts/clang-tools/gen_compile_commands.py \
		-d $(BUILD_DIR) \
		-o $(PWD)/compile_commands.json

clean:
	rm -rf $(BUILD_DIR)
	rm -f compile_commands.json

load:
	sudo insmod $(BUILD_DIR)/$(MODULE).ko

unload:
	sudo rmmod $(MODULE)

endif