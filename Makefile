CC ?= gcc
CUDA_HOME ?= /usr/local/cuda

TARGET := dcgm-job-map
SRC := dcgm-job-map.c

# Build against the CUDA toolkit's nvml.h when there is one; otherwise the ABI-compatible stub in
# tests/stub, so the program and its tests build on machines with no NVIDIA software at all.
NVML_INCLUDE := $(if $(wildcard $(CUDA_HOME)/include/nvml.h),$(CUDA_HOME)/include,tests/stub)

CFLAGS ?= -O3 -Wall -Wextra
CPPFLAGS += -I$(NVML_INCLUDE)
# libnvidia-ml is dlopen'd at runtime, not linked, so this binary starts on nodes with no
# NVIDIA driver installed. Only libdl itself is a real link-time dependency.
LDLIBS += -ldl

TEST_DIR := tests/build
FAKE_NVML := $(TEST_DIR)/libnvidia-ml.so.1

.PHONY: all clean install test

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

install: $(TARGET)
	install -m 0755 $(TARGET) /usr/local/sbin/$(TARGET)

# The fake NVML the tests dlopen in place of the driver's library. Any -I from the environment is
# kept so the fake is built against the same nvml.h as the program.
$(FAKE_NVML): tests/fake_nvml.c tests/stub/nvml.h
	mkdir -p $(TEST_DIR)
	$(CC) $(CPPFLAGS) -Wall -Wextra -shared -fPIC -o $@ $<

test: $(TARGET) $(FAKE_NVML)
	bash tests/run.sh

clean:
	rm -f $(TARGET)
	rm -rf $(TEST_DIR)
