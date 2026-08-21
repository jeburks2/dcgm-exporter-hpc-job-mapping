CC ?= gcc
CUDA_HOME ?= /usr/local/cuda

TARGET := dcgm-job-map
SRC := dcgm-job-map.c

CFLAGS ?= -O3 -Wall -Wextra
CPPFLAGS += -I$(CUDA_HOME)/include
# libnvidia-ml is dlopen'd at runtime, not linked, so this binary starts on nodes with no
# NVIDIA driver installed. Only libdl itself is a real link-time dependency.
LDLIBS += -ldl

.PHONY: all clean install

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

install: $(TARGET)
	install -m 0755 $(TARGET) /usr/local/sbin/$(TARGET)

clean:
	rm -f $(TARGET)
