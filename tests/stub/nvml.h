/*
 * Minimal stand-in for CUDA's nvml.h, used when no CUDA toolkit is installed (GPU-less build and test
 * hosts). It declares only what dcgm-job-map.c and tests/fake_nvml.c use, with the same ABI as the real
 * header: nvmlDevice_t is an opaque pointer, nvmlReturn_t an int-sized enum, and nvmlPciInfo_t the v3
 * layout. Prefer the real header whenever $(CUDA_HOME)/include/nvml.h exists; the Makefile does.
 */
#ifndef NVML_STUB_H
#define NVML_STUB_H

typedef struct nvmlDevice_st *nvmlDevice_t;

typedef enum nvmlReturn_enum
{
    NVML_SUCCESS = 0,
    NVML_ERROR_UNINITIALIZED = 1,
    NVML_ERROR_INVALID_ARGUMENT = 2,
    NVML_ERROR_NOT_SUPPORTED = 3,
    NVML_ERROR_NOT_FOUND = 6,
    NVML_ERROR_UNKNOWN = 999
} nvmlReturn_t;

#define NVML_DEVICE_MIG_DISABLE 0x0
#define NVML_DEVICE_MIG_ENABLE 0x1

#define NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE 32
#define NVML_DEVICE_PCI_BUS_ID_BUFFER_V2_SIZE 16

typedef struct nvmlPciInfo_st
{
    char busIdLegacy[NVML_DEVICE_PCI_BUS_ID_BUFFER_V2_SIZE]; /* "0000:41:00.0" */
    unsigned int domain;
    unsigned int bus;
    unsigned int device;
    unsigned int pciDeviceId;
    unsigned int pciSubSystemId;
    char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE]; /* "00000000:41:00.0" */
} nvmlPciInfo_t;

nvmlReturn_t nvmlInit_v2(void);
nvmlReturn_t nvmlShutdown(void);
nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int *count);
nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t *device);
nvmlReturn_t nvmlDeviceGetHandleByUUID(const char *uuid, nvmlDevice_t *device);
nvmlReturn_t nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int *index);
nvmlReturn_t nvmlDeviceGetMinorNumber(nvmlDevice_t device, unsigned int *minor);
nvmlReturn_t nvmlDeviceGetPciInfo_v3(nvmlDevice_t device, nvmlPciInfo_t *pci);
nvmlReturn_t nvmlDeviceGetMigMode(nvmlDevice_t device, unsigned int *current, unsigned int *pending);
nvmlReturn_t nvmlDeviceGetMaxMigDeviceCount(nvmlDevice_t device, unsigned int *count);
nvmlReturn_t nvmlDeviceGetMigDeviceHandleByIndex(nvmlDevice_t device, unsigned int index, nvmlDevice_t *mig);
nvmlReturn_t nvmlDeviceGetGpuInstanceId(nvmlDevice_t device, unsigned int *id);
nvmlReturn_t nvmlDeviceGetDeviceHandleFromMigDeviceHandle(nvmlDevice_t mig, nvmlDevice_t *device);

#endif
