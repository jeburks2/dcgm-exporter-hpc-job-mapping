/*
 * A fake libnvidia-ml.so.1 for the test suite. dcgm-job-map dlopen()s NVML by name, so putting this
 * library first on LD_LIBRARY_PATH swaps in a node of our choosing: any number of GPUs, in any NVML
 * order, with any minor numbers, PCI addresses and MIG layout -- including the sg235 layout from the
 * October 2026 wrongful-cancel report, where minor order and NVML order disagree.
 *
 * FAKE_NVML_GPUS describes the node, one entry per GPU in NVML enumeration order, comma-separated:
 *
 *     <minor>[@<pci-bus-hex>][:<gi>+<gi>+...]
 *
 * <minor> is the GPU's /dev/nvidia<minor> number. @<bus> gives its PCI bus (domain 0, device 0,
 * function 0); it defaults to NVML index + 1, i.e. NVML order == PCI order. A ":"-list of GPU instance
 * IDs makes the GPU MIG-enabled with exactly those instances, in that MIG-index order.
 *
 *     FAKE_NVML_GPUS="0@01,3@41,1@81,2@c1"   sg235: NVML 1 is /dev/nvidia3, NVML 3 is /dev/nvidia2
 *     FAKE_NVML_GPUS="1:3+5,0"               NVML 0 is MIG-partitioned into instances 3 and 5
 *
 * UUIDs are "GPU-fake-<nvml-index>" and "MIG-fake-<nvml-index>-<gi>". FAKE_NVML_INIT_FAIL=1 makes
 * nvmlInit_v2 fail.
 */
#include <nvml.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_GPUS 64
#define MAX_GIS 8

struct fake_gpu
{
    unsigned int minor;
    unsigned int bus;
    unsigned int gi[MAX_GIS];
    unsigned int gi_count;
};

static struct fake_gpu gpus[MAX_GPUS];
static unsigned int gpu_count = 0;
static int parsed = 0;

/* Handles: a full GPU is 0x1000 + index, MIG slot j of GPU i is 0x2000 + i * 0x10 + j. */
#define GPU_HANDLE(i) ((nvmlDevice_t)(uintptr_t)(0x1000u + (i)))
#define MIG_HANDLE(i, j) ((nvmlDevice_t)(uintptr_t)(0x2000u + (i) * 0x10u + (j)))

static int handle_gpu(nvmlDevice_t h, unsigned int *i)
{
    uintptr_t v = (uintptr_t)h;

    if (v < 0x1000u || v >= 0x1000u + gpu_count)
        return 0;

    *i = (unsigned int)(v - 0x1000u);
    return 1;
}

static int handle_mig(nvmlDevice_t h, unsigned int *i, unsigned int *j)
{
    uintptr_t v = (uintptr_t)h;

    if (v < 0x2000u || v >= 0x2000u + gpu_count * 0x10u)
        return 0;

    *i = (unsigned int)((v - 0x2000u) / 0x10u);
    *j = (unsigned int)((v - 0x2000u) % 0x10u);
    return *j < gpus[*i].gi_count;
}

static void parse_spec(void)
{
    const char *env = getenv("FAKE_NVML_GPUS");
    char copy[1024];
    char *saveptr = NULL;
    char *tok;

    if (parsed)
        return;

    parsed = 1;
    gpu_count = 0;

    if (!env || !*env)
        return;

    snprintf(copy, sizeof(copy), "%s", env);

    for (tok = strtok_r(copy, ",", &saveptr); tok && gpu_count < MAX_GPUS; tok = strtok_r(NULL, ",", &saveptr))
    {
        struct fake_gpu *g = &gpus[gpu_count];
        char *colon = strchr(tok, ':');
        char *at;

        memset(g, 0, sizeof(*g));
        g->bus = gpu_count + 1;

        if (colon)
        {
            char *gi_saveptr = NULL;
            char *gi;

            *colon++ = '\0';
            for (gi = strtok_r(colon, "+", &gi_saveptr); gi && g->gi_count < MAX_GIS;
                 gi = strtok_r(NULL, "+", &gi_saveptr))
                g->gi[g->gi_count++] = (unsigned int)strtoul(gi, NULL, 10);
        }

        at = strchr(tok, '@');
        if (at)
        {
            *at++ = '\0';
            g->bus = (unsigned int)strtoul(at, NULL, 16);
        }

        g->minor = (unsigned int)strtoul(tok, NULL, 10);
        gpu_count++;
    }
}

nvmlReturn_t nvmlInit_v2(void)
{
    const char *fail = getenv("FAKE_NVML_INIT_FAIL");

    parse_spec();
    return (fail && *fail && strcmp(fail, "0") != 0) ? NVML_ERROR_UNKNOWN : NVML_SUCCESS;
}

nvmlReturn_t nvmlShutdown(void)
{
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int *count)
{
    parse_spec();
    *count = gpu_count;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t *device)
{
    parse_spec();
    if (index >= gpu_count)
        return NVML_ERROR_INVALID_ARGUMENT;

    *device = GPU_HANDLE(index);
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetHandleByUUID(const char *uuid, nvmlDevice_t *device)
{
    unsigned int i, gi;

    parse_spec();

    if (sscanf(uuid, "GPU-fake-%u", &i) == 1 && strchr(uuid + 9, '-') == NULL)
    {
        if (i >= gpu_count)
            return NVML_ERROR_NOT_FOUND;

        *device = GPU_HANDLE(i);
        return NVML_SUCCESS;
    }

    if (sscanf(uuid, "MIG-fake-%u-%u", &i, &gi) == 2 && i < gpu_count)
    {
        for (unsigned int j = 0; j < gpus[i].gi_count; j++)
        {
            if (gpus[i].gi[j] == gi)
            {
                *device = MIG_HANDLE(i, j);
                return NVML_SUCCESS;
            }
        }
    }

    return NVML_ERROR_NOT_FOUND;
}

nvmlReturn_t nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int *index)
{
    unsigned int i;

    if (!handle_gpu(device, &i))
        return NVML_ERROR_INVALID_ARGUMENT;

    *index = i;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetMinorNumber(nvmlDevice_t device, unsigned int *minor)
{
    unsigned int i, j;

    if (handle_gpu(device, &i) || handle_mig(device, &i, &j))
    {
        *minor = gpus[i].minor;
        return NVML_SUCCESS;
    }

    return NVML_ERROR_INVALID_ARGUMENT;
}

nvmlReturn_t nvmlDeviceGetPciInfo_v3(nvmlDevice_t device, nvmlPciInfo_t *pci)
{
    unsigned int i;

    if (!handle_gpu(device, &i))
        return NVML_ERROR_INVALID_ARGUMENT;

    memset(pci, 0, sizeof(*pci));
    pci->domain = 0;
    pci->bus = gpus[i].bus;
    pci->device = 0;
    snprintf(pci->busIdLegacy, sizeof(pci->busIdLegacy), "0000:%02X:00.0", gpus[i].bus);
    snprintf(pci->busId, sizeof(pci->busId), "00000000:%02X:00.0", gpus[i].bus);
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetMigMode(nvmlDevice_t device, unsigned int *current, unsigned int *pending)
{
    unsigned int i;

    if (!handle_gpu(device, &i))
        return NVML_ERROR_INVALID_ARGUMENT;

    *current = *pending = gpus[i].gi_count ? NVML_DEVICE_MIG_ENABLE : NVML_DEVICE_MIG_DISABLE;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetMaxMigDeviceCount(nvmlDevice_t device, unsigned int *count)
{
    unsigned int i;

    if (!handle_gpu(device, &i))
        return NVML_ERROR_INVALID_ARGUMENT;

    /* The real call reports a profile-independent maximum, so unpopulated slots are normal. */
    *count = gpus[i].gi_count ? 7 : 0;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetMigDeviceHandleByIndex(nvmlDevice_t device, unsigned int index, nvmlDevice_t *mig)
{
    unsigned int i;

    if (!handle_gpu(device, &i))
        return NVML_ERROR_INVALID_ARGUMENT;

    if (index >= gpus[i].gi_count)
        return NVML_ERROR_NOT_FOUND;

    *mig = MIG_HANDLE(i, index);
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetGpuInstanceId(nvmlDevice_t device, unsigned int *id)
{
    unsigned int i, j;

    if (!handle_mig(device, &i, &j))
        return NVML_ERROR_NOT_SUPPORTED; /* a full GPU has no instance ID, as in real NVML */

    *id = gpus[i].gi[j];
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetDeviceHandleFromMigDeviceHandle(nvmlDevice_t mig, nvmlDevice_t *device)
{
    unsigned int i, j;

    if (!handle_mig(mig, &i, &j))
        return NVML_ERROR_INVALID_ARGUMENT;

    *device = GPU_HANDLE(i);
    return NVML_SUCCESS;
}
