#include <ctype.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <nvml.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define DEFAULT_JOB_MAPPING_DIR "/var/run/dcgm_job_maps"
#define MAX_ITEMS 1024

/*
 * NVML is loaded via dlopen instead of linked at build time: this program also runs on nodes
 * with no NVIDIA GPU and no driver installed, where libnvidia-ml.so.1 does not exist. Linking
 * it directly would keep the program from starting at all on those nodes.
 */
typedef nvmlReturn_t (*nvmlInit_v2_fn)(void);
typedef nvmlReturn_t (*nvmlShutdown_fn)(void);
typedef nvmlReturn_t (*nvmlDeviceGetCount_v2_fn)(unsigned int *deviceCount);
typedef nvmlReturn_t (*nvmlDeviceGetHandleByIndex_v2_fn)(unsigned int index, nvmlDevice_t *device);
typedef nvmlReturn_t (*nvmlDeviceGetMinorNumber_fn)(nvmlDevice_t device, unsigned int *minorNumber);
typedef nvmlReturn_t (*nvmlDeviceGetMigMode_fn)(nvmlDevice_t device, unsigned int *currentMode, unsigned int *pendingMode);
typedef nvmlReturn_t (*nvmlDeviceGetMaxMigDeviceCount_fn)(nvmlDevice_t device, unsigned int *maxMigDevices);
typedef nvmlReturn_t (*nvmlDeviceGetMigDeviceHandleByIndex_fn)(nvmlDevice_t device, unsigned int index, nvmlDevice_t *migDevice);
typedef nvmlReturn_t (*nvmlDeviceGetGpuInstanceId_fn)(nvmlDevice_t device, unsigned int *id);
typedef nvmlReturn_t (*nvmlDeviceGetHandleByUUID_fn)(const char *uuid, nvmlDevice_t *device);
typedef nvmlReturn_t (*nvmlDeviceGetDeviceHandleFromMigDeviceHandle_fn)(nvmlDevice_t migDevice, nvmlDevice_t *device);

static void *nvml_lib = NULL;
static nvmlInit_v2_fn nvmlInit_v2_p;
static nvmlShutdown_fn nvmlShutdown_p;
static nvmlDeviceGetCount_v2_fn nvmlDeviceGetCount_v2_p;
static nvmlDeviceGetHandleByIndex_v2_fn nvmlDeviceGetHandleByIndex_v2_p;
static nvmlDeviceGetMinorNumber_fn nvmlDeviceGetMinorNumber_p;
static nvmlDeviceGetMigMode_fn nvmlDeviceGetMigMode_p;
static nvmlDeviceGetMaxMigDeviceCount_fn nvmlDeviceGetMaxMigDeviceCount_p;
static nvmlDeviceGetMigDeviceHandleByIndex_fn nvmlDeviceGetMigDeviceHandleByIndex_p;
static nvmlDeviceGetGpuInstanceId_fn nvmlDeviceGetGpuInstanceId_p;
static nvmlDeviceGetHandleByUUID_fn nvmlDeviceGetHandleByUUID_p;
static nvmlDeviceGetDeviceHandleFromMigDeviceHandle_fn nvmlDeviceGetDeviceHandleFromMigDeviceHandle_p;

/* Resolve one NVML symbol into *ptr, bailing out of load_nvml() if it's missing. */
#define LOAD_SYM(ptr, name, type)                                         \
    do                                                                    \
    {                                                                     \
        ptr = (type)dlsym(nvml_lib, name);                                \
        if (!ptr)                                                         \
        {                                                                 \
            fprintf(stderr, "dcgm-job-map: libnvidia-ml.so.1 missing symbol %s\n", name); \
            dlclose(nvml_lib);                                            \
            nvml_lib = NULL;                                              \
            return -1;                                                   \
        }                                                                 \
    } while (0)

/*
 * Load libnvidia-ml.so.1 and resolve the NVML entry points used below. Returns -1 silently if
 * the library itself is missing -- the normal case on a GPU-less node -- but reports a missing
 * symbol, since that points to a driver/header version mismatch instead.
 */
static int load_nvml(void)
{
    nvml_lib = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
    if (!nvml_lib)
        return -1;

    LOAD_SYM(nvmlInit_v2_p, "nvmlInit_v2", nvmlInit_v2_fn);
    LOAD_SYM(nvmlShutdown_p, "nvmlShutdown", nvmlShutdown_fn);
    LOAD_SYM(nvmlDeviceGetCount_v2_p, "nvmlDeviceGetCount_v2", nvmlDeviceGetCount_v2_fn);
    LOAD_SYM(nvmlDeviceGetHandleByIndex_v2_p, "nvmlDeviceGetHandleByIndex_v2", nvmlDeviceGetHandleByIndex_v2_fn);
    LOAD_SYM(nvmlDeviceGetMinorNumber_p, "nvmlDeviceGetMinorNumber", nvmlDeviceGetMinorNumber_fn);
    LOAD_SYM(nvmlDeviceGetMigMode_p, "nvmlDeviceGetMigMode", nvmlDeviceGetMigMode_fn);
    LOAD_SYM(nvmlDeviceGetMaxMigDeviceCount_p, "nvmlDeviceGetMaxMigDeviceCount", nvmlDeviceGetMaxMigDeviceCount_fn);
    LOAD_SYM(nvmlDeviceGetMigDeviceHandleByIndex_p, "nvmlDeviceGetMigDeviceHandleByIndex", nvmlDeviceGetMigDeviceHandleByIndex_fn);
    LOAD_SYM(nvmlDeviceGetGpuInstanceId_p, "nvmlDeviceGetGpuInstanceId", nvmlDeviceGetGpuInstanceId_fn);
    LOAD_SYM(nvmlDeviceGetHandleByUUID_p, "nvmlDeviceGetHandleByUUID", nvmlDeviceGetHandleByUUID_fn);
    LOAD_SYM(nvmlDeviceGetDeviceHandleFromMigDeviceHandle_p, "nvmlDeviceGetDeviceHandleFromMigDeviceHandle", nvmlDeviceGetDeviceHandleFromMigDeviceHandle_fn);

    return 0;
}

enum mode
{
    MODE_NONE,
    MODE_INIT,
    MODE_PROLOG,
    MODE_EPILOG
};

static int print_only = 0;
static int nonzero = 0;
static enum mode run_mode = MODE_NONE;

/* Print command usage and behavior. */
static void usage(const char *prog)
{
    printf(
        "Usage: %s -init|-prolog|-epilog [-print] [-nonzero] [-help]\n"
        "\n"
        "Manage dcgm_exporter Slurm job mapping files.\n"
        "\n"
        "Modes:\n"
        "  -init     Create/chmod mapping files for full GPUs and MIG GPU instances.\n"
        "            Refuses to run if SLURM_JOB_ID is set; it must see every device\n"
        "            on the node, not one job's allocated slice of it.\n"
        "  -prolog   Write SLURM_JOB_ID to the allocated mapping files\n"
        "  -epilog   Truncate allocated mapping files and write 0\n"
        "\n"
        "Options:\n"
        "  -print    Print what would be written instead of writing files\n"
        "  -nonzero  Return non-zero on errors\n"
        "  -help     Show this help message\n"
        "\n"
        "Device selection:\n"
        "  -prolog and -epilog run from the Slurm job Prolog/Epilog (slurmd, once\n"
        "  per allocation, outside the job's cgroup) and read the devices Slurm\n"
        "  allocated to the job from CUDA_VISIBLE_DEVICES, falling back to\n"
        "  GPU_DEVICE_ORDINAL if that is unset. Each entry may be an NVML\n"
        "  enumeration ordinal or a GPU/MIG instance UUID.\n"
        "\n"
        "Environment:\n"
        "  DCGM_HPC_JOB_MAPPING_DIR  Directory for job mapping files\n"
        "  SLURM_JOB_ID              Job ID written by -prolog\n"
        "  CUDA_VISIBLE_DEVICES      Devices allocated to the job (-prolog/-epilog)\n"
        "  GPU_DEVICE_ORDINAL        Fallback device list if CUDA_VISIBLE_DEVICES unset\n"
        "\n"
        "Default mapping dir: %s\n",
        prog,
        DEFAULT_JOB_MAPPING_DIR);
}

/* Return the directory dcgm_exporter will read job mapping files from. */
static const char *get_mapping_dir(void)
{
    const char *dir = getenv("DCGM_HPC_JOB_MAPPING_DIR");

    if (!dir || !*dir)
        dir = DEFAULT_JOB_MAPPING_DIR;

    return dir;
}

/*
 * Write or print one mapping file update. chmod_file is only set by -init, to fix up permissions on a
 * freshly created file: dcgm-exporter just needs to read it, since -prolog and -epilog now run as root
 * (via slurmd, from the job Prolog/Epilog) rather than as the job user, so the files no longer need to
 * be user-writable.
 */
static int write_map(const char *dir, const char *map_id, const char *value, int chmod_file)
{
    char path[512];

    snprintf(path, sizeof(path), "%s/%s", dir, map_id);

    if (print_only)
    {
        printf("would write \"%s\" to %s\n", value, path);
        return 0;
    }

    FILE *f = fopen(path, "w"); /* truncates, so a stale job ID never lingers */
    if (!f)
        return -1;

    fprintf(f, "%s\n", value);
    fclose(f);

    if (chmod_file)
        chmod(path, 0644);

    return 0;
}

/* Track mapping IDs so the program does not write the same file twice. */
static int already_written(char written[][32], unsigned int count, const char *map_id)
{
    for (unsigned int i = 0; i < count; i++)
    {
        if (strcmp(written[i], map_id) == 0)
            return 1;
    }

    return 0;
}

/* Add one mapping ID to the local map list, silently dropping it if the fixed-size array is full. */
static void add_map(char maps[][32], unsigned int *count, const char *map_id)
{
    if (*count >= MAX_ITEMS)
        return;

    snprintf(maps[*count], sizeof(maps[0]), "%s", map_id);
    (*count)++;
}

/* Report whether the GPU is currently running in MIG mode. */
static int mig_enabled(nvmlDevice_t gpu)
{
    unsigned int current = 0;
    unsigned int pending = 0;

    if (nvmlDeviceGetMigMode_p(gpu, &current, &pending) != NVML_SUCCESS)
        return 0;

    return current == NVML_DEVICE_MIG_ENABLE;
}

/*
 * Report whether this process may open the GPU's device node. Used only by -init (which runs
 * unrestricted, outside any job) as a sanity check that an NVML-enumerated full GPU has a live
 * /dev entry; it is not a job-allocation test.
 */
static int device_node_accessible(unsigned int minor)
{
    char path[64];
    int fd;

    snprintf(path, sizeof(path), "/dev/nvidia%u", minor);

    fd = open(path, O_RDONLY);
    if (fd < 0)
        return 0;

    close(fd);
    return 1;
}

/*
 * Add every MIG GPU instance NVML exposes for one GPU as "<gpu>.<gi>".
 * Instances outside the job's allocation fail the handle/GI lookup below and are skipped that way.
 */
static void collect_gpu_migs(nvmlDevice_t gpu, unsigned int gpu_id, char maps[][32], unsigned int *count)
{
    unsigned int max_migs = 0;

    if (nvmlDeviceGetMaxMigDeviceCount_p(gpu, &max_migs) != NVML_SUCCESS)
        return;

    for (unsigned int mig_index = 0; mig_index < max_migs; mig_index++)
    {
        nvmlDevice_t mig;
        unsigned int gi = 0;
        char map_id[32];

        if (nvmlDeviceGetMigDeviceHandleByIndex_p(gpu, mig_index, &mig) != NVML_SUCCESS)
            continue;

        if (nvmlDeviceGetGpuInstanceId_p(mig, &gi) != NVML_SUCCESS)
            continue;

        snprintf(map_id, sizeof(map_id), "%u.%u", gpu_id, gi);
        add_map(maps, count, map_id);
    }
}

/*
 * Build the list of mapping IDs for every device on the node: "<gpu>.<gi>" per MIG instance, or "<gpu>"
 * for a full GPU that this process can open. IDs use the GPU's minor number, not its NVML enumeration
 * position, so they stay stable regardless of NVML enumeration order. Used only by -init, which must see
 * every device on the node; -prolog and -epilog use collect_job_maps() instead, below.
 */
static int collect_maps(char maps[][32], unsigned int *count)
{
    unsigned int gpu_count = 0;

    *count = 0;

    if (nvmlDeviceGetCount_v2_p(&gpu_count) != NVML_SUCCESS)
        return -1;

    for (unsigned int gpu_index = 0; gpu_index < gpu_count; gpu_index++)
    {
        nvmlDevice_t gpu;
        unsigned int minor = 0;

        if (nvmlDeviceGetHandleByIndex_v2_p(gpu_index, &gpu) != NVML_SUCCESS)
            continue;

        if (nvmlDeviceGetMinorNumber_p(gpu, &minor) != NVML_SUCCESS)
            continue;

        if (mig_enabled(gpu))
        {
            collect_gpu_migs(gpu, minor, maps, count);
        }
        else if (device_node_accessible(minor))
        {
            char map_id[32];

            snprintf(map_id, sizeof(map_id), "%u", minor);
            add_map(maps, count, map_id);
        }
    }

    return 0;
}

/*
 * Return the env var listing devices Slurm allocated to this job, or NULL if neither is set (e.g. a
 * job that did not request a GPU). CUDA_VISIBLE_DEVICES is preferred since it is what CUDA-based
 * frameworks honor; GPU_DEVICE_ORDINAL is a fallback for sites where only it is configured.
 */
static const char *get_device_env(void)
{
    const char *cvd = getenv("CUDA_VISIBLE_DEVICES");

    if (cvd && *cvd)
        return cvd;

    const char *ordinal = getenv("GPU_DEVICE_ORDINAL");

    if (ordinal && *ordinal)
        return ordinal;

    return NULL;
}

/*
 * Resolve one CUDA_VISIBLE_DEVICES/GPU_DEVICE_ORDINAL entry into the mapping ID(s) it names, appending
 * them to maps/count. An entry is either a plain NVML enumeration ordinal or a UUID; nvmlDeviceGetHandleByUUID
 * accepts both a full GPU's UUID and a MIG instance's UUID, resolving straight to the matching handle.
 * Once resolved, nvmlDeviceGetGpuInstanceId succeeding is what tells a MIG instance handle apart from a
 * full GPU handle, since only a MIG device has an instance ID at all.
 */
static void resolve_device_entry(const char *entry, char maps[][32], unsigned int *count)
{
    nvmlDevice_t dev;
    unsigned int minor = 0;
    unsigned int gi = 0;
    int is_ordinal = 1;

    if (!*entry)
        return;

    for (const char *p = entry; *p; p++)
    {
        if (!isdigit((unsigned char)*p))
        {
            is_ordinal = 0;
            break;
        }
    }

    if (is_ordinal)
    {
        if (nvmlDeviceGetHandleByIndex_v2_p((unsigned int)strtoul(entry, NULL, 10), &dev) != NVML_SUCCESS)
            return;
    }
    else if (nvmlDeviceGetHandleByUUID_p(entry, &dev) != NVML_SUCCESS)
    {
        return;
    }

    if (nvmlDeviceGetGpuInstanceId_p(dev, &gi) == NVML_SUCCESS)
    {
        /* dev is a MIG instance handle: find its parent GPU's minor number to name it "<minor>.<gi>". */
        nvmlDevice_t parent;
        char map_id[32];

        if (nvmlDeviceGetDeviceHandleFromMigDeviceHandle_p(dev, &parent) != NVML_SUCCESS)
            return;

        if (nvmlDeviceGetMinorNumber_p(parent, &minor) != NVML_SUCCESS)
            return;

        snprintf(map_id, sizeof(map_id), "%u.%u", minor, gi);
        add_map(maps, count, map_id);
        return;
    }

    if (nvmlDeviceGetMinorNumber_p(dev, &minor) != NVML_SUCCESS)
        return;

    if (mig_enabled(dev))
    {
        /* Named by ordinal or full-GPU UUID rather than an instance UUID: the allocation doesn't tell us
         * which instance(s) of this MIG GPU the job owns, so map every instance on it. */
        collect_gpu_migs(dev, minor, maps, count);
    }
    else
    {
        char map_id[32];

        snprintf(map_id, sizeof(map_id), "%u", minor);
        add_map(maps, count, map_id);
    }
}

/*
 * Build the mapping ID list for this job from Slurm's allocated-device env var (see get_device_env()).
 * Used by -prolog and -epilog, which now run from the Slurm job Prolog/Epilog: once per allocation, via
 * slurmd as root, outside the job's cgroup. That means neither NVML enumeration nor a device-node open()
 * test is restricted to this job's devices the way it would be inside the job's cgroup, so unlike
 * collect_maps() above, device identity has to come from what Slurm tells us it allocated, not from what
 * this process can see.
 */
static int collect_job_maps(char maps[][32], unsigned int *count)
{
    const char *env = get_device_env();
    char copy[1024];
    char *saveptr = NULL;
    char *tok;

    *count = 0;

    if (!env)
        return 0; /* no GPUs allocated to this job: nothing to map */

    snprintf(copy, sizeof(copy), "%s", env);

    for (tok = strtok_r(copy, ",", &saveptr); tok; tok = strtok_r(NULL, ",", &saveptr))
        resolve_device_entry(tok, maps, count);

    return 0;
}

/* Write one value to each collected mapping file, skipping duplicates. */
static int write_maps(const char *dir, char maps[][32], unsigned int count, const char *value, int chmod_file)
{
    int rc = 0;
    char written[MAX_ITEMS][32];
    unsigned int written_count = 0;

    for (unsigned int i = 0; i < count; i++)
    {
        if (already_written(written, written_count, maps[i]))
            continue;

        if (write_map(dir, maps[i], value, chmod_file) != 0)
            rc = 1;

        snprintf(written[written_count++], sizeof(written[0]), "%s", maps[i]);
    }

    return rc;
}

/* Create root-owned, world-readable files for every full GPU and MIG GPU instance on the node. */
static int run_init(const char *dir)
{
    char maps[MAX_ITEMS][32];
    unsigned int count = 0;

    if (collect_maps(maps, &count) != 0)
        return -1;

    if (!print_only && mkdir(dir, 0755) != 0 && errno != EEXIST)
        return -1;

    return write_maps(dir, maps, count, "0", 1);
}

/* Write SLURM_JOB_ID during prolog, or 0 during epilog, to this job's mapped files. */
static int run_job_mode(const char *dir, const char *value)
{
    char maps[MAX_ITEMS][32];
    unsigned int count = 0;

    if (collect_job_maps(maps, &count) != 0)
        return -1;

    return write_maps(dir, maps, count, value, 0);
}

int main(int argc, char **argv)
{
    int rc = 0;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-init") == 0)
        {
            run_mode = MODE_INIT;
        }
        else if (strcmp(argv[i], "-prolog") == 0)
        {
            run_mode = MODE_PROLOG;
        }
        else if (strcmp(argv[i], "-epilog") == 0)
        {
            run_mode = MODE_EPILOG;
        }
        else if (strcmp(argv[i], "-print") == 0)
        {
            print_only = 1;
        }
        else if (strcmp(argv[i], "-nonzero") == 0)
        {
            nonzero = 1;
        }
        else if (strcmp(argv[i], "-help") == 0 || strcmp(argv[i], "--help") == 0)
        {
            usage(argv[0]);
            return 0;
        }
        else
        {
            fprintf(stderr, "Unknown argument: %s\n\n", argv[i]);
            usage(argv[0]);
            return nonzero ? 2 : 0;
        }
    }

    if (run_mode == MODE_NONE)
    {
        usage(argv[0]);
        return nonzero ? 2 : 0;
    }

    /* -init needs the whole node; inside a job's cgroup it would only see that job's devices. */
    if (run_mode == MODE_INIT)
    {
        const char *job_id = getenv("SLURM_JOB_ID");

        if (job_id && *job_id)
        {
            fprintf(stderr, "-init must not run inside a Slurm job (SLURM_JOB_ID=%s is set)\n", job_id);
            return nonzero ? 1 : 0;
        }
    }

    const char *dir = get_mapping_dir();

    /* No NVIDIA driver on this node is not an error for this program: skip mapping and exit clean. */
    if (load_nvml() != 0)
        return nonzero ? 1 : 0;

    if (nvmlInit_v2_p() != NVML_SUCCESS)
    {
        dlclose(nvml_lib);
        return nonzero ? 1 : 0;
    }

    if (run_mode == MODE_INIT)
    {
        rc = run_init(dir);
    }
    else if (run_mode == MODE_PROLOG)
    {
        const char *job_id = getenv("SLURM_JOB_ID");

        if (!job_id || !*job_id)
        {
            rc = 1;
        }
        else
        {
            rc = run_job_mode(dir, job_id);
        }
    }
    else if (run_mode == MODE_EPILOG)
    {
        rc = run_job_mode(dir, "0");
    }

    nvmlShutdown_p();
    dlclose(nvml_lib);

    /* Without -nonzero, always exit 0 so this program never fails a Slurm prolog/epilog. */
    return nonzero ? rc : 0;
}
