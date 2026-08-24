#include <ctype.h>
#include <dlfcn.h>
#include <errno.h>
#include <nvml.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define DEFAULT_JOB_MAPPING_DIR "/var/run/dcgm_job_maps"
#define MAX_ITEMS 1024

/*
 * NVML is dlopen'd rather than linked: this program also runs on nodes with no NVIDIA driver, where
 * libnvidia-ml.so.1 does not exist and linking it would keep the binary from starting at all.
 */
static void *nvml_lib = NULL;
static nvmlReturn_t (*nvmlInit_v2_p)(void);
static nvmlReturn_t (*nvmlShutdown_p)(void);
static nvmlReturn_t (*nvmlDeviceGetCount_v2_p)(unsigned int *count);
static nvmlReturn_t (*nvmlDeviceGetHandleByIndex_v2_p)(unsigned int index, nvmlDevice_t *device);
static nvmlReturn_t (*nvmlDeviceGetMinorNumber_p)(nvmlDevice_t device, unsigned int *minor);
static nvmlReturn_t (*nvmlDeviceGetMigMode_p)(nvmlDevice_t device, unsigned int *current, unsigned int *pending);
static nvmlReturn_t (*nvmlDeviceGetMaxMigDeviceCount_p)(nvmlDevice_t device, unsigned int *count);
static nvmlReturn_t (*nvmlDeviceGetMigDeviceHandleByIndex_p)(nvmlDevice_t device, unsigned int index, nvmlDevice_t *mig);
static nvmlReturn_t (*nvmlDeviceGetGpuInstanceId_p)(nvmlDevice_t device, unsigned int *id);
static nvmlReturn_t (*nvmlDeviceGetHandleByUUID_p)(const char *uuid, nvmlDevice_t *device);
static nvmlReturn_t (*nvmlDeviceGetDeviceHandleFromMigDeviceHandle_p)(nvmlDevice_t mig, nvmlDevice_t *device);

/* Resolve one NVML symbol into *ptr, bailing out of load_nvml() if it's missing. */
#define LOAD_SYM(ptr, name)                                                                \
    do                                                                                     \
    {                                                                                      \
        ptr = (__typeof__(ptr))dlsym(nvml_lib, name);                                      \
        if (!ptr)                                                                          \
        {                                                                                  \
            fprintf(stderr, "dcgm-job-map: libnvidia-ml.so.1 missing symbol %s\n", name);   \
            dlclose(nvml_lib);                                                             \
            nvml_lib = NULL;                                                               \
            return -1;                                                                     \
        }                                                                                  \
    } while (0)

/*
 * Returns -1 silently when the library is missing -- the normal case on a GPU-less node -- but reports
 * a missing symbol, which points to a driver/header version mismatch instead.
 */
static int load_nvml(void)
{
    nvml_lib = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
    if (!nvml_lib)
        return -1;

    LOAD_SYM(nvmlInit_v2_p, "nvmlInit_v2");
    LOAD_SYM(nvmlShutdown_p, "nvmlShutdown");
    LOAD_SYM(nvmlDeviceGetCount_v2_p, "nvmlDeviceGetCount_v2");
    LOAD_SYM(nvmlDeviceGetHandleByIndex_v2_p, "nvmlDeviceGetHandleByIndex_v2");
    LOAD_SYM(nvmlDeviceGetMinorNumber_p, "nvmlDeviceGetMinorNumber");
    LOAD_SYM(nvmlDeviceGetMigMode_p, "nvmlDeviceGetMigMode");
    LOAD_SYM(nvmlDeviceGetMaxMigDeviceCount_p, "nvmlDeviceGetMaxMigDeviceCount");
    LOAD_SYM(nvmlDeviceGetMigDeviceHandleByIndex_p, "nvmlDeviceGetMigDeviceHandleByIndex");
    LOAD_SYM(nvmlDeviceGetGpuInstanceId_p, "nvmlDeviceGetGpuInstanceId");
    LOAD_SYM(nvmlDeviceGetHandleByUUID_p, "nvmlDeviceGetHandleByUUID");
    LOAD_SYM(nvmlDeviceGetDeviceHandleFromMigDeviceHandle_p, "nvmlDeviceGetDeviceHandleFromMigDeviceHandle");

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

static void usage(const char *prog)
{
    printf(
        "Usage: %s -init|-prolog|-epilog [-print] [-nonzero] [-help]\n"
        "\n"
        "Manage dcgm_exporter Slurm job mapping files.\n"
        "\n"
        "Modes:\n"
        "  -init     Create/chmod mapping files for full GPUs and MIG GPU instances.\n"
        "            Refuses to run if SLURM_JOB_ID is set\n"
        "  -prolog   Write SLURM_JOB_ID to the allocated mapping files\n"
        "  -epilog   Truncate allocated mapping files and write 0\n"
        "\n"
        "Options:\n"
        "  -print    Print what would be written instead of writing files\n"
        "  -nonzero  Return non-zero on errors\n"
        "  -help     Show this help message\n"
        "\n"
        "Device selection:\n"
        "  -prolog and -epilog read the job's allocated devices from SLURM_JOB_GPUS or\n"
        "  CUDA_VISIBLE_DEVICES. Each entry is a GPU/MIG instance UUID, or a Slurm gres\n"
        "  index -- a position in the node's device list, where a MIG-mode GPU\n"
        "  contributes one entry per GPU instance rather than one for the whole GPU.\n"
        "\n"
        "Environment:\n"
        "  DCGM_HPC_JOB_MAPPING_DIR  Directory for job mapping files\n"
        "  SLURM_JOB_ID              Job ID written by -prolog\n"
        "  SLURM_JOB_GPUS            Devices allocated to the job (-prolog/-epilog)\n"
        "  CUDA_VISIBLE_DEVICES      Preferred over SLURM_JOB_GPUS when it holds UUIDs,\n"
        "                            and used as a fallback when SLURM_JOB_GPUS is unset\n"
        "\n"
        "Default mapping dir: %s\n",
        prog,
        DEFAULT_JOB_MAPPING_DIR);
}

static const char *get_mapping_dir(void)
{
    const char *dir = getenv("DCGM_HPC_JOB_MAPPING_DIR");

    return (dir && *dir) ? dir : DEFAULT_JOB_MAPPING_DIR;
}

/* Write or print one mapping file update. chmod_file is set only by -init, on files it just created. */
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

/* Append a mapping ID, ignoring duplicates and silently dropping it if the fixed-size array is full. */
static void add_map(char maps[][32], unsigned int *count, const char *map_id)
{
    for (unsigned int i = 0; i < *count; i++)
    {
        if (strcmp(maps[i], map_id) == 0)
            return;
    }

    if (*count >= MAX_ITEMS)
        return;

    snprintf(maps[*count], sizeof(maps[0]), "%s", map_id);
    (*count)++;
}

static int mig_enabled(nvmlDevice_t gpu)
{
    unsigned int current = 0;
    unsigned int pending = 0;

    if (nvmlDeviceGetMigMode_p(gpu, &current, &pending) != NVML_SUCCESS)
        return 0;

    return current == NVML_DEVICE_MIG_ENABLE;
}

/*
 * One allocatable device: a full GPU, or one MIG GPU instance of one. The table holds these in NVML
 * enumeration order -- GPUs by NVML index, a MIG GPU's instances by MIG index -- which is the order
 * Slurm's gres/gpu NVML autodetect builds the node's GPU gres list in, and so the order its device
 * numbering follows. Devices are identified by the parent GPU's minor number rather than its NVML
 * position, so every mode names the same file for the same physical device.
 */
struct node_device
{
    unsigned int minor; /* parent GPU's minor number, i.e. /dev/nvidia<minor> */
    unsigned int gi;    /* GPU instance ID, meaningful only when is_mig */
    int is_mig;
};

static struct node_device node_devices[MAX_ITEMS];
static unsigned int node_device_count = 0;
static int node_table_built = 0;

/* "<minor>.<gi>" for a MIG instance, "<minor>" for a full GPU. */
static void device_map_id(const struct node_device *dev, char *buf, size_t len)
{
    if (dev->is_mig)
        snprintf(buf, len, "%u.%u", dev->minor, dev->gi);
    else
        snprintf(buf, len, "%u", dev->minor);
}

static void add_node_device(unsigned int minor, int is_mig, unsigned int gi)
{
    if (node_device_count >= MAX_ITEMS)
        return;

    node_devices[node_device_count].minor = minor;
    node_devices[node_device_count].is_mig = is_mig;
    node_devices[node_device_count].gi = gi;
    node_device_count++;
}

static void add_gpu_migs(nvmlDevice_t gpu, unsigned int minor)
{
    unsigned int max_migs = 0;

    if (nvmlDeviceGetMaxMigDeviceCount_p(gpu, &max_migs) != NVML_SUCCESS)
        return;

    for (unsigned int mig_index = 0; mig_index < max_migs; mig_index++)
    {
        nvmlDevice_t mig;
        unsigned int gi = 0;

        /* max_migs is a profile-independent upper bound, so unpopulated slots are expected. */
        if (nvmlDeviceGetMigDeviceHandleByIndex_p(gpu, mig_index, &mig) != NVML_SUCCESS)
            continue;

        if (nvmlDeviceGetGpuInstanceId_p(mig, &gi) != NVML_SUCCESS)
            continue;

        add_node_device(minor, 1, gi);
    }
}

/* Enumerate every allocatable device on the node once, into node_devices[]. */
static int build_node_device_table(void)
{
    unsigned int gpu_count = 0;

    if (node_table_built)
        return 0;

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
            add_gpu_migs(gpu, minor);
        else
            add_node_device(minor, 0, 0);
    }

    node_table_built = 1;
    return 0;
}

static void add_all_node_devices(char maps[][32], unsigned int *count)
{
    for (unsigned int i = 0; i < node_device_count; i++)
    {
        char map_id[32];

        device_map_id(&node_devices[i], map_id, sizeof(map_id));
        add_map(maps, count, map_id);
    }
}

/* Add every MIG instance of one GPU, for an entry naming the GPU but not an instance. */
static void add_gpu_mig_maps(unsigned int minor, char maps[][32], unsigned int *count)
{
    for (unsigned int i = 0; i < node_device_count; i++)
    {
        char map_id[32];

        if (!node_devices[i].is_mig || node_devices[i].minor != minor)
            continue;

        device_map_id(&node_devices[i], map_id, sizeof(map_id));
        add_map(maps, count, map_id);
    }
}

/* Every device on the node, for -init. -prolog/-epilog use collect_job_maps() instead. */
static int collect_maps(char maps[][32], unsigned int *count)
{
    *count = 0;

    if (build_node_device_table() != 0)
        return -1;

    add_all_node_devices(maps, count);
    return 0;
}

static int all_digits(const char *s)
{
    if (!*s)
        return 0;

    for (const char *p = s; *p; p++)
    {
        if (!isdigit((unsigned char)*p))
            return 0;
    }

    return 1;
}

/*
 * Does this list name devices by UUID rather than by number? Every UUID carries a "GPU-"/"MIG-"
 * prefix, so a letter is the tell; a numeric list holds only digits, commas and a range's '-'.
 */
static int is_uuid_list(const char *env)
{
    for (const char *p = env; *p; p++)
    {
        if (isalpha((unsigned char)*p))
            return 1;
    }

    return 0;
}

/* Parse an inclusive range like "4-7". Returns 0 if entry is not one -- a UUID also contains '-'. */
static int parse_range(const char *entry, unsigned long *lo, unsigned long *hi)
{
    const char *dash = strchr(entry, '-');
    char first[16];
    size_t len;

    if (!dash || dash == entry || !all_digits(dash + 1))
        return 0;

    len = (size_t)(dash - entry);
    if (len >= sizeof(first))
        return 0;

    snprintf(first, len + 1, "%s", entry);
    if (!all_digits(first))
        return 0;

    *lo = strtoul(first, NULL, 10);
    *hi = strtoul(dash + 1, NULL, 10);

    return *hi >= *lo;
}

/* How many devices a numeric list names. Used only to compare against what NVML can see. */
static unsigned int count_device_entries(const char *env)
{
    char copy[1024];
    char *saveptr = NULL;
    char *tok;
    unsigned int total = 0;

    snprintf(copy, sizeof(copy), "%s", env);

    for (tok = strtok_r(copy, ",", &saveptr); tok; tok = strtok_r(NULL, ",", &saveptr))
    {
        unsigned long lo = 0;
        unsigned long hi = 0;

        if (parse_range(tok, &lo, &hi))
            total += (unsigned int)(hi - lo + 1);
        else if (all_digits(tok))
            total++;
    }

    return total;
}

/*
 * The env var listing this job's allocated devices, or NULL if none is set (e.g. a job that requested
 * no GPU). *name, when non-NULL, receives which variable it came from.
 *
 * Slurm's Prolog/Epilog environment gives the allocation as device numbers, not UUIDs -- even for a MIG
 * allocation, where the job itself would later see a MIG-<uuid> in CUDA_VISIBLE_DEVICES. SLURM_JOB_GPUS
 * is the authoritative form of that numbering, so it wins; CUDA_VISIBLE_DEVICES takes priority only when
 * it holds UUIDs, which name the exact device and need no numbering assumption at all.
 */
static const char *get_device_env(const char **name)
{
    const char *cvd = getenv("CUDA_VISIBLE_DEVICES");
    const char *job_gpus = getenv("SLURM_JOB_GPUS");

    if (cvd && *cvd && is_uuid_list(cvd))
    {
        if (name)
            *name = "CUDA_VISIBLE_DEVICES";
        return cvd;
    }

    if (job_gpus && *job_gpus)
    {
        if (name)
            *name = "SLURM_JOB_GPUS";
        return job_gpus;
    }

    if (cvd && *cvd)
    {
        if (name)
            *name = "CUDA_VISIBLE_DEVICES";
        return cvd;
    }

    return NULL;
}

/*
 * A device number is a position in the node device table, which is how Slurm numbers the GPU gres it
 * can allocate: on a MIG-mode GPU each instance is separately allocatable and consumes its own number,
 * so number 0 on a node whose first GPU is partitioned is that GPU's first instance, not the whole GPU.
 */
static void resolve_device_number(unsigned int index, char maps[][32], unsigned int *count)
{
    char map_id[32];

    if (index >= node_device_count)
    {
        fprintf(stderr,
                "dcgm-job-map: device %u is past the end of the node's device list (%u devices); "
                "Slurm numbers devices across the whole node, so this usually means NVML is only "
                "showing part of it -- run from the job Prolog/Epilog, not from inside the job\n",
                index, node_device_count);
        return;
    }

    device_map_id(&node_devices[index], map_id, sizeof(map_id));
    add_map(maps, count, map_id);
}

/*
 * Resolve one list entry -- a device number, a range of them, or a UUID -- into the mapping ID(s) it
 * names. nvmlDeviceGetHandleByUUID takes both a full GPU's UUID and a MIG instance's; afterwards,
 * nvmlDeviceGetGpuInstanceId succeeding is what distinguishes the two, since only a MIG device has an
 * instance ID at all.
 */
static void resolve_device_entry(const char *entry, char maps[][32], unsigned int *count)
{
    nvmlDevice_t dev;
    unsigned long lo = 0;
    unsigned long hi = 0;
    unsigned int minor = 0;
    unsigned int gi = 0;

    if (!*entry)
        return;

    if (all_digits(entry))
    {
        resolve_device_number((unsigned int)strtoul(entry, NULL, 10), maps, count);
        return;
    }

    if (parse_range(entry, &lo, &hi))
    {
        for (unsigned long i = lo; i <= hi && i < MAX_ITEMS; i++)
            resolve_device_number((unsigned int)i, maps, count);

        return;
    }

    if (nvmlDeviceGetHandleByUUID_p(entry, &dev) != NVML_SUCCESS)
        return;

    if (nvmlDeviceGetGpuInstanceId_p(dev, &gi) == NVML_SUCCESS)
    {
        /* A MIG instance handle: its parent GPU's minor number completes the "<minor>.<gi>" name. */
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
        /* A full-GPU UUID for a MIG GPU doesn't say which instances the job owns: map them all. */
        add_gpu_mig_maps(minor, maps, count);
    }
    else
    {
        char map_id[32];

        snprintf(map_id, sizeof(map_id), "%u", minor);
        add_map(maps, count, map_id);
    }
}

/*
 * This job's mapping IDs, from Slurm's allocated-device env var. Run from the job Prolog/Epilog, NVML
 * enumeration is not restricted to the job's devices the way it is inside the job's cgroup, so device
 * identity comes from what Slurm says it allocated rather than from what this process can see -- except
 * in the confined case below.
 */
static int collect_job_maps(char maps[][32], unsigned int *count)
{
    const char *name = NULL;
    const char *env = get_device_env(&name);
    char copy[1024];
    char *saveptr = NULL;
    char *tok;

    *count = 0;

    if (!env)
        return 0; /* no GPUs allocated to this job: nothing to map */

    if (build_node_device_table() != 0)
        return -1;

    if (print_only)
        printf("allocated devices from %s=%s\n", name, env);

    /*
     * Slurm numbers devices across the whole node, but a process confined to the job's cgroup -- run
     * from inside the job, or from a Prolog/Epilog under PrologFlags=RunInJob -- sees NVML enumerate
     * only the job's own devices, renumbered from zero, so node-wide numbers would select the wrong
     * ones. When the allocation names exactly as many devices as NVML can see, the visible set *is*
     * the allocation and the numbers aren't needed. That also holds unconfined for a job holding the
     * whole node, and can't misfire on a partial allocation: a confined view never shows more devices
     * than were allocated.
     */
    if (!is_uuid_list(env) && count_device_entries(env) == node_device_count)
    {
        if (print_only)
            printf("all %u NVML-visible devices are allocated to this job: mapping them directly\n", node_device_count);

        add_all_node_devices(maps, count);
        return 0;
    }

    snprintf(copy, sizeof(copy), "%s", env);

    for (tok = strtok_r(copy, ",", &saveptr); tok; tok = strtok_r(NULL, ",", &saveptr))
        resolve_device_entry(tok, maps, count);

    return 0;
}

static int write_maps(const char *dir, char maps[][32], unsigned int count, const char *value, int chmod_file)
{
    int rc = 0;

    for (unsigned int i = 0; i < count; i++)
    {
        if (write_map(dir, maps[i], value, chmod_file) != 0)
            rc = 1;
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
            run_mode = MODE_INIT;
        else if (strcmp(argv[i], "-prolog") == 0)
            run_mode = MODE_PROLOG;
        else if (strcmp(argv[i], "-epilog") == 0)
            run_mode = MODE_EPILOG;
        else if (strcmp(argv[i], "-print") == 0)
            print_only = 1;
        else if (strcmp(argv[i], "-nonzero") == 0)
            nonzero = 1;
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

        rc = (job_id && *job_id) ? run_job_mode(dir, job_id) : 1;
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
