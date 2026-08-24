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
    MODE_EPILOG,
    MODE_LIST
};

/*
 * How to read a plain number in the allocated-device list. Slurm numbers the GPU gres on a node
 * across every device it can allocate, so on a MIG node each *GPU instance* consumes one number
 * and a full GPU's NVML enumeration ordinal is not that number. ORDINAL_GRES resolves a number
 * through the node device table below (Slurm's numbering); ORDINAL_GPU is the old behavior, a
 * direct NVML enumeration index, kept as an escape hatch for sites whose gres.conf lists whole
 * GPUs on MIG-enabled nodes.
 */
enum ordinal_mode
{
    ORDINAL_GRES,
    ORDINAL_GPU
};

static int print_only = 0;
static int nonzero = 0;
static enum mode run_mode = MODE_NONE;
static enum ordinal_mode ordinal_mode = ORDINAL_GRES;

/* Print command usage and behavior. */
static void usage(const char *prog)
{
    printf(
        "Usage: %s -init|-prolog|-epilog|-list [-print] [-nonzero] [-help]\n"
        "\n"
        "Manage dcgm_exporter Slurm job mapping files.\n"
        "\n"
        "Modes:\n"
        "  -init     Create/chmod mapping files for full GPUs and MIG GPU instances.\n"
        "            Refuses to run if SLURM_JOB_ID is set; it must see every device\n"
        "            on the node, not one job's allocated slice of it.\n"
        "  -prolog   Write SLURM_JOB_ID to the allocated mapping files\n"
        "  -epilog   Truncate allocated mapping files and write 0\n"
        "  -list     Print the node device table: the Slurm gres index of every\n"
        "            allocatable device and the mapping file it resolves to\n"
        "\n"
        "Options:\n"
        "  -print    Print what would be written instead of writing files\n"
        "  -nonzero  Return non-zero on errors\n"
        "  -help     Show this help message\n"
        "\n"
        "Device selection:\n"
        "  -prolog and -epilog run from the Slurm job Prolog/Epilog (slurmd, once\n"
        "  per allocation, outside the job's cgroup) and read the devices Slurm\n"
        "  allocated to the job from SLURM_JOB_GPUS, CUDA_VISIBLE_DEVICES or\n"
        "  GPU_DEVICE_ORDINAL. Each entry is either a GPU/MIG instance UUID or a\n"
        "  Slurm gres index -- a position in the node device table (-list), where a\n"
        "  MIG-mode GPU contributes one entry per GPU instance rather than one entry\n"
        "  for the whole GPU.\n"
        "\n"
        "Environment:\n"
        "  DCGM_HPC_JOB_MAPPING_DIR  Directory for job mapping files\n"
        "  SLURM_JOB_ID              Job ID written by -prolog\n"
        "  SLURM_JOB_GPUS            Devices allocated to the job (-prolog/-epilog)\n"
        "  CUDA_VISIBLE_DEVICES      Preferred over SLURM_JOB_GPUS when it holds UUIDs\n"
        "  GPU_DEVICE_ORDINAL        Fallback device list if neither above is set\n"
        "  DCGM_HPC_JOB_MAPPING_ORDINALS\n"
        "                            gres (default) or gpu: how to read a plain\n"
        "                            number in the device list\n"
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
 * One allocatable device on the node: a full GPU, or a single MIG GPU instance of one. The table
 * below holds these in NVML enumeration order -- GPUs by NVML index, and within a MIG-mode GPU its
 * instances by MIG index -- which is the order Slurm's gres/gpu NVML autodetect builds the node's
 * GPU gres list in, and therefore the order its device numbering follows.
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

/* Name the mapping file for one node device: "<minor>.<gi>" for a MIG instance, "<minor>" otherwise. */
static void device_map_id(const struct node_device *dev, char *buf, size_t len)
{
    if (dev->is_mig)
        snprintf(buf, len, "%u.%u", dev->minor, dev->gi);
    else
        snprintf(buf, len, "%u", dev->minor);
}

/* Append one device to the node table, silently dropping it if the fixed-size array is full. */
static void add_node_device(unsigned int minor, int is_mig, unsigned int gi)
{
    if (node_device_count >= MAX_ITEMS)
        return;

    node_devices[node_device_count].minor = minor;
    node_devices[node_device_count].is_mig = is_mig;
    node_devices[node_device_count].gi = gi;
    node_device_count++;
}

/* Add every MIG GPU instance NVML exposes for one GPU to the node table, in MIG index order. */
static void add_gpu_migs(nvmlDevice_t gpu, unsigned int minor)
{
    unsigned int max_migs = 0;

    if (nvmlDeviceGetMaxMigDeviceCount_p(gpu, &max_migs) != NVML_SUCCESS)
        return;

    for (unsigned int mig_index = 0; mig_index < max_migs; mig_index++)
    {
        nvmlDevice_t mig;
        unsigned int gi = 0;

        /* max_migs is the profile-independent upper bound, so unpopulated slots are expected here. */
        if (nvmlDeviceGetMigDeviceHandleByIndex_p(gpu, mig_index, &mig) != NVML_SUCCESS)
            continue;

        if (nvmlDeviceGetGpuInstanceId_p(mig, &gi) != NVML_SUCCESS)
            continue;

        add_node_device(minor, 1, gi);
    }
}

/*
 * Enumerate every allocatable device on the node once, into node_devices[]. Devices are identified by
 * the parent GPU's minor number (its /dev/nvidia<minor> node), not its NVML enumeration position, so
 * every mode agrees on a file name for the same physical device regardless of enumeration order.
 */
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

/* Add every MIG GPU instance of one GPU to a map list, for an entry that names the GPU but not an instance. */
static void collect_gpu_migs(unsigned int minor, char maps[][32], unsigned int *count)
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

/*
 * Build the list of mapping IDs for every device on the node: "<gpu>.<gi>" per MIG instance, or "<gpu>"
 * for a full GPU that this process can open. Used only by -init, which must see every device on the node;
 * -prolog and -epilog use collect_job_maps() instead, below.
 */
static int collect_maps(char maps[][32], unsigned int *count)
{
    *count = 0;

    if (build_node_device_table() != 0)
        return -1;

    for (unsigned int i = 0; i < node_device_count; i++)
    {
        char map_id[32];

        if (!node_devices[i].is_mig && !device_node_accessible(node_devices[i].minor))
            continue;

        device_map_id(&node_devices[i], map_id, sizeof(map_id));
        add_map(maps, count, map_id);
    }

    return 0;
}

/* Print the node device table with the Slurm gres index of each device (-list). */
static int run_list(const char *dir)
{
    if (build_node_device_table() != 0)
        return -1;

    printf("Slurm gres index -> mapping file (%s)\n", dir);

    for (unsigned int i = 0; i < node_device_count; i++)
    {
        char map_id[32];

        device_map_id(&node_devices[i], map_id, sizeof(map_id));

        if (node_devices[i].is_mig)
            printf("  %u\t%s\t(MIG instance %u on /dev/nvidia%u)\n", i, map_id, node_devices[i].gi, node_devices[i].minor);
        else
            printf("  %u\t%s\t(full GPU /dev/nvidia%u)\n", i, map_id, node_devices[i].minor);
    }

    if (node_device_count == 0)
        printf("  (no devices found)\n");

    return 0;
}

/* Report whether every character of s is a digit, i.e. s is a plain device number. */
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
 * Report whether a device list names devices by UUID rather than by number. Every UUID carries its
 * "GPU-"/"MIG-" prefix, so a letter anywhere is the tell; a numeric list holds only digits, commas
 * and the '-' of a range.
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

/* Count the devices a numeric list names, expanding ranges. Used only to compare against what NVML sees. */
static unsigned int count_device_entries(const char *env)
{
    char copy[1024];
    char *saveptr = NULL;
    char *tok;
    unsigned int total = 0;

    snprintf(copy, sizeof(copy), "%s", env);

    for (tok = strtok_r(copy, ",", &saveptr); tok; tok = strtok_r(NULL, ",", &saveptr))
    {
        const char *dash = strchr(tok, '-');

        if (dash && dash != tok && all_digits(dash + 1))
        {
            unsigned long lo = strtoul(tok, NULL, 10);
            unsigned long hi = strtoul(dash + 1, NULL, 10);

            if (hi >= lo)
                total += (unsigned int)(hi - lo + 1);
        }
        else if (all_digits(tok))
        {
            total++;
        }
    }

    return total;
}

/*
 * Return the env var listing devices Slurm allocated to this job, or NULL if none is set (e.g. a job
 * that did not request a GPU). *name, when non-NULL, receives the variable the list came from.
 *
 * In the job Prolog/Epilog environment Slurm exports the allocation as device numbers, not UUIDs --
 * even for a MIG allocation, where the job itself would later see a MIG-<uuid> in CUDA_VISIBLE_DEVICES.
 * SLURM_JOB_GPUS is the authoritative form of that numbering, so it is preferred; CUDA_VISIBLE_DEVICES
 * wins only when it holds UUIDs, which identify the exact device and need no numbering assumption at
 * all. GPU_DEVICE_ORDINAL is a last fallback for sites where only it is configured.
 */
static const char *get_device_env(const char **name)
{
    const char *cvd = getenv("CUDA_VISIBLE_DEVICES");
    const char *job_gpus = getenv("SLURM_JOB_GPUS");
    const char *ordinal = getenv("GPU_DEVICE_ORDINAL");

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

    if (ordinal && *ordinal)
    {
        if (name)
            *name = "GPU_DEVICE_ORDINAL";
        return ordinal;
    }

    return NULL;
}

/*
 * Resolve one device number into the mapping ID it names. Under the default ORDINAL_GRES this is a
 * position in the node device table, which is how Slurm numbers the GPU gres it can allocate: on a
 * MIG-mode GPU each GPU instance is its own allocatable device and gets its own number, so number 0
 * on a node whose first GPU is partitioned is that GPU's first instance, not the whole GPU. Under
 * ORDINAL_GPU it is an NVML enumeration index, naming a whole GPU (and with it every instance on it).
 */
static void resolve_device_number(unsigned int index, char maps[][32], unsigned int *count)
{
    nvmlDevice_t dev;
    unsigned int minor = 0;
    char map_id[32];

    if (ordinal_mode == ORDINAL_GRES)
    {
        if (index >= node_device_count)
        {
            fprintf(stderr,
                    "dcgm-job-map: device %u is past the end of the node device table (%u devices); "
                    "Slurm numbers devices across the whole node, so this usually means NVML is only "
                    "showing part of it -- run from the job Prolog/Epilog, not from inside the job\n",
                    index, node_device_count);
            return;
        }

        device_map_id(&node_devices[index], map_id, sizeof(map_id));
        add_map(maps, count, map_id);
        return;
    }

    if (nvmlDeviceGetHandleByIndex_v2_p(index, &dev) != NVML_SUCCESS)
        return;

    if (nvmlDeviceGetMinorNumber_p(dev, &minor) != NVML_SUCCESS)
        return;

    if (mig_enabled(dev))
    {
        collect_gpu_migs(minor, maps, count);
        return;
    }

    snprintf(map_id, sizeof(map_id), "%u", minor);
    add_map(maps, count, map_id);
}

/*
 * Resolve one entry of the allocated-device list into the mapping ID(s) it names, appending them to
 * maps/count. An entry is either a device number (see resolve_device_number()), an inclusive range of
 * them ("2-5"), or a UUID. nvmlDeviceGetHandleByUUID accepts both a full GPU's UUID and a MIG instance's
 * UUID, resolving straight to the matching handle. Once resolved, nvmlDeviceGetGpuInstanceId succeeding
 * is what tells a MIG instance handle apart from a full GPU handle, since only a MIG device has an
 * instance ID at all.
 */
static void resolve_device_entry(const char *entry, char maps[][32], unsigned int *count)
{
    nvmlDevice_t dev;
    unsigned int minor = 0;
    unsigned int gi = 0;
    const char *dash;

    if (!*entry)
        return;

    if (all_digits(entry))
    {
        resolve_device_number((unsigned int)strtoul(entry, NULL, 10), maps, count);
        return;
    }

    /* A range like "0-3"; a UUID also contains '-', so both sides must be plain numbers. */
    dash = strchr(entry, '-');
    if (dash && dash != entry && all_digits(dash + 1))
    {
        char first[16];
        size_t len = (size_t)(dash - entry);

        if (len < sizeof(first))
        {
            snprintf(first, len + 1, "%s", entry);

            if (all_digits(first))
            {
                unsigned long lo = strtoul(first, NULL, 10);
                unsigned long hi = strtoul(dash + 1, NULL, 10);

                for (unsigned long i = lo; i <= hi && i < MAX_ITEMS; i++)
                    resolve_device_number((unsigned int)i, maps, count);

                return;
            }
        }
    }

    if (nvmlDeviceGetHandleByUUID_p(entry, &dev) != NVML_SUCCESS)
        return;

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
        /* Named by a full-GPU UUID rather than an instance UUID: the allocation doesn't tell us which
         * instance(s) of this MIG GPU the job owns, so map every instance on it. */
        collect_gpu_migs(minor, maps, count);
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
 * Used by -prolog and -epilog, which normally run from the Slurm job Prolog/Epilog: once per allocation,
 * via slurmd as root, outside the job's cgroup. That means neither NVML enumeration nor a device-node
 * open() test is restricted to this job's devices the way it would be inside the job's cgroup, so unlike
 * collect_maps() above, device identity has to come from what Slurm tells us it allocated, not from what
 * this process can see -- except in the confined case handled below.
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
     * Slurm numbers devices across the whole node, but a process confined to the job's cgroup -- this
     * program run from inside the job, or from a Prolog/Epilog under PrologFlags=RunInJob -- sees NVML
     * enumerate only the job's own devices, so those node-wide numbers index the wrong table (and run
     * off the end of it). When the allocation names exactly as many devices as NVML can see, though,
     * the visible set *is* the allocation and the numbers are not needed: map everything visible. That
     * is equally true unconfined for a job holding every device on the node, and it cannot misfire in
     * the partial-allocation case, since a confined view never shows more devices than were allocated.
     */
    if (!is_uuid_list(env) && count_device_entries(env) == node_device_count)
    {
        if (print_only)
            printf("all %u NVML-visible devices are allocated to this job: mapping them directly\n", node_device_count);

        for (unsigned int i = 0; i < node_device_count; i++)
        {
            char map_id[32];

            device_map_id(&node_devices[i], map_id, sizeof(map_id));
            add_map(maps, count, map_id);
        }

        return 0;
    }

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
        else if (strcmp(argv[i], "-list") == 0)
        {
            run_mode = MODE_LIST;
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
    const char *ordinals = getenv("DCGM_HPC_JOB_MAPPING_ORDINALS");

    if (ordinals && strcmp(ordinals, "gpu") == 0)
        ordinal_mode = ORDINAL_GPU;

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
    else if (run_mode == MODE_LIST)
    {
        rc = run_list(dir);
    }

    nvmlShutdown_p();
    dlclose(nvml_lib);

    /* Without -nonzero, always exit 0 so this program never fails a Slurm prolog/epilog. */
    return nonzero ? rc : 0;
}
