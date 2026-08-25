#include <ctype.h>
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <nvml.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <unistd.h>

#define DEFAULT_JOB_MAPPING_DIR "/var/run/dcgm_job_maps"
#define DEFAULT_SLURMD_SPOOL_DIR "/var/spool/slurmd"
#define MAX_ITEMS 1024
#define MAX_JOB_ID 32

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
    MODE_RESET,
    MODE_PROLOG,
    MODE_EPILOG
};

static int print_only = 0;
static int nonzero = 0;
static enum mode run_mode = MODE_NONE;

static void usage(const char *prog)
{
    printf(
        "Usage: %s -init|-reset|-prolog|-epilog [-print] [-nonzero] [-help]\n"
        "\n"
        "Manage dcgm_exporter Slurm job mapping files.\n"
        "\n"
        "Modes:\n"
        "  -init     Create/chmod mapping files for full GPUs and MIG GPU instances,\n"
        "            resetting each to 0 unless it holds the ID of a job slurmd is\n"
        "            still running. Refuses to run if SLURM_JOB_ID is set\n"
        "  -reset    Delete every mapping file, then recreate them all holding 0.\n"
        "            Refuses to run if SLURM_JOB_ID is set\n"
        "  -prolog   Write SLURM_JOB_ID to the allocated mapping files\n"
        "  -epilog   Reset every mapping file holding SLURM_JOB_ID back to 0\n"
        "\n"
        "Options:\n"
        "  -print    Print what would be written instead of writing files\n"
        "  -nonzero  Return non-zero on errors\n"
        "  -help     Show this help message\n"
        "\n"
        "Device selection:\n"
        "  -prolog reads the job's allocated devices from SLURM_JOB_GPUS or\n"
        "  CUDA_VISIBLE_DEVICES. Each entry is a GPU/MIG instance UUID, or a Slurm gres\n"
        "  index -- a position in the node's device list, where a MIG-mode GPU\n"
        "  contributes one entry per GPU instance rather than one for the whole GPU.\n"
        "  -epilog selects no devices: it clears whichever files hold SLURM_JOB_ID.\n"
        "\n"
        "Running jobs:\n"
        "  -init keeps a mapping whose job is still running on the node, found by\n"
        "  listing slurmd's spool directory -- the same local, RPC-free check\n"
        "  `scontrol listjobs` makes. When that directory cannot be read, every\n"
        "  non-zero mapping is kept: a stale ID is corrected by the next job's\n"
        "  prolog, an erased one is not. Use -reset to clear them regardless.\n"
        "\n"
        "Environment:\n"
        "  DCGM_HPC_JOB_MAPPING_DIR  Directory for job mapping files\n"
        "  DCGM_SLURMD_SPOOL_DIR     slurmd spool dir, overriding SlurmdSpoolDir from\n"
        "                            slurm.conf (-init)\n"
        "  SLURM_CONF                slurm.conf to read SlurmdSpoolDir from (-init)\n"
        "  SLURM_JOB_ID              Job ID written by -prolog, matched by -epilog\n"
        "  SLURM_JOB_GPUS            Devices allocated to the job (-prolog)\n"
        "  CUDA_VISIBLE_DEVICES      Preferred over SLURM_JOB_GPUS when it holds UUIDs,\n"
        "                            and used as a fallback when SLURM_JOB_GPUS is unset\n"
        "\n"
        "Default mapping dir: %s\n"
        "Default slurmd spool dir: %s\n",
        prog,
        DEFAULT_JOB_MAPPING_DIR,
        DEFAULT_SLURMD_SPOOL_DIR);
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

/* Read one mapping file's contents, trimmed of the trailing newline write_map() adds. */
static int read_map(const char *dir, const char *map_id, char *buf, size_t len)
{
    char path[512];
    FILE *f;
    size_t n;

    snprintf(path, sizeof(path), "%s/%s", dir, map_id);

    f = fopen(path, "r");
    if (!f)
        return -1;

    n = fread(buf, 1, len - 1, f);
    fclose(f);
    buf[n] = '\0';

    while (n > 0 && isspace((unsigned char)buf[n - 1]))
        buf[--n] = '\0';

    return 0;
}

/* Apply -init's permissions to a file it is leaving untouched otherwise. */
static void chmod_map(const char *dir, const char *map_id)
{
    char path[512];

    snprintf(path, sizeof(path), "%s/%s", dir, map_id);
    chmod(path, 0644);
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

/* Every device on the node, for -init. -prolog uses collect_job_maps() instead. */
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
 * no GPU).
 *
 * Slurm's Prolog/Epilog environment gives the allocation as device numbers, not UUIDs -- even for a MIG
 * allocation, where the job itself would later see a MIG-<uuid> in CUDA_VISIBLE_DEVICES. SLURM_JOB_GPUS
 * is the authoritative form of that numbering, so it wins; CUDA_VISIBLE_DEVICES takes priority only when
 * it holds UUIDs, which name the exact device and need no numbering assumption at all.
 */
static const char *get_device_env(void)
{
    const char *cvd = getenv("CUDA_VISIBLE_DEVICES");
    const char *job_gpus = getenv("SLURM_JOB_GPUS");

    if (cvd && *cvd && is_uuid_list(cvd))
        return cvd;

    if (job_gpus && *job_gpus)
        return job_gpus;

    if (cvd && *cvd)
        return cvd;

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

/* This job's mapping IDs, from Slurm's allocated-device env var (-prolog). */
static int collect_job_maps(char maps[][32], unsigned int *count)
{
    const char *env = get_device_env();
    char copy[1024];
    char *saveptr = NULL;
    char *tok;

    *count = 0;

    if (!env)
        return 0; /* no GPUs allocated to this job: nothing to map */

    if (build_node_device_table() != 0)
        return -1;

    /*
     * Under PrologFlags=RunInJob the prolog runs inside the job's cgroup, where NVML enumerates only
     * the job's own devices. That view is exact, so when it holds as many devices as the allocation
     * names, map all of them and skip the numbering entirely. This is the reliable path: Slurm's
     * device numbers are not always positions in the node's device list -- a job allocated GPU
     * instances 3 and 5 of a mixed-profile MIG GPU is handed "0,1" -- so the fallback below can pick
     * the wrong instance. The check cannot misfire: a confined view never shows more devices than
     * were allocated, and unconfined an equal count means the job holds the whole node.
     */
    if (!is_uuid_list(env) && count_device_entries(env) == node_device_count)
    {
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

/*
 * Job IDs slurmd is currently running on this node, filled in by load_running_jobs().
 *
 * `scontrol listjobs` answers this without a slurmctld RPC by listing SlurmdSpoolDir, and -init does the
 * same: slurmd keeps one unix socket per step there, named "<node>_<jobid>.<stepid>", and gives a batch
 * job its own "job<jobid>" directory. One getdents pass, no daemon contacted, no lock taken.
 */
static char running_jobs[MAX_ITEMS][MAX_JOB_ID];
static unsigned int running_job_count = 0;

/* Read SlurmdSpoolDir out of a slurm.conf, ignoring a trailing comment. */
static int conf_lookup_spool_dir(const char *conf_path, char *out, size_t len)
{
    FILE *f = fopen(conf_path, "r");
    char line[1024];
    int found = 0;

    if (!f)
        return -1;

    while (!found && fgets(line, sizeof(line), f))
    {
        char *p = line;
        char *value;

        while (isspace((unsigned char)*p))
            p++;

        if (strncasecmp(p, "SlurmdSpoolDir", 14) != 0)
            continue;

        p += 14;
        while (isspace((unsigned char)*p))
            p++;

        if (*p++ != '=')
            continue;

        while (isspace((unsigned char)*p))
            p++;

        value = p;
        while (*p && !isspace((unsigned char)*p) && *p != '#')
            p++;
        *p = '\0';

        if (!*value)
            continue;

        snprintf(out, len, "%s", value);
        found = 1;
    }

    fclose(f);
    return found ? 0 : -1;
}

/* Substitute the %n/%h node-name escapes slurm.conf allows in a path. */
static void expand_conf_path(const char *raw, char *out, size_t len)
{
    char host[256];
    char *dot;
    size_t o = 0;

    if (gethostname(host, sizeof(host)) != 0)
        host[0] = '\0';

    host[sizeof(host) - 1] = '\0';
    if ((dot = strchr(host, '.')) != NULL)
        *dot = '\0';

    for (const char *p = raw; *p && o + 1 < len; p++)
    {
        if (*p == '%' && (p[1] == 'n' || p[1] == 'h'))
        {
            for (const char *h = host; *h && o + 1 < len; h++)
                out[o++] = *h;

            p++;
            continue;
        }

        out[o++] = *p;
    }

    out[o] = '\0';
}

/*
 * Where slurmd keeps its per-step sockets. Configless clusters have no /etc/slurm/slurm.conf, so the
 * cached copy slurmd writes under /run is checked too -- the same two locations scontrol tries.
 */
static const char *get_spool_dir(void)
{
    static char resolved[512];
    const char *env = getenv("DCGM_SLURMD_SPOOL_DIR");
    const char *conf = getenv("SLURM_CONF");
    char raw[512];

    if (env && *env)
        return env;

    if (resolved[0])
        return resolved;

    if ((conf && *conf && conf_lookup_spool_dir(conf, raw, sizeof(raw)) == 0) ||
        conf_lookup_spool_dir("/etc/slurm/slurm.conf", raw, sizeof(raw)) == 0 ||
        conf_lookup_spool_dir("/run/slurm/conf/slurm.conf", raw, sizeof(raw)) == 0)
        expand_conf_path(raw, resolved, sizeof(resolved));
    else
        snprintf(resolved, sizeof(resolved), "%s", DEFAULT_SLURMD_SPOOL_DIR);

    return resolved;
}

/* "<node>_<jobid>.<stepid>" -> jobid. A node name may contain '_', so the split is the last one. */
static int step_socket_job_id(const char *name, char *out, size_t len)
{
    const char *sep = strrchr(name, '_');
    const char *end;
    size_t n;

    if (!sep)
        return -1;

    sep++;
    for (end = sep; isdigit((unsigned char)*end); end++)
        ;

    if (end == sep || *end != '.' || !all_digits(end + 1))
        return -1;

    n = (size_t)(end - sep);
    if (n >= len)
        return -1;

    snprintf(out, n + 1, "%s", sep);
    return 0;
}

/* "job<jobid>" -> jobid. slurmd zero-pads the name to five digits, so 42 arrives as "job00042". */
static int batch_dir_job_id(const char *name, char *out, size_t len)
{
    const char *digits = name + 3;

    if (strncmp(name, "job", 3) != 0 || !all_digits(digits))
        return -1;

    while (digits[0] == '0' && digits[1])
        digits++;

    if (strlen(digits) >= len)
        return -1;

    snprintf(out, len, "%s", digits);
    return 0;
}

static void add_running_job(const char *job_id)
{
    for (unsigned int i = 0; i < running_job_count; i++)
    {
        if (strcmp(running_jobs[i], job_id) == 0)
            return;
    }

    if (running_job_count >= MAX_ITEMS)
        return;

    snprintf(running_jobs[running_job_count], sizeof(running_jobs[0]), "%s", job_id);
    running_job_count++;
}

/*
 * Collect this node's running jobs. Returns -1 when the spool directory can't be read, which the caller
 * has to tell apart from a node that is simply idle: an empty list and an unusable one mean the opposite
 * thing for every mapping file on the node.
 */
static int load_running_jobs(void)
{
    const char *spool = get_spool_dir();
    DIR *d = opendir(spool);
    struct dirent *ent;

    running_job_count = 0;

    if (!d)
    {
        fprintf(stderr, "dcgm-job-map: cannot list slurmd spool dir %s: %s\n", spool, strerror(errno));
        return -1;
    }

    while ((ent = readdir(d)) != NULL)
    {
        char job_id[MAX_JOB_ID];

        if (ent->d_name[0] == '.')
            continue;

        if (step_socket_job_id(ent->d_name, job_id, sizeof(job_id)) == 0 ||
            batch_dir_job_id(ent->d_name, job_id, sizeof(job_id)) == 0)
            add_running_job(job_id);
    }

    closedir(d);
    return 0;
}

static int job_is_running(const char *job_id)
{
    for (unsigned int i = 0; i < running_job_count; i++)
    {
        if (strcmp(running_jobs[i], job_id) == 0)
            return 1;
    }

    return 0;
}

/*
 * Reset every mapping file holding this job's ID (-epilog). Deliberately identity-based rather than
 * device-based: the epilog is not confined to the job's cgroup the way a PrologFlags=RunInJob prolog
 * is, so it sees a different device list and would resolve the allocation to different files than the
 * prolog wrote -- clearing the wrong ones and stranding a job ID on the right ones. Matching on the ID
 * clears exactly what the prolog wrote, needs no NVML, and cannot be thrown off by device numbering.
 */
static int clear_job_maps(const char *dir, const char *job_id)
{
    DIR *d = opendir(dir);
    struct dirent *ent;
    int rc = 0;

    if (!d)
        return -1;

    while ((ent = readdir(d)) != NULL)
    {
        char buf[64];

        if (ent->d_name[0] == '.')
            continue;

        if (read_map(dir, ent->d_name, buf, sizeof(buf)) != 0)
            continue;

        if (strcmp(buf, job_id) != 0)
            continue;

        if (write_map(dir, ent->d_name, "0", 0) != 0)
            rc = 1;
    }

    closedir(d);
    return rc;
}

/*
 * Create root-owned, world-readable files for every full GPU and MIG GPU instance on the node, resetting
 * each to 0 -- except one already holding the ID of a job slurmd is still running. Restarting
 * dcgm-exporter re-runs -init under a live allocation, and zeroing that mapping would drop the job's
 * metrics for the rest of its run: nothing writes the ID again until the job's own epilog clears it.
 */
static int run_init(const char *dir)
{
    char maps[MAX_ITEMS][32];
    unsigned int count = 0;
    int jobs_known;
    int rc = 0;

    if (collect_maps(maps, &count) != 0)
        return -1;

    if (!print_only && mkdir(dir, 0755) != 0 && errno != EEXIST)
        return -1;

    /*
     * With no job list to check against, keep every non-zero ID rather than risk erasing a live one. A
     * stale ID survives only until the next job lands on that device and its prolog overwrites it; an
     * erased one is gone for the rest of the job it belonged to. -reset is the way to clear them anyway.
     */
    jobs_known = load_running_jobs() == 0;
    if (!jobs_known)
        rc = 1;

    for (unsigned int i = 0; i < count; i++)
    {
        char held[64];

        if (read_map(dir, maps[i], held, sizeof(held)) == 0 && *held && strcmp(held, "0") != 0 &&
            (!jobs_known || job_is_running(held)))
        {
            if (print_only)
                printf("would keep \"%s\" in %s/%s\n", held, dir, maps[i]);
            else
                chmod_map(dir, maps[i]);

            continue;
        }

        if (write_map(dir, maps[i], "0", 1) != 0)
            rc = 1;
    }

    return rc;
}

/* Delete every mapping file, so a device that no longer exists leaves none behind. */
static int remove_all_maps(const char *dir)
{
    DIR *d = opendir(dir);
    struct dirent *ent;
    int rc = 0;

    if (!d)
        return errno == ENOENT ? 0 : -1;

    while ((ent = readdir(d)) != NULL)
    {
        char path[512];
        struct stat st;

        if (ent->d_name[0] == '.')
            continue;

        snprintf(path, sizeof(path), "%s/%s", dir, ent->d_name);

        /* Mapping files are plain files; leave anything else in the directory alone. */
        if (stat(path, &st) != 0 || !S_ISREG(st.st_mode))
            continue;

        if (print_only)
        {
            printf("would remove %s\n", path);
            continue;
        }

        if (unlink(path) != 0)
            rc = 1;
    }

    closedir(d);
    return rc;
}

/*
 * Discard every mapping, running job or not, and recreate the files from the node's current devices
 * (-reset). This is -init without the running-job check, plus the delete pass that clears out files for
 * devices that are gone -- the mode to run after a MIG reconfiguration, or to fix mappings by hand.
 */
static int run_reset(const char *dir)
{
    char maps[MAX_ITEMS][32];
    unsigned int count = 0;
    int rc = 0;

    if (collect_maps(maps, &count) != 0)
        return -1;

    if (!print_only && mkdir(dir, 0755) != 0 && errno != EEXIST)
        return -1;

    if (remove_all_maps(dir) != 0)
        rc = 1;

    if (write_maps(dir, maps, count, "0", 1) != 0)
        rc = 1;

    return rc;
}

/* Write SLURM_JOB_ID to the mapping files for this job's allocated devices (-prolog). */
static int run_prolog(const char *dir, const char *job_id)
{
    char maps[MAX_ITEMS][32];
    unsigned int count = 0;

    if (collect_job_maps(maps, &count) != 0)
        return -1;

    return write_maps(dir, maps, count, job_id, 0);
}

int main(int argc, char **argv)
{
    int rc = 0;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-init") == 0)
            run_mode = MODE_INIT;
        else if (strcmp(argv[i], "-reset") == 0)
            run_mode = MODE_RESET;
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

    /*
     * -init and -reset need the whole node; inside a job's cgroup they would only see that job's
     * devices, creating files for one job's slice and resetting every other device on the node.
     */
    if (run_mode == MODE_INIT || run_mode == MODE_RESET)
    {
        const char *job_id = getenv("SLURM_JOB_ID");

        if (job_id && *job_id)
        {
            fprintf(stderr,
                    "%s must not run inside a Slurm job (SLURM_JOB_ID=%s is set)\n",
                    run_mode == MODE_INIT ? "-init" : "-reset",
                    job_id);
            return nonzero ? 1 : 0;
        }
    }

    const char *dir = get_mapping_dir();

    /* -epilog matches on the job ID, so it needs no NVML and runs even where the driver is broken. */
    if (run_mode == MODE_EPILOG)
    {
        const char *job_id = getenv("SLURM_JOB_ID");

        if (!job_id || !*job_id)
        {
            fprintf(stderr, "-epilog requires SLURM_JOB_ID\n");
            return nonzero ? 1 : 0;
        }

        rc = clear_job_maps(dir, job_id);
        return nonzero ? rc : 0;
    }

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
    else if (run_mode == MODE_RESET)
    {
        rc = run_reset(dir);
    }
    else if (run_mode == MODE_PROLOG)
    {
        const char *job_id = getenv("SLURM_JOB_ID");

        rc = (job_id && *job_id) ? run_prolog(dir, job_id) : 1;
    }

    nvmlShutdown_p();
    dlclose(nvml_lib);

    /* Without -nonzero, always exit 0 so this program never fails a Slurm prolog/epilog. */
    return nonzero ? rc : 0;
}
