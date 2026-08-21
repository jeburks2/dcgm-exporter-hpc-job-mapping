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
        "  Devices come from NVML as constrained by the Slurm cgroup, not from\n"
        "  CUDA_VISIBLE_DEVICES or SLURM_JOB_GPUS. Run -prolog and -epilog inside\n"
        "  the job cgroup (task prolog/epilog) so only allocated devices are seen.\n"
        "\n"
        "Environment:\n"
        "  DCGM_HPC_JOB_MAPPING_DIR  Directory for job mapping files\n"
        "  SLURM_JOB_ID              Job ID written by -prolog\n"
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

/* Write or print one mapping file update. chmod_file is only set by -init, so files stay job-user writable. */
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
        chmod(path, 0666);

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

    if (nvmlDeviceGetMigMode(gpu, &current, &pending) != NVML_SUCCESS)
        return 0;

    return current == NVML_DEVICE_MIG_ENABLE;
}

/* Report whether this process may open the GPU's device node — the cgroup allocation test for full GPUs. */
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

    if (nvmlDeviceGetMaxMigDeviceCount(gpu, &max_migs) != NVML_SUCCESS)
        return;

    for (unsigned int mig_index = 0; mig_index < max_migs; mig_index++)
    {
        nvmlDevice_t mig;
        unsigned int gi = 0;
        char map_id[32];

        if (nvmlDeviceGetMigDeviceHandleByIndex(gpu, mig_index, &mig) != NVML_SUCCESS)
            continue;

        if (nvmlDeviceGetGpuInstanceId(mig, &gi) != NVML_SUCCESS)
            continue;

        snprintf(map_id, sizeof(map_id), "%u.%u", gpu_id, gi);
        add_map(maps, count, map_id);
    }
}

/*
 * Build the list of mapping IDs for this node: "<gpu>.<gi>" per MIG instance, or "<gpu>" for a full GPU
 * that this process can open. IDs use the GPU's minor number, not its NVML enumeration position — under
 * a cgroup, NVML renumbers visible devices from 0, so the enumeration index isn't stable across an
 * unrestricted -init run and a restricted -prolog/-epilog run, but the minor number is.
 */
static int collect_maps(char maps[][32], unsigned int *count)
{
    unsigned int gpu_count = 0;

    *count = 0;

    if (nvmlDeviceGetCount_v2(&gpu_count) != NVML_SUCCESS)
        return -1;

    for (unsigned int gpu_index = 0; gpu_index < gpu_count; gpu_index++)
    {
        nvmlDevice_t gpu;
        unsigned int minor = 0;

        if (nvmlDeviceGetHandleByIndex_v2(gpu_index, &gpu) != NVML_SUCCESS)
            continue;

        if (nvmlDeviceGetMinorNumber(gpu, &minor) != NVML_SUCCESS)
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

/* Create root-owned, user-writable files for every full GPU and MIG GPU instance on the node. */
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

    if (collect_maps(maps, &count) != 0)
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

    if (nvmlInit_v2() != NVML_SUCCESS)
        return nonzero ? 1 : 0;

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

    nvmlShutdown();

    /* Without -nonzero, always exit 0 so this program never fails a Slurm prolog/epilog. */
    return nonzero ? rc : 0;
}
