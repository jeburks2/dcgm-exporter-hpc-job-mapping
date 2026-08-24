# DCGM Exporter HPC Job Mapping

`dcgm-job-map` is a small C utility for integrating [DCGM Exporter](https://github.com/NVIDIA/dcgm-exporter) with Slurm. It maintains per-GPU or per-MIG GPU instance mapping files containing the Slurm job ID currently using each device.

The mapping files can be consumed by a DCGM Exporter job-mapping configuration so exported GPU metrics can be associated with HPC jobs.

## Requirements

Build time:

- Linux
- GCC or another C compiler compatible with the Makefile
- CUDA toolkit headers, for `nvml.h`

Run time:

- NVIDIA driver providing `libnvidia-ml.so.1`, on any node where GPU mapping should actually happen
- Slurm's `gres/gpu` plugin configured so `SLURM_JOB_GPUS` (or `CUDA_VISIBLE_DEVICES`) is exported to the job `Prolog` and `Epilog`

`libnvidia-ml.so.1` is loaded with `dlopen` at run time rather than linked at build time, so the same binary starts cleanly on GPU-less nodes (login nodes, CPU-only compute nodes) and simply does nothing: `-init` and `-prolog`/`-epilog` find no NVML library, skip GPU mapping, and exit `0`.

## Build

```sh
make
```

By default, the build expects CUDA at `/usr/local/cuda`, for `nvml.h` and the NVML type/constant declarations only — the resulting binary does not link against `libnvidia-ml`. If cuda is installed elsewhere, set `CUDA_HOME` to the correct path before running `make`. For example:

```sh
CUDA_HOME=/opt/cuda-13.2 make
```

Install the utility to `/usr/local/sbin`:

```sh
sudo make install
```

Remove the built executable with:

```sh
make clean
```

## Modes

### Initialize mappings

Run on boot or whenever the available MIG configuration changes:

```sh
sudo dcgm-job-map -init
```

This discovers every GPU on the node through NVML and creates one mapping file per device. A GPU in MIG mode gets one file per GPU instance, named `<gpu-index>.<gpu-instance-id>`; a GPU not in MIG mode gets a single file named `<gpu-index>`. Files are initialized with `0`, owned by root, and made world-readable (`0644`) — `-prolog` and `-epilog` run as root too (see below), so the files no longer need to be writable by the job user. The mapping directory is created during this mode if it does not already exist.

`-init` refuses to run when `SLURM_JOB_ID` is set, i.e. inside a Slurm job. It must see every device on the node to build a complete mapping; running it inside a job would restrict NVML and the device-node accessibility check to that job's allocated devices, quietly narrowing the node-wide file set down to one job's slice and resetting mapping files for every other device on the node in the process. Run `-init` at boot, or from an `ExecStartPre` as shown below, never from a job's prolog or epilog.

The `DCGM_HPC_JOB_MAPPING_DIR` environment variable specifies the directory where mapping files are created. If not set, the default is `/var/run/dcgm_job_maps`. The `-init` mode creates the final directory automatically with mode `0755` when it does not exist. Its parent directory must already exist, and the process must have permission to create it.

To initialize the mapping files automatically when the exporter starts, create a systemd drop-in for `nvidia-dcgm-exporter.service`:

```sh
sudo install -d /etc/systemd/system/nvidia-dcgm-exporter.service.d
sudo tee /etc/systemd/system/nvidia-dcgm-exporter.service.d/hpc-job-map.conf > /dev/null <<'EOF'
[Service]
Environment=DCGM_HPC_JOB_MAPPING_DIR=/var/run/dcgm_job_maps
ExecStartPre=/usr/local/sbin/dcgm-job-map -init
EOF

sudo systemctl daemon-reload
sudo systemctl restart nvidia-dcgm-exporter.service
```

Verify the drop-in and exporter status with:

```sh
systemctl cat nvidia-dcgm-exporter.service
systemctl status nvidia-dcgm-exporter.service
```

The same `DCGM_HPC_JOB_MAPPING_DIR` value must be available to the Slurm prolog and epilog commands so they update the files consumed by DCGM Exporter.

### Slurm job prolog

Configure as the Slurm **job** prolog — `Prolog` in `slurm.conf`, not `TaskProlog`:

```ini
Prolog=/usr/local/sbin/dcgm-job-map -prolog
```

The utility reads `SLURM_JOB_GPUS` (falling back to `CUDA_VISIBLE_DEVICES`) to find the devices Slurm allocated to the job, and writes `SLURM_JOB_ID` to the matching mapping files. See [Device selection](#device-selection) for how each entry is resolved.

By default `Prolog` runs when the job's first step launches on the node. To have the mapping written as soon as the allocation is granted instead — e.g. for an interactive `salloc` session that may never launch a step — add `PrologFlags=Alloc` in `slurm.conf`.

### Slurm job epilog

Configure as the Slurm **job** epilog — `Epilog` in `slurm.conf`, not `TaskEpilog`:

```ini
Epilog=/usr/local/sbin/dcgm-job-map -epilog
```

This uses the same device selection and resets the selected mapping files to `0`.

**Use the job prolog/epilog, not the task prolog/epilog.** `TaskProlog`/`TaskEpilog` run once per *task*, inside `slurmstepd`, for every job step — including `srun` steps launched from within an existing allocation. A job that runs several `srun` steps in sequence (or an interactive `salloc` session followed by `srun`) would have its mapping file reset to `0` by `TaskEpilog` as soon as the first step finishes, even though the allocation — and the rest of the job — is still running:

```text
$ salloc -G 2 -n 2
$ cat /var/run/dcgm_job_maps/*
1474054
1474054
$ srun -G 1 ./train.py
... (runs fine) ...
$ cat /var/run/dcgm_job_maps/*
0        # <- wrong: the allocation is still active, but TaskEpilog already reset this
0
```

`Prolog` and `Epilog` instead run once per **allocation**, via `slurmd` as root, so they fire once when the job starts and once when it ends — not once per step — which is what this tool needs.

## Options

```text
-init      Create/reset mapping files for every full GPU and MIG GPU instance
           on the node; refuses to run inside a Slurm job
-prolog    Write SLURM_JOB_ID to the mapping files for this job's allocated devices
-epilog    Reset the mapping files for this job's allocated devices to 0
-print     Print intended writes without changing files
-nonzero   Return non-zero for invalid arguments, NVML errors, or mapping errors
-help      Show usage information
```

Without `-nonzero`, the process always exits with status `0`. This will ensure that Slurm does not fail the job if the mapping utility fails.

For a safe dry run, set `SLURM_JOB_GPUS` the way Slurm would inside a job's `Prolog`/`Epilog`:

```sh
SLURM_JOB_ID=12345 SLURM_JOB_GPUS=0,1 \
  dcgm-job-map -prolog -print -nonzero
```

## Sample Output

In this example, a MIG node job is allocated one full A100 GPU slice and two 20GB MIG instances. Root runs `-init` to set up the directories; the job `Prolog` writes the job ID to the mapping files for the devices Slurm allocated, and the job `Epilog` resets them to `0`.

```bash
# nvidia-smi -L
GPU 0: NVIDIA A100-SXM4-80GB (UUID: GPU-b614723c-e261-4605-066e-79daab4c00ff)
  MIG 7g.80gb     Device  0: (UUID: MIG-dc34c2fd-4bd7-52ad-a6d0-645dcd1794e4)
GPU 1: NVIDIA A100-SXM4-80GB (UUID: GPU-e52dfcc4-406d-a3d4-f787-4402b535877a)
  MIG 2g.20gb     Device  0: (UUID: MIG-ec4c3916-65fa-5948-b9b5-9f21a2ddfcc0)
  MIG 1g.20gb     Device  1: (UUID: MIG-000d40d3-5924-5906-9879-8b69bf58689b)
  MIG 2g.20gb     Device  2: (UUID: MIG-676485be-683e-5129-83a5-f856e2ed922e)
  MIG 2g.20gb     Device  3: (UUID: MIG-0d944d00-611c-52de-92cf-c348adb292c2)
GPU 2: NVIDIA A100-SXM4-80GB (UUID: GPU-2812eb50-852a-8049-1257-d447ce043b27)
  MIG 2g.20gb     Device  0: (UUID: MIG-f87f8632-7326-508d-9231-525a243d0e26)
  MIG 2g.20gb     Device  1: (UUID: MIG-f859d832-3b8e-50c9-bb99-1c1c5311974b)
  MIG 2g.20gb     Device  2: (UUID: MIG-5f3d1c6b-4e04-57ab-9243-22736f9324cb)
  MIG 1g.20gb     Device  3: (UUID: MIG-03552fa9-803f-5ac8-baa4-a3e05685cd7a)
GPU 3: NVIDIA A100-SXM4-80GB (UUID: GPU-c78d556a-38ea-a307-20b5-e0548e6169d1)
  MIG 2g.20gb     Device  0: (UUID: MIG-ce91cb9e-5612-5c80-8c7a-84199c5ecc5d)
  MIG 2g.20gb     Device  1: (UUID: MIG-c7f9be2f-60d1-5851-a3fb-3e37900e04f2)
  MIG 2g.20gb     Device  2: (UUID: MIG-e7a8b788-1d6a-5b7d-8f7e-068004dfd0d6)
  MIG 1g.20gb     Device  3: (UUID: MIG-150c2a67-74ea-53c3-a121-319ed514b431)

# dcgm-job-map -init -print
would write "0" to /var/run/dcgm_job_maps/0.0
would write "0" to /var/run/dcgm_job_maps/1.3
would write "0" to /var/run/dcgm_job_maps/1.4
would write "0" to /var/run/dcgm_job_maps/1.5
would write "0" to /var/run/dcgm_job_maps/1.6
would write "0" to /var/run/dcgm_job_maps/2.3
would write "0" to /var/run/dcgm_job_maps/2.4
would write "0" to /var/run/dcgm_job_maps/2.5
would write "0" to /var/run/dcgm_job_maps/2.6
would write "0" to /var/run/dcgm_job_maps/3.3
would write "0" to /var/run/dcgm_job_maps/3.4
would write "0" to /var/run/dcgm_job_maps/3.5
would write "0" to /var/run/dcgm_job_maps/3.6


$ salloc -G a100:1,a100.20gb:2
salloc: Nodes gpu001 are ready for job

$ nvidia-smi -L
GPU 0: NVIDIA A100-SXM4-80GB (UUID: GPU-b614723c-e261-4605-066e-79daab4c00ff)
  MIG 7g.80gb     Device  0: (UUID: MIG-dc34c2fd-4bd7-52ad-a6d0-645dcd1794e4)
GPU 1: NVIDIA A100-SXM4-80GB (UUID: GPU-e52dfcc4-406d-a3d4-f787-4402b535877a)
  MIG 2g.20gb     Device  0: (UUID: MIG-ec4c3916-65fa-5948-b9b5-9f21a2ddfcc0)
  MIG 2g.20gb     Device  1: (UUID: MIG-676485be-683e-5129-83a5-f856e2ed922e)
GPU 2: NVIDIA A100-SXM4-80GB (UUID: GPU-2812eb50-852a-8049-1257-d447ce043b27)
GPU 3: NVIDIA A100-SXM4-80GB (UUID: GPU-c78d556a-38ea-a307-20b5-e0548e6169d1)

$ echo $CUDA_VISIBLE_DEVICES
MIG-dc34c2fd-4bd7-52ad-a6d0-645dcd1794e4,MIG-ec4c3916-65fa-5948-b9b5-9f21a2ddfcc0,MIG-676485be-683e-5129-83a5-f856e2ed922e

$ /usr/local/sbin/dcgm-job-map -prolog -print
would write "1474042" to /var/run/dcgm_job_maps/0.0
would write "1474042" to /var/run/dcgm_job_maps/1.3
would write "1474042" to /var/run/dcgm_job_maps/1.5

$ /usr/local/sbin/dcgm-job-map -epilog -print
would write "0" to /var/run/dcgm_job_maps/0.0
would write "0" to /var/run/dcgm_job_maps/1.3
would write "0" to /var/run/dcgm_job_maps/1.5
```

## Device selection

`-prolog` and `-epilog` run once per allocation, from the Slurm job `Prolog`/`Epilog`, via `slurmd` as root — **outside** the job's cgroup. NVML's device enumeration there is not restricted to this job's devices the way it would be inside the cgroup, so device identity instead comes directly from what Slurm tells the job it was allocated: `SLURM_JOB_GPUS`, or `CUDA_VISIBLE_DEVICES` when that is unset. `CUDA_VISIBLE_DEVICES` takes priority when it holds UUIDs, which name the exact device and need no numbering assumption at all.

Every mapping file is named from each GPU's NVML **minor number** (its `/dev/nvidia<minor>` node) — `<minor>` for a full GPU, `<minor>.<gpu-instance-id>` for a MIG instance — never from its position in NVML's device list, so `-init` and `-prolog`/`-epilog` always agree on a file name for the same physical device or instance regardless of how each was run. Each entry in the allocated-device list is resolved to that identity as follows:

- **A plain number** (e.g. `0`, `1`) is a **Slurm gres index**: a position in the node's device list, where a MIG-mode GPU contributes one entry *per GPU instance* rather than one entry for the whole GPU. This is *not* an NVML enumeration ordinal — on a node whose first GPU is partitioned into four instances, gres index `0` is that GPU's first instance and index `4` is the next device, not GPU 4. `slurmd -G` prints Slurm's own view of that list.
- **A range** (e.g. `4-7`) expands to each gres index it covers.
- **A `MIG-...` UUID** identifies one MIG GPU instance directly — `nvmlDeviceGetHandleByUUID` resolves straight to the instance handle, from which the GPU instance ID and the parent GPU's minor number are read.
- **A `GPU-...` UUID** identifies a full GPU directly — looked up with `nvmlDeviceGetHandleByUUID`, then resolved to its minor number. If it turns out to be in MIG mode, the specific instance(s) allocated aren't identifiable from that entry alone, so every instance on that GPU is mapped.

Slurm numbers gres devices across the **whole node**, so those numbers only line up with a whole-node view of NVML. A process confined to the job's cgroup — this program run from inside the job, or from a `Prolog`/`Epilog` under `PrologFlags=RunInJob` — sees NVML enumerate only the job's own devices, renumbered from zero, and node-wide indices would then select the wrong devices or run off the end of the table. One case is handled without the numbers: when the allocation names exactly as many devices as NVML can see, the visible set *is* the allocation, so every visible device is mapped directly. That covers a confined view of the full allocation as well as an unconfined job holding the whole node, and it cannot misfire on a partial allocation, since a confined view never shows more devices than were allocated. Outside that case, run `-prolog`/`-epilog` from the job `Prolog`/`Epilog` without `PrologFlags=RunInJob`.

This assumes the GPU index dcgm-exporter reports for a device matches that device's minor number. That holds whenever DCGM itself runs unrestricted (the normal case for a host-level exporter), since NVML minor numbers and enumeration position coincide outside a cgroup. Confirm it once per node with `nvidia-smi --query-gpu=index,name --format=csv` alongside `ls -la /dev/nvidia[0-9]*` if in doubt.

Verify device selection with `-print` before deploying; it performs the same resolution but writes nothing, and reports which variable the device list came from. `-print` needs `SLURM_JOB_GPUS` or `CUDA_VISIBLE_DEVICES` set in the environment to see anything, since those normally only exist inside an actual job's `Prolog`/`Epilog`:

```sh
SLURM_JOB_ID=12345 SLURM_JOB_GPUS=0,1 \
  dcgm-job-map -prolog -print -nonzero
```

`-prolog` requires a non-empty `SLURM_JOB_ID`. `-epilog` does not require a job ID. Neither requires `SLURM_JOB_GPUS`/`CUDA_VISIBLE_DEVICES` to be set — a job that didn't request a GPU simply has nothing to map.

## License

This project is licensed under the GNU General Public License, version 3. See [LICENSE](LICENSE).
