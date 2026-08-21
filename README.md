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
- Slurm for `-prolog` and `-epilog` integration, configured with `ConstrainDevices=yes`

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

This discovers every GPU on the node through NVML and creates one mapping file per device. A GPU in MIG mode gets one file per GPU instance, named `<gpu-index>.<gpu-instance-id>`; a GPU not in MIG mode gets a single file named `<gpu-index>`. Files are initialized with `0` and made user-writable (`0666`) so a task prolog running as the job user can update them. The mapping directory is created during this mode if it does not already exist.

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

### Slurm task prolog

Run from the Slurm task prolog:

```sh
dcgm-job-map -prolog
```

The utility asks NVML which devices it can reach from inside the job cgroup and writes `SLURM_JOB_ID` to the matching mapping files. It does not read `CUDA_VISIBLE_DEVICES` or `SLURM_JOB_GPUS`.

### Slurm task epilog

Run from the Slurm task epilog:

```sh
dcgm-job-map -epilog
```

This uses the same device selection and resets the selected mapping files to `0`.

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

For a safe dry run:

```sh
SLURM_JOB_ID=12345 \
  dcgm-job-map -prolog -print -nonzero
```

## Sample Output

In this example, a MIG node job is allocated one full A100 GPU slice and two 20GB MIG instances. Root runs `-init` to set up the directories, the prolog writes the job ID to the mapping files for the visible devices, and the epilog resets them to `0`.

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

The Slurm cgroup determines which devices `-prolog` and `-epilog` act on. No environment variable describes the allocation; every mapping file is named from each GPU's NVML **minor number** (its `/dev/nvidia<minor>` node), never from its position in NVML's device list.

The minor number matters because NVML's enumeration position is not stable under a cgroup: inside a restricted job, NVML only enumerates the device nodes the job can see and numbers them `0..N-1` within that restricted view. A job with one visible full GPU will always see it at enumeration position `0`, even if that GPU's minor number, and the id `-init` used to name its mapping file, is `2`. The minor number is unaffected by the cgroup, so it is the one identity `-init` (run unrestricted) and `-prolog`/`-epilog` (run inside the job cgroup) always agree on.

Two different signals are needed, because the cgroup constrains MIG instances and full GPUs differently:

- **MIG GPU instances.** Slurm restricts the `/dev/nvidia-caps` entries for instances outside the allocation, which also hides them from NVML. Every MIG instance NVML enumerates therefore belongs to this job, and is written as `<parent-minor>.<gpu-instance-id>`.
- **Full GPUs.** NVML enumerates every physical GPU on the node whether or not it is allocated, so enumeration alone proves nothing. Instead the utility tries to `open()` the GPU's `/dev/nvidia<minor>` node: the cgroup device controller enforces its allowlist at open time, so a successful open means the GPU is allocated to this job. Only GPUs not in MIG mode need this test.

This assumes the GPU index dcgm-exporter reports for a device matches that device's minor number. That holds whenever DCGM itself runs unrestricted (the normal case for a host-level exporter), since NVML minor numbers and enumeration position coincide outside a cgroup. Confirm it once per node with `nvidia-smi --query-gpu=index,name --format=csv` alongside `ls -la /dev/nvidia[0-9]*` if in doubt.

This requires two things of the Slurm configuration:

- `ConstrainDevices=yes` in `cgroup.conf`. Without it the cgroup imposes no restriction, every device looks allocated, and the utility will claim the whole node for every job.
- `-prolog` and `-epilog` must run **inside the job cgroup**, which means `TaskProlog` and `TaskEpilog` rather than the node-level `Prolog` and `Epilog`. A node-level epilog runs outside the cgroup, sees every device, and would reset mapping files belonging to other jobs still running on the node.

Verify both with `-print` before deploying; `-print` performs the same device selection but writes nothing.

`-prolog` requires a non-empty `SLURM_JOB_ID`. `-epilog` does not require a job ID.

## License

This project is licensed under the GNU General Public License, version 3. See [LICENSE](LICENSE).
