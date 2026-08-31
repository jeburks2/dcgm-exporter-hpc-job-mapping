# dcgm-job-map

A small C utility that records which Slurm job is using each GPU, for use with [DCGM Exporter](https://github.com/NVIDIA/dcgm-exporter)'s HPC job mapping. It maintains one file per GPU (or per MIG GPU instance) containing the job ID currently running on it, so exported GPU metrics can be attributed to jobs.

Four modes cover the lifecycle: `-init` creates the files at boot, `-prolog` stamps the job ID when a job starts, `-epilog` clears it when the job ends, and `-reset` wipes every mapping back to `0`.

## Requirements

**Build:** Linux, GCC or another compiler the Makefile accepts, and the CUDA toolkit headers for `nvml.h`.

**Run:** Slurm with the `gres/gpu` plugin configured, and the NVIDIA driver on any node that should report mappings.

`libnvidia-ml.so.1` is loaded with `dlopen` at run time rather than linked at build time, so a single binary can be deployed fleet-wide. On a node with no driver, such as a login or CPU-only compute node, every mode finds no NVML, skips the mapping, and exits `0`.

## Build

```sh
make
```

The build looks for CUDA in `/usr/local/cuda`, and uses it only for `nvml.h` — the binary does not link against `libnvidia-ml`. Point `CUDA_HOME` elsewhere if needed:

```sh
CUDA_HOME=/opt/cuda-13.2 make
```

Install to `/usr/local/sbin`, and clean up:

```sh
sudo make install
make clean
```

## Setup

### Create the mapping files

Run `-init` at boot:

```sh
sudo dcgm-job-map -init
```

This enumerates every GPU on the node through NVML and creates one file per device, owned by root, mode `0644`. A MIG-mode GPU gets one file per GPU instance named `<minor>.<gpu-instance-id>`; any other GPU gets a single file named `<minor>`. The mapping directory itself is created if absent, so its parent must exist and be writable.

Every file is reset to `0`, except one that already holds the ID of a job still running on the node — see [Keeping live mappings](#keeping-live-mappings). Use [`-reset`](#clearing-every-mapping) to clear those too.

Set `DCGM_HPC_JOB_MAPPING_DIR` to choose the directory; it defaults to `/var/run/dcgm_job_maps`. The same value has to reach the prolog and epilog, or they will update files the exporter never reads.

`-init` refuses to run with `SLURM_JOB_ID` set. It needs a node-wide view, and inside a job's cgroup NVML reports only that job's devices — it would then create files for one job's slice and reset every other device on the node.

To initialize automatically alongside the exporter, drop in a unit override:

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

Check it with `systemctl cat nvidia-dcgm-exporter.service` and `systemctl status nvidia-dcgm-exporter.service`.

### Keeping live mappings

That `ExecStartPre` fires on every exporter restart, not just at boot, and restarts happen while jobs are running. A blanket reset to `0` would strand those jobs: nothing rewrites a mapping between the prolog and the epilog, so the GPUs stay unattributed for the rest of the job.

So `-init` keeps a non-zero mapping whose job is still running on the node. It finds those jobs the same way `scontrol listjobs` does — by listing `SlurmdSpoolDir`, where `slurmd` keeps one unix socket per running step, named `<node>_<jobid>.<stepid>`, plus a `job<jobid>` directory for a batch job. That is one `getdents` call against a local directory: no `slurmctld` RPC, no lock, no measurable cost in an `ExecStartPre`.

```console
# ls /var/spool/slurmd
sdg051_1474107.4294967290  sdg051_1474107.4294967292  cred_state  conf-cache

# dcgm-job-map -init -print
would keep "1474107" in /var/run/dcgm_job_maps/0
would keep "1474107" in /var/run/dcgm_job_maps/1
would write "0" to /var/run/dcgm_job_maps/2
would write "0" to /var/run/dcgm_job_maps/3
```

The spool directory comes from `SlurmdSpoolDir` in `slurm.conf`, read from `$SLURM_CONF`, `/etc/slurm/slurm.conf`, or the cached copy at `/run/slurm/conf/slurm.conf` that a configless cluster gets — the same places `scontrol` looks. `%n` and `%h` in the value expand to the short hostname. Set `DCGM_SLURMD_SPOOL_DIR` to skip the lookup entirely.

Two limits are worth knowing:

- The check is on job **identity**, not placement. A mapping is kept if its job is still on the node — not because that job still holds that particular GPU. In practice the two agree, since the only thing that writes a job ID to a file is that job's own prolog, and the epilog clears it.
- Jobs are visible through their **steps**. An allocation with no step running at that instant — an idle `salloc` on a cluster without `PrologFlags=Contain`, which is what creates the persistent extern step — does not appear in the spool directory and its mapping is reset. `scontrol listjobs` has the same blind spot.

If the spool directory cannot be read at all, `-init` keeps every non-zero mapping and reports the failure on stderr. A stale ID lasts only until the next job lands on that device and its prolog overwrites it; an erased one is unrecoverable for the job it belonged to.

### Clearing every mapping

`-reset` deletes every file in the mapping directory, then recreates one per current device holding `0`:

```sh
sudo dcgm-job-map -reset
```

Unlike `-init` it never keeps a running job's ID, and the delete pass means devices that no longer exist leave no file behind. Run it after a MIG reconfiguration, or to clear mappings left stale by a prolog or epilog that did not complete. Like `-init`, it refuses to run with `SLURM_JOB_ID` set.

### Wire up the prolog and epilog

In `slurm.conf`:

```ini
Prolog=/usr/local/sbin/dcgm-job-map -prolog
Epilog=/usr/local/sbin/dcgm-job-map -epilog
```

`-prolog` reads the devices Slurm allocated to the job and writes `SLURM_JOB_ID` to the matching files. `-epilog` doesn't look at devices at all: it resets every mapping file that holds its own `SLURM_JOB_ID`, so it clears exactly what the prolog wrote.

Both work with or without `PrologFlags=RunInJob`. The prolog adapts to whichever view of the node it gets, and the epilog is unaffected by the setting either way. See [Device selection](#device-selection) for how each case resolves.

`Prolog` runs when the job's first step launches. Add `PrologFlags=Alloc` to have the mapping written as soon as the allocation is granted, which matters for an interactive `salloc` session that may never launch a step.

Use the **job** prolog and epilog, not `TaskProlog`/`TaskEpilog`. The task versions run once per task inside `slurmstepd`, for every step in the job. A job that runs several `srun` steps would have its mapping cleared as soon as the first step exits, while the allocation is still live:

```text
$ salloc -G 2 -n 2
$ cat /var/run/dcgm_job_maps/*
1474054
1474054
$ srun -G 1 ./train.py
... (runs fine) ...
$ cat /var/run/dcgm_job_maps/*
0        # wrong: TaskEpilog cleared these, but the allocation is still running
0
```

`Prolog` and `Epilog` run once per allocation, through `slurmd` as root, so they fire once at job start and once at job end.

## Device selection

Mapping files are named after each GPU's **minor number**, its `/dev/nvidia<minor>` node: `<minor>` for a full GPU and `<minor>.<gpu-instance-id>` for a MIG instance. Minor numbers are stable regardless of NVML enumeration order, so every mode agrees on the file name for a given device no matter how it was invoked.

The allocated devices come from `SLURM_JOB_GPUS`, falling back to `CUDA_VISIBLE_DEVICES`. `CUDA_VISIBLE_DEVICES` takes priority when it holds UUIDs, since those name an exact device. Entries are resolved as follows:

- **A plain number** is a Slurm gres index: a position in the node's device list, where a MIG-mode GPU contributes one entry per GPU instance rather than one for the whole GPU. This is not an NVML enumeration ordinal. On a node whose first GPU is partitioned into four instances, gres index `0` is that GPU's first instance and index `4` is the next device, not GPU 4. Run `slurmd -G` to see Slurm's own view of the list.
- **A range** such as `4-7` expands to each gres index it covers.
- **A `MIG-...` UUID** resolves straight to the instance handle, giving both the GPU instance ID and the parent GPU's minor number.
- **A `GPU-...` UUID** resolves to a full GPU. If that GPU is in MIG mode, the entry doesn't say which instances the job holds, so all of them are mapped.

### `PrologFlags=RunInJob`

The prolog handles both cases. With `RunInJob` it runs inside the job's cgroup, where NVML enumerates only the job's own devices; when that view holds exactly as many devices as the allocation names, the visible set *is* the allocation and the device numbers are never consulted. Without `RunInJob` it sees the whole node and resolves the numbers positionally. Neither can misfire on the count check itself: a confined view never shows more devices than were allocated, and unconfined, an equal count means the job holds the entire node.

The epilog is unaffected either way. It never runs inside the job's cgroup, even with `RunInJob` set, so it always sees the whole node — which is exactly why it matches on job ID rather than resolving devices a second time. Resolving them would clear a different set of files than the prolog wrote and strand the job ID on the rest.

Slurm's numbering is node-global, and it accounts for devices already in use. Two concurrent jobs on a partitioned node land on disjoint files: a single-GPU job takes gres index `0`, and a five-GPU job starting while it runs is handed `1,2,3,4,5`, not `0,1,2,3,4`.

This does assume the GPU index dcgm-exporter reports matches the device's minor number, which holds whenever DCGM runs unrestricted, as a host-level exporter normally does. To confirm on a given node, compare `nvidia-smi --query-gpu=index,name --format=csv` against `ls -la /dev/nvidia[0-9]*`.

## Options

```text
-init      Create mapping files for every full GPU and MIG GPU instance on the
           node, resetting each to 0 unless it holds a still-running job's ID;
           refuses to run inside a Slurm job
-reset     Delete every mapping file, then recreate them all holding 0;
           refuses to run inside a Slurm job
-prolog    Write SLURM_JOB_ID to the mapping files for this job's allocated devices
-epilog    Reset every mapping file holding SLURM_JOB_ID back to 0
-print     Print intended writes without changing files
-nonzero   Return non-zero for invalid arguments, NVML errors, or mapping errors
-help      Show usage information
```

Without `-nonzero` the process always exits `0`, so a failure here never fails the job.

Environment variables:

```text
DCGM_HPC_JOB_MAPPING_DIR  Mapping file directory; default /var/run/dcgm_job_maps
DCGM_SLURMD_SPOOL_DIR     slurmd spool dir, overriding SlurmdSpoolDir from
                          slurm.conf; default /var/spool/slurmd  (-init)
SLURM_CONF                slurm.conf to read SlurmdSpoolDir from  (-init)
SLURM_JOB_ID              Job ID written by -prolog, matched by -epilog
SLURM_JOB_GPUS            Devices allocated to the job  (-prolog)
CUDA_VISIBLE_DEVICES      Preferred over SLURM_JOB_GPUS when it holds UUIDs,
                          and a fallback when SLURM_JOB_GPUS is unset  (-prolog)
```

Both `-prolog` and `-epilog` need a non-empty `SLURM_JOB_ID`. Only `-prolog` reads a device list, and a job that requested no GPU simply has nothing to map. `-epilog` needs no NVML at all, so it still clears its files on a node whose driver is unhealthy.

Use `-print` to check device selection before deploying. It performs the same resolution and writes nothing, but needs the environment a real prolog would have:

```sh
SLURM_JOB_ID=12345 SLURM_JOB_GPUS=0,1 \
  dcgm-job-map -prolog -print -nonzero
```

## Example

A node with four A100s, all in MIG mode. Root initializes the files, then a job takes one 7g.80gb instance and two 2g.20gb instances:

```console
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
```

From inside the resulting allocation, where NVML shows only the three allocated instances:

```console
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

$ dcgm-job-map -prolog -print
would write "1474042" to /var/run/dcgm_job_maps/0.0
would write "1474042" to /var/run/dcgm_job_maps/1.3
would write "1474042" to /var/run/dcgm_job_maps/1.5
```

Once the job ends, the epilog resets those same three files, having found `1474042` in them rather than by resolving devices a second time.

## License

GNU General Public License, version 3. See [LICENSE](LICENSE).
