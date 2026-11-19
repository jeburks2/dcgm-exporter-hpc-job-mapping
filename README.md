# dcgm-job-map

A small C utility that records which Slurm job is using each GPU, for use with [DCGM Exporter](https://github.com/NVIDIA/dcgm-exporter)'s HPC job mapping. It maintains one file per GPU (or per MIG GPU instance) containing the job ID currently running on it, so exported GPU metrics can be attributed to jobs.

Four modes cover the lifecycle: `-init` creates the files at boot, `-prolog` stamps the job ID when a job starts, `-epilog` clears it when the job ends, and `-reset` wipes every mapping back to `0`.

## Requirements

**Build:** GCC or Clang. The CUDA toolkit's `nvml.h` is used when present; without it the build falls back to the ABI-compatible stub in `tests/stub`, so the program and its tests build on a machine with no NVIDIA software at all.

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

### Test

```sh
make test
```

No GPU, driver or Slurm is needed. The suite in `tests/run.sh` runs the real binary against `tests/fake_nvml.c`, a stand-in `libnvidia-ml.so.1` that presents whatever node the test describes — any number of GPUs, in any NVML order, with any minor numbers, PCI addresses and MIG layout — plus a fake `/proc/driver/nvidia/gpus`. Its anchor case is the node layout from the October 2026 wrongful-cancel report, where NVML order and minor-number order disagree; see [Device numbering](#device-numbering).

## Setup

### Create the mapping files

Run `-init` at boot:

```sh
sudo dcgm-job-map -init
```

This enumerates every GPU on the node through NVML and creates one file per device, owned by root, mode `0644`. A MIG-mode GPU gets one file per GPU instance named `<index>.<gpu-instance-id>`; any other GPU gets a single file named `<index>`, where `<index>` is the GPU's NVML index — the number dcgm-exporter labels it with. The mapping directory itself is created if absent, so its parent must exist and be writable.

On a node where `/dev/nvidia<N>` numbering does not follow NVML order, `-init` says so on stderr, one line per GPU. Nothing is wrong on such a node — see [Device numbering](#device-numbering) — but it is the kind of node where a mapping tool that assumes the two agree labels the wrong GPU, so it is worth knowing which ones you have.

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

## Device numbering

Two numberings meet in this tool, and on many nodes they disagree:

- **dcgm-exporter** stamps mapping file `N` onto the GPU it calls `N`: the **NVML index**, which follows PCI bus order (`nvidia-smi --query-gpu=index,pci.bus_id --format=csv`).
- **Slurm** numbers GPUs by **device file**. With `File=/dev/nvidia[0-3]` in `gres.conf`, gres index `s` is `/dev/nvidia<s>`, and `ConstrainDevices=yes` confines the job to exactly that node — so `SLURM_JOB_GPUS=3` means the job runs on the GPU whose **minor number** is 3. Minor numbers follow the driver's probe order, not PCI order.

The two coincide on some nodes and not others. On one real node, NVML indices 0–3 are buses `01`, `41`, `81`, `C1`, but their minor numbers are 0, 3, 1, 2:

```console
# nvidia-smi -q | grep -E 'Minor Number|Bus Id' | paste - -
        Bus Id : 00000000:01:00.0        Minor Number : 0
        Bus Id : 00000000:41:00.0        Minor Number : 3
        Bus Id : 00000000:81:00.0        Minor Number : 1
        Bus Id : 00000000:C1:00.0        Minor Number : 2
```

A job handed `SLURM_JOB_GPUS=3` there runs on bus `41`, which dcgm-exporter calls GPU **1**. So:

- Mapping files are named by **NVML index**: `<index>` for a full GPU, `<index>.<gpu-instance-id>` for a MIG instance.
- A Slurm device number is resolved **through the minor number**: index `s` is the `s`-th device in the node's list ordered by `/dev/nvidia<N>`, which is `/dev/nvidia<s>` itself whenever minors run `0..N-1`. It is never used as an NVML position.

Earlier versions did both the other way round — named files by minor number and read Slurm's index as an NVML position — which is correct only where the two orders coincide and otherwise permutes the `hpc_job` label across the jobs on the node. Anything acting on that label, such as an idle-GPU rule, then acts on the wrong job. `-init` reports every node where the orders differ, and `make test` reproduces the layout above.

## Device selection

The allocated devices come from `SLURM_JOB_GPUS`, falling back to `CUDA_VISIBLE_DEVICES`. `CUDA_VISIBLE_DEVICES` takes priority when it holds UUIDs, since those name an exact device. Entries are resolved as follows:

- **A plain number** is a Slurm gres index: a position in the node's device list ordered by device file, where a MIG-mode GPU contributes one entry per GPU instance rather than one for the whole GPU. This is not an NVML enumeration ordinal, for the reason above. On a node whose `/dev/nvidia0` is partitioned into four instances, gres index `0` is that GPU's first instance and index `4` is the next device, not GPU 4. Run `slurmd -G` to see Slurm's own view of the list.
- **A range** such as `4-7` expands to each gres index it covers.
- **A `MIG-...` UUID** resolves straight to the instance handle, giving the GPU instance ID and the parent GPU.
- **A `GPU-...` UUID** resolves to a full GPU. If that GPU is in MIG mode, the entry doesn't say which instances the job holds, so all of them are mapped.

Slurm's numbering is node-global, and it accounts for devices already in use. Two concurrent jobs on a partitioned node land on disjoint files: a single-GPU job takes gres index `0`, and a five-GPU job starting while it runs is handed `1,2,3,4,5`, not `0,1,2,3,4`.

### `PrologFlags=RunInJob`

With `RunInJob` the prolog runs inside the job's cgroup, where NVML enumerates only the job's own devices — and renumbers them from 0, so NVML's index inside the job is not the one dcgm-exporter uses. The prolog detects this view by comparing NVML's device count against the driver's own list in `/proc/driver/nvidia/gpus`, which procfs serves regardless of the cgroup, and takes each visible GPU's node-wide index from its position in that list (sorted by PCI address, NVML's order). Because the device numbers in `SLURM_JOB_GPUS` are node-wide and a confined view cannot resolve them, the visible set is taken to be the allocation when the two have the same count — which under confinement they always do — and a numeric list that doesn't match is refused rather than guessed at.

Without `RunInJob` the prolog sees the whole node and NVML's index is used directly. The driver list, when readable, is still cross-checked: a GPU whose minor number NVML and the driver disagree on, or that the driver does not list, stops the prolog with nothing written, and a node whose NVML order is not PCI order is flagged on stderr since the `RunInJob` fallback would be unreliable there. Set `DCGM_NVIDIA_PROC_GPUS_DIR` to read the list from elsewhere; if it cannot be read at all, the prolog assumes it is unconfined.

The epilog is unaffected either way. It never runs inside the job's cgroup, even with `RunInJob` set, so it always sees the whole node — which is exactly why it matches on job ID rather than resolving devices a second time. Resolving them would clear a different set of files than the prolog wrote and strand the job ID on the rest.

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
DCGM_NVIDIA_PROC_GPUS_DIR Driver GPU list to read instead of
                          /proc/driver/nvidia/gpus
DCGM_NVML_LIBRARY         NVML library to load instead of libnvidia-ml.so.1
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

From inside the resulting allocation, where NVML shows only the three allocated instances (on this node minor numbers follow NVML order, so the file names match what `nvidia-smi -L` showed above):

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
