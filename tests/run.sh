#!/usr/bin/env bash
#
# dcgm-job-map test suite. Runs the real binary against tests/build/libnvidia-ml.so.1, a fake NVML
# whose node layout is set per test through FAKE_NVML_GPUS (see tests/fake_nvml.c), and a fake
# /proc/driver/nvidia/gpus built per test under a temp dir. No GPU, driver or Slurm is needed.
#
# Usage: make test        (or: bash tests/run.sh, after make and the fake library are built)
#
# The anchor case is sg235 from the 2026-10 wrongful-cancel report: NVML/PCI order 0,1,2,3 is
# /dev/nvidia0,3,1,2. A job handed Slurm index 3 runs on /dev/nvidia3 == NVML GPU 1, so the
# prolog must write file 1 -- not file 2, which is what naming by minor number produced.

set -u

cd "$(dirname "$0")/.."
BIN="$PWD/dcgm-job-map"
FAKE_LIBDIR="$PWD/tests/build"
LIBDIR="$FAKE_LIBDIR"

if [ ! -x "$BIN" ] || [ ! -f "$LIBDIR/libnvidia-ml.so.1" ]; then
    echo "build first: make dcgm-job-map tests/build/libnvidia-ml.so.1" >&2
    exit 2
fi

WORK=$(mktemp -d "${TMPDIR:-/tmp}/dcgm-job-map-test.XXXXXX")
trap 'rm -rf "$WORK"' EXIT

pass=0
fail=0
current=""

# ---- harness --------------------------------------------------------------------------------------

# Start a test case: a fresh mapping dir, spool dir and (absent) proc dir, and a clean environment.
begin() {
    current="$1"
    CASE="$WORK/$(echo "$current" | tr -c 'A-Za-z0-9' '_')"
    MAPS="$CASE/maps"
    SPOOL="$CASE/spool"
    PROC="$CASE/proc-gpus"  # not created unless the test calls proc_gpus
    rm -rf "$CASE"
    mkdir -p "$MAPS" "$SPOOL"
    unset FAKE_NVML_GPUS FAKE_NVML_INIT_FAIL SLURM_JOB_ID SLURM_JOB_GPUS CUDA_VISIBLE_DEVICES
    export DCGM_HPC_JOB_MAPPING_DIR="$MAPS"
    export DCGM_SLURMD_SPOOL_DIR="$SPOOL"
    export DCGM_NVIDIA_PROC_GPUS_DIR="$PROC"
    LIBDIR="$FAKE_LIBDIR"
    export LD_LIBRARY_PATH="$LIBDIR"
}

ok() {
    pass=$((pass + 1))
}

ko() {
    fail=$((fail + 1))
    echo "FAIL: $current: $*" >&2
    [ -s "$CASE/stderr" ] && sed 's/^/    stderr: /' "$CASE/stderr" >&2
    [ -s "$CASE/stdout" ] && sed 's/^/    stdout: /' "$CASE/stdout" >&2
}

# The node as NVML sees it. See tests/fake_nvml.c for the syntax.
nvml() {
    export FAKE_NVML_GPUS="$1"
}

# The node as the driver lists it in /proc/driver/nvidia/gpus: "<bus-hex>=<minor>" per GPU.
proc_gpus() {
    rm -rf "$PROC"
    mkdir -p "$PROC"
    for entry in "$@"; do
        local bus="${entry%%=*}" minor="${entry#*=}"
        local dir
        dir="$PROC/0000:$(printf '%02x' "0x$bus"):00.0"
        mkdir -p "$dir"
        printf 'Model:           Fake GPU\nBus Location:    0000:%02x:00.0\nDevice Minor:    %s\n' \
            "0x$bus" "$minor" > "$dir/information"
    done
}

# Run the binary with the given arguments, capturing exit status and output. It runs from the fake
# library's directory: Linux finds libnvidia-ml.so.1 through LD_LIBRARY_PATH, macOS dyld ignores that
# variable but does try a bare library name against the working directory. The binary itself never
# depends on its working directory, every path it uses being absolute.
run() {
    (cd "$LIBDIR" && exec "$BIN" "$@") > "$CASE/stdout" 2> "$CASE/stderr"
    RC=$?
}

# Make the fake library unfindable, as on a node with no NVIDIA driver.
no_driver() {
    mkdir -p "$CASE/nowhere"
    LIBDIR="$CASE/nowhere"
    export LD_LIBRARY_PATH="$LIBDIR"
}

running_job() {
    touch "$SPOOL/node01_$1.0"
}

seed_map() {
    printf '%s\n' "$2" > "$MAPS/$1"
}

# ---- assertions -----------------------------------------------------------------------------------

expect_rc() {
    [ "$RC" -eq "$1" ] && ok || ko "exit status $RC, expected $1"
}

# expect_map <file> <value>: the mapping file holds exactly this.
expect_map() {
    if [ ! -f "$MAPS/$1" ]; then
        ko "mapping file $1 missing (have: $(ls "$MAPS" | tr '\n' ' '))"
    elif [ "$(cat "$MAPS/$1")" != "$2" ]; then
        ko "mapping file $1 holds '$(cat "$MAPS/$1")', expected '$2'"
    else
        ok
    fi
}

expect_no_map() {
    [ ! -e "$MAPS/$1" ] && ok || ko "mapping file $1 exists, holding '$(cat "$MAPS/$1")'"
}

# expect_maps_exactly <name>...: the mapping dir holds these files and no others.
expect_maps_exactly() {
    local want="" have
    [ $# -gt 0 ] && want=$(printf '%s\n' "$@" | sort | tr '\n' ' ')
    have=$(ls "$MAPS" | sort | tr '\n' ' ')
    [ "$want" = "$have" ] && ok || ko "mapping dir holds '$have', expected '$want'"
}

# expect_job_on <job> <file>...: the job's ID is in exactly these files and in no other.
expect_job_on() {
    local job="$1"; shift
    local f
    for f in "$@"; do
        expect_map "$f" "$job"
    done
    for f in "$MAPS"/*; do
        [ -e "$f" ] || continue
        case " $* " in *" $(basename "$f") "*) continue ;; esac
        if [ "$(cat "$f")" = "$job" ]; then
            ko "job $job also written to $(basename "$f")"
        fi
    done
}

expect_stderr() {
    grep -q -- "$1" "$CASE/stderr" && ok || ko "stderr lacks '$1'"
}

expect_no_stderr() {
    if [ -s "$CASE/stderr" ]; then
        if [ $# -gt 0 ]; then
            grep -q -- "$1" "$CASE/stderr" && ko "stderr contains '$1'" || ok
        else
            ko "stderr not empty"
        fi
    else
        ok
    fi
}

expect_stdout() {
    grep -q -- "$1" "$CASE/stdout" && ok || ko "stdout lacks '$1'"
}

# ---- node layouts ---------------------------------------------------------------------------------

# A node where every numbering agrees: NVML i == PCI rank i == /dev/nvidia<i>.
IDENTITY="0,1,2,3"
# sg235: nvidia-smi index 0..3 is bus 01,41,81,C1; minors are 0,3,1,2 respectively.
SG235="0@01,3@41,1@81,2@c1"
sg235_proc() {
    proc_gpus 01=0 41=3 81=1 c1=2
}

# ---- -prolog: Slurm index -> NVML index -----------------------------------------------------------

begin "prolog on an identity node writes file s for Slurm index s"
nvml "$IDENTITY"; export SLURM_JOB_ID=100 SLURM_JOB_GPUS=2
run -prolog -nonzero
expect_rc 0
expect_job_on 100 2
expect_no_stderr

# The exact table from the report, one job at a time. SLURM_JOB_GPUS -> bus -> NVML index -> file.
begin "prolog on sg235: Slurm index 0 -> bus 01 -> NVML 0 -> file 0"
nvml "$SG235"; export SLURM_JOB_ID=1884965 SLURM_JOB_GPUS=0
run -prolog -nonzero
expect_rc 0
expect_job_on 1884965 0

begin "prolog on sg235: Slurm index 3 -> bus 41 -> NVML 1 -> file 1 (was file 2: job 65790816's GPU)"
nvml "$SG235"; export SLURM_JOB_ID=698246 SLURM_JOB_GPUS=3
run -prolog -nonzero
expect_rc 0
expect_job_on 698246 1
expect_no_map 2

begin "prolog on sg235: Slurm index 1 -> bus 81 -> NVML 2 -> file 2 (was file 3)"
nvml "$SG235"; export SLURM_JOB_ID=1136411 SLURM_JOB_GPUS=1
run -prolog -nonzero
expect_rc 0
expect_job_on 1136411 2

begin "prolog on sg235: Slurm index 2 -> bus C1 -> NVML 3 -> file 3 (was file 1)"
nvml "$SG235"; export SLURM_JOB_ID=1353531 SLURM_JOB_GPUS=2
run -prolog -nonzero
expect_rc 0
expect_job_on 1353531 3

begin "prolog on sg235 with the driver's GPU list present agrees with NVML alone"
nvml "$SG235"; sg235_proc; export SLURM_JOB_ID=698246 SLURM_JOB_GPUS=3
run -prolog -nonzero
expect_rc 0
expect_job_on 698246 1
expect_no_stderr

begin "prolog on sg235: all four jobs at once land on four distinct, correct files"
nvml "$SG235"
for pair in 1884965:0 698246:3 1136411:1 1353531:2; do
    export SLURM_JOB_ID="${pair%%:*}" SLURM_JOB_GPUS="${pair#*:}"
    run -prolog -nonzero
done
expect_map 0 1884965
expect_map 1 698246
expect_map 2 1136411
expect_map 3 1353531

begin "prolog on sg235: a list and a range translate entry by entry"
nvml "$SG235"; export SLURM_JOB_ID=200 SLURM_JOB_GPUS=1-2
run -prolog -nonzero
expect_rc 0
expect_job_on 200 2 3
begin "prolog on sg235: comma list"
nvml "$SG235"; export SLURM_JOB_ID=201 SLURM_JOB_GPUS=0,3
run -prolog -nonzero
expect_job_on 201 0 1

begin "prolog: a whole-node allocation maps every file"
nvml "$SG235"; export SLURM_JOB_ID=300 SLURM_JOB_GPUS=0-3
run -prolog -nonzero
expect_rc 0
expect_job_on 300 0 1 2 3

begin "prolog: non-contiguous minors (nvidia1 absent) rank by minor"
nvml "0,3@30,2@20"; export SLURM_JOB_ID=310 SLURM_JOB_GPUS=1
run -prolog -nonzero
expect_rc 0
expect_job_on 310 2   # Slurm's 2nd device is /dev/nvidia2, which is NVML index 2 here

begin "prolog: CUDA_VISIBLE_DEVICES is a fallback for SLURM_JOB_GPUS"
nvml "$SG235"; export SLURM_JOB_ID=320 CUDA_VISIBLE_DEVICES=3
run -prolog -nonzero
expect_job_on 320 1

begin "prolog: SLURM_JOB_GPUS wins over a numeric CUDA_VISIBLE_DEVICES"
nvml "$SG235"; export SLURM_JOB_ID=321 SLURM_JOB_GPUS=3 CUDA_VISIBLE_DEVICES=0
run -prolog -nonzero
expect_job_on 321 1

begin "prolog: a GPU UUID names the device directly, file still by NVML index"
nvml "$SG235"; export SLURM_JOB_ID=330 CUDA_VISIBLE_DEVICES=GPU-fake-1
run -prolog -nonzero
expect_rc 0
expect_job_on 330 1

begin "prolog: UUIDs in CUDA_VISIBLE_DEVICES beat SLURM_JOB_GPUS"
nvml "$SG235"; export SLURM_JOB_ID=331 SLURM_JOB_GPUS=0 CUDA_VISIBLE_DEVICES=GPU-fake-3
run -prolog -nonzero
expect_job_on 331 3

begin "prolog: an unknown UUID writes nothing"
nvml "$SG235"; export SLURM_JOB_ID=332 CUDA_VISIBLE_DEVICES=GPU-fake-9
run -prolog -nonzero
expect_maps_exactly
expect_stderr "does not know device GPU-fake-9"

begin "prolog: an index past the node's device list writes nothing"
nvml "$SG235"; export SLURM_JOB_ID=340 SLURM_JOB_GPUS=7
run -prolog -nonzero
expect_maps_exactly
expect_stderr "past the end"

begin "prolog: a job with no GPUs writes nothing and succeeds"
nvml "$SG235"; export SLURM_JOB_ID=350
run -prolog -nonzero
expect_rc 0
expect_maps_exactly

begin "prolog: no SLURM_JOB_ID is an error under -nonzero"
nvml "$SG235"; export SLURM_JOB_GPUS=0
run -prolog -nonzero
expect_rc 1
expect_maps_exactly

begin "prolog -print names the file it would write without writing it"
nvml "$SG235"; export SLURM_JOB_ID=99 SLURM_JOB_GPUS=3
run -prolog -print -nonzero
expect_rc 0
expect_stdout "would write \"99\" to $MAPS/1"
expect_maps_exactly

# ---- -prolog under PrologFlags=RunInJob (NVML confined to the job's cgroup) ---------------------

begin "confined prolog: the one visible GPU is named by its node-wide index"
# Job got Slurm index 3 == /dev/nvidia3 == bus 41 == host NVML 1. Inside its cgroup NVML shows
# that one GPU as index 0; only the driver's list can say it is GPU 1 to dcgm-exporter.
nvml "3@41"; sg235_proc; export SLURM_JOB_ID=400 SLURM_JOB_GPUS=3
run -prolog -nonzero
expect_rc 0
expect_job_on 400 1

begin "confined prolog: two visible GPUs"
nvml "1@81,2@c1"; sg235_proc; export SLURM_JOB_ID=410 SLURM_JOB_GPUS=1,2
run -prolog -nonzero
expect_rc 0
expect_job_on 410 2 3

begin "confined prolog: numbers that don't match the visible set are refused, nothing written"
nvml "3@41"; sg235_proc; export SLURM_JOB_ID=420 SLURM_JOB_GPUS=0-1
run -prolog -nonzero
expect_maps_exactly
expect_stderr "inside the job's cgroup"

begin "confined prolog: a UUID resolves through the driver list too"
nvml "3@41"; sg235_proc; export SLURM_JOB_ID=430 CUDA_VISIBLE_DEVICES=GPU-fake-0
run -prolog -nonzero
expect_job_on 430 1

# ---- consistency checks between NVML and the driver ----------------------------------------------

begin "unconfined, NVML not in PCI order: NVML index is used and the node is flagged"
nvml "0@81,1@01"; proc_gpus 81=0 01=1; export SLURM_JOB_ID=500 SLURM_JOB_GPUS=0
run -prolog -nonzero
expect_rc 0
expect_job_on 500 0
expect_stderr "not in PCI bus order"

begin "driver and NVML disagree on a minor number: refuse, write nothing"
nvml "0@01"; proc_gpus 01=5; export SLURM_JOB_ID=510 SLURM_JOB_GPUS=0
run -prolog -nonzero
expect_rc 1
expect_maps_exactly
expect_stderr "driver says /dev/nvidia5"

begin "a GPU NVML shows that the driver doesn't list: refuse, write nothing"
nvml "0@01,1@02"; proc_gpus 01=0; export SLURM_JOB_ID=520 SLURM_JOB_GPUS=0
run -prolog -nonzero
expect_rc 1
expect_maps_exactly
expect_stderr "not in the driver's GPU list"

# ---- MIG -----------------------------------------------------------------------------------------

# NVML 0 is /dev/nvidia1, partitioned into instances 3 and 5; NVML 1 is /dev/nvidia0, whole.
# Slurm's gres list, by device file: [nvidia0] [nvidia1 gi3] [nvidia1 gi5] -> indices 0, 1, 2.
MIGNODE="1:3+5,0"

begin "MIG: Slurm index 0 is the whole GPU on /dev/nvidia0, NVML 1"
nvml "$MIGNODE"; export SLURM_JOB_ID=600 SLURM_JOB_GPUS=0
run -prolog -nonzero
expect_rc 0
expect_job_on 600 1

begin "MIG: Slurm indices 1 and 2 are the instances of /dev/nvidia1, NVML 0"
nvml "$MIGNODE"; export SLURM_JOB_ID=610 SLURM_JOB_GPUS=1
run -prolog -nonzero
expect_job_on 610 0.3
begin "MIG: Slurm index 2"
nvml "$MIGNODE"; export SLURM_JOB_ID=611 SLURM_JOB_GPUS=2
run -prolog -nonzero
expect_job_on 611 0.5

begin "MIG: a MIG UUID names one instance"
nvml "$MIGNODE"; export SLURM_JOB_ID=620 CUDA_VISIBLE_DEVICES=MIG-fake-0-5
run -prolog -nonzero
expect_rc 0
expect_job_on 620 0.5

begin "MIG: a full-GPU UUID of a partitioned GPU maps every instance"
nvml "$MIGNODE"; export SLURM_JOB_ID=630 CUDA_VISIBLE_DEVICES=GPU-fake-0
run -prolog -nonzero
expect_job_on 630 0.3 0.5

begin "MIG: whole-node allocation covers instances and whole GPUs"
nvml "$MIGNODE"; export SLURM_JOB_ID=640 SLURM_JOB_GPUS=0-2
run -prolog -nonzero
expect_job_on 640 1 0.3 0.5

# ---- -init ---------------------------------------------------------------------------------------

begin "init creates one zeroed file per GPU, named by NVML index"
nvml "$SG235"
run -init -nonzero
expect_rc 0
expect_maps_exactly 0 1 2 3
expect_map 0 0; expect_map 1 0; expect_map 2 0; expect_map 3 0

begin "init on sg235 reports that minor order differs from NVML order"
nvml "$SG235"; sg235_proc
run -init -nonzero
expect_stderr "GPU 1 (NVML/dcgm-exporter index) is /dev/nvidia3 (Slurm index 3)"
expect_stderr "GPU 2 (NVML/dcgm-exporter index) is /dev/nvidia1 (Slurm index 1)"
expect_stderr "GPU 3 (NVML/dcgm-exporter index) is /dev/nvidia2 (Slurm index 2)"
expect_stderr "minor numbers do not follow NVML order"
expect_no_stderr "GPU 0 (NVML"

begin "init on an identity node is silent about ordering"
nvml "$IDENTITY"; proc_gpus 01=0 02=1 03=2 04=3
run -init -nonzero
expect_rc 0
expect_no_stderr

begin "init keeps a running job's ID and zeroes a finished job's"
nvml "$SG235"; seed_map 1 4242; seed_map 2 777; running_job 4242
run -init -nonzero
expect_rc 0
expect_map 1 4242
expect_map 2 0
expect_map 0 0
expect_map 3 0

begin "init with an unreadable spool dir keeps every non-zero ID and reports it"
nvml "$SG235"; seed_map 3 555; export DCGM_SLURMD_SPOOL_DIR="$CASE/no-such-spool"
run -init -nonzero
expect_rc 1
expect_map 3 555
expect_map 0 0

begin "init creates MIG instance files named <nvml-index>.<gi>"
nvml "$MIGNODE"
run -init -nonzero
expect_maps_exactly 0.3 0.5 1

begin "init refuses to run inside a job"
nvml "$SG235"; export SLURM_JOB_ID=700
run -init -nonzero
expect_rc 1
expect_maps_exactly

begin "init -print writes nothing"
nvml "$SG235"
run -init -print -nonzero
expect_stdout "would write \"0\" to $MAPS/1"
expect_maps_exactly

# ---- -reset --------------------------------------------------------------------------------------

begin "reset drops stale files and recreates the node's, running jobs or not"
nvml "$SG235"; seed_map 7 1; seed_map 0.3 1; seed_map 1 4242; running_job 4242
run -reset -nonzero
expect_rc 0
expect_maps_exactly 0 1 2 3
expect_map 1 0

begin "reset leaves non-regular entries in the directory alone"
nvml "$IDENTITY"; mkdir "$MAPS/subdir"
run -reset -nonzero
expect_rc 0
[ -d "$MAPS/subdir" ] && ok || ko "subdir removed"

# ---- -epilog -------------------------------------------------------------------------------------

begin "epilog zeroes exactly the files holding its job ID"
nvml "$SG235"; seed_map 0 66; seed_map 2 55; seed_map 3 55; seed_map 1 0
export SLURM_JOB_ID=55
run -epilog -nonzero
expect_rc 0
expect_map 2 0; expect_map 3 0
expect_map 0 66
expect_map 1 0

begin "epilog needs no NVML at all"
seed_map 0 55; export SLURM_JOB_ID=55; no_driver
run -epilog -nonzero
expect_rc 0
expect_map 0 0

begin "epilog without SLURM_JOB_ID is an error"
seed_map 0 55
run -epilog -nonzero
expect_rc 1
expect_map 0 55

# ---- no driver ------------------------------------------------------------------------------------

begin "no libnvidia-ml: prolog exits 0 by default, 1 under -nonzero, writes nothing"
export SLURM_JOB_ID=800 SLURM_JOB_GPUS=0; no_driver
run -prolog
expect_rc 0
run -prolog -nonzero
expect_rc 1
expect_maps_exactly

begin "nvmlInit failure is handled the same way"
nvml "$SG235"; export SLURM_JOB_ID=810 SLURM_JOB_GPUS=0 FAKE_NVML_INIT_FAIL=1
run -prolog
expect_rc 0
run -prolog -nonzero
expect_rc 1
expect_maps_exactly

# ---- errors never fail a Slurm prolog without -nonzero ----------------------------------------------

begin "without -nonzero every failure exits 0"
nvml "0@01"; proc_gpus 01=5; export SLURM_JOB_ID=900 SLURM_JOB_GPUS=0
run -prolog
expect_rc 0
run -bogus
expect_rc 0
run
expect_rc 0

# ---- summary --------------------------------------------------------------------------------------

echo "dcgm-job-map tests: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
