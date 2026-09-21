#!/bin/sh
# Effective CPU + RAM available to this process, cgroup-aware (v1 and v2).

cg_cpus() {
    quota=max; period=100000
    if [ -r /sys/fs/cgroup/cpu.max ]; then                      # cgroup v2
        read -r quota period < /sys/fs/cgroup/cpu.max
    elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then       # cgroup v1
        quota=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
        period=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
        [ "$quota" = "-1" ] && quota=max
    fi
    affinity=$(nproc)                                           # honours cpuset/taskset
    [ "$quota" = max ] && { echo "$affinity"; return; }
    n=$(( (quota + period - 1) / period ))                      # ceil
    [ "$n" -lt 1 ] && n=1
    [ "$n" -lt "$affinity" ] && echo "$n" || echo "$affinity"
}

cg_mem_bytes() {
    lim=
    [ -r /sys/fs/cgroup/memory.max ] && lim=$(cat /sys/fs/cgroup/memory.max)
    [ -z "$lim" ] && [ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ] &&
        lim=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)
    host=$(awk '/^MemTotal:/ {print $2 * 1024}' /proc/meminfo)
    [ -z "$lim" ] || [ "$lim" = max ] && { echo "$host"; return; }
    [ "$lim" -gt "$host" ] && echo "$host" || echo "$lim"       # catches v1's unlimited sentinel
}
