#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Collect the CV pipeline's seed decisions out of triton's compilation cache.

Every compilation that ran the dynamic CV pipeline records which operation
ordering it compiled, as `cv_*` keys in the compilation metadata. Triton writes
that metadata to `<cache>/<hash>/<kernel>.json` beside each kernel, so a run
over a suite of kernels leaves one record per kernel on disk already. This
turns those records into one table.

    # look at what a search run found
    python collect_cv_seeds.py

    # the kernels worth measuring, as a table to replay
    python collect_cv_seeds.py --only-found --format jsonl --out seeds.jsonl

    # replay exactly those orderings
    TRITON_ALWAYS_COMPILE=1 TRITON_ASCEND_CV_SEED_TABLE=seeds.jsonl pytest ...

The JSONL it writes is what `TRITON_ASCEND_CV_SEED_TABLE` reads, so collect and
replay are the same file with nothing in between.

Reads the cache and writes nowhere near it. Standard library only, and no
import of triton, so it runs wherever the cache was copied to.
"""

import argparse
import csv
import json
import os
import sys

# Written by third_party/ascend/backend/cv_seed_table.py. A record is a
# metadata JSON that has this key; everything else in the cache directory is
# some other kernel artifact.
MARKER_KEY = "cv_search_status_name"

# The fields a row carries, in the order a table shows them. Name, hash and
# seed are what a replay needs; the rest is why the seed was chosen, which is
# what a perf run needs when the measurement disagrees with the model.
FIELDS = [
    "kernel_name",
    "hash",
    "seed",
    "status",
    "cost",
    "second_bound",
    "ub_peak",
    "ub_capacity",
    "pipeline_stages",
    "pipeline_depth",
    "usable",
    "total",
    "rc",
]

# Which metadata key each field comes from. kernel_name is taken from
# cv_kernel_name rather than from kernel_name: the seed table is keyed on the
# name the pipeline itself saw, and reading it from anywhere else invites the
# two to drift apart without anyone noticing.
_FROM_METADATA = {
    "kernel_name": "cv_kernel_name",
    "hash": "hash",
    "seed": "cv_seed",
    "status": "cv_search_status_name",
    "cost": "cv_cost",
    "second_bound": "cv_second_bound",
    "ub_peak": "cv_ub_peak",
    "ub_capacity": "cv_ub_capacity",
    "pipeline_stages": "cv_pipeline_stages",
    "pipeline_depth": "cv_pipeline_depth",
    "usable": "cv_variants_usable",
    "total": "cv_variants_total",
    "rc": "cv_rc",
}


def default_cache_dir():
    """Where triton keeps its cache, resolved the way triton resolves it."""
    explicit = os.getenv("TRITON_CACHE_DIR")
    if explicit:
        return explicit
    home = os.getenv("TRITON_HOME") or os.path.expanduser("~")
    return os.path.join(home, ".triton", "cache")


def read_records(cache_dir):
    """Every seed decision in the cache, as rows, plus what could not be read.

    One file per kernel per compilation hash, so there is nothing to merge and
    no ordering to depend on. Unreadable files are counted rather than raised:
    a cache being written to while this runs is normal, and losing a whole
    table to one half-written file in it is not a trade worth making.
    """
    rows = []
    skipped = 0
    for entry in sorted(os.listdir(cache_dir)):
        group = os.path.join(cache_dir, entry)
        if not os.path.isdir(group):
            continue
        for name in sorted(os.listdir(group)):
            if not name.endswith(".json"):
                continue
            path = os.path.join(group, name)
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
            except (OSError, ValueError):
                skipped += 1
                continue
            if not isinstance(metadata, dict) or MARKER_KEY not in metadata:
                continue
            row = {field: metadata.get(key)
                   for field, key in _FROM_METADATA.items()}
            # The directory name is the compilation hash, and it is the one
            # thing here that cannot have been written wrong.
            row["hash"] = row["hash"] or entry
            rows.append(row)
    return rows, skipped


def deduplicate(rows):
    """One row per (kernel, hash), last seen winning.

    The same kernel under several hashes is several rows on purpose: those are
    different shapes or constexprs, and they get different orderings. The same
    pair twice should not happen -- one file holds it -- so this is only here
    so that collecting from two caches at once cannot produce a table with a
    duplicate key, which replay would resolve arbitrarily.
    """
    unique = {}
    for row in rows:
        unique[(row["kernel_name"], row["hash"])] = row
    return list(unique.values())


def sort_key(row, order):
    cost = row["cost"]
    # Unknown cost sorts last whichever way the comparison goes.
    cost_desc = -cost if isinstance(cost, int) else 1
    if order == "kernel":
        return (str(row["kernel_name"]), str(row["hash"]))
    if order == "cost":
        return (cost_desc, str(row["kernel_name"]))
    # "interesting": a chosen ordering first, then the expensive kernels,
    # because that is the order in which measuring them pays off.
    seed = row["seed"]
    chosen = 0 if isinstance(seed, int) and seed > 0 else 1
    return (chosen, cost_desc, str(row["kernel_name"]))


def write_jsonl(rows, stream):
    for row in rows:
        # Compact and one per line, so the file can be appended to and read
        # back line by line, which is what the replay side does.
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(rows, stream):
    writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore",
                            lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: "" if row[field] is None else row[field]
                         for field in FIELDS})


def write_table(rows, stream):
    """Aligned columns, for reading rather than for feeding back in."""
    columns = ["status", "seed", "kernel_name", "hash", "cost", "second_bound",
               "ub_peak", "pipeline_stages", "usable", "total"]

    def cell(row, column):
        value = row[column]
        if value is None:
            return "-"
        if column == "hash":
            return str(value)[:12]
        return str(value)

    widths = {column: len(column) for column in columns}
    for row in rows:
        for column in columns:
            widths[column] = max(widths[column], len(cell(row, column)))

    stream.write("  ".join(column.ljust(widths[column])
                           for column in columns).rstrip() + "\n")
    for row in rows:
        stream.write("  ".join(cell(row, column).ljust(widths[column])
                               for column in columns).rstrip() + "\n")


def summarise(rows, stream):
    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    parts = [f"{status}={count}" for status, count in sorted(counts.items())]
    stream.write(f"{len(rows)} kernel(s): " + ", ".join(parts) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", action="append", metavar="DIR",
                        help="cache directory to read; repeatable."
                             f" Default: {default_cache_dir()}")
    parser.add_argument("--format", choices=("table", "jsonl", "csv"),
                        default="table",
                        help="table to read, jsonl to replay with"
                             " TRITON_ASCEND_CV_SEED_TABLE, csv for a"
                             " spreadsheet")
    parser.add_argument("--out", metavar="FILE",
                        help="write here instead of stdout")
    parser.add_argument("--only-found", action="store_true",
                        help="keep only kernels where an ordering other than"
                             " the untouched pipeline was chosen")
    parser.add_argument("--status", action="append", metavar="NAME",
                        help="keep only these statuses; repeatable")
    parser.add_argument("--sort", choices=("interesting", "cost", "kernel"),
                        default="interesting",
                        help="row order; 'interesting' puts chosen orderings"
                             " first, then the expensive kernels")
    args = parser.parse_args(argv)

    caches = args.cache or [default_cache_dir()]
    rows = []
    skipped = 0
    for cache in caches:
        if not os.path.isdir(cache):
            print(f"collect_cv_seeds: {cache} is not a directory",
                  file=sys.stderr)
            return 2
        found, missed = read_records(cache)
        rows.extend(found)
        skipped += missed

    rows = deduplicate(rows)

    if args.only_found:
        rows = [row for row in rows
                if isinstance(row["seed"], int) and row["seed"] > 0]
    if args.status:
        wanted = set(args.status)
        rows = [row for row in rows if row["status"] in wanted]

    rows.sort(key=lambda row: sort_key(row, args.sort))

    if not rows:
        # Said out loud, because the overwhelmingly likely cause is a cache
        # that was served rather than written.
        print("collect_cv_seeds: no seed decisions found."
              " A kernel already in the cache is not recompiled, so a run"
              " that records anything needs TRITON_ALWAYS_COMPILE=1.",
              file=sys.stderr)

    writers = {"table": write_table, "jsonl": write_jsonl, "csv": write_csv}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            writers[args.format](rows, handle)
        summarise(rows, sys.stderr)
        print(f"collect_cv_seeds: wrote {args.out}", file=sys.stderr)
    else:
        writers[args.format](rows, sys.stdout)
        # Keep stdout a clean table when it is about to be piped somewhere.
        summarise(rows, sys.stdout if args.format == "table" else sys.stderr)

    if skipped:
        print(f"collect_cv_seeds: skipped {skipped} unreadable file(s)",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
