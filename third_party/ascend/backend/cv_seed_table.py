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
"""Reading back, and replaying, the dynamic CV pipeline's choice of ordering.

The variant search inside AddDynamicCVPipeline picks one operation ordering out
of many and compiles it. Which one it picked used to exist only as a line of
stderr, which is unusable for the thing the search is for: running a suite of
kernels, finding the ones where the cost model prefers something other than the
untouched pipeline, and then measuring those on hardware.

Two halves, and they are deliberately asymmetric.

Writing needs no file and no option. The pass publishes its decision as module
attributes; this module copies them into the compilation metadata and triton
itself persists that, as JSON, next to every kernel it compiles. So the record
is written by the ordinary cache machinery -- one file per kernel, keyed by the
compilation hash, through a temp file and a rename, with no two processes ever
writing the same path. Collecting those into a table is then somebody else's
job, and a better one for it: `triton.knobs.compilation.listener` gets the same
metadata handed to it, and the cache directory holds it either way.

Reading does need a file, because its input is a table somebody wrote by hand
or built from those records, and the whole point of it is that it travels --
between runs, between machines, into a commit. `TRITON_ASCEND_CV_SEED_TABLE`
names it; naming it is what turns replay on, so there is no second variable
saying which mode this is.

The table is re-read on every compilation rather than cached for the process.
It costs a stat and a few hundred lines of parsing against a full kernel
compilation, which is nothing, and it means a table edited or extended during a
run takes effect during that run. The mtime is logged on every lookup, because
once the table can change mid-run the only way to explain a result afterwards
is to know which version of the table produced it.
"""

import json
import os
import sys

__all__ = [
    "SEED_TABLE_ENV_VAR",
    "NO_SEED",
    "table_path",
    "lookup_seed",
    "record_decision",
]

SEED_TABLE_ENV_VAR = "TRITON_ASCEND_CV_SEED_TABLE"

# Handed to the pass to mean "decide for yourself", and read back from it to
# mean "I decided nothing". The same number on purpose: both say that no
# ordering was named, and the pass option and the attribute are two ends of one
# conversation.
NO_SEED = -1

# Mirrors CVPipeline::SearchStatus in
# third_party/ascend/include/DynamicCVPipeline/Common/Utils.h. Kept as plain
# numbers on the C++ side because an integer attribute is the only kind the
# Python binding can read; the names live here, where the table is written.
SEARCH_NOT_REQUESTED = 0
SEARCH_SEED_FOUND = 1
SEARCH_BASELINE_BEST = 2
SEARCH_NONE_USABLE = 3
SEARCH_SEED_SUPPLIED = 4
SEARCH_UNSUPPORTED_TARGET = 5

_STATUS_NAMES = {
    SEARCH_NOT_REQUESTED: "no_search",
    SEARCH_SEED_FOUND: "seed_found",
    SEARCH_BASELINE_BEST: "baseline_best",
    SEARCH_NONE_USABLE: "none_usable",
    SEARCH_SEED_SUPPLIED: "from_table",
    SEARCH_UNSUPPORTED_TARGET: "wrong_chip",
}

# Set by AddDynamicCVPipelinePass on its way out, on every exit from it.
_SEED_ATTR = "triton_ascend.dynamic_cv_pipeline.chosen_seed"
_STATUS_ATTR = "triton_ascend.dynamic_cv_pipeline.search_status"
_USABLE_ATTR = "triton_ascend.dynamic_cv_pipeline.variants_usable"
_TOTAL_ATTR = "triton_ascend.dynamic_cv_pipeline.variants_total"

# Already on the module, set by EstimateCVPipelineCostPass and documented as
# part of its output, so these are read and left in place. Removing them would
# take away something users are told they can read off the IR.
_COST_ATTRS = {
    "cv_cost": "ascend.cv_pipeline_estimated_cycles",
    "cv_cost_resource": "ascend.cv_pipeline_cost_resource",
    "cv_cost_recurrence": "ascend.cv_pipeline_cost_recurrence",
    "cv_ub_peak": "ascend.cv_pipeline_cost_ub_peak",
    "cv_ub_capacity": "ascend.cv_pipeline_cost_ub_capacity",
    "cv_pipeline_stages": "ascend.cv_pipeline_cost_pipeline_stages",
    "cv_pipeline_depth": "ascend.cv_pipeline_cost_pipeline_depth",
}

_RC_ATTR = "triton_ascend.dynamic_cv_pipeline.rc"

# The last table that parsed cleanly, so that catching the file mid-write does
# not silently turn a replay run into a search run. Re-reading on every
# compilation is what makes this necessary: it widens the window in which a
# half-written file can be read, so the fallback has to be the previous good
# content rather than nothing.
_last_good = None  # (path, rows) or None


def _log(message):
    # stderr, because that is where the pass and the cost model write, and a
    # line about which seed was used is unreadable if it lands in a different
    # stream from the report it explains.
    print(f"[cv-seed-table] {message}", file=sys.stderr)


def table_path():
    """The table to replay from, or None for ordinary behaviour."""
    path = os.getenv(SEED_TABLE_ENV_VAR)
    return path or None


def _parse_seed(raw):
    """A seed field as an int, or None if the row records that there is none.

    The write side records kernels the search could not help -- nothing beat
    the baseline, nothing compiled, the kernel was never applicable -- and
    those rows have to survive a round trip through a hand-built table, or the
    table cannot answer "have I already looked at this kernel?". Anything that
    is not a number is read as such a row rather than as an error.
    """
    text = str(raw).strip()
    if not text:
        return None
    try:
        return int(text, 10)
    except ValueError:
        return None


def _split_delimited(line):
    # Both separators, because a table assembled from the per-kernel JSON by
    # one script and a table typed by hand rarely agree on which one it is.
    for separator in (";", ","):
        if separator in line:
            return [field.strip().strip('"') for field in line.split(separator)]
    return [line.strip()]


def _parse(text):
    """{(kernel_name, hash): seed-or-None} from JSONL or delimited text.

    Both formats, deliberately. What this reads is whatever the harness built
    out of the per-kernel records, and refusing one of the two obvious
    spellings would only mean a conversion step that can itself be wrong.
    """
    rows = {}
    header = None
    for number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("{"):
            # A truncated last line is the expected failure here, not a reason
            # to throw away every row that did parse.
            try:
                entry = json.loads(line)
            except ValueError:
                _log(f"line {number} is not valid JSON; skipped")
                continue
            if not isinstance(entry, dict):
                continue
            name = entry.get("kernel_name") or entry.get("kernel") or entry.get("name")
            digest = entry.get("hash")
            if name is None or digest is None:
                _log(f"line {number} has no kernel_name/hash pair; skipped")
                continue
            rows[(str(name), str(digest))] = _parse_seed(entry.get("seed", ""))
            continue

        fields = _split_delimited(line)
        lowered = [field.lower() for field in fields]
        if header is None and "seed" in lowered and any(
                field in lowered for field in ("kernel_name", "kernel", "name")):
            header = lowered
            continue

        if header is not None:
            entry = dict(zip(header, fields))
            name = entry.get("kernel_name") or entry.get("kernel") or entry.get("name")
            digest = entry.get("hash")
            seed = entry.get("seed", "")
        elif len(fields) >= 3:
            # No header, so the order is the one the records are written in.
            name, digest, seed = fields[0], fields[1], fields[2]
        else:
            _log(f"line {number} has fewer than three fields and no header; skipped")
            continue

        if not name or not digest:
            _log(f"line {number} has no kernel_name/hash pair; skipped")
            continue
        rows[(str(name), str(digest))] = _parse_seed(seed)
    return rows


def _load(path):
    """Parse the table, keeping the previous content if this read is unusable."""
    global _last_good
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
        mtime = os.path.getmtime(path)
    except OSError as error:
        if _last_good is not None and _last_good[0] == path:
            _log(f"could not re-read {path} ({error}); keeping the {len(_last_good[1])}"
                 " row(s) read earlier")
            return _last_good[1], None
        _log(f"could not read {path} ({error}); no seed will be replayed")
        return {}, None

    rows = _parse(text)
    if not rows and _last_good is not None and _last_good[0] == path:
        # An empty parse where there used to be rows is a file being rewritten,
        # far more often than a table someone truncated on purpose.
        _log(f"{path} parsed to no rows; keeping the {len(_last_good[1])}"
             " row(s) read earlier")
        return _last_good[1], mtime

    _last_good = (path, rows)
    return rows, mtime


def lookup_seed(kernel_name, kernel_hash):
    """The ordering to compile for this kernel, or NO_SEED to decide as usual.

    Never raises and never silently does nothing: a caller that set the table
    variable is expecting a hit, so both a hit and a miss are logged. A miss
    deliberately behaves exactly like no table at all -- the search runs if it
    was asked for -- because the alternative is refusing to compile a kernel
    that was only missing from a list.
    """
    path = table_path()
    if path is None:
        return NO_SEED

    rows, mtime = _load(path)
    stamp = f", mtime {mtime}" if mtime is not None else ""
    key = (str(kernel_name), str(kernel_hash))
    if key not in rows:
        _log(f"{kernel_name} (hash {kernel_hash}) not found in {path}"
             f" ({len(rows)} row(s){stamp}); compiling as if no table had been given")
        return NO_SEED

    seed = rows[key]
    if seed is None or seed < 0:
        _log(f"{kernel_name} (hash {kernel_hash}) is in {path}{stamp} with no usable"
             " seed; compiling as if no table had been given")
        return NO_SEED

    _log(f"{kernel_name} (hash {kernel_hash}) found in {path}{stamp}:"
         f" compiling seed {seed}, no search")
    return seed


def _get_int_attr(mod, name, ascend_module):
    getter = getattr(getattr(ascend_module, "ir", None), "get_int_attr", None)
    if getter is None:
        return None
    try:
        value = getter(mod, name)
    except TypeError:
        return None
    return value if isinstance(value, int) else None


def _remove_attr(mod, name, ascend_module):
    remover = getattr(getattr(ascend_module, "ir", None), "remove_attr", None)
    if remover is not None:
        remover(mod, name)


def _status_name(status, rc):
    """One label per outcome, from the two halves that each know part of it.

    The pass reports what it decided about the ordering; the return code
    reports whether the pipeline then survived. Neither is the whole answer: a
    seed was still chosen on a kernel that fell back, and a kernel the
    pipeline never applied to has no ordering to speak of. Combining them is
    this side's job, because this side is where the vocabulary of the table is
    defined.
    """
    if rc == 2:  # CVPipeline::ERRCODE_IGNORED
        return "ignored"
    if rc:
        return "fallback"
    if status is None:
        return "unknown"
    return _STATUS_NAMES.get(status, f"status_{status}")


def record_decision(mod, metadata, kernel_name, requested_seed):
    """Copy the pipeline's decision into `metadata` and say so in the log.

    Called straight after the pass pipeline has run and before anything
    consumes the return code, since that read removes it.

    Writing into `metadata` is what persists this: triton serialises the whole
    dict to JSON beside every kernel it compiles, and hands the same dict to
    `triton.knobs.compilation.listener`. Nothing here opens a file.
    """
    from triton._C.libtriton import ascend

    seed = _get_int_attr(mod, _SEED_ATTR, ascend)
    status = _get_int_attr(mod, _STATUS_ATTR, ascend)
    usable = _get_int_attr(mod, _USABLE_ATTR, ascend)
    total = _get_int_attr(mod, _TOTAL_ATTR, ascend)
    # Read, not consumed: _adjust_metadata_by_module_result still has to find
    # it. Absent means the pass set no error code, which is success.
    rc = _get_int_attr(mod, _RC_ATTR, ascend) or 0

    # Ours, and read by nobody else, so they go no further than this.
    for attr in (_SEED_ATTR, _STATUS_ATTR, _USABLE_ATTR, _TOTAL_ATTR):
        _remove_attr(mod, attr, ascend)

    if seed is None and status is None:
        # The pass published nothing: an older binding, or a build without the
        # attributes. Say so once rather than writing a row that reads like a
        # kernel the search declined.
        _log(f"{kernel_name}: this build publishes no seed decision;"
             " nothing recorded")
        return

    metadata["cv_kernel_name"] = kernel_name
    metadata["cv_seed"] = seed if seed is not None else NO_SEED
    metadata["cv_search_status"] = status
    metadata["cv_search_status_name"] = _status_name(status, rc)
    metadata["cv_variants_usable"] = usable
    metadata["cv_variants_total"] = total
    metadata["cv_rc"] = rc
    metadata["cv_seed_requested"] = requested_seed

    for key, attr in _COST_ATTRS.items():
        metadata[key] = _get_int_attr(mod, attr, ascend)

    # The search ranks on the larger of two bounds and separates ties on the
    # other one, so the pair is what a row has to carry; which of the two it
    # was is not recoverable from the total alone.
    bounds = [metadata["cv_cost_resource"], metadata["cv_cost_recurrence"]]
    metadata["cv_second_bound"] = min(b for b in bounds if b is not None) if any(
        b is not None for b in bounds) else None

    if _should_log(requested_seed):
        # The one line that answers "which kernel, which ordering, and on whose
        # say-so", next to the cost model's own report in the same stream.
        parts = [
            f"seed {metadata['cv_seed']}",
            metadata["cv_search_status_name"],
            f"source {_source_name(status)}",
        ]
        if total:
            parts.append(f"{usable or 0}/{total} ordering(s) usable")
        cost = metadata["cv_cost"]
        parts.append(f"estimate {cost} cycles" if cost is not None
                     else "no estimate")
        if rc:
            parts.append(f"rc {rc}")
        _log(f"{kernel_name} (hash {metadata.get('hash')}): " + ", ".join(parts))


def _source_name(status):
    """Where the seed came from, in the terms the user chose it with."""
    if status == SEARCH_SEED_SUPPLIED:
        return "table"
    if status in (SEARCH_SEED_FOUND, SEARCH_BASELINE_BEST, SEARCH_NONE_USABLE):
        return "search"
    if os.getenv("TRITON_ASCEND_REORDER_SEED"):
        return "TRITON_ASCEND_REORDER_SEED"
    return "none"


def _should_log(requested_seed):
    """Whether this compilation is one the user is watching.

    An ordinary build of an A5 kernel should not grow a line of stderr because
    this bookkeeping exists. A build that asked for a search, replayed a table,
    or turned the cost model's own report on, is one where the seed is the
    thing being looked at.
    """
    if requested_seed >= 0 or table_path() is not None:
        return True
    if os.getenv("TRITON_ASCEND_CV_VARIANTS"):
        return True
    return bool(os.getenv("TRITON_ASCEND_CV_COST_VERBOSE"))
