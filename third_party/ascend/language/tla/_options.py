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
"""TLA launch options and synchronization policy."""

_EXTRA_OPTIONS = "npuir_extra_options"


def normalize_extra_options(value):
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{_EXTRA_OPTIONS} must be a list or tuple of strings")
    options = tuple(value)
    if any(not isinstance(arg, str) for arg in options):
        raise TypeError(f"{_EXTRA_OPTIONS} must contain only strings")
    if any("\0" in arg for arg in options):
        raise ValueError(f"{_EXTRA_OPTIONS} cannot contain NUL characters")
    return options


_INTRA = "disable-hivm-auto-inject-sync"
_CROSS = "disable-auto-inject-block-sync"


def _flag_override(flag, argv):
    for arg in reversed(argv):
        name, _, value = arg.partition("=")
        if name in ("-" + flag, "--" + flag):
            return value.lower() not in ("false", "0")
    return None


def infer_sync_options(features, disable_block_sync, argv):
    manual_intra = features.has_intra_core_sync
    intra = _flag_override(_INTRA, argv)
    cross = _flag_override(_CROSS, argv)
    if cross is not None and disable_block_sync is not None:
        raise ValueError("Specify the TLA cross-core synchronization switch only once")
    if cross is None:
        cross = disable_block_sync

    if features.uses_pipeline:
        if (manual_intra or features.has_cross_core_sync or features.has_unit_flag or features.has_multi_buffer or intra
                or cross):
            raise ValueError("tla.pipeline owns synchronization and buffer rotation; remove handwritten "
                             "synchronization, buffer version hints and auto-sync disable flags")
        return ()

    if features.has_multi_buffer and (manual_intra or features.has_unit_flag or intra):
        raise ValueError("TLA multibuffering requires compiler-managed intra-core synchronization; remove handwritten "
                         "intra-core synchronization and the auto-sync disable flag")
    if manual_intra and intra is False:
        raise ValueError("TLA cannot combine handwritten and automatic intra-core synchronization")
    if cross is False:
        raise ValueError("TLA requires handwritten cross-core synchronization outside tla.pipeline")

    flags = []
    if manual_intra and intra is None:
        flags.append(f"--{_INTRA}=true")
    if cross is None:
        flags.append(f"--{_CROSS}=true")
    return tuple(flags)
