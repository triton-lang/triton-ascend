import itertools
import json

import pytest

from triton._C.libproton import proton as libproton

_function_ids = itertools.count(9000000)


@pytest.fixture(params=["npu", "cuda", "hip"])
def metadata_session(tmp_path, request):
    backend = request.param
    try:
        session = libproton.start(str(tmp_path / "metadata"), "shadow", "tree", "instrumentation", backend)
    except RuntimeError as error:
        if "support is not built" in str(error):
            pytest.skip(str(error))
        raise
    yield backend
    libproton.finalize(session, "hatchet")


def test_duplicate_metadata_policy(tmp_path, metadata_session):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps({"profile_scratch_size": 4096, "num_warps": 1, "profile_num_units": 1}))
    function = next(_function_ids)
    args = (function, "kernel", [(0, "scope")], [], str(path))
    libproton.init_function_metadata(*args)
    if metadata_session == "npu":
        libproton.init_function_metadata(*args)
    else:
        with pytest.raises(RuntimeError, match="Duplicate function id"):
            libproton.init_function_metadata(*args)
    with pytest.raises(RuntimeError, match="Duplicate function id"):
        libproton.init_function_metadata(function, "other", [(0, "scope")], [], str(path))
    with pytest.raises(RuntimeError, match="Duplicate function id"):
        libproton.init_function_metadata(function, "kernel", [(0, "changed")], [], str(path))
    with pytest.raises(RuntimeError, match="Duplicate function id"):
        libproton.init_function_metadata(function, "kernel", [(0, "scope"), (0, "scope")], [], str(path))
    with pytest.raises(RuntimeError, match="Duplicate function id"):
        libproton.init_function_metadata(function, "kernel", [(0, "scope")], [(0, 1)], str(path))
    path.write_text(json.dumps({"profile_scratch_size": 8192, "num_warps": 1, "profile_num_units": 1}))
    with pytest.raises(RuntimeError, match="Duplicate function id"):
        libproton.init_function_metadata(*args)
