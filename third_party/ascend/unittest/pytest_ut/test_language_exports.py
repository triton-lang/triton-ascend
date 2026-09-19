import triton.language as tl
import triton.language.extra.cann.extension as al


def test_cann_vec_ops_exported_from_triton_language():
    for name in ("extract_slice", "get_element", "insert_slice"):
        assert getattr(tl, name) is getattr(al, name)
        assert name in tl.__all__
