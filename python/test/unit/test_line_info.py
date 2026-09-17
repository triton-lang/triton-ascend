import os
import pytest
 	 
from triton.backends.ascend.utils import _is_debug_line_info_disabled
 	 
class TestDebugLineInfo:
    
    def test_default_not_set(self, monkeypatch):
        monkeypatch.delenv("TRITON_DISABLE_LINE_INFO", raising=False)
        assert _is_debug_line_info_disabled() is False
    
    def test_true_values(self, monkeypatch):
        for value in ["true", "True", "TRUE", "1"]:
            monkeypatch.setenv("TRITON_DISABLE_LINE_INFO", value)
            assert _is_debug_line_info_disabled() is True
    
    def test_false_values(self, monkeypatch):
        for value in ["false", "False", "FALSE", "0"]:
            monkeypatch.setenv("TRITON_DISABLE_LINE_INFO", value)
            assert _is_debug_line_info_disabled() is False
    
    def test_invalid_values(self, monkeypatch):
        for value in ["", "yes", "no", "whatever", "2"]:
            monkeypatch.setenv("TRITON_DISABLE_LINE_INFO", value)
            assert _is_debug_line_info_disabled() is False
