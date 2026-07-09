import os
import glob
import logging
import subprocess

logger = logging.getLogger(__name__)


def get_ascend_devices():
    devices = []
    pci_path = '/sys/bus/pci/devices/*'
    
    for dev in glob.glob(pci_path):
        try:
            vendor_path = os.path.join(dev, 'vendor')
            device_path = os.path.join(dev, 'device')
            
            if os.path.exists(vendor_path):
                with open(vendor_path, 'r') as f:
                    vendor = f.read().strip()
                
                if vendor == "0x19e5" and os.path.exists(device_path):
                    with open(device_path, 'r') as f:
                        device = f.read().strip()
                        devices.append(device)
        except (IOError, OSError) as e:
            logger.warning(f"can not fetch device {dev}: {e}")
            continue
    
    return devices


def check_npu_smi_device():
    try:
        result = subprocess.run(
            ["npu-smi", "info"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
            timeout=100
        )
        if result.returncode == 0:
            output = result.stdout.lower()
            return "ascend910_95" in output or "ascend950" in output or "910_958b" in output
        return False
    except Exception as e:
        logger.warning(f"can not use command: npu-smi info")
        return False


# Lazy module attribute (PEP 562). `is_compile_on_910_95` is computed on first
# access and cached, so importing this module is cheap and the slow
# `npu-smi info` subprocess only runs when the value is actually needed
# (e.g. at kernel compile time) instead of at import time.
_lazy_is_compile_on_910_95 = None
_lazy_computed = False


def _compute_is_compile_on_910_95():
    global _lazy_is_compile_on_910_95, _lazy_computed
    if _lazy_computed:
        return
    ascend_devices = get_ascend_devices()
    pci_condition = any("0xd806" in dev for dev in ascend_devices)
    npu_smi_condition = check_npu_smi_device()
    _lazy_is_compile_on_910_95 = pci_condition or npu_smi_condition
    _lazy_computed = True


def __getattr__(name):
    if name == 'is_compile_on_910_95':
        _compute_is_compile_on_910_95()
        return _lazy_is_compile_on_910_95
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")