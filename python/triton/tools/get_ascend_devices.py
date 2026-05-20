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


def is_a5(arch: str) -> bool:
    """True if ``arch`` (an rtGetSocVersion SOC string) is an A5 / 950-class NPU.

    A5 is the only Ascend generation with SIMT support, 248 KiB usable UB, and
    no FFTS. Production silicon reports ``Ascend910_95xx``; pre-release silicon
    reports ``Ascend950PR_*`` / ``Ascend950DT_*``. Note that A3 (``Ascend910_93xx``)
    shares the ``Ascend910_9`` prefix and must NOT match, hence ``Ascend910_95``.
    """
    return arch.startswith("Ascend910_95") or arch.startswith("Ascend950")


ascend_devices = get_ascend_devices()
pci_condition = any("0xd806" in dev for dev in ascend_devices)
npu_smi_condition = check_npu_smi_device()
is_compile_on_910_95 = pci_condition or npu_smi_condition