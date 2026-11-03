from third_party.ascend.build import get_triton_ascend_patch_file, checkout_file
from third_party.ascend.build.setup_patch import _apply_proton_patch

_apply_proton_patch(False)
patch_files, dev_patch_files = get_triton_ascend_patch_file()
if dev_patch_files:
    checkout_file(dev_patch_files)
if patch_files:
    checkout_file(patch_files)
