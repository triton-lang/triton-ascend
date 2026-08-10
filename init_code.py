from setup_ascend import _get_triton_ascend_patch_file, _checkout_file

patch_files, dev_patch_files = _get_triton_ascend_patch_file()
if dev_patch_files:
    _checkout_file(dev_patch_files)
if patch_files:
    _checkout_file(patch_files)
