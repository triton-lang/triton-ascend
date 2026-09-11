# bisheng-package

Packaging scripts for the `ascendnpu-ir_*.run` installer built by
[`bisheng-build.yml`](../bisheng-build.yml).

The scripts were extracted verbatim from the Huawei-produced OBS package
(`ascendnpu-ir_1.2.0_linux-x86-pr_2863.run`, a Makeself 2.5.0 archive with the
`ASCEND_RUN_PACKAGE` label) so that self-built packages behave exactly like the
prebuilt ones. Only a few adaptations were made:

- `script/install.sh` parses its arguments from `$1` instead of `$3`: the
  original custom makeself header prepended two synthetic arguments before the
  user's args; our header (below) forwards user args verbatim, so `--install` /
  `--install-path=...` land at `$1` / `$2`.
- `BASE_PACKAGE_VERSION` / `PACKAGE_VERSION` (originally `1.1.0`) and the
  `TARGET_BUILD_ARCH` placeholder are substituted at packaging time by
  `bisheng-build.yml`.
- `makeself-header.sh` is the stock makeself 2.5.0 header plus a whitelist
  that forwards `--install`, `--install-path=...`, `--uninstall`, `--version`,
  `--run`, `--full`, `--install-for-all`, and `--upgrade` to `install.sh`
  (stock makeself rejects unknown flags; the original package's custom header
  forwarded these). Passed to `makeself.sh --header ...`.
- `help.txt` is the original `--help-header` text, reproduced from the
  payload's `script/help.sh`.

`set_env.sh` is an empty placeholder; `install.sh` rewrites it with the real
install path at install time.

The installer is Linux-only (like the CANN run packages it mimics); BSD `sed`
on macOS is not supported.

## Installer layout

The payload directory passed to makeself is:

```
bisheng_toolkit/
├── bishengir/
│   ├── bin/   bishengir-compile, bishengir-opt, hivmc, hivmc-a5
│   └── lib/   host.bc + meta_op.{aic,aiv,mix}.{c220,c310}.bc
├── script/    install.sh, uninstall.sh, cann_uninstall.sh
└── set_env.sh
```

`--install --install-path=<path>` installs `bishengir/` under
`<path>/tools/bishengir/` and symlinks the binaries into `<path>/<arch>-linux/bin/`,
which is what `integration-tests-ascend.yml` relies on
(`/usr/local/bisheng/tools/bishengir/bin`).
