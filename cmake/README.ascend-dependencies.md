# Source build dependency inputs

This page lists the authoritative source-dependency input files under `cmake/`
and what consumes each one, so a maintainer updating a dependency knows the
authoritative pin file to edit (plus any applicable local patches) and which
checks to re-run.

Preserve the current Triton baseline: LLVM stays in `llvm-hash.txt` until
the upstream Triton 3.8 migration; the nlohmann-json tag stays in the same
`json-version.txt` text pin upstream Triton uses. Each source dependency has one authoritative pin; a single file format is not required.

| File | Consumers |
| --- | --- |
| `llvm-hash.txt` | Full LLVM commit SHA (Secure Hash Algorithm); Ascend patches remain local files in sorted order. Packaging, the LLVM source-build workflow and the `scripts/build-llvm-project.sh` wrapper each read their own checkout; so do the four installation workflows listed below the table. |
| `json-version.txt` | JSON-library download URL in the Ascend setup hook (`third_party/ascend/build/setup_patch.py`, `get_json_package_info`), reading the tag with `read_text().strip()` like upstream Triton's reader. |
| `bisheng-info.json` | Installer names and SHA-256 for both architectures; the integration, sync and DynamicCVPipeline workflow steps read and verify these inputs. `tla-tests.yml` instead keeps reading the pre-existing `cmake/bisheng-version.txt` pin at its main-dev version. |

Beyond those pins, the existing division of responsibilities is preserved:

- The upstream `setup.py` and `python/build_helpers.py` definitions stay unchanged.
- Ascend package readers, URLs, patch-content hashes and the AlmaLinux ABI
  (Application Binary Interface) exception stay in
  `third_party/ascend/build/setup_patch.py`.
- Explicit toolchain paths and offline behavior retain their existing roles.
- CMake continues to discover installed LLVM/MLIR configs and JSON includes;
  no generic reader, generated mirror or metadata-only CMake module is required.
- BiSheng (the Ascend BiSheng compiler, whose installers are pinned in
  `bisheng-info.json`) workflow steps use `jq -er` for metadata fields; the
  `tla-tests.yml` keeps its existing `cmake/bisheng-version.txt` + `sed` reader;
  LLVM shell steps read the text pin with `cat`; Python/pip remain build prerequisites.
- Packaging uses the Python standard library and does not require jq.

The four installation workflows that read their own checkout of the pinned
LLVM revision are `.github/workflows/integration-tests-ascend.yml`,
`sync-upstream.yml`, `DynamicCVPipeline-ci.yml` and `tla-tests.yml`. They each
build and install Triton-Ascend from that checkout. The integration installer
also retains the main branch's `.tar.gz` payload support; the other three
installers retain their existing `.run` paths. While `tla-tests.yml` reads the
legacy `cmake/bisheng-version.txt` pin, the audit cross-checks that both
BiSheng pins agree — see the `check_bom_consistency.py` paragraph below.

`third_party/ascend/build/check_bom_consistency.py` audits these declared inputs, local patch
names, the installation sources, the legacy BiSheng pin's agreement with
`bisheng-info.json`, the main release-policy LLVM/patch
columns, and workflow cache paths. The BiSheng cross-check is an equality
requirement: it passes only while the legacy `cmake/bisheng-version.txt` pin
agrees with `bisheng-info.json`, which keeps the two pins from drifting apart
until `tla-tests.yml` migrates to the JSON pin. It does not supply production values or
certify arbitrary binaries, runtime environments or complete release matrices.

To update a source dependency, edit its authoritative file and applicable local
patches together. From the repository root (a checkout that includes the files
above) with a Python 3 interpreter and pytest installed — the same environment
the CI check uses — run
`python third_party/ascend/build/check_bom_consistency.py` (a successful run
prints its final line with `passed`) and
`python -m pytest -q third_party/ascend/build_tests` (every test passes; the
count grows only when new metadata checks are added). On a host where `python`
is not aliased, use `python3` as the workflow does. If the BiSheng equality
cross-check fails — `cmake/bisheng-version.txt` disagrees with
`bisheng-info.json` — align the legacy `cmake/bisheng-version.txt` pin with
`bisheng-info.json` (the authoritative pin) and re-run. Compiler cache keys
include the actual metadata and patch bytes; unrelated
documentation changes do not rotate that portion of the key.

Runtime compatibility groups, optional dependency source policies, release
images and download cleanup are separate changes described in
[RFC 2739](https://github.com/triton-lang/triton-ascend/issues/2739).
This repository leaves runtime image/torch defaults and historical release
recipes in their existing roles. Source distribution inclusion is provided by
the existing manifest; it does not replace a real build/install check.
