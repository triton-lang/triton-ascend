# Source build dependency inputs

Preserve the current Triton baseline: LLVM stays in `llvm-hash.txt` until
the upstream 3.8 migration; the nlohmann-json tag uses Intel-style
`json-version.txt`. Each source dependency has one authoritative pin; a single file format is not required.

| File | Consumers |
| --- | --- |
| `llvm-hash.txt` | Full LLVM SHA; Ascend patches remain local files in sorted order. Packaging, the LLVM source-build workflow, wrapper and four installation sources read their own checkout. |
| `json-version.txt` | JSON-library download URL in the Ascend setup hook, using `read_text().strip()` as Intel does. |
| `bisheng-info.json` | Installer names and SHA-256 for both architectures; four CI installation steps read and verify these inputs. |

The upstream `setup.py` and `python/build_helpers.py` definitions stay unchanged.
Ascend package readers, URLs, patch-content hashes and the AlmaLinux ABI
exception stay in `third_party/ascend/build/setup_patch.py`.
Explicit toolchain paths and offline behavior retain their existing roles.
CMake continues to discover installed LLVM/MLIR configs and JSON includes.
No generic reader, generated mirror or metadata-only CMake module is required.
Bisheng workflow steps use `jq -er` for metadata fields. LLVM shell steps
read the text pin with `cat`; Python/pip remain build prerequisites.
Packaging uses the Python standard library and does not require jq.
The integration installer also retains main's `.tar.gz` payload support;
the other three installers retain their existing `.run` paths.

`third_party/ascend/build/check_bom_consistency.py` audits these declared inputs, local patch
names, the four installation sources, the main release-policy LLVM/patch
columns, and workflow cache paths. It does not supply production values or
certify arbitrary binaries, runtime environments or complete release matrices.

To update a source dependency, edit its authoritative file and applicable local
patches together. Run `python third_party/ascend/build/check_bom_consistency.py` and
`python -m pytest -q third_party/ascend/build_tests`.
Compiler cache keys include the actual metadata and patch bytes; unrelated
documentation changes do not rotate that portion of the key.

Runtime compatibility groups, optional dependency source policies, release
images and download cleanup are separate changes described in RFC 2739.
This change leaves runtime image/torch defaults and historical release recipes
in their existing roles. Source distribution inclusion is provided by the
existing manifest; it does not replace a real build/install check.
