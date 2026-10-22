#!/usr/bin/env python3
"""Wait for the prebuilt Ascend LLVM package this build is going to download.

Used by integration-tests-ascend.yml on ``pull_request`` events. When a PR
changes ``cmake/llvm-hash.txt`` or the LLVM patches, the canonical OBS object
(``llvm-builds/<name>.tar.gz``) does not exist yet: ``llvm-build.yml`` builds
it from the PR (without secrets, as workflow artifacts) and the
``llvm-build-upload.yml`` relay publishes it under the PR-scoped prefix
``llvm-builds/prs/<head-sha>/``.

This script polls until one of the two URLs is available:

* canonical exists  -> do nothing, the build downloads it as usual;
* PR-scoped exists  -> export ``TRITON_ASCEND_LLVM_URL_PREFIX`` via
  ``$GITHUB_ENV`` so ``setup_patch.py`` downloads from the PR prefix;
* the "LLVM Build" workflow run for this head SHA failed -> fail fast
  instead of burning the full timeout;
* timeout -> fail with an explanatory message.

Environment variables:
  HEAD_SHA           PR head SHA (enables the PR-scoped prefix check)
  GH_TOKEN           GitHub token for run-status checks (optional)
  GITHUB_REPOSITORY  owner/repo (optional)
  GITHUB_API_URL     API base URL (default https://api.github.com)
  TIMEOUT_MINUTES    overall wait budget (default 240)
  POLL_SECONDS       poll interval (default 60)
  GITHUB_ENV         file to export TRITON_ASCEND_LLVM_URL_PREFIX into
"""

import importlib.util
import json
import os
import sys
import time
import types
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS_FILE = "llvm-build.yml"


def _stub_module(name, **attrs):
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    mod.__path__ = []  # make stubs usable as packages
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod


def ensure_bdist_wheel_importable():
    """setup_patch subclasses ``bdist_wheel`` at import time.

    The CI build containers always ship setuptools/wheel, but stub the
    import when neither is available so this script stays standalone.
    The stub class is never instantiated here.
    """
    for mod_name in ("setuptools.command.bdist_wheel", "wheel.bdist_wheel"):
        try:
            __import__(mod_name)
            return
        except Exception:
            continue
    dummy = type("bdist_wheel", (), {})
    for parent in ("setuptools", "setuptools.command", "wheel"):
        _stub_module(parent)
    _stub_module("setuptools.command.bdist_wheel", bdist_wheel=dummy)
    _stub_module("wheel.bdist_wheel", bdist_wheel=dummy)


def load_setup_patch():
    """Load third_party/ascend/build/setup_patch.py without heavy deps.

    The module lives in a package and does ``from .build_npuir import
    build_npuir`` at import time; register a minimal fake package with a
    stubbed ``build_npuir`` so importing it stays cheap and side-effect
    free.
    """
    ensure_bdist_wheel_importable()
    build_dir = REPO_ROOT / "third_party" / "ascend" / "build"
    pkg_name = "_ascend_build"

    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(build_dir)]
    sys.modules[pkg_name] = pkg

    stub = types.ModuleType(f"{pkg_name}.build_npuir")
    stub.build_npuir = lambda *args, **kwargs: None
    sys.modules[f"{pkg_name}.build_npuir"] = stub

    spec = importlib.util.spec_from_file_location(f"{pkg_name}.setup_patch", build_dir / "setup_patch.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def url_exists(url):
    """HEAD-probe a URL; True only on a definitive 200/206."""

    def probe(req):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.status in (200, 206)
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 405):
                return None  # HEAD not allowed; retry with a ranged GET
            return False
        except (urllib.error.URLError, OSError):
            return False  # transient network error; keep polling

    result = probe(urllib.request.Request(url, method="HEAD"))
    if result is None:
        result = probe(urllib.request.Request(url, headers={"Range": "bytes=0-0"}))
    return bool(result)


def llvm_build_run_failed(head_sha):
    """Check the latest 'LLVM Build' run for head_sha via the GitHub API.

    Returns True when it completed unsuccessfully (fail fast), False
    otherwise (missing/in-flight/success all mean: keep waiting).
    """
    token = os.environ.get("GH_TOKEN", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if not (token and repo and head_sha):
        return False
    api = os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")
    url = (f"{api}/repos/{repo}/actions/workflows/{WORKFLOWS_FILE}/runs"
           f"?head_sha={head_sha}&per_page=5")
    req = urllib.request.Request(
        url, headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            runs = json.load(resp).get("workflow_runs", [])
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"  (run-status check failed, ignoring: {exc})")
        return False
    if not runs:
        return False
    latest = runs[0]  # newest first
    if latest.get("status") == "completed" and latest.get("conclusion") != "success":
        print(f"'LLVM Build' run {latest.get('html_url')} completed with "
              f"conclusion '{latest.get('conclusion')}'")
        return True
    return False


def export_pr_prefix(pr_prefix):
    github_env = os.environ.get("GITHUB_ENV")
    if github_env:
        with open(github_env, "a", encoding="utf-8") as f:
            f.write(f"TRITON_ASCEND_LLVM_URL_PREFIX={pr_prefix}\n")
    # Also export in-process for callers that run this as a function.
    os.environ["TRITON_ASCEND_LLVM_URL_PREFIX"] = pr_prefix


def main():
    head_sha = os.environ.get("HEAD_SHA", "").strip()
    timeout_s = float(os.environ.get("TIMEOUT_MINUTES", "240")) * 60
    poll_s = float(os.environ.get("POLL_SECONDS", "60"))

    setup_patch = load_setup_patch()
    # Compute the canonical URL first: the prefix override must not leak
    # into the canonical probe.
    os.environ.pop("TRITON_ASCEND_LLVM_URL_PREFIX", None)
    info = setup_patch._get_ascend_llvm_package_info(REPO_ROOT)
    if info is None:
        print("No prebuilt LLVM package for this platform; nothing to wait for.")
        return 0

    canonical_url = info["url"]
    canonical_prefix = canonical_url.rsplit("/", 1)[0] + "/"
    pr_prefix = pr_url = None
    if head_sha:
        pr_prefix = f"{canonical_prefix}prs/{head_sha}/"
        pr_url = f"{pr_prefix}{info['name']}.tar.gz"

    print(f"LLVM package: {info['name']}")
    print(f"  canonical: {canonical_url}")
    print(f"  pr-scoped: {pr_url or '<HEAD_SHA not set>'}")

    deadline = time.monotonic() + timeout_s
    attempt = 0
    while True:
        attempt += 1
        if url_exists(canonical_url):
            print(f"Canonical package is available (attempt #{attempt}).")
            return 0
        if pr_url and url_exists(pr_url):
            print(f"PR-scoped package is available (attempt #{attempt}); "
                  f"exporting TRITON_ASCEND_LLVM_URL_PREFIX={pr_prefix}")
            export_pr_prefix(pr_prefix)
            return 0
        if llvm_build_run_failed(head_sha):
            print(
                "ERROR: the LLVM build for this PR head failed; the package "
                "will never appear. Check the 'LLVM Build' run, fix the "
                "patches and push again.", file=sys.stderr)
            return 1
        if time.monotonic() >= deadline:
            print(
                f"ERROR: timed out after {timeout_s / 60:.0f} min waiting for "
                f"either {canonical_url} or {pr_url}. The 'LLVM Build' + "
                f"'LLVM Build Upload (PR)' relay may still be running; "
                f"re-run this workflow once they complete.", file=sys.stderr)
            return 1
        print(f"[attempt #{attempt}] neither URL available yet; "
              f"sleeping {poll_s:.0f}s")
        time.sleep(poll_s)


if __name__ == "__main__":
    sys.exit(main())
