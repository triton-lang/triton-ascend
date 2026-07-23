#!/usr/bin/env python3
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the triton-ascend project.
#
"""
Translate Chinese Markdown files (docs/zh/) to English (docs/en/).

Usage:
    # Translate specific files (paths relative to docs/zh/, no prefix needed)
    python translate_md.py --files "quick_start.md,FAQ.md"
    # Or with full path
    python translate_md.py --files "docs/zh/quick_start.md"
    # Translate all changed/new files
    python translate_md.py --all
    # First-time: translate ALL files except excluded dirs
    python translate_md.py --first-all --exclude-dirs "python-api,triton_api,triton_api_extension,libdevice"
    # Incremental with exclusions
    python translate_md.py --all --exclude-dirs "python-api,triton_api,triton_api_extension,libdevice"
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI

ZH_DIR = Path("docs/zh")
EN_DIR = Path("docs/en")

SYSTEM_PROMPT = ("You are a professional technical documentation translation expert, "
                 "proficient in Chinese-to-English technical document translation.")

TRANSLATION_PROMPT = """Translate the following Chinese technical documentation (Markdown format) into English.

Rules:
1. Preserve all Markdown formatting: headings (#), lists (-, *), code blocks (```), inline code (`), tables, links, images, etc.
2. Keep code blocks, code snippets, command-line examples, and inline code completely unchanged (do NOT translate code).
3. Keep all variable names, function names, class names, file paths, and URLs unchanged.
4. Keep all YAML/TOML/JSON configuration blocks unchanged in content (only translate surrounding text if any).
5. Use standard English technical terminology. Keep technical terms accurate and consistent.
6. For proper nouns (person names, company names, product names, contributor names), keep them as-is.
7. Preserve the exact same number of blank lines and paragraph structure.
8. Return ONLY the translated Markdown content, no extra explanations or comments.
9. If a sentence is too difficult to translate, keep the original Chinese as-is rather than producing incorrect English.

Here is the content to translate:

{content}"""


class MarkdownTranslator:

    def __init__(self, api_key: str, max_concurrent: int = 5):
        self.client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.max_concurrent = max_concurrent

    def _relative_path(self, zh_path: Path) -> Path:
        """Convert a docs/zh/... path to its corresponding docs/en/... path.
        If the filename ends with '_zh', strip the '_zh' suffix."""
        rel = zh_path.relative_to(ZH_DIR)
        if rel.stem.endswith('_zh'):
            rel = rel.parent / (rel.stem[:-3] + rel.suffix)
        return EN_DIR / rel

    async def _call_api(self, content: str, file_info: str = "") -> str | None:
        """Make a single translation API call."""
        prompt = TRANSLATION_PROMPT.format(content=content)
        system = SYSTEM_PROMPT
        if file_info:
            system = f"{SYSTEM_PROMPT} (Translating: {file_info})"
        response = await self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=16384,
            temperature=0.3,
        )
        text = response.choices[0].message.content
        return self._clean_response(text) if text else None

    async def translate_file(self, zh_path: Path) -> bool:
        """Translate a single Markdown file from zh to en."""
        if not zh_path.exists() or zh_path.suffix != ".md":
            print(f"  Skip: {zh_path} (not found or not .md)")
            return False

        # Read source content
        content = zh_path.read_text(encoding="utf-8")
        lines = content.split("\n")

        # Skip empty files
        if not content.strip():
            print(f"  Skip: {zh_path.name} (empty)")
            return False

        en_path = self._relative_path(zh_path)
        print(f"  {zh_path.name} \u2192 {en_path} ({len(lines)} lines)", end=" ", flush=True)

        try:
            result = await self._call_api(content, file_info=str(zh_path))
            if not result:
                print("FAILED (empty response)")
                return False

            # Write to target path
            en_path.parent.mkdir(parents=True, exist_ok=True)
            en_path.write_text(result, encoding="utf-8")
            print("OK")
            return True
        except Exception as e:
            print(f"ERROR: {e}")
            return False

    async def translate_files(self, file_list: list[Path], output_json: str) -> int:
        """Translate a list of files and save results."""
        print(f"Translating {len(file_list)} file(s), max_concurrent={self.max_concurrent}")

        sem = asyncio.Semaphore(self.max_concurrent)

        async def do_translate(path: Path) -> tuple[Path, bool]:
            async with sem:
                ok = await self.translate_file(path)
                return (path, ok)

        tasks = [do_translate(fp) for fp in file_list]
        results = await asyncio.gather(*tasks)

        success_files = []
        for path, ok in results:
            if ok:
                success_files.append(str(self._relative_path(path)))

        print(f"\nResult: {len(success_files)}/{len(file_list)} translated")

        # Always write the JSON, even if no files were translated
        report = {
            "success_files": success_files,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_files": len(file_list),
            "success_count": len(success_files),
        }
        out = Path(output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Results written to {output_json}")

        return 0 if success_files else 1

    @staticmethod
    def _is_excluded(path: Path, exclude_dirs: list[str]) -> bool:
        """Check if a path falls under any of the excluded directories."""
        if not exclude_dirs:
            return False
        try:
            rel = path.relative_to(ZH_DIR)
        except ValueError:
            return False
        # Check if the relative path starts with any of the excluded dirs
        for excl in exclude_dirs:
            if str(rel).startswith(excl + os.sep) or str(rel) == excl:
                return True
        return False

    @staticmethod
    def filter_excluded(file_list: list[Path], exclude_dirs: list[str]) -> list[Path]:
        """Filter out files that are in excluded directories."""
        return [f for f in file_list if not MarkdownTranslator._is_excluded(f, exclude_dirs)]

    @staticmethod
    def find_all_zh_files(exclude_dirs: list[str] | None = None) -> list[Path]:
        """Find ALL .md files in docs/zh/, optionally excluding certain directories."""
        exclude_dirs = exclude_dirs or []
        result = []
        for md_file in ZH_DIR.rglob("*.md"):
            if not MarkdownTranslator._is_excluded(md_file, exclude_dirs):
                result.append(md_file)
        return sorted(result)

    @staticmethod
    def find_changed_zh_files(exclude_dirs: list[str] | None = None) -> list[Path]:
        """Find docs/zh/ .md files that differ from their docs/en/ counterpart
        (or have no en counterpart yet). Uses git diff if available.
        Optionally excludes files under certain directories."""
        exclude_dirs = exclude_dirs or []
        import subprocess

        # Try to get changed files via git diff
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "--diff-filter=ACMUX", "HEAD", "--", "docs/zh/"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                changed = []
                for line in result.stdout.strip().split("\n"):
                    line = line.strip()
                    if line.endswith(".md"):
                        changed.append(Path(line))
                if changed:
                    # Filter out excluded dirs
                    return MarkdownTranslator.filter_excluded(changed, exclude_dirs)
        except Exception:
            pass

        # Fallback: find all .md files in docs/zh/ that differ from docs/en/
        # (handles _zh suffix in filename being stripped for the en counterpart)
        def _to_en_path(zh_md: Path) -> Path:
            rel = zh_md.relative_to(ZH_DIR)
            if rel.stem.endswith('_zh'):
                rel = rel.parent / (rel.stem[:-3] + rel.suffix)
            return EN_DIR / rel

        changed = []
        for md_file in ZH_DIR.rglob("*.md"):
            if MarkdownTranslator._is_excluded(md_file, exclude_dirs):
                continue
            en_file = _to_en_path(md_file)
            if not en_file.exists():
                changed.append(md_file)
            elif md_file.read_text(encoding="utf-8") != en_file.read_text(encoding="utf-8"):
                changed.append(md_file)

        return changed

    @staticmethod
    def _clean_response(response: str) -> str:
        """Strip markdown code block wrappers from API response."""
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            lines = lines[1:]  # remove opening ```
            while lines and lines[-1].strip() == "```":
                lines.pop()
            response = "\n".join(lines).strip()
        return response


def write_empty_json(output_json: str, reason: str = ""):
    """Write an empty result JSON so the workflow can check it."""
    report = {
        "success_files": [],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_files": 0,
        "success_count": 0,
        "note": reason,
    }
    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Empty result written to {output_json} (reason: {reason})")


async def async_main():
    parser = argparse.ArgumentParser(description="Translate docs/zh/ Markdown to docs/en/")
    parser.add_argument("--files",
                        help="Comma-separated file paths to translate (relative to docs/zh/, e.g. quick_start.md)")
    parser.add_argument("--all", action="store_true", help="Translate all changed/new .md files")
    parser.add_argument("--first-all", action="store_true",
                        help="Translate ALL .md files under docs/zh/ (used for first-time translation)")
    parser.add_argument("--exclude-dirs",
                        help="Comma-separated directory names under docs/zh/ to exclude (e.g. python-api,triton_api)")
    parser.add_argument("--output-json", default=os.getenv("OUTPUT_JSON", "/tmp/translation_results.json"))
    parser.add_argument("--api-key", default=os.getenv("DEEPSEEK_API_KEY"))
    parser.add_argument("--max-concurrent", type=int, default=5)
    args = parser.parse_args()

    output_json = args.output_json
    exclude_dirs = [d.strip() for d in args.exclude_dirs.split(",") if d.strip()] if args.exclude_dirs else []

    if exclude_dirs:
        print(f"Excluding directories: {exclude_dirs}")

    # Check API key early
    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        msg = "DEEPSEEK_API_KEY not set"
        print(f"Error: {msg}")
        write_empty_json(output_json, msg)
        return 1

    # Determine file list
    if args.files:
        raw_files = [f.strip() for f in args.files.split(",") if f.strip()]
        file_list = []
        for f in raw_files:
            # Support both "quick_start.md" and "docs/zh/quick_start.md"
            p = Path(f)
            if not p.exists():
                # Try prepending docs/zh/
                p = ZH_DIR / f
            file_list.append(p)
        # Filter out excluded dirs
        file_list = MarkdownTranslator.filter_excluded(file_list, exclude_dirs)
        if not file_list:
            print("All specified files are in excluded directories. Nothing to translate.")
            write_empty_json(output_json, "all files excluded")
            return 0
    elif args.first_all:
        file_list = MarkdownTranslator.find_all_zh_files(exclude_dirs=exclude_dirs)
        if not file_list:
            print("No .md files found under docs/zh/ to translate (all excluded).")
            write_empty_json(output_json, "no files to translate after exclusion")
            return 0
        print(f"First-time translation: found {len(file_list)} file(s) to translate")
    elif args.all:
        file_list = MarkdownTranslator.find_changed_zh_files(exclude_dirs=exclude_dirs)
    else:
        msg = "specify --files, --all, or --first-all"
        print(f"Error: {msg}")
        write_empty_json(output_json, msg)
        return 1

    # Validate files exist
    valid_files = [f for f in file_list if f.exists() and f.suffix == ".md"]
    skipped = [str(f) for f in file_list if f not in valid_files]
    if skipped:
        print(f"Skipped (not found or not .md): {skipped}")

    if not valid_files:
        print("No valid .md files to translate")
        write_empty_json(output_json, "no valid .md files to translate")
        return 0

    translator = MarkdownTranslator(api_key=api_key, max_concurrent=args.max_concurrent)
    return await translator.translate_files(valid_files, output_json)


if __name__ == "__main__":
    sys.exit(asyncio.run(async_main()))
