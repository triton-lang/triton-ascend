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
Sphinx gettext translation workflow for Chinese docs.

This script translates docs/zh/ Chinese documentation to English .po files
using Sphinx gettext and the DeepSeek API.

Path mapping (PO files mirror docs/zh/ directory structure):
    docs/zh/debug_guide/index.md
        → locale/en/LC_MESSAGES/debug_guide/index.po

    The relative path under locale/en/LC_MESSAGES/ (e.g. "debug_guide/index.po")
    is IDENTICAL to the relative path under docs/zh/ (e.g. "debug_guide/index.md")
    except for the .md → .po extension change.

The English build is driven by docs/zh/conf.py (srcdir=docs/zh/,
locale_dirs=['../locale/']); .github/workflows/scripts/readthedocs_checkout.sh
selects it for READTHEDOCS_LANGUAGE=en. Sphinx with language='en' reads
locale/en/LC_MESSAGES/<relative-path>.po to translate
docs/zh/<relative-path>.md into English HTML output.

Directory layout:
    docs/zh/                        Chinese source Markdown files (input)
    locale/                         Sphinx gettext output & translation dir
    locale/*.pot                    Raw .pot files (before organization)
    locale/zh/LC_MESSAGES/          Organized .pot files (source language)
    locale/en/LC_MESSAGES/          Translated .po files (target language, mirrors docs/zh/ structure)

Excluded directories (not translated):
    python-api/
    triton_api/
    triton_api_extension/
    libdevice/

Excluded files (not translated):
    community/CODE_OF_CONDUCT_zh.md
    community/CONTRIBUTING_zh.md
    community/GOVERNANCE_zh.md
    community/SECURITYNOTE_zh.md
    community/community_technical_meeting.md
    community/roadmap_guide.md
    community/CONTRIBUTOR.md
    community/MAINTAINERS.md
    (The English site renders the canonical English document directly via the
    source-read hook in docs/zh/conf.py: the community pages take their English
    source from the repository root (CODE_OF_CONDUCT.md, CONTRIBUTING.md,
    GOVERNANCE.md, SECURITYNOTE.md, CONTRIBUTOR.md, MAINTAINERS.md) or from the
    English docs tree (docs/en/community/community_technical_meeting.md,
    docs/en/community/roadmap_guide.md).)

Usage:
    # First-time: generate .pot files and translate ALL
    python translate_md.py --first-time

    # Incremental: translate only changed .pot files
    python translate_md.py --all

Translation skill:
    Every API call injects the translation skill document
    (.agents/skills/translate/skill.md) as part of the system
    prompt so terminology/glossary stays consistent across all documents.
    Use --skill-doc (or TRANSLATION_SKILL_DOC env var) to override the path.
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

from openai import AsyncOpenAI, AuthenticationError

# ---------------------------------------------------------------------------
# Directory Layout (relative to project root = triton-ascend/)
# ---------------------------------------------------------------------------
# Source:         docs/zh/                              Chinese .md / .rst files
# Pot output:     locale/zh/LC_MESSAGES/                 .pot templates (after reorganize)
# Po output:      locale/en/LC_MESSAGES/                 English .po translations
#                                                         (mirrors docs/zh/ structure)

ZH_DIR = Path("docs/zh")
LOCALE_DIR = Path("docs/locale")
POT_DIR = LOCALE_DIR / "zh" / "LC_MESSAGES"
PO_DIR = LOCALE_DIR / "en" / "LC_MESSAGES"

# PO timestamps use Beijing time (UTC+8) so headers read as local time
# regardless of the CI runner's timezone (GitHub Actions runs on UTC).
_BEIJING_TZ = timezone(timedelta(hours=8))

# ---------------------------------------------------------------------------
# LLM provider configuration
# ---------------------------------------------------------------------------
# The translation engine uses the OpenAI-compatible chat-completions API, so
# any provider with an OpenAI-compatible endpoint can be plugged in by
# overriding the base URL and model name. Both can be overridden via
# LLM_API_BASE / LLM_MODEL / --api-base / --model.
DEFAULT_API_BASE = "https://st8tp3ajl0df3n8b8l8qu.apigateway-cn-beijing.volceapi.com/v1"
DEFAULT_MODEL = "glm-5.2"

# ---------------------------------------------------------------------------
# Exclusions
# ---------------------------------------------------------------------------
# Directories under docs/zh/ that should NOT be translated.
# The path from locale/en/LC_MESSAGES/ mirrors docs/zh/, so excluding here
# means their entire subtree is skipped at both pot and po generation time.
EXCLUDED_DIRS: List[str] = [
    "python-api",
    "triton_api",
    "triton_api_extension",
    "libdevice",
]

# Individual source files (by stem, without extension) to exclude.
# The stem is the filename WITHOUT extension, e.g. "CODE_OF_CONDUCT_zh".
# These community documents are NOT translated because the English site renders
# their canonical English source directly (see the source-read hook in
# docs/zh/conf.py): the .po catalogues of the community pages are not used.
EXCLUDED_FILE_STEMS: List[str] = [
    "CODE_OF_CONDUCT_zh",
    "CONTRIBUTING_zh",
    "GOVERNANCE_zh",
    "SECURITYNOTE_zh",
    "community_technical_meeting",
    "roadmap_guide",
    "CONTRIBUTOR",
    "MAINTAINERS",
]

# ---------------------------------------------------------------------------
# Translation prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = ("You are a professional technical documentation translation expert, "
                 "proficient in Chinese-to-English technical document translation. "
                 "Before translating anything you MUST read and follow the translation "
                 "standard embedded in the skill document below (the key points of the "
                 "Google Developer Documentation Style Guide, https://developers.google.com/style). "
                 "Skipping the translation standard produces non-compliant translations "
                 "that must be redone, so the standard is mandatory for every request.")

BLOCK_TRANSLATION_PROMPT = """Translate the following Chinese text block into English.

Rules:
1. Return ONLY the translated text, no explanations, no markdown fences.
2. Preserve the EXACT original structure: markdown/RST heading markers
   (#, ==, --, ~~), list prefixes (-, 1., 4.1), table separators (|, ----),
   inline markup (`, **, *), links, and code fences. Do NOT renumber,
   reorder, merge, or split lines, paragraphs, list items, table rows, or
   code blocks.
2b. Keep the leading whitespace (spaces/tabs) of EVERY line exactly as in
   the source, especially inside code blocks and indented RST blocks.
3. Follow the translation standard (Google Developer Documentation Style
   Guide): prefer active voice (imperative for instructions), use simple
   present tense, use second person ("you"), write complete sentences with
   explicit subjects and verbs, put articles (a/an/the) before singular
   countable nouns, keep sentences under ~25 words, and use parallel
   structures.
4. Do NOT use foreign words (etc., e.g., i.e., via) or contractions
   (can't, it's, don't) - use "and so on", "for example", "that is",
   "through/by/using", "cannot", "it is", "do not" instead.
5. Proper nouns (product names, environment variables, API identifiers,
   repository names) stay exactly as-is.
6. In code blocks or inline code, translate ONLY the Chinese comments and
   string literals; leave all code syntax, variable names, and keywords
   unchanged.
7. If any sentence is too ambiguous to translate faithfully, keep the
   original Chinese as-is; never guess.

Text to translate:
{content}"""

# ---------------------------------------------------------------------------
# Translation skill document
# ---------------------------------------------------------------------------
# The skill document (.agents/skills/translate/skill.md)
# defines the authoritative Chinese -> English terminology glossary and
# translation rules for this project. It is injected into every translation
# request so that terminology stays consistent across all documents.
#
# The document is designed to be extensible: contributors can add new
# glossary entries / rules to the Markdown file and the translation pipeline
# picks them up automatically on the next run. The path can be overridden
# via the --skill-doc CLI argument or the TRANSLATION_SKILL_DOC env var.
SKILL_DOC_PATH = Path(".agents/skills/translate/skill.md")

_skill_doc_cache: Optional[str] = None


def load_skill_doc(path: Optional[Path] = None) -> str:
    """Load the translation skill document content.

    Returns the raw skill Markdown content, or an empty string when the
    document is missing / unreadable (the pipeline then falls back to the
    plain system prompt).

    The result is cached module-wide so concurrent calls in the same run
    only read the file once. A non-None ``path`` (from --skill-doc) clears
    the cache so a caller-specified document is always honored.
    """
    global _skill_doc_cache
    if path is not None:
        _skill_doc_cache = None
        skill_path = path
    else:
        skill_path = SKILL_DOC_PATH
    if _skill_doc_cache is not None:
        return _skill_doc_cache
    try:
        if not skill_path.exists():
            print(f"  Skill doc not found: {skill_path} (translating without skill)", flush=True)
            _skill_doc_cache = ""
            return _skill_doc_cache
        _skill_doc_cache = skill_path.read_text(encoding="utf-8").strip()
        print(f"  Loaded translation skill doc: {skill_path} ({len(_skill_doc_cache)} chars)", flush=True)
    except OSError as e:
        print(f"  Warning: failed to read skill doc {skill_path}: {e} (translating without skill)", flush=True)
        _skill_doc_cache = ""
    return _skill_doc_cache


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _source_rel_from_pot(pot_path: Path) -> Optional[Path]:
    """Convert a .pot file path to the source file path relative to docs/zh/.

    This is the REVERSE mapping. Given a pot file at e.g.
    locale/zh/LC_MESSAGES/foo/bar.pot, return the source path
    docs/zh/foo/bar.md.

    The relative path under the locale directory structure mirrors docs/zh/
    exactly. For pot files under LC_MESSAGES/, the relative component AFTER
    LC_MESSAGES/ gives the source-relative path. For pot files directly
    under locale/ (before reorganization), the relative path from locale/
    gives the source-relative path.
    """
    try:
        rel = pot_path.relative_to(LOCALE_DIR)
    except ValueError:
        return None
    # If already under LC_MESSAGES/, everything after LC_MESSAGES/ is the source path
    if "LC_MESSAGES" in rel.parts:
        lc_index = rel.parts.index("LC_MESSAGES")
        source_rel = Path(*rel.parts[lc_index + 1:])
    else:
        # Before reorganization: e.g. locale/foo/bar.pot → foo/bar.pot
        source_rel = rel
    # .pot → .md (or .rst, but default to .md)
    return source_rel.with_suffix('.md')


def _get_source_commit(source_md: Path) -> str:
    """Return a stable content fingerprint of a source .md at HEAD.

    Uses the file's blob object id: `git rev-parse HEAD:<repo-rel-path>`.

    Why blob id instead of `git log -1 -- <path>`? GitHub Actions checkouts
    are shallow (fetch-depth: 1) by default, so `git log -- <path>` typically
    returns NOTHING for files whose history predates the shallow boundary and
    the old code fell back to `git rev-parse HEAD`. That made every unrelated
    merge (e.g. an auto-translation PR touching only .po files) change the
    fingerprint, re-selecting every document.

    The blob id equals `git hash-object` of the file's version at HEAD: it
    changes only when the file's CONTENT changes, and it works on shallow clones.
    """
    if not source_md.exists():
        return ""
    try:
        # `HEAD:<path>` needs a path relative to the repo root (cwd).
        cwd = Path.cwd().resolve()
        try:
            rel = source_md.resolve().relative_to(cwd)
        except ValueError:
            return ""
        res = subprocess.run(
            ["git", "rev-parse", f"HEAD:{rel.as_posix()}"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _find_source_for_pot(pot_path: Path) -> Optional[Path]:
    """Locate the actual source file (docs/zh/**) behind a .pot / .po.

    Prefers the .md sibling (existing documents are Markdown), falls back to
    the .rst sibling (e.g. docs/zh/index.rst -> index.po), and returns None
    when neither exists (e.g. sphinx.po has no source document at all).
    """
    rel = pot_path.relative_to(POT_DIR) if POT_DIR is not None else Path()
    md = ZH_DIR / rel.with_suffix('.md')
    if md.exists():
        return md
    rst = ZH_DIR / rel.with_suffix('.rst')
    if rst.exists():
        return rst
    return None


def _po_source_path(po_path: Path) -> Optional[Path]:
    """The source behind a .po path (same relative mapping as _find_source_for_pot)."""
    if PO_DIR is None:
        return None
    try:
        rel = po_path.relative_to(PO_DIR)
    except ValueError:
        rel = po_path
    md = ZH_DIR / rel.with_suffix('.md')
    if md.exists():
        return md
    rst = ZH_DIR / rel.with_suffix('.rst')
    return rst if rst.exists() else None


def _get_po_source_commit(po_path: Path) -> str:
    """Read the X-Source-Commit header field recorded in a .po file."""
    try:
        raw = po_path.read_text(encoding="utf-8")
    except OSError:
        return ""
    m = re.search(r'X-Source-Commit:\s*([0-9a-fA-F]{7,40})', raw)
    return m.group(1) if m else ""


def _is_excluded_pot(pot_path: Path) -> bool:
    """Check if a .pot file corresponds to an excluded source file or directory.

    Exclusion is based on the source-relative path:
    - If the first path component (the subdirectory name) is in EXCLUDED_DIRS
    - If the filename stem is in EXCLUDED_FILE_STEMS
    """
    # Check filename stem
    if pot_path.stem in EXCLUDED_FILE_STEMS:
        return True

    try:
        rel = pot_path.relative_to(LOCALE_DIR)
    except ValueError:
        return False

    parts = rel.parts
    if not parts:
        return False

    # After organization: locale/zh/LC_MESSAGES/<dir>/<file>.pot
    # After LC_MESSAGES/, parts[0] is the subdirectory name
    # Before organization: locale/<dir>/<file>.pot, parts[0] is also subdirectory
    if "LC_MESSAGES" in parts:
        # locale/zh/LC_MESSAGES/subdir/file.pot → check "subdir"
        lc_index = parts.index("LC_MESSAGES")
        if lc_index + 1 < len(parts):
            return parts[lc_index + 1] in EXCLUDED_DIRS
        return False
    else:
        # locale/subdir/file.pot → check "subdir"
        return parts[0] in EXCLUDED_DIRS


# ---------------------------------------------------------------------------
# PO / POT file parsing
# ---------------------------------------------------------------------------


def parse_pot_file(filepath: Path) -> dict:
    """
    Parse a .pot or .po file and return entries dict.

    Returns dict mapping msgid -> entry dict:
    {
        "msgid": str,
        "msgstr": str,
        "translated": bool,
    }
    """
    entries = {}
    if not filepath.exists():
        return entries

    raw = filepath.read_text(encoding="utf-8")

    blocks = raw.split('\n\n')
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if block.startswith('msgid ""') and '# ' not in block.split('\n')[0]:
            continue

        msgid = _extract_po_value(block, 'msgid')
        msgstr = _extract_po_value(block, 'msgstr')

        if msgid is not None:
            entries[msgid] = {
                "msgid": msgid,
                "msgstr": msgstr or "",
                "translated": bool(msgstr and msgstr.strip()),
            }

    return entries


def _extract_po_value(block: str, field: str) -> Optional[str]:
    """Extract the value of a msgid/msgstr field from a PO block."""
    pattern = re.compile(rf'{field}\s+"((?:[^"\\]|\\.)*)"', re.MULTILINE)
    match = pattern.search(block)
    if match:
        raw = match.group(1)
        return raw.replace('\\"', '"').replace('\\\\', '\\')

    if f'{field} ""' in block:
        lines = block.split('\n')
        in_field = False
        parts = []
        for line in lines:
            if line.startswith(f'{field} ""'):
                in_field = True
                continue
            if in_field:
                m = re.match(r'\s*"((?:[^"\\]|\\.)*)"', line)
                if m:
                    parts.append(m.group(1).replace('\\"', '"').replace('\\\\', '\\'))
                else:
                    break
        if parts:
            return ''.join(parts)
        return ""

    return None


def write_po_file(filepath: Path, entries: dict, source_pot: str = "", changed: bool = True, source_commit: str = "",
                  skill_version: str = ""):
    """Write entries dict to a .po file.

    ``changed`` controls whether the POT/PO creation timestamps are stamped:
    - True (content changed): write the current POT-Creation-Date /
      PO-Revision-Date so the diff clearly marks the file as updated.
    - False (content identical): write a stable header WITHOUT the timestamp
      fields, so an unchanged .po file is rewritten byte-identically and
      produces no git diff (keeps incremental translation PRs minimal).

    ``source_commit`` records the git commit hash of the source Chinese .md
    from which this .po was generated. Incremental runs compare it with the
    source's current commit: if equal, the .po is skipped entirely (no
    rewrite, no API calls), so untouched documents never appear in PRs.

    ``skill_version`` records the skill-document version that was in effect
    when this .po was written. Incremental runs compare it with the current
    skill version: when the skill changes (e.g. the glossary was extended),
    affected .po files are re-processed so terminology edits propagate.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    now_str = datetime.now(_BEIJING_TZ).strftime("%Y-%m-%d %H:%M%z")
    lines = []

    # Header.
    lines.append(f'# English translations for triton-ascend docs.\n'
                 f'# Copyright (c) 2025 Huawei Technologies Co., Ltd.\n'
                 f'#\n'
                 f'msgid ""\n'
                 f'msgstr ""\n'
                 f'"Project-Id-Version: triton-ascend-docs\\n"\n')
    if changed:
        lines.append(f'"POT-Creation-Date: {now_str}\\n"\n'
                     f'"PO-Revision-Date: {now_str}\\n"\n')
    if source_commit:
        lines.append(f'"X-Source-Commit: {source_commit}\\n"\n')
    if skill_version:
        lines.append(f'"X-Skill-Version: {skill_version}\\\\n"\n')
    lines.append(f'"Last-Translator: Auto Translation (DeepSeek)\\n"\n'
                 f'"Language-Team: English\\n"\n'
                 f'"Language: en\\n"\n'
                 f'"MIME-Version: 1.0\\n"\n'
                 f'"Content-Type: text/plain; charset=UTF-8\\n"\n'
                 f'"Content-Transfer-Encoding: 8bit\\n"\n'
                 f'"Plural-Forms: nplurals=2; plural=(n != 1);\\n"\n')

    for entry in entries.values():
        msgid = entry["msgid"]
        msgstr = entry.get("msgstr", "")

        if '\n' in msgid:
            lines.append('msgid ""')
            for part in msgid.split('\n'):
                lines.append(f'"{_escape_po(part)}\\n"')
        else:
            lines.append(f'msgid "{_escape_po(msgid)}"')

        msgstr = _escape_enumeration_prefix(msgstr)
        if '\n' in msgstr:
            lines.append('msgstr ""')
            for part in msgstr.split('\n'):
                lines.append(f'"{_escape_po(part)}\\n"')
        else:
            lines.append(f'msgstr "{_escape_po(msgstr)}"')

        lines.append('')

    text = '\n'.join(lines)
    filepath.write_text(text, encoding="utf-8")


def _escape_po(s: str) -> str:
    s = s.replace('\\', '\\\\')
    s = s.replace('"', '\\"')
    return s


def _escape_enumeration_prefix(s: str) -> str:
    """Escape a leading 'N. ' enumeration prefix in a translated message.

    Sphinx's Locale transform re-parses every msgstr via _publish_msgstr().
    When the translated text starts with a digit followed by '. ' (e.g.
    "2. Support parallel compilation of multiple configs"), the parser
    produces an enumerated_list (ordered list) node instead of a paragraph.
    Locale.apply() only accepts paragraph/title/literal nodes, so the
    translation is silently skipped and the original Chinese heading remains
    in the output.

    Escaping the dot ("2\\. Support ...") makes the parser emit a paragraph
    whose asText() is still "2. Support ...", so headings render correctly.

    Only the first line is considered; later lines are left untouched.
    """
    if not s:
        return s
    lines = s.split('\n')
    first = lines[0]
    m = re.match(r'^(\s*\d+)\.(\s)', first)
    if m:
        lines[0] = f"{m.group(1)}\\.{m.group(2)}{first[m.end():]}"
    return '\n'.join(lines)


def _restore_enumeration_prefix(msgid: str, msgstr: str) -> str:
    """Ensure the translated text keeps the 'N. ' enumeration prefix of its msgid.

    The DeepSeek model sometimes drops the list-number prefix when translating
    Chinese headings, e.g.

        msgid  "1. 安装与环境配置"
        msgstr "Installation and Environment Configuration"

    loses the leading "1. ". Since the msgid prefix is structural (list
    numbering), we re-attach it before the text is written to the .po file
    (later _escape_enumeration_prefix turns it into "1\\. " for Sphinx).

    Only the first non-blank line is considered; sub-sequences (e.g.
    "1.1 ") or other leading numbers already present in the translation are
    left untouched.
    """
    if not msgid or not msgstr:
        return msgstr

    # List-number prefixes come in two shapes: "1. " (top-level item) and
    # "4.1 " / "1.1.2 " (nested item, no trailing dot). Match either, and
    # reuse the source's exact prefix form (dot included only when present).
    m = re.match(r'^\s*(\d+(?:\.\d+)*\.?)\s+', msgid)
    if not m:
        return msgstr

    prefix = m.group(1) + " "

    stripped = msgstr.lstrip('\n')
    leading_ws = msgstr[:len(msgstr) - len(stripped)]

    # Already has the (possibly escaped) prefix -> leave it as-is.
    if re.match(r'^\s*\d+(?:\.\d+)*\.?\s+', stripped) or \
            re.match(r'^\s*\d+(?:\.\d+)*\\.\s+', stripped):
        return msgstr

    return leading_ws + prefix + stripped


_EN_TERM_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\s\-_().,/'+]*$")


def _looks_like_english_term(s: str) -> bool:
    """True if ``s`` looks like a plain English glossary value."""
    return bool(s) and bool(_EN_TERM_PATTERN.match(s)) and not re.search(r"[\u4e00-\u9fff]", s)


def _lower_first_word(w: str) -> str:
    """Lowercase the first letter of a word unless it is an acronym (all caps)."""
    if w.isupper() or len(w) <= 1:
        return w
    return w[:1].lower() + w[1:]


def _casing_variants(authoritative: str) -> List[str]:
    """Generate plausible casing variants of an authoritative term.

    E.g. "Ascend Platform" -> ["Ascend platform", "ascend platform"].
    Acronyms (e.g. "NPU") keep their caps. The authoritative spelling itself
    is excluded from the returned variants.
    """
    words = authoritative.split()
    if len(words) < 2:
        return []
    variants = set()
    # Only the first word keeps its case; the rest get lowercase first letters.
    variants.add(" ".join([words[0]] + [_lower_first_word(w) for w in words[1:]]))
    # Every word lowercased.
    variants.add(" ".join(_lower_first_word(w) for w in words))
    # Fully-lowercase fallback.
    variants.add(" ".join(w.lower() for w in words))
    variants.discard(authoritative)
    return list(variants)


def _build_glossary_rules(skill_doc: str) -> List[tuple]:
    """Extract deterministic (variant -> authoritative) replacement pairs from
    the skill document's glossary tables (section 4).

    Each table row is ``| 中文术语 | English (authoritative) | Notes |``. For
    every English form in the second column we generate variants that the LLM
    tends to output instead of the authoritative spelling:

    - separator variants: hyphen/underscore collapsed to spaces, e.g.
      ``("Ascend platform", "Ascend-platform")``;
    - casing variants: same word sequence with a different letter case, e.g.
      ``("Ascend platform", "Ascend Platform")``.

    ``apply_glossary_rules()`` then deterministically enforces the
    authoritative spellings on every translated string, so the pipeline no
    longer depends on the LLM choosing to follow the glossary (DeepSeek tends
    to normalize "Ascend Platform" back to the natural "Ascend platform").

    Only the section-4 tables are scanned and only plain-English values are
    accepted, so explanatory cells (e.g. section 4.9's "删除 "概述：""...)
    are ignored.
    """
    rules: List[tuple] = []
    if not skill_doc:
        return rules
    in_section4 = False
    for line in skill_doc.splitlines():
        stripped = line.strip()
        if stripped.startswith("## 4."):
            in_section4 = True
            continue
        if in_section4 and stripped.startswith("## "):
            break
        if not in_section4:
            continue
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) < 2:
            continue
        en_cell = cells[1]
        if not en_cell or set(en_cell) <= {"-", " ", ":"}:
            continue
        for authoritative in en_cell.split("/"):
            authoritative = authoritative.strip().strip("`").strip()
            if not _looks_like_english_term(authoritative):
                continue
            # Separator variants: hyphen/underscore -> spaces.
            normalized = re.sub(r"[-_]", " ", authoritative)
            normalized = re.sub(r"\s+", " ", normalized).strip()
            if normalized and normalized != authoritative:
                rules.append((normalized, authoritative))
            # Casing variants: same words, different letter case.
            for variant in _casing_variants(authoritative):
                rules.append((variant, authoritative))
    # Longer variants first so e.g. "Ascend platform" wins over any shorter
    # overlapping variant.
    rules.sort(key=lambda r: len(r[0]), reverse=True)
    return rules


def apply_glossary_rules(text: str, rules: List[tuple]) -> str:
    """Replace glossary variants in ``text`` with their authoritative spellings.

    Inline code spans and fenced code blocks are left untouched so identifiers
    and string literals are never rewritten.
    """
    if not text or not rules:
        return text
    protected: List[str] = []

    def _protect(m):
        protected.append(m.group(0))
        return "\x00%d\x00" % (len(protected) - 1)

    text = re.sub(r"```.*?```", _protect, text, flags=re.DOTALL)
    text = re.sub(r"`[^`\n]+`", _protect, text)
    for variant, authoritative in rules:
        text = re.sub(r"(?<![\w])" + re.escape(variant) + r"(?![\w])", authoritative, text)
    return re.sub(r"\x00(\d+)\x00", lambda m: protected[int(m.group(1))], text)


def _get_current_skill_version() -> str:
    """Read the ``version`` field from the skill doc front-matter."""
    try:
        skill = load_skill_doc()
    except Exception:
        return ""
    if not skill:
        return ""
    m = re.search(r"^version:\s*([0-9A-Za-z.+-]+)", skill, re.MULTILINE)
    return m.group(1).strip() if m else ""


def _get_po_skill_version(po_path: Path) -> str:
    """Read the X-Skill-Version header field recorded in a .po file."""
    try:
        raw = po_path.read_text(encoding="utf-8")
    except OSError:
        return ""
    m = re.search(r"X-Skill-Version:\s*([0-9A-Za-z.+-]+)", raw)
    return m.group(1) if m else ""


# ---------------------------------------------------------------------------
# Translation engine
# ---------------------------------------------------------------------------


class PoTranslator:
    """Translate .pot entries to .po using LLM API with translation memory."""

    def __init__(self, api_key: str, skill_doc: Optional[str] = None, api_base: str = "", model: str = ""):
        self.api_base = api_base or DEFAULT_API_BASE
        self.model = model or DEFAULT_MODEL
        # Per-request timeout and retry budget (env-tunable). Without a timeout
        # a hung request (slow gateway, stalled connection) would block the
        # whole sequential translation for minutes per block. When no API key
        # is provided the client stays None so cache-only re-processing works
        # without network access; blocks needing fresh translation will fail
        # closed with a clear message.
        self.client = (AsyncOpenAI(
            api_key=api_key,
            base_url=self.api_base,
            timeout=float(os.getenv("LLM_TIMEOUT", "120")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "1")),
        ) if api_key else None)
        # Number of blocks translated concurrently per document. Raising this
        # shortens total runtime but may hit provider rate limits (429).
        self._concurrency = max(1, int(os.getenv("LLM_CONCURRENCY", "4")))
        # Extra per-block retries for transient failures (timeout / 429 / net
        # jitter). A block is only considered failed after all attempts.
        self._block_retries = max(0, int(os.getenv("LLM_BLOCK_RETRIES", "2")))
        # The skill document text (.agents/skills/translate/skill.md)
        # is injected as part of the system prompt so every translation request
        # follows the project's authoritative terminology glossary and rules.
        self.skill_doc = (skill_doc or "").strip()
        # Deterministic glossary enforcement: (variant, authoritative) pairs
        # parsed from the skill doc's section-4 tables. Applied to every
        # translation before it is written to the .po file, so the exact
        # spellings defined in the glossary (e.g. "Ascend-platform") are always
        # honoured even if the LLM outputs the natural-space variant
        # ("Ascend platform").
        self._glossary_rules: List[tuple] = _build_glossary_rules(self.skill_doc)
        m = re.search(r"^version:\s*([0-9A-Za-z.+-]+)", self.skill_doc, re.MULTILINE)
        self._skill_version = m.group(1).strip() if m else ""

    def _system_prompt(self, context: str = "") -> str:
        """Build the system prompt, injecting the skill document when available."""
        parts = [SYSTEM_PROMPT]
        if self.skill_doc:
            parts.append("Follow the skill document below for the mandatory translation "
                         "standard (Google Developer Documentation Style Guide key points), "
                         "terminology, and structure rules. The standard MUST be read and "
                         "followed before translating anything; its glossary is "
                         "authoritative. Use it for every translation.\n\n"
                         "===== BEGIN TRANSLATION SKILL DOCUMENT =====\n"
                         f"{self.skill_doc}\n"
                         "===== END TRANSLATION SKILL DOCUMENT =====")
        if context:
            parts.append(f"(File: {context})")
        return "\n\n".join(parts)

    async def translate_file(self, pot_path: Path) -> bool:
        """Translate a single .pot file, producing/updating the matching .po file.

        The .po file path mirrors the docs/zh/ source structure under
        locale/en/LC_MESSAGES/. For example:
            docs/zh/debug_guide/index.md
            → pot_path = locale/zh/LC_MESSAGES/debug_guide/index.pot
            → po_path  = locale/en/LC_MESSAGES/debug_guide/index.po
        """
        # Determine po_path: same relative path under en/LC_MESSAGES/ as pot under zh/LC_MESSAGES/
        rel = pot_path.relative_to(POT_DIR)
        po_path = PO_DIR / rel.with_suffix('.po')
        po_path.parent.mkdir(parents=True, exist_ok=True)

        # Record the content fingerprint of the source Chinese document so
        # future incremental runs can skip this file when the source hasn't
        # changed. Documents without a source (e.g. sphinx.po) are skipped.
        source_md = _find_source_for_pot(pot_path)
        if source_md is None:
            print(f"  Skip: {pot_path.name} (no source document)", flush=True)
            return False
        source_commit = _get_source_commit(source_md)

        pot_entries = parse_pot_file(pot_path)
        po_entries = parse_pot_file(po_path)

        if not pot_entries:
            print(f"  Skip: {pot_path.name} (empty .pot)")
            return False

        name = pot_path.stem

        new_entries = {}
        untranslated: List[str] = []
        kept = 0
        # Tracks whether any msgstr actually changed in this run (a new
        # translation succeeded, or a stored translation was repaired by
        # _restore_enumeration_prefix). Only then is the .po file rewritten
        # with fresh POT/PO timestamps; unchanged files are left untouched so
        # incremental PRs don't carry timestamp-only diffs.
        content_changed = False

        for msgid, entry in pot_entries.items():
            existing = po_entries.get(msgid)
            if existing and existing.get("translated"):
                # Self-heal: DeepSeek may have dropped the "N. " enumeration
                # prefix from a Chinese list heading. Re-attach it here so the
                # repaired translation is persisted (find_changed_pot_files
                # detects such files and routes them through this function).
                restored = _restore_enumeration_prefix(msgid, existing.get("msgstr", ""))
                # Glossary enforcement: deterministically fix stored
                # translations too, so terminology edits in skill.md propagate
                # to already-translated entries without re-calling the API.
                restored = apply_glossary_rules(restored, self._glossary_rules)
                if restored != existing.get("msgstr", ""):
                    content_changed = True
                new_entries[msgid] = {
                    "msgid": msgid,
                    "msgstr": restored,
                    "translated": True,
                }
                kept += 1
            else:
                # Skip API calls for entries that contain no Chinese characters
                # (pure ASCII / pure numbers / English identifiers). These are
                # typically table cell numbers, code identifiers, or already-
                # English text extracted by sphinx-build. Passing them to the
                # LLM wastes tokens and risks the model returning a meta
                # response ("no Chinese text to translate") instead of the
                # original string.
                if not re.search(r'[\u4e00-\u9fff]', msgid):
                    new_entries[msgid] = {
                        "msgid": msgid,
                        "msgstr": msgid,
                        "translated": True,
                    }
                    content_changed = True
                    kept += 1
                    continue
                new_entries[msgid] = {
                    "msgid": msgid,
                    "msgstr": "",
                    "translated": False,
                }
                untranslated.append(msgid)

        print(f"  {name}: {len(pot_entries)} entries [{kept} kept, {len(untranslated)} new]", end=" ", flush=True)

        # Translate the pending entries concurrently (bounded by _concurrency)
        # and collect results. Fail-closed: if any entry fails (API error or
        # auth failure), the .po file is NOT written so the workflow never
        # commits a document containing untranslated Chinese text.
        failed = 0
        auth_failed = False

        if untranslated:
            print(f"\n    Translating {len(untranslated)} entry(ies)...", end=" ", flush=True)

            sem = asyncio.Semaphore(self._concurrency)

            async def translate_one(msgid: str) -> tuple:
                async with sem:
                    translation = None
                    for attempt in range(self._block_retries + 1):
                        try:
                            translation = await self._translate_single(msgid, name)
                        except AuthenticationError:
                            return msgid, None, "AUTH"
                        if translation is not None:
                            break
                        if attempt < self._block_retries:
                            await asyncio.sleep(1.0 * (attempt + 1))
                    if translation is None:
                        return msgid, None, None
                    translation = _restore_enumeration_prefix(msgid, translation)
                    translation = apply_glossary_rules(translation, self._glossary_rules)
                    return msgid, translation, "OK"

            results = await asyncio.gather(*(translate_one(msgid) for msgid in untranslated))

            for msgid, translation, status in results:
                if status == "AUTH":
                    auth_failed = True
                elif status is None:
                    failed += 1
                else:
                    new_entries[msgid]["msgstr"] = translation
                    new_entries[msgid]["translated"] = True
                    content_changed = True

            print(" done", end=" ", flush=True)

        if auth_failed:
            print(f"\n  AUTH FAIL: {name} (invalid API key) - .po NOT written", flush=True)
            return False

        if failed > 0:
            print(f"\n  FAIL: {name}: {failed} entry(ies) failed to translate - .po NOT written", flush=True)
            return False

        if not content_changed:
            # Content identical. Still stamp (or refresh) the X-Source-Commit
            # fingerprint when it differs from the current one: after switching
            # from the commit-hash to the blob-id algorithm, or for .po files
            # that predate the field, one stable-header rewrite is needed so the
            # next incremental run can skip this document entirely. No POT/PO
            # timestamps are touched (changed=False keeps them unchanged).
            # Likewise, stamp X-Skill-Version once so a freshly-upgraded skill
            # doc does not re-trigger processing on every subsequent run.
            needs_stamp = source_commit and source_commit != _get_po_source_commit(po_path)
            needs_skill = self._skill_version and self._skill_version != _get_po_skill_version(po_path)
            if needs_stamp or needs_skill:
                write_po_file(po_path, new_entries, str(pot_path), changed=False, source_commit=source_commit,
                              skill_version=self._skill_version)
                print("OK (header refreshed)", flush=True)
                return True
            # Nothing to do: keep the file untouched.
            print("No content change, skip rewriting", flush=True)
            return False

        write_po_file(po_path, new_entries, str(pot_path), changed=True, source_commit=source_commit,
                      skill_version=self._skill_version)
        print("OK")
        return True

    async def _translate_single(self, content: str, context: str = "") -> Optional[str]:
        """Translate a single text string via the configured LLM API."""
        if self.client is None:
            print(f"  No API key configured - block for '{context}' needs fresh translation, skipped", flush=True)
            return None
        prompt = BLOCK_TRANSLATION_PROMPT.replace("{content}", content)
        system = self._system_prompt(context)

        trailing_nl = content.endswith("\n")
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=8192,
                temperature=0.3,
            )
            text = response.choices[0].message.content
            if not text:
                return None
            text = text.strip()
            if trailing_nl and not text.endswith("\n"):
                text += "\n"
            return text
        except AuthenticationError:
            raise
        except Exception as e:
            print(f"API error translating '{context}': {e}")
            return None

    async def translate_files(self, pot_list: list[Path], output_json: str) -> int:
        """Translate a list of .pot files sequentially and save results JSON."""
        print(f"Translating {len(pot_list)} .pot file(s)", flush=True)

        ok_files = []
        success_files = []
        for pot_path in pot_list:
            ok = await self.translate_file(pot_path)
            if ok:
                rel = pot_path.relative_to(POT_DIR)
                po_path = PO_DIR / rel.with_suffix('.po')
                ok_files.append(str(pot_path))
                success_files.append(str(po_path))

        total = len(pot_list)
        ok_count = len(ok_files)
        print(f"\nResult: {ok_count}/{total} translated", flush=True)

        failed_files = [str(p) for p in pot_list if str(p) not in ok_files]
        if failed_files:
            print(f"FAILED ({len(failed_files)} document(s)):", flush=True)
            for f in failed_files:
                print(f"  - {f}", flush=True)

        report = {
            "success_files": success_files,
            "failed_files": failed_files,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_files": total,
            "success_count": ok_count,
            "failed_count": len(failed_files),
        }
        out = Path(output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Results written to {output_json}", flush=True)

        return 0 if ok_count > 0 else 1


# ---------------------------------------------------------------------------
# POT discovery and organization
# ---------------------------------------------------------------------------


def _organize_pot_files() -> int:
    """Move .pot files from locale/ to locale/zh/LC_MESSAGES/.

    After sphinx-build -b gettext, .pot files are output directly under locale/
    preserving the source structure (e.g. locale/debug_guide/index.pot).

    This function moves them to locale/zh/LC_MESSAGES/<same-rel-path>.pot
    (the standard gettext layout expected by Sphinx when reading .po files).

    Files from excluded directories or with excluded filenames are
    discarded (deleted) so they never enter the translation pipeline.
    """
    count = 0
    discarded = 0
    for pot_file in sorted(LOCALE_DIR.rglob("*.pot")):
        try:
            rel = pot_file.relative_to(LOCALE_DIR)
        except ValueError:
            continue
        # Skip files already under LC_MESSAGES/ (from a previous run)
        if "LC_MESSAGES" in rel.parts:
            continue
        # Discard .pot files from excluded directories or filenames
        if _is_excluded_pot(pot_file):
            pot_file.unlink()
            discarded += 1
            continue
        # Move to locale/zh/LC_MESSAGES/<rel_path>, mirroring source structure
        target = POT_DIR / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        pot_file.replace(target)
        count += 1
    if discarded > 0:
        print(f"  Discarded {discarded} .pot file(s) from excluded dirs/files", flush=True)
    return count


def find_all_pot_files() -> list[Path]:
    """Find all .pot files in locale/zh/LC_MESSAGES/, excluding excluded dirs."""
    result = []
    for pot_file in sorted(POT_DIR.rglob("*.pot")):
        if not _is_excluded_pot(pot_file):
            result.append(pot_file)
    return result


def find_changed_pot_files() -> list[Path]:
    """Find .pot files that need (re)translation vs their .po counterpart.

    Two guards can skip a file entirely:

    1. Source-content guard (primary): each .po records the blob id
       (X-Source-Commit) of the Chinese source .md it was generated from. If
       that fingerprint equals the source's current blob id, the document
       content is unchanged -> skip without any msgid comparison or rewrite.

    2. msgid fallback: for .po files that predate the fingerprint field, fall
       back to comparing msgids; also catch missing/empty translations and
       enumeration-prefix self-heal.
    """
    changed = []
    for pot_file in sorted(POT_DIR.rglob("*.pot")):
        if _is_excluded_pot(pot_file):
            continue

        rel = pot_file.relative_to(POT_DIR)
        po_file = PO_DIR / rel.with_suffix('.po')

        # Source-content guard: unchanged source AND unchanged skill version
        # => skip this document entirely. Generated .po files with no source
        # document (e.g. sphinx.po) are also skipped so they never enter the
        # translation pipeline.
        source_md = _find_source_for_pot(pot_file)
        if source_md is None:
            continue
        cur_commit = _get_source_commit(source_md)
        po_commit = _get_po_source_commit(po_file)
        cur_skill = _get_current_skill_version()
        po_skill = _get_po_skill_version(po_file)
        if (cur_commit and po_commit and cur_commit == po_commit and cur_skill and po_skill and cur_skill == po_skill):
            continue

        pot_entries = parse_pot_file(pot_file)
        po_entries = parse_pot_file(po_file)

        # Skill version changed (or not yet recorded): force re-processing of
        # the whole file so every kept translation is re-checked against the
        # new glossary (deterministic terminology enforcement), not just the
        # untranslated entries.
        if cur_skill and (not po_skill or cur_skill != po_skill):
            changed.append(pot_file)
            continue

        for msgid, entry in pot_entries.items():
            existing = po_entries.get(msgid)
            if not existing or not existing.get("translated"):
                changed.append(pot_file)
                break
            # Self-heal: DeepSeek sometimes drops the "N. " prefix when
            # translating Chinese list headings. If the stored translation
            # lost the msgid's enumeration prefix, re-process this file so
            # _restore_enumeration_prefix can repair it without re-translating.
            if _restore_enumeration_prefix(msgid, existing.get("msgstr", "")) != existing.get("msgstr", ""):
                changed.append(pot_file)
                break

    return changed


def run_sphinx_gettext() -> bool:
    """Run sphinx-build -b gettext to generate/update .pot files.

    We override locale_dirs to '.' so the gettext builder outputs .pot
    files directly into the output directory (locale/) rather than
    being confused by the parent-relative '../locale/' path in conf.py.
    Without this override, the gettext builder would generate 0 files.

    After sphinx-build, this function reorganizes the .pot files into
    locale/zh/LC_MESSAGES/ for the rest of the workflow. Excluded
    directories/files are discarded during reorganization.
    """
    print(f"Running: sphinx-build -b gettext docs/zh {LOCALE_DIR}/", flush=True)
    result = subprocess.run(
        [
            sys.executable, "-m", "sphinx", "-b", "gettext", "-q", "-D", "locale_dirs=['.']",
            str(ZH_DIR),
            str(LOCALE_DIR)
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"sphinx-build FAILED (exit {result.returncode})", flush=True)
        if result.stderr:
            print(f"  stderr: {result.stderr.strip()}", flush=True)
        if result.stdout:
            print(f"  stdout: {result.stdout.strip()}", flush=True)
        return False
    if result.stderr:
        print(f"  stderr: {result.stderr.strip()}", flush=True)

    # Reorganize .pot files into locale/zh/LC_MESSAGES/, discarding excluded ones
    moved = _organize_pot_files()
    if moved == 0:
        print(f"  No .pot files found in {LOCALE_DIR} after build!", flush=True)

    # Verify .pot files were created
    pot_count = len(list(POT_DIR.rglob("*.pot")))
    print(f"  Generated {moved} .pot file(s), {pot_count} in {POT_DIR}", flush=True)
    return pot_count > 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def write_empty_json(output_json: str, reason: str = ""):
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
    print(f"Empty result written to {output_json} (reason: {reason})", flush=True)


async def async_main():
    parser = argparse.ArgumentParser(description="Sphinx gettext translation: docs/zh/ -> locale/en/LC_MESSAGES/")
    parser.add_argument("--first-time", action="store_true",
                        help="Generate .pot files via sphinx-build, then translate ALL")
    parser.add_argument("--all", action="store_true", help="Translate only changed .pot files (incremental)")
    parser.add_argument("--skip-gettext", action="store_true", help="Skip sphinx-build gettext step")
    parser.add_argument("--files", help="Comma-separated .pot filenames")
    parser.add_argument("--output-json", default=os.getenv("OUTPUT_JSON", "/tmp/translation_results.json"))
    parser.add_argument(
        "--api-key",
        default=os.getenv("TRANSLATION_ASCEND", os.getenv("LLM_API_KEY", "")),
        help="LLM API key (env: TRANSLATION_ASCEND, fallback LLM_API_KEY)",
    )
    parser.add_argument(
        "--api-base",
        default=os.getenv("LLM_API_BASE", ""),
        help=("OpenAI-compatible API base URL "
              f"(default: {DEFAULT_API_BASE}, e.g. Zhipu https://open.bigmodel.cn/api/paas/v4)"),
    )
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", ""),
                        help=f"Model name (default: {DEFAULT_MODEL}, e.g. Zhipu glm-4-plus)")
    parser.add_argument(
        "--skill-doc",
        default=os.getenv("TRANSLATION_SKILL_DOC", ""),
        help="Path to the translation skill document (Markdown). "
        "Defaults to .agents/skills/translate/skill.md.",
    )
    args = parser.parse_args()

    output_json = args.output_json

    api_key = (args.api_key or os.getenv("TRANSLATION_ASCEND") or os.getenv("LLM_API_KEY"))
    if not api_key:
        print(
            "Warning: no LLM API key set (TRANSLATION_ASCEND / LLM_API_KEY). "
            "Documents whose entries are fully cached will still be re-processed; entries that "
            "need a fresh translation will fail (fail-closed).", flush=True)

    # Load the translation skill document (authoritative terminology glossary
    # and rules). It is injected into every translation request's system prompt.
    skill_doc_path = Path(args.skill_doc) if args.skill_doc else None
    skill_doc = load_skill_doc(skill_doc_path)

    api_base = args.api_base or DEFAULT_API_BASE
    model = args.model or DEFAULT_MODEL
    print(f"LLM provider: {api_base} | model: {model}", flush=True)

    # Step 1: Generate .pot files (unless --skip-gettext)
    if not args.skip_gettext:
        if not run_sphinx_gettext():
            write_empty_json(output_json, "sphinx-build -b gettext failed")
            return 1

    # Step 2: Determine which .pot files to translate
    pot_list = []
    if args.files:
        raw_files = [f.strip() for f in args.files.split(",") if f.strip()]
        for f in raw_files:
            p = Path(f)
            if not p.exists():
                p = POT_DIR / f
            if p.exists() and p.suffix == ".pot" and not _is_excluded_pot(p):
                pot_list.append(p)
    elif args.first_time:
        pot_list = find_all_pot_files()
    elif args.all:
        pot_list = find_changed_pot_files()
    else:
        msg = "specify --first-time, --all, or --files"
        print(f"Error: {msg}", flush=True)
        write_empty_json(output_json, msg)
        return 1

    if pot_list:
        print(f"Found {len(pot_list)} .pot file(s) to translate", flush=True)
        for p in pot_list:
            print(f"  - {p}", flush=True)
    else:
        # No files to translate — write empty result so workflow doesn't get stuck
        reason = "excluded" if args.first_time else "no changes"
        print(f"No .pot files to translate ({reason})", flush=True)
        write_empty_json(output_json, f"no .pot files to translate ({reason})")
        return 0

    translator = PoTranslator(api_key=api_key, skill_doc=skill_doc, api_base=args.api_base, model=args.model)
    return await translator.translate_files(pot_list, output_json)


if __name__ == "__main__":
    sys.exit(asyncio.run(async_main()))
