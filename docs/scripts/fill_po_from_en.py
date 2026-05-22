#!/usr/bin/env python3
"""
fill_po_from_en.py - Extract English translations from en/*.md files
and fill them into the corresponding locale/en/LC_MESSAGES/*.po files.

Usage:
    python docs/scripts/fill_po_from_en.py                  # Process all PO files
    python docs/scripts/fill_po_from_en.py --dry-run         # Preview without writing
    python docs/scripts/fill_po_from_en.py --verbose         # Detailed output
    python docs/scripts/fill_po_from_en.py --po FAQ.po       # Single PO file

Strategy:
    1. Parse both zh/*.md and en/*.md into paragraphs
    2. Align zh-to-en paragraphs using heading-level matching
       (handles structural differences like extra paragraphs in en)
    3. For each PO entry, find its zh paragraph and map to en via the alignment
    4. Table cell entries (no line numbers) matched by sequential position
    5. Fill msgstr with English text
"""

import re
import sys
import argparse
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DOCS_DIR = SCRIPT_DIR.parent  # docs/
assert (DOCS_DIR / "Makefile").exists(), \
    f"Script must be located at docs/scripts/. Current DOCS_DIR={DOCS_DIR}"

PO_DIR = DOCS_DIR / "locale" / "en" / "LC_MESSAGES"
ZH_DIR = DOCS_DIR / "zh"
EN_DIR = DOCS_DIR / "en"


# ═══════════════════════════════════════════════════════════════════════
# Markdown File Parsing
# ═══════════════════════════════════════════════════════════════════════

def strip_code_and_comments(lines):
    """Annotate each line with (line_num, text, is_code, is_sep).

    - is_code: line is inside a code block (``` fences)
    - is_sep:  line is a structural separator (blank, HTML comment, table sep)
    """
    result = []
    in_code = False

    for i, raw_line in enumerate(lines, 1):
        line = raw_line.rstrip('\n')
        stripped = line.strip()

        if stripped.startswith('```'):
            in_code = not in_code
            result.append((i, line, True, False))
            continue

        if in_code:
            result.append((i, line, True, False))
            continue

        if not stripped:
            result.append((i, line, False, True))
            continue

        if stripped.startswith('<!--'):
            result.append((i, line, False, True))
            continue

        if _is_table_separator(stripped):
            result.append((i, line, False, True))
            continue

        result.append((i, line, False, False))

    return result


def _is_table_separator(stripped):
    """Check if line is a markdown table separator (| --- | --- |)."""
    if not stripped.startswith('|'):
        return False
    body = stripped.replace('|', '').replace('-', '').replace(':', '').replace(' ', '').strip()
    return len(body) == 0 and '--' in stripped


def extract_content_paragraphs(annotated_lines):
    """Group consecutive content lines (non-code, non-blank) into paragraphs.

    Returns list of dicts:
      {index, start_line, end_line, text, text_lines, line_numbers, heading_level}
    """
    content_lines = [(num, text) for num, text, is_code, is_sep in annotated_lines
                     if not is_code and not is_sep]

    if not content_lines:
        return []

    paragraphs = []
    current = [content_lines[0]]

    for item in content_lines[1:]:
        prev_num = current[-1][0]
        curr_num = item[0]
        if curr_num - prev_num > 1:
            paragraphs.append(current)
            current = [item]
        else:
            current.append(item)

    if current:
        paragraphs.append(current)

    result = []
    for idx, para in enumerate(paragraphs):
        line_numbers = [p[0] for p in para]
        text_lines = [p[1] for p in para]
        text = '\n'.join(text_lines)

        # Determine heading level
        first_line = text_lines[0].strip() if text_lines else ''
        h_level = 0
        if first_line.startswith('#'):
            h_level = len(first_line) - len(first_line.lstrip('#'))

        result.append({
            'index': idx,
            'start_line': line_numbers[0],
            'end_line': line_numbers[-1],
            'line_numbers': line_numbers,
            'text': text,
            'text_lines': text_lines,
            'heading_level': h_level,
        })

    return result


def calculate_para_mapping(zh_paragraphs, en_paragraphs):
    """Align zh paragraphs to en paragraphs using heading-level matching.

    Handles structural differences: if en has an extra heading that zh doesn't
    (or vice versa), this function finds the correct alignment by matching
    headings by level and adjusting the offset.

    Returns dict: {zh_para_index: en_para_index}
    """
    mapping = {}
    offset = 0
    max_offset_search = 20

    for zh_idx, zh_para in enumerate(zh_paragraphs):
        en_idx = zh_idx + offset

        # Clamp to valid range
        if en_idx < 0:
            en_idx = 0
        if en_idx >= len(en_paragraphs):
            mapping[zh_idx] = None
            continue

        zh_h = zh_para['heading_level']
        en_h = en_paragraphs[en_idx]['heading_level']

        if zh_h > 0 and en_h > 0:
            # Both are headings
            if zh_h == en_h:
                # Same level: good match
                mapping[zh_idx] = en_idx
            elif zh_h < en_h:
                # zh has higher-level heading, en has lower-level heading
                # en might have an extra sub-heading inserted
                # Search forward for the matching heading
                found = False
                for so in range(1, max_offset_search + 1):
                    cand = en_idx + so
                    if cand >= len(en_paragraphs):
                        break
                    if en_paragraphs[cand]['heading_level'] == zh_h:
                        # Found matching heading level
                        offset = cand - zh_idx
                        mapping[zh_idx] = cand
                        found = True
                        break
                if not found:
                    mapping[zh_idx] = None
            else:
                # zh_h > en_h: zh has lower-level heading, en has higher
                # zh might have an extra sub-heading inserted
                found = False
                for so in range(1, max_offset_search + 1):
                    cand = en_idx - so
                    if cand < 0:
                        break
                    if en_paragraphs[cand]['heading_level'] == zh_h:
                        offset = cand - zh_idx
                        mapping[zh_idx] = cand
                        found = True
                        break
                if not found:
                    mapping[zh_idx] = None
        elif zh_h > 0 and en_h == 0:
            # zh is heading, en is not -> search forward
            found = False
            for so in range(1, max_offset_search + 1):
                cand = en_idx + so
                if cand >= len(en_paragraphs):
                    break
                if en_paragraphs[cand]['heading_level'] > 0:
                    offset = cand - zh_idx
                    mapping[zh_idx] = cand
                    found = True
                    break
            if not found:
                mapping[zh_idx] = None
        elif zh_h == 0 and en_h > 0:
            # zh is not heading, en is heading -> zh is missing this heading
            # This means offset should increase
            offset += 1
            # Re-try with new offset
            en_idx = zh_idx + offset
            if en_idx < len(en_paragraphs):
                mapping[zh_idx] = en_idx
            else:
                mapping[zh_idx] = None
        else:
            # Both non-headings: should be aligned
            if en_idx < len(en_paragraphs):
                mapping[zh_idx] = en_idx
            else:
                mapping[zh_idx] = None

    return mapping


def extract_table_cells(annotated_lines):
    """Extract all table cell texts in document order.

    Returns list of (line_number, cell_text).
    """
    cells = []
    for num, line_text, is_code, is_sep in annotated_lines:
        if is_code or is_sep:
            continue
        stripped = line_text.strip()
        if stripped.startswith('|') and stripped.endswith('|'):
            parts = stripped.split('|')
            parts = [p.strip() for p in parts[1:-1]]
            for cell in parts:
                if cell:
                    cells.append((num, cell))
    return cells


# ═══════════════════════════════════════════════════════════════════════
# PO File Handling
# ═══════════════════════════════════════════════════════════════════════

def find_md_files(po_path):
    """Determine zh/en file paths from a PO file's location.

    locale/en/LC_MESSAGES/FAQ.po -> zh/FAQ.md, en/FAQ.md
    locale/en/LC_MESSAGES/examples/index.po -> zh/examples/index.md, en/examples/index.md

    Returns (rel_path_str, zh_path, en_path) or (None, None, None).
    """
    try:
        rel = po_path.relative_to(PO_DIR)
    except ValueError:
        return None, None, None

    # Try .md
    rel_md = rel.with_suffix('.md')
    zh_path = ZH_DIR / rel_md
    en_path = EN_DIR / rel_md

    if not zh_path.exists():
        # Try .rst
        rel_rst = rel.with_suffix('.rst')
        zh_rst = ZH_DIR / rel_rst
        if zh_rst.exists():
            zh_path = zh_rst
            en_path = EN_DIR / rel_rst
        else:
            return None, None, None

    return str(rel_md), zh_path, en_path


def strip_markdown_heading(text):
    """Strip leading # markers from heading text for PO msgstr.

    gettext extracts rendered text (without markdown syntax), so msgstr
    must also be plain text.  '## Quick Start' → 'Quick Start'.
    """
    lines = text.split('\n')
    if lines and lines[0].strip().startswith('#'):
        stripped = lines[0].strip()
        h_level = len(stripped) - len(stripped.lstrip('#'))
        if h_level > 0 and h_level <= 6 and stripped[h_level] == ' ':
            lines[0] = stripped[h_level + 1:]
    return '\n'.join(lines)


def format_po_string(text):
    """Format a string for PO msgstr/msgid, handling escaping and wrapping.

    Returns the content after msgstr  (e.g. '"text"' or '""\\n"line1"\\n"line2"').
    """
    if not text:
        return '""'

    escaped = text.replace('\\', '\\\\').replace('"', '\\"')

    if len(escaped) <= 72 and '\n' not in escaped:
        return f'"{escaped}"'

    # Multi-line
    parts = escaped.split('\n') if '\n' in escaped else [escaped]

    lines = []
    for part in parts:
        while len(part) > 72:
            idx = part.rfind(' ', 0, 73)
            if idx < 40:
                idx = 72
            lines.append(part[:idx])
            part = part[idx:].strip()
        if part:
            lines.append(part)

    result = '""'
    for line in lines:
        result += '\n' + f'"{line}"'

    return result


def process_po_file(po_path, dry_run=False, verbose=False):
    """Process a single PO file: fill msgstr with English translations.

    Returns dict with stats.
    """
    stats = {
        'file': po_path.name,
        'total_entries': 0,
        'translated_para': 0,
        'translated_cell': 0,
        'skipped_no_match': 0,
        'skipped_no_en': 0,
    }

    rel_path, zh_path, en_path = find_md_files(po_path)
    if rel_path is None:
        stats['skipped_no_en'] = 1
        return stats

    if not zh_path.exists():
        stats['skipped_no_en'] = 1
        return stats

    if not en_path.exists():
        stats['skipped_no_en'] = 1
        return stats

    if verbose:
        print(f"\n  Processing: {rel_path}")

    # ── Parse zh and en ────────────────────────────────────────────
    zh_text = zh_path.read_text(encoding='utf-8')
    en_text = en_path.read_text(encoding='utf-8')

    zh_annotated = strip_code_and_comments(zh_text.split('\n'))
    en_annotated = strip_code_and_comments(en_text.split('\n'))

    zh_paragraphs = extract_content_paragraphs(zh_annotated)
    en_paragraphs = extract_content_paragraphs(en_annotated)

    # Build zh line → paragraph mapping
    zh_line_to_para = {}
    for para in zh_paragraphs:
        for ln in range(para['start_line'], para['end_line'] + 1):
            zh_line_to_para[ln] = para['index']

    # Calculate paragraph alignment (zh → en)
    para_mapping = calculate_para_mapping(zh_paragraphs, en_paragraphs)

    # Extract table cells
    zh_cells = extract_table_cells(zh_annotated)
    en_cells = extract_table_cells(en_annotated)

    if verbose:
        print(f"       zh: {len(zh_paragraphs)} paragraphs, {len(zh_cells)} cells")
        print(f"       en: {len(en_paragraphs)} paragraphs, {len(en_cells)} cells")

        # Report alignment
        mismatches = sum(1 for k, v in para_mapping.items()
                         if v is not None and k != v)
        if mismatches:
            print(f"       alignment: {mismatches} zh paras shifted")
        none_count = sum(1 for v in para_mapping.values() if v is None)
        if none_count:
            print(f"       alignment: {none_count} zh paras unmatched")

    # ── Read PO file and find entries ──────────────────────────────
    po_text = po_path.read_text(encoding='utf-8')

    # Find all #: entries
    entry_starts = list(re.finditer(r'^#:\s+', po_text, re.MULTILINE))

    entry_spans = []
    for i, match in enumerate(entry_starts):
        start = match.start()
        if i + 1 < len(entry_starts):
            end = entry_starts[i + 1].start()
        else:
            end = len(po_text)
        entry_spans.append((start, end))

    stats['total_entries'] = len(entry_spans)

    if verbose:
        print(f"       po: {len(entry_spans)} entries")

    # ── Collect replacements ───────────────────────────────────────
    replacements = []  # (start_pos, end_pos, new_text)
    cell_idx = 0

    for start, end in entry_spans:
        entry_text = po_text[start:end]

        # Parse the location reference. The PO header (before the first '#:'
        # line) is already excluded by entry_starts, so all remaining entries
        # are real content entries with '#: path:line' references.
        loc_match = re.match(r'#:\s+(\S+?)(?::(\d+))?\s*\n', entry_text)
        if not loc_match:
            continue

        zh_line_str = loc_match.group(2)

        # Find msgstr position in entry
        msgstr_match = re.search(
            r'msgstr\s+((?:"(?:[^"\\]|\\.)*"\s*\n?)+)',
            entry_text
        )
        if not msgstr_match:
            continue

        captured = msgstr_match.group(1)
        # Strip trailing whitespace/newlines so they stay in the file
        # after replacement (fixes entries being concatenated)
        stripped_captured = captured.rstrip('\n\r ')
        msgstr_abs_start = start + msgstr_match.start(1)
        msgstr_abs_end = msgstr_abs_start + len(stripped_captured)

        en_translation = None

        if zh_line_str is not None:
            # ── Entry WITH line number: paragraph matching ──
            zh_line = int(zh_line_str)

            if zh_line in zh_line_to_para:
                zh_para_idx = zh_line_to_para[zh_line]
                en_para_idx = para_mapping.get(zh_para_idx)

                if en_para_idx is not None and en_para_idx < len(en_paragraphs):
                    en_para = en_paragraphs[en_para_idx]
                    en_text_val = en_para['text']
                    if en_text_val and en_text_val.strip():
                        en_translation = en_text_val
                    else:
                        stats['skipped_no_match'] += 1
                else:
                    stats['skipped_no_match'] += 1
            else:
                stats['skipped_no_match'] += 1

            if en_translation is not None:
                en_translation = strip_markdown_heading(en_translation)
                replacements.append((
                    msgstr_abs_start, msgstr_abs_end,
                    format_po_string(en_translation)
                ))
                stats['translated_para'] += 1
            else:
                # Keep empty msgstr (fallback to Chinese)
                pass

        else:
            # ── Entry WITHOUT line number: table cell matching ──
            if cell_idx < len(en_cells) and cell_idx < len(zh_cells):
                en_cell_text = en_cells[cell_idx][1]
                replacements.append((
                    msgstr_abs_start, msgstr_abs_end,
                    format_po_string(en_cell_text)
                ))
                stats['translated_cell'] += 1
                cell_idx += 1
            else:
                stats['skipped_no_match'] += 1

    # ── Apply replacements (reverse order for position safety) ─────
    replacements.sort(key=lambda x: x[0], reverse=True)

    for r_start, r_end, new_text in replacements:
        po_text = po_text[:r_start] + new_text + po_text[r_end:]

    # ── Write ──────────────────────────────────────────────────────
    if not dry_run and replacements:
        po_path.write_text(po_text, encoding='utf-8')
        if verbose:
            print(f"       -> Wrote {len(replacements)} translations")
    elif replacements:
        if verbose:
            print(f"       -> Would write {len(replacements)} translations (dry-run)")
    else:
        if verbose:
            print(f"       -> No translations")

    return stats


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Fill PO files with English translations from en/*.md',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview changes without writing')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed processing info')
    parser.add_argument('--po-file', '-p', type=str,
                        help='Process a single PO file (name, e.g. FAQ.po)')

    args = parser.parse_args()

    if args.po_file:
        po_files = [PO_DIR / args.po_file]
        if not po_files[0].exists():
            po_files = list(PO_DIR.rglob(args.po_file))
        if not po_files:
            print(f"Error: '{args.po_file}' not found under {PO_DIR}")
            sys.exit(1)
    else:
        po_files = sorted(PO_DIR.rglob('*.po'))

    total = {
        'files': 0,
        'total_entries': 0,
        'translated_para': 0,
        'translated_cell': 0,
        'skipped_no_match': 0,
        'skipped_no_en': 0,
        'files_with_translations': 0,
    }

    print(f"Processing {len(po_files)} PO files...")
    if args.dry_run:
        print("(DRY RUN - no files will be modified)")
    print()

    for po_path in po_files:
        rel = po_path.relative_to(PO_DIR)
        stats = process_po_file(po_path, dry_run=args.dry_run, verbose=args.verbose)

        if not args.verbose:
            if stats['skipped_no_en']:
                status = "SKIP (no en)"
            elif stats['translated_para'] + stats['translated_cell'] == 0:
                status = "??"
            else:
                status = f"OK {stats['translated_para']}p+{stats['translated_cell']}c"
            print(f"  {rel}: {status}")

        total['files'] += 1
        total['total_entries'] += stats['total_entries']
        total['translated_para'] += stats['translated_para']
        total['translated_cell'] += stats['translated_cell']
        total['skipped_no_match'] += stats['skipped_no_match']
        total['skipped_no_en'] += stats['skipped_no_en']
        if stats['translated_para'] + stats['translated_cell'] > 0:
            total['files_with_translations'] += 1

    print()
    print("=" * 60)
    print("Summary:")
    print(f"  Total PO files:          {total['files']}")
    print(f"  Files with translations: {total['files_with_translations']}")
    print(f"  Total entries:           {total['total_entries']}")
    print(f"  Paragraph translations:  {total['translated_para']}")
    print(f"  Table cell translations: {total['translated_cell']}")
    print(f"  Skipped (no en file):    {total['skipped_no_en']}")
    print(f"  Skipped (no match):      {total['skipped_no_match']}")

    untranslated = total['total_entries'] - total['translated_para'] - total['translated_cell'] - total['skipped_no_en'] - total['skipped_no_match']
    if untranslated > 0:
        print(f"  Untranslated (kept empty): {untranslated}")

    if args.dry_run:
        print("\n(Dry run completed - no files modified)")
        print("Run without --dry-run to apply changes.")


if __name__ == '__main__':
    main()
