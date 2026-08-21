"""Edit format parsers, validators, and applicators for code editing benchmarks.

Supports:
1. Search/Replace blocks (Aider standard):
   <<<<<<< SEARCH
   old code
   =======
   new code
   >>>>>>> REPLACE

2. Unified Diffs (udiff / diff):
   --- a/file.py
   +++ b/file.py
   @@ -1,3 +1,3 @@
   -old
   +new

3. Whole file replacement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


@dataclass
class EditBlock:
    """Individual SEARCH / REPLACE editing block."""

    search_content: str
    replace_content: str
    file_path: str | None = None


@dataclass
class EditApplyResult:
    """Outcome of applying an edit or diff to original source code."""

    success: bool
    modified_code: str
    error: str | None = None
    format_type: str = "unknown"
    blocks_applied: int = 0
    total_blocks: int = 0


# ---------------------------------------------------------------------------
# Search / Replace Parsers & Applicators
# ---------------------------------------------------------------------------

_SEARCH_REPLACE_PATTERN = re.compile(
    r"(?:(?:\#|//|--)\s*(?:file:?\s*)?([^\n\r]+\.[a-zA-Z0-9]+)\s*\n)?"
    r"<<<<<<<\s*SEARCH\r?\n"
    r"(.*?)"
    r"=======\r?\n"
    r"(.*?)"
    r">>>>>>>\s*REPLACE",
    re.DOTALL,
)


def parse_search_replace_blocks(text: str) -> list[EditBlock]:
    """Extract all SEARCH / REPLACE blocks from text."""
    blocks: list[EditBlock] = []
    if not text:
        return blocks

    # Strip markdown wrapper fences if the whole text is wrapped
    clean_text = text
    fence_match = re.search(r"```[a-zA-Z0-9_-]*\r?\n(.*?)```", text, re.DOTALL)
    if fence_match and "<<<<<<< SEARCH" in fence_match.group(1):
        clean_text = fence_match.group(1)

    for match in _SEARCH_REPLACE_PATTERN.finditer(clean_text):
        file_path = match.group(1).strip() if match.group(1) else None
        search_part = match.group(2)
        replace_part = match.group(3)
        blocks.append(
            EditBlock(
                search_content=search_part,
                replace_content=replace_part,
                file_path=file_path,
            )
        )

    return blocks


def apply_search_replace(
    original_code: str,
    blocks: list[EditBlock],
    fuzzy: bool = True,
) -> EditApplyResult:
    """Apply SEARCH / REPLACE blocks sequentially to the original code."""
    if not blocks:
        return EditApplyResult(
            success=False,
            modified_code=original_code,
            error="No valid <<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE blocks found.",
            format_type="search_replace",
            blocks_applied=0,
            total_blocks=0,
        )

    current_code = original_code
    applied_count = 0

    for idx, block in enumerate(blocks, start=1):
        search_target = block.search_content

        # Case 1: Empty search block -> prepend or replace entire content
        if not search_target:
            if not current_code.strip():
                current_code = block.replace_content
                applied_count += 1
                continue

        # Case 2: Exact string match
        if search_target in current_code:
            # Replace only the first occurrence
            current_code = current_code.replace(search_target, block.replace_content, 1)
            applied_count += 1
            continue

        # Case 3: Fuzzy / whitespace-normalized line-by-line match
        if fuzzy:
            fuzzy_success, updated_code = _apply_fuzzy_search_replace(
                current_code, search_target, block.replace_content
            )
            if fuzzy_success:
                current_code = updated_code
                applied_count += 1
                continue

        return EditApplyResult(
            success=False,
            modified_code=original_code,
            error=f"Block {idx}/{len(blocks)} SEARCH text not found in source code:\n{search_target[:100]}...",
            format_type="search_replace",
            blocks_applied=applied_count,
            total_blocks=len(blocks),
        )

    return EditApplyResult(
        success=True,
        modified_code=current_code,
        format_type="search_replace",
        blocks_applied=applied_count,
        total_blocks=len(blocks),
    )


def _apply_fuzzy_search_replace(
    code: str,
    search_target: str,
    replace_content: str,
) -> tuple[bool, str]:
    """Attempt whitespace-insensitive and line-trimmed search and replacement."""
    search_lines = [line.strip() for line in search_target.splitlines() if line.strip()]
    if not search_lines:
        return False, code

    code_lines = code.splitlines(keepends=True)
    n_search = len(search_lines)

    for i in range(len(code_lines) - n_search + 1):
        window = [code_lines[i + j].strip() for j in range(n_search)]
        if window == search_lines:
            # Match found at lines [i : i + n_search]
            # Replace this slice with replace_content
            prefix = "".join(code_lines[:i])
            suffix = "".join(code_lines[i + n_search :])
            rep_text = replace_content
            if not rep_text.endswith("\n") and suffix:
                rep_text += "\n"
            return True, prefix + rep_text + suffix

    return False, code


# ---------------------------------------------------------------------------
# Unified Diff Applicator
# ---------------------------------------------------------------------------


def apply_unified_diff(original_code: str, diff_text: str) -> EditApplyResult:
    """Apply a unified diff patch to the given source code."""
    if not diff_text.strip():
        return EditApplyResult(
            success=False,
            modified_code=original_code,
            error="Empty diff text",
            format_type="diff",
        )

    # Strip code block markdown wrappers
    clean_diff = diff_text
    diff_block_match = re.search(r"```(?:diff|patch)?\r?\n(.*?)```", diff_text, re.DOTALL)
    if diff_block_match:
        clean_diff = diff_block_match.group(1)

    orig_lines = original_code.splitlines(keepends=True)
    diff_lines = clean_diff.splitlines()

    # Find hunks
    hunk_indices = [
        i
        for i, line in enumerate(diff_lines)
        if re.match(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@", line)
    ]

    if not hunk_indices:
        # Check if it's a simple search/replace or whole file
        return EditApplyResult(
            success=False,
            modified_code=original_code,
            error="No unified diff hunk headers (@@ -... +... @@) found",
            format_type="diff",
        )

    out_lines: list[str] = []
    curr_orig_idx = 0

    for h_num, h_idx in enumerate(hunk_indices):
        header = diff_lines[h_idx]
        m = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", header)
        if not m:
            continue
        orig_start = max(1, int(m.group(1))) - 1  # 0-indexed

        # Next hunk index or end of diff
        next_h_idx = hunk_indices[h_num + 1] if h_num + 1 < len(hunk_indices) else len(diff_lines)
        hunk_body = diff_lines[h_idx + 1 : next_h_idx]

        # Add lines up to orig_start
        while curr_orig_idx < min(orig_start, len(orig_lines)):
            out_lines.append(orig_lines[curr_orig_idx])
            curr_orig_idx += 1

        for line in hunk_body:
            if not line:
                if curr_orig_idx < len(orig_lines):
                    out_lines.append(orig_lines[curr_orig_idx])
                    curr_orig_idx += 1
                continue

            marker = line[0]
            content = line[1:]

            if marker == " ":
                if curr_orig_idx < len(orig_lines):
                    out_lines.append(orig_lines[curr_orig_idx])
                    curr_orig_idx += 1
                else:
                    out_lines.append(content + "\n")
            elif marker == "-":
                curr_orig_idx += 1
            elif marker == "+":
                out_lines.append(content + "\n")

    # Add remaining trailing lines
    while curr_orig_idx < len(orig_lines):
        out_lines.append(orig_lines[curr_orig_idx])
        curr_orig_idx += 1

    result_code = "".join(out_lines)
    return EditApplyResult(
        success=True,
        modified_code=result_code,
        format_type="diff",
        blocks_applied=len(hunk_indices),
        total_blocks=len(hunk_indices),
    )


# ---------------------------------------------------------------------------
# Whole File Applicator
# ---------------------------------------------------------------------------


def apply_whole_file(original_code: str, new_code: str) -> EditApplyResult:
    """Extract code from markdown codeblock or treat entire string as new code."""
    if not new_code.strip():
        return EditApplyResult(
            success=False,
            modified_code=original_code,
            error="Empty code content provided",
            format_type="whole_file",
        )

    # If wrapped in ```python ... ``` or similar, extract inside
    block_match = re.search(r"```[a-zA-Z0-9_-]*\r?\n(.*?)```", new_code, re.DOTALL)
    if block_match:
        extracted = block_match.group(1)
        return EditApplyResult(
            success=True,
            modified_code=extracted,
            format_type="whole_file",
            blocks_applied=1,
            total_blocks=1,
        )

    return EditApplyResult(
        success=True,
        modified_code=new_code,
        format_type="whole_file",
        blocks_applied=1,
        total_blocks=1,
    )


# ---------------------------------------------------------------------------
# Unified High-Level Applicator
# ---------------------------------------------------------------------------


def apply_edit(
    original_code: str,
    edit_text: str,
    expected_format: Literal["search_replace", "diff", "udiff", "whole_file", "auto"] = "auto",
) -> EditApplyResult:
    """Apply an edit to original source code according to the expected format."""
    if not edit_text.strip():
        return EditApplyResult(
            success=False,
            modified_code=original_code,
            error="Empty edit text",
            format_type=expected_format,
        )

    fmt = expected_format.lower()

    if fmt == "search_replace":
        blocks = parse_search_replace_blocks(edit_text)
        return apply_search_replace(original_code, blocks)

    if fmt in ("diff", "udiff"):
        return apply_unified_diff(original_code, edit_text)

    if fmt == "whole_file":
        return apply_whole_file(original_code, edit_text)

    # Auto-detection
    if "<<<<<<< SEARCH" in edit_text and "=======" in edit_text and ">>>>>>> REPLACE" in edit_text:
        blocks = parse_search_replace_blocks(edit_text)
        return apply_search_replace(original_code, blocks)

    if "@@ -" in edit_text or "diff --git" in edit_text or "--- a/" in edit_text:
        return apply_unified_diff(original_code, edit_text)

    return apply_whole_file(original_code, edit_text)


# ---------------------------------------------------------------------------
# Format Compliance Validator
# ---------------------------------------------------------------------------


def validate_format_compliance(
    edit_text: str,
    expected_format: Literal["search_replace", "diff", "udiff", "whole_file", "auto"],
) -> tuple[bool, str]:
    """Check if model response adhered strictly to the expected format."""
    if not edit_text.strip():
        return False, "Response is empty"

    fmt = expected_format.lower()

    if fmt == "search_replace":
        has_search = "<<<<<<< SEARCH" in edit_text
        has_div = "=======" in edit_text
        has_replace = ">>>>>>> REPLACE" in edit_text
        if has_search and has_div and has_replace:
            return True, "Valid search_replace format"
        return False, "Missing <<<<<<< SEARCH, =======, or >>>>>>> REPLACE markers"

    if fmt in ("diff", "udiff"):
        has_hunk = bool(re.search(r"@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@", edit_text))
        has_header = "diff --git" in edit_text or "--- a/" in edit_text
        if has_hunk or has_header:
            return True, "Valid unified diff format"
        return False, "Missing @@ hunk headers or diff file markers"

    if fmt == "whole_file":
        # Whole file is compliant if non-empty and not just markdown commentary
        return True, "Valid whole_file format"

    # Auto format is compliant if any recognizable structure is found
    return True, "Compliant"
