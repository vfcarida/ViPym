"""Minimal Agentic Scaffolding for SWE-bench Problem Solving.

Supports two problem-solving strategies:
1. "single_shot": Model directly produces a unified diff patch given issue & repo summary.
2. "iterative": Multi-turn exploration agent with tool commands:
   - <view_file path="..." start="..." end="...">
   - <list_files dir="...">
   - <search_dir query="..." dir="...">
   - <submit_patch>...</submit_patch>
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from vipym.core.logger import get_logger
from vipym.interfaces.evaluation import BenchmarkTask
from vipym.interfaces.inference import GenerationRequest, InferenceBackend

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Config & Data Models
# ---------------------------------------------------------------------------


@dataclass
class SWEBenchAgentConfig:
    """Configuration for SWE-bench agentic solver."""

    strategy: Literal["iterative", "single_shot"] = "iterative"
    max_turns: int = 5
    context_window: int = 32000
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 2048
    system_prompt: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SWEBenchAgentConfig:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class AgentAction:
    """Action requested by model."""

    action_type: str  # "view_file", "list_files", "search_dir", "submit_patch", "unknown"
    argument: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentObservation:
    """Environment response to model action."""

    output: str
    success: bool = True


@dataclass
class AgentTurn:
    """Record of a single conversation turn in the agent loop."""

    turn_index: int
    thought: str
    action: AgentAction | None
    observation: AgentObservation | None
    raw_response: str


@dataclass
class AgentResult:
    """Result of agent execution on a benchmark task."""

    patch: str
    turns: list[AgentTurn] = field(default_factory=list)
    total_tokens: int = 0
    completed: bool = True
    error: str | None = None


# ---------------------------------------------------------------------------
# Default Prompts
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM_PROMPT = """You are an expert software engineer resolving GitHub issues in open-source Python repositories.
Your goal is to inspect the problem, understand the codebase, and produce a clean unified diff patch that fixes the issue.

You have access to the following tools:
1. <view_file path="path/to/file" start="1" end="50"/>: View lines of a file.
2. <list_files dir="path/to/dir"/>: List directory contents.
3. <search_dir query="term" dir="path/to/dir"/>: Search codebase for terms or function names.
4. <submit_patch>
diff --git a/... b/...
...
</submit_patch>: Submit the final unified diff patch to resolve the issue.

Instructions:
- Examine the issue description carefully.
- Use tools to locate the relevant files and bug location.
- When ready, submit your fix in standard unified diff format (`diff --git a/... b/...`).
"""

_SINGLE_SHOT_SYSTEM_PROMPT = """You are an expert software engineer resolving GitHub issues.
Analyze the following problem statement and repository context, then output a standard unified diff patch (`diff --git a/... b/...`) resolving the issue.
Wrap your patch in a ```diff ... ``` code block.
"""


# ---------------------------------------------------------------------------
# SWEBenchAgent
# ---------------------------------------------------------------------------


class SWEBenchAgent:
    """Agent that interacts with an LLM backend to produce code patches for SWE-bench instances."""

    def __init__(self, config: SWEBenchAgentConfig | None = None) -> None:
        self.config = config or SWEBenchAgentConfig()

    def solve(
        self,
        task: BenchmarkTask,
        backend: InferenceBackend | Any,
        repo_files: dict[str, str] | Path | None = None,
    ) -> AgentResult:
        """Run agent to solve the given SWE-bench task."""
        if self.config.strategy == "single_shot":
            return self._solve_single_shot(task, backend)
        return self._solve_iterative(task, backend, repo_files)

    # ------------------------------------------------------------------
    # Single-shot Mode
    # ------------------------------------------------------------------

    def _solve_single_shot(
        self,
        task: BenchmarkTask,
        backend: InferenceBackend | Any,
    ) -> AgentResult:
        prompt = self._format_single_shot_prompt(task)
        response_text, token_count = self._call_backend(backend, prompt)

        patch = self.extract_patch(response_text)
        turn = AgentTurn(
            turn_index=0,
            thought="Single-shot patch generation",
            action=AgentAction(action_type="submit_patch", argument=patch),
            observation=AgentObservation(output="Patch generated"),
            raw_response=response_text,
        )
        return AgentResult(
            patch=patch,
            turns=[turn],
            total_tokens=token_count,
            completed=bool(patch.strip()),
        )

    def _format_single_shot_prompt(self, task: BenchmarkTask) -> str:
        sys_p = self.config.system_prompt or _SINGLE_SHOT_SYSTEM_PROMPT
        repo = task.metadata.get("repo", "Repository")
        base_commit = task.metadata.get("base_commit", "HEAD")
        hints = task.metadata.get("hints_text", "")
        hint_str = f"\nHints:\n{hints}\n" if hints else ""

        return (
            f"{sys_p}\n\n"
            f"=== Repository: {repo} (base commit: {base_commit}) ===\n"
            f"=== Problem Statement ===\n"
            f"{task.prompt}\n"
            f"{hint_str}\n"
            f"Provide your unified diff patch below:\n"
        )

    # ------------------------------------------------------------------
    # Iterative Mode
    # ------------------------------------------------------------------

    def _solve_iterative(
        self,
        task: BenchmarkTask,
        backend: InferenceBackend | Any,
        repo_files: dict[str, str] | Path | None = None,
    ) -> AgentResult:
        conversation: list[dict[str, str]] = []
        turns: list[AgentTurn] = []
        total_tokens = 0
        final_patch = ""

        system_msg = self.config.system_prompt or _DEFAULT_SYSTEM_PROMPT
        init_user_msg = self._format_initial_user_prompt(task)

        conversation.append({"role": "system", "content": system_msg})
        conversation.append({"role": "user", "content": init_user_msg})

        for turn_idx in range(self.config.max_turns):
            prompt = self._build_prompt_from_conversation(conversation)
            response_text, token_count = self._call_backend(backend, prompt)
            total_tokens += token_count

            thought, action = self._parse_action(response_text)

            if action is None or action.action_type == "submit_patch":
                # Check if patch is embedded in submit_patch or in raw response
                patch_cand = action.argument if (action and action.argument) else self.extract_patch(response_text)
                if patch_cand.strip():
                    final_patch = patch_cand
                elif turn_idx == self.config.max_turns - 1:
                    final_patch = self.extract_patch(response_text)

                turns.append(
                    AgentTurn(
                        turn_index=turn_idx,
                        thought=thought,
                        action=action,
                        observation=AgentObservation(output="Patch submitted" if final_patch else "No patch found"),
                        raw_response=response_text,
                    )
                )
                if final_patch.strip() or turn_idx == self.config.max_turns - 1:
                    break

            # Execute tool action
            observation = self._execute_action(action, repo_files, task)
            turns.append(
                AgentTurn(
                    turn_index=turn_idx,
                    thought=thought,
                    action=action,
                    observation=observation,
                    raw_response=response_text,
                )
            )

            # Update conversation history
            conversation.append({"role": "assistant", "content": response_text})
            conversation.append({"role": "user", "content": f"<observation>\n{observation.output}\n</observation>"})

            # Check context window
            conversation = self._truncate_conversation(conversation)

        if not final_patch.strip() and turns:
            # Fallback: attempt to parse patch from any turn
            for t in reversed(turns):
                cand = self.extract_patch(t.raw_response)
                if cand.strip():
                    final_patch = cand
                    break

        return AgentResult(
            patch=final_patch,
            turns=turns,
            total_tokens=total_tokens,
            completed=bool(final_patch.strip()),
        )

    def _format_initial_user_prompt(self, task: BenchmarkTask) -> str:
        repo = task.metadata.get("repo", "Repository")
        base_commit = task.metadata.get("base_commit", "HEAD")
        hints = task.metadata.get("hints_text", "")
        hint_str = f"\nHints:\n{hints}\n" if hints else ""

        return (
            f"=== Repository: {repo} (base commit: {base_commit}) ===\n"
            f"=== Issue Statement ===\n"
            f"{task.prompt}\n"
            f"{hint_str}\n"
            f"Please investigate the bug using available tools and submit your fix."
        )

    def _build_prompt_from_conversation(self, conversation: list[dict[str, str]]) -> str:
        lines: list[str] = []
        for msg in conversation:
            role = msg["role"].upper()
            content = msg["content"]
            lines.append(f"[{role}]\n{content}\n")
        return "\n".join(lines)

    def _truncate_conversation(self, conversation: list[dict[str, str]]) -> list[dict[str, str]]:
        """Keep system and first user message, trim oldest middle turns if exceeding context threshold."""
        total_chars = sum(len(m["content"]) for m in conversation)
        # Rough estimate: 4 chars per token
        max_chars = self.config.context_window * 4
        if total_chars <= max_chars or len(conversation) <= 3:
            return conversation

        # Retain index 0 (system) and index 1 (problem statement)
        header = conversation[:2]
        tail = conversation[2:]
        while sum(len(m["content"]) for m in header + tail) > max_chars and len(tail) > 2:
            tail.pop(0)  # remove oldest assistant/user interaction

        return header + tail

    # ------------------------------------------------------------------
    # Tool Parser & Execution
    # ------------------------------------------------------------------

    def _parse_action(self, text: str) -> tuple[str, AgentAction | None]:
        """Extract thought and structured tool action from model output."""
        # 1. Check for <submit_patch>...</submit_patch>
        patch_match = re.search(r"<submit_patch>(.*?)</submit_patch>", text, re.DOTALL | re.IGNORECASE)
        if patch_match:
            thought = text[: patch_match.start()].strip()
            patch_content = patch_match.group(1).strip()
            return thought, AgentAction(action_type="submit_patch", argument=patch_content)

        # 2. Check for <view_file path="..." start="..." end="..."/>
        view_match = re.search(
            r'<view_file\s+path=["\']([^"\']+)["\'](?:\s+start=["\']?(\d+)["\']?)?(?:\s+end=["\']?(\d+)["\']?)?\s*/?>',
            text,
            re.IGNORECASE,
        )
        if view_match:
            thought = text[: view_match.start()].strip()
            path = view_match.group(1)
            start = int(view_match.group(2)) if view_match.group(2) else None
            end = int(view_match.group(3)) if view_match.group(3) else None
            return thought, AgentAction(
                action_type="view_file",
                argument=path,
                params={"path": path, "start": start, "end": end},
            )

        # 3. Check for <list_files dir="..."/>
        list_match = re.search(r'<list_files(?:\s+dir=["\']([^"\']*)["\'])?\s*/?>', text, re.IGNORECASE)
        if list_match:
            thought = text[: list_match.start()].strip()
            dir_path = list_match.group(1) or "."
            return thought, AgentAction(action_type="list_files", argument=dir_path, params={"dir": dir_path})

        # 4. Check for <search_dir query="..." dir="..."/>
        search_match = re.search(
            r'<search_dir\s+query=["\']([^"\']+)["\'](?:\s+dir=["\']([^"\']*)["\'])?\s*/?>',
            text,
            re.IGNORECASE,
        )
        if search_match:
            thought = text[: search_match.start()].strip()
            query = search_match.group(1)
            dir_path = search_match.group(2) or "."
            return thought, AgentAction(
                action_type="search_dir",
                argument=query,
                params={"query": query, "dir": dir_path},
            )

        # 5. Check if raw unified diff is present directly in response
        if "diff --git" in text or "--- a/" in text:
            patch_text = self.extract_patch(text)
            if patch_text.strip():
                return text, AgentAction(action_type="submit_patch", argument=patch_text)

        return text.strip(), None

    def _execute_action(
        self,
        action: AgentAction | None,
        repo_files: dict[str, str] | Path | None,
        task: BenchmarkTask,
    ) -> AgentObservation:
        if action is None:
            return AgentObservation(
                output="No recognizable tool call was found. Use <view_file>, <list_files>, <search_dir>, or <submit_patch>.",
                success=False,
            )

        if action.action_type == "view_file":
            path = action.params.get("path") or action.argument
            start = action.params.get("start")
            end = action.params.get("end")
            return self._tool_view_file(path, start, end, repo_files, task)

        if action.action_type == "list_files":
            dir_path = action.params.get("dir") or action.argument or "."
            return self._tool_list_files(dir_path, repo_files, task)

        if action.action_type == "search_dir":
            query = action.params.get("query") or action.argument
            dir_path = action.params.get("dir") or "."
            return self._tool_search_dir(query, dir_path, repo_files, task)

        if action.action_type == "submit_patch":
            return AgentObservation(output="Patch submitted.", success=True)

        return AgentObservation(output=f"Unknown tool action: {action.action_type}", success=False)

    # ------------------------------------------------------------------
    # Simulated / Local Tool Implementations
    # ------------------------------------------------------------------

    def _tool_view_file(
        self,
        path: str,
        start: int | None,
        end: int | None,
        repo_files: dict[str, str] | Path | None,
        task: BenchmarkTask,
    ) -> AgentObservation:
        content: str | None = None

        if isinstance(repo_files, dict):
            # Normalise path
            clean_path = path.lstrip("./").lstrip("/")
            for k, v in repo_files.items():
                if k.lstrip("./").lstrip("/") == clean_path or k.endswith(clean_path):
                    content = v
                    break
        elif isinstance(repo_files, Path) and repo_files.is_dir():
            target = repo_files / path.lstrip("/")
            if target.exists() and target.is_file():
                try:
                    content = target.read_text(encoding="utf-8", errors="replace")
                except Exception as e:
                    return AgentObservation(output=f"Error reading file {path}: {e}", success=False)

        if content is None:
            # Check if task metadata provides mock file contents
            mock_files = task.metadata.get("files", {})
            if isinstance(mock_files, dict) and path in mock_files:
                content = mock_files[path]

        if content is None:
            return AgentObservation(output=f"File not found: {path}", success=False)

        lines = content.splitlines()
        s = max(1, start or 1) - 1
        e = min(len(lines), end or len(lines))
        selected_lines = [f"{i+1}: {line}" for i, line in enumerate(lines[s:e], start=s)]
        return AgentObservation(output="\n".join(selected_lines), success=True)

    def _tool_list_files(
        self,
        dir_path: str,
        repo_files: dict[str, str] | Path | None,
        task: BenchmarkTask,
    ) -> AgentObservation:
        clean_dir = dir_path.lstrip("./").lstrip("/")
        results: list[str] = []

        if isinstance(repo_files, dict):
            for k in repo_files:
                k_clean = k.lstrip("./").lstrip("/")
                if not clean_dir or k_clean.startswith(clean_dir):
                    results.append(k_clean)
        elif isinstance(repo_files, Path) and repo_files.is_dir():
            target_dir = repo_files / clean_dir if clean_dir else repo_files
            if target_dir.exists() and target_dir.is_dir():
                for p in target_dir.rglob("*"):
                    if p.is_file():
                        results.append(str(p.relative_to(repo_files)).replace("\\", "/"))

        if not results:
            mock_files = task.metadata.get("files", {})
            if isinstance(mock_files, dict):
                results = list(mock_files.keys())

        if not results:
            return AgentObservation(output=f"No files found in directory: {dir_path}")

        return AgentObservation(output="\n".join(sorted(results[:100])), success=True)

    def _tool_search_dir(
        self,
        query: str,
        dir_path: str,
        repo_files: dict[str, str] | Path | None,
        task: BenchmarkTask,
    ) -> AgentObservation:
        matches: list[str] = []
        files_dict: dict[str, str] = {}

        if isinstance(repo_files, dict):
            files_dict = repo_files
        elif isinstance(repo_files, Path) and repo_files.is_dir():
            for p in repo_files.rglob("*.py"):
                try:
                    rel = str(p.relative_to(repo_files)).replace("\\", "/")
                    files_dict[rel] = p.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
        else:
            files_dict = task.metadata.get("files", {})

        for file_path, content in files_dict.items():
            for i, line in enumerate(content.splitlines(), start=1):
                if query.lower() in line.lower():
                    matches.append(f"{file_path}:{i}: {line.strip()}")
                    if len(matches) >= 30:
                        break
            if len(matches) >= 30:
                break

        if not matches:
            return AgentObservation(output=f"No matches found for query '{query}' in {dir_path}")

        return AgentObservation(output="\n".join(matches), success=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _call_backend(self, backend: Any, prompt: str) -> tuple[str, int]:
        """Send prompt to backend and return (generated_text, token_count)."""
        if hasattr(backend, "generate"):
            req = GenerationRequest(
                prompt=prompt,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
            resp = backend.generate(req)
            tokens = getattr(resp, "completion_tokens", len(resp.generated_text) // 4)
            return resp.generated_text, tokens

        if callable(backend):
            out = backend(prompt)
            if isinstance(out, str):
                return out, len(out) // 4
            if hasattr(out, "generated_text"):
                return out.generated_text, getattr(out, "completion_tokens", len(out.generated_text) // 4)

        return "", 0

    @staticmethod
    def extract_patch(text: str) -> str:
        """Extract standard unified diff patch from markdown fences or raw diff text."""
        if not text:
            return ""

        # 1. Try markdown code block with diff tag
        diff_block_match = re.search(r"```(?:diff|patch)?\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if diff_block_match:
            candidate = diff_block_match.group(1).strip()
            if "diff --git" in candidate or "--- a/" in candidate or "@@" in candidate:
                return candidate

        # 2. Try <submit_patch> tags
        xml_match = re.search(r"<submit_patch>(.*?)</submit_patch>", text, re.DOTALL | re.IGNORECASE)
        if xml_match:
            return xml_match.group(1).strip()

        # 3. Find start of unified diff header in raw text
        diff_start = text.find("diff --git")
        if diff_start != -1:
            return text[diff_start:].strip()

        alt_start = text.find("--- a/")
        if alt_start != -1:
            return text[alt_start:].strip()

        return ""
