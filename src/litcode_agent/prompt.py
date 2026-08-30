"""Explainable system-prompt composition."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path


BASE_BEHAVIOR = """You are LitCode Agent, a careful coding assistant.
Inspect relevant files before editing. Make the smallest changes that solve the
task, verify them, and report evidence. Treat tool errors as feedback. Never
claim a command succeeded unless its result says so."""


@dataclass(frozen=True, slots=True)
class PromptSection:
    name: str
    content: str
    source: str


class PromptBuilder:
    def __init__(self, workspace: Path, max_iterations: int) -> None:
        self.workspace = workspace.resolve()
        self.max_iterations = max_iterations

    def sections(self) -> tuple[PromptSection, ...]:
        result = [PromptSection("base_behavior", BASE_BEHAVIOR, "builtin")]
        result.append(
            PromptSection(
                "environment",
                f"Workspace: {self.workspace}\nPlatform: {platform.system()}",
                "runtime",
            )
        )
        instructions = self.workspace / "AGENTS.md"
        if instructions.is_file():
            result.append(
                PromptSection(
                    "project_instructions",
                    instructions.read_text(encoding="utf-8"),
                    str(instructions),
                )
            )
        result.append(
            PromptSection(
                "runtime_limits",
                f"Stop after at most {self.max_iterations} model iterations.",
                "settings",
            )
        )
        return tuple(result)

    def build(self) -> str:
        return "\n\n".join(
            f"## {section.name}\n{section.content}" for section in self.sections()
        )
