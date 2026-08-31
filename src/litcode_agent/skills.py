"""Agent Skills discovery with strict, workspace-local boundaries."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

SKILL_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True, slots=True)
class SkillMetadata:
    name: str
    description: str


@dataclass(frozen=True, slots=True)
class Skill:
    name: str
    description: str
    content: str
    root: Path

    def resource_paths(self, limit: int = 10) -> tuple[str, ...]:
        result: list[str] = []
        for path in sorted(self.root.rglob("*")):
            if len(result) >= limit:
                break
            if path.name == "SKILL.md" or path.is_symlink() or not path.is_file():
                continue
            resolved = path.resolve(strict=True)
            if resolved.is_relative_to(self.root):
                result.append(path.relative_to(self.root).as_posix())
        return tuple(result)


class SkillCatalog:
    """A validated catalog; invalid entries remain visible as issues."""

    def __init__(self, skills: dict[str, Skill], issues: tuple[str, ...]) -> None:
        self._skills = skills
        self.issues = issues

    @classmethod
    def discover(cls, workspace: Path) -> SkillCatalog:
        root = workspace.resolve() / ".agents" / "skills"
        if not root.is_dir():
            return cls({}, ())
        skills: dict[str, Skill] = {}
        issues: list[str] = []
        for directory in sorted(root.iterdir(), key=lambda item: item.name):
            source = directory / "SKILL.md"
            try:
                if directory.is_symlink() or source.is_symlink():
                    raise ValueError("Skill 目录或 SKILL.md 不得是符号链接")
                if not directory.is_dir() or not source.is_file():
                    continue
                resolved = source.resolve(strict=True)
                if not resolved.is_relative_to(root):
                    raise ValueError("Skill 路径逃逸工作区目录")
                skill = _read_skill(source, directory.name)
                if skill.name in skills:
                    raise ValueError(f"Skill 名称重复：{skill.name}")
                skills[skill.name] = skill
            except (OSError, UnicodeDecodeError, ValueError, yaml.YAMLError) as error:
                issues.append(f"{directory.name}: {error}")
        return cls(skills, tuple(issues))

    def metadata(self) -> tuple[SkillMetadata, ...]:
        return tuple(
            SkillMetadata(skill.name, skill.description)
            for skill in sorted(self._skills.values(), key=lambda item: item.name)
        )

    def load(self, name: str) -> Skill:
        try:
            return self._skills[name]
        except KeyError as error:
            available = "、".join(sorted(self._skills)) or "无"
            raise ValueError(f"找不到 Skill：{name}；可用：{available}") from error


def _read_skill(source: Path, directory_name: str) -> Skill:
    raw = source.read_text(encoding="utf-8")
    if not raw.startswith("---\n"):
        raise ValueError("SKILL.md 缺少 YAML frontmatter")
    marker = raw.find("\n---", 4)
    if marker == -1:
        raise ValueError("SKILL.md frontmatter 未闭合")
    closing_end = marker + 4
    if closing_end < len(raw) and raw[closing_end] not in {"\n", "\r"}:
        raise ValueError("SKILL.md frontmatter 结束标记无效")
    data = yaml.safe_load(raw[4:marker])
    if not isinstance(data, dict):
        raise ValueError("Skill frontmatter 必须是对象")
    name = data.get("name")
    description = data.get("description")
    if not isinstance(name, str) or not SKILL_NAME.fullmatch(name) or len(name) > 64:
        raise ValueError("Skill name 必须是 1–64 位小写字母、数字或单连字符")
    if name != directory_name:
        raise ValueError(f"Skill name 必须与目录名一致：期望 {directory_name}，实际 {name}")
    if (
        not isinstance(description, str)
        or not description.strip()
        or len(description) > 1024
    ):
        raise ValueError("Skill description 必须是 1–1024 字符的非空文本")
    content = raw[closing_end:].lstrip("\r\n")
    return Skill(name, description.strip(), content, source.parent.resolve())
