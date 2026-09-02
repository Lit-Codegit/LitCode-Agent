"""Create, install, validate, and share standard Agent Skills."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence
from urllib.parse import urlparse

from litcode_agent.skills import (
    COMPATIBLE_PROJECT_SKILL_ROOTS,
    PROJECT_SKILL_ROOT,
    SKILL_NAME,
    Skill,
    read_skill,
)

SkillScope = Literal["project", "user"]

MAX_SKILL_FILES = 1_000
MAX_SKILL_BYTES = 25 * 1024 * 1024


class SkillManagementError(ValueError):
    """A user-facing Skill management failure."""


@dataclass(frozen=True, slots=True)
class ManagedSkill:
    skill: Skill
    scope: SkillScope


@dataclass(frozen=True, slots=True)
class SyncLink:
    agent: str
    skill: str
    destination: Path
    created: bool


AGENT_DIRECTORIES: Mapping[str, tuple[str, str]] = {
    "codex": (".agents/skills", ".codex/skills"),
    "claude-code": (".claude/skills", ".claude/skills"),
    "opencode": (".agents/skills", ".config/opencode/skills"),
    "cursor": (".agents/skills", ".cursor/skills"),
    "gemini-cli": (".agents/skills", ".gemini/skills"),
    "github-copilot": (".agents/skills", ".copilot/skills"),
}


def default_user_skill_root(environ: Mapping[str, str] | None = None) -> Path:
    values = os.environ if environ is None else environ
    home = values.get("HOME")
    return (Path(home).expanduser() if home else Path.home()) / ".agents" / "skills"


class SkillManager:
    def __init__(self, workspace: Path, user_root: Path | None = None) -> None:
        self.workspace = workspace.expanduser().resolve()
        self.user_root = (user_root or default_user_skill_root()).expanduser().resolve()

    def root(self, scope: SkillScope) -> Path:
        if scope == "project":
            return self.workspace / PROJECT_SKILL_ROOT
        if scope == "user":
            return self.user_root
        raise SkillManagementError(f"未知 Skill scope：{scope}")

    def list(self, scope: Literal["all", "project", "user"] = "all") -> tuple[ManagedSkill, ...]:
        roots: list[tuple[SkillScope, Path]] = []
        if scope in {"all", "user"}:
            roots.append(("user", self.user_root))
        if scope in {"all", "project"}:
            roots.extend(
                ("project", self.workspace / relative)
                for relative in (*COMPATIBLE_PROJECT_SKILL_ROOTS, PROJECT_SKILL_ROOT)
            )
        found: dict[str, ManagedSkill] = {}
        for item_scope, root in roots:
            if not root.is_dir():
                continue
            for directory in sorted(root.iterdir(), key=lambda item: item.name):
                source = directory / "SKILL.md"
                if directory.is_symlink() or source.is_symlink() or not source.is_file():
                    continue
                try:
                    skill = read_skill(source, directory.name)
                except (OSError, UnicodeDecodeError, ValueError):
                    continue
                found[skill.name] = ManagedSkill(skill, item_scope)
        return tuple(sorted(found.values(), key=lambda item: item.skill.name))

    def create(
        self,
        name: str,
        description: str,
        *,
        scope: SkillScope = "project",
        resources: Sequence[str] = (),
    ) -> Skill:
        _validate_name(name)
        description = description.strip()
        if not description or len(description) > 1_024:
            raise SkillManagementError("description 必须是 1–1024 字符的非空文本")
        allowed = {"scripts", "references", "assets"}
        unknown = set(resources) - allowed
        if unknown:
            raise SkillManagementError(f"未知资源目录：{'、'.join(sorted(unknown))}")
        destination = self.root(scope) / name
        if destination.exists() or destination.is_symlink():
            raise SkillManagementError(f"Skill 已存在：{destination}")
        destination.mkdir(parents=True)
        source = destination / "SKILL.md"
        source.write_text(
            "---\n"
            f"name: {name}\n"
            f"description: {_yaml_scalar(description)}\n"
            "---\n\n"
            f"# {name}\n\n"
            "描述这个 Skill 应改变 Agent 哪些决策，以及何时停止。\n",
            encoding="utf-8",
        )
        for resource in sorted(set(resources)):
            (destination / resource).mkdir()
        return read_skill(source, name)

    def validate(self, target: str, scope: Literal["all", "project", "user"] = "all") -> Skill:
        candidate = Path(target).expanduser()
        if candidate.exists() or candidate.is_symlink():
            directory = candidate if candidate.is_dir() else candidate.parent
            return _validate_skill_tree(directory)
        matches = [item.skill for item in self.list(scope) if item.skill.name == target]
        if not matches:
            raise SkillManagementError(f"找不到 Skill：{target}")
        return _validate_skill_tree(matches[0].root)

    def install(
        self,
        source: str,
        *,
        name: str | None = None,
        scope: SkillScope = "project",
        timeout_seconds: float = 60.0,
    ) -> Skill:
        local = Path(source).expanduser()
        if local.is_dir():
            selected = local.resolve()
            skill = _validate_skill_tree(selected)
            if name is not None and skill.name != name:
                raise SkillManagementError(
                    f"请求安装 {name}，但目录中的 Skill 名称是 {skill.name}"
                )
            return self._stage_copy(selected, skill, scope)

        remote, direct_path = _normalise_git_source(source)
        requested = name or (Path(direct_path).name if direct_path else None)
        with tempfile.TemporaryDirectory(prefix="litcode-skill-") as temporary:
            checkout = Path(temporary) / "repository"
            _run_git(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--filter=blob:none",
                    remote,
                    str(checkout),
                ],
                timeout_seconds,
            )
            candidates = _skill_directories(checkout)
            if direct_path:
                direct = (checkout / direct_path).resolve()
                if not direct.is_relative_to(checkout.resolve()):
                    raise SkillManagementError("GitHub Skill 路径逃逸仓库")
                candidates = [item for item in candidates if item.resolve() == direct]
            if requested:
                candidates = [item for item in candidates if item.name == requested]
            if not candidates:
                label = requested or direct_path or source
                raise SkillManagementError(f"来源中找不到 Skill：{label}")
            if len(candidates) > 1:
                names = "、".join(sorted(item.name for item in candidates)[:20])
                raise SkillManagementError(f"来源包含多个 Skill，请用 --name 选择：{names}")
            skill = _validate_skill_tree(candidates[0])
            return self._stage_copy(candidates[0], skill, scope)

    def sync(
        self,
        names: Sequence[str] = (),
        *,
        scope: SkillScope = "project",
        agents: Sequence[str] = (),
    ) -> tuple[SyncLink, ...]:
        available = {item.skill.name: item.skill for item in self.list(scope)}
        selected_names = tuple(names) or tuple(sorted(available))
        missing = [name for name in selected_names if name not in available]
        if missing:
            raise SkillManagementError(f"找不到 Skill：{'、'.join(missing)}")
        selected_agents = tuple(agents) or self._detected_agents(scope)
        unknown = set(selected_agents) - set(AGENT_DIRECTORIES)
        if unknown:
            raise SkillManagementError(f"不支持的 Agent：{'、'.join(sorted(unknown))}")
        links: list[SyncLink] = []
        canonical_root = self.root(scope).resolve()
        for agent in selected_agents:
            target_root = self._agent_root(agent, scope)
            if target_root.resolve() == canonical_root:
                continue
            for skill_name in selected_names:
                source = available[skill_name].root.resolve()
                destination = target_root / skill_name
                if destination.is_symlink():
                    if destination.resolve() == source:
                        links.append(SyncLink(agent, skill_name, destination, False))
                        continue
                    raise SkillManagementError(f"拒绝覆盖其他符号链接：{destination}")
                if destination.exists():
                    raise SkillManagementError(f"拒绝覆盖已有路径：{destination}")
                target_root.mkdir(parents=True, exist_ok=True)
                destination.symlink_to(source, target_is_directory=True)
                links.append(SyncLink(agent, skill_name, destination, True))
        return tuple(links)

    def _stage_copy(self, source: Path, skill: Skill, scope: SkillScope) -> Skill:
        destination = self.root(scope) / skill.name
        if destination.exists() or destination.is_symlink():
            raise SkillManagementError(f"Skill 已存在：{destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{skill.name}-", dir=destination.parent))
        try:
            _copy_tree(source, staging)
            installed = _validate_skill_tree(staging, expected_name=skill.name)
            staging.replace(destination)
            return Skill(installed.name, installed.description, installed.content, destination)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def _detected_agents(self, scope: SkillScope) -> tuple[str, ...]:
        detected: list[str] = []
        for agent in AGENT_DIRECTORIES:
            root = self._agent_root(agent, scope)
            marker = root.parent
            if marker.exists():
                detected.append(agent)
        return tuple(detected)

    def _agent_root(self, agent: str, scope: SkillScope) -> Path:
        try:
            project_path, user_path = AGENT_DIRECTORIES[agent]
        except KeyError as error:
            raise SkillManagementError(f"不支持的 Agent：{agent}") from error
        if scope == "project":
            return self.workspace / project_path
        return self.user_root.parents[1] / user_path


def _validate_name(name: str) -> None:
    if not SKILL_NAME.fullmatch(name) or len(name) > 64:
        raise SkillManagementError("Skill name 必须是 1–64 位小写字母、数字或单连字符")


def _yaml_scalar(value: str) -> str:
    return "'" + value.replace("'", "''").replace("\n", " ") + "'"


def _validate_skill_tree(directory: Path, expected_name: str | None = None) -> Skill:
    directory = directory.resolve(strict=True)
    if not directory.is_dir():
        raise SkillManagementError(f"Skill 不是目录：{directory}")
    count = 0
    size = 0
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise SkillManagementError(f"Skill 不得包含符号链接：{path}")
        if not path.is_file():
            continue
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(directory):
            raise SkillManagementError(f"Skill 路径逃逸：{path}")
        count += 1
        size += path.stat().st_size
        if count > MAX_SKILL_FILES:
            raise SkillManagementError(f"Skill 文件数超过上限 {MAX_SKILL_FILES}")
        if size > MAX_SKILL_BYTES:
            raise SkillManagementError(f"Skill 大小超过上限 {MAX_SKILL_BYTES} bytes")
    source = directory / "SKILL.md"
    name = expected_name or directory.name
    try:
        return read_skill(source, name)
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise SkillManagementError(str(error)) from error


def _copy_tree(source: Path, destination: Path) -> None:
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_symlink():
            raise SkillManagementError(f"Skill 不得包含符号链接：{path}")
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def _skill_directories(repository: Path) -> list[Path]:
    return sorted(
        (path.parent for path in repository.rglob("SKILL.md") if not path.is_symlink()),
        key=lambda item: item.as_posix(),
    )


def _normalise_git_source(source: str) -> tuple[str, str | None]:
    if source.startswith("git@") or source.startswith("ssh://"):
        return source, None
    if source.count("/") == 1 and not source.startswith(("/", ".")):
        owner, repository = source.split("/", 1)
        if owner and repository:
            return f"https://github.com/{owner}/{repository.removesuffix('.git')}.git", None
    parsed = urlparse(source)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise SkillManagementError("source 必须是本地目录、Git 仓库或 GitHub URL")
    if parsed.username or parsed.password:
        raise SkillManagementError("URL 不得内嵌凭据；请使用 Git credential helper")
    parts = [part for part in parsed.path.split("/") if part]
    if parsed.netloc.lower() == "github.com" and len(parts) >= 2:
        remote = f"https://github.com/{parts[0]}/{parts[1].removesuffix('.git')}.git"
        if len(parts) >= 5 and parts[2] == "tree":
            return remote, "/".join(parts[4:])
        return remote, None
    return source, None


def _run_git(command: list[str], timeout_seconds: float) -> None:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as error:
        raise SkillManagementError("未找到 git，无法下载远程 Skill") from error
    except subprocess.TimeoutExpired as error:
        raise SkillManagementError("下载 Skill 超时") from error
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[-2_000:]
        raise SkillManagementError(f"git 下载失败：{detail or completed.returncode}")
