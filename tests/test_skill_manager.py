from pathlib import Path

import pytest

from litcode_agent.skill_manager import SkillManagementError, SkillManager
from litcode_agent.skills import SkillCatalog


def write_skill(root: Path, name: str, description: str = "test skill") -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n",
        encoding="utf-8",
    )
    return directory


def test_create_uses_standard_project_layout_and_optional_resources(tmp_path: Path) -> None:
    manager = SkillManager(tmp_path, tmp_path / "user-skills")

    skill = manager.create(
        "review-code",
        "Review code using this repository's conventions.",
        resources=("references",),
    )

    assert skill.root == (tmp_path / ".litcode" / "skills" / "review-code").resolve()
    assert (skill.root / "references").is_dir()
    assert manager.validate("review-code").name == "review-code"


def test_project_skill_overrides_same_named_user_skill(tmp_path: Path) -> None:
    user_root = tmp_path / "user-skills"
    write_skill(user_root, "shared", "user description")
    write_skill(tmp_path / ".litcode" / "skills", "shared", "project description")

    catalog = SkillCatalog.discover(tmp_path, user_root)
    manager = SkillManager(tmp_path, user_root)

    assert catalog.load("shared").description == "project description"
    assert [(item.skill.name, item.scope) for item in manager.list()] == [
        ("shared", "project")
    ]


def test_native_project_skill_overrides_compatible_layouts(tmp_path: Path) -> None:
    write_skill(tmp_path / ".agent" / "skills", "shared", "singular description")
    write_skill(tmp_path / ".agents" / "skills", "shared", "standard description")
    write_skill(tmp_path / ".litcode" / "skills", "shared", "native description")

    catalog = SkillCatalog.discover(tmp_path)
    manager = SkillManager(tmp_path, tmp_path / "user-skills")

    assert catalog.load("shared").description == "native description"
    assert manager.list("project")[0].skill.description == "native description"


def test_install_from_local_directory_rejects_existing_destination(tmp_path: Path) -> None:
    source = write_skill(tmp_path / "source", "portable")
    manager = SkillManager(tmp_path / "workspace", tmp_path / "user-skills")
    manager.workspace.mkdir()

    installed = manager.install(str(source))

    assert installed.name == "portable"
    assert (installed.root / "SKILL.md").is_file()
    with pytest.raises(SkillManagementError, match="已存在"):
        manager.install(str(source))


def test_validate_rejects_skill_resource_symlink(tmp_path: Path) -> None:
    directory = write_skill(tmp_path, "unsafe")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    (directory / "reference.txt").symlink_to(outside)
    manager = SkillManager(tmp_path, tmp_path / "user-skills")

    with pytest.raises(SkillManagementError, match="符号链接"):
        manager.validate(str(directory))


def test_sync_links_canonical_skill_without_overwriting_existing_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manager = SkillManager(workspace, tmp_path / "home" / ".agents" / "skills")
    skill = manager.create("portable", "Share this skill.")

    links = manager.sync(["portable"], agents=["claude-code"])

    destination = workspace / ".claude" / "skills" / "portable"
    assert destination.is_symlink()
    assert destination.resolve() == skill.root
    assert links[0].created
    assert not manager.sync(["portable"], agents=["claude-code"])[0].created

    destination.unlink()
    destination.mkdir()
    with pytest.raises(SkillManagementError, match="拒绝覆盖"):
        manager.sync(["portable"], agents=["claude-code"])
