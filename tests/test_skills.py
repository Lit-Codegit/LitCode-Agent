from pathlib import Path

from litcode_agent.prompt import PromptBuilder
from litcode_agent.skills import SkillCatalog
from litcode_agent.tools.skills import LoadSkillTool


def write_skill(root: Path, name: str, body: str = "Follow the workflow.") -> None:
    directory = root / ".agents" / "skills" / name
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Use for {name} tasks.\n---\n\n{body}\n",
        encoding="utf-8",
    )


def test_prompt_discloses_only_skill_metadata(tmp_path: Path) -> None:
    write_skill(tmp_path, "release-notes", "SECRET WORKFLOW BODY")
    catalog = SkillCatalog.discover(tmp_path)

    prompt = PromptBuilder(tmp_path, 10, catalog.metadata()).build()

    assert "release-notes" in prompt
    assert "Use for release-notes tasks." in prompt
    assert "SECRET WORKFLOW BODY" not in prompt


def test_load_skill_returns_body_and_resource_names_on_demand(tmp_path: Path) -> None:
    write_skill(tmp_path, "release-notes", "Read references/policy.md only if needed.")
    skill_root = tmp_path / ".agents" / "skills" / "release-notes"
    (skill_root / "references").mkdir()
    (skill_root / "references" / "policy.md").write_text(
        "RESOURCE CONTENT MUST STAY LAZY", encoding="utf-8"
    )
    catalog = SkillCatalog.discover(tmp_path)

    result = LoadSkillTool(catalog, 10_000).execute({"name": "release-notes"})

    assert not result.is_error
    assert "Read references/policy.md only if needed." in result.content
    assert "references/policy.md" in result.content
    assert "RESOURCE CONTENT MUST STAY LAZY" not in result.content


def test_invalid_skill_is_reported_without_hiding_valid_skills(tmp_path: Path) -> None:
    write_skill(tmp_path, "valid-skill")
    invalid = tmp_path / ".agents" / "skills" / "Bad_Name"
    invalid.mkdir(parents=True)
    (invalid / "SKILL.md").write_text(
        "---\nname: other-name\ndescription: broken\n---\nbody",
        encoding="utf-8",
    )

    catalog = SkillCatalog.discover(tmp_path)

    assert [item.name for item in catalog.metadata()] == ["valid-skill"]
    assert len(catalog.issues) == 1
    assert "Bad_Name" in catalog.issues[0]


def test_skill_discovery_does_not_follow_symlinks(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside-skill"
    write_skill(outside, "escaped")
    skills = tmp_path / ".agents" / "skills"
    skills.mkdir(parents=True)
    (skills / "escaped").symlink_to(
        outside / ".agents" / "skills" / "escaped", target_is_directory=True
    )

    catalog = SkillCatalog.discover(tmp_path)

    assert catalog.metadata() == ()
    assert any("符号链接" in issue for issue in catalog.issues)


def test_skill_metadata_is_escaped_in_system_catalog(tmp_path: Path) -> None:
    directory = tmp_path / ".agents" / "skills" / "safe-name"
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        "---\nname: safe-name\ndescription: 'Use <carefully> & explicitly.'\n---\nbody",
        encoding="utf-8",
    )

    catalog = SkillCatalog.discover(tmp_path)
    prompt = PromptBuilder(tmp_path, 10, catalog.metadata()).build()

    assert "Use &lt;carefully&gt; &amp; explicitly." in prompt
    assert "Use <carefully>" not in prompt
