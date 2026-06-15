"""Tests for Phase 3 — file-based skill loading (loader.py).

Covers: FileSkill, load_skills_from_dir, load_all_skills, SkillConfig,
SkillSource, assert_within_base, and manifest integration for allowed-tools
and user-invocable filtering.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from openai_agents_skills.hooks import _build_manifest
from openai_agents_skills.loader import (
    FileSkill,
    SkillConfig,
    SkillSource,
    _parse_frontmatter,
    _parse_skill_file,
    _SkillFields,
    assert_within_base,
    load_all_skills,
    load_skills_from_dir,
)

# ---------------------------------------------------------------------------
# Path to the checked-in fixture directory
# ---------------------------------------------------------------------------

_FIXTURES_DIR = Path(__file__).parent / "fixtures" / ".agent" / "skills"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fields(
    name: str = "test",
    description: str = "Test skill.",
    always_on: bool = False,
    allowed_tools: list[str] | None = None,
    user_invocable: bool = True,
    license_: str = "",
    compatibility: str = "",
    metadata: dict[str, str] | None = None,
) -> _SkillFields:
    return _SkillFields(
        name=name,
        description=description,
        always_on=always_on,
        allowed_tools=allowed_tools or [],
        argument_hint="",
        user_invocable=user_invocable,
        license_=license_,
        compatibility=compatibility,
        metadata=metadata or {},
    )


def _skill_dir(tmp_path: Path, name: str, content: str) -> Path:
    """Write a SKILL.md into a <name>/ subdirectory under tmp_path."""
    skill_dir = tmp_path / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
    return skill_dir


_MINIMAL_SKILL_MD = """\
---
description: A minimal skill.
---
Do the minimal thing.
"""

_FULL_SKILL_MD = """\
---
name: full-skill
description: A fully-specified skill.
license: MIT
compatibility: Requires Python 3.11+
metadata:
  author: test-org
  version: "1.0"
allowed-tools:
  - tool_a
  - tool_b
argument-hint: "[target]"
user-invocable: true
---
Run on $ARGUMENTS.
"""


# ===========================================================================
# FileSkill
# ===========================================================================


class TestFileSkill:
    """Tests for the FileSkill concrete Skill subclass."""

    async def test_get_prompt_blocks_returns_user_role_message(self, tmp_path: Path) -> None:
        skill = FileSkill(
            fields=_make_fields(name="greet", description="Greets."),
            body="Say hello!",
            file_path=tmp_path / "greet" / "SKILL.md",
        )
        blocks = await skill.get_prompt_blocks(None, None)
        assert blocks == [{"role": "user", "content": "Say hello!"}]

    async def test_get_prompt_blocks_applies_arguments_substitution(self, tmp_path: Path) -> None:
        skill = FileSkill(
            fields=_make_fields(name="connect", description="Connects."),
            body="Connect to $ARGUMENTS.",
            file_path=tmp_path / "connect" / "SKILL.md",
        )
        blocks = await skill.get_prompt_blocks(None, None, args="myserver 8080")
        assert blocks[0]["content"] == "Connect to myserver 8080."

    async def test_get_prompt_blocks_applies_variable_substitution(self, tmp_path: Path) -> None:
        skill = FileSkill(
            fields=_make_fields(name="run", description="Runs."),
            body="Run ID is ${RUN_ID}.",
            file_path=tmp_path / "run" / "SKILL.md",
            variables={"RUN_ID": "abc-123"},
        )
        blocks = await skill.get_prompt_blocks(None, None)
        assert blocks[0]["content"] == "Run ID is abc-123."

    async def test_get_prompt_blocks_no_args_returns_body_unchanged(self, tmp_path: Path) -> None:
        skill = FileSkill(
            fields=_make_fields(),
            body="Static content.",
            file_path=tmp_path / "static" / "SKILL.md",
        )
        blocks = await skill.get_prompt_blocks(None, None)
        assert blocks[0]["content"] == "Static content."

    # ------------------------------------------------------------------
    # Caching behaviour
    # ------------------------------------------------------------------

    async def test_cache_same_args_returns_identical_list_object(self, tmp_path: Path) -> None:
        skill = FileSkill(
            fields=_make_fields(),
            body="Content.",
            file_path=tmp_path / "x" / "SKILL.md",
        )
        first = await skill.get_prompt_blocks(None, None, args="")
        second = await skill.get_prompt_blocks(None, None, args="")
        assert first is second  # same object reference

    async def test_cache_different_args_produce_independent_entries(self, tmp_path: Path) -> None:
        skill = FileSkill(
            fields=_make_fields(),
            body="Value: $ARGUMENTS.",
            file_path=tmp_path / "x" / "SKILL.md",
        )
        blocks_a = await skill.get_prompt_blocks(None, None, args="alpha")
        blocks_b = await skill.get_prompt_blocks(None, None, args="beta")
        assert blocks_a is not blocks_b
        assert blocks_a[0]["content"] == "Value: alpha."
        assert blocks_b[0]["content"] == "Value: beta."

    async def test_cache_empty_args_cached_separately_from_nonempty(self, tmp_path: Path) -> None:
        skill = FileSkill(
            fields=_make_fields(),
            body="Value: $ARGUMENTS",
            file_path=tmp_path / "x" / "SKILL.md",
        )
        empty = await skill.get_prompt_blocks(None, None, args="")
        filled = await skill.get_prompt_blocks(None, None, args="hello")
        assert empty is not filled
        assert empty[0]["content"] == "Value: $ARGUMENTS"
        assert filled[0]["content"] == "Value: hello"

    # ------------------------------------------------------------------
    # Attributes
    # ------------------------------------------------------------------

    def test_attributes_set_from_fields(self, tmp_path: Path) -> None:
        fields = _make_fields(
            name="my-skill",
            description="My description.",
            allowed_tools=["tool_x"],
            user_invocable=False,
            license_="MIT",
            compatibility="Requires Python 3.11+",
            metadata={"author": "test-org"},
        )
        skill = FileSkill(fields=fields, body="body", file_path=tmp_path / "SKILL.md")
        assert skill.name == "my-skill"
        assert skill.description == "My description."
        assert skill.always_on is False
        assert skill.allowed_tools == ["tool_x"]
        assert skill.user_invocable is False
        assert skill.license_ == "MIT"
        assert skill.compatibility == "Requires Python 3.11+"
        assert skill.metadata == {"author": "test-org"}

    def test_allowed_tools_is_instance_copy(self, tmp_path: Path) -> None:
        """FileSkill.allowed_tools must not share the class-level list."""
        original = ["tool_a"]
        fields = _make_fields(allowed_tools=original)
        skill = FileSkill(fields=fields, body="b", file_path=tmp_path / "SKILL.md")
        skill.allowed_tools.append("tool_b")
        assert original == ["tool_a"], "mutating instance list must not affect fields list"

    def test_source_stored(self, tmp_path: Path) -> None:
        skill = FileSkill(
            fields=_make_fields(),
            body="b",
            file_path=tmp_path / "SKILL.md",
            source=SkillSource.USER,
        )
        assert skill.source == SkillSource.USER

    def test_file_path_stored(self, tmp_path: Path) -> None:
        p = tmp_path / "skill" / "SKILL.md"
        skill = FileSkill(fields=_make_fields(), body="b", file_path=p)
        assert skill.file_path == p

    # ------------------------------------------------------------------
    # enabled_when gate
    # ------------------------------------------------------------------

    def test_is_enabled_true_by_default(self, tmp_path: Path) -> None:
        """A FileSkill with no enabled_when predicate is always enabled."""
        skill = FileSkill(fields=_make_fields(), body="b", file_path=tmp_path / "SKILL.md")
        assert skill.enabled_when is None
        assert skill.is_enabled(None, None) is True

    @pytest.mark.parametrize("verdict", [True, False])
    def test_is_enabled_delegates_to_enabled_when(self, tmp_path: Path, verdict: bool) -> None:
        """When a predicate is set, is_enabled returns its result."""
        skill = FileSkill(fields=_make_fields(), body="b", file_path=tmp_path / "SKILL.md")
        skill.enabled_when = lambda context, agent: verdict
        assert skill.is_enabled(None, None) is verdict

    def test_enabled_when_receives_context_and_agent(self, tmp_path: Path) -> None:
        """The predicate is called with the exact context and agent objects."""
        received: list[tuple[Any, Any]] = []
        skill = FileSkill(fields=_make_fields(), body="b", file_path=tmp_path / "SKILL.md")

        def predicate(context: Any, agent: Any) -> bool:
            received.append((context, agent))
            return True

        skill.enabled_when = predicate
        ctx = object()
        agent = object()
        skill.is_enabled(ctx, agent)
        assert received == [(ctx, agent)]


# ===========================================================================
# _parse_frontmatter
# ===========================================================================


class TestParseFrontmatter:
    def test_parses_simple_frontmatter(self) -> None:
        content = "---\nname: foo\ndescription: bar\n---\nBody text.\n"
        fields, body = _parse_frontmatter(content)
        assert fields == {"name": "foo", "description": "bar"}
        assert body == "Body text.\n"

    def test_no_frontmatter_returns_empty_dict_and_full_content(self) -> None:
        content = "Just a plain body."
        fields, body = _parse_frontmatter(content)
        assert fields == {}
        assert body == "Just a plain body."

    def test_missing_closing_delimiter_returns_empty_dict(self) -> None:
        content = "---\nname: foo\n"
        fields, body = _parse_frontmatter(content)
        assert fields == {}

    def test_list_value_in_frontmatter(self) -> None:
        content = "---\nallowed-tools:\n  - tool_a\n  - tool_b\n---\nbody\n"
        fields, body = _parse_frontmatter(content)
        assert fields["allowed-tools"] == ["tool_a", "tool_b"]

    def test_body_stripped_of_leading_newline_after_closing_delimiter(self) -> None:
        content = "---\nname: x\n---\nHello.\n"
        _, body = _parse_frontmatter(content)
        assert body.startswith("Hello.")


# ===========================================================================
# _parse_skill_file
# ===========================================================================


class TestParseSkillFile:
    def test_returns_fields_and_body_on_valid_input(self) -> None:
        content = "---\ndescription: My skill.\n---\nDo things.\n"
        result = _parse_skill_file(content, "my-skill")
        assert result is not None
        fields, body = result
        assert fields.description == "My skill."
        assert body == "Do things."

    def test_directory_name_used_as_default_skill_name(self) -> None:
        content = "---\ndescription: A skill.\n---\nbody\n"
        result = _parse_skill_file(content, "my-dir")
        assert result is not None
        fields, _ = result
        assert fields.name == "my-dir"

    def test_name_field_overrides_directory_name(self) -> None:
        content = "---\nname: custom-name\ndescription: A skill.\n---\nbody\n"
        result = _parse_skill_file(content, "my-dir")
        assert result is not None
        fields, _ = result
        assert fields.name == "custom-name"

    def test_missing_description_returns_none(self) -> None:
        content = "---\nname: no-desc\n---\nbody\n"
        assert _parse_skill_file(content, "no-desc") is None

    def test_empty_description_returns_none(self) -> None:
        content = "---\ndescription: \n---\nbody\n"
        assert _parse_skill_file(content, "empty") is None

    def test_optional_fields_default_correctly(self) -> None:
        content = "---\ndescription: Minimal.\n---\nbody\n"
        result = _parse_skill_file(content, "minimal")
        assert result is not None
        fields, _ = result
        assert fields.always_on is False
        assert fields.allowed_tools == []
        assert fields.argument_hint == ""
        assert fields.user_invocable is True
        assert fields.license_ == ""
        assert fields.compatibility == ""
        assert fields.metadata == {}

    def test_allowed_tools_parsed_as_list(self) -> None:
        content = "---\ndescription: D.\nallowed-tools:\n  - t_a\n  - t_b\n---\nbody\n"
        result = _parse_skill_file(content, "x")
        assert result is not None
        assert result[0].allowed_tools == ["t_a", "t_b"]

    def test_user_invocable_false_parsed(self) -> None:
        content = "---\ndescription: D.\nuser-invocable: false\n---\nbody\n"
        result = _parse_skill_file(content, "x")
        assert result is not None
        assert result[0].user_invocable is False

    def test_name_with_path_separator_raises(self) -> None:
        content = "---\nname: ../evil\ndescription: D.\n---\nbody\n"
        with pytest.raises(ValueError, match="invalid"):
            _parse_skill_file(content, "x")

    def test_allowed_tools_parsed_from_space_separated_string(self) -> None:
        content = "---\ndescription: D.\nallowed-tools: tool_a tool_b\n---\nbody\n"
        result = _parse_skill_file(content, "x")
        assert result is not None
        assert result[0].allowed_tools == ["tool_a", "tool_b"]

    def test_allowed_tools_string_with_complex_names(self) -> None:
        content = "---\ndescription: D.\nallowed-tools: Bash(git:*) Bash(jq:*) Read\n---\nbody\n"
        result = _parse_skill_file(content, "x")
        assert result is not None
        assert result[0].allowed_tools == ["Bash(git:*)", "Bash(jq:*)", "Read"]

    def test_license_field_parsed(self) -> None:
        content = "---\ndescription: D.\nlicense: Apache-2.0\n---\nbody\n"
        result = _parse_skill_file(content, "x")
        assert result is not None
        assert result[0].license_ == "Apache-2.0"

    def test_compatibility_field_parsed(self) -> None:
        content = "---\ndescription: D.\ncompatibility: Requires Python 3.11+\n---\nbody\n"
        result = _parse_skill_file(content, "x")
        assert result is not None
        assert result[0].compatibility == "Requires Python 3.11+"

    def test_metadata_field_parsed(self) -> None:
        content = (
            "---\ndescription: D.\nmetadata:\n  author: test-org\n  version: '1.0'\n---\nbody\n"
        )
        result = _parse_skill_file(content, "x")
        assert result is not None
        assert result[0].metadata == {"author": "test-org", "version": "1.0"}

    def test_name_with_uppercase_raises(self) -> None:
        content = "---\nname: myskill\ndescription: D.\n---\nbody\n"
        result = _parse_skill_file(content, "myskill")
        assert result is not None  # lowercase is valid

        content = "---\nname: MySkill\ndescription: D.\n---\nbody\n"
        with pytest.raises(ValueError, match="invalid"):
            _parse_skill_file(content, "myskill")

    def test_name_with_consecutive_hyphens_raises(self) -> None:
        content = "---\nname: pdf--processing\ndescription: D.\n---\nbody\n"
        with pytest.raises(ValueError, match="consecutive hyphens"):
            _parse_skill_file(content, "pdf--processing")

    def test_name_with_leading_hyphen_raises(self) -> None:
        content = "---\nname: -skill\ndescription: D.\n---\nbody\n"
        with pytest.raises(ValueError, match="invalid"):
            _parse_skill_file(content, "x")

    def test_name_too_long_raises(self) -> None:
        long_name = "a" * 65
        content = f"---\nname: {long_name}\ndescription: D.\n---\nbody\n"
        with pytest.raises(ValueError, match="64 characters"):
            _parse_skill_file(content, long_name)

    def test_description_too_long_raises(self) -> None:
        long_desc = "a" * 1025
        content = f"---\ndescription: {long_desc}\n---\nbody\n"
        with pytest.raises(ValueError, match="1024 characters"):
            _parse_skill_file(content, "x")

    def test_compatibility_too_long_raises(self) -> None:
        long_compat = "a" * 501
        content = f"---\ndescription: D.\ncompatibility: {long_compat}\n---\nbody\n"
        with pytest.raises(ValueError, match="500 characters"):
            _parse_skill_file(content, "x")


# ===========================================================================
# assert_within_base
# ===========================================================================


class TestAssertWithinBase:
    def test_path_inside_base_passes(self, tmp_path: Path) -> None:
        child = tmp_path / "subdir" / "file.txt"
        child.parent.mkdir()
        child.touch()
        assert_within_base(child, tmp_path)  # must not raise

    def test_path_equal_to_base_passes(self, tmp_path: Path) -> None:
        assert_within_base(tmp_path, tmp_path)  # must not raise

    def test_path_outside_base_raises(self, tmp_path: Path) -> None:
        outside = tmp_path.parent / "other"
        with pytest.raises(ValueError, match="escapes base directory"):
            assert_within_base(outside, tmp_path)

    def test_dotdot_escape_attempt_raises(self, tmp_path: Path) -> None:
        escape = tmp_path / ".." / "escape"
        with pytest.raises(ValueError, match="escapes base directory"):
            assert_within_base(escape, tmp_path)


# ===========================================================================
# load_skills_from_dir
# ===========================================================================


class TestLoadSkillsFromDir:
    async def test_returns_file_skill_per_subdirectory_with_skill_md(self, tmp_path: Path) -> None:
        _skill_dir(tmp_path, "skill-a", _MINIMAL_SKILL_MD)
        _skill_dir(tmp_path, "skill-b", _MINIMAL_SKILL_MD.replace("minimal", "second"))

        results = await load_skills_from_dir(tmp_path, SkillSource.PROJECT)
        names = {skill.name for skill, _ in results}
        assert "skill-a" in names or "skill-b" in names
        assert len(results) == 2

    async def test_subdirectory_without_skill_md_silently_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "no-skill").mkdir()
        _skill_dir(tmp_path, "has-skill", _MINIMAL_SKILL_MD)

        results = await load_skills_from_dir(tmp_path, SkillSource.PROJECT)
        assert len(results) == 1

    async def test_missing_top_level_directory_returns_empty(self) -> None:
        missing = Path("/nonexistent/path/that/does/not/exist")
        results = await load_skills_from_dir(missing, SkillSource.PROJECT)
        assert results == []

    async def test_empty_directory_returns_empty(self, tmp_path: Path) -> None:
        results = await load_skills_from_dir(tmp_path, SkillSource.PROJECT)
        assert results == []

    async def test_returns_canonical_path(self, tmp_path: Path) -> None:
        _skill_dir(tmp_path, "skill-a", _MINIMAL_SKILL_MD)
        results = await load_skills_from_dir(tmp_path, SkillSource.PROJECT)
        assert len(results) == 1
        skill, canon_path = results[0]
        assert canon_path == (tmp_path / "skill-a" / "SKILL.md").resolve()

    async def test_source_stored_on_file_skill(self, tmp_path: Path) -> None:
        _skill_dir(tmp_path, "skill-a", _MINIMAL_SKILL_MD)
        results = await load_skills_from_dir(tmp_path, SkillSource.USER)
        assert results[0][0].source == SkillSource.USER

    async def test_variables_threaded_to_skill(self, tmp_path: Path) -> None:
        content = "---\ndescription: Var skill.\n---\nRun ID is ${RUN_ID}.\n"
        _skill_dir(tmp_path, "var-skill", content)
        results = await load_skills_from_dir(
            tmp_path, SkillSource.PROJECT, variables={"RUN_ID": "xyz"}
        )
        assert len(results) == 1
        blocks = await results[0][0].get_prompt_blocks(None, None)
        assert "xyz" in blocks[0]["content"]

    async def test_skill_missing_description_is_skipped(self, tmp_path: Path) -> None:
        content = "---\nname: no-desc\n---\nbody\n"
        _skill_dir(tmp_path, "no-desc", content)
        results = await load_skills_from_dir(tmp_path, SkillSource.PROJECT)
        assert results == []

    async def test_all_frontmatter_fields_loaded_from_fixture(self) -> None:
        """Integration test against the checked-in SKILL.md fixtures."""
        results = await load_skills_from_dir(_FIXTURES_DIR, SkillSource.PROJECT)
        skill_map = {skill.name: skill for skill, _ in results}

        assert "test-skill" in skill_map
        ts = skill_map["test-skill"]
        assert ts.description == "A test skill for automated tests."

    async def test_with_args_fixture_loaded(self) -> None:
        results = await load_skills_from_dir(_FIXTURES_DIR, SkillSource.PROJECT)
        skill_map = {skill.name: skill for skill, _ in results}
        assert "with-args" in skill_map
        wa = skill_map["with-args"]
        blocks = await wa.get_prompt_blocks(None, None, args="myhost 9090")
        assert "myhost 9090" in blocks[0]["content"]

    async def test_with_tools_fixture_has_allowed_tools(self) -> None:
        results = await load_skills_from_dir(_FIXTURES_DIR, SkillSource.PROJECT)
        skill_map = {skill.name: skill for skill, _ in results}
        assert "with-tools" in skill_map
        assert skill_map["with-tools"].allowed_tools == ["tool_a", "tool_b"]

    async def test_not_user_invocable_fixture_has_flag_false(self) -> None:
        results = await load_skills_from_dir(_FIXTURES_DIR, SkillSource.PROJECT)
        skill_map = {skill.name: skill for skill, _ in results}
        assert "not-user-invocable" in skill_map
        assert skill_map["not-user-invocable"].user_invocable is False

    @pytest.mark.skipif(os.name == "nt", reason="symlinks may require privileges on Windows")
    async def test_realpath_deduplication_via_symlink(self, tmp_path: Path) -> None:
        """A symlink pointing to the same SKILL.md is only loaded once."""
        real_dir = tmp_path / "real"
        _skill_dir(real_dir, "skill-a", _MINIMAL_SKILL_MD)

        # Create a second directory that is a symlink to the real one.
        link_dir = tmp_path / "link"
        link_dir.symlink_to(real_dir)

        real_results = await load_skills_from_dir(real_dir, SkillSource.PROJECT)
        link_results = await load_skills_from_dir(link_dir, SkillSource.PROJECT)

        # Both point to the same canonical path.
        assert len(real_results) == 1
        assert len(link_results) == 1
        real_canon = real_results[0][1]
        link_canon = link_results[0][1]
        assert real_canon == link_canon


# ===========================================================================
# load_all_skills
# ===========================================================================


class TestLoadAllSkills:
    async def test_returns_populated_registry(self, tmp_path: Path) -> None:
        project_dir = tmp_path / "project" / ".agent" / "skills"
        _skill_dir(project_dir, "my-skill", _MINIMAL_SKILL_MD)

        config = SkillConfig(
            project_dir=project_dir,
            user_dir=tmp_path / "nonexistent-user",
        )
        registry = await load_all_skills(cwd=tmp_path, config=config)
        assert "skill-a" not in registry.skill_names
        skill = registry.get("my-skill")
        assert skill.description == "A minimal skill."

    async def test_user_layer_wins_over_project_layer_on_name_conflict(
        self, tmp_path: Path
    ) -> None:
        user_dir = tmp_path / "user" / ".agent" / "skills"
        project_dir = tmp_path / "project" / ".agent" / "skills"

        user_content = "---\nname: shared\ndescription: User version.\n---\nUser body.\n"
        project_content = "---\nname: shared\ndescription: Project version.\n---\nProject body.\n"
        _skill_dir(user_dir, "shared", user_content)
        _skill_dir(project_dir, "shared", project_content)

        config = SkillConfig(user_dir=user_dir, project_dir=project_dir)
        registry = await load_all_skills(cwd=tmp_path, config=config)

        skill = registry.get("shared")
        assert skill.description == "User version."

    async def test_missing_user_and_project_dirs_returns_empty_registry(
        self, tmp_path: Path
    ) -> None:
        config = SkillConfig(
            user_dir=tmp_path / "no-user",
            project_dir=tmp_path / "no-project",
        )
        registry = await load_all_skills(cwd=tmp_path, config=config)
        assert registry.skill_names == []

    async def test_extra_dirs_loaded(self, tmp_path: Path) -> None:
        extra_dir = tmp_path / "extra"
        _skill_dir(extra_dir, "extra-skill", _MINIMAL_SKILL_MD)

        config = SkillConfig(
            user_dir=tmp_path / "no-user",
            project_dir=tmp_path / "no-project",
            extra_dirs=[extra_dir],
        )
        registry = await load_all_skills(cwd=tmp_path, config=config)
        assert "extra-skill" in registry.skill_names

    async def test_user_wins_over_extra(self, tmp_path: Path) -> None:
        user_dir = tmp_path / "user"
        extra_dir = tmp_path / "extra"

        user_content = "---\nname: shared\ndescription: User version.\n---\nU.\n"
        extra_content = "---\nname: shared\ndescription: Extra version.\n---\nE.\n"
        _skill_dir(user_dir, "shared", user_content)
        _skill_dir(extra_dir, "shared", extra_content)

        config = SkillConfig(
            user_dir=user_dir,
            project_dir=tmp_path / "no-project",
            extra_dirs=[extra_dir],
        )
        registry = await load_all_skills(cwd=tmp_path, config=config)
        assert registry.get("shared").description == "User version."

    async def test_variables_from_config_threaded_through(self, tmp_path: Path) -> None:
        project_dir = tmp_path / "project"
        content = "---\ndescription: Var skill.\n---\nRun ID: ${RUN_ID}.\n"
        _skill_dir(project_dir, "var-skill", content)

        config = SkillConfig(
            user_dir=tmp_path / "no-user",
            project_dir=project_dir,
            variables={"RUN_ID": "test-42"},
        )
        registry = await load_all_skills(cwd=tmp_path, config=config)
        skill = registry.get("var-skill")
        blocks = await skill.get_prompt_blocks(None, None)
        assert "test-42" in blocks[0]["content"]

    @pytest.mark.skipif(os.name == "nt", reason="symlinks may require privileges on Windows")
    async def test_realpath_deduplication_across_layers(self, tmp_path: Path) -> None:
        """If user_dir and project_dir point to the same physical directory, load once."""
        real_dir = tmp_path / "real"
        _skill_dir(real_dir, "shared-skill", _MINIMAL_SKILL_MD)

        link_dir = tmp_path / "link"
        link_dir.symlink_to(real_dir)

        config = SkillConfig(user_dir=real_dir, project_dir=link_dir)
        registry = await load_all_skills(cwd=tmp_path, config=config)

        # Skill must appear exactly once.
        assert registry.skill_names.count("shared-skill") == 1
        assert len(registry.all_skills) == 1

    async def test_default_project_dir_resolved_from_cwd(self, tmp_path: Path) -> None:
        """load_all_skills uses <cwd>/.agent/skills/ when project_dir is None."""
        skills_dir = tmp_path / ".agent" / "skills"
        _skill_dir(skills_dir, "cwd-skill", _MINIMAL_SKILL_MD)

        config = SkillConfig(user_dir=tmp_path / "no-user")
        registry = await load_all_skills(cwd=tmp_path, config=config)
        assert "cwd-skill" in registry.skill_names

    async def test_no_config_uses_defaults_and_does_not_raise(self, tmp_path: Path) -> None:
        """Calling with config=None must not raise even if dirs are absent."""
        registry = await load_all_skills(cwd=tmp_path)
        assert isinstance(registry.skill_names, list)


# ===========================================================================
# Manifest integration — allowed-tools and user-invocable
# ===========================================================================


class TestManifestIntegration:
    async def test_allowed_tools_appear_in_manifest(self, tmp_path: Path) -> None:
        results = await load_skills_from_dir(_FIXTURES_DIR, SkillSource.PROJECT)
        skills = [s for s, _ in results]
        manifest = _build_manifest(skills)
        assert "Allowed tools: tool_a, tool_b" in manifest

    async def test_no_allowed_tools_absent_from_manifest(self, tmp_path: Path) -> None:
        results = await load_skills_from_dir(_FIXTURES_DIR, SkillSource.PROJECT)
        skills = [s for s, _ in results]
        manifest = _build_manifest(skills)
        # test-skill has no allowed-tools; its line should not mention "Allowed tools"
        lines = [ln for ln in manifest.splitlines() if "test-skill" in ln]
        assert lines, "test-skill should appear in manifest"
        assert "Allowed tools" not in lines[0]

    async def test_non_user_invocable_skill_excluded_from_manifest(self, tmp_path: Path) -> None:
        results = await load_skills_from_dir(_FIXTURES_DIR, SkillSource.PROJECT)
        skills = [s for s, _ in results]
        manifest = _build_manifest(skills)
        assert "not-user-invocable" not in manifest

    async def test_user_invocable_skill_appears_in_manifest(self, tmp_path: Path) -> None:
        results = await load_skills_from_dir(_FIXTURES_DIR, SkillSource.PROJECT)
        skills = [s for s, _ in results]
        manifest = _build_manifest(skills)
        assert "test-skill" in manifest

    def test_manifest_with_inline_skill_having_allowed_tools(self) -> None:
        """_build_manifest works for non-FileSkill instances with allowed_tools."""
        from openai_agents_skills import Skill

        class ToolySkill(Skill):
            name = "tooled"
            description = "Has tools."
            allowed_tools: list[str] = ["alpha", "beta"]

            async def get_prompt_blocks(self, context, agent, args: str = "") -> list[Any]:
                return []

        manifest = _build_manifest([ToolySkill()])
        assert "Allowed tools: alpha, beta" in manifest

    def test_manifest_excludes_user_invocable_false_inline_skill(self) -> None:
        from openai_agents_skills import Skill

        class HiddenSkill(Skill):
            name = "hidden"
            description = "Hidden."
            user_invocable = False

            async def get_prompt_blocks(self, context, agent, args: str = "") -> list[Any]:
                return []

        manifest = _build_manifest([HiddenSkill()])
        assert manifest == ""


# ===========================================================================
# SkillSource
# ===========================================================================


class TestSkillSource:
    def test_all_source_values_are_strings(self) -> None:
        for source in SkillSource:
            assert isinstance(source.value, str)

    def test_source_names(self) -> None:
        assert SkillSource.BUNDLED.value == "bundled"
        assert SkillSource.USER.value == "user"
        assert SkillSource.PROJECT.value == "project"
        assert SkillSource.EXTRA.value == "extra"
