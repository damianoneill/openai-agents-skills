"""Tests for authored instruction forms (default / concise).

Covers the public scope helpers (InstructionForm, instruction_form_scope,
get_instruction_form) and the FileSkill ``concise-instructions`` field.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from openai_agents_skills import (
    InstructionForm,
    Skill,
    get_instruction_form,
    instruction_form_scope,
)
from openai_agents_skills._state import _run_state
from openai_agents_skills.loader import FileSkill, _parse_skill_file, _SkillFields


def _make_fields(
    name: str = "test",
    description: str = "Test skill.",
    concise_instructions: str = "",
) -> _SkillFields:
    return _SkillFields(
        name=name,
        description=description,
        concise_instructions=concise_instructions,
    )


# ===========================================================================
# Public scope and accessor
# ===========================================================================


class TestInstructionFormScope:
    def test_enum_has_default_and_concise(self) -> None:
        assert InstructionForm.DEFAULT.value == "default"
        assert InstructionForm.CONCISE.value == "concise"

    def test_accessor_default_without_scope(self) -> None:
        assert get_instruction_form() is InstructionForm.DEFAULT

    def test_scope_sets_and_restores_form(self) -> None:
        assert get_instruction_form() is InstructionForm.DEFAULT
        with instruction_form_scope(InstructionForm.CONCISE):
            assert get_instruction_form() is InstructionForm.CONCISE
        assert get_instruction_form() is InstructionForm.DEFAULT

    def test_scope_restores_on_exception(self) -> None:
        with pytest.raises(RuntimeError):
            with instruction_form_scope(InstructionForm.CONCISE):
                assert get_instruction_form() is InstructionForm.CONCISE
                raise RuntimeError("boom")
        assert get_instruction_form() is InstructionForm.DEFAULT

    def test_invalid_selector_raises_at_entry(self) -> None:
        with pytest.raises(ValueError):
            with instruction_form_scope("nonsense"):  # type: ignore[arg-type]
                pytest.fail("scope body must not run for an invalid selector")
        assert get_instruction_form() is InstructionForm.DEFAULT

    def test_invalid_selector_leaves_enclosing_selection(self) -> None:
        with instruction_form_scope(InstructionForm.CONCISE):
            with pytest.raises(ValueError):
                with instruction_form_scope("nonsense"):  # type: ignore[arg-type]
                    pass
            assert get_instruction_form() is InstructionForm.CONCISE

    def test_nested_default_overrides_inherited_concise(self) -> None:
        with instruction_form_scope(InstructionForm.CONCISE):
            with instruction_form_scope(InstructionForm.DEFAULT):
                assert get_instruction_form() is InstructionForm.DEFAULT
            assert get_instruction_form() is InstructionForm.CONCISE

    def test_accessor_does_not_create_run_state(self) -> None:
        _run_state.set(None)
        get_instruction_form()
        assert _run_state.get() is None


# ===========================================================================
# FileSkill file authoring
# ===========================================================================


class TestFileSkillConcise:
    def _skill(self, tmp_path: Path, body: str, concise: str = "") -> FileSkill:
        return FileSkill(
            fields=_make_fields(name="evidence", concise_instructions=concise),
            body=body,
            file_path=tmp_path / "evidence" / "SKILL.md",
        )

    async def test_default_body_without_scope(self, tmp_path: Path) -> None:
        skill = self._skill(tmp_path, body="Full body.", concise="Short body.")
        blocks = await skill.get_prompt_blocks(None, None)
        assert blocks == [{"role": "user", "content": "Full body."}]

    async def test_concise_body_under_concise_scope(self, tmp_path: Path) -> None:
        skill = self._skill(tmp_path, body="Full body.", concise="Short body.")
        with instruction_form_scope(InstructionForm.CONCISE):
            blocks = await skill.get_prompt_blocks(None, None)
        assert blocks == [{"role": "user", "content": "Short body."}]

    async def test_missing_concise_falls_back_to_default(self, tmp_path: Path) -> None:
        skill = self._skill(tmp_path, body="Full body.", concise="")
        with instruction_form_scope(InstructionForm.CONCISE):
            blocks = await skill.get_prompt_blocks(None, None)
        assert blocks == [{"role": "user", "content": "Full body."}]

    async def test_fallback_reuses_default_cache_object(self, tmp_path: Path) -> None:
        skill = self._skill(tmp_path, body="Full body.", concise="")
        default_blocks = await skill.get_prompt_blocks(None, None)
        with instruction_form_scope(InstructionForm.CONCISE):
            concise_blocks = await skill.get_prompt_blocks(None, None)
        assert concise_blocks is default_blocks

    async def test_fallback_emits_debug_diagnostic(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        skill = self._skill(tmp_path, body="Full body.", concise="")
        with caplog.at_level(logging.DEBUG, logger="openai_agents_skills.loader"):
            with instruction_form_scope(InstructionForm.CONCISE):
                await skill.get_prompt_blocks(None, None)
        messages = " ".join(r.getMessage() for r in caplog.records)
        assert "evidence" in messages
        assert "Full body." not in messages

    async def test_fallback_after_default_render_still_logs(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Default renders first and populates the (DEFAULT, args) cache entry; the
        # later concise request must still emit the fallback diagnostic and reuse
        # that cached object rather than logging nothing.
        skill = self._skill(tmp_path, body="Full body.", concise="")
        default_blocks = await skill.get_prompt_blocks(None, None)
        with caplog.at_level(logging.DEBUG, logger="openai_agents_skills.loader"):
            with instruction_form_scope(InstructionForm.CONCISE):
                concise_blocks = await skill.get_prompt_blocks(None, None)
        messages = " ".join(r.getMessage() for r in caplog.records)
        assert "evidence" in messages
        assert concise_blocks is default_blocks

    async def test_concise_body_applies_argument_substitution(self, tmp_path: Path) -> None:
        skill = self._skill(tmp_path, body="Full $ARGUMENTS.", concise="Short $ARGUMENTS.")
        with instruction_form_scope(InstructionForm.CONCISE):
            blocks = await skill.get_prompt_blocks(None, None, "target")
        assert blocks == [{"role": "user", "content": "Short target."}]

    async def test_alternating_forms_same_args_return_correct_body(self, tmp_path: Path) -> None:
        skill = self._skill(tmp_path, body="Full body.", concise="Short body.")
        default_first = await skill.get_prompt_blocks(None, None)
        with instruction_form_scope(InstructionForm.CONCISE):
            concise = await skill.get_prompt_blocks(None, None)
        default_again = await skill.get_prompt_blocks(None, None)
        assert default_first[0]["content"] == "Full body."
        assert concise[0]["content"] == "Short body."
        assert default_again is default_first

    async def test_same_form_repeat_returns_identical_object(self, tmp_path: Path) -> None:
        skill = self._skill(tmp_path, body="Full body.", concise="Short body.")
        with instruction_form_scope(InstructionForm.CONCISE):
            first = await skill.get_prompt_blocks(None, None)
            second = await skill.get_prompt_blocks(None, None)
        assert first is second


# ===========================================================================
# Parsing the concise-instructions frontmatter field
# ===========================================================================


class TestParseConciseInstructions:
    def test_parses_concise_instructions(self) -> None:
        content = "---\ndescription: A skill.\nconcise-instructions: Short form.\n---\nFull form.\n"
        parsed = _parse_skill_file(content, "evidence")
        assert parsed is not None
        fields, _body = parsed
        assert fields.concise_instructions == "Short form."

    def test_absent_concise_instructions_is_empty(self) -> None:
        content = "---\ndescription: A skill.\n---\nFull form.\n"
        parsed = _parse_skill_file(content, "evidence")
        assert parsed is not None
        fields, _body = parsed
        assert fields.concise_instructions == ""

    def test_blank_concise_instructions_is_absent(self) -> None:
        content = '---\ndescription: A skill.\nconcise-instructions: "   "\n---\nFull form.\n'
        parsed = _parse_skill_file(content, "evidence")
        assert parsed is not None
        fields, _body = parsed
        assert fields.concise_instructions == ""

    def test_null_concise_instructions_is_absent(self) -> None:
        content = "---\ndescription: A skill.\nconcise-instructions:\n---\nFull form.\n"
        parsed = _parse_skill_file(content, "evidence")
        assert parsed is not None
        fields, _body = parsed
        assert fields.concise_instructions == ""

    def test_non_string_concise_instructions_is_rejected(self) -> None:
        content = (
            "---\ndescription: A skill.\nconcise-instructions:\n  - a\n  - b\n---\nFull form.\n"
        )
        with pytest.raises(ValueError):
            _parse_skill_file(content, "evidence")


# ===========================================================================
# Python authoring
# ===========================================================================


class TestPythonAuthoring:
    async def test_python_skill_reads_active_form(self) -> None:
        class EvidenceSkill(Skill):
            name = "evidence"
            description = "Reports evidence."

            async def get_prompt_blocks(
                self, context: Any, agent: Any, args: str = ""
            ) -> list[Any]:
                if get_instruction_form() is InstructionForm.CONCISE:
                    return [{"role": "user", "content": "Short."}]
                return [{"role": "user", "content": "Full."}]

        skill = EvidenceSkill()
        default_blocks = await skill.get_prompt_blocks(None, None)
        with instruction_form_scope(InstructionForm.CONCISE):
            concise_blocks = await skill.get_prompt_blocks(None, None)
        assert default_blocks == [{"role": "user", "content": "Full."}]
        assert concise_blocks == [{"role": "user", "content": "Short."}]
