"""Tests for Skill base class."""

from __future__ import annotations

from typing import Any

import pytest

from openai_agents_skills import Skill

# ---------------------------------------------------------------------------
# Minimal concrete implementation used to test class-level defaults
# ---------------------------------------------------------------------------


class _MinimalSkill(Skill):
    """Smallest valid Skill subclass — used only to test inherited defaults."""

    async def get_prompt_blocks(self, args: str = "") -> list[Any]:
        return []


# ---------------------------------------------------------------------------
# Skill base class — abstract enforcement
# ---------------------------------------------------------------------------


class TestSkillAbstract:
    def test_skill_base_class_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError):
            Skill()  # type: ignore[abstract]

    def test_subclass_without_get_prompt_blocks_cannot_be_instantiated(self) -> None:
        class IncompleteSkill(Skill):
            pass

        with pytest.raises(TypeError):
            IncompleteSkill()  # type: ignore[abstract]

    def test_subclass_with_get_prompt_blocks_can_be_instantiated(self) -> None:
        sk = _MinimalSkill()
        assert sk is not None


# ---------------------------------------------------------------------------
# Skill base class — defaults
# ---------------------------------------------------------------------------


class TestSkillDefaults:
    def test_name_default_is_empty_string(self) -> None:
        sk = _MinimalSkill()
        assert sk.name == ""

    def test_description_default_is_empty_string(self) -> None:
        sk = _MinimalSkill()
        assert sk.description == ""

    def test_when_to_use_default_is_empty_string(self) -> None:
        sk = _MinimalSkill()
        assert sk.when_to_use == ""

    def test_is_enabled_defaults_to_true(self) -> None:
        sk = _MinimalSkill()
        assert sk.is_enabled() is True


# ---------------------------------------------------------------------------
# Skill subclasses
# ---------------------------------------------------------------------------


class TestSkillSubclass:
    async def test_concrete_subclass_returns_blocks(self) -> None:
        class ConciseSkill(Skill):
            name = "concise"
            description = "Be concise."

            async def get_prompt_blocks(self, args: str = "") -> list[Any]:
                return [{"role": "user", "content": "Be concise."}]

        blocks = await ConciseSkill().get_prompt_blocks()
        assert blocks == [{"role": "user", "content": "Be concise."}]

    async def test_get_prompt_blocks_receives_args(self) -> None:
        class EchoSkill(Skill):
            name = "echo"
            description = "Echoes args into a block."

            async def get_prompt_blocks(self, args: str = "") -> list[Any]:
                return [{"role": "user", "content": args}]

        blocks = await EchoSkill().get_prompt_blocks(args="hello world")
        assert blocks[0]["content"] == "hello world"

    async def test_get_prompt_blocks_default_args_is_empty_string(self) -> None:
        received: list[str] = []

        class RecordingSkill(Skill):
            name = "recording"
            description = "Records args value."

            async def get_prompt_blocks(self, args: str = "") -> list[Any]:
                received.append(args)
                return []

        await RecordingSkill().get_prompt_blocks()
        assert received == [""]

    async def test_get_prompt_blocks_can_return_multiple_blocks(self) -> None:
        class MultiSkill(Skill):
            name = "multi"
            description = "Returns multiple blocks."

            async def get_prompt_blocks(self, args: str = "") -> list[Any]:
                return [
                    {"role": "user", "content": "block one"},
                    {"role": "user", "content": "block two"},
                ]

        blocks = await MultiSkill().get_prompt_blocks()
        assert len(blocks) == 2
        assert blocks[0]["content"] == "block one"
        assert blocks[1]["content"] == "block two"

    async def test_get_prompt_blocks_can_return_empty_list(self) -> None:
        class EmptySkill(Skill):
            name = "empty"
            description = "Returns no blocks."

            async def get_prompt_blocks(self, args: str = "") -> list[Any]:
                return []

        assert await EmptySkill().get_prompt_blocks() == []

    def test_subclass_can_disable_via_is_enabled(self) -> None:
        class DisabledSkill(Skill):
            name = "disabled"
            description = "Always disabled."

            def is_enabled(self) -> bool:
                return False

            async def get_prompt_blocks(self, args: str = "") -> list[Any]:
                return []

        assert DisabledSkill().is_enabled() is False

    def test_is_enabled_can_be_dynamic(self) -> None:
        class ToggleSkill(Skill):
            name = "toggle"
            description = "Toggled externally."

            def __init__(self, active: bool) -> None:
                self._active = active

            def is_enabled(self) -> bool:
                return self._active

            async def get_prompt_blocks(self, args: str = "") -> list[Any]:
                return []

        assert ToggleSkill(active=True).is_enabled() is True
        assert ToggleSkill(active=False).is_enabled() is False

    def test_class_attributes_are_overridable(self) -> None:
        class NamedSkill(Skill):
            name = "my_skill"
            description = "My description."
            when_to_use = "Use when you need my_skill."

            async def get_prompt_blocks(self, args: str = "") -> list[Any]:
                return []

        sk = NamedSkill()
        assert sk.name == "my_skill"
        assert sk.description == "My description."
        assert sk.when_to_use == "Use when you need my_skill."


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


class TestSkillRepr:
    def test_repr_includes_class_name_and_skill_name(self) -> None:
        class MySkill(Skill):
            name = "my_skill"
            description = "A test skill."

            async def get_prompt_blocks(self, args: str = "") -> list[Any]:
                return []

        result = repr(MySkill())

        assert "MySkill" in result
        assert "my_skill" in result

    def test_repr_with_empty_name(self) -> None:
        """A skill with the default empty name still has a valid repr."""

        class NoNameSkill(Skill):
            async def get_prompt_blocks(self, args: str = "") -> list[Any]:
                return []

        result = repr(NoNameSkill())

        assert "NoNameSkill" in result
        assert "name=" in result

    def test_repr_format(self) -> None:
        class CitationSkill(Skill):
            name = "citation"
            description = "Cite sources."

            async def get_prompt_blocks(self, args: str = "") -> list[Any]:
                return []

        assert repr(CitationSkill()) == "CitationSkill(name='citation')"
