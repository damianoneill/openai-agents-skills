"""Tests for Skill base class, SkillProtocol, and @skill decorator."""

from __future__ import annotations

from typing import Any

import pytest

from openai_agents_skills import Skill, SkillProtocol, skill

# ---------------------------------------------------------------------------
# Skill base class
# ---------------------------------------------------------------------------


class TestSkillDefaults:
    def test_name_default_is_empty_string(self) -> None:
        sk = Skill()
        assert sk.name == ""

    def test_description_default_is_empty_string(self) -> None:
        sk = Skill()
        assert sk.description == ""

    def test_when_to_use_default_is_empty_string(self) -> None:
        sk = Skill()
        assert sk.when_to_use == ""

    def test_is_enabled_defaults_to_true(self) -> None:
        sk = Skill()
        assert sk.is_enabled() is True

    async def test_get_prompt_blocks_raises_not_implemented(self) -> None:
        sk = Skill()
        with pytest.raises(NotImplementedError):
            await sk.get_prompt_blocks()

    async def test_not_implemented_error_names_subclass(self) -> None:
        class MySkill(Skill):
            pass

        with pytest.raises(NotImplementedError, match="MySkill"):
            await MySkill().get_prompt_blocks()


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

        assert DisabledSkill().is_enabled() is False

    def test_is_enabled_can_be_dynamic(self) -> None:
        class ToggleSkill(Skill):
            name = "toggle"
            description = "Toggled externally."

            def __init__(self, active: bool) -> None:
                self._active = active

            def is_enabled(self) -> bool:
                return self._active

        assert ToggleSkill(active=True).is_enabled() is True
        assert ToggleSkill(active=False).is_enabled() is False

    def test_class_attributes_are_overridable(self) -> None:
        class NamedSkill(Skill):
            name = "my_skill"
            description = "My description."
            when_to_use = "Use when you need my_skill."

        sk = NamedSkill()
        assert sk.name == "my_skill"
        assert sk.description == "My description."
        assert sk.when_to_use == "Use when you need my_skill."


# ---------------------------------------------------------------------------
# SkillProtocol
# ---------------------------------------------------------------------------


class TestSkillProtocol:
    def test_skill_subclass_satisfies_protocol(self) -> None:
        class ConcreteSkill(Skill):
            name = "concrete"
            description = "Concrete skill."

            async def get_prompt_blocks(self, args: str = "") -> list[Any]:
                return []

        assert isinstance(ConcreteSkill(), SkillProtocol)

    def test_duck_typed_object_satisfies_protocol(self) -> None:
        class DuckSkill:
            name = "duck"
            description = "Duck-typed skill."

            async def get_prompt_blocks(self, args: str = "") -> list[Any]:
                return []

        assert isinstance(DuckSkill(), SkillProtocol)

    def test_object_missing_name_does_not_satisfy_protocol(self) -> None:
        class NoName:
            description = "missing name attribute"

            async def get_prompt_blocks(self, args: str = "") -> list[Any]:
                return []

        assert not isinstance(NoName(), SkillProtocol)

    def test_object_missing_description_does_not_satisfy_protocol(self) -> None:
        class NoDescription:
            name = "no_desc"

            async def get_prompt_blocks(self, args: str = "") -> list[Any]:
                return []

        assert not isinstance(NoDescription(), SkillProtocol)

    def test_object_missing_get_prompt_blocks_does_not_satisfy_protocol(self) -> None:
        class NoMethod:
            name = "no_method"
            description = "missing method"

        assert not isinstance(NoMethod(), SkillProtocol)

    def test_base_skill_satisfies_protocol(self) -> None:
        assert isinstance(Skill(), SkillProtocol)


# ---------------------------------------------------------------------------
# @skill decorator
# ---------------------------------------------------------------------------


class TestSkillDecorator:
    def test_decorator_attaches_skill_name(self) -> None:
        @skill(name="my_skill")
        def factory() -> Skill:
            return Skill()

        assert factory.__skill_name__ == "my_skill"

    def test_decorator_attaches_description(self) -> None:
        @skill(name="sk", description="A useful skill.")
        def factory() -> Skill:
            return Skill()

        assert factory.__skill_description__ == "A useful skill."

    def test_decorator_default_description_is_empty_string(self) -> None:
        @skill(name="sk")
        def factory() -> Skill:
            return Skill()

        assert factory.__skill_description__ == ""

    def test_decorated_factory_remains_callable(self) -> None:
        class ConcreteSkill(Skill):
            name = "concrete"
            description = "Concrete."

            async def get_prompt_blocks(self, args: str = "") -> list[Any]:
                return []

        @skill(name="concrete", description="Concrete.")
        def factory() -> Skill:
            return ConcreteSkill()

        result = factory()
        assert isinstance(result, Skill)

    def test_decorator_does_not_call_the_factory(self) -> None:
        call_count = 0

        @skill(name="lazy")
        def factory() -> Skill:
            nonlocal call_count
            call_count += 1
            return Skill()

        assert call_count == 0

    def test_decorator_preserves_function_identity(self) -> None:
        def factory() -> Skill:
            return Skill()

        decorated = skill(name="sk")(factory)
        assert decorated is factory
