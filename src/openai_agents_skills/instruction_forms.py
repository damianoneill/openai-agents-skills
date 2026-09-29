"""Authored instruction forms — default and concise.

Applications select an instruction form for an enclosed execution with
:func:`instruction_form_scope`.  Skills read the active form with
:func:`get_instruction_form`.  The form is an immutable value held in a
dedicated :class:`~contextvars.ContextVar`, separate from mutable run state,
so concurrent runs and inherited child tasks stay isolated.

The scope owns the selection.  It is set before the SDK creates tasks and its
token is reset in the caller's ``finally`` path.  Handoffs, deferred rendering
and the ``invoke_skill`` tool all observe the enclosing scope because they
render through :meth:`Skill.get_prompt_blocks`.

Example::

    from openai_agents_skills import InstructionForm, instruction_form_scope

    with instruction_form_scope(InstructionForm.CONCISE):
        result = await Runner.run(agent, question, hooks=hooks)
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from enum import StrEnum


class InstructionForm(StrEnum):
    """The instruction form to render.

    These are alternatives, not ordered levels.  ``DEFAULT`` is the output an
    existing skill produces today; ``CONCISE`` is an authored shorter form.

    Attributes:
        DEFAULT: The skill's existing instructions.
        CONCISE: An authored concise alternative supplied by the skill.
    """

    DEFAULT = "default"
    CONCISE = "concise"


_instruction_form: ContextVar[InstructionForm] = ContextVar(
    "skill_instruction_form", default=InstructionForm.DEFAULT
)


def get_instruction_form() -> InstructionForm:
    """Return the instruction form selected for the current execution.

    Performs a non-creating read of the scoped selection.  Outside any explicit
    or inherited :func:`instruction_form_scope`, returns
    :attr:`InstructionForm.DEFAULT`.  Does not touch or create mutable run state.

    Returns:
        The active :class:`InstructionForm`.
    """
    return _instruction_form.get()


@contextmanager
def instruction_form_scope(
    form: InstructionForm = InstructionForm.DEFAULT,
) -> Iterator[None]:
    """Select *form* for the enclosed execution and restore the previous form on exit.

    The token belongs to the context that entered the scope and is reset there,
    normally in the same task.  A child task deliberately created inside the
    scope inherits the selection; resetting the parent's binding does not clear
    an inherited child context.

    Args:
        form: The :class:`InstructionForm` to select.  Defaults to
            :attr:`InstructionForm.DEFAULT`.

    Raises:
        ValueError: If *form* is not a valid :class:`InstructionForm`.  Raised
            at scope entry, before the body runs, leaving any enclosing
            selection unchanged.
    """
    token = _instruction_form.set(InstructionForm(form))
    try:
        yield
    finally:
        _instruction_form.reset(token)
