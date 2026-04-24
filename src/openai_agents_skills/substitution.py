"""Argument substitution for skill prompt templates.

Provides :func:`substitute_args`, which replaces ``$ARGUMENTS`` and ``${KEY}``
variable patterns in a prompt template string.

Unsafe values (null bytes, YAML boundaries, bidirectional overrides, role headers)
are rejected before substitution to prevent prompt-injection attacks.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled patterns used for validation
# ---------------------------------------------------------------------------

# U+202A–U+202E (LRE, RLE, PDF, LRO, RLO) and U+2066–U+2069 (LRI, RLI, FSI, PDI).
_BIDI_OVERRIDES_RE: re.Pattern[str] = re.compile(r"[‪-‮⁦-⁩]")

# YAML document-start and document-end boundary markers.
_YAML_BOUNDARY_RE: re.Pattern[str] = re.compile(r"\n---\n|\n\.\.\.\n")

# Chat-template role injection markers.
_ROLE_HEADER_RE: re.Pattern[str] = re.compile(r"\n(?:Human:|Assistant:|<\|im_start\|>)")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_value(value: str, label: str) -> None:
    """Validate that *value* is safe to substitute into a prompt template.

    Args:
        value: The candidate substitution string.
        label: A short identifier used in error messages.

    Raises:
        ValueError: When *value* contains null bytes, YAML frontmatter boundary
            sequences, Unicode bidirectional override characters, or role/system
            header sequences.
    """
    if "\x00" in value:
        raise ValueError(f"Substitution value for {label!r} contains a null byte (\\x00).")
    if _YAML_BOUNDARY_RE.search(value):
        raise ValueError(
            f"Substitution value for {label!r} contains a YAML frontmatter boundary"
            r" ('\n---\n' or '\n...\n')."
        )
    if _BIDI_OVERRIDES_RE.search(value):
        raise ValueError(
            f"Substitution value for {label!r} contains a Unicode bidirectional"
            " override character (U+202A–U+202E or U+2066–U+2069)."
        )
    if _ROLE_HEADER_RE.search(value):
        raise ValueError(
            f"Substitution value for {label!r} contains a role/system header sequence"
            r" ('\nHuman:', '\nAssistant:', or '\n<|im_start|>')."
        )


def _literal_replacer(value: str) -> Callable[[re.Match[str]], str]:
    """Return a replacement callable that always yields *value* literally.

    Avoids backslash interpretation that would occur if *value* were passed
    directly as the ``repl`` string to :func:`re.sub`.
    """

    def _replace(_match: re.Match[str]) -> str:
        return value

    return _replace


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def substitute_args(
    template: str,
    raw_args: str,
    variables: dict[str, str] | None = None,
) -> str:
    """Replace ``$ARGUMENTS`` and variable patterns in *template*.

    Applies substitutions in two ordered passes:

    1. **Arguments** — if ``$ARGUMENTS`` appears in *template*, it is replaced
       with *raw_args*.  If ``$ARGUMENTS`` is absent and *raw_args* is non-empty,
       the arguments are appended to the template as ``ARGUMENTS: <value>``.
    2. **Caller-supplied variables** — ``${KEY}`` and ``$KEY``
       (identifier-bounded) patterns are replaced with values from *variables*.
       A ``DEBUG``-level log entry is emitted for any ``${KEY}`` whose key is
       absent from *variables*; the pattern is left unchanged.

    Each substituted value is validated before use; see *Raises* for details.

    Args:
        template: The prompt template string, potentially containing
            ``$ARGUMENTS`` or ``${KEY}`` / ``$KEY`` patterns.
        raw_args: The full argument string passed by the caller.  Empty or
            whitespace-only input means no argument substitution occurs.
        variables: Optional mapping of variable names to replacement values.
            Pass ``None`` to skip variable substitution entirely.

    Returns:
        The template string after all applicable substitutions.

    Raises:
        ValueError: If any substituted value contains null bytes, YAML
            frontmatter boundary sequences, Unicode bidirectional override
            characters, or role/system header sequences.

    Example::

        >>> substitute_args("Fix issue $ARGUMENTS", "42")
        'Fix issue 42'

        >>> substitute_args("Deploy to $ARGUMENTS in ${ENV}", "prod", {"ENV": "us-east-1"})
        'Deploy to prod in us-east-1'

        >>> substitute_args("Run the workflow.", "extra context")
        'Run the workflow.\\n\\nARGUMENTS: extra context'
    """
    args_present = bool(raw_args.strip())

    if args_present:
        _validate_value(raw_args, "$ARGUMENTS")

    if variables is not None:
        for key, val in variables.items():
            braced_present = f"${{{key}}}" in template
            unbraced_present = bool(
                re.search(r"\$" + re.escape(key) + r"(?![a-zA-Z0-9_])", template)
            )
            if braced_present or unbraced_present:
                _validate_value(val, f"${{{key}}}")

    # ------------------------------------------------------------------
    # Step 1: $ARGUMENTS substitution.
    # ------------------------------------------------------------------
    if args_present:
        if "$ARGUMENTS" in template:
            template = template.replace("$ARGUMENTS", raw_args)
        else:
            template = f"{template}\n\nARGUMENTS: {raw_args}"

    # ------------------------------------------------------------------
    # Step 2: Caller-supplied variable substitution.
    # ------------------------------------------------------------------
    if variables is not None:
        for key in re.findall(r"\$\{([^}]+)\}", template):
            if key not in variables:
                logger.debug(
                    "Template contains ${%s} but key not in variables; left unreplaced.",
                    key,
                )

        for key, val in variables.items():
            template = template.replace(f"${{{key}}}", val)
            template = re.sub(
                r"\$" + re.escape(key) + r"(?![a-zA-Z0-9_])",
                _literal_replacer(val),
                template,
            )

    return template
