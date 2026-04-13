"""Argument substitution for skill prompt templates.

Provides :func:`substitute_args`, which replaces ``$arg_name``, ``$N`` positional,
and ``${KEY}`` / ``$KEY`` variable patterns in a prompt template string.

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
_BIDI_OVERRIDES_RE: re.Pattern[str] = re.compile(r"[\u202a-\u202e\u2066-\u2069]")

# YAML document-start and document-end boundary markers.
_YAML_BOUNDARY_RE: re.Pattern[str] = re.compile(r"\n---\n|\n\.\.\.\n")

# Chat-template role injection markers.
_ROLE_HEADER_RE: re.Pattern[str] = re.compile(r"\n(?:Human:|Assistant:|<\|im_start\|>)")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_value(value: str, label: str) -> None:
    """Validate that *value* is safe to substitute into a prompt template.

    Checks for content that could be used to inject instructions, override
    the role structure, or smuggle control characters into a downstream model
    prompt.

    Args:
        value: The candidate substitution string.
        label: A short identifier used in error messages (e.g. ``"$1"`` or
            ``"${RUN_ID}"``).

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
            " override character (U+202A\u2013U+202E or U+2066\u2013U+2069)."
        )
    if _ROLE_HEADER_RE.search(value):
        raise ValueError(
            f"Substitution value for {label!r} contains a role/system header sequence"
            r" ('\nHuman:', '\nAssistant:', or '\n<|im_start|>')."
        )


def _literal_replacer(value: str) -> Callable[[re.Match[str]], str]:
    """Return a :func:`re.sub` replacement callable that always yields *value* literally.

    Using a callable avoids backslash interpretation that would occur if *value*
    were passed directly as the ``repl`` string argument to :func:`re.sub`.

    Args:
        value: The literal replacement string.

    Returns:
        A callable suitable for the *repl* argument of :func:`re.sub`.
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
    arg_names: list[str],
    variables: dict[str, str] | None = None,
) -> str:
    """Replace argument, positional, and variable patterns in *template*.

    Applies substitutions in three ordered passes:

    1. **Named args** — whitespace-split *raw_args* are mapped positionally to
       *arg_names* and each ``$name`` token (with identifier-boundary guard) is
       replaced.  ``$hostname`` is *not* replaced when an arg_name is ``"host"``.
    2. **Positional fallback** — ``$1``, ``$2``, ... are replaced with the
       corresponding whitespace-split parts of *raw_args*.
    3. **Caller-supplied variables** — ``${KEY}`` and ``$KEY``
       (identifier-bounded) patterns are replaced with the matching value from
       *variables*.  A ``DEBUG``-level log entry is emitted for any ``${KEY}``
       pattern whose key is absent from *variables*; the pattern is left
       unchanged.

    Each substituted value is validated before use; see *Raises* for details.
    Unknown ``$pattern`` tokens that do not match any arg name, positional index,
    or variable key are left unchanged.

    Args:
        template: The prompt template string, potentially containing ``$name``,
            ``$N``, ``${KEY}``, or ``$KEY`` patterns.
        raw_args: A whitespace-separated string of argument values.  Empty or
            whitespace-only input disables named and positional substitution.
        arg_names: Ordered list of argument names that *raw_args* parts map to.
            Pass an empty list to skip named substitution; positional ``$N``
            substitution still occurs.
        variables: Optional mapping of variable names to replacement values.
            Both ``${KEY}`` and ``$KEY`` (identifier-bounded) forms are replaced.
            Pass ``None`` to skip variable substitution entirely.

    Returns:
        The template string after all applicable substitutions.

    Raises:
        ValueError: If any substituted value contains null bytes
            (``\\x00``), YAML frontmatter boundary sequences
            (``\\n---\\n`` or ``\\n...\\n``), Unicode bidirectional override
            characters (U+202A\u2013U+202E or U+2066\u2013U+2069), or role/system
            header sequences (``\\nHuman:``, ``\\nAssistant:``, or
            ``\\n<|im_start|>``).

    Example::

        >>> substitute_args("Connect to $host on $port", "myserver 8080", ["host", "port"])
        'Connect to myserver on 8080'

        >>> substitute_args(
        ...     "Run on $1 with ${RUN_ID}",
        ...     "prod",
        ...     [],
        ...     variables={"RUN_ID": "abc123"},
        ... )
        'Run on prod with abc123'
    """
    # Split raw_args into positional parts; empty/whitespace-only yields no parts.
    parts: list[str] = raw_args.split() if raw_args.strip() else []

    # ------------------------------------------------------------------
    # Validate all arg values before any substitution.
    # ------------------------------------------------------------------
    for i, val in enumerate(parts):
        _validate_value(val, f"${i + 1}")

    # ------------------------------------------------------------------
    # Validate variable values for keys referenced in the original template.
    # ------------------------------------------------------------------
    if variables is not None:
        for key, val in variables.items():
            braced_present = f"${{{key}}}" in template
            unbraced_present = bool(
                re.search(r"\$" + re.escape(key) + r"(?![a-zA-Z0-9_])", template)
            )
            if braced_present or unbraced_present:
                _validate_value(val, f"${{{key}}}")

    # ------------------------------------------------------------------
    # Step 1: Named arg substitution ($arg_name with identifier boundary).
    # ------------------------------------------------------------------
    if arg_names and parts:
        for i, name in enumerate(arg_names):
            if i < len(parts):
                template = re.sub(
                    r"\$" + re.escape(name) + r"(?![a-zA-Z0-9_])",
                    _literal_replacer(parts[i]),
                    template,
                )

    # ------------------------------------------------------------------
    # Step 2: Positional fallback ($1, $2, ...).
    # ------------------------------------------------------------------
    if parts:
        for i, val in enumerate(parts):
            template = re.sub(
                r"\$" + str(i + 1) + r"(?!\d)",
                _literal_replacer(val),
                template,
            )

    # ------------------------------------------------------------------
    # Step 3: Caller-supplied variable substitution.
    # ------------------------------------------------------------------
    if variables is not None:
        # Emit DEBUG for any ${KEY} patterns in the (partially-substituted)
        # template whose key is absent from variables.
        for key in re.findall(r"\$\{([^}]+)\}", template):
            if key not in variables:
                logger.debug(
                    "Template contains ${%s} but key not in variables; left unreplaced.",
                    key,
                )

        for key, val in variables.items():
            # Braced form: ${KEY}.
            template = template.replace(f"${{{key}}}", val)
            # Unbraced form: $KEY with identifier boundary.
            template = re.sub(
                r"\$" + re.escape(key) + r"(?![a-zA-Z0-9_])",
                _literal_replacer(val),
                template,
            )

    return template
