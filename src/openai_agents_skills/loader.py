"""File-based skill loading — FileSkill, SkillConfig, SkillSource, and loaders.

Skills stored as ``SKILL.md`` files are loaded from one or more base directories
and registered in a :class:`~openai_agents_skills.registry.SkillRegistry`.

Two layers are searched by default, in priority order (user wins on name conflict):

- **User layer** (personal, cross-repo): ``~/.agent/skills/<name>/SKILL.md``
- **Project layer** (repo-specific, checked in): ``<cwd>/.agent/skills/<name>/SKILL.md``

Additional directories may be supplied via :attr:`SkillConfig.extra_dirs`.

Usage::

    from pathlib import Path
    from openai_agents_skills import load_all_skills, SkillHooks
    from agents import Agent, Runner

    registry = await load_all_skills(cwd=Path.cwd())

    agent = Agent(
        name="Assistant",
        instructions="You are helpful.",
        hooks=SkillHooks(registry=registry),
    )

    result = await Runner.run(agent, "Run the payments workflow.")
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from .registry import SkillRegistry
from .skills import Skill
from .substitution import substitute_args

if TYPE_CHECKING:
    from agents import Agent, RunContextWrapper


_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Source trust levels
# ---------------------------------------------------------------------------


class SkillSource(StrEnum):
    """Trust level of a loaded :class:`FileSkill`.

    The source is recorded at load time and surfaces in log output.  Enforcement
    rules (e.g. restricting ``context: fork`` for ``EXTRA`` skills) are deferred
    to Phase 6.

    Attributes:
        BUNDLED: Compiled into the package — fully trusted.
        USER: Loaded from ``~/.agent/skills/`` — user-trusted.
        PROJECT: Loaded from ``<cwd>/.agent/skills/`` — project-trusted.
        EXTRA: Loaded from a caller-supplied extra directory — caller-trusted.
    """

    BUNDLED = "bundled"
    USER = "user"
    PROJECT = "project"
    EXTRA = "extra"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SkillConfig:
    """Configuration for :func:`load_all_skills`.

    Attributes:
        extra_dirs: Additional directories to load skills from, after the user
            and project layers.  Each directory is treated as a root containing
            ``<name>/SKILL.md`` subdirectories, identical to the user/project
            convention.
        user_dir: Override the user-layer base directory.  Defaults to
            ``~/.agent/skills/`` when ``None``.
        project_dir: Override the project-layer base directory.  Defaults to
            ``<cwd>/.agent/skills/`` when ``None``.
        variables: Mapping of variable names to values threaded through to
            :func:`~openai_agents_skills.substitution.substitute_args` for
            ``${KEY}`` substitution in skill bodies.  Common uses:
            ``{"RUN_ID": str(uuid.uuid4()), "USER": "alice"}``.

    Example::

        import uuid
        from pathlib import Path
        from openai_agents_skills import SkillConfig, load_all_skills

        config = SkillConfig(
            variables={"RUN_ID": str(uuid.uuid4()), "USER": "alice"},
        )
        registry = await load_all_skills(cwd=Path.cwd(), config=config)
    """

    extra_dirs: list[Path] = field(default_factory=list)
    user_dir: Path | None = None
    project_dir: Path | None = None
    variables: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal: parsed frontmatter fields
# ---------------------------------------------------------------------------


@dataclass
class _SkillFields:
    """Parsed frontmatter metadata for a SKILL.md file.

    All fields map directly to their YAML frontmatter counterparts.  The loader
    populates this struct before constructing a :class:`FileSkill`.
    """

    name: str
    description: str
    always_on: bool = False
    allowed_tools: list[str] = field(default_factory=list)
    argument_hint: str = ""
    user_invocable: bool = True
    license_: str = ""
    compatibility: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# FileSkill
# ---------------------------------------------------------------------------


class FileSkill(Skill):
    """A concrete :class:`~openai_agents_skills.skills.Skill` loaded from a SKILL.md file.

    ``get_prompt_blocks`` returns the Markdown body (after the frontmatter block)
    as a single user-role message.  ``$arg_name`` positional substitutions and
    caller-supplied ``${VAR}`` variable substitutions are applied on each call via
    :func:`~openai_agents_skills.substitution.substitute_args`.

    Results are cached by the ``args`` string so that repeated calls with the same
    arguments avoid redundant string work.  The cache is instance-local and never
    invalidated — reload the registry to pick up on-disk changes.

    Attributes:
        source: The :class:`SkillSource` trust level this skill was loaded from.
        file_path: Absolute path to the SKILL.md file on disk.

    Example::

        # Typically created by load_skills_from_dir; shown here for clarity.
        from pathlib import Path
        from openai_agents_skills.loader import FileSkill, _SkillFields

        fields = _SkillFields(name="greet", description="Greets the user.")
        skill = FileSkill(fields=fields, body="Say hello!", file_path=Path("greet/SKILL.md"))
        blocks = await skill.get_prompt_blocks()
        # [{"role": "user", "content": "Say hello!"}]
    """

    def __init__(
        self,
        fields: _SkillFields,
        body: str,
        file_path: Path,
        source: SkillSource = SkillSource.PROJECT,
        variables: dict[str, str] | None = None,
    ) -> None:
        """Initialise a FileSkill from parsed frontmatter and body text.

        Args:
            fields: Parsed frontmatter metadata.
            body: Markdown body text (after the frontmatter block), possibly
                containing ``$arg_name``, ``$N``, or ``${VAR}`` patterns.
            file_path: Path to the source SKILL.md file, stored for diagnostics.
            source: Trust level of the directory this file was loaded from.
            variables: Caller-supplied variable mapping forwarded to
                :func:`~openai_agents_skills.substitution.substitute_args`.
        """
        self.name = fields.name
        self.description = fields.description
        self.always_on = fields.always_on
        self.allowed_tools: list[str] = list(fields.allowed_tools)
        self.user_invocable: bool = fields.user_invocable
        self.license_: str = fields.license_
        self.compatibility: str = fields.compatibility
        self.metadata: dict[str, str] = dict(fields.metadata)
        self.source = source
        self.file_path = file_path
        self._body = body
        self._variables: dict[str, str] = dict(variables) if variables else {}
        self._cache: dict[str, list[Any]] = {}

    async def get_prompt_blocks(
        self,
        context: RunContextWrapper[Any] | None,
        agent: Agent[Any] | None,
        args: str = "",
    ) -> list[Any]:
        """Return the skill body as a user-role prompt block.

        Applies :func:`~openai_agents_skills.substitution.substitute_args` to the
        stored body before wrapping it.  Results are cached by *args* — the same
        list object is returned on repeated calls with identical arguments (useful
        for identity checks in tests).

        Args:
            args: Optional whitespace-separated argument values.  Passed to
                :func:`~openai_agents_skills.substitution.substitute_args` for
                ``$arg_name`` / ``$N`` substitution.

        Returns:
            A one-element list containing a ``{"role": "user", "content": <body>}``
            dict.

        Raises:
            ValueError: If any argument or variable value contains an unsafe
                pattern (see
                :func:`~openai_agents_skills.substitution.substitute_args`).
        """
        if args not in self._cache:
            body = substitute_args(self._body, args, self._variables)
            self._cache[args] = [{"role": "user", "content": body}]
        return self._cache[args]


# ---------------------------------------------------------------------------
# Path traversal guard
# ---------------------------------------------------------------------------


def assert_within_base(path: Path, base: Path) -> None:
    """Assert that *path* resolves to a descendant of *base*.

    Resolves both paths to their canonical (symlink-free) forms before
    comparing so that symbolic links cannot be used to escape the base
    directory.

    Args:
        path: Candidate path to validate.
        base: Expected root directory that *path* must be inside.

    Raises:
        ValueError: If the resolved *path* is not a descendant of the resolved
            *base*.
    """
    resolved = path.resolve()
    resolved_base = base.resolve()
    if not resolved.is_relative_to(resolved_base):
        raise ValueError(
            f"Path {path!r} resolves to {resolved!r}, which escapes base directory"
            f" {resolved_base!r}."
        )


# ---------------------------------------------------------------------------
# Internal: low-level I/O and parsing helpers
# ---------------------------------------------------------------------------


async def _read_skill_file(path: Path) -> str:
    """Read *path* without blocking the event loop.

    Wraps :meth:`Path.read_text` in :func:`asyncio.to_thread` so the calling
    coroutine yields control to other tasks while I/O is in progress.

    Args:
        path: Path to the file to read.

    Returns:
        The full text content of the file, decoded as UTF-8.
    """
    return await asyncio.to_thread(path.read_text, encoding="utf-8")


_SKILL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*[a-z0-9]$|^[a-z0-9]$")


def _validate_skill_name(name: str) -> None:
    """Validate a skill name per the agentskills.io specification.

    Rules: 1-64 chars, lowercase letters/digits/hyphens only, no leading or
    trailing hyphen, no consecutive hyphens.

    Raises:
        ValueError: If the name violates any rule.
    """
    if len(name) > 64:
        raise ValueError(f"Skill name exceeds 64 characters: {name!r}")
    if not _SKILL_NAME_RE.match(name):
        raise ValueError(
            f"Skill name {name!r} is invalid: must contain only lowercase letters,"
            " numbers, and hyphens, and must not start or end with a hyphen."
        )
    if "--" in name:
        raise ValueError(f"Skill name {name!r} contains consecutive hyphens.")


def _check_no_path_seps(value: str, field_name: str) -> None:
    """Reject frontmatter values that contain path-traversal characters.

    Args:
        value: The frontmatter field value to check.
        field_name: Human-readable field name used in the error message.

    Raises:
        ValueError: If *value* contains ``/``, ``\\``, or ``..``.
    """
    if "/" in value or "\\" in value or ".." in value:
        raise ValueError(
            f"Frontmatter field {field_name!r} contains invalid path characters: {value!r}"
        )


def _parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Split a SKILL.md file into a frontmatter dict and a body string.

    Expects content of the form::

        ---
        key: value
        ---
        Body text here.

    The leading ``---\\n`` delimiter and the closing ``\\n---`` delimiter are
    consumed.  The body begins at the first character after the closing delimiter
    (and its trailing newline, if present).

    Args:
        content: Full text content of a SKILL.md file.

    Returns:
        A ``(fields_dict, body)`` tuple.  Returns ``({}, content)`` when the
        content does not begin with ``---\\n``, the closing delimiter is missing,
        or the YAML cannot be parsed.
    """
    if not content.startswith("---\n"):
        return {}, content

    # Locate the closing '---' delimiter (must be preceded by a newline).
    rest = content[4:]  # skip the opening '---\n'
    close_idx = rest.find("\n---")
    if close_idx == -1:
        return {}, content

    fm_text = rest[:close_idx]
    after_sep = rest[close_idx + 4 :]  # characters after '\n---'
    # Consume an optional newline immediately after the closing '---'.
    body = after_sep[1:] if after_sep.startswith("\n") else after_sep

    try:
        data = yaml.safe_load(fm_text)
    except yaml.YAMLError as exc:
        _log.debug("YAML parse error in frontmatter: %s", exc)
        return {}, content

    if not isinstance(data, dict):
        return {}, content

    return data, body


def _parse_skill_file(content: str, dir_name: str) -> tuple[_SkillFields, str] | None:
    """Parse a SKILL.md file's content into structured metadata and a body string.

    Args:
        content: Full text of the SKILL.md file.
        dir_name: Name of the containing directory, used as the default skill
            name when the ``name`` frontmatter field is absent.

    Returns:
        A ``(_SkillFields, body)`` tuple on success, or ``None`` when the file
        has no valid ``description`` field (required) or contains invalid
        frontmatter values.

    Raises:
        ValueError: If the ``name`` frontmatter field or any entry in
            ``arguments`` contains path-separator or ``..`` characters.
    """
    fields_dict, body = _parse_frontmatter(content)

    desc_raw = fields_dict.get("description")
    description = str(desc_raw).strip() if desc_raw is not None else ""
    if not description:
        _log.debug(
            "Skipping skill directory %r: 'description' frontmatter field is missing or empty.",
            dir_name,
        )
        return None
    if len(description) > 1024:
        raise ValueError(f"Skill in {dir_name!r}: 'description' exceeds 1024 characters.")

    raw_name = fields_dict.get("name", dir_name)
    name = str(raw_name).strip() if raw_name else dir_name
    if not name:
        name = dir_name
    _validate_skill_name(name)
    if "name" in fields_dict and name != dir_name:
        _log.debug("Skill name %r does not match directory name %r.", name, dir_name)

    allowed_tools: list[str] = []
    if "allowed-tools" in fields_dict:
        tools_value = fields_dict["allowed-tools"]
        if isinstance(tools_value, str):
            allowed_tools = tools_value.split()
        elif isinstance(tools_value, list):
            allowed_tools = [str(t) for t in tools_value]
        else:
            _log.debug("Skill %r: 'allowed-tools' is not a string or list; ignoring.", name)

    user_invocable_raw = fields_dict.get("user-invocable", True)
    user_invocable = bool(user_invocable_raw)

    license_ = str(fields_dict.get("license", "")).strip()

    compatibility = str(fields_dict.get("compatibility", "")).strip()
    if len(compatibility) > 500:
        raise ValueError(f"Skill {name!r}: 'compatibility' exceeds 500 characters.")

    metadata: dict[str, str] = {}
    if "metadata" in fields_dict:
        meta_value = fields_dict["metadata"]
        if isinstance(meta_value, dict):
            metadata = {str(k): str(v) for k, v in meta_value.items()}
        else:
            _log.debug("Skill %r: 'metadata' field is not a mapping; ignoring.", name)

    skill_fields = _SkillFields(
        name=name,
        description=description,
        always_on=bool(fields_dict.get("always-on", False)),
        allowed_tools=allowed_tools,
        argument_hint=str(fields_dict.get("argument-hint", "")).strip(),
        user_invocable=user_invocable,
        license_=license_,
        compatibility=compatibility,
        metadata=metadata,
    )
    return skill_fields, body.strip()


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------


async def load_skills_from_dir(
    dir_path: Path,
    source: SkillSource,
    variables: dict[str, str] | None = None,
) -> list[tuple[FileSkill, Path]]:
    """Load all ``SKILL.md`` skills from immediate subdirectories of *dir_path*.

    Each immediate subdirectory that contains a ``SKILL.md`` file is treated as
    one skill.  Subdirectories without a ``SKILL.md`` file are silently skipped.
    A missing or non-directory *dir_path* is also silently skipped (no error on
    a fresh project that has not yet created ``.agent/skills/``).

    File reads are performed concurrently via :func:`asyncio.gather` and wrapped
    in :func:`asyncio.to_thread` so the event loop is never blocked.

    Args:
        dir_path: Root directory to scan for ``<name>/SKILL.md`` subdirectories.
        source: Trust level to record on every :class:`FileSkill` loaded from
            this directory.
        variables: Optional variable mapping forwarded to
            :func:`~openai_agents_skills.substitution.substitute_args` for
            ``${KEY}`` substitution in skill bodies.

    Returns:
        A list of ``(FileSkill, canonical_path)`` pairs, where
        ``canonical_path`` is the resolved absolute path of the ``SKILL.md``
        file (used for deduplication by the caller).  The list preserves
        filesystem iteration order.
    """
    if not dir_path.exists() or not dir_path.is_dir():
        return []

    # Collect candidate SKILL.md paths from immediate subdirectories.
    candidates: list[tuple[str, Path]] = []
    for subdir in sorted(dir_path.iterdir()):
        if not subdir.is_dir():
            continue
        skill_file = subdir / "SKILL.md"
        if not skill_file.exists():
            continue
        candidates.append((subdir.name, skill_file))

    if not candidates:
        return []

    # Read all files concurrently.
    contents = await asyncio.gather(
        *[_read_skill_file(skill_file) for _, skill_file in candidates],
        return_exceptions=True,
    )

    results: list[tuple[FileSkill, Path]] = []
    for (dir_name, skill_file), content_or_exc in zip(candidates, contents):
        if isinstance(content_or_exc, BaseException):
            _log.warning(
                "Failed to read %s: %s",
                skill_file,
                content_or_exc,
            )
            continue

        content: str = content_or_exc

        # Path traversal guard: ensure the resolved path stays inside dir_path.
        try:
            assert_within_base(skill_file, dir_path)
        except ValueError as exc:
            _log.warning("Path traversal rejected for %s: %s", skill_file, exc)
            continue

        try:
            parsed = _parse_skill_file(content, dir_name)
        except ValueError as exc:
            _log.warning("Invalid frontmatter in %s: %s", skill_file, exc)
            continue

        if parsed is None:
            # Missing required field; already logged by _parse_skill_file.
            continue

        skill_fields, body = parsed
        skill = FileSkill(
            fields=skill_fields,
            body=body,
            file_path=skill_file,
            source=source,
            variables=variables,
        )
        canonical = skill_file.resolve()
        results.append((skill, canonical))
        _log.debug(
            "Loaded skill %r from %s (source=%s)",
            skill.name,
            skill_file,
            source.value,
        )

    return results


async def load_all_skills(
    cwd: Path,
    config: SkillConfig | None = None,
) -> SkillRegistry:
    """Load skills from user, project, and optional extra directories.

    Directories are resolved in priority order:

    1. **User layer** — ``~/.agent/skills/`` (or ``config.user_dir``).
    2. **Project layer** — ``<cwd>/.agent/skills/`` (or ``config.project_dir``).
    3. **Extra layers** — each directory in ``config.extra_dirs``, in order.

    Deduplication rules applied in this order:

    - Skills with the same **canonical file path** (resolved via
      :func:`Path.resolve`) are loaded only once, regardless of how many
      directories point to the same file (e.g. via symbolic links).
    - Skills with the same **name** are deduplicated by priority: user-layer
      skills win over project-layer skills; earlier extra dirs win over later
      ones.

    The returned :class:`~openai_agents_skills.registry.SkillRegistry` is
    ready to pass to
    :class:`~openai_agents_skills.hooks.SkillHooks` or
    :class:`~openai_agents_skills.hooks.RunSkillHooks`.

    Args:
        cwd: Working directory used to resolve the default project-layer path
            (``<cwd>/.agent/skills/``).
        config: Optional loader configuration.  Pass ``None`` to use all
            defaults (user and project dirs only, no extra dirs, no variables).

    Returns:
        A populated :class:`~openai_agents_skills.registry.SkillRegistry`.

    Example::

        from pathlib import Path
        from openai_agents_skills import load_all_skills, SkillHooks
        from agents import Agent

        registry = await load_all_skills(cwd=Path.cwd())
        agent = Agent(
            name="Assistant",
            instructions="You are helpful.",
            hooks=SkillHooks(registry=registry),
        )
    """
    cfg = config or SkillConfig()
    variables: dict[str, str] = cfg.variables

    user_dir = cfg.user_dir if cfg.user_dir is not None else Path.home() / ".agent" / "skills"
    project_dir = cfg.project_dir if cfg.project_dir is not None else cwd / ".agent" / "skills"

    # Build the ordered list of (directory, source) pairs.
    dirs: list[tuple[Path, SkillSource]] = [
        (user_dir, SkillSource.USER),
        (project_dir, SkillSource.PROJECT),
        *((d, SkillSource.EXTRA) for d in cfg.extra_dirs),
    ]

    # Load all directories concurrently.
    groups = await asyncio.gather(
        *[load_skills_from_dir(d, source, variables) for d, source in dirs],
        return_exceptions=True,
    )

    registry = SkillRegistry()
    seen_paths: set[Path] = set()
    seen_names: set[str] = set()

    for group_or_exc in groups:
        if isinstance(group_or_exc, BaseException):
            _log.warning("Error loading skill directory: %s", group_or_exc)
            continue

        group: list[tuple[FileSkill, Path]] = group_or_exc
        for skill, canonical_path in group:
            # Deduplicate by canonical file path (symlink-safe).
            if canonical_path in seen_paths:
                _log.debug(
                    "Skipping duplicate path %s (already loaded as a different dir entry).",
                    canonical_path,
                )
                continue
            seen_paths.add(canonical_path)

            # Deduplicate by skill name (higher-priority layers win).
            if skill.name in seen_names:
                _log.debug(
                    "Skipping skill %r from %s: a higher-priority layer already registered"
                    " a skill with this name.",
                    skill.name,
                    skill.file_path,
                )
                continue
            seen_names.add(skill.name)

            registry.register(skill)

    return registry
