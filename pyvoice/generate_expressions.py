import functools
import logging
from itertools import chain
from typing import Iterable, Mapping, Optional, Sequence

import jedi
from cachetools import LRUCache, cached
from lsprotocol.types import Position

from pyvoice.custom_jedi_classes import Project
from pyvoice.inference import (
    get_keyword_names,
    get_scopes,
    instance_attributes,
    module_public_names,
)
from pyvoice.speakify import speak_single_item
from pyvoice.types import ExpressionItem, ExpressionSettings, ScopeSettings

__all__ = [
    "generate_nested",
    "into_item",
    "with_prefix",
    "get_expressions",
]

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1024 * 8)
def into_item(value: str) -> ExpressionItem:
    spoken = speak_single_item(value)
    return ExpressionItem(value=value, spoken=spoken)


@functools.lru_cache(maxsize=512)
def with_prefix(prefix: str, name: jedi.api.classes.BaseName) -> ExpressionItem:
    if prefix:
        prefix = prefix + "."
    n = name.name
    if name.type == "function":
        n = n + "()"
    return into_item(f"{prefix}{n}")


default_levels: Mapping[str, int] = {
    "module": 1,
    "instance": 2,
    "variable": 2,
    "param": 2,
    "statement": 2,
}


@cached(
    cache=LRUCache(maxsize=512 * 4),
    key=lambda name, prefix, level, project: (
        name.full_name,
        prefix,
        level,
        project.path,
    ),
)
def generate_nested(
    name: jedi.api.classes.Name,
    prefix: str,
    level: Optional[int] = None,
    project: Optional[Project] = None,
) -> Sequence[ExpressionItem]:
    return list(_generate_nested(name, prefix, level, project))


def _generate_nested(
    name: jedi.api.classes.Name,
    prefix: str,
    level: Optional[int] = None,
    project: Optional[Project] = None,
    forwarded: bool = False,
) -> Iterable[ExpressionItem]:
    if level is None:
        level = default_levels.get(name.type, 1)
        yield from generate_nested(name, prefix, level, project)
    else:
        if level <= 0:
            return
        if name.type == "module":
            if name.name == "pytest":
                level += 1
            for n in module_public_names(project, name.full_name):
                yield with_prefix(prefix, n)
                yield from _generate_nested(n, prefix, level - 1, project)
        elif name.type == "instance":
            for n in instance_attributes(name.full_name, project):
                yield with_prefix(prefix, n)
                if (
                    n.type in ["instance", "variable", "statement", "param", "property"]
                    and (not name.name.startswith("_") or forwarded)
                    and not n.name.startswith("_")
                ):
                    yield from _generate_nested(
                        n, f"{prefix}.{n.name}", level - 1, project
                    )
        elif name.type in ["variable", "statement", "param", "property"]:
            for n in name.infer():
                yield from _generate_nested(
                    n,
                    prefix,
                    level,
                    project,
                    forwarded=(not name.name.startswith("_") or prefix == "self"),
                )
        elif name.type == "function":
            return
        elif name.type == "class" and hasattr(name, "defined_names"):
            for n in name.defined_names():
                if n.type == "statement":
                    yield with_prefix(prefix, n)


@cached(cache=LRUCache(maxsize=4))
def _get_expressions_from_builtins(
    project: Project, settings: ScopeSettings
) -> Sequence[ExpressionItem]:
    if not settings.enabled:
        return []
    return [
        with_prefix("", x)
        for x in project.get_script(code="").complete()
        if not x.name.startswith("_") and not x.type == "keyword"
    ]


def _get_expressions_from_scope(
    scope: jedi.api.classes.Name, project: Project, settings: ScopeSettings
) -> Sequence[ExpressionItem]:
    if not settings.enabled:
        return []
    output = []
    for n in scope.defined_names():
        output.append(with_prefix("", n))
        output.extend(
            generate_nested(
                n,
                n.name if n.type != "function" else "",
                None,
                project,
            )
        )
        if settings.signature:
            output.extend(into_item(k) for k in get_keyword_names(n))
    return output


def get_expressions(
    script: jedi.api.Script, settings: ExpressionSettings, pos: Position
) -> Sequence[ExpressionItem]:
    project = script._inference_state.project
    containing_scopes = list(get_scopes(script, pos))

    try:
        local_scope, *non_local_scopes, global_scope = containing_scopes
    except ValueError:
        local_scope = None
        non_local_scopes = []
        global_scope = containing_scopes[0]

    expressions_from_builtins = _get_expressions_from_builtins(
        project, settings.builtins
    )
    expressions_from_locals = (
        _get_expressions_from_scope(local_scope, project, settings.locals)
        if local_scope
        else []
    )
    expressions_from_globals = _get_expressions_from_scope(
        global_scope, project, settings.globals
    )
    expressions_from_non_locals = list(
        chain.from_iterable(
            _get_expressions_from_scope(scope, project, settings.nonlocals)
            for scope in non_local_scopes
            if scope.type == "function"
        )
    )
    logger.debug(
        "Found %s local expressions, %s global expressions,"
        " %s builtin expressions, %s nonlocal expressions",
        len(expressions_from_locals),
        len(expressions_from_globals),
        len(expressions_from_builtins),
        len(expressions_from_non_locals),
    )
    output = list(
        x
        for x in chain(
            expressions_from_builtins,
            expressions_from_locals,
            expressions_from_globals,
            expressions_from_non_locals,
        )
        if "__" not in x.value
    )

    if len(output) > settings.limit:
        output = output[: settings.limit]
    return output
