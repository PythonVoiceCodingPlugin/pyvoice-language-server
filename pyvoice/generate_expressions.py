import functools
from typing import Optional, Sequence

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
from pyvoice.types import ExpressionItem, ExpressionSettings

__all__ = [
    "generate_nested",
    "into_item",
    "with_prefix",
    "get_expressions",
]


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


default_levels = {"module": 1, "instance": 2, "variable": 2, "param": 2, "statement": 2}


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
) -> Sequence[str]:
    return list(_generate_nested(name, prefix, level, project))


def _generate_nested(
    name: jedi.api.classes.Name,
    prefix: str,
    level: Optional[int] = None,
    project: Optional[Project] = None,
):
    if level is None:
        level = default_levels.get(name.type, 1)
    if level <= 0:
        return
    if name.type == "module":
        if name.name == "pytest":
            level += 1
        for n in module_public_names(project, name.full_name):
            yield with_prefix(prefix, n)
            yield from _generate_nested(n, prefix, level - 1)
    elif name.type == "instance":
        for n in instance_attributes(name.full_name, project):
            yield with_prefix(prefix, n)
            if (
                n.type in ["instance", "variable", "statement", "param"]
                and not name.name.startswith("_")
                and not n.name.startswith("_")
                and True
            ):
                yield from _generate_nested(n, f"{prefix}.{n.name}", level - 1, project)
    elif name.type in ["variable", "statement", "param"]:
        for n in name.infer():
            yield from _generate_nested(n, prefix, level, project)
    elif name.type == "function":
        return
    elif name.type == "class" and hasattr(name, "defined_names"):
        for n in name.defined_names():
            if n.type == "statement":
                yield with_prefix(prefix, n)


def get_expressions(
    script: jedi.api.Script, settings: ExpressionSettings, pos: Optional[Position]
) -> Sequence[ExpressionItem]:
    global_names = script.get_names()
    output = []
    for n in global_names:
        output.append(with_prefix("", n))
        output.extend(
            generate_nested(
                n,
                n.name if n.type != "function" else "",
                None,
                script._inference_state.project,
            )
        )
        output.extend(into_item(k) for k in get_keyword_names(n))
    if pos:
        containing_scopes = list(get_scopes(script, pos))
        for scope in containing_scopes:
            if scope.type == "function":
                for n in scope.defined_names():
                    output.append(with_prefix("", n))
                    output.extend(
                        generate_nested(
                            n,
                            n.name if n.type != "function" else "",
                            None,
                            script._inference_state.project,
                        )
                    )
    output = [x for x in set(output) if "__" not in x.value]
    if len(output) < 2000:
        output = output[:2000]
    return output
