import functools
from typing import Optional, Sequence, Set

import jedi
from cachetools import LRUCache, cached
from lsprotocol.types import Position

from pyvoice.custom_jedi_classes import Project

__all__ = [
    "join_names",
    "generate_nested",
    "instance_attributes",
    "module_public_names",
    "get_keyword_names",
    "ignored_names",
    "with_prefix",
    "get_scopes",
    "pretty_scope_list",
    "module_public_names_fuzzy",
]


@functools.lru_cache(maxsize=512)
def with_prefix(prefix: str, name: jedi.api.classes.Name):
    if prefix:
        prefix = prefix + "."
    n = name.name
    if name.type == "function":
        n = n + "()"
    return f"{prefix}{n}"


default_levels = {"module": 1, "instance": 2, "variable": 2, "param": 2, "statement": 2}


@functools.lru_cache(maxsize=128)
def instance_attributes(
    full_name: str, project: Project
) -> Sequence[jedi.api.classes.BaseName]:
    if full_name is None:
        return []
    text = f"""
import {full_name.split('.')[0]}
_ : {full_name}
_."""
    small_script = project.get_script(code=text)
    return [
        x
        for x in small_script.complete()
        if "__" not in x.name and "leave" not in x.name and "visit" not in x.name
    ]


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
):
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
        # if "def " in name.get_line_code():
        #     for n in name.defined_names():
        #         yield with_prefix(prefix, n)
        #         yield from generate_nested(n, n.name, None, project)
    elif name.type == "class" and name.name.endswith("Targets"):
        #        return
        for n in name.defined_names():
            yield with_prefix(prefix, n)
        #     yield from generate_nested(n, prefix, level - 1)


@cached(cache=LRUCache(maxsize=512 * 4), key=lambda n: n.full_name)
def get_keyword_names(n: jedi.api.classes.Name):
    output = []
    for signature in n.get_signatures():
        output.extend(p.name for p in signature.params)
    return output


@functools.lru_cache()
def ignored_names(project: Project):
    return {x.full_name for x in jedi.Script("", project=project).complete()}


def _get_module__all__(names: Sequence[jedi.api.classes.Name]) -> Set[str]:
    try:
        all_name = next(x for x in names if x.name == "__all__")
        return set(
            x.data._get_payload()
            for literal_sequence in all_name.infer()
            for x in literal_sequence._name._value.py__iter__()
        )
    except (AttributeError, StopIteration):
        return {}


@functools.lru_cache(maxsize=128)
def module_public_names(
    project: Project,
    module_name: str,
) -> Sequence[jedi.api.classes.BaseName]:
    small_script = project.get_script(
        code=f"from {module_name} import \n",
    )
    completions = small_script.complete()

    module__all__ = _get_module__all__(completions)
    if module__all__ is not None:
        return [name for name in completions if name.name in module__all__]
    else:
        return [name for name in completions if not name.name.startswith("_")]


def module_public_names_fuzzy(
    project: Project, current_path: str, module_name: str, name: str
) -> Sequence[jedi.api.classes.BaseName]:
    ignore = ignored_names(project)
    return [
        name
        for name in jedi.Script(
            f"from {module_name} import *\n{name.replace(' ','')}",
            project=project,
            path=current_path,
        ).complete(fuzzy=True)
        if name.full_name not in ignore and name.full_name
    ]


def join_names(a: str, b: str) -> str:
    if a and b:
        return f"{a}.{b}"
    else:
        return f"{a or b}"


def get_scopes(script: jedi.Script, pos: Position):
    scope = script.get_context(pos.line + 1, None)
    while scope:
        yield scope
        scope = scope.parent()


def pretty_scope_list(containing_scopes):
    return " > ".join(
        x.description if x.type != "module" else "mod " + x.full_name
        for x in reversed(containing_scopes)
    )
