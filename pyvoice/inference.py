import functools
import logging
from typing import Optional, Sequence, Set, cast

import jedi
from cachetools import LRUCache, cached
from lsprotocol.types import Position

from pyvoice.custom_jedi_classes import Project

logger = logging.getLogger(__name__)
__all__ = [
    "join_names",
    "instance_attributes",
    "module_public_names",
    "get_keyword_names",
    "ignored_names",
    "get_scopes",
    "pretty_scope_list",
    "module_public_names_fuzzy",
]


@functools.lru_cache(maxsize=128)
def instance_attributes(
    full_name: Optional[str], project: Project
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


@cached(cache=LRUCache(maxsize=512 * 4), key=lambda n: n.full_name)
def get_keyword_names(n: jedi.api.classes.BaseName) -> Set[str]:
    return {
        cast(str, p.name) for signature in n.get_signatures() for p in signature.params
    }


@functools.lru_cache()
def ignored_names(project: Project):
    return {x.full_name for x in jedi.Script("", project=project).complete()}


def _get_module__all__(names: Sequence[jedi.api.classes.Name]) -> Optional[Set[str]]:
    try:
        all_name = next(x for x in names if x.name == "__all__")
        return set(
            x.data._get_payload()
            for literal_sequence in all_name.infer()
            for x in literal_sequence._name._value.py__iter__()
        )
    except (AttributeError, StopIteration):
        return None


@functools.lru_cache(maxsize=128)
def module_public_names(
    project: Project,
    module_name: str,
) -> Sequence[jedi.api.classes.BaseName]:
    small_script = project.get_script(
        code=f"from {module_name} import ",
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

    # this will not work for relative import but anyhow
    public_names = module_public_names(project, module_name)
    public_names_identifiers = {x.name for x in public_names}
    candidates = [
        name
        for name in jedi.Script(
            f"from {module_name} import {name.replace(' ','')}",
            project=project,
            path=current_path,
        ).complete(fuzzy=True)
        if name.full_name
    ]
    return [x for x in candidates if x.name in public_names_identifiers]


def join_names(a: str, b: str) -> str:
    if a and b:
        return f"{a}.{b}"
    else:
        return f"{a or b}"


def get_scopes(script: jedi.Script, pos: Position):
    try:
        scope = script.get_context(pos.line + 1, None)
    except ValueError:
        # this is 100% a hack that should not be here
        # but sometimes we are asked to generate spoken
        # hints BEFORE  textdocument/didChange arrives
        # 100% not where it should be solved but for now
        # quick and dirty
        logger.debug("Race condition in get_scopes vs textdocument/didChange")
        scope = script.get_context(None, None)
    while scope:
        yield scope
        scope = scope.parent()


def pretty_scope_list(containing_scopes):
    return " > ".join(
        x.description if x.type != "module" else "mod " + x.full_name
        for x in reversed(containing_scopes)
    )
