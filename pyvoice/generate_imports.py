import functools
import itertools
from pathlib import Path
from typing import List, Sequence, Tuple

import toml
from cachetools import LRUCache, cached
from importlib_metadata import Distribution
from requirements_detector import find_requirements
from requirements_detector.exceptions import RequirementsNotFound
from stdlibs import module_names as stdlib_module_names

from pyvoice.custom_jedi_classes import Project
from pyvoice.inference import module_public_names
from pyvoice.speakify import speak_single_item
from pyvoice.types import ModuleItem

__all__ = [
    "get_modules",
    "get_top_level_dependencies_modules",
    "get_stdlib_modules",
    "get_project_modules",
    "get_extra_subsymbols",
    "get_modules_from_distribution",
    "relative_path_to_item",
    "get_top_level_dependencies_names",
]


def get_top_level_dependencies_names(project: Project) -> Sequence[str]:
    try:
        p = project.path / "pyproject.toml"
        data = toml.loads(p.read_text())
        return (
            data.get("project", {}).get("dependencies", [])
            or data["tool"]["poetry"]["dependencies"].keys()
        )
    except Exception:
        try:
            return [x.name for x in find_requirements(project.path)]
        except RequirementsNotFound:
            return []


@functools.lru_cache()
def get_modules_from_distribution(project: Project, name: str) -> Sequence[ModuleItem]:
    try:
        return [
            relative_path_to_item(f)
            for distribution in Distribution.discover(
                name=name, path=project.get_environment().get_sys_path()
            )
            for f in distribution.files  # type: ignore
            if f.suffix == ".py"
        ]

    except Exception:
        if name != "python":
            raise ValueError(name) from None
        return []


def get_top_level_dependencies_modules(project: Project):
    return [
        x
        for dependency_name in get_top_level_dependencies_names(project)
        for x in get_modules_from_distribution(project, dependency_name)
    ]


@functools.lru_cache()
def relative_path_to_item(x: Path) -> ModuleItem:
    if x.name == "__init__.py":
        return relative_path_to_item(x.parent)
    if len(x.parts) == 1:
        return ModuleItem(
            spoken=speak_single_item(" ".join(x.parts).replace(".py", "")),
            module=x.name.replace(".py", ""),
            name=None,
        )
    return ModuleItem(
        spoken=speak_single_item(" ".join(x.parts).replace(".py", "")),
        module=".".join(x.parts[:-1]).replace(".py", ""),
        name=x.stem,
    )


@functools.lru_cache()
def get_stdlib_modules(project: Project):
    return [
        ModuleItem(spoken=speak_single_item(x), module=x, name=None)
        for x in stdlib_module_names
        if not x.startswith("_")
    ]


@functools.lru_cache()
def get_extra_subsymbols(project: Project, key_value_pairs: Sequence[Tuple[str, str]]):
    output = [
        ModuleItem(
            spoken=speak_single_item(f"{spoken_prefix} {name.name}"),
            module=module_name,
            name=name.name,
        )
        for module_name, spoken_prefix in key_value_pairs
        for name in module_public_names(project, module_name)
    ]
    return output


@cached(
    cache=LRUCache(maxsize=4),
    key=lambda project: (project, project.path.stat().st_mtime),
)
def get_project_modules(project: Project):
    output = [
        relative_path_to_item(x)
        for y in project.path.iterdir()
        if not y.name.startswith(".") and y.is_dir()
        for x in map(
            lambda p: p.relative_to(project.path),
            Path(y).glob("**/*.py"),
        )
        if len(x.parts) > 1 and "." not in x.parts[0]
    ]
    return output


def get_modules(
    project: Project, extra_subsymbols: Sequence[Tuple[str, str]]
) -> List[ModuleItem]:
    return list(
        itertools.chain(
            get_stdlib_modules(project),
            get_top_level_dependencies_modules(project),
            get_project_modules(project),
            get_extra_subsymbols(project, extra_subsymbols),
        )
    )
