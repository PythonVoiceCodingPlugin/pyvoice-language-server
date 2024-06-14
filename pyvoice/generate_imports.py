import functools
import itertools
from pathlib import Path
from typing import List, Optional, Sequence

import toml
from cachetools import LRUCache, cached
from importlib_metadata import Distribution
from requirements_detector import find_requirements
from requirements_detector.exceptions import RequirementsNotFound
from requirements_detector.requirement import DetectedRequirement
from setuptools.discovery import (
    FlatLayoutModuleFinder,
    FlatLayoutPackageFinder,
    find_package_path,
)
from stdlibs import module_names as stdlib_module_names

from pyvoice.custom_jedi_classes import Project
from pyvoice.inference import module_public_names
from pyvoice.speakify import speak_single_item
from pyvoice.types import (
    ImportSettings,
    ModuleItem,
    ProjectImportsSettings,
    StdlibImportsSettings,
    SymbolsImportsSettings,
    ThirdPartyImportsSettings,
)

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


@cached(
    cache=LRUCache(maxsize=4),
    key=lambda project: (project.path, project.path.stat().st_mtime),
)
def _get_pyproject_toml(project: Project) -> Optional[dict]:
    try:
        p = project.path / "pyproject.toml"
        return toml.loads(p.read_text())
    except Exception:
        return None


# def _get_dependencies_from_pyproject_toml(project: Project) -> Sequence[str]:
def _get_pep621_dependencies(project: Project) -> Sequence[str]:
    pyproject_toml = _get_pyproject_toml(project)
    if pyproject_toml is None:
        raise RequirementsNotFound("pyproject.toml not found")
    try:
        raw = list(pyproject_toml["project"]["dependencies"].keys())
        parsed = [DetectedRequirement.parse(x) for x in raw]
        return [
            x.name
            for x in parsed
            if x is not None and x.name is not None and x.name != "python"
        ]

    except (KeyError, AttributeError) as e:
        raise RequirementsNotFound(
            "No pep621 dependencies found in pyproject.toml"
        ) from e


def _get_poetry_dependencies(project: Project) -> Sequence[str]:
    pyproject_toml = _get_pyproject_toml(project)
    if pyproject_toml is None:
        raise RequirementsNotFound("pyproject.toml not found")
    try:
        return list(pyproject_toml["tool"]["poetry"]["dependencies"].keys())
    except (KeyError, AttributeError) as e:
        raise RequirementsNotFound(
            "No poetry dependencies found in pyproject.toml"
        ) from e


def _get_traditional_dependencies(project: Project) -> Sequence[str]:
    return [x.name for x in find_requirements(project.path)]


def get_top_level_dependencies_names(project: Project) -> Sequence[str]:
    for method in (
        _get_pep621_dependencies,
        _get_poetry_dependencies,
        _get_traditional_dependencies,
    ):
        try:
            return method(project)
        except RequirementsNotFound:
            pass
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


@cached(cache=LRUCache(maxsize=4))
def get_top_level_dependencies_modules(
    project: Project, settings: ThirdPartyImportsSettings
):
    if not settings.enabled:
        return []
    detected = get_top_level_dependencies_names(project)
    detected.extend(settings.include_dists)
    names = filter(lambda x: x not in settings.exclude_dists, set(detected))
    return [
        x
        for dependency_name in names
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


@cached(cache=LRUCache(maxsize=4))
def get_stdlib_modules(
    project: Project, settings: StdlibImportsSettings
) -> List[ModuleItem]:
    if not settings.enabled:
        return []
    return [
        ModuleItem(spoken=speak_single_item(x), module=x, name=None)
        for x in stdlib_module_names
        if not x.startswith("_")
    ]


@cached(
    cache=LRUCache(maxsize=4),
)
def get_extra_subsymbols(project: Project, settings: SymbolsImportsSettings):
    if not settings.enabled:
        return []
    output = [
        ModuleItem(
            spoken=speak_single_item(f"{name.name}"),
            module=module_name,
            name=name.name,
        )
        for module_name in settings.modules
        for name in module_public_names(project, module_name)
    ]
    return output


@cached(
    cache=LRUCache(maxsize=4),
    key=lambda project, settings: (
        project,
        project.path.stat().st_mtime,
        settings,
    ),
)
def get_project_modules(project: Project, settings: ProjectImportsSettings):
    if not settings.enabled:
        return []

    # try src layout first
    src_path = project.path.joinpath("src")
    if src_path.exists() and src_path.is_dir():
        return [
            relative_path_to_item(x)
            for x in map(
                lambda p: p.relative_to(src_path),
                src_path.rglob("*.py"),
            )
        ]
    # if that fails, we allow for a mixure of what setuptools
    # calls flat layout package and flat layout module
    output = []
    top_modules: List[str] = FlatLayoutModuleFinder.find(project.path)
    output.extend(
        ModuleItem(spoken=speak_single_item(x), module=x, name=None)
        for x in top_modules
    )
    packages: List[str] = FlatLayoutPackageFinder.find(project.path)
    output.extend(
        relative_path_to_item(x)
        for pkg in packages
        for x in map(
            lambda p: p.relative_to(project.path),
            Path(find_package_path(pkg, {}, project.path)).glob("*.py"),
        )
    )
    return output


def get_modules(
    project: Project,
    settings: ImportSettings,
) -> List[ModuleItem]:
    return list(
        itertools.chain(
            get_stdlib_modules(project, settings.stdlib),
            get_top_level_dependencies_modules(project, settings.third_party),
            get_project_modules(project, settings.project),
            get_extra_subsymbols(project, settings.explicit_symbols),
        )
    )
