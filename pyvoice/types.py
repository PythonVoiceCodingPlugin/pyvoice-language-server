import enum
from pathlib import Path
from typing import NewType, Optional, Tuple

import attrs
from cattrs import Converter
from pygls.server import LanguageServer

__all__ = [
    "SpokenKind",
    "ModuleItem",
    "RelativePath",
    "ProjectSettings",
    "Settings",
    "register_custom_hooks",
]


class SpokenKind(enum.Flag):
    IMPORTABLE = enum.auto()


@attrs.define
class ModuleItem:
    spoken: str
    module: str
    name: Optional[str] = attrs.field(default=None)
    asname: Optional[str] = attrs.field(default=None)
    kind: SpokenKind = attrs.field(default=SpokenKind.IMPORTABLE)


RelativePath = NewType("RelativePath", Path)


@attrs.define
class ProjectSettings:
    # The base path where your python project is located.
    # It can be either absolute or relative to the path of the sublime project.
    path: RelativePath = attrs.field(default=RelativePath("."))

    # The path to the root of the virtual environment.
    # It can be either absolute or relative to the sublime project path.
    environment_path: Optional[RelativePath] = attrs.field(default=None)

    # A list of paths to override the sys path if needed.
    # WARNING: This will COMPLETELY override the sys.path.
    # Leave this null to generate sys.path from the environment.
    sys_path: Optional[Tuple[RelativePath, ...]] = attrs.field(default=None)

    # Adds these paths at the end of the sys path.
    added_sys_path: Tuple[RelativePath, ...] = attrs.field(factory=tuple)

    # If enabled (default), adds paths from local directories.
    # Otherwise, you will have to rely on your packages being properly
    # configured on the sys.path.
    smart_sys_path: bool = attrs.field(default=True)


# Settings for the various generators of imports


@attrs.frozen
class StdlibImportsSettings:
    # Enable or disable this generator.
    enabled: bool = attrs.field(default=True)


@attrs.frozen
class ThirdPartyImportsSettings:
    """
    Settings for customizing the generation of spoken hints for
    third-party modules. Pyvoice will try to automatically
    discover the  dependencies of your project  and  is going
    to generate hints for their modules. In order to do so,pyvoice
    will try:
    - pep621 dependencies in pyproject.toml
    - poetry dependencies in pyproject.toml
    - traditional requirements.txt
    NOTE: By default hints would be generated ONLY for your top level dependencies
    aka distributions that you directly depend on, not transiet dependencies.
    default behavior is not satisfactory you can add/exclude
    distributions from the settings below.
    """

    # Enable or disable this generator.
    enabled: bool = attrs.field(default=True)

    # A list of third-party distributions to include modules from.
    include_dists: Tuple[str, ...] = attrs.field(factory=tuple)

    # A list of third-party distributions to exclude.
    exclude_dists: Tuple[str, ...] = attrs.field(factory=tuple)


@attrs.frozen
class ProjectImportsSettings:
    # Enable or disable this generator.
    enabled: bool = attrs.field(default=True)


@attrs.frozen
class SymbolsImportsSettings:
    # Enable or disable this generator.
    enabled: bool = attrs.field(default=True)

    # A list of modules to generate spoken hints for
    # their defined symbols.
    modules: Tuple[str, ...] = attrs.field(factory=tuple)


@attrs.define
class ImportSettings:
    stdlib: StdlibImportsSettings = attrs.field(factory=StdlibImportsSettings)
    third_party: ThirdPartyImportsSettings = attrs.field(
        factory=ThirdPartyImportsSettings
    )
    project: ProjectImportsSettings = attrs.field(factory=ProjectImportsSettings)
    explicit_symbols: SymbolsImportsSettings = attrs.field(
        factory=SymbolsImportsSettings
    )


# Settings for the various generators of spoken expressions


@attrs.define
class ScopeSettings:
    # Enable or disable the generation of spoken hints for this scope
    enabled: bool = attrs.field(default=True)

    # Generate hints from the parameter names of the
    # signatures of the symbols defined in this scope.
    signature: bool = attrs.field(default=True)


@attrs.define
class ExpressionSettings:
    # settings for the local scope
    locals: ScopeSettings = attrs.field(factory=ScopeSettings)
    # settings for the non-local scope
    nonlocals: ScopeSettings = attrs.field(factory=ScopeSettings)
    # settings for the global scope
    globals: ScopeSettings = attrs.field(factory=ScopeSettings)
    # settings for the builtin scope
    builtins: ScopeSettings = attrs.field(factory=ScopeSettings)

    # an upper bound on the number of expressions to generate
    # this is needed to avoid generating too many expressions
    # which can be overwhelming for the grammar compiler of
    # the programming by voice system.
    limit: int = attrs.field(default=2000)


@attrs.define
class SpokenSettings:
    imports: ImportSettings = attrs.field(default=ImportSettings())
    expressions: ExpressionSettings = attrs.field(default=ExpressionSettings())


@attrs.define
class Settings:
    project: ProjectSettings = attrs.field(default=ProjectSettings())
    spoken: SpokenSettings = attrs.field(default=SpokenSettings())


def register_custom_hooks(server: LanguageServer):
    converter: Converter = server.lsp._converter

    def rel_path_hook(value, _):
        base_path = Path(server.workspace.root_path)
        p = converter.structure(value, Path)
        if p.is_absolute():
            return p
        return (base_path / p).absolute()

    converter.register_structure_hook(RelativePath, rel_path_hook)
