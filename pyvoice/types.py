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
    path: RelativePath = attrs.field(default=RelativePath("."))
    environment_path: Optional[RelativePath] = attrs.field(default=None)
    sys_path: Optional[Tuple[RelativePath, ...]] = attrs.field(default=None)
    added_sys_path: Tuple[RelativePath, ...] = attrs.field(default=tuple())
    smart_sys_path: bool = attrs.field(default=True)


@attrs.define
class Settings:
    project: ProjectSettings = attrs.field(default=ProjectSettings())


def register_custom_hooks(server: LanguageServer):
    converter: Converter = server.lsp._converter

    def rel_path_hook(value, _):
        base_path = Path(server.workspace.root_path)
        p = converter.structure(value, Path)
        if p.is_absolute():
            return p
        return (base_path / p).absolute()

    converter.register_structure_hook(RelativePath, rel_path_hook)
