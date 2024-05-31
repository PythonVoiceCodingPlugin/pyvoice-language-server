import enum
from pathlib import Path
from typing import List, NewType, Optional, Tuple

import attrs


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
    environment_path: RelativePath = attrs.field(default=RelativePath(".venv"))
    sys_path: Optional[Tuple[RelativePath, ...]] = attrs.field(default=None)
    added_sys_path: Tuple[RelativePath, ...] = attrs.field(default=tuple())
    smart_sys_path: bool = attrs.field(default=True)


@attrs.define
class Settings:
    project: ProjectSettings = attrs.field(default=ProjectSettings())
