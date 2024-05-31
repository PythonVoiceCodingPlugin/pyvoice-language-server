import enum
from pathlib import Path
from typing import Optional, Tuple

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


@attrs.frozen
class ProjectSettings:
    path: Path = attrs.field(default=Path("."))
    environment_path: Path = attrs.field(default=Path(".venv"))
    sys_path: Optional[Tuple[Path]] = attrs.field(default=None)
    added_sys_path: Tuple[Path] = attrs.field(default=tuple())
    smart_sys_path: bool = attrs.field(default=True)


@attrs.define
class Settings:
    project: ProjectSettings = attrs.field(default=ProjectSettings())
