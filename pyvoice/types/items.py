import enum
from enum import Enum, Flag
from typing import Optional, Sequence

from .models import Model


class SpokenKind(enum.Flag):
    IMPORTABLE = enum.auto()


class SpokenItem(Model):
    spoken: str = ""
    kind: SpokenKind


class ModuleItem(SpokenItem):
    kind = SpokenKind.IMPORTABLE
    module: str
    name: Optional[str]
    asname: Optional[str]
