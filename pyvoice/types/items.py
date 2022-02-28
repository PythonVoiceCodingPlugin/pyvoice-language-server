import enum
from typing import Optional

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
