import enum
from enum import Enum, Flag
from typing import Iterable, Optional, Sequence

from pydantic import dataclasses

from .models import Model


class SpokenKind(enum.Flag):
    IMPORTABLE = enum.auto()


class SpokenItem(Model):
    """item that can be spoken"""

    spoken: str = ""
    kind: SpokenKind


class ModuleItem(SpokenItem):
    kind = SpokenKind.IMPORTABLE
    module: str
    name: Optional[str]
    asname: Optional[str]
