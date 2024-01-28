import enum
from typing import Optional

import attrs


class SpokenKind(enum.Flag):
    IMPORTABLE = enum.auto()


# @attrs.define
# class SpokenItem():
#     """item that can be spoken"""

#     kind: SpokenKind
#     spoken: str = attrs.field(default="")


@attrs.define
class ModuleItem:
    spoken: str
    module: str
    name: Optional[str] = attrs.field(default=None)
    asname: Optional[str] = attrs.field(default=None)
    kind: SpokenKind = attrs.field(default=SpokenKind.IMPORTABLE)
