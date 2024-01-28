"""Utility functions for converting to TextEdit.


These code has been borrowed with small adaptations from jedi-language-server
and is licensed accordingly.
This module is a bridge between `jedi.Refactoring` and
`pygls.types.TextEdit` types
"""


import ast
import difflib
from bisect import bisect_right
from typing import List, NamedTuple

from lsprotocol.types import Position, Range, TextEdit
from pygls.workspace import Document


def is_valid_python(code: str) -> bool:
    """Check whether Python code is syntactically valid."""
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return True


_OPCODES_CHANGE = {"replace", "delete", "insert"}


def lsp_text_edits(document: Document, new_code: str) -> List[TextEdit]:
    """Take a jedi `ChangedFile` and convert to list of text edits.

    Handles inserts, replaces, and deletions within a text file.

    Additionally, makes sure returned code is syntactically valid Python.
    """
    old_code = document.source
    position_lookup = PositionLookup(old_code)
    text_edits = []
    for opcode in get_opcodes(old_code, new_code):
        if opcode.op in _OPCODES_CHANGE:
            start = position_lookup.get(opcode.old_start)
            end = position_lookup.get(opcode.old_end)
            new_text = new_code[opcode.new_start : opcode.new_end]
            text_edits.append(
                TextEdit(
                    range=Range(start=start, end=end),
                    new_text=new_text,
                )
            )
    return text_edits


class Opcode(NamedTuple):
    """Typed opcode.

    Op can be one of the following values:
        'replace':  a[i1:i2] should be replaced by b[j1:j2]
        'delete':   a[i1:i2] should be deleted.
            Note that j1==j2 in this case.
        'insert':   b[j1:j2] should be inserted at a[i1:i1].
            Note that i1==i2 in this case.
        'equal':    a[i1:i2] == b[j1:j2]
    """

    op: str
    old_start: int
    old_end: int
    new_start: int
    new_end: int


def get_opcodes(old: str, new: str) -> List[Opcode]:
    """Obtain typed opcodes from two files (old and new)"""
    diff = difflib.SequenceMatcher(a=old, b=new)
    return [Opcode(*opcode) for opcode in diff.get_opcodes()]


# pylint: disable=too-few-public-methods
class PositionLookup:
    """Data structure to convert a byte offset in a file to a line number and
    character."""

    def __init__(self, code: str) -> None:
        # Create a list saying at what offset in the file each line starts.
        self.line_starts = []
        offset = 0
        for line in code.splitlines(keepends=True):
            self.line_starts.append(offset)
            offset += len(line)

    def get(self, offset: int) -> Position:
        """Get the position in the file that corresponds to the given
        offset."""
        line = bisect_right(self.line_starts, offset) - 1
        character = offset - self.line_starts[line]
        return Position(line=line, character=character)
