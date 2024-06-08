from itertools import groupby
from typing import List

from parso import parse

from pyvoice.types import ModuleItem

__all__ = ["add_imports_to_module", "add_imports_to_code"]


def add_imports_to_module(module, items: List[ModuleItem]) -> None:
    """add import statements to a module"""
    new_nodes = []
    for module_name, values_iter in groupby(items, lambda x: x.module):
        values = list(values_iter)
        names = [
            x.name if not x.asname else f"{x.name} as {x.asname}"
            for x in values
            if x.name
        ]
        if names:
            new_nodes.append(
                parse(f"from {module_name} import {', '.join(names)}\n").children[0]
            )
        if any(x.name is None for x in values):
            new_nodes.append(parse(f"import {module_name}\n").children[0])
    start = 0
    try:
        if module.children[0].children[0].type == "string":
            start = 1
    except (IndexError, AttributeError):
        pass
    for node in new_nodes:
        node.parent = module
        module.children.insert(start, node)


def add_imports_to_code(code: str, items: List[ModuleItem]) -> str:
    """add import statements to a code string"""
    module = parse(code)
    add_imports_to_module(module, items)
    return module.get_code()
