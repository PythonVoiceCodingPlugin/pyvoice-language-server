import functools
import os
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, TypeVar

import jedi
import libcst as cst
import speakit
from jedi.inference.names import ImportName, SubModuleName
from libcst import codemod as transformations
from pydantic import validate_arguments
from pygls import protocol
from pygls.lsp.methods import INITIALIZE
from pygls.lsp.types import InitializeParams, InitializeResult, WorkspaceEdit
from pygls.protocol import LanguageServerProtocol, lsp_method
from pygls.server import LanguageServer

from pyvoice.types.items import ModuleItem

from .text_edit_utils import lsp_text_edits

protocol.deserialize_command = lambda p: p
F = TypeVar("F", bound=Callable)


class MyProtocol(LanguageServerProtocol):
    @lsp_method(INITIALIZE)
    def lsp_initialize(self, params: InitializeParams) -> InitializeResult:
        x = super().lsp_initialize(params)
        self._server.project = jedi.Project(
            self._server.workspace.root_path,
            environment_path=os.path.join(self._server.workspace.root_path, ".venv"),
        )

        return x


class PyVoiceLanguageServer(LanguageServer):
    def __init__(
        self, loop=None, protocol_cls=LanguageServerProtocol, max_workers: int = 2
    ):
        super().__init__(loop, protocol_cls, max_workers)

    def command(
        self, command_name: str
    ) -> Callable[[F], Callable[["PyVoiceLanguageServer", Any], Any]]:
        """Decorator used to register custom commands.

        Example:
            @ls.command('myCustomCommand')
            def my_cmd(ls, a, b, c):
                pass
        """

        def wrapper(f: F):
            f = validate_arguments(config=dict(arbitrary_types_allowed=True))(f)

            def function(server: PyVoiceLanguageServer, args):
                return f(server, *args)

            return self.lsp.fm.command(command_name)(function)

        return wrapper

    def send_voice(self, command: str, *args, **kwargs):
        server.send_notification(
            "voice/sendRpc", {"command": command, "params": args or kwargs}
        )


server = PyVoiceLanguageServer(protocol_cls=MyProtocol)


def speak_single_item(x):
    # if re.match(r"[A-Z_]+", x):
    #     return x.lower().replace("_", " ")

    s = speakit.split_symbol(x)
    s = " ".join([(x.upper() if len(x) in [2, 3] else x) for x in s.split()])
    return s


def speak_items(l):
    return {speak_single_item(x): x for x in l}


def with_prefix(prefix: str, name: jedi.api.classes.Name):
    if prefix:
        prefix = prefix + "."
    n = name.name
    if name.type == "function":
        n = n + "()"
    return f"{prefix}{n}"


default_levels = {"module": 1, "instance": 3}


def generate_nested(
    name: jedi.api.classes.Name,
    prefix: str,
    level: Optional[int] = None,
    project: Optional[jedi.Project] = None,
):
    if level is None:
        level = default_levels.get(name.type, 1)
    if level <= 0:
        return
    if name.type == "module":
        for n in module_public_names(project, name.full_name):
            yield with_prefix(prefix, n)
            yield from generate_nested(n, prefix, level - 1)
    elif name.type == "instance":
        for n in name.defined_names():
            yield with_prefix(prefix, n)
            yield from generate_nested(n, prefix, level - 1)
    else:
        for n in name.defined_names():
            yield with_prefix(prefix, n)
            yield from generate_nested(n, prefix, level - 1)


@functools.lru_cache()
def ignored_names(project: jedi.Project):
    return {x.full_name for x in jedi.Script("", project=project).complete()}


def module_public_names(
    project: jedi.Project, module_name: str
) -> Sequence[jedi.api.classes.BaseName]:
    ignore = ignored_names(project)
    return [
        name
        for name in jedi.Script(
            f"from {module_name} import *\n", project=project
        ).complete()
        if name.full_name not in ignore and name.full_name
    ]


@functools.lru_cache()
def get_builtin_modules(project: jedi.Project):
    output = [
        ModuleItem(
            spoken=speak_single_item(f"{x} {name.name}"), module=x, name=name.name
        )
        for x in ["typing"]
        for name in module_public_names(project, x)
    ]
    return output


def get_modules(project: jedi.Project):
    output = [
        ModuleItem(
            spoken=speak_single_item(" ".join(x.parts[1:])),
            module=".".join(x.parts[:-1]),
            name=x.stem,
        )
        for x in map(
            lambda p: p.relative_to(project.path),
            Path(project.path).glob("[!.]*\\**\\*.py"),
        )
        if len(x.parts) > 1 and "." not in x.parts[0]
    ]
    return output + get_builtin_modules(project)


@server.command("get_spoken")
def function(server: PyVoiceLanguageServer, doc_uri: str):
    server.show_message("Validating json...")
    document = server.workspace.get_document(doc_uri)
    s = jedi.Script(code=document.source, path=document.path, project=server.project)
    x = s.get_names()
    server.show_message(f"{x}")
    output = []
    for n in x:
        output.append(with_prefix("", n))
        output.extend(generate_nested(n, n.name, None, server.project))

    output = [x for x in output if "__" not in x]
    server.show_message(len(output))
    server.send_voice("enhance", speak_items(output))
    server.send_voice("enhance_import", get_modules(server.project))


@server.command("add_import")
def function_add_import(server: PyVoiceLanguageServer, doc_uri: str, item: ModuleItem):
    document = server.workspace.get_document(doc_uri)
    wrapper = cst.MetadataWrapper(cst.parse_module(document.source))
    context = transformations.CodemodContext(wrapper=wrapper)
    transformer = transformations.visitors.AddImportsVisitor(  # type: ignore[attr-defined]
        context,
        [transformations.visitors.ImportItem(item.module, item.name, item.asname)],
    )
    result = transformations.transform_module(transformer, document.source)
    edit = WorkspaceEdit(changes={doc_uri: lsp_text_edits(document, result.code)})
    server.apply_edit(edit)
