import functools
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, TypeVar, Union

import jedi
import libcst as cst
import speakit
import toml
from importlib_metadata import Distribution
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

# you code new codeproject.get_environment()
protocol.deserialize_command = lambda p: p
F = TypeVar("F", bound=Callable)


class MyProtocol(LanguageServerProtocol):
    @lsp_method(INITIALIZE)
    def lsp_initialize(self, params: InitializeParams) -> InitializeResult:
        x = super().lsp_initialize(params)
        self._server.project = jedi.Project(
            self._server.workspace.root_path,
            environment_path=Path(self._server.workspace.root_path) / ".venv",
        )
        return x


class PyVoiceLanguageServer(LanguageServer):
    project: jedi.Project

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

            self.lsp.fm.command(command_name)(function)
            return f

        return wrapper

    def send_voice(self, command: str, *args, **kwargs):
        server.send_notification(
            "voice/sendRpc", {"command": command, "params": args or kwargs}
        )


server = PyVoiceLanguageServer(protocol_cls=MyProtocol)

# server.workspace


def speak_single_item(x):
    # if re.match(r"[A-Z_]+", x):
    #     return x.lower().replace("_", " ")abc
    x = (
        x.replace("python_voice_coding_plugin.typing", "")
        .replace("python_voice_coding_plugin.types", "")
        .replace("python_voice_coding_plugin", "root")
    )

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


default_levels = {"module": 1, "instance": 3, "variable": 3, "param": 3, "statement": 3}


@functools.lru_cache(maxsize=128)
def instance_attributes(
    full_name: str, project: jedi.Project
) -> Sequence[jedi.api.classes.BaseName]:
    if full_name is None:
        return []
    text = f"""
import {full_name.split('.')[0]}
_ : {full_name}
_."""
    s = jedi.Script(text, project=project)
    return [
        x
        for x in s.complete()
        if "__" not in x.name and "leave" not in x.name and "visit" not in x.name
    ]


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
        for n in instance_attributes(name.full_name, project):
            yield with_prefix(prefix, n)
            if n.type in ["instance"]:
                yield from generate_nested(n, f"{prefix}.{n.name}", level - 1, project)
    elif name.type in ["variable", "statement", "param"]:
        for n in name.infer():
            yield from generate_nested(n, prefix, level, project)
    elif name.type == "function":
        if "def " in name.get_line_code():
            for n in name.defined_names():
                yield with_prefix(prefix, n)
                yield from generate_nested(n, n.name, None, project)

    else:
        return
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


def get_top_level_dependencies_names(project: jedi.Project) -> Sequence[str]:
    p = project.path / "pyproject.toml"
    data = toml.loads(p.read_text())
    return data["tool"]["poetry"]["dependencies"].keys()


@functools.lru_cache()
def get_modules_from_distribution(
    project: jedi.Project, name: str
) -> Sequence[ModuleItem]:
    try:
        return [
            relative_path_to_item(f)
            for distribution in Distribution.discover(
                name=name, path=project.get_environment().get_sys_path()
            )
            for f in distribution.files
            if f.suffix == ".py"
        ]

    except Exception:
        if name != "python":
            raise ValueError(name)
        return []


def get_top_level_dependencies_modules(project: jedi.Project):
    return [
        x
        for dependency_name in get_top_level_dependencies_names(project)
        for x in get_modules_from_distribution(project, dependency_name)
    ]


def relative_path_to_item(x: Path) -> ModuleItem:
    if x.name == "__init__.py":
        return relative_path_to_item(x.parent)
    if len(x.parts) == 1:
        return ModuleItem(
            spoken=speak_single_item(" ".join(x.parts).replace(".py", "")),
            module=x.name.replace(".py", ""),
            name=None,
        )
    return ModuleItem(
        spoken=speak_single_item(" ".join(x.parts).replace(".py", "")),
        module=".".join(x.parts[:-1]).replace(".py", ""),
        name=x.stem,
    )


@functools.lru_cache()
def get_builtin_modules(project: jedi.Project):
    output = [
        ModuleItem(
            spoken=speak_single_item(f"{x} {name.name}"), module=x, name=name.name
        )
        for x in [
            "typing",
            "importlib_metadata",
            "enum",
            "pathlib",
            "python_voice_coding_plugin.typing",
            "python_voice_coding_plugin.types",
        ]
        for name in module_public_names(project, x)
    ]
    return output


# server project Environmentserver.a


def get_modules(project: jedi.Project):
    output = [
        relative_path_to_item(x)
        for x in map(
            lambda p: p.relative_to(project.path),
            Path(project.path).glob("[!.]*\\**\\*.py"),
        )
        if len(x.parts) > 1 and "." not in x.parts[0]
    ]
    return (
        output
        + get_builtin_modules(project)
        + get_top_level_dependencies_modules(project)
    )


@server.command("get_spoken")
def function(server: PyVoiceLanguageServer, doc_uri: str):
    server.show_message("Validating json...")
    document = server.workspace.get_document(doc_uri)
    s = jedi.Script(code=document.source, path=document.path, project=server.project)
    x = s.get_names()
    server.show_message(f"{len(x)}")
    output = []
    for n in x:
        output.append(with_prefix("", n))
        output.extend(
            generate_nested(
                n, n.name if n.type != "function" else "", None, server.project
            )
        )

    output = [x for x in sorted(set(output)) if "__" not in x]
    server.show_message(len(output))
    server.send_voice("enhance", speak_items(output))
    server.send_voice("enhance_import", get_modules(server.project))


@server.command("add_import")
def function_add_import(
    server: PyVoiceLanguageServer,
    doc_uri: str,
    items: Union[ModuleItem, List[ModuleItem]],
):
    server.show_message(f"{items}")
    document = server.workspace.get_document(doc_uri)
    wrapper = cst.MetadataWrapper(cst.parse_module(document.source))
    context = transformations.CodemodContext(wrapper=wrapper)
    items = items if isinstance(items, list) else [items]
    transformer = transformations.visitors.AddImportsVisitor(  # type: ignore[attr-defined]
        context,
        [
            transformations.visitors.ImportItem(item.module, item.name, item.asname)
            for item in items
        ],
    )
    result = transformations.transform_module(transformer, document.source)
    edit = WorkspaceEdit(changes={doc_uri: lsp_text_edits(document, result.code)})
    server.apply_edit(edit)


def join_names(a: str, b: str) -> str:
    if a and b:
        return f"{a}.{b}"
    else:
        return f"{a or b}"


@server.command("from_import")
def function_from_import(server: PyVoiceLanguageServer, item: ModuleItem):
    module_name = join_names(item.module, item.name)
    s = [
        ModuleItem(spoken=speak_single_item(x.name), module=module_name, name=x.name)
        for x in module_public_names(server.project, module_name)
    ]
    server.send_voice("enhance_from_import", s)


def module_public_names_fuzzy(
    project: jedi.Project, current_path: str, module_name: str, name: str
) -> Sequence[jedi.api.classes.BaseName]:
    ignore = ignored_names(project)
    return [
        name
        for name in jedi.Script(
            f"from {module_name} import *\n{name.replace(' ','')}",
            project=project,
            path=current_path,
        ).complete(fuzzy=True)
        if name.full_name not in ignore and name.full_name
    ]


@server.command("from_import_fuzzy")
def function_from_import_fuzzy(
    server: PyVoiceLanguageServer,
    doc_uri: str,
    item: ModuleItem,
    name: str,
    every: bool,
):
    module_name = join_names(item.module, item.name)
    document = server.workspace.get_document(doc_uri)
    choices = module_public_names_fuzzy(
        server.project, document.path, module_name, name
    )
    choices = list(
        sorted(
            choices,
            reverse=True,
            key=lambda x: x.full_name == join_names(module_name, x.name),
        )
    )
    chosen = choices if every else [choices[0]]
    server.show_message(f"{choices}")
    items = [ModuleItem(spoken="", module=module_name, name=x.name) for x in chosen]
    function_add_import(server, doc_uri, items)
    # server.send_voice("enhance_from_import", s)


# b
