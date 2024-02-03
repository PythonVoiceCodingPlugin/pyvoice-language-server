import functools
import re
import sys  # noqa
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, TypeVar, Union  # noqa

import jedi
import libcst as cst
import toml
from importlib_metadata import Distribution
from jedi.inference.names import ImportName, SubModuleName
from libcst import codemod as transformations
from lsprotocol.types import (
    INITIALIZE,
    WORKSPACE_DID_CHANGE_CONFIGURATION,
    DidChangeConfigurationParams,
    InitializeParams,
    InitializeResult,
    Position,
    WorkspaceEdit,
)
from pygls import protocol
from pygls.protocol import LanguageServerProtocol, lsp_method
from pygls.server import LanguageServer
from requirements_detector import find_requirements

from pyvoice.types.items import ModuleItem

from .text_edit_utils import lsp_text_edits

protocol.deserialize_command = lambda p: p
F = TypeVar("F", bound=Callable)


class MyProtocol(LanguageServerProtocol):
    @lsp_method(INITIALIZE)
    def lsp_initialize(self, params: InitializeParams) -> InitializeResult:
        x = super().lsp_initialize(params)
        venv_path = Path(self._server.workspace.root_path) / ".venv"
        self._server.project = jedi.Project(
            self._server.workspace.root_path,
            environment_path=venv_path if venv_path.exists() else None,
        )
        return x


class PyVoiceLanguageServer(LanguageServer):
    project: jedi.Project

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
            import inspect

            # import cattrs
            # # from lsprotocol.converters import get_converter
            # c = cattrs.Converter()

            def function(server: PyVoiceLanguageServer, args):
                f_args = list(inspect.signature(f).parameters.values())[1:]
                new_args = [
                    server.lsp._converter.structure(value, arg_type.annotation)
                    for arg_type, value in zip(f_args, args)
                ]
                # raise ValueError(f"{f_args} {args} {new_args}")
                return f(server, *new_args)

            self.lsp.fm.command(command_name)(function)
            return f

        return wrapper

    def send_voice(self, command: str, *args, **kwargs):
        server.send_notification(
            "voice/sendRpc", {"command": command, "params": args or kwargs}
        )


server = PyVoiceLanguageServer(
    name="pyvoice", version="0.0.0a2", protocol_cls=MyProtocol
)


@server.feature("workspace/didChangeConfiguration")
def workspace_did_change_configuration(
    ls: PyVoiceLanguageServer, params: DidChangeConfigurationParams
):
    venv_path = params.settings.get("venvPath", ".venv")
    venv_path = Path(venv_path) if venv_path else None
    if venv_path and not venv_path.is_absolute():
        venv_path = Path(ls.workspace.root_path) / venv_path
    ls.show_message("Validating Did change configuration...{}".format(venv_path))
    ls.project = jedi.Project(
        ls.workspace.root_path,
        environment_path=venv_path if venv_path.exists() else None,
    )


# def speak_single_item(x):
#     if re.match(r"[A-Z_]+", x):
#         x = x.lower().replace("_", " ")
#     s = speakit.split_symbol(x)
#     s = " ".join([(x.upper() if len(x) in [2, 3] else x.lower()) for x in s.split()])
#     return s


# lets rewrite this

# pattern = re.compile(r"\W+")

pattern = re.compile(r"[A-Z][a-z]+|[A-Z]+|[a-z]+|\d")
digits_to_names = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "fife",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}


@functools.lru_cache(maxsize=8192)
def speak_single_item(text):
    return " ".join(
        [
            digits_to_names.get(w, w.upper() if len(w) < 3 else w.lower())
            for w in pattern.findall(text)
            if w
        ]
    )


def speak_items(l):
    return {speak_single_item(x): x for x in l}


@functools.lru_cache(maxsize=512)
def with_prefix(prefix: str, name: jedi.api.classes.Name):
    if prefix:
        prefix = prefix + "."
    n = name.name
    if name.type == "function":
        n = n + "()"
    return f"{prefix}{n}"


default_levels = {"module": 1, "instance": 2, "variable": 2, "param": 2, "statement": 2}


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
    if hasattr(project, "_inference_state"):
        s._inference_state = project._inference_state
    return [
        x
        for x in s.complete()
        if "__" not in x.name and "leave" not in x.name and "visit" not in x.name
    ]


from cachetools import LRUCache, cached


@cached(
    cache=LRUCache(maxsize=512 * 4),
    key=lambda name, prefix, level, project: (
        name.full_name,
        prefix,
        level,
        project.path,
    ),
)
def generate_nested(
    name: jedi.api.classes.Name,
    prefix: str,
    level: Optional[int] = None,
    project: Optional[jedi.Project] = None,
):
    return list(_generate_nested(name, prefix, level, project))


def _generate_nested(
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
        if name.name == "pytest":
            level += 1
        for n in module_public_names(project, name.full_name):
            yield with_prefix(prefix, n)
            yield from _generate_nested(n, prefix, level - 1)
    elif name.type == "instance":
        for n in instance_attributes(name.full_name, project):
            yield with_prefix(prefix, n)
            if (
                n.type in ["instance", "variable", "statement", "param"]
                and not name.name.startswith("_")
                and not n.name.startswith("_")
                and True
            ):
                yield from _generate_nested(n, f"{prefix}.{n.name}", level - 1, project)
    elif name.type in ["variable", "statement", "param"]:
        for n in name.infer():
            yield from _generate_nested(n, prefix, level, project)
    elif name.type == "function":
        return
        # if "def " in name.get_line_code():
        #     for n in name.defined_names():
        #         yield with_prefix(prefix, n)
        #         yield from generate_nested(n, n.name, None, project)
    elif name.type == "class" and name.name.endswith("Targets"):
        #        return
        for n in name.defined_names():
            yield with_prefix(prefix, n)
        #     yield from generate_nested(n, prefix, level - 1)


@cached(cache=LRUCache(maxsize=512 * 4), key=lambda n: n.full_name)
def get_keyword_names(n: jedi.api.classes.Name):
    output = []
    for signature in n.get_signatures():
        output.extend(p.name for p in signature.params)
    return output


@functools.lru_cache()
def ignored_names(project: jedi.Project):
    return {x.full_name for x in jedi.Script("", project=project).complete()}


@functools.lru_cache(maxsize=128)
def module_public_names(
    project: jedi.Project, module_name: str
) -> Sequence[jedi.api.classes.BaseName]:
    ignore = ignored_names(project)
    small_script = jedi.Script(f"from {module_name} import ", project=project)
    if hasattr(project, "_inference_state"):
        small_script._inference_state = project._inference_state

    return [
        name
        for name in small_script.complete()
        if name.full_name not in ignore and name.full_name
    ]


def get_top_level_dependencies_names(project: jedi.Project) -> Sequence[str]:
    try:
        p = project.path / "pyproject.toml"
        data = toml.loads(p.read_text())
        return (
            data.get("project", {}).get("dependencies", [])
            or data["tool"]["poetry"]["dependencies"].keys()
        )
    except:
        try:
            return [x.name for x in find_requirements(project.path)]
        except RequirementsNotFound:
            return []


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


from requirements_detector.exceptions import RequirementsNotFound


@functools.lru_cache()
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
            "datetime",
            "itertools",
        ]
        for name in module_public_names(project, x)
    ]
    return output


# server project Environmentserver.a


@cached(
    cache=LRUCache(maxsize=4),
    key=lambda project: (project, project.path.stat().st_mtime),
)
def get_project_modules(project: jedi.Project):
    output = [
        relative_path_to_item(x)
        for y in project.path.iterdir()
        if not y.name.startswith(".") and y.is_dir()
        for x in map(
            lambda p: p.relative_to(project.path),
            Path(y).glob("**/*.py"),
        )
        if len(x.parts) > 1 and "." not in x.parts[0]
    ]
    return output


def get_modules(project: jedi.Project):
    return (
        get_project_modules(project)
        + get_builtin_modules(project)
        + get_top_level_dependencies_modules(project)
    )


@server.command("get_spoken")
def function(
    server: PyVoiceLanguageServer,
    doc_uri: str,
    pos: Position = None,
    generate_importables: bool = True,
):
    document = server.workspace.get_document(doc_uri)
    s = jedi.Script(code=document.source, path=document.path, project=server.project)
    if hasattr(server.project, "_inference_state"):
        s._inference_state = server.project._inference_state
    else:
        server.project._inference_state = s._inference_state
    if generate_importables:
        imp = get_modules(server.project)
        server.send_voice("enhance_spoken", "importable", imp)
    else:
        imp = None
    global_names = s.get_names()
    output = []
    for n in global_names:
        output.append(with_prefix("", n))
        output.extend(
            generate_nested(
                n, n.name if n.type != "function" else "", None, server.project
            )
        )
        output.extend(get_keyword_names(n))
    if pos:
        f = s.get_context(pos.line + 1, pos.character)
        if f.type == "function":
            for n in f.defined_names():
                output.append(with_prefix("", n))
                t = list(
                    generate_nested(
                        n, n.name if n.type != "function" else "", None, server.project
                    )
                )
                output.extend(t)
    output = [x for x in sorted(set(output)) if "__" not in x]
    if len(output) < 2000:
        output = output[:2000]
    d = speak_items(output)
    server.send_voice(
        "enhance_spoken",
        "expression",
        [{"spoken": k, "value": v} for k, v in d.items()],
    )
    if imp is not None:
        server.show_message(f"{len(output)} expressions, {len(imp)} imports")
    else:
        server.show_message(f"{len(output)} expressions, skipped imports")


# op server.send_voice( )


@server.command("add_import")
def function_add_import(
    server: PyVoiceLanguageServer,
    doc_uri: str,
    items: ModuleItem
    # items: Union[ModuleItem, List[ModuleItem]],
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
    server.send_voice("enhance_spoken", "subsymbol", s)


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
    # server.send_voice("enhance_from_import", s) server.send_voice()
    # "pyvoice.types""pathlib.Path"
