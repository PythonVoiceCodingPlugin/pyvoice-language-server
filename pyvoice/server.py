import functools
import itertools
import logging
import sys  # noqa
from itertools import groupby
from pathlib import Path
from typing import (  # noqa
    Any,
    Callable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import jedi
import toml
from cachetools import LRUCache, cached
from importlib_metadata import Distribution
from lsprotocol.types import (
    WORKSPACE_DID_CHANGE_CONFIGURATION,
    DidChangeConfigurationParams,
    Position,
    WorkspaceEdit,
)
from parso import parse
from pygls import protocol
from pygls.protocol import LanguageServerProtocol, default_converter
from pygls.server import LanguageServer
from requirements_detector import find_requirements
from requirements_detector.exceptions import RequirementsNotFound
from stdlibs import module_names as stdlib_module_names

from pyvoice.project import Project
from pyvoice.speakify import speak_items, speak_single_item
from pyvoice.types import ModuleItem, RelativePath, Settings

from .text_edit_utils import lsp_text_edits

protocol.deserialize_command = lambda p: p
F = TypeVar("F", bound=Callable)


logger = logging.getLogger(__name__)


class MyProtocol(LanguageServerProtocol):
    # @lsp_method(INITIALIZE)
    # def lsp_initialize(self, params: InitializeParams) -> InitializeResult:
    #     x = super().lsp_initialize(params)
    #     venv_path = Path(self._server.workspace.root_path) / ".venv"
    #     self._server.project = Project(
    #         self._server.workspace.root_path,
    #         environment_path=venv_path if venv_path.exists() else None,
    #     )
    #     # self._server.extra_subsymbols = {}
    #     return x
    pass


class PyVoiceLanguageServer(LanguageServer):
    def __init__(self, *args, **kwargs):
        converter_factory = kwargs.pop("converter_factory", default_converter)

        def wrapper_factory():
            converter = converter_factory()

            def hook(value, _):
                base_path = Path(self.workspace.root_path)
                p = converter.structure(value, Path)
                if p.is_absolute():
                    return p
                return (base_path / p).absolute()

            converter.register_structure_hook(RelativePath, hook)
            return converter

        kwargs["converter_factory"] = wrapper_factory
        super().__init__(*args, **kwargs)

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

            def function(server: PyVoiceLanguageServer, args):
                f_args = list(inspect.signature(f).parameters.values())[1:]
                new_args = [
                    server.lsp._converter.structure(value, arg_type.annotation)
                    for arg_type, value in zip(f_args, args)
                ]
                return f(server, *new_args)

            self.lsp.fm.command(command_name)(function)
            return f

        return wrapper

    @property
    def project(self) -> Project:
        try:
            if (
                self._last_workspace_root_path == self.workspace.root_path
                or self._last_configuration_settings == self.configuration_settings
            ):
                return self._project
        except AttributeError:
            logger.info(
                "Creating jedi project from %s", self.configuration_settings.project
            )
            self._last_workspace_root_path = self.workspace.root_path
            self._last_configuration_settings = self.configuration_settings
            self._project = Project.from_settings(self.configuration_settings.project)
            return self._project

    @property
    def configuration_settings(self) -> Settings:
        return getattr(self, "_configuration_settings", Settings())

    def send_voice(self, command: str, *args, **kwargs):
        server.send_notification(
            "voice/sendRpc", {"command": command, "params": args or kwargs}
        )


server = PyVoiceLanguageServer(
    name="pyvoice", version="0.0.0a2", protocol_cls=MyProtocol
)


def _dotted_dict_to_normal(d: dict, prefix=""):
    output = {}
    for k, v in d.items():
        if isinstance(v, dict):
            output.update(_dotted_dict_to_normal(v, f"{prefix}{k}."))
        else:
            output[f"{prefix}{k}"] = v
    return output


@server.feature(WORKSPACE_DID_CHANGE_CONFIGURATION)
def workspace_did_change_configuration(
    ls: PyVoiceLanguageServer, params: DidChangeConfigurationParams
):
    ls._configuration_settings = ls.lsp._converter.structure(params.settings, Settings)
    # ls.project = Project.from_settings(
    #     ls.configuration_settings.project, Path(ls.workspace.root_path)
    # )


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
    full_name: str, project: Project
) -> Sequence[jedi.api.classes.BaseName]:
    if full_name is None:
        return []
    text = f"""
import {full_name.split('.')[0]}
_ : {full_name}
_."""
    small_script = project.get_script(code=text)
    return [
        x
        for x in small_script.complete()
        if "__" not in x.name and "leave" not in x.name and "visit" not in x.name
    ]


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
    project: Optional[Project] = None,
):
    return list(_generate_nested(name, prefix, level, project))


def _generate_nested(
    name: jedi.api.classes.Name,
    prefix: str,
    level: Optional[int] = None,
    project: Optional[Project] = None,
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
def ignored_names(project: Project):
    return {x.full_name for x in jedi.Script("", project=project).complete()}


@functools.lru_cache(maxsize=128)
def module_public_names(
    project: Project, module_name: str
) -> Sequence[jedi.api.classes.BaseName]:
    ignore = ignored_names(project)
    small_script = project.get_script(
        code=f"from {module_name} import *\n",
    )
>>>>>>> 568b51b... create dedicated custom project class

    return [
        name
        for name in small_script.complete()
        if name.full_name not in ignore and name.full_name
    ]


def get_top_level_dependencies_names(project: Project) -> Sequence[str]:
    try:
        p = project.path / "pyproject.toml"
        data = toml.loads(p.read_text())
        return (
            data.get("project", {}).get("dependencies", [])
            or data["tool"]["poetry"]["dependencies"].keys()
        )
    except Exception:
        try:
            return [x.name for x in find_requirements(project.path)]
        except RequirementsNotFound:
            return []


@functools.lru_cache()
def get_modules_from_distribution(project: Project, name: str) -> Sequence[ModuleItem]:
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
            raise ValueError(name) from None
        return []


def get_top_level_dependencies_modules(project: Project):
    return [
        x
        for dependency_name in get_top_level_dependencies_names(project)
        for x in get_modules_from_distribution(project, dependency_name)
    ]


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
def get_stdlib_modules(project: Project):
    return [
        ModuleItem(spoken=speak_single_item(x), module=x, name=None)
        for x in stdlib_module_names
        if not x.startswith("_")
    ]


@functools.lru_cache()
def get_extra_subsymbols(project: Project, key_value_pairs: Sequence[Tuple[str, str]]):
    output = [
        ModuleItem(
            spoken=speak_single_item(f"{spoken_prefix} {name.name}"),
            module=module_name,
            name=name.name,
        )
        for module_name, spoken_prefix in key_value_pairs
        for name in module_public_names(project, module_name)
    ]
    return output


@cached(
    cache=LRUCache(maxsize=4),
    key=lambda project: (project, project.path.stat().st_mtime),
)
def get_project_modules(project: Project):
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


def get_modules(
    project: Project, extra_subsymbols: Sequence[Tuple[str, str]]
) -> List[ModuleItem]:
    return list(
        itertools.chain(
            get_stdlib_modules(project),
            get_top_level_dependencies_modules(project),
            get_project_modules(project),
            get_extra_subsymbols(project, extra_subsymbols),
        )
    )


def get_scopes(script: jedi.Script, pos: Position):
    scope = script.get_context(pos.line + 1, None)
    while scope:
        yield scope
        scope = scope.parent()


def pretty_scope_list(containing_scopes):
    return " > ".join(
        x.description if x.type != "module" else "mod " + x.full_name
        for x in reversed(containing_scopes)
    )


@server.command("get_spoken")
def function(
    server: PyVoiceLanguageServer,
    doc_uri: str,
    pos: Position = None,
    generate_importables: bool = True,
):
    document = server.workspace.get_document(doc_uri)
    s = server.project.get_script(document=document)
    if generate_importables:
        imp = get_modules(server.project, tuple())
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
        containing_scopes = list(get_scopes(s, pos))
        for scope in containing_scopes:
            if scope.type == "function":
                for n in scope.defined_names():
                    output.append(with_prefix("", n))
                    output.extend(
                        generate_nested(
                            n,
                            n.name if n.type != "function" else "",
                            None,
                            server.project,
                        )
                    )
    output = [x for x in sorted(set(output)) if "__" not in x]
    if len(output) < 2000:
        output = output[:2000]
    d = speak_items(output)
    server.send_voice(
        "enhance_spoken",
        "expression",
        [{"spoken": k, "value": v} for k, v in d.items()],
    )
    scope_message = "inside " + pretty_scope_list(containing_scopes) if pos else ""
    if imp is not None:
        logger.info(f"{len(imp)} imports, {len(output)} expressions {scope_message}")
        # server.show_message(f"{len(output)} expressions, {len(imp)} imports")
    else:
        logger.info(f"{len(output)} expressions {scope_message}")
        # server.show_message(f"{len(output)} expressions, skipped imports")


# op server.send_voice( )
def add_imports_to_module(module, items: list[ModuleItem]) -> None:
    """add import statements to a module"""
    new_nodes = []
    for module_name, values in groupby(items, lambda x: x.module):
        values = list(values)
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


def add_imports_to_code(code: str, items: list[ModuleItem]) -> str:
    """add import statements to a code string"""
    module = parse(code)
    add_imports_to_module(module, items)
    return module.get_code()


@server.command("add_import")
def function_add_import(
    server: PyVoiceLanguageServer,
    doc_uri: str,
    items: ModuleItem,
    # items: Union[ModuleItem, List[ModuleItem]],
):
    server.show_message(f"{items}")
    document = server.workspace.get_document(doc_uri)
    result = add_imports_to_code(document.source, items)
    edit = WorkspaceEdit(changes={doc_uri: lsp_text_edits(document, result)})
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
    project: Project, current_path: str, module_name: str, name: str
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
