import logging
import sys  # noqa
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
from lsprotocol.types import (
    WORKSPACE_DID_CHANGE_CONFIGURATION,
    DidChangeConfigurationParams,
    MessageActionItem,
    MessageType,
    Position,
    ShowMessageRequestParams,
    WorkspaceEdit,
)
from pygls import protocol
from pygls.server import LanguageServer

from pyvoice.custom_jedi_classes import Project
from pyvoice.generate_expressions import get_expressions
from pyvoice.generate_imports import get_modules
from pyvoice.inference import (
    get_scopes,
    join_names,
    module_public_names,
    module_public_names_fuzzy,
    pretty_scope_list,
)
from pyvoice.logging import configure_logging
from pyvoice.speakify import speak_single_item
from pyvoice.transformations import add_imports_to_code
from pyvoice.types import ModuleItem, Settings, register_custom_hooks

from .text_edit_utils import lsp_text_edits

protocol.deserialize_command = lambda p: p
F = TypeVar("F", bound=Callable)


logger = logging.getLogger(__name__)


class PyVoiceLanguageServer(LanguageServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        register_custom_hooks(self)

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
                try:
                    return f(server, *new_args)
                except jedi.api.environment.InvalidPythonEnvironment as e:
                    logger.error(e, stack_info=False)
                    # self.show_message_request(
                    #     str(e), "hello" * 10, "world" * 10, callback=None
                    # )

            self.lsp.fm.command(command_name)(function)
            return f

        return wrapper

    @property
    def project(self) -> Project:
        try:
            return self._project
        except AttributeError:
            logger.debug(
                "Creating new jedi project from %s", self.configuration_settings.project
            )
            self._project = Project.from_settings(self.configuration_settings.project)
            return self._project

    @property
    def configuration_settings(self) -> Settings:
        return getattr(self, "_configuration_settings", Settings())

    def send_voice(self, command: str, *args, **kwargs):
        server.send_notification(
            "voice/sendRpc", {"command": command, "params": args or kwargs}
        )

    def show_message_request(self, message: str, *actions, callback):
        return self.lsp.send_request(
            "window/showMessageRequest",
            ShowMessageRequestParams(
                message=message,
                type=MessageType.Info,
                actions=[MessageActionItem(title=action) for action in actions],
            ),
        )


server = PyVoiceLanguageServer(name="pyvoice", version="0.0.0b1")


@server.feature(WORKSPACE_DID_CHANGE_CONFIGURATION)
def workspace_did_change_configuration(
    ls: PyVoiceLanguageServer, params: DidChangeConfigurationParams
):
    ls._configuration_settings = ls.lsp._converter.structure(params.settings, Settings)
    configure_logging(ls, ls.configuration_settings.logging)
    try:
        del ls._project
    except AttributeError:
        pass

    try:
        env = ls.project.get_environment()
        logger.info(
            "Successfully created jedi project at %s with python %s",
            ls.project.path,
            repr(env).split(":", maxsplit=1)[1][:-1],
        )
    except jedi.api.environment.InvalidPythonEnvironment as e:
        logger.error(e, stack_info=False)


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
        imp = get_modules(server.project, server.configuration_settings.spoken.imports)
        server.send_voice("enhance_spoken", "importable", imp)
    else:
        imp = None
    if pos:
        containing_scopes = list(get_scopes(s, pos))
    expressions = get_expressions(
        s, server.configuration_settings.spoken.expressions, pos
    )
    server.send_voice("enhance_spoken", "expression", expressions)
    scope_message = "inside " + pretty_scope_list(containing_scopes) if pos else ""
    if imp is not None:
        logger.info(
            f"{len(imp)} imports, {len(expressions)} expressions {scope_message}"
        )
        # server.show_message(f"{len(expressions)} expressions, {len(imp)} imports")
    else:
        logger.info(f"{len(expressions)} expressions {scope_message}")
        # server.show_message(f"{len(expressions)} expressions, skipped imports")


@server.command("add_import")
def function_add_import(
    server: PyVoiceLanguageServer,
    doc_uri: str,
    items: Union[ModuleItem, List[ModuleItem]],
):
    server.show_message(f"{items}")
    document = server.workspace.get_document(doc_uri)
    result = add_imports_to_code(
        document.source, items if isinstance(items, list) else [items]
    )
    edit = WorkspaceEdit(changes={doc_uri: lsp_text_edits(document, result)})
    server.apply_edit(edit)


@server.command("from_import")
def function_from_import(server: PyVoiceLanguageServer, item: ModuleItem):
    module_name = join_names(item.module, item.name)
    public_names = module_public_names(server.project, module_name)
    public_names_identifiers = [x.name for x in public_names]
    s = [
        ModuleItem(spoken=speak_single_item(x.name), module=module_name, name=x.name)
        for x in public_names
    ]
    logger.info(
        "Found %s subsymbols in %s: %s", len(s), module_name, public_names_identifiers
    )
    server.send_voice("enhance_spoken", "subsymbol", s)


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
