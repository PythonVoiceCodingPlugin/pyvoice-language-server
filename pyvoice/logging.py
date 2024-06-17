import logging

from lsprotocol.types import MessageType
from pygls.server import LanguageServer

from pyvoice.types import LoggingSettings


class LanguageServerLogHandler(logging.Handler):
    def __init__(self, language_server: LanguageServer):
        super().__init__()
        self.language_server = language_server

    def emit(self, record):
        try:
            if record.levelno >= logging.ERROR:
                msg_type = MessageType.Error
            elif record.levelno >= logging.WARNING:
                msg_type = MessageType.Warning
            elif record.levelno >= logging.INFO:
                msg_type = MessageType.Info
            else:
                msg_type = MessageType.Debug
            message = self.format(record)
            self.language_server.show_message_log(message, msg_type)
        except Exception:  # noqa
            pass  # noqa


_default_logging_settings = LoggingSettings()


def configure_logging(
    language_server: LanguageServer,
    settings: LoggingSettings = _default_logging_settings,
):
    level = getattr(logging, settings.level.upper())

    formatter = logging.Formatter("%(name)s - %(message)s - %(asctime)s")
    handler = LanguageServerLogHandler(language_server)
    handler.setFormatter(formatter)

    logger = logging.getLogger(__package__)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(level)
