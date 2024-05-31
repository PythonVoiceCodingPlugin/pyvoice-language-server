import logging
from pathlib import Path

import jedi

from pyvoice.types import ProjectSettings

logger = logging.getLogger(__name__)


class Project(jedi.Project):
    def __init__(self, *args, **kwargs):
        super(Project, self).__init__(*args, **kwargs)
        self._inference_state = jedi.inference.InferenceState(self)

    def get_script(self, *, code=None, path=None, document=None):
        if document:
            code = document.source
            path = document.path
        s = jedi.Script(code=code, path=path, project=self)
        s._inference_state = self._inference_state
        return s

    @staticmethod
    def from_settings(settings: ProjectSettings, base_path: Path):
        def get_path(path, key):
            if path.is_absolute() and path.exists():
                return path
            elif (base_path / path).exists():
                return (base_path / path).absolute()
            else:
                logger.error("Path %s for %s does not exist", path, key)
                return None

        return Project(
            path=get_path(settings.path, "path"),
            environment_path=(get_path(settings.environment_path, "environment_path")),
            added_sys_path=(
                tuple(get_path(p, "added_sys_path") for p in settings.added_sys_path)
            ),
            sys_path=(
                tuple(get_path(p, "sys_path") for p in settings.sys_path)
                if settings.sys_path is not None
                else None
            ),
            smart_sys_path=settings.smart_sys_path,
        )
