import logging
from pathlib import Path

import jedi
from cachetools import LRUCache, cached

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
    def from_settings(settings: ProjectSettings):
        return Project(
            path=settings.path,
            environment_path=settings.environment_path,
            added_sys_path=settings.added_sys_path,
            sys_path=settings.sys_path,
            smart_sys_path=settings.smart_sys_path,
        )
