import logging

import jedi
from jedi.api.environment import create_environment, get_cached_default_environment

from pyvoice.types import ProjectSettings

logger = logging.getLogger(__name__)


class Project(jedi.Project):
    def __init__(self, *args, **kwargs):
        super(Project, self).__init__(*args, **kwargs)
        self._inference_state = jedi.inference.InferenceState(self)

    # monkey patching jedi.Project.get_environment
    # use safe=True to enforce safety checks
    # avoid execution of the environment python executable
    # if it is not deemed trusted
    def get_environment(self):
        if self._environment is None:
            if self._environment_path is not None:
                self._environment = create_environment(
                    self._environment_path, safe=True
                )
            else:
                self._environment = get_cached_default_environment()
        return self._environment

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

    # overwrite in order to accommodate for src layout
    def _get_sys_path(
        self, inference_state, add_parent_paths=True, add_init_paths=False
    ):

        result = super()._get_sys_path(
            inference_state, add_parent_paths, add_init_paths
        )
        src_path = self._path / "src"
        if src_path.exists() and src_path.is_dir():
            result.insert(0, str(src_path))
            # result.remove(str(self._path))
        return result
