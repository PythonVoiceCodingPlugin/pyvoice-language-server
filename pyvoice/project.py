import jedi


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
