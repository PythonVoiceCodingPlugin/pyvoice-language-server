# pyvoice

[![PyPI](https://img.shields.io/pypi/v/pyvoice-language-server.svg)](https://pypi.org/project/pyvoice-language-server/)
[![Status](https://img.shields.io/pypi/status/pyvoice-language-server.svg)](https://pypi.org/project/pyvoice-language-server/)
[![Python Version](https://img.shields.io/pypi/pyversions/pyvoice-language-server)](https://pypi.org/project/pyvoice-language-server)
[![License](https://img.shields.io/pypi/l/pyvoice-language-server)](https://opensource.org/licenses/GPL-3.0)
[![Gitter](https://badges.gitter.im/PythonVoiceCodingPlugin/pyvoice.svg)](https://app.gitter.im/?updated=1.11.31#/room/#pythonvoicecodingpluginpyvoice:gitter.im)


This is the language server component for the [pyvoice](https://github.com/PythonVoiceCodingPlugin/) system. 

<div>
<img src="https://raw.githubusercontent.com/PythonVoiceCodingPlugin/assets/main/pyvoice_logo.png" align="right" height=320 width=320/>
</div>

<!-- MarkdownTOC  autolink="true" -->

- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Security considerations](#security-considerations)
- [Performance Limitations](#performance-limitations)
- [License](#license)

<!-- /MarkdownTOC -->


This is where all the heavy business logic and semantic analysis of the source code the user is editing is performed. [pygls](https://github.com/openlawlibrary/pygls) and [lsprotocol](https://github.com/microsoft/lsprotocol) are used as the base for the LSP server implementation, and the [jedi](https://github.com/davidhalter/jedi) library is used for the static analysis.

## Features

The server is capable of generating hints for

- expressions, whose recognition should be boosted, based on the contents of the file your editing, as well as the scope within each your current selection lies. You should expect hints for
   - variables from  the local scope, all parent nonlocals scopes and the global scope
   - for variables representing modules, you should also expect access to their ***public*** attributes one level deep

> [!TIP]
> For example, if you have an import
> ```python
> import typing 
> ```
> you can expect hints to be generated for
> - `typing.Optional`
> - `typing.Union`
> - `typing.List`
> etc...
   - For enums and classes with constants, expect access to these constant attributes only
   - for variables that represent instances of classes, you can expect nested access to their attributes up to three levels deep or until the method is reached
   - finally for all the variables hints to be generated for their keyword arguments if they are callable (NOT for their nested methods)

> [!TIP]
> for example, if you have a local variable, called `server` and it has an attribute `project` which itself has has a method `get_environment`,
> then a speech hint would be generated, so that if the user says `server project get environment` , the voice coding system will insert `server.project.get_environment()` via ***keyboard*** Please note, how formatting was handled automatically

- importables from  your project, mainly modules from
   - the standard library
   - third-party dependencies, only top level not transitive
   - your project's codebase
   
but also symbols from modules that you explicitly specify

> [!TIP]
> With the default settings, you should expect hints for all items in the typing module. So for example,
> if you say `typing Optional`, the voice coding system will insert `from typing import Optional` via ***sending a command to the language server***


For more information, see [Configuration](#configuration)


## Installation

Normally you should not have to install this package manually, as the editor plug-ins for Sublime and VSCode should automatically install it for you. However, if you want a more fine-grained control, you can install the *pyvoice* executable from [PyPI](https://pypi.org/project/pyvoice-language-server/)

- with vanilla pip

```console
pip install pyvoice-language-server
```

- or (preferred) via [pipx](https://pipxproject.github.io/pipx/) so that it gets installed in an isolated environment.

```console
pipx install pyvoice-language-server
```

these commands will make the the 

```console
pyvoice
```

executable available in your command line


## Configuration

pyvoice expects something that looks like the following content ( without the comments and not in dotted dict format obviously) via workspace/didChangeConfiguration notification in order to customize how it processes each of your projects


```json
{
    "settings": {
        // Project settings used for configuring jedi Project

        // The base path where your python project is located.
        // Like all paths in this settings file it can be either
        // absolute or relative to the path of the sublime project.
        // By default, it resumed to be  the same folder that you have open
        // in the sublime window, but if your pipe and project is part
        // of a larger repository ,you should set this to the subdirectory 
        // containing your python code.for example:
        //      "project.path": "backend"
        "project.path": ".",

        // The path to the virtual environment for this project
        // again either absolute or relative to the sublime folder.
        // (NOT the project.path setting above)
        //
        // Furthermore, it could point either to the root of that environment,
        // or the python binary inside that environment 
        //      - "project.environmentPath": ".venv"
        //      - "project.environmentPath": ".venv/bin/python"
        //      - "project.environmentPath": ".venv\\Scripts\\python.exe"
        //
        //
        // WARNING
        //
        // while jedi will not execute python code from your code base
        // it WILL execute the python binary of the associated environment!
        // There are automated checks in place to guard against binaries 
        // that could for example have been injected by an attacker inside
        // vcs controlled files. You can find more about this mechanism in 
        //      https://jedi.readthedocs.io/en/latest/_modules/jedi/api/environment.html
        //      under the _is_safe function.
        //
        // However, it is still a good idea to only point this setting
        // to environments that you trust.
        "project.environmentPath": null,

        // A list of paths to override the sys path if needed.
        // Leave this null to generate sys.path from the environment.
        // WARNING: This will COMPLETELY override the sys.path 
        // generated from the environment.
        "project.sysPath": null,

        // Adds these paths at the end of the sys path.
        "project.addedSysPath": [],

        // If enabled (default), adds paths from local directories.
        // Otherwise, you will have to rely on your packages being properly configured on the sys.path.
        "project.smartSysPath": true,



        // Hints.Imports settings


        // Enable or disable the generation of stdlib imports.
        "hints.imports.stdlib.enabled": true,


        // Enable or disable the generation of third-party imports.
        // Pyvoice will try to automatically discover the  dependencies 
        // of your project  and  is going
        // to generate hints for their modules. In order to do so,pyvoice
        // will try:
        //
        // - pep621 dependencies in pyproject.toml
        // - poetry dependencies in pyproject.toml
        // - options.install_requires in setup.cfg
        // - traditional requirements.txt
        // 
        // NOTE: By default hints would be generated ONLY for your top level dependencies
        // aka distributions that you directly depend on, not transiet dependencies.
        "hints.imports.thirdParty.enabled": true,

        // A list of third-party distributions to include modules from.
        "hints.imports.thirdParty.includeDists": [],

        // A list of third-party distributions to exclude.
        "hints.imports.thirdParty.excludeDists": [],



        // Enable or disable the generation of project imports.
        // This generator is going to scan your project folders
        // recursively for pure python modules and generate hints for them.
        // This is performed in a manner somewhat similar to setuptools 
        //  - if the project follows a source layout, the src/ folder would be scanned
        //  - if the projects follow flat layout, hints would be generated
        //    for top level modules and recursively for top level packages
        //    that match the filters used by setuptools.discover
        //  - the explicit layout, where user maps in their pyproject.toml
        //    a set of packages/module names to folders containg their respective
        //    respective code is not supported to yet
        "hints.imports.project.enabled": true,

        // Enable or disable the generation of import hints for symbols
        // So far, all of the imports hints generators are targeting modules
        // However, there are cases where symbols like functions or classes
        // that are used so frequently in your project
        // that you want to be able to import them in a single step
        // and without having to speak the name of the containing module
        //
        // For example you might want to be able to speak
        //      `import optional`
        // in order to insert
        //      `from typing import Optional`
        //
        // The explicit symbols generator allows you to define a 
        // list of modules that you want to generate hints for their symbols
        "hints.imports.explicitSymbols.enabled": true,

        // A list of modules to generate hints hints for their defined symbols.
        "hints.imports.explicitSymbols.modules": ["typing"],



        // Hints.Expressions settings


        // Enable or disable the generation of hints hints for local scope.
        "hints.expressions.locals.enabled": true,
        // Hints from param names of the signatures of local scope variables. 
        "hints.expressions.locals.signature": true,


        // Enable or disable the generation of hints hints for non-local scope.
        "hints.expressions.nonlocals.enabled": true,
        // Hints from param names of the signatures of non-local scope variables. 
        "hints.expressions.nonlocals.signature": true,


        // Enable or disable the generation of hints hints for global scope.
        "hints.expressions.globals.enabled": true,
        // Hints from param names of the signatures of global scope variables. 
        "hints.expressions.globals.signature": true,


        // Enable or disable the generation of hints hints for built-in scope.
        "hints.expressions.builtins.enabled": true,
        // Hints from param names of the signatures of built-in scope variables. 
        "hints.expressions.builtins.signature": true,


        // An upper bound on the number of expressions to generate.
        "hints.expressions.limit": 2000,



        // Logging settings
        // Set the logging level for the pyvoice executable.
        "logging.level": "INFO"
    }
}
```

## Security considerations

> [!WARNING]
> This project is based on jedi and while jedi will not execute/import python code from your code base in order to perform static analysis, by design it IS GOING TO execute the python binary of the associated environment! That is something very important to keep in mind!!!
> There are automated checks in place to guard against malicious binaries that could for example have been injected by an attacker inside using your version control system but it is still important for you to make sure you trust the contents of the folders you're working on before pointing pyvoice to them. You can find more about the security mechanism in 
>      https://jedi.readthedocs.io/en/latest/_modules/jedi/api/environment.html
>      under the _is_safe function
>



## Performance Limitations

Performance the first time you generate hints for a new file can be pretty slow. However, thanks to heavy caching of intermediate results , subsequent requests to generate speech hints dramatically speed up. That being said, a couple of things to note

> [!NOTE]
> This project has not been tested against large codebases so you have to check for yourself whether performance would be satisfactory in your use case, possibly making suitable configuration changes.

> [!WARNING]
> 
> In order to meet the development time constraints, prioritize simplicity of implementation and minimize the exposure to low-level jedi internals, a trade-off between correctness and the performance of the caching mechanism had to be made

## License

Distributed under the terms of the [GPL 3.0 license](https://opensource.org/licenses/GPL-3.0), *pyvoice* is free and open source software.

The codebase also contains snippets of code borrowed from:

- [jedi-language-server](https://github.com/pappasam/jedi-language-server) by [pappasam](https://github.com/pappasam) (text_edit_utils)
- [jedi](https://github.com/davidhalter/jedi) by [davidhalter](https://github.com/davidhalter) (override certain methods)
- [mrob95](https://github.com/mrob95)'s grammar files for (speakify)

These are licensed under the respective licenses of the projects. The same goes for all the dependencies of pyvoice.

Also, this project was generated from [@cjolowicz](https://github.com/cjolowicz)'s [Hypermodern Python Cookiecutter](https://github.com/cjolowicz/cookiecutter-hypermodern-python) template.

