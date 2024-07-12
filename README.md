# pyvoice

[![PyPI](https://img.shields.io/pypi/v/pyvoice.svg)](https://pypi.org/project/pyvoice/)
[![Status](https://img.shields.io/pypi/status/pyvoice.svg)](https://pypi.org/project/pyvoice/)
[![Python Version](https://img.shields.io/pypi/pyversions/pyvoice)](https://pypi.org/project/pyvoice)
[![License](https://img.shields.io/pypi/l/pyvoice)](https://opensource.org/licenses/GPL-3.0)
[![Tests](https://github.com/PythonVoiceCodingPlugin/pyvoice/workflows/Tests/badge.svg)](https://github.com/PythonVoiceCodingPlugin/pyvoice/actions?workflow=Tests)


This is the language server component for the [pyvoice](https://github.com/PythonVoiceCodingPlugin/) system. 

<div>
<img src="https://github.com/PythonVoiceCodingPlugin/assets/blob/main/pyvoice_logo.png" align="right" height=320 width=320/>
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

* TODO

## Installation

Normally you should not have to install this package manually, as the editor plug-ins for Sublime and VSCode should automatically install it for you. However, if you want a more fine-grained control, you can install *pyvoice* from [PyPI](https://pypi.org/) with vanilla pip

```console
pip install pyvoice
```

or (preferred) via [pipx](https://pipxproject.github.io/pipx/) so that it gets installed in an isolated environment.

```console
pipx install pyvoice
```


## Configuration

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

