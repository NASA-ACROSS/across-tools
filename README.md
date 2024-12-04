
# across-tools

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)

[![PyPI](https://img.shields.io/pypi/v/across-tools?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/across-tools/)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/ACROSS-Team/across-tools/smoke-test.yml)](https://github.com/ACROSS-Team/across-tools/actions/workflows/smoke-test.yml)
[![Codecov](https://codecov.io/gh/ACROSS-Team/across-tools/branch/main/graph/badge.svg)](https://codecov.io/gh/ACROSS-Team/across-tools)
[![Read The Docs](https://img.shields.io/readthedocs/across-tools)](https://across-tools.readthedocs.io/)

This project was automatically generated using the LINCC-Frameworks 
[python-project-template](https://github.com/lincc-frameworks/python-project-template).

A repository badge was added to show that this project uses the python-project-template, however it's up to
you whether or not you'd like to display it!

For more information about the project template see the 
[documentation](https://lincc-ppt.readthedocs.io/en/latest/).

## Dev Guide - Getting Started

Before installing any dependencies or writing code, it's a great idea to create a
virtual environment. LINCC-Frameworks engineers primarily use `conda` to manage virtual
environments. If you have conda installed locally, you can run the following to
create and activate a new environment.

```
>> conda create -n <env_name> python=3.10
>> conda activate <env_name>
```

Once you have created a new environment, you can install this project for local
development using the following commands:

```
>> ./.setup_dev.sh
>> conda install pandoc
```

Notes:
1. `./.setup_dev.sh` will initialize pre-commit for this local repository, so
   that a set of tests will be run prior to completing a local commit. For more
   information, see the Python Project Template documentation on 
   [pre-commit](https://lincc-ppt.readthedocs.io/en/latest/practices/precommit.html)
2. Install `pandoc` allows you to verify that automatic rendering of Jupyter notebooks
   into documentation for ReadTheDocs works as expected. For more information, see
   the Python Project Template documentation on
   [Sphinx and Python Notebooks](https://lincc-ppt.readthedocs.io/en/latest/practices/sphinx.html#python-notebooks)

# The .github directory

This directory contains various configurations and .yml files that are used to
define GitHub actions and behaviors.

## Workflows

The .yml files in ``./workflows`` are used to define the various continuous 
integration scripts that will be run on your behalf e.g. nightly as a smoke check,
or when you create a new PR.

For more information about CI and workflows, look here: https://lincc-ppt.readthedocs.io/en/latest/practices/ci.html

## Configurations

Templates for various different issue types are defined in ``./ISSUE_TEMPLATE``
and a pull request template is defined as ``pull_request_template.md``. Adding,
removing, and modifying these templates to suit the needs of your project is encouraged.

For more information about these templates, look here: https://lincc-ppt.readthedocs.io/en/latest/practices/issue_pr_templating.html


Or if you still have questions contact us: https://lincc-ppt.readthedocs.io/en/latest/source/contact.html