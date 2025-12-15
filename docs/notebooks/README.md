# Example Notebooks

This directory contains Jupyter notebooks that demonstrate how to use the across-tools library.

## Notebooks

- **getting_started.ipynb**: Introduction to the core functionality of across-tools, including ephemeris calculation, visibility constraints, and footprint analysis.

## Running Notebooks

These notebooks are designed to be run with the across-tools package installed. To set up your environment:

```bash
pip install across-tools
```

Or for development:

```bash
pip install -e '.[dev]'
```

## Documentation Integration

Notebooks in this directory are automatically rendered and included in the ReadTheDocs documentation.
To add a new notebook:

1. Create the notebook in this directory
2. Add an entry to `../notebooks.rst` referencing your notebook
3. Include a markdown cell at the beginning with a `# Title` that will be used in the table of contents

## Pre-executed Notebooks

For notebooks that require large datasets, external API access, or significant compute resources,
place them in the `../pre_executed/` directory instead. These notebooks should be run manually
and their outputs saved before committing.
