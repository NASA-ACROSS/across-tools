
# across-tools

[![Unit test smoke test](https://github.com/ACROSS-Team/across-tools/actions/workflows/smoke-test.yml/badge.svg)](https://github.com/ACROSS-Team/across-tools/actions/workflows/smoke-test.yml)

<!-- [![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)

[![PyPI](https://img.shields.io/pypi/v/across-tools?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/across-tools/)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/ACROSS-Team/across-tools/smoke-test.yml)](https://github.com/ACROSS-Team/across-tools/actions/workflows/smoke-test.yml)
[![Codecov](https://codecov.io/gh/ACROSS-Team/across-tools/branch/main/graph/badge.svg)](https://codecov.io/gh/ACROSS-Team/across-tools)
[![Read The Docs](https://img.shields.io/readthedocs/across-tools)](https://across-tools.readthedocs.io/) -->

The `across-tools` repository is the NASA-ACROSS library to perform specific astronomy-related calculations on the data hosted in the `across-server`. This repository is responsible for instrument bandpasses, footprint, ephemeris, two-line element (tle) analysis, and observatory/instrument visibility calculations. 

## Installation

This repository can be installed on python (`version >=3.10`) environments.

```sh
#activate your python environment (conda, mamba, venv... etc)
git clone https://github.com/ACROSS-Team/across-tools.git
cd across-tools
python -m pip install .
```

_Direct `pypi` installation coming soon!_

## Usage

Below is a small example on how to utilize the tools to initialize a simple footprint, and perform a healpix query on a projected instrument footprint.

```python
from across.tools import Coordinate
from across.tools import Polygon
from across.tools.footprint import Footprint

simple_polygon = Polygon(
   coordinates=[
      Coordinate(ra=-0.5, dec=0.5),
      Coordinate(ra=0.5, dec=0.5),
      Coordinate(ra=0.5, dec=-0.5),
      Coordinate(ra=-0.5, dec=-0.5),
      Coordinate(ra=-0.5, dec=0.5)
   ]
)
footprint = Footprint(detectors=[simple_polygon])

projected_footprint = footprint.project(
   coordinate=Coordinate(ra=42, dec=42), roll_angle=45
)
pixels = projected_footprint.query_pixels()
```

## Contributing

Found a bug? Want to make a feature request? Or create a pull request? Navigate to our [Contributing](CONTRIBUTING.md) document for more instructions!

## Other Links

[Open Science at NASA](https://science.nasa.gov/open-science/)
